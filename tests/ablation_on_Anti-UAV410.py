import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import random
from tqdm import tqdm

from dataloader410 import AntiUAV410Dataset, collate_fn_superior
from SuperiorDETR import SuperiorDETR

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
ROOT_DIR_410 = r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT\Anti-UAV410"
NUM_RUNS = 100  # Definido para 100 execuções

transform = {
    "visible": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "infrared": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.449], [0.226])
    ])
}

# ============================================================
# 1. WRAPPER DE ABLAÇÃO (ESPELHAMENTO IR -> VIS)
# ============================================================
class AblationModelWrapper(torch.nn.Module):
    def __init__(self, model, mode="ir_only"):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, vis_frames, ir_frames):
        if self.mode == "ir_only":
            new_vis_frames = []
            for seq in ir_frames:
                new_seq = []
                for frame in seq:
                    grad_x = torch.abs(frame[:, :, 1:] - frame[:, :, :-1])
                    grad_x = F.pad(grad_x, (0, 1, 0, 0))
                    grad_y = torch.abs(frame[:, 1:, :] - frame[:, :-1, :])
                    grad_y = F.pad(grad_y, (0, 0, 0, 1))
                    pseudo_rgb = torch.cat([frame, grad_x, grad_y], dim=0)
                    new_seq.append(pseudo_rgb)
                new_vis_frames.append(new_seq)
            vis_frames = new_vis_frames
            
        return self.model(vis_frames, ir_frames)

# ============================================================
# 2. UTILITÁRIOS DE GEOMETRIA
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

def box_xywh_to_xyxy(x):
    x_tl, y_tl, w, h = x.unbind(-1)
    return torch.stack([x_tl, y_tl, x_tl + w, y_tl + h], dim=-1)

def calculate_iou(boxes1, boxes2):
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

def generalized_box_iou(boxes1, boxes2):
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter + 1e-6
    iou = inter / union
    lt_c = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1] + 1e-6
    return iou - (area_c - union) / area_c

# ============================================================
# 3. CRITÉRIO AJUSTADO (L1 + GIoU focado no IR)
# ============================================================
class MultimodalCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, batch):
        p_ir_raw = outputs["pred_boxes_ir"] # Formato cxcywh
        B, T, N, _ = p_ir_raw.shape
        p_ir_xyxy = box_cxcywh_to_xyxy(p_ir_raw)
        
        orig_sizes_ir = outputs["orig_sizes"][1] 
        device = p_ir_raw.device
        gt_ir = batch["boxes_ir"].to(device)
        exist_ir = batch["exist_ir"].to(device)

        metrics = {"loss": torch.tensor(0.0, device=device), "iou_ir": [], "msa_ir": [], "count": 0}

        for b in range(B):
            w_i, h_i = float(orig_sizes_ir[b][0]), float(orig_sizes_ir[b][1])
            scale_i = torch.tensor([w_i, h_i, w_i, h_i], device=device)
            
            gt_i_norm_xyxy = box_xywh_to_xyxy(gt_ir[b]) / scale_i
            gt_i_norm_cxcywh = gt_ir[b] / scale_i
            
            mask_i = exist_ir[b] > 0

            if mask_i.any():
                dist = torch.abs(p_ir_xyxy[b][mask_i] - gt_i_norm_xyxy[mask_i].unsqueeze(1)).sum(-1)
                best_indices = dist.argmin(dim=-1)
                idx_range = torch.arange(mask_i.sum(), device=device)
                
                b_ir_xyxy = p_ir_xyxy[b][mask_i][idx_range, best_indices]
                b_ir_raw  = p_ir_raw[b][mask_i][idx_range, best_indices]

                loss_l1 = F.l1_loss(b_ir_raw, gt_i_norm_cxcywh[mask_i])
                loss_giou = (1.0 - torch.diag(generalized_box_iou(b_ir_xyxy, gt_i_norm_xyxy[mask_i]))).mean()
                
                metrics["loss"] += (5.0 * loss_l1 + 2.0 * loss_giou)
                metrics["count"] += 1

                iou_i = calculate_iou(b_ir_xyxy, gt_i_norm_xyxy[mask_i])
                metrics["iou_ir"].append(iou_i.mean().item())
                metrics["msa_ir"].append((iou_i > 0.5).float().mean().item())

        denom = metrics["count"] + 1e-6
        return {
            "loss": metrics["loss"] / denom,
            "iou_ir_avg": np.mean(metrics["iou_ir"]) if metrics["iou_ir"] else 0.0,
            "msa_ir_avg": np.mean(metrics["msa_ir"]) if metrics["msa_ir"] else 0.0
        }

# ============================================================
# 4. EXECUÇÃO
# ============================================================
def run_epoch(model, loader, criterion, run_idx=0):
    model.eval()
    logs = {"loss": [], "iou_ir_avg": [], "msa_ir_avg": []}
    pbar = tqdm(loader, desc=f">> RUN {run_idx+1}/{NUM_RUNS}", leave=False)
    
    for batch in pbar:
        vis, ir = batch["vis_frames"], batch["ir_frames"]
        with torch.no_grad():
            outputs = model(vis, ir)
            res = criterion(outputs, batch)
        
        for k in logs.keys():
            val = res[k].item() if torch.is_tensor(res[k]) else res[k]
            logs[k].append(val)
            
        pbar.set_postfix({"Loss": f"{np.mean(logs['loss']):.4f}", "IoU": f"{np.mean(logs['iou_ir_avg']):.4f}"})
    return {k: np.mean(v) for k, v in logs.items()}

def run_final_test_410():
    print("\n" + "="*60)
    print(f"   SUPERIORDETR - 410 EVALUATION (AVERAGE OVER {NUM_RUNS} RUNS)")
    print("="*60)

    # Dataset e Loader (carregados uma vez para eficiência)
    test_dataset = AntiUAV410Dataset(ROOT_DIR_410, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_superior)
    
    BEST_MODEL_PATH = "checkpoints/superior_detr_best.pth"
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"[!] Erro: Checkpoint {BEST_MODEL_PATH} não encontrado.")
        return

    # Lista para armazenar resultados de cada run
    all_runs_results = []

    for i in range(NUM_RUNS):
        # Gera uma semente aleatória para esta execução
        current_seed = random.randint(1, 100000)
        set_seed(current_seed)

        # Inicializa Modelo e Critério para cada run (para garantir independência da semente)
        model = SuperiorDETR(d_model=256, n_queries=20).to(DEVICE)
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)
        
        model_wrapped = AblationModelWrapper(model, mode="ir_only")
        criterion = MultimodalCriterion()

        # Executa o teste
        res = run_epoch(model_wrapped, test_loader, criterion, run_idx=i)
        all_runs_results.append(res)

    # Cálculo das médias finais
    final_avg = {k: np.mean([r[k] for r in all_runs_results]) for k in all_runs_results[0].keys()}

    print("\n" + "-"*45)
    print(f" {'MÉTRICA (MÉDIA 100 RUNS)':<25} | {'VALOR':<10}")
    print("-" * 45)
    print(f" {'Loss (L1+GIoU)':<25} | {final_avg['loss']:.4f}")
    print(f" {'IoU (Thermal)':<25} | {final_avg['iou_ir_avg']:.4f}")
    print(f" {'MSA (Thermal)':<25} | {final_avg['msa_ir_avg']:.4f}")
    print("-" * 45)
    print("=" * 60)

if __name__ == "__main__":
    run_final_test_410()
