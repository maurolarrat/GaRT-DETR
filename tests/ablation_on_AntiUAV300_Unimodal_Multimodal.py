import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from tqdm import tqdm
import random

from dataloader import AntiUAVRGBTDataset, collate_fn_superior
from SuperiorDETR import SuperiorDETR

# ============================================================
# CONFIGURAÇÕES
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
ROOT_DIR = r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT"
NUM_RUNS=50
GLOBAL_SEED=2029

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
# 1. WRAPPER COM LÓGICA DE REDUNDÂNCIA (ESPELHAMENTO)
# ============================================================
class AblationModelWrapper(torch.nn.Module):
    def __init__(self, model, mode="dual"):
        super().__init__()
        self.model = model
        self.mode = mode
        # Estatísticas para re-normalização
        self.vis_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
        self.vis_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
        self.ir_mean = torch.tensor([0.449]).view(1, 1, 1).to(DEVICE)
        self.ir_std = torch.tensor([0.226]).view(1, 1, 1).to(DEVICE)

    def forward(self, vis_frames, ir_frames):
        # CASO 1: Visível Falhou (IR Only)
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

        # CASO 2: IR Falhou (Visible Only)
        elif self.mode == "visible_only":
            new_ir_frames = []
            for seq in vis_frames:
                new_seq = []
                for frame in seq:
                    # 1. Desnormalizar para voltar ao range [0, 1]
                    frame_raw = frame * self.vis_std + self.vis_mean
                    
                    # 2. Saliência: Pega o valor máximo entre R, G, B
                    # Isso destaca o drone melhor que a média se ele tiver cor distinta
                    thermal_sim, _ = torch.max(frame_raw, dim=0, keepdim=True)
                    
                    # 3. Soften: Remove texturas finas do visível (faz o backbone IR gostar mais do dado)
                    thermal_sim = transforms.functional.gaussian_blur(thermal_sim, [5, 5], sigma=1.0)
                    
                    # 4. Ajuste de Contraste: Simula o range dinâmico do sensor térmico
                    thermal_sim = (thermal_sim - thermal_sim.min()) / (thermal_sim.max() - thermal_sim.min() + 1e-6)
                    
                    # 5. Re-normalizar com as estatísticas que o braço IR espera
                    thermal_sim = (thermal_sim - self.ir_mean) / self.ir_std
                    
                    new_seq.append(thermal_sim)
                new_ir_frames.append(new_seq)
            ir_frames = new_ir_frames
            
        return self.model(vis_frames, ir_frames)

# ============================================================
# 2. FUNÇÕES AUXILIARES DE CAIXA
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

def box_xywh_to_xyxy(x):
    x_tl, y_tl, w, h = x.unbind(-1)
    return torch.stack([x_tl, y_tl, x_tl + w, y_tl + h], dim=-1)

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

def calculate_iou(boxes1, boxes2):
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

# ============================================================
# 3. CRITÉRIO QUE CONSIDERA DADOS REPLICADOS
# ============================================================
class MultimodalCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, batch, exist_weight=1.0):
        p_vis = outputs["pred_boxes_vis"] 
        p_ir  = outputs["pred_boxes_ir"] 
        B, T, N, _ = p_vis.shape
        
        p_vis_xyxy = box_cxcywh_to_xyxy(p_vis)
        p_ir_xyxy  = box_cxcywh_to_xyxy(p_ir)
        
        pred_ev, pred_ei, pred_eg = outputs["exist_vis"], outputs["exist_ir"], outputs["exist"]
        orig_sizes_vis, orig_sizes_ir = outputs["orig_sizes"] 

        device = p_vis.device
        gt_vis = batch["boxes_vis"].to(device)
        gt_ir = batch["boxes_ir"].to(device)
        exist_vis = batch["exist_vis"].to(device)
        exist_ir = batch["exist_ir"].to(device)

        metrics = {
            "loss": torch.tensor(0.0, device=device),
            "iou_vis": [], "msa_vis": [],
            "iou_ir": [], "msa_ir": [],
            "count": 0
        }

        for b in range(B):
            w_v_img, h_v_img = float(orig_sizes_vis[b][0]), float(orig_sizes_vis[b][1])
            w_i_img, h_i_img = float(orig_sizes_ir[b][0]), float(orig_sizes_ir[b][1])
            scale_v = torch.tensor([w_v_img, h_v_img, w_v_img, h_v_img], device=device)
            scale_i = torch.tensor([w_i_img, h_i_img, w_i_img, h_i_img], device=device)
            
            gt_v_norm = box_xywh_to_xyxy(gt_vis[b]) / scale_v
            gt_i_norm = box_xywh_to_xyxy(gt_ir[b]) / scale_i
            gt_v_cxcywh = gt_vis[b] / scale_v
            gt_i_cxcywh = gt_ir[b] / scale_i
            
            mask_v, mask_i = exist_vis[b] > 0, exist_ir[b] > 0
            valid_mask = mask_v | mask_i 

            if valid_mask.any():
                # MATCHING DINÂMICO SIMPLIFICADO
                # Com a redundância, assume-se que ambos os braços estão ativos
                # e contribuem para a predição.
                v_preds_match = (p_vis_xyxy[b][valid_mask] + p_ir_xyxy[b][valid_mask]) / 2.0
                
                M = v_preds_match.size(0)
                target_ref = torch.where(mask_v[valid_mask].unsqueeze(1), gt_v_norm[valid_mask], gt_i_norm[valid_mask])
                
                dist = torch.abs(v_preds_match - target_ref.unsqueeze(1)).sum(-1)
                best_indices = dist.argmin(dim=-1) 
                idx_range = torch.arange(M, device=device)

                b_vis_xyxy = p_vis_xyxy[b][valid_mask][idx_range, best_indices]
                b_ir_xyxy  = p_ir_xyxy[b][valid_mask][idx_range, best_indices]
                b_vis_raw  = p_vis[b][valid_mask][idx_range, best_indices]
                b_ir_raw   = p_ir[b][valid_mask][idx_range, best_indices]

                loss_box_batch = torch.tensor(0.0, device=device)
                
                # CÁLCULO DE LOSS (Sempre calcula se houver GT, independente de ablação)
                if mask_v[valid_mask].any():
                    mv = mask_v[valid_mask]
                    l1_v = F.l1_loss(b_vis_raw[mv], gt_v_cxcywh[valid_mask][mv])
                    giou_v = (1.0 - torch.diag(generalized_box_iou(b_vis_xyxy[mv], gt_v_norm[valid_mask][mv]))).mean()
                    loss_box_batch += (5.0 * l1_v + 2.0 * giou_v)
                
                if mask_i[valid_mask].any():
                    mi = mask_i[valid_mask]
                    l1_i = F.l1_loss(b_ir_raw[mi], gt_i_cxcywh[valid_mask][mi])
                    giou_i = (1.0 - torch.diag(generalized_box_iou(b_ir_xyxy[mi], gt_i_norm[valid_mask][mi]))).mean()
                    loss_box_batch += (5.0 * l1_i + 2.0 * giou_i)

                target_exist = torch.zeros((M, N), device=device)
                target_exist[idx_range, best_indices] = 1.0
                loss_ce_v = F.binary_cross_entropy_with_logits(pred_ev[b][valid_mask], target_exist)
                loss_ce_i = F.binary_cross_entropy_with_logits(pred_ei[b][valid_mask], target_exist)
                loss_ce_g = F.binary_cross_entropy_with_logits(pred_eg[b][valid_mask], target_exist)
                loss_bg = (torch.sigmoid(pred_eg[b][valid_mask])[target_exist == 0]**2).mean() 
                
                metrics["loss"] += (loss_box_batch + exist_weight * (0.33*loss_ce_v + 0.33*loss_ce_i + 0.33*loss_ce_g + 0.1*loss_bg))
                metrics["count"] += 1

                # -----------------------------------------------------------
                # CÁLCULO DE MÉTRICAS (IOU/MSA)
                # Calcula IoU sempre que houver GT (mask_v), pois o Wrapper
                # garantiu que há dados (originais ou replicados) entrando.
                # -----------------------------------------------------------
                with torch.no_grad():
                    if mask_v.any():
                        mv = mask_v[valid_mask]
                        # Calcula IoU mesmo se o dado for do IR espelhado
                        iou_v = calculate_iou(b_vis_xyxy[mv], gt_v_norm[valid_mask][mv])
                        metrics["iou_vis"].append(iou_v.mean().item())
                        metrics["msa_vis"].append((iou_v > 0.5).float().mean().item())
                        
                    if mask_i.any():
                        mi = mask_i[valid_mask]
                        # Calcula IoU mesmo se o dado for do VIS espelhado
                        iou_i = calculate_iou(b_ir_xyxy[mi], gt_i_norm[valid_mask][mi])
                        metrics["iou_ir"].append(iou_i.mean().item())
                        metrics["msa_ir"].append((iou_i > 0.5).float().mean().item())

        denom = metrics["count"] + 1e-6
        avg_iou_v = np.mean(metrics["iou_vis"]) if metrics["iou_vis"] else 0.0
        avg_iou_i = np.mean(metrics["iou_ir"]) if metrics["iou_ir"] else 0.0
        avg_msa_v = np.mean(metrics["msa_vis"]) if metrics["msa_vis"] else 0.0
        avg_msa_i = np.mean(metrics["msa_ir"]) if metrics["msa_ir"] else 0.0

        # Pega os gates reais do modelo
        g_v = outputs.get("gate_vis_avg", torch.tensor(0.5)).item()
        g_i = outputs.get("gate_ir_avg", torch.tensor(0.5)).item()
        sum_gates = g_v + g_i + 1e-6
        
        w_v = g_v / sum_gates
        w_i = g_i / sum_gates

        # O Global considera a performance do sensor espelhado
        return {
            "loss": metrics["loss"] / denom,
            "iou_global": (w_v * avg_iou_v) + (w_i * avg_iou_i),
            "msa_global": (w_v * avg_msa_v) + (w_i * avg_msa_i),
            "iou_vis_avg": avg_iou_v,
            "msa_vis_avg": avg_msa_v,
            "iou_ir_avg": avg_iou_i,
            "msa_ir_avg": avg_msa_i
        }

# ============================================================
# FUNÇÃO DE EXECUÇÃO DE ÉPOCA (REVISADA PARA MULTI-RUN)
# ============================================================
def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE, exist_weight=1.0, current_run=1):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    epoch_logs = {
        "loss": [], 
        "iou_global": [], "msa_global": [], 
        "iou_vis_avg": [], "msa_vis_avg": [],
        "iou_ir_avg": [], "msa_ir_avg": [],
        "gate_vis_avg": [], "gate_ir_avg": [],
    }

    # Barra de progresso indica qual run executando
    pbar = tqdm(loader, desc=f">> RUN {current_run}/{NUM_RUNS}", leave=False)
    
    for batch in pbar:
        vis, ir = batch["vis_frames"], batch["ir_frames"]
        
        with torch.set_grad_enabled(is_train):
            outputs = model(vis, ir)
            res = criterion(outputs, batch, exist_weight=exist_weight)

        # Captura métricas do critério
        for k, v in res.items():
            if k in epoch_logs:
                val = v.item() if torch.is_tensor(v) else v
                epoch_logs[k].append(val)

        # Captura os Gates (G_VIS e G_IR)
        for g_key in ["gate_vis_avg", "gate_ir_avg"]:
            if g_key in outputs:
                val = outputs[g_key].item() if torch.is_tensor(outputs[g_key]) else outputs[g_key]
                epoch_logs[g_key].append(val)
        
        pbar.set_postfix({"IoU": f"{np.mean(epoch_logs['iou_global']):.4f}"})

    return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in epoch_logs.items()}

# ============================================================
# FUNÇÃO DE SUÍTE DE TESTE (INDIVIDUAL POR RUN)
# ============================================================
def perform_test_suite(mode="dual", seed=42, current_run=1):
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    test_dataset = AntiUAVRGBTDataset(ROOT_DIR, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, 
                             batch_size=8, 
                             shuffle=False, 
                             collate_fn=collate_fn_superior,
                             worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
                             generator=g)

    model = SuperiorDETR(d_model=256, n_queries=20).to(DEVICE)
    BEST_MODEL_PATH = "checkpoints/superior_detr_best.pth"

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)    
    else:
        print(f"Erro: Checkpoint não encontrado.")
        return None

    model_wrapped = AblationModelWrapper(model, mode=mode)
    
    # current_run para o log de progresso
    results = run_epoch(model_wrapped, test_loader, MultimodalCriterion(), current_run=current_run)
    return results

# ============================================================
# FUNÇÃO FINAL (TABELA COMPLETA COM MÉDIA ± STD)
# ============================================================
def run_final_test():
    modes = ["dual", "ir_only", "visible_only"]
    summary_stats = {}

    for mode in modes:
        print(f"\n>>> INICIANDO AVALIAÇÃO ESTOCÁSTICA ({NUM_RUNS} RUNS): Modo {mode.upper()}")
        mode_results = []
        
        for i in range(NUM_RUNS):
            # Gera uma semente aleatória para cada uma das 100 runs
            random_seed = random.randint(1, 1000000)
            res = perform_test_suite(mode, seed=random_seed, current_run=i+1)
            if res:
                mode_results.append(res)
        
        if mode_results:
            keys = mode_results[0].keys()
            summary_stats[mode] = {
                "mean": {k: np.mean([r[k] for r in mode_results]) for k in keys},
                "std":  {k: np.std([r[k] for r in mode_results]) for k in keys}
            }

    # Restaurando exatamente as suas colunas, agora com suporte a Média ± σ
    line_width = 195
    print("\n" + "="*line_width)
    print(f"RESUMO FINAL: MÉDIA ± DESVIO PADRÃO (σ) DE {NUM_RUNS} EXECUÇÕES")
    print("="*line_width)
    
    # Header formatado para alinhar com os dados largos
    header = (f"{'MODO':<14} | "
              f"{'IoU G (±σ)':<18} | {'IoU V (±σ)':<18} | {'IoU I (±σ)':<18} | "
              f"{'MSA G (±σ)':<18} | {'MSA V (±σ)':<18} | {'MSA I (±σ)':<18} | "
              f"{'G_VIS (±σ)':<16} | {'G_IR (±σ)':<16}")
    print(header)
    print("-" * line_width)

    for m in modes:
        if m in summary_stats:
            avg = summary_stats[m]["mean"]
            std = summary_stats[m]["std"]
            
            # Helper para formatar a célula Média ± Std
            def f_cell(val_key, precision=4):
                m_val = avg[val_key]
                s_val = std[val_key]
                return f"{m_val:.{precision}f}±{s_val:.{precision}f}"

            print(f"{m.upper():<14} | "
                  f"{f_cell('iou_global'):<18} | {f_cell('iou_vis_avg'):<18} | {f_cell('iou_ir_avg'):<18} | "
                  f"{f_cell('msa_global'):<18} | {f_cell('msa_vis_avg'):<18} | {f_cell('msa_ir_avg'):<18} | "
                  f"{f_cell('gate_vis_avg', 3):<16} | {f_cell('gate_ir_avg', 3):<16}")
    print("="*line_width)

if __name__ == "__main__":
    # Semente mestre para que o sorteio das 100 sub-sementes seja repetível
    random.seed(GLOBAL_SEED)
    run_final_test()
