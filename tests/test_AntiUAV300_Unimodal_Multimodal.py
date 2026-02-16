import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from tqdm import tqdm

from dataloader import AntiUAVRGBTDataset, collate_fn_superior
from SuperiorDETR import SuperiorDETR

# ============================================================
# CONFIGURAÇÕES
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 300
ROOT_DIR = r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT"
LEARNING_RATE = 1e-4

transform = {
    "visible": transforms.Compose([
        transforms.ToTensor(), # Converte para 0-1
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Escala para a ResNet 3 canais
    ]),
    "infrared": transforms.Compose([
        transforms.ToTensor(), # Converte para 0-1
        transforms.Normalize([0.449], [0.226]) # Escala para o canal térmico do ImageNet EfficientNet-B0 1 canal
    ])
}

# ============================================================
# UTILITÁRIOS 
# ============================================================

class AblationModelWrapper(torch.nn.Module):
    def __init__(self, model, mode="dual"):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, vis_frames, ir_frames):
        if self.mode == "ir_only":
            new_vis = []
            for seq in vis_frames:
                new_vis.append([torch.zeros_like(frame) for frame in seq])
            vis_frames = new_vis
            
        elif self.mode == "visible_only":
            new_ir = []
            for seq in ir_frames:
                new_ir.append([torch.zeros_like(frame) for frame in seq])
            ir_frames = new_ir
            
        return self.model(vis_frames, ir_frames)
    
def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

def box_xywh_to_xyxy(x):
    x_tl, y_tl, w, h = x.unbind(-1)
    return torch.stack([x_tl, y_tl, x_tl + w, y_tl + h], dim=-1)

def calculate_iou(boxes1, boxes2):
    """ boxes1: [N, 4] xyxy, boxes2: [N, 4] xyxy """
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

# ============================================================
# CRITÉRIO COM MATCHER DINÂMICO
# ============================================================

def generalized_box_iou(boxes1, boxes2):
    """
    Calcula a Generalized IoU entre dois conjuntos de caixas.
    boxes1, boxes2: [N, 4] no formato XYXY
    """
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


class MultimodalCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, batch, exist_weight=1.0):
        p_vis = outputs["pred_boxes_vis"] 
        p_ir  = outputs["pred_boxes_ir"] 
        B, T, N, _ = p_vis.shape
        
        p_vis_xyxy = box_cxcywh_to_xyxy(p_vis)
        p_ir_xyxy  = box_cxcywh_to_xyxy(p_ir)
        
        p_mean_xyxy = (p_vis_xyxy + p_ir_xyxy) / 2.0

        pred_ev, pred_ei, pred_eg = outputs["exist_vis"], outputs["exist_ir"], outputs["exist"]
        orig_sizes_vis, orig_sizes_ir = outputs["orig_sizes"] 

        device = p_vis.device
        gt_vis = batch["boxes_vis"].to(device)
        gt_ir = batch["boxes_ir"].to(device)
        exist_vis = batch["exist_vis"].to(device)
        exist_ir = batch["exist_ir"].to(device)

        # Acesso os frames diretamente do dicionário batch 
        vis_frames_batch = batch["vis_frames"] # Lista de B sequências
        ir_frames_batch = batch["ir_frames"]   # Lista de B sequências

        metrics = {
            "loss": torch.tensor(0.0, device=device),
            "iou_vis": [], "msa_vis": [],
            "iou_ir": [], "msa_ir": [],
            "count": 0
        }

        for b in range(B):
            # DETECÇÃO DE MODALIDADE ATIVA
            # Se o sensor foi zerado (Ablação/Dropout), o frame é uma constante.
            # O desvio padrão (std) de uma imagem constante é 0.
            # Uso o primeiro frame [0] da sequência para checar.
            vis_is_active = vis_frames_batch[b][0].std() > 1e-4
            ir_is_active  = ir_frames_batch[b][0].std() > 1e-4

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
                # MATCHING DINÂMICO
                # Defini o que o Matcher vai usar como referência baseada no que está ativo
                if vis_is_active and ir_is_active:
                    v_preds_match = (p_vis_xyxy[b][valid_mask] + p_ir_xyxy[b][valid_mask]) / 2.0
                elif ir_is_active:
                    v_preds_match = p_ir_xyxy[b][valid_mask] # No IR_ONLY, olha só pro IR
                else:
                    v_preds_match = p_vis_xyxy[b][valid_mask] # No VISIBLE_ONLY, olha só pro VIS
                
                M = v_preds_match.size(0)
                target_ref = torch.where(mask_v[valid_mask].unsqueeze(1), gt_v_norm[valid_mask], gt_i_norm[valid_mask])
                
                # O cálculo da distância agora é justo
                dist = torch.abs(v_preds_match - target_ref.unsqueeze(1)).sum(-1)
                best_indices = dist.argmin(dim=-1) 
                idx_range = torch.arange(M, device=device)

                # Seleção das caixas finais para o Loss e Métricas
                b_vis_xyxy = p_vis_xyxy[b][valid_mask][idx_range, best_indices]
                b_ir_xyxy  = p_ir_xyxy[b][valid_mask][idx_range, best_indices]
                b_vis_raw  = p_vis[b][valid_mask][idx_range, best_indices]
                b_ir_raw   = p_ir[b][valid_mask][idx_range, best_indices]

                loss_box_batch = torch.tensor(0.0, device=device)
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

                with torch.no_grad():
                    if mask_v.any():
                        mv = mask_v[valid_mask]
                        iou_v = calculate_iou(b_vis_xyxy[mv], gt_v_norm[valid_mask][mv])
                        metrics["iou_vis"].append(iou_v.mean().item())
                        metrics["msa_vis"].append((iou_v > 0.5).float().mean().item())
                    if mask_i.any():
                        mi = mask_i[valid_mask]
                        iou_i = calculate_iou(b_ir_xyxy[mi], gt_i_norm[valid_mask][mi])
                        metrics["iou_ir"].append(iou_i.mean().item())
                        metrics["msa_ir"].append((iou_i > 0.5).float().mean().item())

        denom = metrics["count"] + 1e-6
        avg_iou_v = np.mean(metrics["iou_vis"]) if metrics["iou_vis"] else 0.0
        avg_iou_i = np.mean(metrics["iou_ir"]) if metrics["iou_ir"] else 0.0
        avg_msa_v = np.mean(metrics["msa_vis"]) if metrics["msa_vis"] else 0.0
        avg_msa_i = np.mean(metrics["msa_ir"]) if metrics["msa_ir"] else 0.0

        g_v = outputs.get("gate_vis_avg", torch.tensor(0.5)).item()
        g_i = outputs.get("gate_ir_avg", torch.tensor(0.5)).item()
        sum_gates = g_v + g_i + 1e-6
        
        w_v = g_v / sum_gates
        w_i = g_i / sum_gates

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
# FUNÇÃO DE EXECUÇÃO DE ÉPOCA
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE, exist_weight=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    epoch_logs = {
        "loss": [], 
        "iou_global": [], "msa_global": [], 
        "iou_vis_avg": [], "msa_vis_avg": [],
        "iou_ir_avg": [], "msa_ir_avg": [],
        "gate_vis_avg": [], "gate_ir_avg": [],
        "gate_vis_std": [], "gate_ir_std": [] 
    }

    desc = ">> TREINO" if is_train else ">> VALIDAÇÃO"
    pbar = tqdm(loader, desc=desc)
    
    for batch in pbar:
        vis, ir = batch["vis_frames"], batch["ir_frames"]
        
        with torch.set_grad_enabled(is_train):
            outputs = model(vis, ir)
            res = criterion(outputs, batch, exist_weight=exist_weight)

            if is_train:
                optimizer.zero_grad()
                res["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
        # LOG DE MÉTRICAS DO CRITÉRIO (Loss, IoU)
        for k, v in res.items():
            if k in epoch_logs:
                val = v.item() if torch.is_tensor(v) else v
                epoch_logs[k].append(val)

        # LOG DOS GATES (Vindo do forward do modelo)
        for g_key in ["gate_vis_avg", "gate_ir_avg", "gate_vis_std", "gate_ir_std"]:
            if g_key in outputs:
                val = outputs[g_key]
                val = val.item() if torch.is_tensor(val) else val
                epoch_logs[g_key].append(val)
        
        pbar.set_postfix({
            "Loss": f"{np.mean(epoch_logs['loss']):.4f}",
            "IoU": f"{np.mean(epoch_logs['iou_global']):.4f}",
            "G_V": f"{np.mean(epoch_logs['gate_vis_avg']):.2f}±{np.mean(epoch_logs['gate_vis_std']):.2f}" if epoch_logs['gate_vis_avg'] else "N/A",
            "G_I": f"{np.mean(epoch_logs['gate_ir_avg']):.2f}±{np.mean(epoch_logs['gate_ir_std']):.2f}" if epoch_logs['gate_ir_avg'] else "N/A"
        })

    return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in epoch_logs.items()}

# ============================================================
# 3. FUNÇÕES DE TESTE
# ============================================================
def perform_test_suite(mode="dual"):
    print(f"\n>>> INICIANDO TESTE: Modo {mode.upper()}")
    
    test_dataset = AntiUAVRGBTDataset(ROOT_DIR, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_superior)

    model = SuperiorDETR(d_model=256, n_queries=20).to(DEVICE)
    BEST_MODEL_PATH = "checkpoints/superior_detr_best.pth"

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)    
    else:
        print(f"Erro: Checkpoint não achado.")
        return None

    model_wrapped = AblationModelWrapper(model, mode=mode)
    
    results = run_epoch(
        model_wrapped, test_loader, MultimodalCriterion(), 
        optimizer=None, device=DEVICE, exist_weight=1.0
    )
    return results

def run_final_test():
    modes = ["dual", "ir_only", "visible_only"]
    final_summary = {}

    for mode in modes:
        res = perform_test_suite(mode)
        if res: final_summary[mode] = res

    print("\n" + "="*115)
    header = (f"{'MODO':<14} | "
            f"{'IoU G':<8} | {'IoU V':<8} | {'IoU I':<8} | "
            f"{'MSA G':<8} | {'MSA V':<8} | {'MSA I':<8} | "
            f"{'G_VIS':<8} | {'G_IR':<8}") 
    print(header)
    print("-" * 115)

    for m in modes:
        if m in final_summary:
            r = final_summary[m]
            print(f"{m.upper():<14} | "
                f"{r['iou_global']:.4f} | {r['iou_vis_avg']:.4f} | {r['iou_ir_avg']:.4f} | "
                f"{r['msa_global']:.4f} | {r['msa_vis_avg']:.4f} | {r['msa_ir_avg']:.4f} | "
                f"{r['gate_vis_avg']:.4f} | {r['gate_ir_avg']:.4f}")
    print("="*115)

if __name__ == "__main__":
    run_final_test()
