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

# No seu script de treino, ajuste assim:
transform = {
    "visible": transforms.Compose([
        transforms.ToTensor(), # Converte para 0-1
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Escala para a ResNet
    ]),
    "infrared": transforms.Compose([
        transforms.ToTensor(), # Converte para 0-1
        transforms.Normalize([0.449], [0.226]) # Escala para o canal térmico EfficientNet-B0 1 canal
    ])
}

# ============================================================
# UTILITÁRIOS DE COORDENADAS 
# ============================================================

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

def box_xywh_to_xyxy(x):
    # Dataloader retorna [x, y, w, h] (top-left)
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
    # 1. IoU Padrão
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter + 1e-6
    iou = inter / union

    # 2. Área do Menor Invólucro Convexo (C)
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
        "iou_vis_avg": [], "msa_vis_avg": [], # Adicionado
        "iou_ir_avg": [], "msa_ir_avg": [],   # Adicionado
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
        for g_key in ["gate_vis_avg", "gate_ir_avg", "gate_ir_std"]:
            if g_key in outputs:
                val = outputs[g_key]
                val = val.item() if torch.is_tensor(val) else val
                epoch_logs[g_key].append(val)
        
        # O Visível é global por imagem 
        if "gate_vis_avg" in outputs:
            epoch_logs["gate_vis_std"].append(0.0)
        
        # MANTÉM O POSTFIX, mas adiconei o desvio padrão visualmente
        pbar.set_postfix({
            "Loss": f"{np.mean(epoch_logs['loss']):.4f}",
            "IoU": f"{np.mean(epoch_logs['iou_global']):.4f}",
            
            "G_V": f"{np.mean(epoch_logs['gate_vis_avg']):.2f}±{np.mean(epoch_logs['gate_vis_std']):.2f}" if epoch_logs['gate_vis_avg'] else "N/A",
            "G_I": f"{np.mean(epoch_logs['gate_ir_avg']):.2f}±{np.mean(epoch_logs['gate_ir_std']):.2f}"
        })

    return {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in epoch_logs.items()}

# ============================================================
# SCRIPT DE EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    train_dataset = AntiUAVRGBTDataset(ROOT_DIR, split="train", transform=transform)
    val_dataset = AntiUAVRGBTDataset(ROOT_DIR, split="test", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn_superior, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        collate_fn=collate_fn_superior, num_workers=0
    )

    model = SuperiorDETR(d_model=256, n_queries=20).to(DEVICE)
    criterion = MultimodalCriterion()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # -----------------------------------------------------------------------------------------------------
    # apenas enquanto uso o Kendal nos parãmetros dos gates
    # Filtro os parâmetros que contêm "learnable_bias" para o grupo dos Gates
    gate_params = [p for n, p in model.named_parameters() if "learnable_bias" in n]
    # Filtro os parâmetros que NÃO contêm "learnable_bias" para o grupo base
    base_params = [p for n, p in model.named_parameters() if "learnable_bias" not in n]

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': LEARNING_RATE},
        {'params': gate_params, 'lr': LEARNING_RATE * 10, 'weight_decay': 0.0} # Gates sem decay para não esmagar o aprendizado de Kendall
    ], weight_decay=1e-4)
    # -----------------------------------------------------------------------------------------------------

    os.makedirs("checkpoints", exist_ok=True)
    RESUME_PATH = "checkpoints/superior_detr_checkpoint.pth"
    start_epoch = 0
    best_msa = 0.0

    if os.path.exists(RESUME_PATH):
        print(f"\n[*] Encontrado checkpoint: {RESUME_PATH}. Carregando...")
        checkpoint = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_msa = checkpoint['best_msa']
        print(f"[+] Retomando da Época {start_epoch+1}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'='*30} ÉPOCA {epoch+1}/{NUM_EPOCHS} {'='*30}")

        # Lógica de Warm-up de Existência
        # Nas primeiras 10 épocas, focamos 80% na caixa e 20% na existência.
        # Após a época 10, usamos o peso total (1.0).
        current_exist_weight = 0.2 if epoch < 10 else 1.0
        if epoch == 10: print("\n[!] Warm-up finalizado. Peso de existência agora é 1.0")

        # Executa Treino com o peso atual
        train_results = run_epoch(model, train_loader, criterion, 
                                optimizer=optimizer, exist_weight=current_exist_weight)

        # Executa Validação (sempre peso 1.0 para métricas reais)
        val_results = run_epoch(model, val_loader, criterion, 
                                optimizer=None, exist_weight=1.0)

        #percent_padding_train = 100 * train_dataset.padding_count / train_dataset.total_calls
        #percent_padding_val = 100 * val_dataset.padding_count / val_dataset.total_calls

        #print(f"{percent_padding_train:.2f}% padding no treino.")
        #print(f"{percent_padding_val:.2f}% padding na validação.")
        print("-" * 40)
        print(f"\n[RESULTADOS DA ÉPOCA {epoch+1}]")
        print(f"{'MÉTRICA':<12} | {'TREINO':<10} | {'VALIDAÇÃO':<10}")
        print("-" * 40)
        
        # PRINT DOS RESULTADOS
        # Ordena para garantir que Loss e IoU apareçam primeiro
        for k in sorted(train_results.keys()):
            if "std" in k: continue  # Pula o STD pois ele é impresso junto com o AVG
            
            if "avg" in k:
                std_k = k.replace("avg", "std")
                t_out = f"{train_results[k]:.2f} (±{train_results.get(std_k, 0.0):.2f})"
                v_out = f"{val_results[k]:.2f} (±{val_results.get(std_k, 0.0):.2f})"
            else:
                t_out = f"{train_results[k]:.4f}"
                v_out = f"{val_results[k]:.4f}"

            print(f"{k.upper():<12} | {t_out:<15} | {v_out:<15}")

        current_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_msa': best_msa,
        }
        torch.save(current_checkpoint, RESUME_PATH)

        if val_results['msa_global'] > best_msa:
            best_msa = val_results['msa_global']
            torch.save(model.state_dict(), "checkpoints/superior_detr_best.pth")
            print(f"\n[!] Novo recorde de MSA: {best_msa:.4f}. Peso salvo.")
