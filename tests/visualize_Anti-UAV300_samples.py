import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Importando do seu arquivo
from dataloader import AntiUAVRGBTDataset, collate_fn_superior

# 1. CONFIGURAÇÃO
transform = {
    "visible": transforms.Compose([transforms.ToTensor()]),
    "infrared": transforms.Compose([transforms.ToTensor()])
}

dataset = AntiUAVRGBTDataset(
    root_dir=r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT", 
    split="train",
    transform=transform,
    temporal_window=5,      
    max_frames_per_seq=5    
)

# O DataLoader com shuffle=True garante que cada next(iter()) pegue um vídeo aleatório
loader = DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=True, 
    collate_fn=collate_fn_superior
)

# 2. EXECUÇÃO DO TESTE (Pega um batch aleatório a cada vez)
try:
    # Cria o iterador e pega o primeiro batch (que será aleatório devido ao shuffle)
    batch = next(iter(loader))
    print(f"Vídeo sorteado: {batch['seq_names'][0]}")
except Exception as e:
    print(f"Erro ao carregar batch: {e}")
    raise e

# 3. MAPEAMENTO DE DADOS
b_idx = 0
vis_seq = batch["vis_frames"][b_idx] 
ir_seq = batch["ir_frames"][b_idx]   
boxes_v = batch["boxes_vis"][b_idx]  
boxes_i = batch["boxes_ir"][b_idx]   
exist_v = batch["exist_vis"][b_idx]
exist_i = batch["exist_ir"][b_idx]

# 4. LOOP DE VISUALIZAÇÃO
for t in range(len(vis_seq)):
    # vis_seq[t] shape: (3, H, W) -> (H, W, 3)
    img_vis = vis_seq[t].permute(1, 2, 0).cpu().numpy()
    
    # ir_seq[t] shape: (1, H, W) -> (H, W)
    img_ir = ir_seq[t].squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # VISIBLE
    ax[0].imshow(img_vis)
    ax[0].set_title(f"VISIBLE - T={t} | Exist: {int(exist_v[t])}")
    if exist_v[t] > 0:
        bx = boxes_v[t] # [x, y, w, h]
        rect = patches.Rectangle((bx[0], bx[1]), bx[2], bx[3], 
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
    
    # INFRARED
    ax[1].imshow(img_ir, cmap='gray')
    ax[1].set_title(f"INFRARED - T={t} | Exist: {int(exist_i[t])}")
    if exist_i[t] > 0:
        bx = boxes_i[t]
        rect = patches.Rectangle((bx[0], bx[1]), bx[2], bx[3], 
                                 linewidth=2, edgecolor='cyan', facecolor='none')
        ax[1].add_patch(rect)

    plt.suptitle(f"Sequência Aleatória: {batch['seq_names'][b_idx]}")
    plt.tight_layout()
    plt.show()
