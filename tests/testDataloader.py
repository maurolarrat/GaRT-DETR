import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Importando do seu arquivo (certifique-se que o nome do arquivo seja dataloader.py)
from dataloader import AntiUAVRGBTDataset, collate_fn_superior

# 1. CONFIGURAÇÃO
# Note que não usamos Resize aqui para manter a dimensão original
transform = {
    "visible": transforms.Compose([transforms.ToTensor()]),
    "infrared": transforms.Compose([transforms.ToTensor()])
}

dataset = AntiUAVRGBTDataset(
    root_dir=r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT", # Ajuste seu caminho
    split="train",
    transform=transform,
    temporal_window=5,      # Janela curta para teste
    max_frames_per_seq=5    # Poucos frames para teste
)

# Batch size pode ser > 1 porque o collate_fn agora suporta listas
loader = DataLoader(
    dataset, 
    batch_size=2, 
    shuffle=True, 
    collate_fn=collate_fn_superior
)

# 2. EXECUÇÃO DO TESTE
try:
    batch = next(iter(loader))
    print(f"Sequências carregadas: {batch['seq_names']}")
except Exception as e:
    print(f"Erro ao carregar batch: {e}")
    exit()

# Vamos visualizar o primeiro exemplo do batch (índice 0)
b_idx = 0
vis_seq = batch["vis_frames"][b_idx] # Lista de tensores [T, C, H, W]
ir_seq = batch["ir_frames"][b_idx]   # Lista de tensores [T, C, H, W]
boxes_v = batch["boxes_vis"][b_idx]  # Tensor [T, 4]
boxes_i = batch["boxes_ir"][b_idx]   # Tensor [T, 4]
exist_v = batch["exist_vis"][b_idx]
exist_i = batch["exist_ir"][b_idx]

# Loop pelos frames da janela temporal
for t in range(len(vis_seq)):
    # Conversão de tensor para numpy para o matplotlib
    # vis_seq[t] é (3, H, W) -> permute para (H, W, 3)
    img_vis = vis_seq[t].permute(1, 2, 0).numpy()
    # ir_seq[t] é (1, H, W) -> squeeze para (H, W)
    img_ir = ir_seq[t].squeeze().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot VISIBLE
    ax[0].imshow(img_vis)
    ax[0].set_title(f"VISIBLE - Frame {t} | Exist: {int(exist_v[t])}")
    if exist_v[t] > 0:
        # box: [x, y, w, h] em pixels reais
        bx = boxes_v[t]
        rect = patches.Rectangle((bx[0], bx[1]), bx[2], bx[3], 
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
    
    # Plot INFRARED
    ax[1].imshow(img_ir, cmap='gray')
    ax[1].set_title(f"INFRARED - Frame {t} | Exist: {int(exist_i[t])}")
    if exist_i[t] > 0:
        bx = boxes_i[t]
        rect = patches.Rectangle((bx[0], bx[1]), bx[2], bx[3], 
                                 linewidth=2, edgecolor='cyan', facecolor='none')
        ax[1].add_patch(rect)

    plt.suptitle(f"Sequência: {batch['seq_names'][b_idx]}")
    plt.tight_layout()
    plt.show()

    # Se quiser ver apenas o primeiro frame de cada sequência, descomente o break
    # break