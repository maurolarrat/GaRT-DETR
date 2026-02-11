import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Importando seus arquivos
from dataloader import AntiUAVRGBTDataset, collate_fn_superior
from SuperiorDETR import preprocess_batch 

# 1. SETUP
dataset = AntiUAVRGBTDataset(
    root_dir=r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT", # Ajuste seu caminho
    split="train",
    temporal_window=5,
    max_frames_per_seq=5
)

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_superior)

# 2. SIMULAÇÃO DO QUE OCORRE NO MODELO
batch = next(iter(loader))

# O modelo agora retorna 4 valores para suportar resoluções diferentes por modalidade
target_size = (224, 224)
x_rgb, x_ir, orig_sizes_vis, orig_sizes_ir = preprocess_batch(
    batch['vis_frames'], 
    batch['ir_frames'], 
    target_size=target_size
)

# 3. VERIFICAÇÃO VISUAL
b_idx = 0
t_idx = 0
global_idx = b_idx * 5 + t_idx 

# Imagens redimensionadas
img_vis_resized = x_rgb[global_idx].permute(1, 2, 0).cpu().numpy()
img_ir_resized = x_ir[global_idx].squeeze().cpu().numpy()

# GTs originais em pixels
gt_p_v = batch['boxes_vis'][b_idx][t_idx]
gt_p_i = batch['boxes_ir'][b_idx][t_idx]

# Tamanhos originais (CADA UM COM O SEU)
W_v, H_v = orig_sizes_vis[b_idx]
W_i, H_i = orig_sizes_ir[b_idx]

# Função de ajuste considerando que as resoluções originais divergem
def get_resized_box(gt_pixel, w_orig, h_orig, target_w, target_h):
    # Normaliza pela resolução real do sensor que gerou a imagem
    nx = gt_pixel[0] / w_orig
    ny = gt_pixel[1] / h_orig
    nw = gt_pixel[2] / w_orig
    nh = gt_pixel[3] / h_orig
    return [nx * target_w, ny * target_h, nw * target_w, nh * target_h]

# Aplica o redimensionamento usando o W/H específico de cada modalidade
box_v_res = get_resized_box(gt_p_v, W_v, H_v, target_size[1], target_size[0])
box_i_res = get_resized_box(gt_p_i, W_i, H_i, target_size[1], target_size[0])

# 4. PLOT
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(img_vis_resized)
ax[0].set_title(f"VIS ({W_v}x{H_v} -> {target_size})")
ax[0].add_patch(patches.Rectangle((box_v_res[0], box_v_res[1]), box_v_res[2], box_v_res[3], 
                                 linewidth=2, edgecolor='r', facecolor='none'))

ax[1].imshow(img_ir_resized, cmap='gray')
ax[1].set_title(f"IR ({W_i}x{H_i} -> {target_size})")
ax[1].add_patch(patches.Rectangle((box_i_res[0], box_i_res[1]), box_i_res[2], box_i_res[3], 
                                 linewidth=2, edgecolor='cyan', facecolor='none'))

plt.suptitle(f"Sequência: {batch['seq_names'][b_idx]} - Resoluções Independentes")
plt.show()

print(f"Resolução VIS original: {W_v}x{H_v}")
print(f"Resolução IR original: {W_i}x{H_i}")