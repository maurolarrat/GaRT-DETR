import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataloader410 import AntiUAV410Dataset # Certifique-se que o nome do arquivo está correto

# Configurações de caminho
ROOT_410 = r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT\Anti-UAV410"

def test_visualization():
    # 1. Instanciar o Dataset (sem transforms pesados para facilitar a visualização)
    dataset = AntiUAV410Dataset(
        root_dir=ROOT_410,
        split="test", 
        temporal_window=30,
        max_frames_per_seq=5 # Pega apenas 5 frames para o plot não ficar gigante
    )

    print(f"[*] Dataset carregado. Total de sequências: {len(dataset)}")

    # 2. Pegar um item aleatório
    sample_idx = random.randint(0, len(dataset) - 1)
    batch = dataset[sample_idx]

    seq_name = batch["seq_name"]
    ir_frames = batch["ir_frames"] # Lista de Tensores [1, H, W]
    boxes_ir = batch["boxes_ir"]   # Tensor [N, 4] -> [x, y, w, h]
    exist_ir = batch["exist_ir"]   # Tensor [N]

    print(f"[*] Visualizando Sequência: {seq_name}")

    # 3. Plotar os frames
    num_frames = len(ir_frames)
    fig, axes = plt.subplots(1, num_frames, figsize=(20, 5))
    if num_frames == 1: axes = [axes]

    for i in range(num_frames):
        # Converter Tensor (C, H, W) para Numpy (H, W) para o plt.imshow
        img_np = ir_frames[i].squeeze().numpy()
        
        axes[i].imshow(img_np, cmap='gray')
        
        # Desenhar o retângulo se o drone existir no frame
        if exist_ir[i] > 0:
            x, y, w, h = boxes_ir[i]
            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=2, edgecolor='r', facecolor='none', label='GT IR'
            )
            axes[i].add_patch(rect)
            axes[i].set_title(f"Frame {i}\n[EXIST]")
        else:
            axes[i].set_title(f"Frame {i}\n[NOT EXIST]")
        
        axes[i].axis('off')

    plt.suptitle(f"Validação Anti-UAV410 - Sequência: {seq_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import random
    test_visualization()
