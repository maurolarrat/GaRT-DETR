import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import random

'''
Esta classe gerencia o carregamento de vídeos (sequências de imagens)
que possuem dois "olhos": um Visível (RGB) e um Infravermelho (Térmico).
'''
class AntiUAVRGBTDataset(Dataset):
    def __init__(
        self,
        root_dir: str,           # Pasta raiz onde estão os dados
        split: str = "train",    # 'train' ou 'val'
        transform=None,          # Redimensionamento e conversão para tensor
        max_frames_per_seq: int = None,
        temporal_window: int = 10, # Quantos frames consecutivos o modelo verá
        augment=False,
        val_modality: str = "RANDOM"
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.temporal_window = temporal_window

        # 1. Localiza a pasta do split (ex: data/train)
        self.split_dir = os.path.join(root_dir, split)

        # 2. Lista apenas as subpastas (cada pasta é um vídeo/sequência)
        folder_names = sorted([
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ])

        self.sequences = [os.path.join(self.split_dir, d) for d in folder_names]

        # 3. CACHE: Vamos ler todos os JSONs de anotação agora para não travar o treino depois
        self.annotation_cache = {}
        print(f"Carregando anotações do split {split} na RAM...")

        for d in folder_names:
            seq_path = os.path.join(self.split_dir, d)

            # Abre as anotações das duas modalidades
            with open(os.path.join(seq_path, "visible.json"), "r") as f:
                v_data = json.load(f)
            with open(os.path.join(seq_path, "infrared.json"), "r") as f:
                ir_data = json.load(f)

            # 4. FILTRO DE QUALIDADE: Só aceita frames que tenham as 4 coordenadas [x, y, w, h]
            # Isso evita tentar treinar o modelo com frames corrompidos ou sem marcação
            valid_indices = [
                i for i in range(len(v_data["gt_rect"]))
                if len(v_data["gt_rect"][i]) == 4 and len(ir_data["gt_rect"][i]) == 4
            ]

            # Guarda tudo na memória RAM para acesso instantâneo
            self.annotation_cache[d] = {
                "gt_rect_vis": v_data["gt_rect"],
                "gt_rect_ir": ir_data["gt_rect"],
                "exist_vis": v_data["exist"],
                "exist_ir": ir_data["exist"],
                "files_vis": sorted(os.listdir(os.path.join(seq_path, "visible"))),
                "files_ir": sorted(os.listdir(os.path.join(seq_path, "infrared"))),
                "valid_indices": valid_indices
            }

    def __len__(self):
        # O tamanho do dataset é o número total de vídeos/pastas
        return len(self.sequences)

    def __getitem__(self, idx):
        # 5. SELEÇÃO DA JANELA TEMPORAL:
        # Quando o treino pede um item, escolhemos um trecho do vídeo.
        seq_path = self.sequences[idx]
        seq_name = os.path.basename(seq_path)

        data = self.annotation_cache[seq_name]
        valid_indices = data["valid_indices"]

        # Se o vídeo for longo o suficiente, sorteamos um ponto de início aleatório
        if len(valid_indices) >= self.temporal_window:
            start = random.randint(0, len(valid_indices) - self.temporal_window)
            selected_indices = valid_indices[start:start + self.temporal_window]
        else:
            # Se o vídeo for curto demais, repetimos o último frame (padding) para completar a janela
            selected_indices = valid_indices + \
                [valid_indices[-1]] * (self.temporal_window - len(valid_indices))

        vis_tensors, ir_tensors = [], []
        gt_v_list, gt_ir_list = [], []
        exist_list = []

        # 6. PROCESSAMENTO DOS FRAMES DA JANELA
        for k in selected_indices:
            # Carrega imagem Visível (RGB) e Térmica (L - Tons de cinza)
            v_img = Image.open(
                os.path.join(seq_path, "visible", data["files_vis"][k])
            ).convert("RGB")

            ir_img = Image.open(
                os.path.join(seq_path, "infrared", data["files_ir"][k])
            ).convert("L")

            Wv, Hv = v_img.size
            Wir, Hir = ir_img.size

            # Aplica transformações (Resize, Normalização, etc)
            v_t = self.transform['visible'](v_img) if isinstance(self.transform, dict) else self.transform(v_img)
            ir_t = self.transform['infrared'](ir_img) if isinstance(self.transform, dict) else self.transform(ir_img)

            vis_tensors.append(v_t)
            ir_tensors.append(ir_t)

            # 7. NORMALIZAÇÃO DAS CAIXAS (Bounding Boxes):
            # Converte de pixels [x, y, w, h] para coordenadas relativas [0, 1] 
            # usando o formato [center_x, center_y, width, height]
            x, y, w, h = data["gt_rect_vis"][k]
            gt_v_list.append([
                (x + w / 2) / Wv,
                (y + h / 2) / Hv,
                w / Wv,
                h / Hv
            ])

            x, y, w, h = data["gt_rect_ir"][k]
            gt_ir_list.append([
                (x + w / 2) / Wir,
                (y + h / 2) / Hir,
                w / Wir,
                h / Hir
            ])

            # 8. VERIFICAÇÃO DE EXISTÊNCIA:
            # O drone só "existe" neste frame se estiver visível em AMBAS as câmeras
            exist_list.append(
                1 if (data["exist_vis"][k] and data["exist_ir"][k]) else 0
            )

        # Retorna o pacote completo para um único vídeo
        return {
            # x_input terá o shape: [Janela, 4, H, W] -> (3 canais RGB + 1 canal Térmico)
            "x_input": torch.cat(
                [torch.stack(vis_tensors), torch.stack(ir_tensors)], dim=1
            ),
            "boxes_vis": torch.tensor(gt_v_list, dtype=torch.float32),
            "boxes_ir": torch.tensor(gt_ir_list, dtype=torch.float32),
            "exist": torch.tensor(exist_list, dtype=torch.float32),
            "seq_name": seq_name
        }

'''
Esta função junta vários vídeos sorteados em um único "Batch" (Lote).
Ela empilha os tensores para que o modelo possa processar vários vídeos em paralelo na GPU.
'''
def collate_fn_superior(batch):
    return {
        "x_input": torch.stack([item["x_input"] for item in batch]), # [Batch, Janela, 4, H, W]
        "boxes_vis": torch.stack([item["boxes_vis"] for item in batch]),
        "boxes_ir": torch.stack([item["boxes_ir"] for item in batch]),
        "exist": torch.stack([item["exist"] for item in batch]),
        "seq_names": [item["seq_name"] for item in batch]
    }
