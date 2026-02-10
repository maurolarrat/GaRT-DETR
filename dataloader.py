import os
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import random


class AntiUAVRGBTDataset(Dataset):
    '''
    Dataset para sequências RGBT.
    - Cada modalidade mantém sua própria GT
    - Existência local (RGB / IR)
    - Existência global (RGB OR IR)
    '''

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        temporal_window: int = 30,
        max_frames_per_seq: int = 10,   
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.temporal_window = temporal_window
        self.max_frames_per_seq = max_frames_per_seq  

        # Pasta do split
        self.split_dir = os.path.join(root_dir, split)

        # Lista de sequências
        folder_names = sorted([
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ])
        self.sequences = folder_names

        # Cache de anotações
        self.annotation_cache = {}
        print(f"Carregando anotações do split {split} na RAM...")

        for seq in folder_names:
            seq_path = os.path.join(self.split_dir, seq)

            with open(os.path.join(seq_path, "visible.json"), "r") as f:
                v_data = json.load(f)

            with open(os.path.join(seq_path, "infrared.json"), "r") as f:
                ir_data = json.load(f)

            # Frames válidos: GT definida nas duas modalidades
            valid_indices = [
                i for i in range(len(v_data["gt_rect"]))
                if len(v_data["gt_rect"][i]) == 4 and len(ir_data["gt_rect"][i]) == 4
            ]

            self.annotation_cache[seq] = {
                "gt_rect_vis": v_data["gt_rect"],
                "gt_rect_ir": ir_data["gt_rect"],
                "exist_vis": v_data["exist"],
                "exist_ir": ir_data["exist"],
                "files_vis": sorted(os.listdir(os.path.join(seq_path, "visible"))),
                "files_ir": sorted(os.listdir(os.path.join(seq_path, "infrared"))),
                "valid_indices": valid_indices
            }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_name = self.sequences[idx]
        seq_path = os.path.join(self.split_dir, seq_name)

        data = self.annotation_cache[seq_name]
        valid_indices = data["valid_indices"]

        # Aplica MAX_FRAMES
        if self.max_frames_per_seq is not None:
            valid_indices = valid_indices[:self.max_frames_per_seq]

        # Se a sequência tiver menos frames que a janela, completa repetindo último frame
        if len(valid_indices) <= self.temporal_window:
            selected_indices = valid_indices + [valid_indices[-1]] * (self.temporal_window - len(valid_indices))
        else:
            # Escolhe um trecho aleatório de tamanho >= temporal_window
            start_idx = random.randint(0, len(valid_indices) - self.temporal_window)
            end_idx = len(valid_indices)  # pode ir até o final ou limitar a MAX_FRAMES
            segment = valid_indices[start_idx:end_idx]

            # Seleciona temporal_window frames uniformemente no segmento
            idxs = np.linspace(0, len(segment)-1, num=self.temporal_window, dtype=int)
            selected_indices = [segment[i] for i in idxs]

        vis_tensors, ir_tensors = [], []
        gt_vis, gt_ir = [], []

        exist_vis_list = []
        exist_ir_list = []
        exist_global_list = []

        for k in selected_indices:
            # Imagens
            v_img = Image.open(
                os.path.join(seq_path, "visible", data["files_vis"][k])
            ).convert("RGB")

            ir_img = Image.open(
                os.path.join(seq_path, "infrared", data["files_ir"][k])
            ).convert("L")

            Wv, Hv = v_img.size
            Wir, Hir = ir_img.size

            # Transformações
            if isinstance(self.transform, dict):
                v_t = self.transform["visible"](v_img)
                ir_t = self.transform["infrared"](ir_img)
            else:
                v_t = self.transform(v_img)
                ir_t = self.transform(ir_img)

            vis_tensors.append(v_t)
            ir_tensors.append(ir_t)

            # GT RGB
            x, y, w, h = data["gt_rect_vis"][k]
            gt_vis.append([
                (x + w / 2) / Wv,
                (y + h / 2) / Hv,
                w / Wv,
                h / Hv
            ])

            # GT IR
            x, y, w, h = data["gt_rect_ir"][k]
            gt_ir.append([
                (x + w / 2) / Wir,
                (y + h / 2) / Hir,
                w / Wir,
                h / Hir
            ])

            # Existências
            exist_vis = data["exist_vis"][k]
            exist_ir = data["exist_ir"][k]

            exist_vis_list.append(exist_vis)
            exist_ir_list.append(exist_ir)
            exist_global_list.append(1 if (exist_vis or exist_ir) else 0)

        return {
            # Inputs separados por canal (não há fusão semântica)
            "x_input": torch.cat(
                [torch.stack(vis_tensors), torch.stack(ir_tensors)], dim=1
            ),

            # GTs
            "boxes_vis": torch.tensor(gt_vis, dtype=torch.float32),
            "boxes_ir": torch.tensor(gt_ir, dtype=torch.float32),

            # Existências
            "exist": torch.tensor(exist_global_list, dtype=torch.float32),
            "exist_vis": torch.tensor(exist_vis_list, dtype=torch.float32),
            "exist_ir": torch.tensor(exist_ir_list, dtype=torch.float32),

            "seq_name": seq_name
        }


def collate_fn_superior(batch):
    return {
        "x_input": torch.stack([b["x_input"] for b in batch]),
        "boxes_vis": torch.stack([b["boxes_vis"] for b in batch]),
        "boxes_ir": torch.stack([b["boxes_ir"] for b in batch]),

        "exist": torch.stack([b["exist"] for b in batch]),
        "exist_vis": torch.stack([b["exist_vis"] for b in batch]),
        "exist_ir": torch.stack([b["exist_ir"] for b in batch]),

        "seq_names": [b["seq_name"] for b in batch]
    }
