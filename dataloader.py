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
    - Mantém dimensões originais das imagens.
    - GTs em pixels reais [x, y, w, h].
    - Retorna frames como lista para permitir resoluções variadas.
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

        self.split_dir = os.path.join(root_dir, split)

        folder_names = sorted([
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ])
        self.sequences = folder_names

        self.annotation_cache = {}
        print(f"Carregando anotações do split {split} na RAM...")

        for seq in folder_names:
            seq_path = os.path.join(self.split_dir, seq)

            with open(os.path.join(seq_path, "visible.json"), "r") as f:
                v_data = json.load(f)

            with open(os.path.join(seq_path, "infrared.json"), "r") as f:
                ir_data = json.load(f)

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

        if self.max_frames_per_seq is not None:
            valid_indices = valid_indices[:self.max_frames_per_seq]

        if len(valid_indices) <= self.temporal_window:
            selected_indices = valid_indices + [valid_indices[-1]] * (self.temporal_window - len(valid_indices))
        else:
            start_idx = random.randint(0, len(valid_indices) - self.temporal_window)
            selected_indices = valid_indices[start_idx : start_idx + self.temporal_window]

        vis_frames, ir_frames = [], []
        gt_vis, gt_ir = [], []
        exist_vis_list, exist_ir_list = [], []

        for k in selected_indices:
            # Carrega imagens originais
            v_img = Image.open(os.path.join(seq_path, "visible", data["files_vis"][k])).convert("RGB")
            ir_img = Image.open(os.path.join(seq_path, "infrared", data["files_ir"][k])).convert("L")

            # Aplica transform apenas se existir (ex: ToTensor), sem Resize.
            if self.transform:
                v_t = self.transform["visible"](v_img) if isinstance(self.transform, dict) else self.transform(v_img)
                ir_t = self.transform["infrared"](ir_img) if isinstance(self.transform, dict) else self.transform(ir_img)
            else:
                v_t = torch.from_numpy(np.array(v_img)).permute(2,0,1).float() / 255.0
                ir_t = torch.from_numpy(np.array(ir_img)).unsqueeze(0).float() / 255.0

            vis_frames.append(v_t)
            ir_frames.append(ir_t)

            # GTs Reais em Pixels
            gt_vis.append(data["gt_rect_vis"][k] if data["exist_vis"][k] > 0 else [0,0,0,0])
            gt_ir.append(data["gt_rect_ir"][k] if data["exist_ir"][k] > 0 else [0,0,0,0])

            exist_vis_list.append(data["exist_vis"][k])
            exist_ir_list.append(data["exist_ir"][k])

        return {
            "vis_frames": vis_frames, # Lista de Tensores (C, H, W)
            "ir_frames": ir_frames,   # Lista de Tensores (C, H, W)
            "boxes_vis": torch.tensor(gt_vis, dtype=torch.float32),
            "boxes_ir": torch.tensor(gt_ir, dtype=torch.float32),
            "exist_vis": torch.tensor(exist_vis_list, dtype=torch.float32),
            "exist_ir": torch.tensor(exist_ir_list, dtype=torch.float32),
            "seq_name": seq_name
        }

def collate_fn_superior(batch):
    '''
    Collate que mantém frames como listas de listas, 
    já que as resoluções podem variar entre sequências.
    '''
    return {
        "vis_frames": [b["vis_frames"] for b in batch], # Lista[Batch][Tempo]
        "ir_frames": [b["ir_frames"] for b in batch],   # Lista[Batch][Tempo]
        "boxes_vis": torch.stack([b["boxes_vis"] for b in batch]),
        "boxes_ir": torch.stack([b["boxes_ir"] for b in batch]),
        "exist_vis": torch.stack([b["exist_vis"] for b in batch]),
        "exist_ir": torch.stack([b["exist_ir"] for b in batch]),
        "seq_names": [b["seq_name"] for b in batch]
    }