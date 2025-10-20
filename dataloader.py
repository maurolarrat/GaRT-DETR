import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np

# ==========================================================
# DATASET ANTI-UAV-RGBT COM AUDITORIA + GERAÇÃO DE VÍDEO
# ==========================================================
class AntiUAVRGBTDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", transform=None,
                 audit=False, filter_attributes: list = None, max_frames_per_seq: int = None):
        """
        Dataset Anti-UAV-RGBT
        - Lê todas as subpastas de train, val, test
        - Filtra frames inválidos (gt_rect incompleto)
        - Associa atributos das sequências com base em label_new/*.json
        - filter_attributes: lista de atributos que a sequência deve possuir (ex: ["OV", "FM"])
        - max_frames_per_seq: número máximo de frames a carregar por sequência
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform # transform = {"visible": transform_vis, "infrared": transform_ir}
        self.audit = audit
        self.filter_attributes = filter_attributes
        self.max_frames_per_seq = max_frames_per_seq

        self.split_dir = os.path.join(root_dir, split)
        self.label_file = os.path.join(root_dir, "label_new", f"{split}.json")

        # === Carrega atributos globais ===
        with open(self.label_file, "r") as f:
            self.seq_attributes = json.load(f)

        # === Busca recursiva por subpastas de sequência ===
        self.sequences = []
        for root, dirs, files in os.walk(self.split_dir):
            for d in dirs:
                seq_path = os.path.join(root, d)
                if (os.path.exists(os.path.join(seq_path, "visible")) and
                    os.path.exists(os.path.join(seq_path, "infrared")) and
                    os.path.exists(os.path.join(seq_path, "visible.json")) and
                    os.path.exists(os.path.join(seq_path, "infrared.json"))):

                    seq_attrs = self.seq_attributes.get(d, [])
                    if self.filter_attributes is None or any(attr in seq_attrs for attr in self.filter_attributes):
                        self.sequences.append(seq_path)

        self.sequences = sorted(self.sequences)

        if self.audit:
            print(f"[INFO] {split.upper()} → {len(self.sequences)} sequências encontradas com filtro {self.filter_attributes}")
            print(f"[INFO] Atributos carregados de: {self.label_file}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_path = self.sequences[idx]
        seq_name = os.path.basename(seq_path)

        vis_path = os.path.join(seq_path, "visible")
        ir_path = os.path.join(seq_path, "infrared")
        vis_anno = os.path.join(seq_path, "visible.json")
        ir_anno = os.path.join(seq_path, "infrared.json")

        vis_imgs = sorted(os.listdir(vis_path))
        ir_imgs = sorted(os.listdir(ir_path))

        with open(vis_anno, "r") as f:
            vis_data = json.load(f)
        with open(ir_anno, "r") as f:
            ir_data = json.load(f)

        if self.audit:
            print(f"\n[SEQ: {seq_name}]")
            print(f"Frames visíveis: {len(vis_imgs)} | infravermelhos: {len(ir_imgs)}")
            print(f"gt_rect visível: {len(vis_data['gt_rect'])} | exist: {len(vis_data['exist'])}")

        # === Filtra frames inválidos ===
        valid_indices = [i for i in range(len(vis_data["gt_rect"]))
                     if len(vis_data["gt_rect"][i]) == 4 and len(ir_data["gt_rect"][i]) == 4]
        
        if self.audit:
            print(f"Frames inválidos removidos: {len(vis_data['gt_rect']) - len(valid_indices)}")

        # === Limita frames ===
        if self.max_frames_per_seq is not None:
            step = max(1, len(valid_indices) // self.max_frames_per_seq)
            valid_indices = valid_indices[::step][:self.max_frames_per_seq]

        vis_tensors, ir_tensors = [], []
        gt_rect_vis_list, gt_rect_ir_list, exist_list = [], [], []

        # Obtém dimensões do primeiro frame (assumindo que são constantes)
        # Note: PyTorch/TensorFlow espera C x H x W
        
        # Carrega a primeira imagem para obter W e H
        first_vis_img = Image.open(os.path.join(vis_path, vis_imgs[0]))
        W_vis, H_vis = first_vis_img.size # PIL.Image.size retorna (W, H)
        first_ir_img = Image.open(os.path.join(ir_path, ir_imgs[0]))
        W_ir, H_ir = first_ir_img.size
        
        for i in valid_indices:
            v = Image.open(os.path.join(vis_path, vis_imgs[i])).convert("RGB")
            ir = Image.open(os.path.join(ir_path, ir_imgs[i])).convert("L")

            if self.transform:
                if isinstance(self.transform, dict):
                    v = self.transform['visible'](v)
                    ir = self.transform['infrared'](ir)
                else:
                    v = self.transform(v)
                    ir = self.transform(ir)

            vis_tensors.append(v)
            ir_tensors.append(ir)

            # GT
            gt_rect_vis_list.append(vis_data["gt_rect"][i])
            gt_rect_ir_list.append(ir_data["gt_rect"][i])

            # 'exist' continua compartilhado
            exist_list.append(vis_data["exist"][i])

        # --- CONVERSÃO E NORMALIZAÇÃO DO GT (CORREÇÃO APLICADA AQUI) ---
        
        # 1. VISÍVEL
        # [x, y, w, h] (pixel) -> [xc, yc, w, h] (normalizado)
        gt_rect_vis_px = torch.tensor(gt_rect_vis_list, dtype=torch.float32)
        
        # Converte x_top_left, y_top_left para xc, yc (em pixels)
        gt_rect_vis_px[:, 0] = gt_rect_vis_px[:, 0] + gt_rect_vis_px[:, 2] / 2  # xc_pixel = x + w/2
        gt_rect_vis_px[:, 1] = gt_rect_vis_px[:, 1] + gt_rect_vis_px[:, 3] / 2  # yc_pixel = y + h/2
        
        # Normaliza (xc/W, yc/H, w/W, h/H)
        gt_rect_vis = gt_rect_vis_px.clone()
        gt_rect_vis[:, [0, 2]] /= W_vis # x e w normalizados por W
        gt_rect_vis[:, [1, 3]] /= H_vis # y e h normalizados por H


        # 2. INFRAVERMELHO
        # [x, y, w, h] (pixel) -> [xc, yc, w, h] (normalizado)
        gt_rect_ir_px = torch.tensor(gt_rect_ir_list, dtype=torch.float32)
        
        # Converte x_top_left, y_top_left para xc, yc (em pixels)
        gt_rect_ir_px[:, 0] = gt_rect_ir_px[:, 0] + gt_rect_ir_px[:, 2] / 2 
        gt_rect_ir_px[:, 1] = gt_rect_ir_px[:, 1] + gt_rect_ir_px[:, 3] / 2
        
        # Normaliza (xc/W, yc/H, w/W, h/H)
        gt_rect_ir = gt_rect_ir_px.clone()
        gt_rect_ir[:, [0, 2]] /= W_ir 
        gt_rect_ir[:, [1, 3]] /= H_ir 

        # --- FIM DA CORREÇÃO ---
        
        vis_stack = torch.stack(vis_tensors)
        ir_stack = torch.stack(ir_tensors)
        exist_tensor = torch.tensor(exist_list, dtype=torch.float32)
        attributes = self.seq_attributes.get(seq_name, [])

        if self.audit:
            print(f"Frames válidos: {len(valid_indices)}")
            print(f"Atributos: {attributes}")
            print(f"Box VIS (Normalizada): {gt_rect_vis[0]}") # <-- Novo debug para verificar
    
        return {
            "visible": vis_stack,
            "infrared": ir_stack,
            "gt_rect_vis": gt_rect_vis,
            "gt_rect_ir": gt_rect_ir,
            "exist": exist_tensor,
            "seq_name": seq_name
        }
    
    # ... Restante do código (generate_video_from_sequence e collate_fn_multimodal)
    def generate_video_from_sequence(self, seq_name, output_path="output_seq.mp4", show_infrared=False):
        """
        Gera um vídeo combinando os frames visíveis (ou infravermelhos) com as boxes do JSON e
        exibe os atributos no canto superior da tela.
        """
        # === Localiza sequência ===
        seq_path = None
        for p in self.sequences:
            if os.path.basename(p) == seq_name:
                seq_path = p
                break
        if seq_path is None:
            raise ValueError(f"Sequência '{seq_name}' não encontrada em {self.split_dir}")

        vis_path = os.path.join(seq_path, "visible")
        ir_path = os.path.join(seq_path, "infrared")
        vis_json = os.path.join(seq_path, "visible.json")
        ir_json = os.path.join(seq_path, "infrared.json")

        # === Carrega JSONs ===
        with open(vis_json, "r") as f:
            vis_data = json.load(f)
        with open(ir_json, "r") as f:
            ir_data = json.load(f)

        vis_imgs = sorted(os.listdir(vis_path))
        
        # Usando cv2 para garantir o formato BGR esperado para escrita
        first_frame = cv2.imread(os.path.join(vis_path, vis_imgs[0])) 
        if first_frame is None:
             raise IOError(f"Não foi possível carregar o primeiro frame em {vis_path}")

        H, W = first_frame.shape[:2] # cv2 retorna (H, W)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # VideoWriter espera (W, H)
        out = cv2.VideoWriter(output_path, fourcc, 15, (W, H)) 

        attributes = self.seq_attributes.get(seq_name, [])
        attr_text = ", ".join(attributes)

        # === Percorre frames válidos nos dois canais ===
        for i in range(len(vis_data["gt_rect"])):
            rect_vis = vis_data["gt_rect"][i]
            rect_ir = ir_data["gt_rect"][i]

            # Pula frames inválidos em qualquer canal
            if len(rect_vis) != 4 or len(rect_ir) != 4:
                continue

            frame_path = os.path.join(ir_path if show_infrared else vis_path, vis_imgs[i])
            
            # Carrega e converte para BGR
            if show_infrared:
                 # Se for IR (monocromático), carregue como escala de cinza e converta para BGR
                frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = cv2.imread(frame_path)
            
            if frame is None:
                continue
            
            # Garante que o frame tem o tamanho correto antes de escrever
            if frame.shape[1] != W or frame.shape[0] != H:
                 frame = cv2.resize(frame, (W, H))

            exist_flag = vis_data["exist"][i]
            if exist_flag > 0:
                # Desenha a caixa de visível (pode ser alterado para IR se show_infrared=True)
                x, y, w, h = map(int, rect_vis)
                # Verifica se a caixa é razoável antes de desenhar
                if w > 0 and h > 0:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # === Adiciona texto ===
            cv2.putText(frame, f"Seq: {seq_name}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Atributos: {attr_text}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            out.write(frame)

        out.release()
        print(f"[VIDEO] Vídeo gerado: {output_path}")

def collate_fn_multimodal(batch):
    """
    Collate function para AntiUAVRGBTDataset + VisionTransformerMultimodal.
    Ajustes:
    - Trunca todas as sequências pelo menor número de frames.
    - Empilha tensores em batch.
    """
    # Determina o menor comprimento temporal
    min_len = min(sample["visible"].shape[0] for sample in batch)

    for sample in batch:
        # Trunca frames
        sample["visible"] = sample["visible"][:min_len]
        sample["infrared"] = sample["infrared"][:min_len]
        sample["gt_rect_vis"] = sample["gt_rect_vis"][:min_len]
        sample["gt_rect_ir"] = sample["gt_rect_ir"][:min_len]
        sample["exist"] = sample["exist"][:min_len]

    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
        else:
            collated[key] = [sample[key] for sample in batch]

    return collated

# ==========================================================
# EXEMPLO DE USO
# ==========================================================
if __name__ == "__main__":
    root = r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT"
    dataset = AntiUAVRGBTDataset(root, split="train", audit=False, filter_attributes=["FM", "SV","TS","LR","DBC","TC","IC","OC","OV","VE"])
    print(f"Total: {len(dataset)} sequências válidas")

    # Gerar vídeo de uma sequência específica
    dataset.generate_video_from_sequence("20190925_101846_1_1",
                                         output_path="demo_seq.mp4",
                                         show_infrared=False)
