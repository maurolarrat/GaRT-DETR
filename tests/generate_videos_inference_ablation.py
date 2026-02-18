import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import os

from SuperiorDETR import SuperiorDETR

class AblationModelWrapper(torch.nn.Module):
    def __init__(self, model, mode="dual", device="cuda"):
        super().__init__()
        self.model = model
        self.mode = mode
        self.device = device
        # Estatísticas para re-normalização
        self.vis_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.vis_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        self.ir_mean = torch.tensor([0.449]).view(1, 1, 1).to(device)
        self.ir_std = torch.tensor([0.226]).view(1, 1, 1).to(device)

    def forward(self, vis_frames, ir_frames):
        # Lógica idêntica ao seu script de teste
        if self.mode == "ir_only":
            new_vis_frames = []
            for seq in ir_frames:
                new_seq = []
                for frame in seq:
                    grad_x = torch.abs(frame[:, :, 1:] - frame[:, :, :-1])
                    grad_x = F.pad(grad_x, (0, 1, 0, 0))
                    grad_y = torch.abs(frame[:, 1:, :] - frame[:, :-1, :])
                    grad_y = F.pad(grad_y, (0, 0, 0, 1))
                    pseudo_rgb = torch.cat([frame, grad_x, grad_y], dim=0)
                    new_seq.append(pseudo_rgb)
                new_vis_frames.append(new_seq)
            vis_frames = new_vis_frames

        elif self.mode == "visible_only":
            new_ir_frames = []
            for seq in vis_frames:
                new_seq = []
                for frame in seq:
                    frame_raw = frame * self.vis_std + self.vis_mean
                    thermal_sim, _ = torch.max(frame_raw, dim=0, keepdim=True)
                    thermal_sim = transforms.functional.gaussian_blur(thermal_sim, [5, 5], sigma=1.0)
                    thermal_sim = (thermal_sim - thermal_sim.min()) / (thermal_sim.max() - thermal_sim.min() + 1e-6)
                    thermal_sim = (thermal_sim - self.ir_mean) / self.ir_std
                    new_seq.append(thermal_sim)
                new_ir_frames.append(new_seq)
            ir_frames = new_ir_frames
            
        return self.model(vis_frames, ir_frames)

# ============================================================
# CONFIGURAÇÕES DE EXECUÇÃO
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/superior_detr_best.pth"
VIDEO_VIS_PATH = r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT\test\20190926_134054_1_1\visible.mp4"
VIDEO_IR_PATH = r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT\test\20190926_134054_1_1\infrared.mp4"

# MODOS: "dual", "ir_only", "visible_only"
MODE = "visible_only" 
OUTPUT_PATH = f"inferencia_completa_{MODE}.mp4"

def post_process_box(box_norm, orig_w, orig_h):
    cx, cy, w, h = box_norm
    x1, y1 = (cx - 0.5 * w) * orig_w, (cy - 0.5 * h) * orig_h
    x2, y2 = (cx + 0.5 * w) * orig_w, (cy + 0.5 * h) * orig_h
    return [int(x1), int(y1), int(x2), int(y2)]

# 1. Carregar Modelo Base
base_model = SuperiorDETR(d_model=256, n_queries=20).to(DEVICE)
base_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False))
base_model.eval()

# 2. Encapsular no seu Wrapper de Ablação
model = AblationModelWrapper(base_model, mode=MODE, device=DEVICE)

# 3. Setup de Vídeo
cap_vis = cv2.VideoCapture(VIDEO_VIS_PATH)
cap_ir = cv2.VideoCapture(VIDEO_IR_PATH)
fps = int(cap_vis.get(cv2.CAP_PROP_FPS))
w_v, h_v = int(cap_vis.get(3)), int(cap_vis.get(4))
w_i, h_i = int(cap_ir.get(3)), int(cap_ir.get(4))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_v * 2, h_v))

transform_vis = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform_ir = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.449], [0.226])])

pbar = tqdm(total=int(cap_vis.get(7)), desc=f"Gerando Vídeo [{MODE}]")

while cap_vis.isOpened() and cap_ir.isOpened():
    ret_v, frame_v = cap_vis.read()
    ret_i, frame_i = cap_ir.read()
    if not ret_v or not ret_i: break

    # Preparação (O Wrapper espera listas de listas como o Dataloader)
    img_v_rgb = cv2.cvtColor(frame_v, cv2.COLOR_BGR2RGB)
    img_i_gray = cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY) if len(frame_i.shape) == 3 else frame_i
    
    t_v = transform_vis(img_v_rgb).to(DEVICE)
    t_i = transform_ir(img_i_gray).to(DEVICE)

    with torch.no_grad():
        # Passamos como [[tensor]] para simular Batch=1, JanelaTemporal=1
        outputs = model([[t_v]], [[t_i]])
        
        conf_idx = torch.argmax(outputs["exist"][0, 0])
        coords_v = post_process_box(outputs["pred_boxes_vis"][0, 0, conf_idx].cpu().numpy(), w_v, h_v)
        coords_i = post_process_box(outputs["pred_boxes_ir"][0, 0, conf_idx].cpu().numpy(), w_i, h_i)
        
        g_v = outputs.get("gate_vis_avg", 0.5)
        g_i = outputs.get("gate_ir_avg", 0.5)

    # Desenho
    cv2.rectangle(frame_v, (coords_v[0], coords_v[1]), (coords_v[2], coords_v[3]), (0, 255, 0), 2)
    
    # Proteção para o canvas IR (OpenCV scn=3 error)
    ir_canvas = cv2.cvtColor(frame_i, cv2.COLOR_GRAY2BGR) if len(frame_i.shape) == 2 else frame_i.copy()
    cv2.rectangle(ir_canvas, (coords_i[0], coords_i[1]), (coords_i[2], coords_i[3]), (0, 0, 255), 2)

    # Textos Informativos
    cv2.putText(frame_v, f"MODE: {MODE} | Gate V: {g_v:.2f}", (10, 30), 2, 0.7, (255,255,255), 2)
    cv2.putText(ir_canvas, f"Gate I: {g_i:.2f}", (10, 30), 2, 0.7, (255,255,255), 2)

    # Concatenação e escrita
    combined = np.hstack((frame_v, cv2.resize(ir_canvas, (w_v, h_v))))
    out.write(combined)
    pbar.update(1)

cap_vis.release(); cap_ir.release(); out.release(); pbar.close()
