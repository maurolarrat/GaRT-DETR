
import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import os

from SuperiorDETR import SuperiorDETR

# ============================================================
# CONFIGURAÇÕES
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/superior_detr_best.pth"
VIDEO_VIS_PATH = r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT\test\20190926_134054_1_1\visible.mp4"
VIDEO_IR_PATH = r"C:\Users\Micro\Documents\sourcecode\Anti-UAV-RGBT\test\20190926_134054_1_1\infrared.mp4"
OUTPUT_PATH = "resultado_rgbt_final.mp4"

# 1. Carregar Modelo
model = SuperiorDETR(d_model=256, n_queries=20).to(DEVICE)
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

if 'model_state_dict' in state_dict:
    model.load_state_dict(state_dict['model_state_dict'])
else:
    model.load_state_dict(state_dict)
model.eval()

# NORMALIZAÇÃO IDENTICA AO TREINO
transform_vis = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_ir = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.449], [0.226])
])

def post_process_box(box_norm, orig_w, orig_h):
    cx, cy, w, h = box_norm
    x1 = (cx - 0.5 * w) * orig_w
    y1 = (cy - 0.5 * h) * orig_h
    x2 = (cx + 0.5 * w) * orig_w
    y2 = (cy + 0.5 * h) * orig_h
    return [int(x1), int(y1), int(x2), int(y2)]

# ============================================================
# SETUP DOS VÍDEOS COM DIMENSÕES INDEPENDENTES
# ============================================================
cap_vis = cv2.VideoCapture(VIDEO_VIS_PATH)
cap_ir = cv2.VideoCapture(VIDEO_IR_PATH)

fps = int(cap_vis.get(cv2.CAP_PROP_FPS))
frame_count = int(cap_vis.get(cv2.CAP_PROP_FRAME_COUNT))

# Captura resoluções reais de cada arquivo
w_v = int(cap_vis.get(cv2.CAP_PROP_FRAME_WIDTH))   # 1920
h_v = int(cap_vis.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 1080
w_i = int(cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))    # 640
h_i = int(cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))   # 512

# O vídeo de saída será 2x a largura do RGB (1920*2) pela altura do RGB (1080)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w_v * 2, h_v))

pbar = tqdm(total=frame_count, desc="Gravando RGBT")

while cap_vis.isOpened() and cap_ir.isOpened():
    ret_v, frame_v = cap_vis.read()
    ret_i, frame_i = cap_ir.read()
    
    if not ret_v or not ret_i:
        break

    # PREPARAÇÃO (NORMALIZAÇÃO)
    img_v_rgb = cv2.cvtColor(frame_v, cv2.COLOR_BGR2RGB)
    img_i_gray = cv2.cvtColor(frame_i, cv2.COLOR_BGR2GRAY) if len(frame_i.shape) == 3 else frame_i.copy()

    t_v = transform_vis(img_v_rgb).to(DEVICE)
    t_i = transform_ir(img_i_gray).to(DEVICE)

    with torch.no_grad():
        # Batch=1, Tempo=1
        outputs = model([[t_v]], [[t_i]])
        conf_idx = torch.argmax(outputs["exist"][0, 0]) 
        
        box_vis = outputs["pred_boxes_vis"][0, 0, conf_idx].cpu().numpy()
        box_ir = outputs["pred_boxes_ir"][0, 0, conf_idx].cpu().numpy()
        
        # Cada caixa usa sua própria largura/altura original
        coords_v = post_process_box(box_vis, w_v, h_v) # Usa 1920x1080
        coords_i = post_process_box(box_ir, w_i, h_i)  # Usa 640x512

    # DESENHO
    # RGB
    cv2.rectangle(frame_v, (coords_v[0], coords_v[1]), (coords_v[2], coords_v[3]), (0, 255, 0), 2)
    cv2.putText(frame_v, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # IR (Garante 3 canais para desenhar em vermelho)
    if len(frame_i.shape) == 2:
        ir_canvas = cv2.cvtColor(frame_i, cv2.COLOR_GRAY2BGR)
    else:
        ir_canvas = frame_i.copy()

    cv2.rectangle(ir_canvas, (coords_i[0], coords_i[1]), (coords_i[2], coords_i[3]), (0, 0, 255), 2)
    cv2.putText(ir_canvas, "IR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # CONCATENAÇÃO LADO A LADO
    # Redimensiona o IR para o tamanho do RGB para caber no vídeo final
    ir_resized = cv2.resize(ir_canvas, (w_v, h_v))
    
    combined = np.hstack((frame_v, ir_resized))
    out.write(combined)
    pbar.update(1)

pbar.close()
cap_vis.release()
cap_ir.release()
out.release()
print(f"\n[+] Vídeo salvo com dimensões independentes: {OUTPUT_PATH}")
