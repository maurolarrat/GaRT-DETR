# config.py

import torch

# --- Hiperparâmetros do Modelo e Treinamento ---

# Configurações do Treinamento
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 40
DROPOUT = 0.2

# Arquitetura (Baseada em Transformer/Visão)
PATCH_SIZE = 32
D_MODEL = 512
NUM_HEADS = 8       # D_MODEL=512 deve ser divisivel por NUM_HEADS=4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512 #1024

# --- Configurações de Ambiente e Dados ---

# Define o dispositivo (GPU se disponível, senão CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configurações do DataLoader
NUM_WORKERS = 0

# Configurações Específicas de Sequência/Dados
MAX_FRAMES_PER_SEQ = 10
CHANNEL_TO_PLOT = "visible" # Mude para "infrared" se quiser o IR