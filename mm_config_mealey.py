import os
import random

# --- CONFIGURAÇÕES DE CAMINHO (AJUSTE CRÍTICO AQUI) ---
# DATA_ROOT deve apontar DIRETAMENTE para a pasta que contém 'train', 'val' e 'test' (se existir).

# >>> AJUSTE ESSA LINHA PARA O SEU CAMINHO EXATO:
# Exemplo se as pastas 'train' e 'val' estiverem DENTRO de MMNTT:
DATA_ROOT = r'C:\Users\Micro\Documents\sourcecode\MMNTT' 
# Se as pastas 'train' e 'val' estiverem DENTRO de uma subpasta 'MMNTT_Dataset' no MMNTT:
# DATA_ROOT = r'C:\Users\Micro\Documents\sourcecode\MMNTT\MMNTT_Dataset'

# Se você estiver usando o caminho absoluto que parece estar correto, use o primeiro exemplo acima.

# Pasta onde sero salvos os arquivos de split (seqXX,timestamp)
SPLIT_DIR = os.path.join(DATA_ROOT, 'annotation_splits') 

# Cria o diretório de splits se não existir
os.makedirs(SPLIT_DIR, exist_ok=True)

# Path dos arquivos de split (definidos como constantes)
TRAIN_SPLIT_PATH = os.path.join(SPLIT_DIR, "train.txt")
VAL_SPLIT_PATH = os.path.join(SPLIT_DIR, "val.txt")
TEST_SPLIT_PATH = os.path.join(SPLIT_DIR, "test.txt")


# --- CONFIGURAÇÕES DE TREINAMENTO E HARDWARE ---
CONFIG = {
    # Treinamento
    "batch_size": 8, 
    "num_workers": 0, # Alterado para 0 como padrão para evitar problemas de multiprocessing (comum em Windows)
    "learning_rate": 1e-4, 
    "epochs": 20, 
    "dropout_rate": 0.1, # Adicionado dropout rate para o Transformer
    
    # Arquitetura do Modelo
    "image_size": 224, 
    "modal_feature_dim": 512, 
    "transformer_hidden_dim": 256,
    "num_classes": 4, # quatro tipos de drones.
    
    # Sequencialidade (BPTT)
    "sequence_length": 10, # T: O número de passos de tempo para BPTT
    # Usado para determinar o GT do estado simblico (Candidate -> Confirmed)
    "latency_frames": 3, 
    
    # --- Limite de Sequência para Debug ---
    "max_seq_index_for_debug": 1, 
}

# --- ATUALIZAÇÃO FINAL DO CONFIG COM OS PATHS ---
CONFIG["data_root"] = DATA_ROOT
CONFIG["train_split_path"] = TRAIN_SPLIT_PATH
CONFIG["val_split_path"] = VAL_SPLIT_PATH
CONFIG["test_split_path"] = TEST_SPLIT_PATH


# --- FUNÇÃO AUXILIAR PARA GERAÇÃO DE SPLITS ---

def generate_splits(data_root, split_dir, train_ratio=0.70, val_ratio=0.15):
    """
    Gera arquivos de anotação (train/val/test.txt) no formato 'seqXX,timestamp'
    lendo a estrutura de diretórios do MMAUD ('Data/train/seqXX/Image/').
    """
    
    # Verifica se os splits já existem
    if os.path.exists(TRAIN_SPLIT_PATH) and os.path.exists(VAL_SPLIT_PATH) and os.path.exists(TEST_SPLIT_PATH):
        print("Splits já existem. Pulando a regeneração.")
        return
    
    # Tenta ler a pasta 'train' no DATA_ROOT
    train_folder = os.path.join(data_root, 'train')
    all_lines = []

    if os.path.isdir(train_folder):
        for seq_folder in os.listdir(train_folder):
            seq_path = os.path.join(train_folder, seq_folder)
            image_path = os.path.join(seq_path, 'Image')
            
            # Adiciona frames se for um diretório de sequência e tiver imagens
            if os.path.isdir(seq_path) and os.path.isdir(image_path):
                for filename in os.listdir(image_path):
                    if filename.endswith('.png'):
                        timestamp_id = filename.rsplit('.', 1)[0]
                        all_lines.append(f"{seq_folder},{timestamp_id}")
    
    if not all_lines:
        # Se falhar, é aqui que a mensagem de ERRO é mostrada
        print(f"ERRO: Nenhuma amostra encontrada em {train_folder}. Verifique a estrutura de pastas.")
        return

    random.shuffle(all_lines)
    total = len(all_lines)
    
    # Ratios de divisão
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    train_split = all_lines[:train_size]
    val_split = all_lines[train_size : train_size + val_size]
    test_split = all_lines[train_size + val_size :]

    # Salvamento dos splits
    splits = {
        'train.txt': (train_split, TRAIN_SPLIT_PATH), 
        'val.txt': (val_split, VAL_SPLIT_PATH), 
        'test.txt': (test_split, TEST_SPLIT_PATH)
    }
    
    for name, (data, path) in splits.items():
        with open(path, 'w') as f:
            f.write('\n'.join(data))
        print(f"Criado {name} com {len(data)} amostras.")
    
if __name__ == "__main__":
    generate_splits(DATA_ROOT, SPLIT_DIR)
