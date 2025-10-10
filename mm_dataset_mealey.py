import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from collections import defaultdict
import math
import pandas as pd 
from typing import List, Tuple, Dict, Any

# Configuração de Log: Mude para False para desligar as mensagens de Padding/Truncamento
LOG_INCONSISTENCY = False 

# Dimensão padrão para features NPY que o Transformer espera
DEFAULT_FEATURE_DIM = 512 

class MMDataset(Dataset):
    """
    Custom PyTorch Dataset para carregar SEQUENCIAS de frames (chunks de tamanho T).
    Implementa a logica de Ground Truth do estado simbolico (Mealy Transducer)
    usando a latencia configurada.

    Estados Mealy:
    - 0: Background
    - 1: Candidate (Detected, mas não persistiu por L frames futuros)
    - 2: Confirmed (Detected E persistiu por L frames futuros)
    - -1: Absent/Ignore (GT de classe original é -1)
    """
    def __init__(self, split_file_path, data_root, config, phase='train'):
        self.data_root = data_root
        self.config = config
        self.phase = phase
        
        # Parâmetros principais
        self.img_size = config.get('image_size', 224)
        self.feature_dim = config.get('modal_feature_dim', DEFAULT_FEATURE_DIM)
        self.sequence_length = config.get('sequence_length', 10) 
        # Latencia para Confirmed (Estado 2): N frames ADIANTE devem ser Detectados (1)
        self.latency_frames = config.get('latency_frames', 3) 

        # O split_folder aponta para a pasta onde estao as sequencias (e.g., .../MMNTT/train)
        if self.phase == 'test' and 'val' in data_root: 
            self.split_folder = data_root 
        elif self.phase in ['val', 'test']:
            self.split_folder = data_root 
        else:
            # Se for treino, presumimos que a estrutura e DATA_ROOT/phase
            self.split_folder = os.path.join(self.data_root, self.phase) 

        # 1. Carregamento dos metadados de split
        with open(split_file_path, 'r') as f:
            all_annotations = [line.strip() for line in f.readlines() if line.strip()]
        
        # 2. Agrupamento por Sequencia e Filtragem de Debug
        self.annotations_by_sequence = defaultdict(list)
        max_seq_index = config.get('max_seq_index_for_debug', None)
        
        # O split_file_path lista seqXX,timestamp. Vamos agrupar os frames validos.
        for line in all_annotations:
            if ',' not in line:
                # Caso o split seja de seq folders (Teste Dummy), precisamos listar os frames
                sequence_folder = line
                image_path = os.path.join(self.split_folder, sequence_folder, 'Image')
                if os.path.isdir(image_path):
                    for filename in os.listdir(image_path):
                        if filename.endswith('.png'):
                            timestamp_id = filename.rsplit('.', 1)[0]
                            self.annotations_by_sequence[sequence_folder].append(timestamp_id)
            else:
                sequence_folder, timestamp_id = line.split(',')
                self.annotations_by_sequence[sequence_folder].append(timestamp_id)

            # Logica de Limite de Sequencia para Debug
            if max_seq_index is not None:
                try:
                    seq_number = int(sequence_folder[3:])
                    if seq_number > max_seq_index:
                        if sequence_folder in self.annotations_by_sequence:
                            del self.annotations_by_sequence[sequence_folder]
                        continue
                except ValueError:
                    continue 

        # 3. Mapeamento de GT Simbolico (Mealy State) e Indexacao de Chunks
        self.symbolic_gt_map = {}
        self.sequence_chunks = []
        
        for seq_folder, timestamp_list in self.annotations_by_sequence.items():
            if not timestamp_list:
                continue

            # Garante ordem temporal
            timestamp_list.sort(key=float) 
            
            # 3.1. Carrega todos os GTs de Posicao e Classe (Raw) para a sequencia
            raw_gt_pos_list = []
            raw_gt_class_list = []
            
            for timestamp_id in timestamp_list:
                sample_base_path = os.path.join(self.split_folder, seq_folder)
                
                # Prioriza a leitura de GT de classe NPY (0, 1 ou -1)
                gt_class = self._load_npy(os.path.join(sample_base_path, 'class'), timestamp_id, is_class=True)
                raw_gt_class_list.append(gt_class.item())
                
                # Carrega GT Posicao (usado apenas para referencia ou no modo de regressor puro)
                gt_pos = self._load_npy(os.path.join(sample_base_path, 'ground_truth'), timestamp_id, gt_dim=3)
                raw_gt_pos_list.append(gt_pos) 
                
            # 3.2. Calcula o GT Simbolico (Mealy State: 0, 1, 2, -1) para a sequencia completa
            symbolic_gt_array = self._calculate_mealy_gt_for_sequence(raw_gt_class_list, self.latency_frames)
            self.symbolic_gt_map[seq_folder] = (symbolic_gt_array, raw_gt_pos_list, timestamp_list)
            
            # 3.3. Cria os chunks de indices (seq_folder, start_index)
            num_frames = len(timestamp_list)
            num_chunks = math.floor(num_frames / self.sequence_length)
            
            for i in range(num_chunks):
                start = i * self.sequence_length
                # Armazenamos a pasta e o indice de inicio
                self.sequence_chunks.append((seq_folder, start)) 
                
        print(f"Carregado {len(self.annotations_by_sequence)} sequencias. Total de {len(self.sequence_chunks)} chunks de tamanho {self.sequence_length}.")
        if self.phase == 'train' and max_seq_index is not None:
            print(f"DEBUG: Limitando a sequencias ate 'seq{max_seq_index:04d}'.")

    def __len__(self):
        # O tamanho do dataset agora e o numero total de chunks
        return len(self.sequence_chunks)

    def __getitem__(self, index):
        # Retorna um chunk temporal de tamanho T (self.sequence_length)
        seq_folder, start_index = self.sequence_chunks[index]
        
        symbolic_gt_array, raw_gt_pos_list, timestamp_list = self.symbolic_gt_map[seq_folder]
        
        end_index = start_index + self.sequence_length
        chunk_timestamps = timestamp_list[start_index:end_index]
        
        sequence_data = defaultdict(list)
        
        for t, timestamp_id in enumerate(chunk_timestamps):
            sample_base_path = os.path.join(self.split_folder, seq_folder) 
            
            # --- CARREGAMENTO DE INPUTS ---
            sequence_data['image'].append(self._load_image(os.path.join(sample_base_path, 'Image'), timestamp_id))
            sequence_data['lidar'].append(self._load_npy(os.path.join(sample_base_path, 'lidar_360'), timestamp_id, feature_dim=self.feature_dim))
            sequence_data['radar'].append(self._load_npy(os.path.join(sample_base_path, 'radar_enhance_pcl'), timestamp_id, feature_dim=self.feature_dim))
            # Ajuste no caminho para features de audio, que estao na raiz dos dados
            # Assume que a estrutura é: DATA_ROOT/audio_features/seq_folder/timestamp.npy
            sequence_data['audio'].append(self._load_npy(os.path.join(self.data_root, 'audio_features', seq_folder), timestamp_id, feature_dim=self.feature_dim)) 
            
            # --- CARREGAMENTO DE LABELS (Ground Truth) ---
            
            # 1. GT Class (Symbolic Mealy State): Pre-calculado [T]
            # O GT ja esta no formato 0, 1, 2 ou -1.
            gt_class = torch.tensor(symbolic_gt_array[start_index + t], dtype=torch.long)
            sequence_data['gt_class'].append(gt_class)
            
            # 2. GT Posicao: Do NPY raw
            gt_pos = raw_gt_pos_list[start_index + t]
            # Garante que gt_pos não é um tensor escalar vazio
            if gt_pos.dim() == 0 or gt_pos.numel() == 0: 
                gt_pos = torch.zeros(3, dtype=torch.float) 
            
            sequence_data['gt_pos'].append(gt_pos)

            # 3. Metadados
            sequence_data['timestamp'].append(f"{seq_folder},{timestamp_id}") 
            
        # O timestamp deve ser uma lista de strings, nao um tensor empilhado
        timestamps = sequence_data.pop('timestamp')
        
        # Concatena os tensores na dimensao do tempo (T). O shape sera [T, ...]
        stacked_data = {
            key: torch.stack(value, dim=0) for key, value in sequence_data.items()
        }
        
        # Adiciona o metadata de volta como uma lista de strings
        stacked_data['timestamp'] = timestamps 
        
        # O shape final de cada tensor em stacked_data (exceto 'timestamp') e: [T, BATCH_DIMENSIONS]
        return stacked_data

    # --- FUNCOES DE CALCULO E AUXILIARES ---
    
    def _calculate_mealy_gt_for_sequence(self, raw_gt_class_list: List[int], latency: int) -> List[int]:
        """
        Converte a lista de Ground Truths de classe brutos (0, 1, -1) para 
        os estados simbolicos do Transdutor Mealy (0: Bkg, 1: Cand, 2: Conf, -1: Absent).
        
        Regra:
        - Se o GT bruto for -1 (Absent/Unknown), o GT simbolico e -1 (ignorado pela loss).
        - Se o GT bruto for 0 (Background), o GT simbolico e 0.
        - Se o GT bruto for 1 (Detected):
            - É verificado se a deteccao persiste por 'latency' quadros ADIANTE (i+1 até i+L).
            - Se sim (Confirmed), o GT simbolico e 2.
            - Se nao (Candidate), o GT simbolico e 1.
        """
        seq_len = len(raw_gt_class_list)
        symbolic_gt = []

        # 0: Background, 1: Candidate, 2: Confirmed, -1: Absent/Ignore
        
        for i in range(seq_len):
            raw_class = raw_gt_class_list[i]
            
            if raw_class == -1:
                # O estado 'Absent' (Drone fora do frame de treino/nao rastreado) 
                # deve ser ignorado pela loss, usando -1.
                symbolic_gt.append(-1) 
            
            elif raw_class == 0:
                # Estado 'Background'
                symbolic_gt.append(0)
                
            elif raw_class == 1:
                # Estado 'Detected' (1). Aplicamos a logica de latencia (look-ahead).
                
                is_confirmed = True
                # Verifica se os proximos 'latency' frames (i+1 a i+latency) tambem sao 'Detected' (1)
                for j in range(1, latency + 1):
                    # Se o indice ultrapassar o fim da sequencia OU o proximo frame NAO for 1
                    if (i + j >= seq_len) or (raw_gt_class_list[i + j] != 1):
                        is_confirmed = False
                        break
                        
                if is_confirmed:
                    # Se persistir por L frames futuros: Confirmed
                    symbolic_gt.append(2)
                else:
                    # Se nao persistir por L frames: Candidate
                    symbolic_gt.append(1)
            
            else:
                # Fallback para qualquer outro valor inesperado, trata como Background
                symbolic_gt.append(0)

        return symbolic_gt


    def _load_image(self, base_path, timestamp_id):
        img_path = os.path.join(base_path, f"{timestamp_id}.png")
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            # Normalização (0-255 -> 0.0-1.0)
            return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            if LOG_INCONSISTENCY: print(f"AVISO: Falha ao carregar imagem {img_path}. {e}")
            # Retorna tensor de zeros em caso de falha (3 canais, 224x224)
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.float)

    def _load_npy(self, base_path, timestamp_id, feature_dim=None, gt_dim=None, is_class=False):
        npy_path = os.path.join(base_path, f"{timestamp_id}.npy")
        
        try:
            data = np.load(npy_path, allow_pickle=True)
            
            if is_class:
                # O GT de classe (original) é um escalar (0, 1 ou -1)
                return torch.tensor(data.item(), dtype=torch.long)
            
            data_tensor = torch.from_numpy(data).float()
            if data_tensor.ndim > 1:
                data_tensor = data_tensor.flatten()
            
            # Logica de Padding/Truncamento para Features
            if feature_dim:
                current_length = data_tensor.size(0)
                if current_length > feature_dim:
                    data_tensor = data_tensor[:feature_dim]
                    if LOG_INCONSISTENCY: print(f"AVISO: {npy_path} Truncado de {current_length} para {feature_dim}")
                elif current_length < feature_dim:
                    padding_needed = feature_dim - current_length
                    padding = torch.zeros(padding_needed, dtype=torch.float)
                    data_tensor = torch.cat((data_tensor, padding), 0)
                    if LOG_INCONSISTENCY: print(f"AVISO: {npy_path} Padding de {current_length} para {feature_dim}")
                
                return data_tensor.reshape(feature_dim)
            
            # Logica para GT de Posicao (normalmente dim 3)
            if gt_dim:
                if data_tensor.size(0) >= gt_dim:
                    return data_tensor[:gt_dim].float()
                else:
                    padding_needed = gt_dim - data_tensor.size(0)
                    padding = torch.zeros(padding_needed, dtype=torch.float)
                    return torch.cat((data_tensor, padding), 0).float()
                
            return data_tensor.float()

        except Exception as e:
            # Fallback em caso de arquivo NPY ausente
            if LOG_INCONSISTENCY: print(f"AVISO: Falha ao carregar NPY {npy_path}. {e}")
            if is_class:
                # Retorna -1 para classe se o arquivo de GT estiver ausente (Ignorar)
                return torch.tensor(-1, dtype=torch.long) 
            if gt_dim:
                return torch.zeros(gt_dim, dtype=torch.float) 
            if feature_dim:
                return torch.zeros(feature_dim, dtype=torch.float)
            return torch.zeros(1, dtype=torch.float)
