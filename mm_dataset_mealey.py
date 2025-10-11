import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # Adicionado tqdm para o pre-carregamento do GT
import os
import numpy as np
from PIL import Image
from collections import defaultdict
import math
from typing import List, Tuple, Dict, Any
import random
from sklearn.model_selection import train_test_split 
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data.dataloader import default_collate

# --- NOVAS CONSTANTES PARA PCL FEATURE EXTRACTION ---
PCL_POINT_DIM = 3 
PCL_NUM_STATS = 3 
DEFAULT_FEATURE_DIM = PCL_POINT_DIM * PCL_NUM_STATS
MAX_POINTS_TO_PROCESS = 20000 
# ----------------------------------------------------

# Configuracao de Log: Mude para True para ver as mensagens de sincronizacao/fallback
LOG_INCONSISTENCY = False 

# Lista das subpastas/modalidades fixas dentro de 'Mavic3'
MODALITIES = [
    'ground_truth',
    'image',
    'lidar_360',
    'livox_avia',
    'radar_enhance_pcl'
]

# Mapeamento do nome da pasta para a chave no dicionario de saida
OUTPUT_KEY_MAP = {
    'ground_truth': 'gt_pos', 
    'image': 'image',
    'lidar_360': 'lidar_360',
    'livox_avia': 'livox_avia',
    'radar_enhance_pcl': 'radar' 
}

# =========================================================================
# CLASSE AUXILIAR DE NORMALIZAÇÃO DE POSIÇÃO
# =========================================================================
class PositionNormalizer:
    """ Armazena e aplica as estatísticas de normalização do GT de Posição. """
    def __init__(self, pos_data: np.ndarray = None):
        if pos_data is not None:
            # Calcular a media e std deviation sobre o conjunto de TREINO
            self.mean = np.mean(pos_data, axis=0, keepdims=True)
            self.std = np.std(pos_data, axis=0, keepdims=True)
            # Evita divisão por zero
            self.std[self.std == 0] = 1e-6 
        else:
            self.mean = None
            self.std = None

    def normalize(self, pos_tensor: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return pos_tensor
        
        # Converte np.array para torch.Tensor para a operação
        # Note que pos_tensor pode estar na GPU, o broadcast funciona se o normalizer estiver na CPU
        mean_t = torch.from_numpy(self.mean).float() 
        std_t = torch.from_numpy(self.std).float()

        # Garante que o tensor GT e as estatísticas tenham as mesmas dimensões
        return (pos_tensor - mean_t.to(pos_tensor.device)) / std_t.to(pos_tensor.device)

    def denormalize(self, pos_tensor: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return pos_tensor

        mean_t = torch.from_numpy(self.mean).float().to(pos_tensor.device)
        std_t = torch.from_numpy(self.std).float().to(pos_tensor.device)
        
        return pos_tensor * std_t + mean_t
# =========================================================================


class Mavic3Dataset(Dataset):
    # Definindo as chaves de PCLs para usar no __getitem__ e no collate_fn
    PCL_KEYS = ['lidar_360', 'livox_avia', 'radar'] 

    # CORREÇÃO CRÍTICA: Adicionar 'normalizer' como argumento e inicializar corretamente
    def __init__(self, data_root: str, all_timestamps: List[str], config: Dict[str, Any], normalizer: PositionNormalizer = None):
        super().__init__()
        self.data_root = data_root
        self.config = config
        self.img_size = config.get('image_size', 224)
        # O feature_dim do PCL sera 9, a menos que o config diga o contrario
        self.feature_dim = config.get('pcl_feature_dim', DEFAULT_FEATURE_DIM) 
        self.sequence_length = config.get('sequence_length', 1)
        self.all_timestamps = all_timestamps
        # CORREÇÃO: Recebe o objeto normalizer
        self.normalizer = normalizer 
        self.modality_paths = {mod: os.path.join(self.data_root, mod) for mod in MODALITIES}
        self.PCL_BRUTE_PATHS = ['lidar_360', 'livox_avia', 'radar_enhance_pcl']

        for mod, path in self.modality_paths.items():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Subpasta de modalidade '{mod}' nao encontrada em: {self.data_root}")

        print(f"Dataset inicializado com {len(self.all_timestamps)} amostras.")

    def __len__(self):
        if self.sequence_length > 1:
            return math.floor(len(self.all_timestamps) / self.sequence_length)
        else:
            return len(self.all_timestamps)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.sequence_length > 1:
            start_index = index * self.sequence_length
            end_index = start_index + self.sequence_length
            chunk_timestamps = self.all_timestamps[start_index:end_index]
        else:
            chunk_timestamps = [self.all_timestamps[index]]

        sequence_data = defaultdict(list)

        for timestamp_id in chunk_timestamps:
            sample_data = self._load_single_sample(timestamp_id)
            for key, value in sample_data.items():
                sequence_data[key].append(value)

        # ---------------------------------------------------------------------------------------
        # APENAS GT e IMAGEM serao empilhados. PCLs serao retornadas como lista de tensores.
        # ---------------------------------------------------------------------------------------
        stacked_data = {}
        for key, value_list in sequence_data.items():
            if key == 'timestamp':
                stacked_data['timestamp'] = sequence_data.get('timestamp', chunk_timestamps)
                continue
            
            # PCLs (que agora sao features de tamanho FIXO): Devem ser empilhadas como o resto
            if key in self.PCL_KEYS:
                # Se PCLs estao retornando features de tamanho FIXO (9), elas devem ser empilhadas aqui!
                stacked_data[key] = torch.stack(value_list, dim=0) 
            else:
                # DADOS FIXOS (Imagem, GT): Empilha ao longo da dimensao da sequencia (T)
                stacked_data[key] = torch.stack(value_list, dim=0)

        # Gera gt_class a partir de gt_pos
        gt_pos_tensor = stacked_data['gt_pos']
        # CORREÇÃO: Norma só deve ser calculada APÓS a normalização se a loss for normalizada
        # Assumindo que o modelo prediz a posição NORMALIZADA, a distância ainda funciona.
        distance_norm = torch.linalg.norm(gt_pos_tensor, dim=1) 
        stacked_data['gt_class'] = (distance_norm > 1e-6).long()

        return stacked_data

    def _load_single_sample(self, timestamp_id: str) -> Dict[str, Any]:
        sample_data = {}
        sample_data['timestamp'] = timestamp_id
        for modality, path in self.modality_paths.items():
            output_key = OUTPUT_KEY_MAP.get(modality, modality)
            
            if modality == 'image':
                sample_data[output_key] = self._load_image(path, timestamp_id)
            
            elif modality == 'ground_truth':
                # GT: Mantem padding/truncamento para garantir o tamanho de saida fixo (3)
                sample_data[output_key] = self._load_npy_exact(path, timestamp_id, target_dim=3)
            
            elif modality in self.PCL_BRUTE_PATHS:
                # PCLs: Carrega o cache de features ou gera o cache e retorna as features.
                sample_data[output_key] = self._load_pcl_data(path, timestamp_id)
        
        return sample_data

    # --- FUNÇÃO DE EXTRAÇÃO DE FEATURES (Com amostragem) ---
    def _extract_pcl_features(self, data_tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Calcula features estatísticas (Mean, Max, StdDev) e garante o tamanho de saida fixo."""
        
        N_points = data_tensor.size(0)
        
        # 1. Fallback para PCLs vazias/inválidas
        if N_points == 0 or data_tensor.ndim != 2 or data_tensor.size(1) < PCL_POINT_DIM:
            return torch.zeros(target_dim, dtype=torch.float)

        # 2. TRUNCAMENTO (Amostragem Aleatória)
        if N_points > MAX_POINTS_TO_PROCESS:
            # Garante amostragem determinística para caching consistente
            # Note: seed fixo no Python 'random' nao garante determinismo do torch.randperm
            indices = torch.randperm(N_points, generator=torch.Generator().manual_seed(42))[:MAX_POINTS_TO_PROCESS] 
            data_tensor = data_tensor[indices]
            if LOG_INCONSISTENCY: print(f"AVISO PCL: Amostrado para {MAX_POINTS_TO_PROCESS} pontos.")
        
        # 3. EXTRAÇÃO DE FEATURES ESTATÍSTICAS (Mean, Max, StdDev)
        data_tensor = data_tensor[:, :PCL_POINT_DIM] # Garantir que so usa as 3 dimensoes principais
        
        mean_features = torch.mean(data_tensor, dim=0) 
        max_features = torch.max(data_tensor, dim=0).values 
        std_features = torch.nan_to_num(torch.std(data_tensor, dim=0), 0.) 
        
        final_features = torch.cat([mean_features, max_features, std_features], dim=0)
        
        # 4. Ajuste final do tamanho (Garantir que a saida e target_dim)
        current_length = final_features.size(0)
        if current_length > target_dim:
            final_features = final_features[:target_dim]
        elif current_length < target_dim:
            padding_needed = target_dim - current_length
            padding = torch.zeros(padding_needed, dtype=torch.float)
            final_features = torch.cat((final_features, padding), 0)
            
        return final_features


    # --- FUNÇÃO PCL: CARREGAR/GERAR FEATURE (Cache Lazy) ---
    def _load_pcl_data(self, base_path: str, timestamp_id: str) -> torch.Tensor:
        """Tenta carregar feature .pt. Se nao existir, carrega .npy, gera o .pt, salva e retorna."""
        
        # 1. Define caminhos
        modality = base_path.split(os.sep)[-1]
        feature_path = os.path.join(base_path, f"{timestamp_id}.pt")

        # 2. Tenta carregar o cache (.pt)
        if os.path.exists(feature_path):
            try:
                feature_tensor = torch.load(feature_path)
                return feature_tensor
            except Exception as e:
                if LOG_INCONSISTENCY: print(f"AVISO: Cache .pt corrompido para {modality} {timestamp_id}. Regenerando. {e}")
        
        # 3. Se o cache nao existe/falhou, carrega os NPYs brutos (operaçao lenta)
        paths = self._find_all_npy_by_int_prefix(base_path, timestamp_id)
        POINT_DIM = 3
        tensors = []
        
        if paths:
            for p in paths:
                try:
                    data = np.load(p, allow_pickle=True).astype(np.float32)
                    data_tensor = torch.from_numpy(data).float()
                    
                    if data_tensor.ndim >= 2 and data_tensor.shape[-1] >= POINT_DIM: 
                        tensors.append(data_tensor)
                    elif data_tensor.ndim == 1 and data_tensor.numel() > 0 and data_tensor.numel() % POINT_DIM == 0:
                        N_points = data_tensor.numel() // POINT_DIM
                        tensors.append(data_tensor.reshape(N_points, POINT_DIM))
                except Exception as e:
                    if LOG_INCONSISTENCY: print(f"Falha ao carregar PCL bruto {p}: {e}")
                    
            data_tensor_bruto = torch.cat(tensors, dim=0) if tensors else torch.zeros((0, POINT_DIM), dtype=torch.float)
        else:
            data_tensor_bruto = torch.zeros((0, POINT_DIM), dtype=torch.float)
            
        # 4. Extrai features
        final_features = self._extract_pcl_features(data_tensor_bruto, self.feature_dim)
        
        # 5. Salva o cache (.pt) e retorna
        try:
            torch.save(final_features, feature_path)
            if LOG_INCONSISTENCY: print(f"Cache gerado e salvo com sucesso em {feature_path}")
        except Exception as e:
            if LOG_INCONSISTENCY: print(f"AVISO CRÍTICO: Falha ao salvar cache em {feature_path}. {e}")
            
        return final_features

    # --- FUNÇÃO DE BUSCA DE TODOS OS NPY PELO PREFIXO (PARTE INTEIRA) ---
    def _find_all_npy_by_int_prefix(self, base_path: str, timestamp_id: str) -> List[str]:
        # (Função mantida como no original, não alterada)
        timestamp_int = int(float(timestamp_id))
        candidates = []
        try:
            for filename in os.listdir(base_path):
                if filename.endswith('.npy'):
                    try:
                        if int(float(filename.rsplit('.',1)[0])) == timestamp_int:
                            candidates.append(os.path.join(base_path, filename))
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass

        return candidates
    
    # --- FUNCAO DE CARREGAMENTO DE IMAGEM (Usa o prefixo/loose) ---
    def _load_image(self, base_path: str, timestamp_id: str) -> torch.Tensor:
        """Busca o PRIMEIRO arquivo .png que tem a mesma parte inteira do timestamp (segundo)."""
        timestamp_int_str = timestamp_id.split('.')[0]
        
        found_path = None
        try:
            for filename in os.listdir(base_path):
                if filename.endswith('.png') and filename.startswith(timestamp_int_str):
                    found_path = os.path.join(base_path, filename)
                    break 
        except FileNotFoundError:
            pass 

        if found_path:
            try:
                img = Image.open(found_path).convert('RGB')
                img = img.resize((self.img_size, self.img_size))
                return torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0
            except Exception as e:
                if LOG_INCONSISTENCY: print(f"Falha ao carregar ou processar imagem {found_path}: {e}")
                return torch.zeros((3,self.img_size,self.img_size), dtype=torch.float)
        else:
            if LOG_INCONSISTENCY: print(f"Falha de sincronizacao (somente int) para imagem {timestamp_id}. Retornando ZEROS.")
            return torch.zeros((3,self.img_size,self.img_size), dtype=torch.float)
        
    # --- FUNCAO NPY: CARREGAMENTO EXATO (PARA GROUND TRUTH) ---
    def _load_npy_exact(self, base_path: str, timestamp_id: str, target_dim: int) -> torch.Tensor:
        """Carrega o Ground Truth usando o timestamp_id EXATO (referencia de indice) e aplica NORMALIZAÇÃO."""
        full_path = os.path.join(base_path, f"{timestamp_id}.npy")
        data_tensor = torch.zeros(0)

        if os.path.exists(full_path):
            try:
                data = np.load(full_path, allow_pickle=True)
                # Garante que o GT é (3,)
                data_tensor = torch.from_numpy(data).float().flatten()[:target_dim]
            except Exception as e:
                if LOG_INCONSISTENCY: print(f"Falha ao carregar ou processar GT exato {full_path}: {e}")
                data_tensor = torch.zeros(0)

        # 1. Fallback se não carregou
        if data_tensor.numel() == 0:
            if LOG_INCONSISTENCY: print(f"Falha na busca EXATA para GT {timestamp_id}. Retornando ZEROS.")
            return torch.zeros(target_dim) if target_dim else torch.zeros(1)
        
        # 2. Aplica padding se necessário (apenas para o caso raro de arquivo com < 3 dim)
        if data_tensor.numel() < target_dim:
            padding = torch.zeros(target_dim - data_tensor.numel())
            data_tensor = torch.cat([data_tensor, padding])
            
        # 3. CORREÇÃO CRÍTICA: Aplica NORMALIZAÇÃO no Ground Truth.
        if self.normalizer:
            # O normalizer espera (N_samples, D). Aqui N_samples = 1, D = 3.
            data_tensor = self.normalizer.normalize(data_tensor.unsqueeze(0)).squeeze(0)
            
        return data_tensor


# Funcao de divisao de dataset
def get_mavic3_datasets(data_root: str, config: Dict[str, Any], test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42) -> Dict[str, Mavic3Dataset]:
    """
    Funcao auxiliar para carregar todos os timestamps e dividir em Treino, Validacao e Teste.
    CORREÇÃO CRÍTICA: Pré-carrega o GT de treino para calcular as estatísticas de normalização.
    """
    gt_path = os.path.join(data_root, 'ground_truth')
    if not os.path.isdir(gt_path):
        raise FileNotFoundError(f"Diretorio de Ground Truth nao encontrado em {gt_path}")

    all_timestamps = sorted([f.rsplit('.',1)[0] for f in os.listdir(gt_path) if f.endswith('.npy')])
    if not all_timestamps:
        raise ValueError(f"Nenhum arquivo .npy encontrado em {gt_path}")

    train_val_timestamps, test_timestamps = train_test_split(all_timestamps, test_size=test_size, random_state=random_state, shuffle=True)
    val_ratio_in_train_val = val_size / (1 - test_size)
    train_timestamps, val_timestamps = train_test_split(train_val_timestamps, test_size=val_ratio_in_train_val, random_state=random_state, shuffle=True)

    print(f"\nDivisao do Dataset (Total: {len(all_timestamps)} amostras):")
    print(f" - Treino: {len(train_timestamps)}")
    print(f" - Validacao: {len(val_timestamps)}")
    print(f" - Teste: {len(test_timestamps)}")

    # ----------------------------------------------------------------------
    # CORREÇÃO 1: Pré-carregar GT de Treino para Estatísticas de Normalização
    # ----------------------------------------------------------------------
    train_gt_positions = []
    
    # Adicionado tqdm para visualização do processo, já que é lento.
    for timestamp_id in tqdm(train_timestamps, desc="Pre-carregando GT para normalização"):
        full_path = os.path.join(gt_path, f"{timestamp_id}.npy")
        try:
            # Carrega e flatten para garantir que seja (D,)
            data = np.load(full_path, allow_pickle=True).astype(np.float32).flatten()
            if data.size >= 3:
                train_gt_positions.append(data[:3])
        except Exception as e:
            if LOG_INCONSISTENCY: print(f"Falha ao pré-carregar GT {full_path}: {e}")

    if not train_gt_positions:
        # Fallback para o caso de falha de carregamento
        print("AVISO: Falha ao carregar Ground Truth de treino. Normalização desativada.")
        pos_normalizer = PositionNormalizer(np.zeros((1,3))) 
    else:
        train_gt_positions_np = np.stack(train_gt_positions) # Shape: (N_samples, 3)
        pos_normalizer = PositionNormalizer(train_gt_positions_np)
        print(f"Normalizador de Posição inicializado. Média: {pos_normalizer.mean.mean():.4f}, Std: {pos_normalizer.std.mean():.4f}")
    # ----------------------------------------------------------------------

    # CORREÇÃO 2: Passar o objeto normalizador para todos os Datasets
    return {
        'train': Mavic3Dataset(data_root, train_timestamps, config, normalizer=pos_normalizer),
        'val': Mavic3Dataset(data_root, val_timestamps, config, normalizer=pos_normalizer),
        'test': Mavic3Dataset(data_root, test_timestamps, config, normalizer=pos_normalizer)
    }