import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
from typing import Dict, Any

# =========================================================================
# CONFIGURAÇÃO
# -------------------------------------------------------------------------
# IMPORT CONFIG (Assumimos que CONFIG está definido e tem as chaves necessárias)
from mm_config_mealey import CONFIG 

NUM_SYMBOLIC_STATES = 4 

# =========================================================================
# ENCODER DE PCL: PointNet Simplificado
# -------------------------------------------------------------------------

class PointNetEncoder(nn.Module):
    """
    Encoder simplificado para Nuvem de Pontos (PCL) baseado na arquitetura PointNet.
    Mapeia a entrada (L_max, D_point) para um vetor de feature fixo (D_feature).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # MLPs para mapear pontos individuais
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        # O último MLP de feature-map será tratado pela Proj_Modal no Transformer.
        self.out_dim = out_dim
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        # x tem a forma (B*T, L_max, D_point)
        # PointNet espera (B, D_point, L_max)
        x = x.transpose(2, 1) 
        
        x = self.mlp1(x) 
        # x agora é (B*T, D_feature, L_max)
        
        # Global Max Pooling (GMP): Cria o vetor de features fixo D_feature
        # A GMP ignora os zeros de padding (se a máscara for aplicada corretamente)
        # No entanto, a implementação padrão do GMP em PyTorch é robusta para zeros de padding.
        # A maneira mais limpa de lidar com zeros é garantir que eles sejam o valor mínimo.
        
        # Máscara (opcional, mas recomendado)
        if padding_mask is not None:
             # Expande a máscara para a dimensão da feature (D_feature)
             expanded_mask = padding_mask.unsqueeze(1).expand(-1, x.size(1), -1)
             # Substitui os valores de padding por um valor muito pequeno (-inf)
             x[expanded_mask] = -float('inf')
        
        # Max Pooling (redução da dimensão L_max)
        x = torch.max(x, 2)[0] # Resultado: (B*T, D_feature)
        
        # x = self.final_mlp(x) # Se precisarmos de um MLP final (não usamos aqui, apenas na Proj_Modal)
        
        return x # (B*T, D_feature)

# =========================================================================
# UTILITY: STOCHASTIC DEPTH (DropPath)
# ... (Mantido inalterado)

class DropPath(nn.Module):
    # ... (Implementação omitida por concisão, é a mesma) ...
    """Implementação simplificada de Stochastic Depth (DropPath) para regularização."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

# =========================================================================
# NEURAL MEALY CORE (Aprimorado)
# ... (Mantido inalterado)
class NeuralMealyLayer(nn.Module):
    # ... (Implementação omitida por concisão, é a mesma) ...
    """
    Simula o core Neural-Mealy com regularização aprimorada para coerência temporal.
    Introduz Dropout e Temperature Scaling para 'Soft Transitions'.
    """
    def __init__(self, hidden_dim, num_explainable_states, dropout_rate, temp_scale=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_explainable_states = num_explainable_states
        self.dropout_rate = dropout_rate
        self.state_updater = nn.GRUCell(hidden_dim, hidden_dim) 
        combined_dim = hidden_dim * 2
        self.transition_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.state_transition_head = nn.Linear(hidden_dim, num_explainable_states)
        self.final_class_head = nn.Linear(hidden_dim, 1) 
        self.temperature = nn.Parameter(torch.tensor(temp_scale), requires_grad=False)
        
    def forward(self, fused_embedding, prev_state=None):
        B = fused_embedding.size(0)
        if prev_state is None:
            prev_state = torch.zeros(B, self.hidden_dim, device=fused_embedding.device)
        next_state = self.state_updater(fused_embedding, prev_state)
        combined_input = torch.cat((fused_embedding, next_state), dim=1) 
        processed_input = self.transition_mlp(combined_input)
        pred_class_logits = self.final_class_head(processed_input)
        state_transition_logits_raw = self.state_transition_head(processed_input)
        state_transition_logits = state_transition_logits_raw / self.temperature
        return next_state, pred_class_logits, state_transition_logits


# =========================================================================
# MULTIMODAL TRANSFORMER (AJUSTADO)
# -------------------------------------------------------------------------

class MultiModalTransformer(nn.Module):
    """
    Modelo Multimodal baseado em Transformer.
    Ajustado para lidar com PCLs (B, T, L_max, D_point) via PointNet.
    """
    def __init__(self, config=CONFIG, pos_mean=None, pos_std=None):
        super().__init__()
        
        hidden_dim = config['transformer_hidden_dim']
        # feature_dim agora é a saída do PointNet (PointNet_feature_dim)
        pointnet_feature_dim = CONFIG['modal_feature_dim'] 
        point_dim = CONFIG['pcl_feature_dim']
        dropout_rate = CONFIG.get('dropout_rate', 0.2)
        
        self.pos_mean = pos_mean if pos_mean is not None else torch.zeros(3)
        self.pos_std = pos_std if pos_std is not None else torch.ones(3)
        
        # --- Backbones e Encoders de Features ---
        
        # Visual (CNN)
        self.visual_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.visual_backbone.fc = nn.Identity() 
        self.visual_dim = 512 
        
        # PCL (PointNetEncoder)
        self.pcl_encoder = PointNetEncoder(in_dim=point_dim, out_dim=pointnet_feature_dim)
        
        # --- Projeções para Hidden Dim ---
        
        self.proj_visual = nn.Linear(self.visual_dim, hidden_dim)
        # O PointNet já gera features, então proj_modal mapeia PointNet_feature_dim -> hidden_dim
        self.proj_modal = nn.Linear(pointnet_feature_dim, hidden_dim) 
        
        # Fusion token
        self.fusion_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=1024, 
            dropout=dropout_rate, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.post_fusion_norm = nn.LayerNorm(hidden_dim)
        self.drop_path = DropPath(drop_prob=dropout_rate)
        
        # Neural-Mealy Core
        self.mealy_core = NeuralMealyLayer(
            hidden_dim, 
            NUM_SYMBOLIC_STATES, 
            dropout_rate,
            temp_scale=config.get('mealy_temp_scale', 0.5)
        )
        
        # Head de posição com inicialização adaptada (mantido inalterado)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 3)
        )
        self._init_pos_head()

    def _init_pos_head(self):
        for m in self.pos_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _process_pcl(self, pcl_data: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """ 
        Processa dados de PCL (B, T, L_max, D_point) -> (B, T, D_feature).
        """
        if pcl_data.ndim == 4:
            # 1. Achata (B, T, L_max, D) -> (B*T, L_max, D) para o PointNet
            pcl_flat = pcl_data.view(-1, pcl_data.size(2), pcl_data.size(3))
        else:
            # Caso especial T=1, já está (B, L_max, D)
            pcl_flat = pcl_data
        
        # 2. Extrai features (B*T, D_feature)
        F_pcl_flat = self.pcl_encoder(pcl_flat)
        
        # 3. Reconstroi para (B, T, D_feature)
        F_pcl = F_pcl_flat.view(B, T, -1)
        
        return F_pcl

    def forward(self, image, lidar_360, livox_avia, radar, prev_mealy_state=None):
        # image é (B, T, C, H, W)
        # PCLs são (B, T, L_max, D_point)
        B, T = image.size(0), image.size(1)
        
        # 1. Processamento de Imagem: Achata B e T para CNN, depois reestrutura
        # (B, T, C, H, W) -> (B*T, C, H, W)
        image_flat = image.view(-1, image.size(2), image.size(3), image.size(4)) 
        
        F_V_raw_flat = self.visual_backbone(image_flat) # (B*T, visual_dim)
        
        # Projeta e reestrutura para (B, T, hidden_dim)
        F_V_flat = self.proj_visual(F_V_raw_flat) # (B*T, hidden_dim)
        F_V = F_V_flat.view(B, T, -1) # (B, T, hidden_dim)
        
        # 2. Processamento de PCLs: (B, T, L_max, D_point) -> (B, T, D_feature) -> (B, T, hidden_dim)
        # Note que a função _process_pcl é capaz de lidar com L_max variável (padding)
        F_L360_raw = self._process_pcl(lidar_360, B, T)
        F_LA_raw = self._process_pcl(livox_avia, B, T)
        F_R_raw = self._process_pcl(radar, B, T)
        
        # Projeção final para hidden_dim
        F_L360 = self.proj_modal(F_L360_raw)
        F_LA = self.proj_modal(F_LA_raw)
        F_R = self.proj_modal(F_R_raw)
        
        # 3. Criação da Sequência Multimodal
        # Sequence é (B, T, 4 * hidden_dim) - NÃO, precisamos que o Transformer processe a sequência de features.
        
        # Concatenamos as features modais: (B, T, N_modals, hidden_dim)
        modal_features = torch.stack((F_V, F_L360, F_LA, F_R), dim=2) # (B, T, 4, hidden_dim)

        # Achata (B, T, N_modals, hidden_dim) para (B*T, N_modals, hidden_dim)
        # O Transformer fará a fusão DENTRO de cada timestamp (T)
        input_sequence_transformer = modal_features.view(B*T, 4, -1)
        
        # 4. Inserção do Fusion Token e Processamento Transformer
        
        # Fusion token: (1, 1, hidden_dim) -> (B*T, 1, hidden_dim)
        fusion_token_batch = self.fusion_token.expand(B*T, -1, -1)
        input_sequence_with_token = torch.cat((input_sequence_transformer, fusion_token_batch), dim=1)
        
        F_Encoded = self.transformer_encoder(input_sequence_with_token)
        # Pega o token de CLS/Fusion (último)
        F_Final_raw = F_Encoded[:, -1, :] # (B*T, hidden_dim)
        F_Final_stabilized = self.post_fusion_norm(F_Final_raw + self.drop_path(F_Final_raw))
        
        # 5. Processamento Temporal (Neural-Mealy)
        # Reestrutura (B*T, hidden_dim) para (T, B, hidden_dim) para iterar no GRU
        F_Mealy_Input = F_Final_stabilized.view(B, T, -1).transpose(0, 1) # (T, B, hidden_dim)

        next_mealy_state = prev_mealy_state
        all_class_logits = []
        all_state_logits = []
        
        # Loop sequencial sobre T
        for t in range(T):
            current_embedding = F_Mealy_Input[t] # (B, hidden_dim)
            next_mealy_state, pred_class_logits, state_transition_logits = self.mealy_core(
                current_embedding, next_mealy_state
            )
            all_class_logits.append(pred_class_logits.unsqueeze(0))
            all_state_logits.append(state_transition_logits.unsqueeze(0))

        # Reestrutura a saída para (B, T, D)
        pred_class_logits_seq = torch.cat(all_class_logits, dim=0).transpose(0, 1) # (B, T, 1)
        state_transition_logits_seq = torch.cat(all_state_logits, dim=0).transpose(0, 1) # (B, T, NUM_SYMBOLIC_STATES)
        
        # Predição de posição usa o estado final da sequência (next_mealy_state)
        # next_mealy_state agora é (B, hidden_dim) (o último estado)
        pred_pos_norm = self.pos_head(next_mealy_state) # (B, 3)
        pred_pos = pred_pos_norm * self.pos_std.to(pred_pos_norm.device) + self.pos_mean.to(pred_pos_norm.device)
        
        return {
            # Posicao e Estado são a saída do ÚLTIMO passo de tempo (T-1)
            'pred_pos': pred_pos, 
            # Logits de classe e transição são para TODOS os T passos
            'pred_class_seq': pred_class_logits_seq, 
            'mealy_state_logits_seq': state_transition_logits_seq,
            # Estado Mealy final para ser alimentado no próximo batch (se for o caso)
            'final_mealy_state': next_mealy_state 
        }
    
    '''
    O que foi alterado e por quê?
Novo Encoder de PCL (PointNetEncoder):

Propósito: É o feature extractor que transforma a entrada padronizada de PCL (L_max, D_point) em um vetor de feature de tamanho fixo (D_feature). Isso é essencial, pois o Transformer só pode trabalhar com vetores de tamanho fixo.

Integração: Adicionei self.pcl_encoder na inicialização do MultiModalTransformer.

_process_pcl (Função Auxiliar):

Propósito: Lida com a nova dimensão (B, T, L_max, D_point). Ele achata B e T para (B*T, L_max, D_point), passa pelo pcl_encoder e reestrutura para (B, T, D_feature).

Processamento no forward:

Entrada: O forward agora espera que todas as entradas tenham as dimensões de Batch e Sequência Temporal (B, T) na frente.

Imagens: As imagens (B, T, C, H, W) são achatadas para (B*T, C, H, W) antes de passar pelo visual_backbone (ResNet), e reestruturadas para (B, T, hidden_dim) depois.

PCLs: As PCLs (B, T, L_max, D_point) são processadas pela nova função _process_pcl e projetadas, resultando em (B, T, hidden_dim).

Fusão Intratemporal (Transformer):

A fusão é feita dentro de cada passo de tempo. O vetor de features multimodais (B, T, N_modals, hidden_dim) é achatado em (B*T, N_modals, hidden_dim). O Transformer faz a fusão e extrai o token final (B*T, hidden_dim).

Processamento Temporal (Neural-Mealy):

O resultado da fusão (B*T, hidden_dim) é reestruturado em (T, B, hidden_dim).

O loop sequencial for t in range(T) passa a entrada pelo self.mealy_core (que usa nn.GRUCell), mantendo o estado next_mealy_state de forma correta ao longo do tempo.

Saída da Posição:

A predição de posição (pred_pos) e o estado final (final_mealy_state) são gerados apenas a partir do último estado (t=T-1) do GRU, pois é o estado que contém toda a informação da sequência.

As saídas de classe e estado Mealy agora são retornadas como sequências completas (_seq) de tamanho (B, T, D).
    '''