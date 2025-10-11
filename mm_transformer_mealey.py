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
        
        # Máscara (opcional, mas recomendado)
        if padding_mask is not None:
             # Expande a máscara para a dimensão da feature (D_feature)
             expanded_mask = padding_mask.unsqueeze(1).expand(-1, x.size(1), -1)
             # Substitui os valores de padding por um valor muito pequeno (-inf)
             x[expanded_mask] = -float('inf')
        
        # Max Pooling (redução da dimensão L_max)
        x = torch.max(x, 2)[0] # Resultado: (B*T, D_feature)
        
        return x # (B*T, D_feature)

# =========================================================================
# UTILITY: STOCHASTIC DEPTH (DropPath)
# -------------------------------------------------------------------------

class DropPath(nn.Module):
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
# -------------------------------------------------------------------------
class NeuralMealyLayer(nn.Module):
    """
    Simula o core Neural-Mealy com regularização aprimorada para coerência temporal.
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
# MULTIMODAL TRANSFORMER (CORRIGIDO PARA PREDICAO TEMPORAL)
# -------------------------------------------------------------------------

class MultiModalTransformer(nn.Module):
    """
    Modelo Multimodal baseado em Transformer.
    Ajustado para lidar com PCLs (B, T, L_max, D_point) via PointNet.
    Corrigido para prever a posição em tempo real, em cada passo Mealy.
    """
    def __init__(self, config=CONFIG, pos_mean=None, pos_std=None):
        super().__init__()
        
        hidden_dim = config['transformer_hidden_dim']
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
        
        # Head de posição (Predicts 3D position from the Mealy state)
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
        """Inicializa o Pos Head com pesos menores para evitar explosão inicial."""
        # Novo valor de ganho, reduzido para 0.01 (ou até 0.001 se necessário)
        XAVIER_GAIN = 0.01 
        
        # Itera sobre os módulos para aplicar a inicialização
        for m in self.pos_head:
            if isinstance(m, nn.Linear):
                # Inicializa os pesos com ganho reduzido
                nn.init.xavier_uniform_(m.weight, gain=XAVIER_GAIN)
                # Garante que o bias seja zero
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
        F_L360_raw = self._process_pcl(lidar_360, B, T)
        F_LA_raw = self._process_pcl(livox_avia, B, T)
        F_R_raw = self._process_pcl(radar, B, T)
        
        # Projeção final para hidden_dim
        F_L360 = self.proj_modal(F_L360_raw)
        F_LA = self.proj_modal(F_LA_raw)
        F_R = self.proj_modal(F_R_raw)
        
        # 3. Criação da Sequência Multimodal
        
        # Concatenamos as features modais: (B, T, N_modals, hidden_dim)
        modal_features = torch.stack((F_V, F_L360, F_LA, F_R), dim=2) # (B, T, 4, hidden_dim)

        # Achata (B, T, N_modals, hidden_dim) para (B*T, N_modals, hidden_dim)
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
        all_pos_predictions = []      # Denormalizada (em metros)
        all_pos_norm_predictions = [] # NOVO: Normalizada (para cálculo de Loss)

        # Loop sequencial sobre T
        for t in range(T):
            current_embedding = F_Mealy_Input[t] # (B, hidden_dim)
            next_mealy_state, pred_class_logits, state_transition_logits = self.mealy_core(
                current_embedding, next_mealy_state
            )
            
            # Predição de posição no domínio normalizado
            pred_pos_norm_t = self.pos_head(next_mealy_state) # (B, 3) <--- NORMALIZED
            all_pos_norm_predictions.append(pred_pos_norm_t.unsqueeze(0)) 

            # Denormalização para métricas (exibir o erro em metros)
            pred_pos_t = pred_pos_norm_t * self.pos_std.to(pred_pos_norm_t.device) + self.pos_mean.to(pred_pos_norm_t.device)
            all_pos_predictions.append(pred_pos_t.unsqueeze(0)) 

            all_class_logits.append(pred_class_logits.unsqueeze(0))
            all_state_logits.append(state_transition_logits.unsqueeze(0))

        # Reestrutura as saídas sequenciais para (B, T, D)
        pred_class_logits_seq = torch.cat(all_class_logits, dim=0).transpose(0, 1) # (B, T, 1)
        state_transition_logits_seq = torch.cat(all_state_logits, dim=0).transpose(0, 1) # (B, T, NUM_SYMBOLIC_STATES)
        pred_pos_seq = torch.cat(all_pos_predictions, dim=0).transpose(0, 1) # (B, T, 3)
        pred_pos_norm_seq = torch.cat(all_pos_norm_predictions, dim=0).transpose(0, 1) # NOVO: (B, T, 3)
        
        # Para compatibilidade com o script de treinamento externo (que usa T=1)
        if T == 1:
            final_pred_pos = pred_pos_seq.squeeze(1) # (B, 3)
            final_pred_pos_norm = pred_pos_norm_seq.squeeze(1) # NOVO: (B, 3)
        else:
            final_pred_pos = pred_pos_seq
            final_pred_pos_norm = pred_pos_norm_seq # NOVO: (B, T, 3)

        return {
            # Posicao em tempo real (Se T=1, é (B, 3); se T>1, é (B, T, 3))
            'pred_pos': final_pred_pos,                   # Denormalizado
            'pred_pos_norm': final_pred_pos_norm,         # NOVO: Normalizado (Usado para Loss no script de treino)
            # Logits de classe e transição são para TODOS os T passos
            'pred_class_seq': pred_class_logits_seq, 
            'mealy_state_logits_seq': state_transition_logits_seq,
            # Estado Mealy final para ser alimentado no próximo batch (se for o caso)
            'final_mealy_state': next_mealy_state 
        }