import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
# Assumimos que o CONFIG de mm_config.py será importado ou passado
# from .mm_config import CONFIG 

# Definição das classes de estado simbólico para auditabilidade
NUM_SYMBOLIC_STATES = 3  # Ex: 0: Idle, 1: Candidate, 2: Confirmed

class NeuralMealyLayer(nn.Module):
    """
    Simula o core Neural Finite State Transducer (NFST) / Neural-Mealy.
    Toma o embedding fundido do Transformer e o estado anterior para:
    1. Calcular o próximo estado (simula a função de transição delta).
    2. Emitir uma predição simbólica de estado (simula a função de saída lambda).
    """
    def __init__(self, hidden_dim, num_states):
        super().__init__()
        
        # GRUCell: Implementa a dinâmica do FST, capturando a dependência temporal
        # Input: embedding fundido (Hidden_Dim), Hidden: estado Mealy anterior (Hidden_Dim)
        self.state_updater = nn.GRUCell(hidden_dim, hidden_dim) 
        
        # Head de Predição de Estado: Classifica o estado simbólico (e.g., Idle, Confirmed)
        self.state_head = nn.Linear(hidden_dim, num_states)
        
    def forward(self, fused_embedding, prev_state=None):
        """
        Args:
            fused_embedding (Tensor): Feature fundida pelo Transformer [B, H].
            prev_state (Tensor, optional): Estado interno do Mealy do passo anterior [B, H]. 
                                           É None no primeiro passo ou em testes de frames únicos.
        Returns:
            next_state (Tensor): O novo estado interno Mealy [B, H], usado para predições.
            state_pred (Tensor): Logits para a classificação simbólica do estado [B, num_states].
        """
        B = fused_embedding.size(0)
        
        # Se nenhum estado anterior for fornecido (início da sequência/batch), inicializa com zeros.
        if prev_state is None:
            prev_state = torch.zeros(B, fused_embedding.size(-1), 
                                     device=fused_embedding.device)
            
        # Calcula o próximo estado
        next_state = self.state_updater(fused_embedding, prev_state)
        
        # Gera a predição simbólica de estado a partir do novo estado
        state_pred = self.state_head(next_state)
        
        return next_state, state_pred

class MultiModalTransformer(nn.Module):
    """
    Modelo Multimodal baseado em Transformer com Core Neural-Mealy para fusão estruturada.
    """
    def __init__(self, config):
        super().__init__()
        
        hidden_dim = config['transformer_hidden_dim']
        feature_dim = config['modal_feature_dim']
        num_classes = config['num_classes']
        
        # 1. Backbones de Feature Extraction
        
        # Visual: ResNet18
        self.visual_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.visual_backbone.fc = nn.Identity() 
        self.visual_dim = 512 # Saída do ResNet18 sem FC
        
        # Projeções para harmonizar todas as features na dimensão do Transformer
        self.proj_visual = nn.Linear(self.visual_dim, hidden_dim)
        self.proj_modal = nn.Linear(feature_dim, hidden_dim)
        
        # 2. Transformer Encoder (Cross-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=512, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. NOVO: Neural-Mealy Core (NFST)
        self.mealy_core = NeuralMealyLayer(hidden_dim, NUM_SYMBOLIC_STATES)
        
        # 4. Heads de Predição (Agora consomem a saída estruturada do Mealy Core)
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # Posição [X, Y, Z] (Regressão)
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes) # Classificação
        )

    def forward(self, image, lidar, radar, audio, prev_mealy_state=None):
        
        # 1. Extração e Projeção de Features
        F_V_raw = self.visual_backbone(image)
        
        F_V = self.proj_visual(F_V_raw).unsqueeze(1)    # [B, 1, Hidden_Dim]
        F_L = self.proj_modal(lidar).unsqueeze(1)      # [B, 1, Hidden_Dim]
        F_R = self.proj_modal(radar).unsqueeze(1)      # [B, 1, Hidden_Dim]
        F_A = self.proj_modal(audio).unsqueeze(1)      # [B, 1, Hidden_Dim]
        
        # 2. Fusão: Criação da Sequência de Tokens [B, 4, Hidden_Dim]
        sequence = torch.cat((F_V, F_L, F_R, F_A), dim=1) 
        
        # Processamento pelo Transformer (Cross-Attention)
        F_Encoded = self.transformer_encoder(sequence) 
        
        # F_Final: Usamos o token da Imagem (primeiro token) como o representante fundido
        # [B, Hidden_Dim]
        F_Final = F_Encoded[:, 0, :] 
        
        # 3. Transdução de Estado (Neural-Mealy Core)
        # O Mealy Core toma o feature fundido e o estado anterior
        next_mealy_state, pred_state_logits = self.mealy_core(F_Final, prev_mealy_state)
        
        # 4. Predição: Posição e Classe são feitas com o NOVO VETOR DE ESTADO
        # Isso garante que a predição está estruturalmente ligada ao estado temporal
        pred_pos = self.pos_head(next_mealy_state)
        pred_class = self.class_head(next_mealy_state)
        
        # Retornamos as 3 saídas (Posição, Classe e Logits do Estado Simbólico)
        # E o next_mealy_state, que deve ser passado para o próximo passo de tempo (no loop de treinamento/inferência)
        return {
            'pred_pos': pred_pos, 
            'pred_class': pred_class, 
            'pred_state_logits': pred_state_logits,
            'next_mealy_state': next_mealy_state
        }
