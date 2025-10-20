import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

# ------------------------------
# VISUAL TRANSFORMER MULTIMODAL
# ------------------------------

class PatchEmbedding(nn.Module):
    """
    Patch Embedding avançado para Vision Transformer multimodal (visível + infravermelho).

    Funcionalidades:
    - Suporta imagens RGB (3 canais) e IR (1 canal)
    - Positional embeddings learnable
    - LayerNorm opcional após projeção
    - Preparação para concatenar tokens de múltiplas modalidades

    USO:
    # Para visível
    # Para imagens 1920x1080
    patch_embed_vis = PatchEmbedding(d_model=512, img_size=(1080, 1920), patch_size=16, n_channels=3)
    # Para infravermelho
    patch_embed_ir  = PatchEmbedding(d_model=512, img_size=(1080, 1920), patch_size=16, n_channels=1)
    
    tokens_vis = patch_embed_vis(visible_frames)   # (B, P, d_model)
    tokens_ir  = patch_embed_ir(infrared_frames)   # (B, P, d_model)
    
    # Concatenar os patches ao longo da dimensão de tokens
    tokens = torch.cat([tokens_vis, tokens_ir], dim=1)  # (B, 2*P, d_model)
    """

    def __init__(self, d_model, img_size, patch_size, n_channels, use_norm=True):
        super().__init__()
        self.d_model = d_model
        self.H, self.W = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.use_norm = use_norm

        self.H_patches = self.H // self.patch_size
        self.W_patches = self.W // self.patch_size
        self.num_patches = self.H_patches * self.W_patches

        # Conv2d para projeção de patches
        self.linear_project = nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        self.norm = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        assert C == self.n_channels, f"Esperava {self.n_channels} canais, recebeu {C}"

        # Mesclar batch e tempo: (B*T, C, H, W)
        x = x.view(B*T, C, H, W)
        x = self.linear_project(x)  # (B*T, d_model, H_patch, W_patch)

        x = x.flatten(2).transpose(1, 2)  # (B*T, num_patches, d_model)
        # Adicionar positional embedding (broadcast para T frames)
        x = x + self.pos_embed.unsqueeze(0)  # (1, num_patches, d_model) -> broadcast

        x = self.norm(x)
        # Voltar para (B, T*num_patches, d_model)
        x = x.view(B, T*self.num_patches, self.d_model)
        return x

# --------------------------------------------------
# METADATA EMBEDDING
# --------------------------------------------------
class MetadataEmbedding(nn.Module):
    def __init__(self, d_model, max_frames=100):
        super().__init__()
        self.gt_rect_embed = nn.Linear(4, d_model)
        self.exist_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, gt_rect, exist):
        B, T, _ = gt_rect.shape
        gt_tokens = self.gt_rect_embed(gt_rect)       # (B, T, d_model)
        exist_tokens = self.exist_embed(exist.long()) # (B, T, d_model)
        tokens = gt_tokens + exist_tokens + self.pos_embed[:, :T, :]
        return tokens

# --------------------------------------------------
# MULTIMODAL EMBEDDING
# --------------------------------------------------
class MultimodalEmbedding(nn.Module):
    '''
    Fusão multimodal no nível dos metadados, ou fusão por atenção cruzada (cross-attention).
    Etapa 1: Tokenização e Concatenação aqui no MultimodalEmbedding.
    Etapa 2: Interação e Fusão posterior no TransformerEncoder.
    '''
    def __init__(self, d_model, size_vis, img_size_ir, patch_size, use_norm=True):
        super().__init__()
        self.patch_vis = PatchEmbedding(d_model, size_vis, patch_size, n_channels=3, use_norm=use_norm)
        self.patch_ir  = PatchEmbedding(d_model, img_size_ir, patch_size, n_channels=1, use_norm=use_norm)
        self.metadata_embed = MetadataEmbedding(d_model)

    def forward(self, visible_frames, infrared_frames, gt_rect_vis, gt_rect_ir, exist):
        # primeiro cria dois conjuntos independentes de tokens de imagem.
        tokens_vis = self.patch_vis(visible_frames)   # (B, P_vis, d_model)
        tokens_ir  = self.patch_ir(infrared_frames)   # (B, P_ir, d_model)

        # Tokens de metadados para cada modalidade.
        # Cria T tokens (um para cada frame) que representam o bounding box e o status de existência do drone no espectro visível.
        meta_vis = self.metadata_embed(gt_rect_vis, exist)  # (B, T, d_model)
        # Cria outros T tokens que representam o bounding box e o status de existência no espectro infravermelho.
        meta_ir  = self.metadata_embed(gt_rect_ir, exist)   # (B, T, d_model)

        # Em seguida, une tudo em uma única sequência longa.
        # Neste ponto, os tokens meta_vis e meta_ir estão no mesmo "espaço" (o mesmo tensor), mas eles ainda não trocaram informações. 
        # Eles são como duas equipes na mesma sala, mas que ainda não conversaram.
        # A p´roxima etapa da fusão vai ocorrer no TransformerEncoder.
        return torch.cat([tokens_vis, tokens_ir, meta_vis, meta_ir], dim=1)


class AttentionHead(nn.Module):
    """
    Cabeça única de atenção escalada compatível com embeddings multimodais.
    AttentionHead (O CORAÇÃO DA FUSÃO).
    É aqui que a "conversa" realmente acontece. O forward recebe x (nosso X_camada_0).
    """
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, d_head)
        self.W_k = nn.Linear(d_model, d_head)
        self.W_v = nn.Linear(d_model, d_head)

    def forward(self, x, mask=None):
        """
        x é (B, N, d_model)
        x contém [patches_vis | patches_ir | meta_vis | meta_ir]
    
        1. GERAR Q, K, V
        Estas projeções são aplicadas a CADA TOKEN em 'x'.

        x: (B, N, d_model) → tokens de todas as modalidades.
        mask: (B, N) ou (B, 1, N, N) → 0 para ignorar tokens.
        """
        # Q: Matriz de "Perguntas" (Queries). Q[:, i, :] é a "pergunta" que o Token i fará.
        Q = self.W_q(x)
        # K: Matriz de "Chaves" (Keys). K[:, j, :] é a "etiqueta" do Token j.
        K = self.W_k(x)
        # V: Matriz de "Valores" (Values). V[:, j, :] é a "informação" que o Token j oferece.
        V = self.W_v(x)

        # Scaled Dot-Product Attention.
        # # 2. CALCULAR SCORES (A "CONVERSA")
        # # Q * K.T -> (B, N, d_head) @ (B, d_head, N) = (B, N, N)
        # A matriz scores é a parte crucial. 
        # scores[b, i, j] mede a afinidade (a "relevância da resposta") entre a "pergunta" do Token i e a "etiqueta" do Token j.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, N, N)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # 3. NORMALIZAR SCORES.
        # Agora, cada linha i da matriz attn soma 1. 
        # Ela representa os "pesos" que o Token i deve dar a todos os outros tokens (incluindo ele mesmo).
        attn = F.softmax(scores, dim=-1)
        # 4. CRIAR A SAÍDA (A "FUSÃO") como uma Média Ponderada.
        out = torch.matmul(attn, V)  # (B, N, d_head)
        '''
        É exatamente aqui que a fusão acontece. 
        O token de saída out no índice i (de meta_vis_t5) agora é uma mistura de si mesmo, dos patches e dos outros metadados (meta_ir), 
        com base nos pesos de atenção aprendidos.
        Este out é então retornado para o MultiHeadAttention, concatenado, projetado, retornado para o TransformerEncoderLayer, 
        somado com o original (x + attn_out), e finalmente passado para a próxima camada, onde todo o processo se repete, 
        refinando ainda mais a fusão.
        '''
        return out, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention compatível com embeddings multimodais.
    Este módulo é outro "gerente". Ele gerencia várias "cabeças" de atenção que rodam em paralelo.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.heads = nn.ModuleList([AttentionHead(d_model, self.d_head) for _ in range(num_heads)])
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        x: (B, N, d_model) → saída do embedding multimodal.
        # x aqui é o X_camada_0, no exemplo.
        mask: máscara opcional
        """
        head_outs = []
        attn_maps = []
        #  Loop sobre as 8 cabeças (num_heads=8)
        for head in self.heads:
            # ** IMPORTANTE **
            # Cada 'head' recebe o tensor 'x' COMPLETO.
            out, attn = head(x, mask)
            head_outs.append(out)
            attn_maps.append(attn)

        # Concatenar ao longo da dimensão de heads.
        # Concatena os resultados de todas as cabeças.
        concat = torch.cat(head_outs, dim=-1)  # (B, N, d_model)
        # Projeção linear final
        out = self.out_proj(concat)            # (B, N, d_model)
        # 'out' é o 'attn_out' da camada anterior
        return out, attn_maps

# --------------------------------------------------
# Transformer Encoder Layer
# --------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        '''
        Este módulo é o "bloco" de construção. O forward dele faz duas coisas, mas a fusão acontece na primeira parte:
        '''
        # Multi-head attention com residual + norm.
        # x aqui é o X_camada_0 (na primeira iteração).
        # --- AQUI ACONTECE A FUSÃO ---
        attn_out, _ = self.mha(x, mask) # Chama o MultiHeadAttention
        # attn_out é o tensor (B, N, d_model) com as informações misturadas.
        # A linha abaixo aplica a "conexão residual":
        x = x + self.dropout1(attn_out)
        # x agora é X_camada_0_fundido (pronto para o Feedforward network - FFN).
        x = self.norm1(x)

        # Feedforward com residual + norm.
        # --- AQUI NÃO HÁ FUSÃO ---
        # A FFN processa cada token individualmente
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        # x agora é X_camada_1 (pronto para a próxima camada do Encoder).
        x = self.norm2(x)
        return x

# --------------------------------------------------
# Transformer Encoder completo
# --------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, mask=None):
        '''
        # x aqui é o X_camada_0
    
        # 1ª Iteração do Loop (Camada 0)
        x = self.layers[0](x, mask)  # Chama o TransformerEncoderLayer
        # x agora é X_camada_1 (já fundido 1x)
        
        # 2ª Iteração do Loop (Camada 1)
        x = self.layers[1](x, mask) # Chama o TransformerEncoderLayer com X_camada_1
        # x agora é X_camada_2 (fundido 2x)
        
        # ... repete num_layers vezes ...
        '''
        for layer in self.layers:
            x = layer(x, mask)
        return x  # (B, num_tokens, d_model)

# --------------------------------------------------
# Vision Transformer Multimodal
# --------------------------------------------------
class VisionTransformerMultimodal(nn.Module):
    def __init__(self, img_size_vis, img_size_ir, patch_size,
                 d_model=512, num_heads=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # ---------------------------
        # Embedding multimodal com tamanhos diferentes
        # ---------------------------
        self.embedding = MultimodalEmbedding(
            d_model,
            size_vis=img_size_vis,
            img_size_ir=img_size_ir,
            patch_size=patch_size
        )

        # ---------------------------
        # Transformer Encoder
        # - Integra informações de todas as modalidades
        # ---------------------------
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dim_feedforward, dropout)

        # ---------------------------
        # Heads de saída
        # - box_head: previsão de bounding box (gt_rect)
        # - exist_head: previsão de existência
        # ---------------------------
        self.box_head = nn.Linear(d_model, 4)            # gt_rect
        self.exist_head = nn.Linear(d_model, 1)          # exist

        # ---- inicialização do bias para boxes (opcional mas útil) ----
        # valor inicial: centro (0.5,0.5) e tamanho pequeno (0.2,0.2)
        # evita que o modelo comece com w,h ~ 1.0 (caixa gigante).
        with torch.no_grad():
            if hasattr(self.box_head, 'bias') and self.box_head.bias is not None:
                self.box_head.bias.copy_(torch.tensor([0.5, 0.5, 0.2, 0.2], dtype=self.box_head.bias.dtype))

    def forward(self, visible_frames, infrared_frames, gt_rect_vis, gt_rect_ir, exist):
        # ---------------------------
        # 1. Extrair tokens multimodais
        # ---------------------------
        tokens = self.embedding(visible_frames, infrared_frames, gt_rect_vis, gt_rect_ir, exist)
        
        # ---------------------------
        # 2. Processar tokens no Transformer Encoder
        # ---------------------------
        encoded_tokens = self.encoder(tokens)  # (B, num_total_tokens, d_model)
        
        # ---------------------------
        # 3. Fusão explícita de meta tokens por frame
        # ---------------------------
        B = encoded_tokens.size(0)
        T = gt_rect_vis.size(1)  # número de frames
        d_model = encoded_tokens.size(2)

        # Os meta_tokens visíveis estão concatenados antes dos meta_tokens IR
        frame_tokens_vis = encoded_tokens[:, -2*T:-T, :]  # (B, T, d_model)
        frame_tokens_ir  = encoded_tokens[:, -T:, :]      # (B, T, d_model)

        # --- NÃO usar sigmoid nas caixas: deixa os gradientes fluírem melhor ---
        pred_boxes_vis = self.box_head(frame_tokens_vis)  # (B, T, 4)
        pred_boxes_ir  = self.box_head(frame_tokens_ir)   # (B, T, 4)

        # Mantemos sigmoid só para a saída de existência
        pred_exist     = torch.sigmoid(self.exist_head(frame_tokens_vis)).squeeze(-1)  # (B, T)

        # Clamp para forçar as predições de caixa em [0,1] durante o treino
        pred_boxes_vis = torch.sigmoid(self.box_head(frame_tokens_vis))
        pred_boxes_ir  = torch.sigmoid(self.box_head(frame_tokens_ir))

        return {
            "pred_boxes_vis": pred_boxes_vis,
            "pred_boxes_ir": pred_boxes_ir,
            "pred_exist": pred_exist
        }





