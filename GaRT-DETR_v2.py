import math
import timm 
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF # Necessário para o blur

# ============================================================
# UTILS E MÓDULOS DE SUPORTE
# ============================================================

def preprocess_batch(vis_frames_list, ir_frames_list, target_size=(224, 224)):
    # B = número de sequências no batch (batch size temporal)
    # Cada elemento de vis_frames_list corresponde a uma sequência completa
    B = len(vis_frames_list)
    # T = número de frames por sequência
    # Assume-se que todas as sequências têm o mesmo comprimento temporal
    T = len(vis_frames_list[0])
    # Listas que irão armazenar TODOS os frames processados (flatten em B*T)
    processed_vis, processed_ir = [], []
    # Listas para guardar os tamanhos originais (W, H) de cada sequência
    # Usado depois para reescalar bounding boxes para o espaço original
    orig_sizes_vis, orig_sizes_ir = [], []
    # Loop sobre cada sequência do batch
    for b in range(B):
        # Obtém altura e largura do PRIMEIRO frame RGB da sequência b
        # shape esperado: [C, H, W]
        h_v, w_v = vis_frames_list[b][0].shape[-2:]
        # Guarda o tamanho original no formato (W, H), padrão usado em DETR
        orig_sizes_vis.append((w_v, h_v))
        # Obtém altura e largura do PRIMEIRO frame IR da sequência b
        h_i, w_i = ir_frames_list[b][0].shape[-2:]
        # Guarda o tamanho original do IR separadamente
        orig_sizes_ir.append((w_i, h_i))
        # Loop temporal: percorre cada frame da sequência
        for t in range(T):
            # v = frame RGB no tempo t da sequência b
            # i = frame IR correspondente
            v, i = vis_frames_list[b][t], ir_frames_list[b][t]
            # Redimensiona o frame RGB para target_size
            # unsqueeze(0): adiciona dimensão de batch para o interpolate
            # bilinear: apropriado para imagens naturais
            # align_corners=False: evita distorções geométricas
            processed_vis.append(F.interpolate(v.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False))
            # Redimensiona o frame IR da mesma forma
            # Mantém alinhamento espacial consistente entre RGB e IR
            processed_ir.append(F.interpolate(i.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False))
    # Concatena todos os frames RGB ao longo da dimensão batch
    # Resultado: tensor de shape [B*T, C, H, W]
    # Esse flatten temporal permite que o backbone processe todos os frames
    # como se fossem imagens independentes, mantendo eficiência
    return torch.cat(processed_vis), torch.cat(processed_ir), orig_sizes_vis, orig_sizes_ir

class SpatialGatedFusionBlock(nn.Module):
    def __init__(self, d_model, nhead, temperature=2.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.temperature = temperature
        self.spatial_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
        # Técnica de Kendall: Parâmetro treinado automaticamente pelo otimizador
        # O bias controlado
        self.learnable_bias = nn.Parameter(torch.tensor([10.0]))
        # Zera o bias aleatório do PyTorch. evita o valor acima + um valor aleatoriodo pytorch
        nn.init.constant_(self.spatial_gate[-1].bias, 0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, f_main, f_aux):
        spatial_logits = self.spatial_gate(f_aux) 
        # Aplicamos o bias aprendido antes do sigmoid
        conf = torch.sigmoid((spatial_logits + self.learnable_bias) / self.temperature) 
        
        f_fused, _ = self.cross_attn(f_main, f_aux, f_aux)

        s_mean = conf.mean(dim=1) 
        s_std  = conf.std(dim=1) 
        
        return self.norm(f_main + conf * f_fused), (s_mean, s_std)

# ============================================================
# 1. BACKBONE RGBT
# ============================================================

class RGBTBackbone(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        # BRAÇO RGB — ResNet18 pré-treinada em ImageNet
        # Carrega a ResNet18 padrão com pesos ImageNet
        # Fornece uma extração robusta de textura, bordas e padrões visuais
        rgb_net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Camadas iniciais da ResNet (conv1 + bn + relu + maxpool + layer1)
        # Saída em escala 1/4 com 64 canais
        # Usada como feature de alta resolução espacial
        self.rgb_low_level = nn.Sequential(*list(rgb_net.children())[:5]) 
        # Camadas intermediárias (layer2 + layer3)
        # Saída em escala 1/16 com 256 canais
        # Fornece semântica mais abstrata, adequada para atenção global
        self.rgb_deep = nn.Sequential(*list(rgb_net.children())[5:7]) 
        # BRAÇO IR — EfficientNet-B0 (entrada monocanal)
        # EfficientNet-B0 via timm:
        # - in_chans=1: compatível com imagens térmicas
        # - features_only=True: retorna mapas intermediários
        # - out_indices=(1, 3): seleciona dois níveis de escala
        self.ir_net = timm.create_model(
            'efficientnet_b0', 
            pretrained=True, 
            in_chans=3, 
            features_only=True,
            out_indices=(1, 3) 
        )
        # Agora adaptamos cirurgicamente a primeira camada para 1 canal
        old_conv = self.ir_net.conv_stem
        new_conv = nn.Conv2d(1, old_conv.out_channels, 
                             kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, 
                             padding=old_conv.padding, bias=False)
        
        with torch.no_grad():
            # média dos pesos dos 3 canais RGB para o canal único IR
            new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True)
        self.ir_net.conv_stem = new_conv
        # PROJEÇÕES DE CANAL PARA d_model
        # Projeta a feature RGB profunda:
        # ResNet18 layer3 -> 256 canais
        # Conv 1x1 ajusta o espaço para d_model (dimensão comum do Transformer)
        self.proj_rgb = nn.Conv2d(256, d_model, 1)
        # Projeta a feature IR profunda:
        # EfficientNet index 3 -> 112 canais (escala 1/16)
        # Ajustado explicitamente para evitar mismatch silencioso
        self.proj_ir = nn.Conv2d(112, d_model, 1)
        # Projeção de alta resolução:
        # RGB low-level: 64 canais (escala 1/4)
        # IR low-level: 24 canais (index 1 do EfficientNet)
        # Concatenação total: 88 canais → projetados para d_model
        # Essa feature NÃO entra no Transformer principal,
        # sendo usada depois para refinamento local (zoom espacial)
        self.proj_high_res = nn.Conv2d(64 + 24, d_model, 1) 
        # BLOCOS DE FUSÃO CRUZADA COM GATING
        # RGB recebe informação do IR ponderada por um gate aprendido
        self.rgb_enhanced_by_ir = SpatialGatedFusionBlock(d_model, nhead)
        # IR recebe informação do RGB ponderada por outro gate independente
        self.ir_enhanced_by_rgb = SpatialGatedFusionBlock(d_model, nhead)
        # Bottleneck final:
        # Concatena RGB e IR já fundidos (2 * d_model)
        # Reduz novamente para d_model
        self.bottleneck = nn.Linear(d_model * 2, d_model)

    def forward(self, x_rgb, x_ir):
        # EXTRAÇÃO RGB
        # Extração de features RGB em alta resolução (1/4)
        f_rgb_low = self.rgb_low_level(x_rgb)     
        # Extração de features RGB profundas (1/16) 
        f_rgb_deep = self.rgb_deep(f_rgb_low)      
        # EXTRAÇÃO IR
        #  EfficientNet retorna uma lista de features nos índices escolhidos
        ir_features = self.ir_net(x_ir)
        # Feature IR de alta resolução (1/4, 24 canais)
        f_ir_low = ir_features[0] 
        # Feature IR profunda (1/16, 112 canais)
        f_ir_deep = ir_features[1] 
        # FEATURE DE ALTA RESOLUÇÃO (RGB + IR)
        # Concatena RGB e IR low-level preservando alinhamento espacial
        # Projeta para d_model
        # Essa feature é mantida em formato 2D (H, W)
        f_high_res = self.proj_high_res(torch.cat([f_rgb_low, f_ir_low], dim=1))
        # PROJEÇÃO PARA ESPAÇO DO TRANSFORMER
        # RGB profundo:
        # - projeta canais
        # - flatten espacial (H*W)
        # - permuta para formato [B, Tokens, d_model]
        f_rgb = self.proj_rgb(f_rgb_deep).flatten(2).permute(0, 2, 1)
        # IR profundo:
        # mesmo pipeline do RGB para alinhamento semântico
        f_ir  = self.proj_ir(f_ir_deep).flatten(2).permute(0, 2, 1)
        # FUSÃO CRUZADA COM GATING
        # RGB recebe contexto do IR, ponderado por confiança aprendida
        f_rgb_fused, (conf_ir_m, conf_ir_s)   = self.rgb_enhanced_by_ir(f_rgb, f_ir)
        # IR recebe contexto do RGB, ponderado por outra confiança
        # Agora capturamos a tupla (média, std) vinda do SpatialGatedFusionBlock
        f_ir_fused, (conf_rgb_m, conf_rgb_s) = self.ir_enhanced_by_rgb(f_ir, f_rgb)
        # AGREGAÇÃO FINAL
        # Concatena as duas representações já fundidas
        # Bottleneck reduz para d_model
        fused = self.bottleneck(torch.cat([f_rgb_fused, f_ir_fused], dim=-1))
        # Retorna:
        # - fused: memória multimodal pronta para o Transformer
        # - confs: scores médios de gate (úteis para diagnóstico)
        # - f_high_res: feature espacial para refinamento posterior
        return fused, (conf_rgb_m, conf_ir_m, conf_rgb_s, conf_ir_s), f_high_res

# ============================================================
# 2. REFINEMENT LAYER (MELHORIA 3: SOFT ROI-ATTENTION)
# ============================================================

class RefinementLayer(nn.Module):
    '''
    Ela implementa uma "Atenção ROI Soft". Ao contrário de um DETR padrão que olha para a imagem inteira em todas as camadas, 
    essa versão usa o attn_bias para forçar cada query a olhar prioritariamente para perto de onde ela acha que o drone está (ref_points).
    Conforme o layer_idx aumenta, o foco fica mais fechado (o drone é um objeto pequeno), melhorando a precisão do MSA.
    '''
    def __init__(self, d_model, nhead, layer_idx=0):
        super().__init__()
        # Armazena o número de cabeças de atenção para uso no bias
        self.nhead = nhead
        # Índice da camada (usado para ajustar o foco da atenção gaussiana)
        self.layer_idx = layer_idx
        # Camada de Self-Attention para comunicação entre as queries (objetos)
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # Camada de Cross-Attention para buscar informações na memória visual (backbone)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # MLP para processamento das features extraídas após a atenção
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        # Normalizações de camada para estabilidade do treinamento (Pre-norm/Post-norm)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, q, memory, ref_points):
        # Aplica Self-Attention com conexão residual e normalização
        q = self.norm1(q + self.self_attn(q, q, q)[0])
        # Obtém Batch size, número de Queries e tamanho da Memória
        B, N, _ = q.shape
        M = memory.shape[1]
        grid_side = int(math.sqrt(M))
        # Calcula o lado do grid (ex: 14 para memória 196) para mapeamento espacial
        # Grid dinâmico para evitar desalinhamento se M não for quadrado perfeito
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, grid_side, device=q.device), 
            torch.linspace(0, 1, grid_side, device=q.device), indexing='ij'
        )
        # Cria as coordenadas (x, y) do grid de memória normalizadas entre 0 e 1
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2)[:, :M, :]
        # Calcula a distância Euclidiana entre cada query (ref_points) e cada ponto do grid
        dist = torch.cdist(ref_points[:, :, :2], grid) 
        # Define o raio de atenção: fica mais "fino" (focado) conforme as camadas avançam
        sigma = 0.5 / (self.layer_idx + 1)
        # Gera um bias negativo (Gaussian-like) para silenciar regiões distantes do ref_point
        attn_bias = -(dist / sigma).pow(2) 
        # Replica o bias para todas as cabeças de atenção do MultiheadAttention
        attn_bias = attn_bias.repeat_interleave(self.nhead, dim=0)
        # Cross-Attention utilizando o bias espacial para focar na região da predição atual
        q_focussed, _ = self.cross_attn(q, memory, memory, attn_mask=attn_bias)
        # Soma o resultado focado à query original (Residual) e normaliza
        q = self.norm2(q + q_focussed)
        # Refinamento final das features através do MLP e última normalização
        q = self.norm3(q + self.mlp(q))
        # Retorna as queries refinadas para a próxima camada ou predição
        return q

# ============================================================
# 3. SUPERIOR DETR
# ============================================================

class SuperiorDETR(nn.Module):
    def __init__(self, d_model=256, n_queries=30, n_layers=6, img_size=(224, 224)):
        super().__init__()
        # Armazena a resolução de entrada alvo para o pré-processamento
        self.img_size = img_size
        # Dimensão latente interna do Transformer (Canais)
        self.d_model = d_model

        # Estatísticas de normalização para o espelhamento (Buffers não são treináveis)
        self.register_buffer("vis_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("vis_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("ir_mean", torch.tensor([0.449]).view(1, 1, 1, 1))
        self.register_buffer("ir_std", torch.tensor([0.226]).view(1, 1, 1, 1))

        # Instancia o extrator de características multimodal (RGB + Infravermelho)
        self.backbone = RGBTBackbone(d_model)
        # Embedding aprendido para as consultas (propostas de objetos)
        self.query_embed = nn.Embedding(n_queries, d_model)
        # Pontos de referência (x, y, w, h) iniciais para cada query
        self.ref_points = nn.Embedding(n_queries, 4)
        # Inicialização Proporcional (Grid) para cobrir a imagem uniformemente
        with torch.no_grad():
            # Calcula a densidade do grid baseada no número de queries
            side_x = int(math.sqrt(n_queries))
            side_y = n_queries // side_x
            # Gera coordenadas espaciais distribuídas entre 0.1 e 0.9 do frame
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0.1, 0.9, side_y),
                torch.linspace(0.1, 0.9, side_x),
                indexing='ij'
            )
            # Reorganiza o grid para o formato de lista de coordenadas [N, 2]
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            
            # Se sobrar query por conta da divisão, as outras ficam no centro
            self.ref_points.weight.data.fill_(0.5) 
            # Sobrescreve as primeiras queries com as posições do grid gerado
            self.ref_points.weight.data[:grid.size(0), :2] = grid
            # Define o tamanho inicial das caixas como 5% da imagem (foco em objetos pequenos)
            self.ref_points.weight.data[:, 2:] = 0.05
        # Cabeça de predição de Bounding Box para o braço Visível
        self.bbox_head_vis = nn.Linear(d_model, 4)
        # Cabeça de predição de Bounding Box para o braço Infravermelho
        self.bbox_head_ir  = nn.Linear(d_model, 4)
        # Cabeças de classificação: probabilidade de existência no Visível, IR e Global
        self.exist_vis_head = nn.Linear(d_model, 1)
        self.exist_ir_head  = nn.Linear(d_model, 1)
        self.exist_glb_head = nn.Linear(d_model, 1) 
        # Encoder do Transformer para refinar a memória multimodal globalmente
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, 1024, batch_first=True), num_layers=2)
        # Lista de camadas de refinamento iterativo com Atenção ROI-Soft
        self.layers = nn.ModuleList([RefinementLayer(d_model, 8, layer_idx=i) for i in range(n_layers)])
        # Compressor para integrar as features de alta resolução extraídas via Grid Sample
        self.local_compressor = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
    
    def _mirror_ir_to_vis(self, x_ir):
        """ Gera Pseudo-RGB a partir do Infravermelho via Gradientes """
        grad_x = torch.abs(x_ir[:, :, :, 1:] - x_ir[:, :, :, :-1])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.abs(x_ir[:, :, 1:, :] - x_ir[:, :, :-1, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        return torch.cat([x_ir, grad_x, grad_y], dim=1)

    def _mirror_vis_to_ir(self, x_rgb):
        """ Gera Pseudo-IR a partir do Visível via Saliência e Blur """
        x_raw = x_rgb * self.vis_std + self.vis_mean
        thermal_sim, _ = torch.max(x_raw, dim=1, keepdim=True)
        thermal_sim = TF.gaussian_blur(thermal_sim, [5, 5], sigma=1.0)
        
        b_min = thermal_sim.view(thermal_sim.size(0), -1).min(1)[0].view(-1, 1, 1, 1)
        b_max = thermal_sim.view(thermal_sim.size(0), -1).max(1)[0].view(-1, 1, 1, 1)
        thermal_sim = (thermal_sim - b_min) / (b_max - b_min + 1e-6)
        return (thermal_sim - self.ir_mean) / self.ir_std

    def forward(self, vis_frames, ir_frames, force_mode=None):
        # Obtém dimensões do batch (B) e profundidade temporal (T)
        B, T = len(vis_frames), len(vis_frames[0])
        # Redimensiona e normaliza os frames de ambos os sensores
        x_rgb, x_ir, o_vis, o_ir = preprocess_batch(vis_frames, ir_frames, target_size=self.img_size)

        # LÓGICA DE ESPELHAMENTO / FALHA SIMULADA
        mode = force_mode 
        if self.training and mode is None:
            p = torch.rand(1).item()
            if p < 0.15: mode = "ir_only"
            elif p < 0.30: mode = "visible_only"
            else: mode = "dual"

        if mode == "ir_only":
            x_rgb = self._mirror_ir_to_vis(x_ir)
        elif mode == "visible_only":
            x_ir = self._mirror_vis_to_ir(x_rgb)

        # Processa a memória no Encoder e remodela para separar Batch e Tempo
        memory_all, gates_info, high_res_feat = self.backbone(x_rgb, x_ir)
        # Remodela features de alta resolução para uso no mecanismo de zoom (grid_sample)
        conf_rgb_m, conf_ir_m, conf_rgb_s, conf_ir_s = gates_info
        # Inicializa listas para armazenar predições de toda a sequência
        memory_all = self.encoder(memory_all).view(B, T, -1, self.d_model)
        # Remodela features de alta resolução para uso no mecanismo de zoom (grid_sample)
        high_res_feat = high_res_feat.view(B, T, self.d_model, 56, 56)
        # Inicializa listas para armazenar predições de toda a sequência
        all_boxes, all_ev, all_ei, all_eg = [], [], [], []
        # Prepara as queries iniciais repetindo o embedding para o batch
        Q_t = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        # Converte pontos de referência para espaço Logit para atualizações numéricas estáveis
        ref_vis_logits = torch.logit(self.ref_points.weight.unsqueeze(0).repeat(B, 1, 1).clamp(1e-4, 1-1e-4))
        # Inicializa os logits do IR como cópia do visível (ponto de partida comum)
        ref_ir_logits  = ref_vis_logits.clone()
        # Listas para guardar caixas específicas de cada sensor por frame
        all_boxes_vis, all_boxes_ir = [], [] # Substitui o all_boxes antigo
        # Variáveis para persistência temporal (estado do frame anterior)
        last_ev, last_ei, last_eg = None, None, None
        # Loop através de cada frame na sequência temporal
        for t in range(T):
            # Mecanismo de Propagação Temporal: usa confiança do frame t-1 para guiar t
            if t > 0 and last_eg is not None:
                # Calcula a confiança máxima entre as cabeças para decidir a taxa de atualização
                combined_conf = torch.max(torch.max(last_ev, last_ei), last_eg)
                # Alpha define o quanto manter da predição anterior (Smooth Tracking)
                alpha = torch.sigmoid(combined_conf).unsqueeze(-1) 
                # Interpolação linear das Queries entre o estado atual e o embedding base
                Q_t = Q_t * alpha + self.query_embed.weight * (1.0 - alpha)
                # Logits base das caixas de referência
                base_logits = torch.logit(self.ref_points.weight.unsqueeze(0).clamp(1e-4, 1-1e-4))
                # Propaga a localização das caixas para o próximo frame baseada na confiança
                ref_vis_logits = ref_vis_logits * alpha + base_logits * (1.0 - alpha)
                ref_ir_logits  = ref_ir_logits * alpha + base_logits * (1.0 - alpha)
            # Seleciona memória e features de alta resolução do frame atual
            mem_t, hr_t = memory_all[:, t], high_res_feat[:, t]
            # Refinamento progressivo através das camadas do Transformer
            for i, layer in enumerate(self.layers):
                # Calcula a média geométrica das predições Vis/IR para centralizar a atenção
                ref_mean = torch.sigmoid((ref_vis_logits + ref_ir_logits) / 2.0)
                # Executa a camada de atenção espacialmente focada (Soft-ROI)
                Q_t = layer(Q_t, mem_t, ref_mean)
                # Intervenção de Alta Resolução (Zoom) na camada intermediária
                if i == 2: # High-Res Zoom
                    B, N, _ = Q_t.shape
                    # O zoom também utiliza a referência média
                    # Cria grid de amostragem focado na caixa média atual
                    sampling_grid = self._make_sampling_grid(ref_mean)
                    # Expande features de alta resolução para processar cada query individualmente
                    hr_t_exp = hr_t.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, self.d_model, 56, 56)
                    # Ajusta grid para o formato exigido pelo grid_sample
                    grid_exp = sampling_grid.reshape(B * N, 7, 7, 2)
                    # Recorta e redimensiona (ROI Align manual) o patch de alta resolução
                    local_f = F.grid_sample(hr_t_exp, grid_exp, align_corners=False)
                    # Agrega informação espacial do patch e integra na Query via compressor
                    local_f = local_f.view(B, N, self.d_model, -1).mean(dim=-1) 
                    Q_t = Q_t + 0.1 * self.local_compressor(local_f)
                # Atualização desacoplada: cada sensor refina seu próprio deslocamento (delta)
                # Isso permite que a caixa IR e Visível divirjam se houver paralaxe ou oclusão
                ref_vis_logits = ref_vis_logits + self.bbox_head_vis(Q_t) * 0.1
                ref_ir_logits  = ref_ir_logits  + self.bbox_head_ir(Q_t) * 0.1
            # Calcula probabilidades de existência (classificação) para o frame atual
            ev_t = self.exist_vis_head(Q_t).squeeze(-1)
            ei_t = self.exist_ir_head(Q_t).squeeze(-1)
            eg_t = self.exist_glb_head(Q_t).squeeze(-1)
            # Converte logits finais para coordenadas [0, 1] e armazena
            all_boxes_vis.append(torch.sigmoid(ref_vis_logits))
            all_boxes_ir.append(torch.sigmoid(ref_ir_logits))
            # Armazena scores de confiança para supervisão e uso no próximo passo temporal
            all_ev.append(ev_t)
            all_ei.append(ei_t)
            all_eg.append(eg_t)
            # Atualiza o estado "anterior" para a próxima iteração do loop t
            last_ev, last_ei, last_eg = ev_t, ei_t, eg_t
        # Retorna dicionário completo com predições, métricas de fusão e metadados
        return {
            "pred_boxes_vis": torch.stack(all_boxes_vis, dim=1),
            "pred_boxes_ir":  torch.stack(all_boxes_ir, dim=1),
            "exist_vis":  torch.stack(all_ev, dim=1),
            "exist_ir":   torch.stack(all_ei, dim=1),
            "exist":      torch.stack(all_eg, dim=1),
            "gate_scores": gates_info,        
            "gate_vis_avg": conf_rgb_m.mean(), # Usando os valores desempacotados
            "gate_ir_avg":  conf_ir_m.mean(),  # Agora usa conf_ir_m
            "gate_vis_std": conf_rgb_s.mean(), # Opcional: log do desvio do RGB
            "gate_ir_std":  conf_ir_s.mean(),  # Agora temos o std do IR também
            "orig_sizes": (o_vis, o_ir)
        }

    def _make_sampling_grid(self, ref_t, size=7):
        # Obtém Batch Size (B) e Número de Queries (N) das caixas de referência
        B, N, _ = ref_t.shape
        # Converte os centros (x, y) de [0, 1] para o espaço [-1, 1] exigido pelo grid_sample
        # Fórmula: (v * 2) - 1
        centers = (ref_t[:, :, :2] * 2 - 1)
        # Define a escala do recorte (w, h) baseada nas dimensões da Bounding Box
        # Multiplicamos por 1.5 para dar uma "margem" extra ao redor do drone (contexto local)
        scales = ref_t[:, :, 2:].view(B, N, 1, 2) * 1.5
        # Cria uma linha linear de -1 a 1 com 'size' pontos (ex: 7 pontos)
        patch_range = torch.linspace(-1, 1, size, device=ref_t.device)
        # Gera um grid 2D local unitário (como se fosse uma mini-imagem centralizada em 0,0)
        gy, gx = torch.meshgrid(patch_range, patch_range, indexing='ij')
        # Empilha e formata o grid local para [1, 1, total_pontos, 2]
        rel_grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        # Projeta o grid unitário para o local correto:
        # 1. Multiplica o grid relativo pela escala da caixa (dimensionamento)
        # 2. Adiciona o centro da caixa (posicionamento no frame global)
        # 3. Clamp(-1, 1) garante que o recorte não tente sair das bordas da imagem
        return (centers.view(B, N, 1, 2) + rel_grid * scales).clamp(-1, 1)