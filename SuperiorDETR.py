import math
import timm 
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

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

class GatedFusionBlock(nn.Module):
    def __init__(self, d_model, nhead, temperature=2.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.temperature = temperature # <--- O fator de suavização
        
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )
        # Inicializa bias em 0 para que sigmoid(0/T) seja exatamente 0.5
        nn.init.constant_(self.gate[-1].bias, 0.0) 
        self.norm = nn.LayerNorm(d_model)

    def forward(self, f_main, f_aux):
        # Aplicamos a temperatura na divisão antes da Sigmoid
        # conf = sigmoid(logits / T)
        gate_logits = self.gate(f_aux.mean(dim=1))
        conf = torch.sigmoid(gate_logits / self.temperature).unsqueeze(1) 
        
        f_fused, _ = self.cross_attn(f_main, f_aux, f_aux)
        
        # O fluxo de informação de f_aux para f_main agora é mais "resistente" 
        # a ser zerado abruptamente por variações de gradiente no início.
        return self.norm(f_main + conf * f_fused), conf
    
class GatedFusionBlock_sigmoid_simples_old(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1) # Saída linear para ser usada com Sigmoid
        )
        # gate para começar em 0.5 (neutro)
        nn.init.constant_(self.gate[-1].bias, 0.0) 
        self.norm = nn.LayerNorm(d_model)

    def forward(self, f_main, f_aux):
        # Confiança aprendida puramente pelos dados, sem clamps artificiais
        conf = torch.sigmoid(self.gate(f_aux.mean(dim=1))).unsqueeze(1) 
        
        f_fused, _ = self.cross_attn(f_main, f_aux, f_aux)
        
        # f_main recebe a informação de f_aux pesada pela confiança
        # Tratamento igualitário: o modelo decide o escalar conf [0, 1]
        return self.norm(f_main + conf * f_fused), conf

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
        # Isso remove o erro de "Unexpected keys" e mantém o aprendizado
        old_conv = self.ir_net.conv_stem
        new_conv = nn.Conv2d(1, old_conv.out_channels, 
                             kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, 
                             padding=old_conv.padding, bias=False)
        
        with torch.no_grad():
            # Fazemos a média dos pesos dos 3 canais RGB para o canal único IR
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
        self.rgb_enhanced_by_ir = GatedFusionBlock(d_model, nhead)
        # IR recebe informação do RGB ponderada por outro gate independente
        self.ir_enhanced_by_rgb = GatedFusionBlock(d_model, nhead)
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
        # ============================================================
        # TESTE DE SANIDADE: APAGAR RGB (Temporário) - passou!
        # Ao zerar esses dois tensores, o modelo não recebe NADA do Visível.
        # Ele será forçado a reconstruir o objeto apenas via braço IR.
        # ============================================================
        #f_rgb_low = torch.zeros_like(f_rgb_low)
        #f_rgb_deep = torch.zeros_like(f_rgb_deep)
        # ============================================================
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
        f_rgb_fused, conf_ir = self.rgb_enhanced_by_ir(f_rgb, f_ir)
        # IR recebe contexto do RGB, ponderado por outra confiança
        f_ir_fused, conf_rgb = self.ir_enhanced_by_rgb(f_ir, f_rgb)
        # AGREGAÇÃO FINAL
        # Concatena as duas representações já fundidas
        # Bottleneck reduz para d_model
        fused = self.bottleneck(torch.cat([f_rgb_fused, f_ir_fused], dim=-1))
        # Retorna:
        # - fused: memória multimodal pronta para o Transformer
        # - confs: scores médios de gate (úteis para diagnóstico)
        # - f_high_res: feature espacial para refinamento posterior
        return fused, (conf_rgb, conf_ir), f_high_res

# ============================================================
# 2. REFINEMENT LAYER (MELHORIA 3: SOFT ROI-ATTENTION)
# ============================================================

class RefinementLayer(nn.Module):
    def __init__(self, d_model, nhead, layer_idx=0):
        super().__init__()
        self.nhead = nhead
        self.layer_idx = layer_idx
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, q, memory, ref_points):
        q = self.norm1(q + self.self_attn(q, q, q)[0])
        
        B, N, _ = q.shape
        M = memory.shape[1]
        grid_side = int(math.sqrt(M))
        
        # Grid dinâmico para evitar desalinhamento se M não for quadrado perfeito
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, grid_side, device=q.device), 
            torch.linspace(0, 1, grid_side, device=q.device), indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 2)[:, :M, :]
        
        dist = torch.cdist(ref_points[:, :, :2], grid) 
        sigma = 0.5 / (self.layer_idx + 1)
        
        attn_bias = -(dist / sigma).pow(2) 
        attn_bias = attn_bias.repeat_interleave(self.nhead, dim=0)

        q_focussed, _ = self.cross_attn(q, memory, memory, attn_mask=attn_bias)
        q = self.norm2(q + q_focussed)
        q = self.norm3(q + self.mlp(q))
        return q

# ============================================================
# 3. SUPERIOR DETR
# ============================================================

class SuperiorDETR(nn.Module):
    def __init__(self, d_model=256, n_queries=30, n_layers=6, img_size=(224, 224)):
        super().__init__()
        self.img_size = img_size
        self.d_model = d_model
        self.backbone = RGBTBackbone(d_model)
        self.query_embed = nn.Embedding(n_queries, d_model)

        self.ref_points = nn.Embedding(n_queries, 4)
        # Inicialização Proporcional (Grid)
        with torch.no_grad():
            # Cria um grid (ex: se n_queries=20, tentamos algo perto de 4x5 ou 2x10)
            side_x = int(math.sqrt(n_queries))
            side_y = n_queries // side_x
            
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0.1, 0.9, side_y),
                torch.linspace(0.1, 0.9, side_x),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            
            # Se sobrar query por conta da divisão, as outras ficam no centro
            self.ref_points.weight.data.fill_(0.5) 
            self.ref_points.weight.data[:grid.size(0), :2] = grid
            # Tamanho inicial pequeno (5% do frame)
            self.ref_points.weight.data[:, 2:] = 0.05

        self.bbox_head = nn.Linear(d_model, 4)
        self.exist_vis_head = nn.Linear(d_model, 1)
        self.exist_ir_head  = nn.Linear(d_model, 1)
        self.exist_glb_head = nn.Linear(d_model, 1) 

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, 1024, batch_first=True), num_layers=2)
        self.layers = nn.ModuleList([RefinementLayer(d_model, 8, layer_idx=i) for i in range(n_layers)])
        self.local_compressor = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, vis_frames, ir_frames):
        B, T = len(vis_frames), len(vis_frames[0])
        x_rgb, x_ir, o_vis, o_ir = preprocess_batch(vis_frames, ir_frames, target_size=self.img_size)

        memory_all, gates, high_res_feat = self.backbone(x_rgb, x_ir)
        memory_all = self.encoder(memory_all).view(B, T, -1, self.d_model)
        high_res_feat = high_res_feat.view(B, T, self.d_model, 56, 56)

        all_boxes, all_ev, all_ei, all_eg = [], [], [], []
        Q_t = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        ref_t_logits = torch.logit(self.ref_points.weight.unsqueeze(0).repeat(B, 1, 1).clamp(1e-4, 1-1e-4))
        
        # Inicializamos os estados anteriores como None
        last_ev, last_ei, last_eg = None, None, None

        for t in range(T):
            # Lógica Temporal Robusta
            if t > 0 and last_eg is not None:
                # Pegamos a confiança máxima entre os 3 sinais do frame anterior
                # Isso garante compatibilidade total com o seu MultimodalCriterion
                combined_conf = torch.max(torch.max(last_ev, last_ei), last_eg)
                alpha = torch.sigmoid(combined_conf).unsqueeze(-1) 
                
                Q_t = Q_t * alpha + self.query_embed.weight * (1.0 - alpha)
                base_logits = torch.logit(self.ref_points.weight.unsqueeze(0).clamp(1e-4, 1-1e-4))
                ref_t_logits = ref_t_logits * alpha + base_logits * (1.0 - alpha)
            
            mem_t, hr_t = memory_all[:, t], high_res_feat[:, t]
            
            for i, layer in enumerate(self.layers):
                ref_t = torch.sigmoid(ref_t_logits)
                Q_t = layer(Q_t, mem_t, ref_t)
                if i == 2: # High-Res Zoom
                    B, N, _ = Q_t.shape # 8, 20
                    sampling_grid = self._make_sampling_grid(ref_t) # [8, 20, 49, 2]
                    
                    # FATO: Expandimos a imagem para que cada query tenha sua própria "cópia" para amostrar
                    # [8, 256, 56, 56] -> [8, 20, 256, 56, 56] -> [160, 256, 56, 56]
                    hr_t_exp = hr_t.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, self.d_model, 56, 56)
                    
                    # FATO: Ajustamos o grid para ser 160 amostragens de 7x7 (total 49 pontos)
                    # [8, 20, 49, 2] -> [160, 7, 7, 2]
                    grid_exp = sampling_grid.reshape(B * N, 7, 7, 2)
                    
                    # Agora sim: 160 imagens sendo amostradas por 160 grids
                    local_f = F.grid_sample(hr_t_exp, grid_exp, align_corners=False) # [160, 256, 7, 7]
                    
                    # Voltamos para o shape da Query: [8, 20, 256]
                    local_f = local_f.view(B, N, self.d_model, -1).mean(dim=-1) 
                    
                    Q_t = Q_t + 0.1 * self.local_compressor(local_f)

                delta = self.bbox_head(Q_t) # Sem tanh() aqui para permitir variação no espaço logit
                ref_t_logits = ref_t_logits + delta * 0.1

            # Calculamos as saídas das 3 cabeças
            ev_t = self.exist_vis_head(Q_t).squeeze(-1)
            ei_t = self.exist_ir_head(Q_t).squeeze(-1)
            eg_t = self.exist_glb_head(Q_t).squeeze(-1)

            # Guardamos para o retorno
            all_boxes.append(ref_t)
            all_ev.append(ev_t)
            all_ei.append(ei_t)
            all_eg.append(eg_t)
            
            # Atualizamos os estados para o próximo frame (t+1)
            last_ev, last_ei, last_eg = ev_t, ei_t, eg_t

        return {
            "pred_boxes": torch.stack(all_boxes, dim=1),
            "exist_vis":  torch.stack(all_ev, dim=1),
            "exist_ir":   torch.stack(all_ei, dim=1),
            "exist":      torch.stack(all_eg, dim=1),
            "gate_scores": gates, 
            "gate_vis_avg": gates[0].mean(), 
            "gate_ir_avg": gates[1].mean(),
            "orig_sizes": (o_vis, o_ir)
        }

    def _make_sampling_grid(self, ref_t, size=7):
        B, N, _ = ref_t.shape
        centers = (ref_t[:, :, :2] * 2 - 1)
        scales = ref_t[:, :, 2:].view(B, N, 1, 2) * 1.5
        patch_range = torch.linspace(-1, 1, size, device=ref_t.device)
        gy, gx = torch.meshgrid(patch_range, patch_range, indexing='ij')
        rel_grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
        return (centers.view(B, N, 1, 2) + rel_grid * scales).clamp(-1, 1)