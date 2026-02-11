import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

# ============================================================
# UTILS E MÓDULOS DE SUPORTE
# ============================================================

def preprocess_batch(vis_frames_list, ir_frames_list, target_size=(224, 224)):
    B = len(vis_frames_list)
    T = len(vis_frames_list[0])
    processed_vis, processed_ir = [], []
    orig_sizes_vis, orig_sizes_ir = [], []

    for b in range(B):
        h_v, w_v = vis_frames_list[b][0].shape[-2:]
        orig_sizes_vis.append((w_v, h_v))
        h_i, w_i = ir_frames_list[b][0].shape[-2:]
        orig_sizes_ir.append((w_i, h_i))
        
        for t in range(T):
            v, i = vis_frames_list[b][t], ir_frames_list[b][t]
            processed_vis.append(F.interpolate(v.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False))
            processed_ir.append(F.interpolate(i.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False))

    return torch.cat(processed_vis), torch.cat(processed_ir), orig_sizes_vis, orig_sizes_ir

class GatedFusionBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        self.gate_bias = nn.Parameter(torch.ones(1) * 0.5) 
        self.norm = nn.LayerNorm(d_model)

    def forward(self, f_main, f_aux):
        conf = self.gate(f_aux.mean(dim=1)).unsqueeze(1) 
        conf = torch.clamp(conf + self.gate_bias, 0.1, 0.9)
        f_fused, _ = self.cross_attn(f_main, f_aux * conf, f_aux * conf)
        return self.norm(f_main + f_fused), conf

# ============================================================
# 1. BACKBONE RGBT
# ============================================================

class RGBTBackbone(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        rgb_net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        ir_net  = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        ir_net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        self.rgb_low_level = nn.Sequential(*list(rgb_net.children())[:5]) 
        self.ir_low_level  = nn.Sequential(*list(ir_net.children())[:5])
        self.rgb_deep = nn.Sequential(*list(rgb_net.children())[5:7]) 
        self.ir_deep  = nn.Sequential(*list(ir_net.children())[5:7])  
        
        self.proj_rgb = nn.Conv2d(256, d_model, 1)
        self.proj_ir  = nn.Conv2d(256, d_model, 1)
        self.proj_high_res = nn.Conv2d(64, d_model, 1) 

        self.rgb_enhanced_by_ir = GatedFusionBlock(d_model, nhead)
        self.ir_enhanced_by_rgb = GatedFusionBlock(d_model, nhead)
        self.bottleneck = nn.Linear(d_model * 2, d_model)

    def forward(self, x_rgb, x_ir):
        low_rgb = self.rgb_low_level(x_rgb)
        low_ir  = self.ir_low_level(x_ir)
        f_high_res = self.proj_high_res(low_rgb + low_ir) 

        f_rgb = self.proj_rgb(self.rgb_deep(low_rgb)).flatten(2).permute(0, 2, 1)
        f_ir  = self.proj_ir(self.ir_deep(low_ir)).flatten(2).permute(0, 2, 1)

        f_rgb_fused, conf_ir = self.rgb_enhanced_by_ir(f_rgb, f_ir)
        f_ir_fused, conf_rgb = self.ir_enhanced_by_rgb(f_ir, f_rgb)

        fused = self.bottleneck(torch.cat([f_rgb_fused, f_ir_fused], dim=-1))
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
        self.ref_points  = nn.Embedding(n_queries, 4) 
        nn.init.constant_(self.ref_points.weight[:, 2:], 0.05)

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
        ref_t = self.ref_points.weight.unsqueeze(0).repeat(B, 1, 1)
        last_exist = None

        for t in range(T):
            # Melhoria 1: Identidade Temporal Suave (Sem flicker de hard threshold)
            if t > 0 and last_exist is not None:
                alpha = torch.sigmoid(last_exist).unsqueeze(-1) # Confiança contínua
                Q_t = Q_t * alpha + self.query_embed.weight * (1.0 - alpha)
                ref_t = ref_t * alpha + self.ref_points.weight.unsqueeze(0) * (1.0 - alpha)
            
            mem_t, hr_t = memory_all[:, t], high_res_feat[:, t]
            
            for i, layer in enumerate(self.layers):
                Q_t = layer(Q_t, mem_t, ref_t)
                if i == 2: # Melhoria 4: High-Res Zoom
                    sampling_grid = self._make_sampling_grid(ref_t)
                    local_f = F.grid_sample(hr_t, sampling_grid, align_corners=False)
                    local_f = local_f.permute(0, 2, 3, 1).mean(dim=2) 
                    Q_t = Q_t + 0.1 * self.local_compressor(local_f)

                delta = self.bbox_head(Q_t).tanh() * 0.2
                ref_t = torch.sigmoid(torch.logit(ref_t.clamp(1e-4, 1-1e-4)) + delta)

            eg_t = self.exist_glb_head(Q_t)
            all_boxes.append(ref_t)
            all_ev.append(self.exist_vis_head(Q_t))
            all_ei.append(self.exist_ir_head(Q_t))
            all_eg.append(eg_t)
            last_exist = eg_t.squeeze(-1)

        return {
            "pred_boxes": torch.stack(all_boxes, dim=1),
            "exist_vis": torch.sigmoid(torch.stack(all_ev, dim=1)).squeeze(-1),
            "exist_ir": torch.sigmoid(torch.stack(all_ei, dim=1)).squeeze(-1),
            "exist": torch.max(torch.sigmoid(torch.stack(all_eg, dim=1)), 
                     torch.max(torch.sigmoid(torch.stack(all_ev, dim=1)), 
                               torch.sigmoid(torch.stack(all_ei, dim=1)))).squeeze(-1),
            "gate_scores": gates, 
            "gate_vis_avg": gates[0].mean(), "gate_ir_avg": gates[1].mean(),
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