"""Pi0.5 完整模型定义 — 从 full_model_noconv.pt 构建, 用于 ONNX 导出.

包含 SigLIP Vision Tower + PaliGemma Encoder + Gemma Expert Decoder + 动作输出头.
仅依赖 torch, 独立可运行.

架构:
  images(3x224x224) + prompt → SigLIP → multi_modal_projector → 
  embed_tokens(prompt) → PaliGemma Encoder(prefix KV cache) →
  10x denoise: embed_suffix → Expert Decoder(with KV cache) → action_out →
  actions [1, 50, 32]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════ SigLIP Vision Tower ═══════════════════════════

class SigLIPAttention(nn.Module):
    def __init__(self, hidden=1152, nheads=16):
        super().__init__()
        self.nheads = nheads
        self.hd = hidden // nheads
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.nheads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.nheads, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.nheads, self.hd).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).contiguous().view(b, s, -1)
        return self.out_proj(x)


class SigLIPMLP(nn.Module):
    def __init__(self, hidden=1152, inter=4304):
        super().__init__()
        self.fc1 = nn.Linear(hidden, inter)
        self.fc2 = nn.Linear(inter, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SigLIPEncoderLayer(nn.Module):
    def __init__(self, hidden=1152, nheads=16, inter=4304, eps=1e-6):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps)
        self.self_attn = SigLIPAttention(hidden, nheads)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = SigLIPMLP(hidden, inter)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SigLIPVisionTower(nn.Module):
    """SigLIP ViT: patch_embed → 27 encoder layers → post_layernorm."""
    def __init__(self, hidden=1152, nheads=16, inter=4304, nlayers=27, patch_size=14, image_size=224):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2  # 256
        self.patch_embedding = nn.Conv2d(3, hidden, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Embedding(num_patches, hidden)
        self.layers = nn.ModuleList([SigLIPEncoderLayer(hidden, nheads, inter, eps=1e-6) for _ in range(nlayers)])
        self.post_layernorm = nn.LayerNorm(hidden, eps=1e-6)
        self.register_buffer("position_ids", torch.arange(num_patches).unsqueeze(0))

    def forward(self, pixel_values):
        """pixel_values: [B, 3, 224, 224] → [B, 256, 1152]"""
        x = self.patch_embedding(pixel_values)  # [B, 1152, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [B, 256, 1152]
        x = x + self.position_embedding(self.position_ids)
        for layer in self.layers:
            x = layer(x)
        return self.post_layernorm(x)


# ═══════════════════════════ Gemma RMSNorm ═══════════════════════════

class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: y = x_normed * (1 + weight)"""
    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden))
        self.eps = eps

    def forward(self, x):
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = (x * torch.rsqrt(var + self.eps)).to(x.dtype)
        return normed * (1.0 + self.weight)


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm for pi0.5 expert layers. dense: [3*hidden, hidden] → (scale, shift, gate)"""
    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(hidden, 3 * hidden, bias=True)

    def forward(self, x, cond):
        """Returns (normed, gate). Gate used for gated residual connections."""
        dtype = x.dtype
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        proj = self.dense(cond)
        if proj.dim() < normed.dim():
            proj = proj.unsqueeze(1)
        scale, shift, gate = proj.chunk(3, dim=-1)
        normed = normed * (1 + scale.to(torch.float32)) + shift.to(torch.float32)
        return normed.to(dtype), gate.to(dtype)


# ═══════════════════════════ RoPE ═══════════════════════════

class RoPE(nn.Module):
    """RoPE matching reference's compute_rope exactly:
    timescale = 10000^((2/D)*arange(D/2)), radians = pos / timescale
    """
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        # Use reference's exact formula: freq_exponents → timescale
        d_half = dim // 2
        freq_exponents = (2.0 / dim) * torch.arange(d_half, dtype=torch.float32)
        self.register_buffer("timescale", base ** freq_exponents)  # [D/2]

    def compute(self, position_ids, device):
        """Compute cos/sin on-the-fly (matches reference's compute_rope)."""
        d_half = self.dim // 2
        freq_exponents = (2.0 / self.dim) * torch.arange(d_half, dtype=torch.float32, device=device)
        timescale = 10000.0 ** freq_exponents
        radians = position_ids[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
        radians = radians[..., None, :]   # [B, S, 1, D/2]
        emb = torch.cat((radians, radians), dim=-1)  # [B, S, 1, D]
        return torch.cos(emb), torch.sin(emb)

    def apply(self, x, position_ids):
        """Apply RoPE. x: [B, nH, S, hd], position_ids: [B, S]"""
        cos, sin = self.compute(position_ids, x.device)
        # cos/sin: [B, S, 1, D] → need [B, 1, S, D] for (B,nH,S,D)
        cos = cos.squeeze(-2).unsqueeze(-3)  # [B, 1, S, D]
        sin = sin.squeeze(-2).unsqueeze(-3)
        dtype = x.dtype
        x = x.float()
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        out = x * cos + torch.cat((-x2, x1), dim=-1) * sin
        return out.to(dtype)


# ═══════════════════════════ PaliGemma (Backbone) Layer ═══════════════════════════

class GemmaDecoderLayer(nn.Module):
    """PaliGemma backbone transformer layer (Gemma 2B style).
    
    Supports both Linear and Conv2d projections. Call prepare_conv() to match
    the reference model's Conv2d numerical behavior.
    """
    def __init__(self, hidden=2048, nq=8, nkv=1, hd=256, inter=16384):
        super().__init__()
        self.hidden = hidden
        self.nq, self.nkv, self.hd = nq, nkv, hd
        self.kv_groups = nq // nkv
        self.input_layernorm = GemmaRMSNorm(hidden)
        self.post_attention_layernorm = GemmaRMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, nq * hd, bias=False)
        self.k_proj = nn.Linear(hidden, nkv * hd, bias=False)
        self.v_proj = nn.Linear(hidden, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nq * hd, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.rope = RoPE(hd, max_seq_len=2048)
        self._use_conv = False

    def prepare_conv(self):
        """Convert Linear projections to Conv2d to match reference numerical behavior."""
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            linear = getattr(self, name)
            conv = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=False)
            conv.weight.data.copy_(linear.weight[:, :, None, None])
            conv.to(device=linear.weight.device, dtype=linear.weight.dtype)
            setattr(self, f"{name}_conv", conv)
            delattr(self, name)
        self._use_conv = True

    def _proj(self, x, name):
        """Apply projection (Conv2d or Linear)."""
        if self._use_conv:
            b, s, d = x.shape
            hs = x.reshape(b, s, 1, d).transpose(1, 3)  # [B, d, 1, S]
            out = getattr(self, f"{name}_conv")(hs)
            return out.transpose(1, 3).reshape(b, s, -1)
        return getattr(self, name)(x)

    def _mlp_forward(self, x):
        """MLP with Conv2d or Linear."""
        if self._use_conv:
            b, s, d = x.shape
            hs = x.reshape(b, s, 1, d).transpose(1, 3)
            out = getattr(self, 'down_proj_conv')(
                F.gelu(getattr(self, 'gate_proj_conv')(hs), approximate="tanh") *
                getattr(self, 'up_proj_conv')(hs))
            return out.transpose(1, 3).reshape(b, s, -1)
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

    def forward_with_cache(self, x, position_ids, attention_mask=None):
        """Forward pass that returns KV cache."""
        b, s, _ = x.shape
        res = x
        x = self.input_layernorm(x)
        
        # Projections → (B, S, nH, hd)
        q = self._proj(x, 'q_proj').view(b, s, self.nq, self.hd)
        k = self._proj(x, 'k_proj').view(b, s, self.nkv, self.hd)
        v = self._proj(x, 'v_proj').view(b, s, self.nkv, self.hd)
        
        # RoPE in (B, S, H, D) format
        cos, sin = self.rope.compute(position_ids, x.device)  # (B, S, 1, D)
        dtype = q.dtype
        q_f, k_f = q.float(), k.float()
        q1, q2 = q_f[..., :self.hd//2], q_f[..., self.hd//2:]
        k1, k2 = k_f[..., :self.hd//2], k_f[..., self.hd//2:]
        q = (q_f * cos + torch.cat((-q2, q1), dim=-1) * sin).to(dtype)
        k = (k_f * cos + torch.cat((-k2, k1), dim=-1) * sin).to(dtype)
        
        # Save KV cache
        cached_k = k.transpose(1, 2).clone()  # (B, nkv, S, hd)
        cached_v = v.transpose(1, 2).clone()
        
        # Attention
        q_t = q.transpose(1, 2)  # (B, nq, S, hd)
        k_t = k.transpose(1, 2).unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.nq, s, self.hd)
        v_t = v.transpose(1, 2).unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.nq, s, self.hd)
        
        scores = torch.matmul(q_t, k_t.transpose(2, 3)) * (self.hd ** -0.5)
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :s]
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
        
        attn_out = torch.matmul(attn, v_t).transpose(1, 2).contiguous().reshape(b, s, -1)
        attn_out = self._proj(attn_out, 'o_proj')
        
        x = res + attn_out
        res = x
        x = self.post_attention_layernorm(x)
        x = res + self._mlp_forward(x)
        return x, cached_k, cached_v


# ═══════════════════════════ Expert Decoder Layer (pi0.5) ═══════════════════════════

class ExpertDecoderLayer(nn.Module):
    """Expert decoder layer with KV-cache from backbone prefix and AdaRMSNorm."""
    def __init__(self, hidden=1024, nq=8, nkv=1, hd=256, inter=4096):
        super().__init__()
        self.hidden = hidden
        self.nq, self.nkv, self.hd = nq, nkv, hd
        self.kv_groups = nq // nkv
        self.input_layernorm = AdaRMSNorm(hidden)
        self.post_attention_layernorm = AdaRMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, nq * hd, bias=False)
        self.k_proj = nn.Linear(hidden, nkv * hd, bias=False)
        self.v_proj = nn.Linear(hidden, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nq * hd, hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.rope = RoPE(hd, max_seq_len=2048, base=10000.0)
        self._use_conv = False

    def prepare_conv(self):
        """Convert Linear projections to Conv2d to match reference numerical behavior."""
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            linear = getattr(self, name)
            conv = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=False)
            conv.weight.data.copy_(linear.weight[:, :, None, None])
            conv.to(device=linear.weight.device, dtype=linear.weight.dtype)
            setattr(self, f"{name}_conv", conv)
            delattr(self, name)
        self._use_conv = True

    def _proj(self, x, name):
        if self._use_conv:
            b, s, d = x.shape
            hs = x.reshape(b, s, 1, d).transpose(1, 3)
            out = getattr(self, f"{name}_conv")(hs)
            return out.transpose(1, 3).reshape(b, s, -1)
        return getattr(self, name)(x)

    def forward(self, x, position_ids, adarms_cond, prefix_k=None, prefix_v=None, attention_mask=None):
        """
        x: [B, suffix_len, 1024]
        position_ids: [B, suffix_len] (offset by prefix_len)
        adarms_cond: [B, 1024] (time conditioning)
        prefix_k: [B, nkv, prefix_len, hd] (cached from backbone)
        prefix_v: [B, nkv, prefix_len, hd]
        attention_mask: [B, 1, suffix_len, prefix_len+suffix_len]
        """
        b, s, _ = x.shape
        # Input layernorm with gate
        normed, gate = self.input_layernorm(x, adarms_cond)
        q = self._proj(normed, 'q_proj').view(b, s, self.nq, self.hd).transpose(1, 2)
        k = self._proj(normed, 'k_proj').view(b, s, self.nkv, self.hd).transpose(1, 2)
        v = self._proj(normed, 'v_proj').view(b, s, self.nkv, self.hd).transpose(1, 2)
        q = self.rope.apply(q, position_ids)
        k = self.rope.apply(k, position_ids)
        # Prepend prefix KV cache
        if prefix_k is not None:
            k = torch.cat([prefix_k, k], dim=2)
            v = torch.cat([prefix_v, v], dim=2)
        kv_len = k.shape[2]
        k = k.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.nq, kv_len, self.hd)
        v = v.unsqueeze(2).expand(-1, -1, self.kv_groups, -1, -1).reshape(b, self.nq, kv_len, self.hd)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hd)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1)
        attn_out = self._proj(
            torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1), 'o_proj')
        # Gated residual for attention
        x = x + attn_out * gate
        # Post-attention layernorm with gate
        normed, gate = self.post_attention_layernorm(x, adarms_cond)
        if self._use_conv:
            b_s, s_s, d_s = normed.shape
            hs = normed.reshape(b_s, s_s, 1, d_s).transpose(1, 3)
            mlp_out = getattr(self, 'down_proj_conv')(
                F.gelu(getattr(self, 'gate_proj_conv')(hs), approximate="tanh") *
                getattr(self, 'up_proj_conv')(hs))
            mlp_out = mlp_out.transpose(1, 3).reshape(b_s, s_s, -1)
        else:
            mlp_out = self.down_proj(F.gelu(self.gate_proj(normed), approximate="tanh") * self.up_proj(normed))
        # Gated residual for MLP
        return x + mlp_out * gate


# ═══════════════════════════ Complete Pi0.5 Model ═══════════════════════════

def sinusoidal_pos_embedding(t, dim, min_p=4e-3, max_p=4.0):
    """与基准 UnifiedFlowMatching.scaling_factor 一致, 使用 float64 精度."""
    assert dim % 2 == 0
    # 与基准一致: scaling_factor 用 float64 计算
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=t.device, dtype=torch.float64)
    period = min_p * (max_p / min_p) ** fraction
    scaling = (1.0 / period) * 2 * math.pi
    # sin/cos 也在 float64 下计算, 然后转回 float32
    angle = t.unsqueeze(-1).to(torch.float64) * scaling.unsqueeze(0)
    return torch.cat([angle.sin(), angle.cos()], dim=-1).to(torch.float32)


def make_att_2d_masks(pad_masks, att_masks):
    """从 openpi big_vision 复制. 显式 cast 以兼容 ONNX."""
    cumsum = torch.cumsum(att_masks.long(), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :].long() * pad_masks[:, :, None].long()
    return att_2d.bool() & pad_2d.bool()


class Pi05FullModel(nn.Module):
    """完整 pi0.5 模型: SigLIP + PaliGemma + Expert Decoder.
    
    流程:
    1. embed_prefix: SigLIP编码图像 + embed_tokens编码语言 → prefix embeddings
    2. encode_prefix: PaliGemma encoder → KV cache (18 layers)
    3. denoise_loop: 10步 Euler flow-matching 去噪
       - embed_suffix: noisy_actions → action_in_proj + time_mlp
       - 每步: expert decoder(suffix_emb, KV_cache, time_emb) → velocity → Euler更新
    4. action_out_proj → actions [B, 50, 32]
    """

    def __init__(self,
                 vlm_hidden=2048, vlm_nq=8, vlm_nkv=1, vlm_hd=256, vlm_inter=16384, vlm_layers=18,
                 expert_hidden=1024, expert_nq=8, expert_nkv=1, expert_hd=256, expert_inter=4096, expert_layers=18,
                 siglip_hidden=1152, siglip_nheads=16, siglip_inter=4304, siglip_layers=27,
                 vocab_size=257152, action_dim=32, chunk_size=50, num_steps=10):
        super().__init__()
        self.vlm_hidden = vlm_hidden
        self.expert_hidden = expert_hidden
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_steps = num_steps

        # SigLIP Vision Tower
        self.vision_tower = SigLIPVisionTower(siglip_hidden, siglip_nheads, siglip_inter, siglip_layers)
        # Multi-modal projector
        self.multi_modal_projector = nn.Linear(siglip_hidden, vlm_hidden)
        # Language embedding
        self.embed_tokens = nn.Embedding(vocab_size, vlm_hidden)
        # PaliGemma backbone layers
        self.backbone_layers = nn.ModuleList([
            GemmaDecoderLayer(vlm_hidden, vlm_nq, vlm_nkv, vlm_hd, vlm_inter)
            for _ in range(vlm_layers)
        ])
        # Expert decoder layers
        self.expert_layers = nn.ModuleList([
            ExpertDecoderLayer(expert_hidden, expert_nq, expert_nkv, expert_hd, expert_inter)
            for _ in range(expert_layers)
        ])
        # Expert final norm (AdaRMS — used as final norm in pi0.5)
        self.expert_final_norm = AdaRMSNorm(expert_hidden)
        # Action projections
        self.action_in_proj = nn.Linear(action_dim, expert_hidden)
        self.time_mlp_in = nn.Linear(expert_hidden, expert_hidden)
        self.time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, action_dim)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """编码图像和语言 → prefix embeddings, pad_masks, att_masks."""
        embs = []
        pad_masks = []
        att_masks_list = []

        for img, mask in zip(images, img_masks):
            img_emb = self.vision_tower(img)  # [B, 256, 1152]
            img_emb = self.multi_modal_projector(img_emb)  # [B, 256, 2048]
            b, n = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(mask[:, None].expand(b, n))
            att_masks_list.extend([0] * n)

        # Language
        lang_emb = self.embed_tokens(lang_tokens)  # [B, T, 2048]
        lang_emb = lang_emb * math.sqrt(self.vlm_hidden)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks_list.extend([0] * lang_emb.shape[1])

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        b = pad_masks.shape[0]
        att_masks = torch.tensor(att_masks_list, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(b, -1)
        return embs, pad_masks, att_masks

    def encode_prefix(self, prefix_embs, prefix_pad_masks, prefix_att_masks):
        """PaliGemma encoder: 生成 KV cache (每层一对 K, V)."""
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        att_4d = torch.where(att_2d[:, None, :, :], 0.0, -2.3819763e38).to(prefix_embs.dtype)
        position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1

        kv_cache = []  # list of (K, V) per layer
        x = prefix_embs
        for layer in self.backbone_layers:
            x, cached_k, cached_v = layer.forward_with_cache(x, position_ids, att_4d)
            kv_cache.append((cached_k, cached_v))
        return kv_cache, prefix_pad_masks

    def embed_suffix(self, noisy_actions, time_emb):
        """Action + time embedding → suffix embeddings + adarms_cond."""
        act_emb = self.action_in_proj(noisy_actions)  # [B, 50, 1024]
        adarms_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))  # [B, 1024]
        return act_emb, adarms_cond

    def denoise_step(self, suffix_embs, adarms_cond, kv_cache, prefix_pad_masks, position_ids, attention_mask):
        """一步去噪: expert decoder with KV cache.
        
        backbone and expert share same KV geometry: nkv=1, hd=256.
        KV cache from backbone can be directly concatenated.
        """
        x = suffix_embs
        for i, layer in enumerate(self.expert_layers):
            pk, pv = kv_cache[i]
            x = layer(x, position_ids, adarms_cond, pk, pv, attention_mask)
        x, _ = self.expert_final_norm(x, adarms_cond)
        return self.action_out_proj(x.float())  # velocity [B, 50, 32]

    @torch.no_grad()
    def forward(self, images, img_masks, lang_tokens, lang_masks, noise=None):
        """完整推理: images + prompt → actions.
        
        Args:
            images: list of [B, 3, 224, 224] tensors (3 cameras)
            img_masks: list of [B] bool tensors
            lang_tokens: [B, 200] long tensor
            lang_masks: [B, 200] bool tensor
            noise: [B, 50, 32] or None (random if None)
        
        Returns:
            actions: [B, 50, 32]
        """
        b = lang_tokens.shape[0]
        device = lang_tokens.device

        # 1. Encode prefix (once)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks)
        kv_cache, prefix_pad_masks = self.encode_prefix(
            prefix_embs, prefix_pad_masks, prefix_att_masks)

        # 2. Setup denoise
        if noise is None:
            noise = torch.randn(b, self.chunk_size, self.action_dim, device=device)
        x_t = noise
        dt = -1.0 / self.num_steps

        # Pre-compute suffix attention structure
        suffix_len = self.chunk_size  # 50
        prefix_len = prefix_pad_masks.shape[1]
        suffix_pad_masks = torch.ones(b, suffix_len, dtype=torch.bool, device=device)
        suffix_att = torch.zeros(b, suffix_len, dtype=torch.bool, device=device)
        suffix_att[:, 0] = True  # first action token is causal boundary
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att)
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(b, suffix_len, prefix_len)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        full_att_4d = torch.where(full_att_2d[:, None, :, :], 0.0, -2.3819763e38).to(prefix_embs.dtype)

        prefix_offsets = torch.sum(prefix_pad_masks.long(), dim=-1)[:, None]
        suffix_pos_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1

        # 3. 10-step Euler flow-matching
        time_val = 1.0
        for step in range(self.num_steps):
            time_tensor = torch.tensor([time_val], dtype=torch.float32, device=device).expand(b)
            time_emb = sinusoidal_pos_embedding(time_tensor, self.expert_hidden)

            suffix_embs, adarms_cond = self.embed_suffix(x_t, time_emb)
            v_t = self.denoise_step(suffix_embs, adarms_cond, kv_cache,
                                     prefix_pad_masks, suffix_pos_ids, full_att_4d)
            x_t = x_t + dt * v_t
            time_val += dt

        return x_t


# ═══════════════════════════ 从 full_model_noconv.pt 加载权重 ═══════════════════════════

def _load_siglip(model, sd, prefix="paligemma_with_expert.vision_tower.vision_model."):
    """加载 SigLIP 权重."""
    vt = model.vision_tower
    vt.patch_embedding.weight.data.copy_(sd[f"{prefix}embeddings.patch_embedding.weight"])
    vt.patch_embedding.bias.data.copy_(sd[f"{prefix}embeddings.patch_embedding.bias"])
    vt.position_embedding.weight.data.copy_(sd[f"{prefix}embeddings.position_embedding.weight"])
    vt.post_layernorm.weight.data.copy_(sd[f"{prefix}post_layernorm.weight"])
    vt.post_layernorm.bias.data.copy_(sd[f"{prefix}post_layernorm.bias"])
    for i, layer in enumerate(vt.layers):
        lp = f"{prefix}encoder.layers.{i}."
        layer.layer_norm1.weight.data.copy_(sd[f"{lp}layer_norm1.weight"])
        layer.layer_norm1.bias.data.copy_(sd[f"{lp}layer_norm1.bias"])
        layer.layer_norm2.weight.data.copy_(sd[f"{lp}layer_norm2.weight"])
        layer.layer_norm2.bias.data.copy_(sd[f"{lp}layer_norm2.bias"])
        layer.self_attn.q_proj.weight.data.copy_(sd[f"{lp}self_attn.q_proj.weight"])
        layer.self_attn.q_proj.bias.data.copy_(sd[f"{lp}self_attn.q_proj.bias"])
        layer.self_attn.k_proj.weight.data.copy_(sd[f"{lp}self_attn.k_proj.weight"])
        layer.self_attn.k_proj.bias.data.copy_(sd[f"{lp}self_attn.k_proj.bias"])
        layer.self_attn.v_proj.weight.data.copy_(sd[f"{lp}self_attn.v_proj.weight"])
        layer.self_attn.v_proj.bias.data.copy_(sd[f"{lp}self_attn.v_proj.bias"])
        layer.self_attn.out_proj.weight.data.copy_(sd[f"{lp}self_attn.out_proj.weight"])
        layer.self_attn.out_proj.bias.data.copy_(sd[f"{lp}self_attn.out_proj.bias"])
        layer.mlp.fc1.weight.data.copy_(sd[f"{lp}mlp.fc1.weight"])
        layer.mlp.fc1.bias.data.copy_(sd[f"{lp}mlp.fc1.bias"])
        layer.mlp.fc2.weight.data.copy_(sd[f"{lp}mlp.fc2.weight"])
        layer.mlp.fc2.bias.data.copy_(sd[f"{lp}mlp.fc2.bias"])


def _load_backbone(model, sd, prefix="paligemma_with_expert.", use_conv=False):
    """加载 PaliGemma backbone 权重."""
    model.embed_tokens.weight.data.copy_(sd[f"{prefix}paligemma_language_model_embed_tokens.weight"])
    model.multi_modal_projector.weight.data.copy_(sd[f"{prefix}multi_modal_projector.weight"])
    model.multi_modal_projector.bias.data.copy_(sd[f"{prefix}multi_modal_projector.bias"])
    conv_suffix = "_conv" if use_conv else ""
    for i, layer in enumerate(model.backbone_layers):
        lp = f"{prefix}paligemma_language_model_layers.{i}."
        layer.input_layernorm.weight.data.copy_(sd[f"{lp}input_layernorm.weight"])
        layer.post_attention_layernorm.weight.data.copy_(sd[f"{lp}post_attention_layernorm.weight"])
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            src_key = f"{lp}self_attn.{proj}{conv_suffix}.weight"
            getattr(layer, f"{proj}{conv_suffix}").weight.data.copy_(sd[src_key])
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            src_key = f"{lp}mlp.{proj}{conv_suffix}.weight"
            getattr(layer, f"{proj}{conv_suffix}").weight.data.copy_(sd[src_key])


def _load_expert(model, sd, prefix="paligemma_with_expert.", use_conv=False):
    """加载 Expert decoder 权重."""
    conv_suffix = "_conv" if use_conv else ""
    for i, layer in enumerate(model.expert_layers):
        lp = f"{prefix}gemma_expert_model_layers.{i}."
        layer.input_layernorm.dense.weight.data.copy_(sd[f"{lp}input_layernorm.dense.weight"])
        layer.input_layernorm.dense.bias.data.copy_(sd[f"{lp}input_layernorm.dense.bias"])
        layer.post_attention_layernorm.dense.weight.data.copy_(sd[f"{lp}post_attention_layernorm.dense.weight"])
        layer.post_attention_layernorm.dense.bias.data.copy_(sd[f"{lp}post_attention_layernorm.dense.bias"])
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            src_key = f"{lp}self_attn.{proj}{conv_suffix}.weight"
            getattr(layer, f"{proj}{conv_suffix}").weight.data.copy_(sd[src_key])
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            src_key = f"{lp}mlp.{proj}{conv_suffix}.weight"
            getattr(layer, f"{proj}{conv_suffix}").weight.data.copy_(sd[src_key])
    # Expert final norm
    model.expert_final_norm.dense.weight.data.copy_(sd[f"{prefix}gemma_expert_model_norm.dense.weight"])
    model.expert_final_norm.dense.bias.data.copy_(sd[f"{prefix}gemma_expert_model_norm.dense.bias"])


def _load_action_heads(model, sd):
    """加载 action/time projections."""
    model.action_in_proj.weight.data.copy_(sd["action_in_proj.weight"])
    model.action_in_proj.bias.data.copy_(sd["action_in_proj.bias"])
    model.action_out_proj.weight.data.copy_(sd["action_out_proj.weight"])
    model.action_out_proj.bias.data.copy_(sd["action_out_proj.bias"])
    model.time_mlp_in.weight.data.copy_(sd["time_mlp_in.weight"])
    model.time_mlp_in.bias.data.copy_(sd["time_mlp_in.bias"])
    model.time_mlp_out.weight.data.copy_(sd["time_mlp_out.weight"])
    model.time_mlp_out.bias.data.copy_(sd["time_mlp_out.bias"])


def load_pi05_from_checkpoint(checkpoint_path, device="cpu"):
    """从 full_model.pt 或 full_model_noconv.pt 加载完整 Pi0.5 模型.
    
    自动检测 conv/noconv 格式:
    - full_model.pt: 含 *_conv.weight 键, 使用 Conv2d 投影 (与基准一致)
    - full_model_noconv.pt: 含 *.weight 键, 使用 Linear 投影
    
    Args:
        checkpoint_path: 权重文件路径
        device: 加载设备
    
    Returns:
        Pi05FullModel
    """
    import time as _time
    print(f"加载 checkpoint: {checkpoint_path}")
    t0 = _time.time()
    sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"  {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.1f}M params, "
          f"耗时 {_time.time() - t0:.1f}s")

    # Auto-detect conv format
    use_conv = any("_conv.weight" in k for k in sd.keys())
    print(f"  格式: {'Conv2d (full_model.pt)' if use_conv else 'Linear (full_model_noconv.pt)'}")

    model = Pi05FullModel()
    if use_conv:
        for layer in model.backbone_layers:
            layer.prepare_conv()
        for layer in model.expert_layers:
            layer.prepare_conv()

    t0 = _time.time()
    _load_siglip(model, sd)
    _load_backbone(model, sd, use_conv=use_conv)
    _load_expert(model, sd, use_conv=use_conv)
    _load_action_heads(model, sd)
    print(f"  权重加载完成, 耗时 {_time.time() - t0:.1f}s")

    model = model.to(device).eval()
    return model
