"""Step 2: Decomposed ONNX 导出 — 遵循 reflex-vla decomposed.py 模式.

导出 2 个 ONNX 子模型 (与 reflex-vla 的 Pi05DecomposedInference 兼容):
  1. vlm_prefix.onnx   — 图像+语言 → 36个KV cache + prefix_pad_masks
  2. expert_denoise.onnx — KV cache + prefix_pad_masks + noise → actions

I/O 命名与 reflex-vla 一致:
  prefix 输入: img_base, img_wrist_l, img_wrist_r, mask_base, mask_wrist_l, mask_wrist_r, lang_tokens, lang_masks
  prefix 输出: past_k_0..17, past_v_0..17, prefix_pad_masks
  expert 输入: past_k_0..17, past_v_0..17, prefix_pad_masks, noise
  expert 输出: actions

用法:
    conda activate openpi311
    python step2_export_onnx.py [--per-step]
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from pi05_full_model import (
    Pi05FullModel, load_pi05_from_checkpoint,
    sinusoidal_pos_embedding, make_att_2d_masks,
)

# ═══════════════════════════ Constants (reflex-vla compatible) ═══════════════════════════

PI05_PALIGEMMA_LAYERS = 18
PI05_KV_HEADS = 1
PI05_HEAD_DIM = 256
BATCH_SIZE = 1
IMAGE_SIZE = 224
LANG_TOKENS = 200
VISION_PATCHES_PER_VIEW = 256


def past_kv_names():
    """Generate KV cache tensor names — matches reflex-vla decomposed.py."""
    names = []
    for i in range(PI05_PALIGEMMA_LAYERS):
        names.append(f"past_k_{i}")
        names.append(f"past_v_{i}")
    return names


def prefix_seq_len():
    return 3 * VISION_PATCHES_PER_VIEW + LANG_TOKENS  # 968


# ═══════════════════════════ VLM Prefix Wrapper ═══════════════════════════

class VLMPrefixWrapper(nn.Module):
    """Wraps SigLIP + PaliGemma encoder.
    
    输入: 3 images [1,3,224,224] + 3 masks [1] + lang_tokens [1,200] + lang_masks [1,200]
    输出: 36 KV tensors (past_k_0..17, past_v_0..17, each [1,1,968,256]) + prefix_pad_masks [1,968]
    
    与 reflex-vla Pi05PrefixWrapper 的 I/O 签名完全一致.
    """
    def __init__(self, model: Pi05FullModel):
        super().__init__()
        self.vision_tower = model.vision_tower
        self.multi_modal_projector = model.multi_modal_projector
        self.embed_tokens = model.embed_tokens
        self.backbone_layers = model.backbone_layers
        self.vlm_hidden = model.vlm_hidden

    def forward(self, img_base, img_wrist_l, img_wrist_r,
                mask_base, mask_wrist_l, mask_wrist_r,
                lang_tokens, lang_masks):
        # 1. Embed images
        images = [img_base, img_wrist_l, img_wrist_r]
        img_masks = [mask_base, mask_wrist_l, mask_wrist_r]
        
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

        # 2. Embed language
        lang_emb = self.embed_tokens(lang_tokens) * math.sqrt(self.vlm_hidden)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks_list.extend([0] * lang_emb.shape[1])

        prefix_embs = torch.cat(embs, dim=1)  # [B, 968, 2048]
        prefix_pad_masks = torch.cat(pad_masks, dim=1)  # [B, 968]
        b = prefix_pad_masks.shape[0]
        att_masks = torch.tensor(att_masks_list, dtype=torch.bool, device=prefix_pad_masks.device)
        att_masks = att_masks[None, :].expand(b, -1)

        # 3. Compute attention mask
        att_2d = make_att_2d_masks(prefix_pad_masks, att_masks)
        att_4d = torch.where(att_2d[:, None, :, :], 0.0, -2.3819763e38).to(prefix_embs.dtype)
        position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1

        # 4. Forward through backbone, collecting KV cache
        kv_list = []  # flat: k_0, v_0, k_1, v_1, ...
        x = prefix_embs
        for layer in self.backbone_layers:
            x, cached_k, cached_v = layer.forward_with_cache(x, position_ids, att_4d)
            kv_list.append(cached_k)
            kv_list.append(cached_v)

        # Return: 36 KV tensors + prefix_pad_masks
        return tuple(kv_list) + (prefix_pad_masks,)


# ═══════════════════════════ Expert Denoise Wrapper (baked loop) ═══════════════════════════

class ExpertDenoiseWrapper(nn.Module):
    """Wraps Expert decoder with baked 10-step Euler flow-matching loop.
    
    输入: 36 KV tensors + prefix_pad_masks [1,968] + noise [1,50,32]
    输出: actions [1,50,32]
    
    与 reflex-vla Pi05ExpertWrapper (baked-loop) 兼容.
    """
    def __init__(self, model: Pi05FullModel, num_steps: int = 10):
        super().__init__()
        self.expert_layers = model.expert_layers
        self.expert_final_norm = model.expert_final_norm
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        self.time_mlp_in = model.time_mlp_in
        self.time_mlp_out = model.time_mlp_out
        self.expert_hidden = model.expert_hidden
        self.chunk_size = model.chunk_size
        self.num_steps = num_steps
        # Pre-compute scaling factor in float64 → store as float32 buffer (ONNX-friendly)
        dim = model.expert_hidden
        fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        scaling = (1.0 / period * 2 * math.pi).to(torch.float32)
        self.register_buffer("_scaling_factor", scaling)

    def forward(self, *args):
        """args: past_k_0, past_v_0, ..., past_k_17, past_v_17, prefix_pad_masks, noise"""
        n_kv = PI05_PALIGEMMA_LAYERS * 2  # 36
        past_flat = args[:n_kv]
        prefix_pad_masks = args[n_kv]
        noise = args[n_kv + 1]

        # Reconstruct KV cache as list of (k, v) tuples
        kv_cache = []
        for i in range(PI05_PALIGEMMA_LAYERS):
            kv_cache.append((past_flat[2 * i], past_flat[2 * i + 1]))

        b = noise.shape[0]
        device = noise.device

        # Pre-compute suffix attention structure
        suffix_len = self.chunk_size
        prefix_len = prefix_pad_masks.shape[1]
        suffix_pad = torch.ones(b, suffix_len, dtype=torch.bool, device=device)
        suffix_att = torch.zeros(b, suffix_len, dtype=torch.bool, device=device)
        suffix_att[:, 0] = True
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(b, suffix_len, prefix_len)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        full_att_4d = torch.where(full_att_2d[:, None, :, :], 0.0, -2.3819763e38).to(noise.dtype)

        prefix_offsets = torch.sum(prefix_pad_masks.long(), dim=-1)[:, None]
        suffix_pos_ids = prefix_offsets + torch.cumsum(suffix_pad.long(), dim=1) - 1

        # 10-step Euler flow-matching
        dt = -1.0 / self.num_steps
        x_t = noise
        for step in range(self.num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.full((b,), time_val, dtype=torch.float32, device=device)
            # ONNX-friendly time embedding: use pre-computed scaling factor (float32)
            sin_input = self._scaling_factor[None, :] * time_tensor[:, None]
            time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

            # Embed suffix
            suffix_embs = self.action_in_proj(x_t)
            adarms_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))

            # Expert decoder
            h = suffix_embs
            for i, layer in enumerate(self.expert_layers):
                pk, pv = kv_cache[i]
                h = layer(h, suffix_pos_ids, adarms_cond, pk, pv, full_att_4d)
            h, _ = self.expert_final_norm(h, adarms_cond)
            v_t = self.action_out_proj(h.float())

            x_t = x_t + dt * v_t

        return x_t


# ═══════════════════════════ Expert Per-Step Wrapper ═══════════════════════════

class ExpertPerStepWrapper(nn.Module):
    """Single-step expert wrapper (per-step mode).
    
    输入: 36 KV tensors + prefix_pad_masks + x_t [1,50,32] + t [1]
    输出: v_t [1,50,32] (velocity)
    
    Runtime 在 Python 中驱动 Euler loop (适合 per-step caching).
    """
    def __init__(self, model: Pi05FullModel):
        super().__init__()
        self.expert_layers = model.expert_layers
        self.expert_final_norm = model.expert_final_norm
        self.action_in_proj = model.action_in_proj
        self.action_out_proj = model.action_out_proj
        self.time_mlp_in = model.time_mlp_in
        self.time_mlp_out = model.time_mlp_out
        self.expert_hidden = model.expert_hidden
        self.chunk_size = model.chunk_size
        # Pre-compute scaling factor in float64 → store as float32 buffer (ONNX-friendly)
        dim = model.expert_hidden
        fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        scaling = (1.0 / period * 2 * math.pi).to(torch.float32)
        self.register_buffer("_scaling_factor", scaling)

    def forward(self, *args):
        """args: past_k_0..17, past_v_0..17, prefix_pad_masks, x_t, t"""
        n_kv = PI05_PALIGEMMA_LAYERS * 2
        past_flat = args[:n_kv]
        prefix_pad_masks = args[n_kv]
        x_t = args[n_kv + 1]
        t = args[n_kv + 2]

        kv_cache = []
        for i in range(PI05_PALIGEMMA_LAYERS):
            kv_cache.append((past_flat[2 * i], past_flat[2 * i + 1]))

        b = x_t.shape[0]
        device = x_t.device
        suffix_len = self.chunk_size
        prefix_len = prefix_pad_masks.shape[1]

        suffix_pad = torch.ones(b, suffix_len, dtype=torch.bool, device=device)
        suffix_att = torch.zeros(b, suffix_len, dtype=torch.bool, device=device)
        suffix_att[:, 0] = True
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(b, suffix_len, prefix_len)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        full_att_4d = torch.where(full_att_2d[:, None, :, :], 0.0, -2.3819763e38).to(x_t.dtype)

        prefix_offsets = torch.sum(prefix_pad_masks.long(), dim=-1)[:, None]
        suffix_pos_ids = prefix_offsets + torch.cumsum(suffix_pad.long(), dim=1) - 1

        time_emb_input = self._scaling_factor[None, :] * t[:, None]
        time_emb = torch.cat([torch.sin(time_emb_input), torch.cos(time_emb_input)], dim=1)
        suffix_embs = self.action_in_proj(x_t)
        adarms_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))

        h = suffix_embs
        for i, layer in enumerate(self.expert_layers):
            pk, pv = kv_cache[i]
            h = layer(h, suffix_pos_ids, adarms_cond, pk, pv, full_att_4d)
        h, _ = self.expert_final_norm(h, adarms_cond)
        return self.action_out_proj(h.float())


# ═══════════════════════════ Export Functions ═══════════════════════════

def export_vlm_prefix(model, out_dir, opset=17):
    """导出 vlm_prefix.onnx."""
    print("导出 vlm_prefix.onnx ...")
    wrapper = VLMPrefixWrapper(model).eval()
    B = BATCH_SIZE
    dummy = dict(
        img_base=torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE),
        img_wrist_l=torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE),
        img_wrist_r=torch.randn(B, 3, IMAGE_SIZE, IMAGE_SIZE),
        mask_base=torch.ones(B, dtype=torch.bool),
        mask_wrist_l=torch.ones(B, dtype=torch.bool),
        mask_wrist_r=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 257152, (B, LANG_TOKENS), dtype=torch.long),
        lang_masks=torch.ones(B, LANG_TOKENS, dtype=torch.bool),
    )
    output_names = past_kv_names() + ["prefix_pad_masks"]

    path = os.path.join(out_dir, "vlm_prefix.onnx")
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        tuple(dummy.values()),
        path,
        opset_version=opset,
        input_names=list(dummy.keys()),
        output_names=output_names,
    )
    print(f"  → {path} ({time.time() - t0:.1f}s)")
    return path


def export_expert_denoise(model, out_dir, num_steps=10, per_step=False, opset=17):
    """导出 expert_denoise.onnx."""
    mode_str = "per-step" if per_step else f"baked num_steps={num_steps}"
    print(f"导出 expert_denoise.onnx ({mode_str}) ...")

    B = BATCH_SIZE
    seq_len = prefix_seq_len()
    chunk = model.chunk_size
    action_dim = model.action_dim

    kv_dummies = {}
    kv_names = past_kv_names()
    for name in kv_names:
        kv_dummies[name] = torch.randn(B, PI05_KV_HEADS, seq_len, PI05_HEAD_DIM)
    
    expert_dummy = dict(kv_dummies)
    expert_dummy["prefix_pad_masks"] = torch.ones(B, seq_len, dtype=torch.bool)

    if per_step:
        wrapper = ExpertPerStepWrapper(model).eval()
        expert_dummy["x_t"] = torch.randn(B, chunk, action_dim)
        expert_dummy["t"] = torch.full((B,), 1.0, dtype=torch.float32)
        output_names = ["v_t"]
    else:
        wrapper = ExpertDenoiseWrapper(model, num_steps).eval()
        expert_dummy["noise"] = torch.randn(B, chunk, action_dim)
        output_names = ["actions"]

    path = os.path.join(out_dir, "expert_denoise.onnx")
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        tuple(expert_dummy.values()),
        path,
        opset_version=opset,
        input_names=list(expert_dummy.keys()),
        output_names=output_names,
    )
    print(f"  → {path} ({time.time() - t0:.1f}s)")
    return path


def write_reflex_config(out_dir, model, num_steps=10, per_step=False, target="orin"):
    """写入 reflex_config.json (与 Pi05DecomposedInference 兼容)."""
    cfg = {
        "model_type": "pi05_decomposed",
        "target": target,
        "num_denoising_steps": num_steps,
        "chunk_size": model.chunk_size,
        "action_chunk_size": model.chunk_size,
        "action_dim": model.action_dim,
        "opset": 17,
        "export_kind": "decomposed",
        "decomposed": {
            "vlm_prefix_onnx": "vlm_prefix.onnx",
            "expert_denoise_onnx": "expert_denoise.onnx",
            "paligemma_layers": PI05_PALIGEMMA_LAYERS,
            "kv_heads": PI05_KV_HEADS,
            "head_dim": PI05_HEAD_DIM,
            "past_kv_tensor_names": past_kv_names(),
            "per_step_expert": per_step,
        },
    }
    path = os.path.join(out_dir, "reflex_config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  → {path}")


def verify_onnx(out_dir):
    """验证 ONNX 模型可加载."""
    import onnxruntime as ort
    for name in ["vlm_prefix.onnx", "expert_denoise.onnx"]:
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            print(f"  ✗ {name} 不存在")
            continue
        try:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            ins = len(sess.get_inputs())
            outs = len(sess.get_outputs())
            print(f"  ✓ {name}: {ins} inputs, {outs} outputs")
            for inp in sess.get_inputs():
                print(f"      in:  {inp.name} {inp.shape} {inp.type}")
            for out in sess.get_outputs():
                print(f"      out: {out.name} {out.shape} {out.type}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")


# ═══════════════════════════ Main ═══════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pi0.5 decomposed ONNX 导出 (reflex-vla 兼容)")
    parser.add_argument("--checkpoint", type=str,
                        default="/data1/pengtao/robot/reflex-vla/checkpoint/30000_fold_shirt/torch/full_model_noconv.pt")
    parser.add_argument("--out-dir", type=str, default="/data1/pengtao/robot/reflex-vla/orin_pi05_server/onnx_models")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--per-step", action="store_true", default=False,
                        help="Per-step export (Python-driven Euler loop)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--target", type=str, default="orin")
    parser.add_argument("--verify", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model = load_pi05_from_checkpoint(args.checkpoint, device="cpu")

    print("\n" + "=" * 60)
    print(f" Pi0.5 Decomposed ONNX Export (reflex-vla 兼容)")
    print(f" 模式: {'per-step' if args.per_step else f'baked {args.num_steps}-step'}")
    print("=" * 60)

    t0 = time.time()
    export_vlm_prefix(model, args.out_dir, args.opset)
    export_expert_denoise(model, args.out_dir, args.num_steps, args.per_step, args.opset)
    write_reflex_config(model=model, out_dir=args.out_dir, num_steps=args.num_steps,
                        per_step=args.per_step, target=args.target)
    elapsed = time.time() - t0
    print(f"\n导出完成, 耗时 {elapsed:.1f}s")

    if args.verify:
        print("\n验证 ONNX 模型:")
        verify_onnx(args.out_dir)


if __name__ == "__main__":
    main()
