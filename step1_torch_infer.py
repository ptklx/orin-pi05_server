"""Step 1: PyTorch 完整推理验证.

用 test_openpi_example.pkl 验证模型加载和推理正确性.
输入: 3张图片 + 关节状态 + 任务文本 → 输出: actions [1, 50, 32]

用法:
    conda activate openpi311
    python step1_torch_infer.py
"""

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pi05_full_model import load_pi05_from_checkpoint, sinusoidal_pos_embedding


# ═══════════════════════════ 数据预处理 ═══════════════════════════

def resize_with_pad(image_np, height=224, width=224):
    """Resize image with padding (与 openpi resize_with_pad 一致)."""
    img = Image.fromarray(image_np)
    cur_w, cur_h = img.size
    if cur_w == width and cur_h == height:
        return np.array(img)
    ratio = max(cur_w / width, cur_h / height)
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    result = Image.new(resized.mode, (width, height), 0)
    pad_h = max(0, (height - new_h) // 2)
    pad_w = max(0, (width - new_w) // 2)
    result.paste(resized, (pad_w, pad_h))
    return np.array(result)


def preprocess_image(image_np, height=224, width=224):
    """图片预处理: resize → normalize to [-1, 1] → [B, C, H, W]."""
    if image_np.shape[0] == 3:  # CHW → HWC
        image_np = np.transpose(image_np, (1, 2, 0))
    image_np = resize_with_pad(image_np, height, width)
    img = image_np / 255.0 * 2.0 - 1.0  # float64 intermediate (match reference)
    img = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 224, 224]
    return img


class NormStats:
    """归一化统计量."""
    def __init__(self, stats_dict):
        self.mean = np.array(stats_dict["mean"], dtype=np.float32)
        self.std = np.array(stats_dict["std"], dtype=np.float32)
        self.q01 = np.array(stats_dict["q01"], dtype=np.float32) if "q01" in stats_dict else None
        self.q99 = np.array(stats_dict["q99"], dtype=np.float32) if "q99" in stats_dict else None


def normalize_state(x, stats):
    """与基准一致: mean/std 归一化 (use_quantiles=False)."""
    return (x - stats.mean) / (stats.std + 1e-6)


def unnormalize_actions(actions, stats):
    """与基准一致: mean/std 反归一化 (use_quantiles=False)."""
    mean = torch.from_numpy(stats.mean).to(actions.device)
    std = torch.from_numpy(stats.std).to(actions.device)
    return actions * (std + 1e-6) + mean


def tensor_sample_text(tensor, max_items=8):
    """与基准一致的输出格式."""
    flat = tensor.detach().cpu().reshape(-1)
    sliced = flat[:max_items].tolist()
    return ", ".join(f"{v:.6f}" for v in sliced)


def prepare_state_prompt(state, task_text, tokenizer, max_len=200):
    """Pi0.5: state 离散化嵌入到 prompt 中.
    
    state → quantize to 256 bins → "Task: xxx, State: 128 42 ...;\\nAction: "
    → tokenize with PaliGemma tokenizer → pad to max_len
    """
    # Discretize state (match reference: no clip, allow negative values)
    bins = np.linspace(-1, 1, 257)[:-1]
    discretized = np.digitize(state, bins) - 1
    state_str = " ".join(map(str, discretized))

    cleaned_text = task_text.strip().replace("_", " ").replace("\n", " ")
    full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "

    tokenized = tokenizer(
        full_prompt,
        padding="max_length",
        padding_side="right",
        max_length=max_len,
        return_tensors="pt",
    )
    return tokenized["input_ids"], tokenized["attention_mask"].bool()


def load_pkl_example(path):
    """加载 test_openpi_example.pkl."""
    with open(path, "rb") as f:
        return pickle.load(f)


def pad_vector(v, new_dim):
    """Pad vector to new_dim."""
    if len(v) >= new_dim:
        return v[:new_dim]
    padded = np.zeros(new_dim, dtype=v.dtype)
    padded[:len(v)] = v
    return padded


# ═══════════════════════════ Main ═══════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pi0.5 PyTorch 推理验证")
    parser.add_argument("--checkpoint", type=str,
                        default="/data1/pengtao/robot/reflex-vla/checkpoint/30000_fold_shirt/torch/full_model.pt")
    parser.add_argument("--norm-stats", type=str,
                        default="/data1/pengtao/robot/reflex-vla/checkpoint/30000_fold_shirt/assets/agilex/norm_stats.json")
    parser.add_argument("--tokenizer", type=str,
                        default="/data1/pengtao/robot/openpi/checkpoint/paligemma-3b-pt-224")
    parser.add_argument("--pkl", type=str,
                        default="/data1/pengtao/robot/reflex-vla/orin_deploy/test_openpi_example.pkl")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--noise-path", type=str, default="/data1/pengtao/robot/reflex-vla/orin_deploy/1x50x32_random_numbers.npy",
                        help="固定 noise .npy 路径 (与基准对齐)")
    args = parser.parse_args()

    device = args.device

    # 1. 加载模型
    print("=" * 60)
    print(" Pi0.5 PyTorch 完整推理验证")
    print("=" * 60)
    model = load_pi05_from_checkpoint(args.checkpoint, device=device)

    # 2. 加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer 加载完成: {args.tokenizer}")

    # 3. 加载归一化统计
    with open(args.norm_stats) as f:
        raw_stats = json.load(f)["norm_stats"]
    state_stats = NormStats(raw_stats["state"])
    action_stats = NormStats(raw_stats["actions"])
    print(f"归一化统计加载完成")

    # 4. 加载测试数据
    example = load_pkl_example(args.pkl)
    print(f"\n测试数据:")
    for k, v in example.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    # 5. 预处理
    # 图片
    front = example.get("observation_image_front", example.get("observation_front_image"))
    left = example.get("observation_image_left", example.get("observation_left_wrist_image"))
    right = example.get("observation_image_right", example.get("observation_right_wrist_image"))

    images = [
        preprocess_image(front).to(device),
        preprocess_image(left).to(device),
        preprocess_image(right).to(device),
    ]
    img_masks = [torch.ones(1, dtype=torch.bool, device=device) for _ in range(3)]

    # State: 合并关节 → pad → mean/std normalize (与基准一致)
    left_joint = example.get("observation_joint_position_left", np.zeros(7))
    right_joint = example.get("observation_joint_position_right", np.zeros(7))
    state = np.concatenate([left_joint, right_joint]).astype(np.float32)
    state = pad_vector(state, args.action_dim)
    state_normalized = normalize_state(state, state_stats)

    # Task
    task_text = example["task"]
    if isinstance(task_text, list):
        task_text = task_text[0]

    # Tokenize
    lang_tokens, lang_masks = prepare_state_prompt(state_normalized, task_text, tokenizer)
    lang_tokens = lang_tokens.to(device)
    lang_masks = lang_masks.to(device)

    print(f"\n预处理完成:")
    print(f"  Images: 3 × {images[0].shape}")
    print(f"  State (raw): {state[:14]}")
    print(f"  State (normalized): {state_normalized[:14]}")
    print(f"  Prompt: {task_text}")
    print(f"  Lang tokens: {lang_tokens.shape}")

    # 6. 加载 noise (优先使用固定 noise 与基准对齐)
    if args.noise_path and os.path.exists(args.noise_path):
        noise = torch.from_numpy(np.load(args.noise_path)).to(torch.float32).to(device)
        print(f"  使用固定 noise: {args.noise_path}")
    else:
        noise = torch.randn(1, 50, args.action_dim, dtype=torch.float32, device=device)
        print(f"  使用随机 noise")

    # 7. 推理
    print(f"\n开始推理...")
    with torch.no_grad():
        t0 = time.perf_counter()
        actions = model(images, img_masks, lang_tokens, lang_masks, noise=noise)
        elapsed = (time.perf_counter() - t0) * 1000

    # 输出格式与基准 onnx_infer_demo.py 一致
    actions = actions[:, :, :args.action_dim]
    actions_unnorm = unnormalize_actions(actions, action_stats)
    print(f"total_inference: {elapsed:.3f} ms")
    print(f"action_shape: {tuple(actions.shape)}")
    print(f"action_sample: [{tensor_sample_text(actions_unnorm[0, 0, :8])}]")
    print(f"actions unnorm: {actions_unnorm}")


if __name__ == "__main__":
    main()


'''
tensor([[[ 2.3801e-01,  1.4231e+00, -6.7347e-01,  ...,  4.5169e-07,
           3.4747e-07,  7.5611e-07],
         [-2.3391e-01, -3.5207e-01,  1.1499e+00,  ..., -7.2717e-07,
          -9.2781e-07,  5.5572e-07],
         [ 3.5762e-01,  5.9089e-02, -1.0223e+00,  ...,  1.5331e-06,
          -5.7409e-08, -2.7995e-06],
         ...,
         [ 1.0025e-01,  9.2038e-01, -1.3875e+00,  ...,  5.4799e-07,
           1.2798e-06, -1.0427e-06],
         [ 2.0042e-01, -2.5295e-01,  1.1176e+00,  ...,  1.1817e-06,
          -7.9044e-07, -6.5318e-07],
         [ 7.4492e-01,  6.2988e-01,  1.0835e-01,  ...,  1.4484e-06,
          -1.5284e-06,  8.6491e-07]]])
'''