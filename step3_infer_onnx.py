"""Step 3: Decomposed ONNX 推理 — 遵循 reflex-vla Pi05DecomposedInference 模式.

加载 vlm_prefix.onnx + expert_denoise.onnx, 用测试数据运行端到端推理,
输出 actions 和分阶段耗时.

用法:
    conda activate openpi311
    python step3_infer_onnx.py [--onnx-dir ./onnx_models]
"""

import argparse
import json
import os
import pickle
import time
from collections import OrderedDict
import einops
import numpy as np
import onnxruntime as ort


# ═══════════════════════════ 数据预处理 (从 step1 复制, 避免跨文件依赖) ═══════════════════════════

def resize_with_pad(image_np, height=224, width=224):
    from PIL import Image
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
    if image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))
    image_np = resize_with_pad(image_np, height, width)
    img = image_np / 255.0 * 2.0 - 1.0  # float64 intermediate (match reference)
    return img.astype(np.float32).transpose(2, 0, 1)[np.newaxis]  # [1, 3, 224, 224]


def normalize_state(x, mean, std):
    """与基准一致: mean/std 归一化 (use_quantiles=False)."""
    return (x - mean) / (std + 1e-6)


def unnormalize_actions(actions, mean, std):
    """与基准一致: mean/std 反归一化 (use_quantiles=False)."""
    return actions * (std + 1e-6) + mean


def tensor_sample_text(array, max_items=8):
    """与基准一致的输出格式."""
    flat = array.reshape(-1)
    sliced = flat[:max_items].tolist()
    return ", ".join(f"{v:.6f}" for v in sliced)


def pad_vector(v, new_dim):
    if len(v) >= new_dim:
        return v[:new_dim]
    padded = np.zeros(new_dim, dtype=v.dtype)
    padded[:len(v)] = v
    return padded


def prepare_state_prompt(state, task_text, tokenizer, max_len=200):
    bins = np.linspace(-1, 1, 257)[:-1]
    discretized = np.digitize(state, bins) - 1
    state_str = " ".join(map(str, discretized))
    cleaned_text = task_text.strip().replace("_", " ").replace("\n", " ")
    full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
    tokenized = tokenizer(
        full_prompt, padding="max_length", padding_side="right",
        max_length=max_len, return_tensors="np",
    )
    return tokenized["input_ids"].astype(np.int64), tokenized["attention_mask"].astype(bool)


# ═══════════════════════════ StageTimer (与基准一致) ═══════════════════════════

class StageTimer:
    def __init__(self):
        self._stats = OrderedDict()

    def add(self, stage_name, elapsed_ms):
        item = self._stats.setdefault(stage_name, {"total_ms": 0.0, "count": 0})
        item["total_ms"] += elapsed_ms
        item["count"] += 1

    def print_summary(self, title="module_timings"):
        print(title)
        for stage_name, item in self._stats.items():
            avg_ms = item["total_ms"] / item["count"]
            print(f"  {stage_name}: total={item['total_ms']:.3f} ms, count={item['count']}, avg={avg_ms:.3f} ms")


# ═══════════════════════════ ONNX Runtime Wrapper ═══════════════════════════

class DecomposedONNXRunner:
    """2-file decomposed ONNX 推理 (reflex-vla Pi05DecomposedInference 模式)."""

    def __init__(self, onnx_dir, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        config_path = os.path.join(onnx_dir, "reflex_config.json")
        with open(config_path) as f:
            self.cfg = json.load(f)

        dcfg = self.cfg["decomposed"]
        self.paligemma_layers = dcfg["paligemma_layers"]
        self.kv_heads = dcfg["kv_heads"]
        self.head_dim = dcfg["head_dim"]
        self.kv_names = dcfg["past_kv_tensor_names"]
        self.per_step = dcfg.get("per_step_expert", False)

        # Load ONNX sessions
        vlm_path = os.path.join(onnx_dir, dcfg["vlm_prefix_onnx"])
        expert_path = os.path.join(onnx_dir, dcfg["expert_denoise_onnx"])

        print(f"加载 VLM prefix: {vlm_path}")
        t0 = time.perf_counter()
        self.vlm_sess = ort.InferenceSession(vlm_path, providers=providers)
        print(f"  耗时: {(time.perf_counter() - t0) * 1000:.0f} ms")

        print(f"加载 Expert denoise: {expert_path}")
        t0 = time.perf_counter()
        self.expert_sess = ort.InferenceSession(expert_path, providers=providers)
        print(f"  耗时: {(time.perf_counter() - t0) * 1000:.0f} ms")

        # Cache input/output names
        self.vlm_input_names = [i.name for i in self.vlm_sess.get_inputs()]
        self.vlm_output_names = [o.name for o in self.vlm_sess.get_outputs()]
        self.expert_input_names = [i.name for i in self.expert_sess.get_inputs()]
        self.expert_output_names = [o.name for o in self.expert_sess.get_outputs()]

    def run_vlm_prefix(self, img_base, img_wrist_l, img_wrist_r,
                       mask_base, mask_wrist_l, mask_wrist_r,
                       lang_tokens, lang_masks):
        """运行 VLM prefix: images + lang → KV cache + prefix_pad_masks."""
        feed = {
            "img_base": img_base,
            "img_wrist_l": img_wrist_l,
            "img_wrist_r": img_wrist_r,
            "mask_base": mask_base,
            "mask_wrist_l": mask_wrist_l,
            "mask_wrist_r": mask_wrist_r,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }
        t0 = time.perf_counter()
        outputs = self.vlm_sess.run(self.vlm_output_names, feed)
        elapsed = (time.perf_counter() - t0) * 1000

        # Parse outputs: 36 KV tensors + prefix_pad_masks
        kv_dict = {}
        for i, name in enumerate(self.vlm_output_names[:-1]):
            kv_dict[name] = outputs[i]
        prefix_pad_masks = outputs[-1]
        return kv_dict, prefix_pad_masks, elapsed

    def run_expert_denoise(self, kv_dict, prefix_pad_masks, noise):
        """运行 Expert denoise: KV cache + noise → actions."""
        feed = dict(kv_dict)
        feed["prefix_pad_masks"] = prefix_pad_masks
        if self.per_step:
            return self._run_expert_per_step(feed, noise)
        else:
            feed["noise"] = noise
            t0 = time.perf_counter()
            outputs = self.expert_sess.run(self.expert_output_names, feed)
            elapsed = (time.perf_counter() - t0) * 1000
            return outputs[0], elapsed

    def _run_expert_per_step(self, feed_base, noise):
        """Per-step Euler loop (Python-driven)."""
        num_steps = self.cfg["num_denoising_steps"]
        dt = -1.0 / num_steps
        x_t = noise.copy()
        total_ms = 0
        for step in range(num_steps):
            t_val = 1.0 + step * dt
            feed = dict(feed_base)
            feed["x_t"] = x_t
            feed["t"] = np.full((1,), t_val, dtype=np.float32)
            t0 = time.perf_counter()
            outputs = self.expert_sess.run(self.expert_output_names, feed)
            total_ms += (time.perf_counter() - t0) * 1000
            v_t = outputs[0]
            x_t = x_t + dt * v_t
        return x_t, total_ms

    def infer(self, img_base, img_wrist_l, img_wrist_r,
              mask_base, mask_wrist_l, mask_wrist_r,
              lang_tokens, lang_masks, noise):
        """端到端推理: images + lang + noise → actions."""
        kv_dict, prefix_pad_masks, vlm_ms = self.run_vlm_prefix(
            img_base, img_wrist_l, img_wrist_r,
            mask_base, mask_wrist_l, mask_wrist_r,
            lang_tokens, lang_masks,
        )
        actions, expert_ms = self.run_expert_denoise(kv_dict, prefix_pad_masks, noise)
        return actions, vlm_ms, expert_ms




image_height = 224
image_width = 224
action_dim=32
OBS_ROBOT= 'observation.state'
############load data 

def _parse_image(image) -> np.ndarray:    
    image = np.asarray(image)    
    if np.issubdtype(image.dtype, np.floating):        
        image = (255 * image).astype(np.uint8)    
    if image.shape[0] == 3:        
        image = einops.rearrange(image, "c h w -> h w c")    
    return image

def taihu_data_handle_pt( data: dict) -> dict:
    state = np.concatenate([data["observation_joint_position_left"], data["observation_joint_position_right"]])
    front_image = (
    data["observation_front_image"] if "observation_front_image" in data else
    data["observation_image_front"] if "observation_image_front" in data else
    np.zeros((224, 224, 3), dtype=np.uint8)
    )
    front_image = _parse_image(front_image)
    base_image_left=(data["observation_left_wrist_image"] if "observation_left_wrist_image" in data else
    data["observation_image_left"] if "observation_image_left" in data else
    np.zeros((224, 224, 3), dtype=np.uint8)  )
    base_image_left = _parse_image(base_image_left)
    base_image_right=(data["observation_right_wrist_image"] if "observation_right_wrist_image" in data else
    data["observation_image_right"] if "observation_image_right" in data else
    np.zeros((224, 224, 3), dtype=np.uint8)  )
    base_image_right = _parse_image(base_image_right)

    names = ("observation.images.front", "observation.images.left_wrist", "observation.images.right_wrist")
    # images = (base_image, wrist_image, np.zeros_like(base_image))
    images = (front_image, base_image_left, base_image_right)

    inputs = dict(zip(names, images, strict=True))
    inputs[OBS_ROBOT] = state
    
    # 处理actions数据，保持批次维度
    if "actions" in data:
        actions = data["actions"]
        if actions.shape[1] < action_dim:
            padded_actions = np.zeros((actions.shape[0], action_dim))
            padded_actions[:, :actions.shape[1]] = actions
            inputs["actions"] = padded_actions  
            # print(f"Actions padded to shape: {padded_actions.shape}")
        else:
            inputs["actions"] = actions

    if "task" in data:
        inputs["task"] = data["task"]
    if "prompt" in data:
        inputs["task"] = data["prompt"]

    return inputs


from dataclasses import dataclass as _dataclass
from transformers import AutoTokenizer

@_dataclass
class pi05Config:

    policy_type: str = "pi05_onnx"

class pi05_create_infer:
    def __init__(self):
        self.config = pi05Config()
        path_file = os.path.dirname(os.path.abspath(__file__))
        onnx_dir = os.path.join(path_file, "onnx_models")
        tokenizer_path = os.path.join(path_file, "data_param/paligemma-3b-pt-224")
        norm_stats_path = os.path.join(path_file, "data_param/norm_stats.json")

        self.runner =  DecomposedONNXRunner(onnx_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        with open(norm_stats_path) as f:
            raw_stats = json.load(f)["norm_stats"]
        self.state_mean = np.array(raw_stats["state"]["mean"], dtype=np.float32)
        self.state_std = np.array(raw_stats["state"]["std"], dtype=np.float32)
        self.action_mean = np.array(raw_stats["actions"]["mean"], dtype=np.float32)
        self.action_std = np.array(raw_stats["actions"]["std"], dtype=np.float32)

    def forward(self,batch,noise):
        state = pad_vector(batch.get("observation.state"), action_dim)
        state_normalized = normalize_state(state, self.state_mean, self.state_std)
        front = batch.get("observation.images.front")
        left = batch.get("observation.images.left_wrist")
        right = batch.get("observation.images.right_wrist")
        img_base = preprocess_image(front).astype(np.float32)
        img_wrist_l = preprocess_image(left).astype(np.float32)
        img_wrist_r = preprocess_image(right).astype(np.float32)
        task_text = batch["task"]
        if isinstance(task_text, list):
            task_text = task_text[0]

        lang_tokens, lang_masks = prepare_state_prompt(state_normalized, task_text, self.tokenizer)
        mask_base = np.array([True])
        mask_wrist_l = np.array([True])
        mask_wrist_r = np.array([True])

        timer = StageTimer()
        vlm_times, expert_times, total_times = [], [], []
        t0 = time.perf_counter()
        actions, vlm_ms, expert_ms = self.runner.infer(img_base, img_wrist_l, img_wrist_r,
                     mask_base, mask_wrist_l, mask_wrist_r,
                     lang_tokens, lang_masks,  noise)
        total_ms = (time.perf_counter() - t0) * 1000
        vlm_times.append(vlm_ms)
        expert_times.append(expert_ms)
        total_times.append(total_ms)
        timer.add("trt.vlm_prefix", vlm_ms)
        timer.add("trt.expert_denoise", expert_ms)
        print(f"  Run : VLM={vlm_ms:.1f}ms  Expert={expert_ms:.1f}ms  Total={total_ms:.1f}ms")
        actions = actions[:, :, :action_dim]
        actions_unnorm = unnormalize_actions(actions, self.action_mean, self.action_std)

        return  actions_unnorm

    def select_action(self,inputs,noise):
        inputs=taihu_data_handle_pt(inputs)
        action= self.forward(inputs,noise= noise)
        outputs = {
            'actions': action[0]
        }
        return outputs




# ═══════════════════════════ Main ═══════════════════════════

def main():
    path_file = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Pi0.5 decomposed ONNX 推理 (reflex-vla 兼容)")
    parser.add_argument("--onnx-dir", type=str, default=os.path.join(path_file, "onnx_models"))
    parser.add_argument("--norm-stats", type=str,
                        default=os.path.join(path_file, "data_param/norm_stats.json"))
    parser.add_argument("--tokenizer", type=str,
                        default=os.path.join(path_file, "data_param/paligemma-3b-pt-224"))
    parser.add_argument("--pkl", type=str,
                        default=os.path.join(path_file, "data_param/test_openpi_example.pkl"))
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--noise-path", type=str, default=os.path.join(path_file, "data_param/1x50x32_random_numbers.npy"),
                        help="固定 noise .npy 路径 (与基准对齐)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing")
    parser.add_argument("--repeat", type=int, default=1, help="Timing runs")
    args = parser.parse_args()

    print("=" * 60)
    print(" Pi0.5 Decomposed ONNX 推理 (reflex-vla 兼容)")
    print("=" * 60)

    # 1. Load ONNX runner
    runner = DecomposedONNXRunner(args.onnx_dir)

    # 2. Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer}")

    # 3. Load norm stats
    with open(args.norm_stats) as f:
        raw_stats = json.load(f)["norm_stats"]
    state_mean = np.array(raw_stats["state"]["mean"], dtype=np.float32)
    state_std = np.array(raw_stats["state"]["std"], dtype=np.float32)
    action_mean = np.array(raw_stats["actions"]["mean"], dtype=np.float32)
    action_std = np.array(raw_stats["actions"]["std"], dtype=np.float32)

    # 4. Load test data
    with open(args.pkl, "rb") as f:
        example = pickle.load(f)

    front = example.get("observation_image_front", example.get("observation_front_image"))
    left = example.get("observation_image_left", example.get("observation_left_wrist_image"))
    right = example.get("observation_image_right", example.get("observation_right_wrist_image"))

    img_base = preprocess_image(front).astype(np.float32)
    img_wrist_l = preprocess_image(left).astype(np.float32)
    img_wrist_r = preprocess_image(right).astype(np.float32)

    mask_base = np.array([True])
    mask_wrist_l = np.array([True])
    mask_wrist_r = np.array([True])

    # State
    left_joint = example.get("observation_joint_position_left", np.zeros(7))
    right_joint = example.get("observation_joint_position_right", np.zeros(7))
    state = np.concatenate([left_joint, right_joint]).astype(np.float32)
    state = pad_vector(state, args.action_dim)
    state_normalized = normalize_state(state, state_mean, state_std)

    task_text = example["task"]
    if isinstance(task_text, list):
        task_text = task_text[0]

    lang_tokens, lang_masks = prepare_state_prompt(state_normalized, task_text, tokenizer)

    # Fixed noise (优先使用固定 noise 与基准对齐)
    if args.noise_path and os.path.exists(args.noise_path):
        noise = np.load(args.noise_path).astype(np.float32)
        print(f"  使用固定 noise: {args.noise_path}")
    else:
        rng = np.random.RandomState(42)
        noise = rng.randn(1, 50, args.action_dim).astype(np.float32)
        print(f"  使用随机 noise (seed=42)")

    print(f"\n输入:")
    print(f"  Images: 3 × {img_base.shape}")
    print(f"  Lang tokens: {lang_tokens.shape}")
    print(f"  Noise: {noise.shape}")
    print(f"  Task: {task_text}")

    # 5. Warmup
    print(f"\nWarmup ({args.warmup} runs)...")
    for _ in range(args.warmup):
        runner.infer(img_base, img_wrist_l, img_wrist_r,
                     mask_base, mask_wrist_l, mask_wrist_r,
                     lang_tokens, lang_masks, noise)

    # 6. Timed runs
    print(f"\nTiming ({args.repeat} runs)...")
    timer = StageTimer()
    vlm_times = []
    expert_times = []
    total_times = []
    for i in range(args.repeat):
        t0 = time.perf_counter()
        actions, vlm_ms, expert_ms = runner.infer(
            img_base, img_wrist_l, img_wrist_r,
            mask_base, mask_wrist_l, mask_wrist_r,
            lang_tokens, lang_masks, noise,
        )
        total_ms = (time.perf_counter() - t0) * 1000
        vlm_times.append(vlm_ms)
        expert_times.append(expert_ms)
        total_times.append(total_ms)
        timer.add("onnx.vlm_prefix", vlm_ms)
        timer.add("onnx.expert_denoise", expert_ms)
        print(f"  Run {i+1}: VLM={vlm_ms:.1f}ms  Expert={expert_ms:.1f}ms  Total={total_ms:.1f}ms")

    # 7. Summary
    print(f"\n{'─' * 50}")
    print(f" 耗时统计 (avg over {args.repeat} runs):")
    print(f"   VLM prefix:     {np.mean(vlm_times):.1f} ms")
    print(f"   Expert denoise: {np.mean(expert_times):.1f} ms")
    print(f"   Total e2e:      {np.mean(total_times):.1f} ms")
    print(f"{'─' * 50}")

    # 8. Output (格式与基准 onnx_infer_demo.py 一致)
    actions = actions[:, :, :args.action_dim]
    actions_unnorm = unnormalize_actions(actions, action_mean, action_std)
    print(f"total_inference: {np.mean(total_times):.3f} ms")
    print(f"action_shape: {tuple(actions.shape)}")
    # print(f"action_sample: [{tensor_sample_text(actions_unnorm[0, 0, :8])}]")
    print(f"actions_unnorm: {actions_unnorm}")
    timer.print_summary("module_timings")





def main1():
    pi05_prefer = pi05_create_infer()
    path_file = os.path.dirname(os.path.abspath(__file__))
    npz_file_path = os.path.join(path_file, "data_param/1x50x32_random_numbers.npy")
    noise = np.load(npz_file_path).astype(np.float32)
  
    pkl_file_path = os.path.join(path_file, "data_param/test_openpi_example.pkl")
    with open(pkl_file_path, "rb") as f:
        example = pickle.load(f)

    start_time = time.time()
    output = pi05_prefer.select_action(example,noise)
    end_time = time.time()
    elapsed_time = (end_time - start_time)*1000  # 计算耗时（秒）
    print(f"推理耗时: {elapsed_time:.3f} ms")
    print("Action shape:", output["actions"].shape)
    print("Action values:", output)

if __name__ == "__main__":
    # main()
    main1()
'''
 [[[ 2.7954746e-03  2.1799803e-02 -1.0964930e-02 ...  1.5130117e-09
   -1.1346080e-09 -4.4485859e-09]
  [ 4.2071287e-03  2.5050640e-02 -1.2509048e-02 ...  3.7311612e-09
    3.8830192e-10 -2.6684785e-09]
  [ 3.5559237e-03  2.3899317e-02 -8.6373687e-03 ...  1.6073362e-09
   -7.9631807e-10  9.8165864e-10]
  ...
  [ 4.4989977e-03  9.3474388e-03 -1.5643239e-03 ...  3.8242245e-09
   -6.7200512e-10 -1.5134514e-09]
  [ 4.6507251e-03  9.7117424e-03 -3.1303763e-03 ...  1.6755610e-09
    9.3492492e-10 -3.0207261e-09]
  [ 3.9627403e-03  9.9452734e-03 -2.9978156e-03 ...  2.1906048e-09
    2.0916016e-09 -4.1741059e-09]]]


'''