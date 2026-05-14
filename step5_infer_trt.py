"""Step 5: TensorRT 引擎推理 — 与 step3 ONNX 推理格式一致.

加载 vlm_prefix.engine + expert_denoise.engine, 运行端到端推理,
输出 actions 和分阶段耗时.

用法:
    conda activate pi05_trt
    cd /home/agi/pengtao/robot_pi05
    python orin_deploy/step5_infer_trt.py [--trt-dir orin_deploy/trt_engines_fp32]

注意: 必须使用 fp32 engines. fp16 因 additive attention mask (-2.38e38)
      超出 fp16 范围导致 NaN 传播, KV cache 全层错误.
"""

import argparse
import einops
import json
import os
import pickle
import time
from collections import OrderedDict

import einops
import numpy as np

try:
    import tensorrt as trt
    import torch
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("警告: tensorrt 或 torch 不可用, 将使用 trtexec 基准测试模式")


# ═══════════════════════════ 数据预处理 (与 step3 一致) ═══════════════════════════

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
    img = image_np.astype(np.float32) / 255.0 * 2.0 - 1.0
    return img.transpose(2, 0, 1)[np.newaxis]


def normalize_state(x, mean, std):
    return (x - mean) / (std + 1e-6)


def unnormalize_actions(actions, mean, std):
    return actions * (std + 1e-6) + mean


def tensor_sample_text(array, max_items=8):
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
    tokenizer.padding_side = "right"
    tokenized = tokenizer(
        full_prompt, padding="max_length",
        max_length=max_len, return_tensors="np",
    )
    return tokenized["input_ids"].astype(np.int64), tokenized["attention_mask"].astype(bool)


# ═══════════════════════════ StageTimer ═══════════════════════════

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


# ═══════════════════════════ TensorRT Engine Runner ═══════════════════════════

_NP_DTYPE = {
    trt.float32: np.float32, trt.float16: np.float16,
    trt.int32: np.int32, trt.int64: np.int64, trt.bool: np.bool_, trt.int8: np.int8,
}
_TORCH_DTYPE = {
    trt.float32: torch.float32, trt.float16: torch.float16,
    trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool, trt.int8: torch.int8,
}


class TRTEngineRunner:
    """使用 torch.cuda 管理内存的 TensorRT 引擎 (无需 pycuda)."""

    def __init__(self, engine_path, device="cuda:0"):
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(self.device)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = {}
        self.outputs = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            trt_dtype = self.engine.get_tensor_dtype(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            torch_dtype = _TORCH_DTYPE.get(trt_dtype, torch.float32)
            buf = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self.context.set_tensor_address(name, buf.data_ptr())
            info = {"shape": shape, "np_dtype": _NP_DTYPE.get(trt_dtype, np.float32),
                    "torch_dtype": torch_dtype, "buffer": buf}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = info
            else:
                self.outputs[name] = info

    def infer(self, feed_dict):
        """运行推理. feed_dict: {name: np.ndarray}."""
        with torch.cuda.stream(self.stream):
            for name, arr in feed_dict.items():
                info = self.inputs[name]
                src = torch.from_numpy(np.ascontiguousarray(arr)).to(info["torch_dtype"])
                info["buffer"].copy_(src)
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return {name: info["buffer"].cpu().numpy() for name, info in self.outputs.items()}


class DecomposedTRTRunner:
    """2-engine decomposed TensorRT 推理."""

    def __init__(self, trt_dir):
        config_path = os.path.join(trt_dir, "reflex_config.json")
        with open(config_path) as f:
            self.cfg = json.load(f)

        dcfg = self.cfg["decomposed"]
        self.per_step = dcfg.get("per_step_expert", False)

        vlm_path = os.path.join(trt_dir, "vlm_prefix.engine")
        expert_path = os.path.join(trt_dir, "expert_denoise.engine")

        print(f"加载 VLM prefix engine: {vlm_path}")
        t0 = time.perf_counter()
        self.vlm_runner = TRTEngineRunner(vlm_path)
        print(f"  耗时: {(time.perf_counter() - t0) * 1000:.0f} ms")

        print(f"加载 Expert denoise engine: {expert_path}")
        t0 = time.perf_counter()
        self.expert_runner = TRTEngineRunner(expert_path)
        print(f"  耗时: {(time.perf_counter() - t0) * 1000:.0f} ms")

    def run_vlm_prefix(self, feed_dict):
        t0 = time.perf_counter()
        results = self.vlm_runner.infer(feed_dict)
        elapsed = (time.perf_counter() - t0) * 1000

        prefix_pad_masks = results.pop("prefix_pad_masks")
        kv_dict = results
        return kv_dict, prefix_pad_masks, elapsed

    def run_expert_denoise(self, kv_dict, prefix_pad_masks, noise):
        feed = dict(kv_dict)
        feed["prefix_pad_masks"] = prefix_pad_masks
        feed["noise"] = noise
        t0 = time.perf_counter()
        results = self.expert_runner.infer(feed)
        elapsed = (time.perf_counter() - t0) * 1000
        actions = list(results.values())[0]
        return actions, elapsed

    def infer(self, feed_dict, noise):
        kv_dict, prefix_pad_masks, vlm_ms = self.run_vlm_prefix(feed_dict)
        actions, expert_ms = self.run_expert_denoise(kv_dict, prefix_pad_masks, noise)
        return actions, vlm_ms, expert_ms


# ═══════════════════════════ trtexec Benchmark (无 pycuda 依赖) ═══════════════════════════

def run_trtexec_benchmark(trt_dir, warmup=5, iterations=20):
    """使用 trtexec --loadEngine 进行基准测试 (不需要 pycuda)."""
    import subprocess

    trtexec = "/usr/src/tensorrt/bin/trtexec"
    results = {}

    for name in ["vlm_prefix", "expert_denoise"]:
        engine_path = os.path.join(trt_dir, f"{name}.engine")
        if not os.path.exists(engine_path):
            print(f"跳过 {name}: engine 文件不存在")
            continue

        cmd = [
            trtexec,
            f"--loadEngine={engine_path}",
            f"--warmUp={warmup * 1000}",  # ms
            f"--iterations={iterations}",
            "--useSpinWait",
            "--fp16",
        ]
        print(f"\n运行 trtexec 基准测试: {name} ...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr

        # 解析输出
        for line in output.split("\n"):
            if "mean" in line.lower() and "latency" in line.lower():
                print(f"  {line.strip()}")
            if "GPU Compute Time" in line or "Throughput" in line:
                print(f"  {line.strip()}")

        results[name] = output

    return results


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
        trt_dir = os.path.join(path_file, "trt_engines")
        tokenizer_path = os.path.join(path_file, "data_param/paligemma-3b-pt-224")
        norm_stats_path = os.path.join(path_file, "data_param/norm_stats.json")

        self.runner = DecomposedTRTRunner(trt_dir)
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
        feed_dict = {
            "img_base": img_base,
            "img_wrist_l": img_wrist_l,
            "img_wrist_r": img_wrist_r,
            "mask_base": mask_base,
            "mask_wrist_l": mask_wrist_l,
            "mask_wrist_r": mask_wrist_r,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }
        timer = StageTimer()
        vlm_times, expert_times, total_times = [], [], []
        t0 = time.perf_counter()
        actions, vlm_ms, expert_ms = self.runner.infer(feed_dict, noise)
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
    path_file = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Pi0.5 TensorRT 推理 benchmark")
    parser.add_argument("--trt-dir", type=str, default=os.path.join(path_file, "trt_engines"))
    parser.add_argument("--norm-stats", type=str,
                        default=os.path.join(path_file, "data_param/norm_stats.json"))
    parser.add_argument("--tokenizer", type=str,
                        default=os.path.join(path_file, "data_param/paligemma-3b-pt-224"))
    parser.add_argument("--pkl", type=str,
                        default=os.path.join(path_file, "data_param/test_openpi_example.pkl"))
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--noise-path", type=str, default=os.path.join(path_file, "data_param/1x50x32_random_numbers.npy"))
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--repeat", type=int, default=1, help="Timing runs")
    parser.add_argument("--trtexec-only", action="store_true",
                        help="仅使用 trtexec 基准测试 (不需要 pycuda)")
    args = parser.parse_args()

    print("=" * 60)
    print(" Pi0.5 TensorRT 推理 Benchmark")
    print("=" * 60)

    # trtexec-only 模式: 纯延迟测试
    if args.trtexec_only or not HAS_TRT:
        if not HAS_TRT:
            print("pycuda/tensorrt 不可用, 使用 trtexec 基准测试模式")
        run_trtexec_benchmark(args.trt_dir, warmup=args.warmup, iterations=args.repeat * 10)
        return

    # 完整推理模式
    runner = DecomposedTRTRunner(args.trt_dir)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    with open(args.norm_stats) as f:
        raw_stats = json.load(f)["norm_stats"]
    state_mean = np.array(raw_stats["state"]["mean"], dtype=np.float32)
    state_std = np.array(raw_stats["state"]["std"], dtype=np.float32)
    action_mean = np.array(raw_stats["actions"]["mean"], dtype=np.float32)
    action_std = np.array(raw_stats["actions"]["std"], dtype=np.float32)

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

    left_joint = example.get("observation_joint_position_left", np.zeros(7))
    right_joint = example.get("observation_joint_position_right", np.zeros(7))
    state = np.concatenate([left_joint, right_joint]).astype(np.float32)

    state = pad_vector(state, args.action_dim)
    state_normalized = normalize_state(state, state_mean, state_std)

    task_text = example["task"]
    if isinstance(task_text, list):
        task_text = task_text[0]

    lang_tokens, lang_masks = prepare_state_prompt(state_normalized, task_text, tokenizer)

    if args.noise_path and os.path.exists(args.noise_path):
        noise = np.load(args.noise_path).astype(np.float32)
    else:
        rng = np.random.RandomState(42)
        noise = rng.randn(1, 50, args.action_dim).astype(np.float32)

    feed_dict = {
        "img_base": img_base,
        "img_wrist_l": img_wrist_l,
        "img_wrist_r": img_wrist_r,
        "mask_base": mask_base,
        "mask_wrist_l": mask_wrist_l,
        "mask_wrist_r": mask_wrist_r,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
    }

    # Warmup
    print(f"\nWarmup ({args.warmup} runs)...")
    for _ in range(args.warmup):
        runner.infer(feed_dict, noise)

    # Timed runs
    print(f"Timing ({args.repeat} runs)...")
    timer = StageTimer()
    vlm_times, expert_times, total_times = [], [], []
    for i in range(args.repeat):
        t0 = time.perf_counter()
        actions, vlm_ms, expert_ms = runner.infer(feed_dict, noise)
        total_ms = (time.perf_counter() - t0) * 1000
        vlm_times.append(vlm_ms)
        expert_times.append(expert_ms)
        total_times.append(total_ms)
        timer.add("trt.vlm_prefix", vlm_ms)
        timer.add("trt.expert_denoise", expert_ms)
        print(f"  Run {i+1}: VLM={vlm_ms:.1f}ms  Expert={expert_ms:.1f}ms  Total={total_ms:.1f}ms")

    print(f"\n{'─' * 50}")
    print(f" TensorRT 耗时统计 (avg over {args.repeat} runs):")
    print(f"   VLM prefix:     {np.mean(vlm_times):.1f} ms")
    print(f"   Expert denoise: {np.mean(expert_times):.1f} ms")
    print(f"   Total e2e:      {np.mean(total_times):.1f} ms")
    print(f"{'─' * 50}")

    actions = actions[:, :, :args.action_dim]
    actions_unnorm = unnormalize_actions(actions, action_mean, action_std)
    print(f"total_inference: {np.mean(total_times):.3f} ms")
    print(f"action_shape: {tuple(actions.shape)}")
    print(f"action_sample: [{tensor_sample_text(actions_unnorm[0, 0, :8])}]")
    timer.print_summary("module_timings")
    print(f"actions_unnorm: {actions_unnorm}")
    # 对比 trtexec 基准
    # print("\n\n--- trtexec 原始基准测试 ---")
    # run_trtexec_benchmark(args.trt_dir, warmup=3, iterations=20)

#####



def main1():
    pi05_prefer = pi05_create_infer()
    path_file = os.path.dirname(os.path.abspath(__file__))
    npz_file_path = os.path.join(path_file, "data_param/1x50x32_random_numbers.npy")
    npz_file = np.load(npz_file_path)
    noise =  torch.from_numpy(npz_file).to(torch.float32)
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

 {'actions': array([[ 2.7859993e-03,  2.1946788e-02, -1.1114836e-02, ...,
         1.5382181e-09, -1.1480719e-09, -4.4500679e-09],
       [ 4.2109247e-03,  2.5279284e-02, -1.2748480e-02, ...,
         3.6691246e-09,  4.3185713e-10, -2.6882410e-09],
       [ 3.5441536e-03,  2.3908854e-02, -8.7801218e-03, ...,
         1.5474957e-09, -8.3879181e-10,  1.0535099e-09],
       ...,
       [ 4.5227055e-03,  8.9397430e-03, -1.6880035e-03, ...,
         3.7962402e-09, -7.1074657e-10, -1.4625884e-09],
       [ 4.6579111e-03,  9.4474554e-03, -2.8849840e-03, ...,
         1.6899437e-09,  9.1126168e-10, -3.0058980e-09],
       [ 3.9308462e-03,  9.6578598e-03, -2.8645992e-03, ...,
         2.1904802e-09,  2.0787350e-09, -4.1913268e-09]], dtype=float32)}

'''