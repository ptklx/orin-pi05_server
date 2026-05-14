#!/bin/bash
# Step 4: ONNX → TensorRT 转换脚本 (在 Orin AGX 上运行)
#
# 遵循 reflex-vla trt_build.py 模式:
#   - 2 个 engine: vlm_prefix.engine + expert_denoise.engine
#   - 默认 FP32 精度 (FP16 因 additive attention mask -2.38e38 溢出而不可用)
#   - workspace = 16384 MiB
#
# 前置条件:
#   - JetPack 6.x (TensorRT 10.x)
#   - trtexec 已在 PATH 中
#
# 用法:
#   chmod +x step4_build_trt.sh
#   ./step4_build_trt.sh [onnx_dir] [trt_dir]
#
# 注意: 不要使用 --fp16, 该模型的 additive attention mask 使用 -2.38e38,
#       远超 fp16 表示范围 (±65504), 会导致 NaN 传播, KV cache 全层错误.
#       fp32 精度 cosine=1.000, fp16 cosine=0.114.

set -e

ONNX_DIR="${1:-./onnx_models}"
TRT_DIR="${2:-./trt_engines}"
# 默认 fp32 (空字符串), 可通过 FP16_FLAG=--fp16 覆盖 (不推荐)
FP16="${FP16_FLAG:-}"
WORKSPACE=16384  # MiB, Orin AGX 64GB 统一内存

PRECISION="FP32"
[[ -n "$FP16" ]] && PRECISION="FP16 (警告: 精度可能不达标)"

mkdir -p "$TRT_DIR"

echo "============================================================"
echo " Pi0.5 Decomposed ONNX → TensorRT (reflex-vla 兼容)"
echo " ONNX 目录: $ONNX_DIR"
echo " TRT 目录:  $TRT_DIR"
echo " 精度:      $PRECISION"
echo " Workspace: ${WORKSPACE} MiB"
echo "============================================================"

# 1. VLM Prefix
echo ""
echo "[1/2] 转换 vlm_prefix.onnx → vlm_prefix.engine ..."
time /usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX_DIR/vlm_prefix.onnx" \
    --saveEngine="$TRT_DIR/vlm_prefix.engine" \
    $FP16 \
    --memPoolSize=workspace:${WORKSPACE}MiB \
    2>&1 | tee /dev/stderr | tail -20

# 2. Expert Denoise
echo ""
echo "[2/2] 转换 expert_denoise.onnx → expert_denoise.engine ..."
time /usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX_DIR/expert_denoise.onnx" \
    --saveEngine="$TRT_DIR/expert_denoise.engine" \
    $FP16 \
    --memPoolSize=workspace:${WORKSPACE}MiB \
    2>&1 | tee /dev/stderr | tail -20

echo ""
echo "============================================================"
echo " 转换完成!"
echo " VLM prefix engine:  $TRT_DIR/vlm_prefix.engine"
echo " Expert engine:      $TRT_DIR/expert_denoise.engine"
echo "============================================================"

# 复制配置
cp "$ONNX_DIR/reflex_config.json" "$TRT_DIR/reflex_config.json" 2>/dev/null || true
echo " Config: $TRT_DIR/reflex_config.json"

ls -lh "$TRT_DIR/"
