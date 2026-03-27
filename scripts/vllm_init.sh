#!/usr/bin/env bash
# ============================================================
# Fractal — vLLM Initialization Script
# Optimized for Intel Arc Pro B70 B-series (Xe2 / Battlemage)
# ============================================================
set -euo pipefail

echo "╔══════════════════════════════════════════════╗"
echo "║  Fractal — vLLM Inference Engine Startup     ║"
echo "║  Target: Intel Arc Pro B70 (XPU)             ║"
echo "╚══════════════════════════════════════════════╝"

# ── Intel oneAPI / Level Zero GPU Configuration ──
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER="${ZE_ENABLE_PCI_ID_DEVICE_ORDER:-1}"
export SYCL_CACHE_PERSISTENT="${SYCL_CACHE_PERSISTENT:-1}"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export NEOReadDebugKeys=1
export ClDeviceGlobalMemSizeAvailablePercent=100

# ── vLLM Target Device ──
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-xpu}"

# ── Dynamic Quantization Configuration ──
# Supports AWQ, GPTQ, and FP8 — auto-detects from model config
QUANTIZATION_METHOD="${VLLM_QUANTIZATION:-awq}"
echo "[INIT] Quantization method: ${QUANTIZATION_METHOD}"

# ── Model Configuration ──
MODEL_PATH="${VLLM_MODEL_PATH:-/models/default}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

# ── Validate model path ──
if [ ! -d "${MODEL_PATH}" ] && [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[WARN] Model path '${MODEL_PATH}' does not exist or lacks config.json."
    echo "[WARN] vLLM will attempt to download from HuggingFace Hub."
fi

# ── Detect quantization from model config (dynamic) ──
if [ -f "${MODEL_PATH}/config.json" ]; then
    # Auto-detect quantization from model metadata
    if grep -q '"quantization_config"' "${MODEL_PATH}/config.json" 2>/dev/null; then
        DETECTED_QUANT=$(python3 -c "
import json
with open('${MODEL_PATH}/config.json') as f:
    cfg = json.load(f)
qcfg = cfg.get('quantization_config', {})
print(qcfg.get('quant_method', '${QUANTIZATION_METHOD}'))
" 2>/dev/null || echo "${QUANTIZATION_METHOD}")
        echo "[INIT] Auto-detected quantization from model: ${DETECTED_QUANT}"
        QUANTIZATION_METHOD="${DETECTED_QUANT}"
    fi
fi

# ── Build vLLM launch command ──
VLLM_ARGS=(
    "--model" "${MODEL_PATH}"
    "--host" "${HOST}"
    "--port" "${PORT}"
    "--max-model-len" "${MAX_MODEL_LEN}"
    "--gpu-memory-utilization" "${GPU_MEMORY_UTILIZATION}"
    "--tensor-parallel-size" "${TENSOR_PARALLEL_SIZE}"
    "--device" "${VLLM_TARGET_DEVICE}"
    "--trust-remote-code"
    "--disable-log-requests"
    "--enforce-eager"
)

# Add quantization if not "none"
if [ "${QUANTIZATION_METHOD}" != "none" ]; then
    VLLM_ARGS+=("--quantization" "${QUANTIZATION_METHOD}")
fi

# ── Intel XPU-specific optimizations ──
# Eager mode is required for XPU (no CUDA graph equivalent yet)
# Enforce eager is already set above

# ── Optional: Load config overrides from JSON ──
CONFIG_FILE="${VLLM_CONFIG_FILE:-/app/config/vllm_config.json}"
if [ -f "${CONFIG_FILE}" ]; then
    echo "[INIT] Loading additional config from ${CONFIG_FILE}"
    # Parse JSON config for additional vLLM args
    EXTRA_ARGS=$(python3 -c "
import json
with open('${CONFIG_FILE}') as f:
    cfg = json.load(f)
for k, v in cfg.get('extra_args', {}).items():
    print(f'--{k}')
    print(str(v))
" 2>/dev/null || echo "")
    
    if [ -n "${EXTRA_ARGS}" ]; then
        while IFS= read -r line; do
            VLLM_ARGS+=("${line}")
        done <<< "${EXTRA_ARGS}"
    fi
fi

echo "[INIT] Launching vLLM OpenAI-compatible API server..."
echo "[INIT] Command: python -m vllm.entrypoints.openai.api_server ${VLLM_ARGS[*]}"
echo "────────────────────────────────────────────────"

exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
