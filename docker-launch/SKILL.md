---
name: docker-launch
description: >
  Standardized Docker container launch and model serving recipes for AMD ROCm MI355X.
  Covers required container flags (--ipc=host, --privileged, --security-opt), model-specific
  sglang/vllm launch parameters, and benchmark commands. Use whenever starting a container
  for model inference, testing, or benchmarking. Prevents common failures from missing flags
  (SIGKILL from seccomp, IPC exhaustion from missing --ipc=host, GPU access from missing devices).
  Triggered by: "start a container", "launch model server", "run inference", "benchmark model",
  "docker run", any task requiring a running model endpoint.
---

# Docker Launch & Model Serving on AMD ROCm MI355X

**Use these standardized recipes when launching containers or model servers.**
Missing flags cause silent failures (SIGKILL, OOM, GPU inaccessibility) that waste hours.

## 1. Required Container Flags

Every AMD ROCm container MUST include these flags:

```bash
docker run -d \
    --name <container-name> \
    --ipc=host \
    --network=host \
    --privileged \
    --shm-size 32G \
    --ulimit core=0:0 \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    <image> <command>
```

### Why Each Flag Matters

| Flag | Purpose | Failure without it |
|------|---------|-------------------|
| `--ipc=host` | Share host IPC namespace for PyTorch shared memory | OOM / SIGKILL when loading models via shared memory |
| `--privileged` | Full device access for GPU operations | Permission denied on some GPU memory operations |
| `--shm-size 32G` | Shared memory for data loaders and IPC | Bus errors (SIGBUS) during multi-process operations |
| `--ulimit core=0:0` | Prevent ~200GB GPU core dumps on crashes | Disk full from massive core dumps |
| `--security-opt seccomp=unconfined` | Disable seccomp filtering | SIGKILL from blocked syscalls |
| `--security-opt apparmor=unconfined` | Disable AppArmor restrictions | Process killed by security policy |
| `--device=/dev/kfd --device=/dev/dri` | GPU device access | No GPU visible |
| `--group-add video` | Video group for GPU access | Permission denied on /dev/kfd |

## 2. Environment Variables

```bash
# Common for all models
-e HF_HUB_OFFLINE=1              # Use cached models only (air-gapped)
-e GPU_COREDUMP_ENABLE=0          # Prevent GPU core dumps

# SGLang-specific
-e SGLANG_ROCM_FUSED_DECODE_MLA=0  # Required for triton backend stability
-e SGLANG_USE_AITER=1              # Enable AITER prefill kernels (for compatible models)
```

## 3. Model Launch Recipes

### Qwen3.5-397B-A17B (SGLang, TP=8)

```bash
python3 -m sglang.launch_server \
    --model-path /sgl-workspace/models/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/<hash> \
    --served-model-name Qwen3.5-397B-A17B \
    --tp 8 \
    --trust-remote-code \
    --attention-backend triton \
    --mem-fraction-static 0.80 \
    --max-mamba-cache-size 128 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --watchdog-timeout 1200 \
    --host 0.0.0.0 --port 30000
```

Key: `--max-mamba-cache-size 128` is critical for the 45 recurrent DeltaNet layers (+45% perf vs default 64).

### Kimi-K2.5 (SGLang, TP=8, Hybrid Attention)

```bash
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_USE_AITER=1
/opt/venv/bin/python3 -m sglang.launch_server \
    --model-path moonshotai/Kimi-K2.5 \
    --tp 8 \
    --trust-remote-code \
    --decode-attention-backend triton \
    --prefill-attention-backend aiter \
    --mem-fraction-static 0.85 \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    --host 0.0.0.0 --port 30000
```

Key: Hybrid attention — triton for decode, aiter ASM kernels for prefill. Median decode 23.5ms (42.6 tok/s).

#### With Eagle3 Speculative Decoding (1.7-1.8x speedup)

```bash
# Same env vars as above, add:
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lightseekorg/kimi-k2.5-eagle3 \
    --mem-fraction-static 0.75   # Lower to accommodate draft model
```

### GLM-5-FP8 (SGLang, TP=8)

```bash
python3 -m sglang.launch_server \
    --model-path zai-org/GLM-5-FP8 \
    --served-model-name glm-5-fp8 \
    --tp 8 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --mem-fraction-static 0.80 \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --chunked-prefill-size 131072 \
    --watchdog-timeout 1200 \
    --port 30000
```

Key: Native Sparse Attention (NSA) with tilelang backend. TP=4 possible with `--mem-fraction-static 0.60 --disable-cuda-graph`.

### vLLM Models (General Template)

```bash
/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model <model-path> \
    --tensor-parallel-size <TP> \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000
```

Note: vLLM uses `/usr/bin/python3` in rocm/vllm-dev images, NOT `/opt/venv/bin/python3`.

## 4. Benchmark Commands

### Single-batch latency (SGLang)

```bash
python3 -m sglang.bench_one_batch_server \
    --model None \
    --base-url http://localhost:30000 \
    --batch-size 1 \
    --input-len 1024 \
    --output-len 512
```

### Full benchmark sweep

```bash
for INPUT_LEN in 1024 8192 16384; do
  for OUTPUT_LEN in 512 1024; do
    echo "====== Input: ${INPUT_LEN}, Output: ${OUTPUT_LEN} ======"
    python3 -m sglang.bench_one_batch_server \
        --model None --base-url http://localhost:30000 \
        --batch-size 1 --input-len $INPUT_LEN --output-len $OUTPUT_LEN
  done
done
```

### Online serving benchmark

```bash
for con in 16 32 64 128; do
    PROMPTS=$(($con * 5))
    python3 -m sglang.bench_serving \
        --dataset-name random \
        --random-input-len 3200 \
        --random-output-len 800 \
        --num-prompt $PROMPTS \
        --random-range-ratio 1.0 \
        --max-concurrency $con
done
```

## 5. Process Cleanup (Safe)

Kill model server processes WITHOUT killing the agent:

```bash
# Safe — only matches python server processes, not agent shells
kill -9 $(pgrep -f 'python3 -m (sglang|vllm)') 2>/dev/null; sleep 2
```

**NEVER use**: `kill -9 $(pgrep -f 'sglang\|vllm')` — this matches the agent's own shell and causes exit 137 (self-kill).

## 6. Common Pitfalls

1. **SGLang Python path**: Use `/opt/venv/bin/python3` in `rocm/sgl-dev` images
2. **vLLM Python path**: Use `/usr/bin/python3` in `rocm/vllm-dev` images
3. **PYTHONPATH pollution**: Always `unset PYTHONPATH` before running test commands
4. **Model loading time**: 15-30 minutes for large models — don't restart server repeatedly
5. **HIP_VISIBLE_DEVICES**: Use to restrict GPU subset (e.g., `HIP_VISIBLE_DEVICES=0,1,2,3` for TP=4)
