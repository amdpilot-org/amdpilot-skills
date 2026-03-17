---
name: env-probe
description: >
  Inspect AMD/ROCm Docker runtime environment before writing any code. Use BEFORE torch.compile,
  CUDAGraph capture, or any kernel optimization. Identifies the Docker image (rocm/vllm, rocm/sgl-dev
  tags with date stamps), detects ROCm/vLLM/SGLang versions, hidden framework defaults (inductor
  max_autotune, triton.cudagraphs), known Docker-specific bugs (hipBLASLt solver crash, FP8
  flash attn), and missing packages. Outputs CRITICAL/WARNING/INFO report with recommended fixes.
  Triggered by: starting work in an AMD Docker, "check environment", "why is torch.compile hanging",
  "env probe", "what Docker am I in", Phase 0 of any AMD optimization experiment.
---

# AMD/ROCm Docker Environment Probe

**Run this before writing any optimization code.** AMD Docker images silently set framework defaults
that differ from stock PyTorch. These hidden defaults cause stalls, crashes, and wrong results that
are impossible to diagnose by looking at code alone.

## Why This Exists

Two problems this solves:

1. **Image identity**: We frequently switch between Docker images from `rocm/vllm` and
   `rocm/sgl-dev` (see [Docker Hub: rocm/vllm](https://hub.docker.com/r/rocm/vllm/tags),
   [Docker Hub: rocm/sgl-dev](https://hub.docker.com/r/rocm/sgl-dev/tags)). Tags include date
   stamps (e.g., `v0.5.9-rocm720-mi35x-20260317`) and the exact combination of ROCm version,
   GPU architecture target, and framework version changes constantly. The probe tells you exactly
   which environment you are in.

2. **Hidden defaults**: ROCm Docker images override PyTorch/Triton defaults at the system level.
   For example, `max_autotune=True` as a global default means `torch.compile(mode="default")`
   benchmarks every GEMM across ATEN+TRITON+CPP backends. These are invisible to `pip list` or
   `rocm-smi`.

## Known Docker Image Families

| Image | Hub | Typical tag pattern | Contents |
|-------|-----|---------------------|----------|
| `rocm/vllm` | [tags](https://hub.docker.com/r/rocm/vllm/tags) | `v0.14.0_amd_dev`, `rocm7.0.0_vllm_0.11.2_YYYYMMDD` | vLLM + ROCm + PyTorch |
| `rocm/sgl-dev` | [tags](https://hub.docker.com/r/rocm/sgl-dev/tags) | `v0.5.9-rocm720-mi35x-YYYYMMDD`, `v0.5.9-rocm700-mi30x-YYYYMMDD` | SGLang + ROCm + PyTorch |

Tag naming conventions for `rocm/sgl-dev`:
- `v{sglang_version}-rocm{rocm_version}-mi{gpu_arch}-{date}`
- `rocm720` = ROCm 7.2.0, `rocm700` = ROCm 7.0.0
- `mi35x` = MI355X (gfx950), `mi30x` = MI300X (gfx942)
- Date stamp `YYYYMMDD` identifies the build — behavior can differ between daily builds

## How to Use

### Step 1: Run the probe script

```bash
python /path/to/env_probe.py
```

Or copy the probe script from [references/env_probe.py](references/env_probe.py) and run it
inside your Docker container. Self-contained — no deps beyond PyTorch.

### Step 2: Read the output

The report starts with **Docker / Image Identity** (ROCm version, vLLM/SGLang version, GPU arch),
then shows issues by severity:

| Level | Meaning | Action |
|-------|---------|--------|
| **CRITICAL** | Will cause hangs, crashes, or silent wrong results | **Must fix before proceeding** |
| **WARNING** | Suboptimal default, will hurt performance | Fix before benchmarking |
| **INFO** | Informational, no action needed | Document for reproducibility |

### Step 3: Apply fixes

Each CRITICAL/WARNING item includes a recommended fix. Apply at the top of your script, before
any `torch.compile()` or `torch.cuda.CUDAGraph()` call.

## What the Probe Checks

### Category 0: Docker Container Identity (NEW)
- Whether running inside Docker
- Docker image tag (from env vars, label files, `/proc/1/environ`)
- ROCm version (from `/opt/rocm/.info/version`)
- vLLM version (import or pip)
- SGLang version (import or pip)
- Synthesized environment summary line

### Category 1: Surface Facts (versions, hardware)
- Python version, PyTorch version, Triton version
- GPU architecture (gfx target from rocminfo)
- AITER, Composable Kernel, flash-attn availability and versions
- hipBLASLt availability

### Category 2: Runtime Behavior Defaults (the hidden landmines)
- `torch._inductor.config.max_autotune` — if True, causes indefinite stall with torch.compile
- `torch._inductor.config.max_autotune_gemm_backends` — which backends inductor will benchmark
- `torch._inductor.config.triton.cudagraphs` — unstable on ROCm
- `torch._inductor.config.triton.cudagraph_trees` — unstable on ROCm
- `torch._inductor.config.memory_planning` — causes deep recursion crash on ROCm
- `torch._dynamo.config.cache_size_limit` — too small causes recompilation loops
- `torch.backends.cudnn.benchmark` and `allow_tf32` defaults

### Category 3: Known Bug Markers
- hipBLASLt solver discovery (HIPBLAS_STATUS_NOT_INITIALIZED)
- FP8 flash attention availability
- gfx950/gfx942 ASM GEMM kernel availability
- AITER function signatures (argument combos that were broken in older versions)

### Category 4: Environment Variables
- `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`
- `HSA_ENABLE_SDMA`, `HIP_FORCE_DEV_KERNARG`
- `PYTORCH_TUNABLEOP_ENABLED`, `PYTORCH_TUNABLEOP_TUNING`
- `TORCH_COMPILE_DEBUG`, `TORCHINDUCTOR_*` overrides

## Recommended Inductor Configuration for ROCm

When the probe flags inductor defaults as CRITICAL, apply this configuration block before any
`torch.compile()` call:

```python
import torch._inductor.config as inductor_config
import torch._dynamo.config as dynamo_config

# Prevent indefinite GEMM autotuning stall
inductor_config.max_autotune = False
inductor_config.max_autotune_gemm_backends = "ATEN"

# Disable unstable triton cudagraphs on ROCm
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False

# Prevent deep recursion crash
inductor_config.memory_planning = False

# Prevent cache eviction / recompilation loops
dynamo_config.cache_size_limit = 128
```

See [references/inductor-rocm-defaults.md](references/inductor-rocm-defaults.md) for the full
explanation of each setting and when you might want to override them.

## Integration with Other Skills

- **amd-rocm-porting**: Run env-probe as Phase 0.5 (after Phase 0 environment setup, before Phase 1 porting)
- **amd-kernel-optimization**: Run env-probe before profiling baseline
- **gpu-profiling**: Probe checks that rocprofv3 is available and functional

## Adding New Checks

When you discover a new Docker-specific gotcha, add it to `references/env_probe.py`:
1. Add the check function
2. Add it to the appropriate category in `run_all_checks()`
3. Include the severity level (CRITICAL/WARNING/INFO) and recommended fix
4. Document the failure mode (what happens if the agent doesn't know about this)

This skill is meant to grow — every experiment that hits an environment issue should contribute
a new check back to the probe.
