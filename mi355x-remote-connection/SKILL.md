---
name: mi355x-remote-connection
description: >
  Connect to and validate an AMD MI355X (gfx950) GPU node for remote development and
  experiment execution. Use this skill when starting work on a remote MI355X node, verifying
  the environment before launching experiments, or setting up a development container. Covers
  host-level checks (rocm-smi, GPU availability, disk space), container entry and validation
  (ROCm version, HIP env, model cache), and safe remote workspace management (worktree, sync,
  cleanup). Triggered by: "connect to MI355X", "check node", "remote GPU setup", "verify
  environment", "node health check", Phase 0 of any experiment launch.
---

# MI355X Remote Connection Skill

**Use this skill before starting any work on a remote MI355X (gfx950) GPU node.** It standardizes
environment validation, container access, and workspace management so that agents can treat remote
GPU nodes as a unified execution backend.

## Why This Exists

When working with remote AMD GPU nodes, agents must handle:

1. **Node-level validation**: Is the node reachable? Are GPUs healthy? Is there enough disk space?
2. **Container selection**: Which Docker image matches the task (rocm/sgl-dev vs rocm/vllm-dev)?
3. **Environment verification**: Are ROCm, HIP, PyTorch, and model caches correctly configured?
4. **Workspace safety**: How to avoid clobbering in-progress work while testing changes?

Without a standardized skill, each session reinvents these checks, risks using wrong containers,
and wastes time debugging environment issues that could have been caught upfront.

## Node Specification

| Property | Value |
|----------|-------|
| GPU | 8x AMD Instinct MI355X |
| Architecture | gfx950 |
| ROCm Version | 7.2.0 (check `/opt/rocm/.info/version`) |
| FP8 dtype | `torch.float8_e4m3fn` (NOT `e4m3fnuz` — that's gfx942) |
| Local storage | `/data` (NVMe, ~2TB free) |
| NFS | `/mnt/dcgpuval` (slow, use local `/data` for models when possible) |
| Local models | `/data/meta-llama/`, `/data/Kimi-K2.5/`, `/data/google-bert/` |
| NFS models | `/mnt/dcgpuval/huggingface/` (Qwen3.5, GLM-4.7, GLM-5, etc.) |

## Step 1: Host-Level Health Check

Run these checks before entering any container:

```bash
# 1. Verify node is reachable
hostname && whoami

# 2. Check GPU status — all 8 GPUs should be visible and healthy
rocm-smi --showuse --showtemp --showpower

# 3. Check GPU architecture — must be gfx950
rocminfo | grep -o 'gfx[0-9a-f]*' | head -1
# Expected: gfx950

# 4. Check GPU memory availability
rocm-smi --showmeminfo vram --json

# 5. Check disk space — local NVMe should have >100GB free
df -h /data /
# /data is preferred for model storage (NVMe, fast)
# / is the system disk

# 6. Check running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | head -10

# 7. Check if any GPU is in use (occupied by running experiments)
# Look for processes using GPU memory
rocm-smi --showpids 2>/dev/null || rocm-smi --showuse
```

### Interpreting Results

| Check | Healthy | Action if unhealthy |
|-------|---------|---------------------|
| `gfx950` in rocminfo | Yes | Wrong node — do not proceed |
| All 8 GPUs visible | Yes | Check node health, rocm-smi/driver status, confirm you're on host (not inside container) |
| GPU temp < 90C | Yes | Wait for cooldown or check cooling |
| `/data` > 100GB free | Yes | Clean old experiment artifacts |
| No zombie containers | Yes | `docker rm -f <name>` if stale |

## Step 2: Container Selection and Entry

### Docker Image Families

| Image | Use for | Tag pattern |
|-------|---------|-------------|
| `rocm/sgl-dev` | SGLang experiments | `v0.5.9-rocm720-mi35x-YYYYMMDD` |
| `rocm/vllm-dev` | vLLM experiments | `rocm7.2.0_vllm_0.14.0_YYYYMMDD` |

Tag decoding for `rocm/sgl-dev`:
- `v{sglang}` — SGLang version
- `rocm{NNN}` — ROCm version (`720` = 7.2.0, `700` = 7.0.0)
- `mi35x` — MI355X / gfx950 (vs `mi30x` = MI300X / gfx942)
- `YYYYMMDD` — build date (behavior can differ between daily builds)

### Enter an existing container

```bash
# List available containers
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

# Enter a running container
docker exec -it <container_name> bash
```

### Start a new container (if needed)

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size 64g \
  -v /data:/data \
  -v /mnt/dcgpuval:/mnt/dcgpuval \
  -e HF_HOME=/data/huggingface \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --name my-experiment \
  rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260317 \
  bash
```

Key flags:
- `--device=/dev/kfd --device=/dev/dri` — GPU access on ROCm
- `--group-add video --group-add render` — permission groups for GPU
- `--shm-size 64g` — shared memory for multi-GPU communication
- `-v /data:/data` — mount local NVMe for fast model loading
- `-e HF_HOME=/data/huggingface` — use local cache, not NFS

## Step 3: In-Container Environment Validation

After entering the container, run these checks:

```bash
# 1. Verify ROCm version
cat /opt/rocm/.info/version
# Expected: 7.2.0 or similar

# 2. Verify GPU visibility inside container
rocm-smi --showid --json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{len([k for k in d if \"card\" in k.lower()])} GPUs visible')"

# 3. Verify PyTorch + ROCm
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 4. Check HIP environment variables
env | grep -E '^(HIP_|ROCR_|HSA_|PYTORCH_|HF_)' | sort

# 5. Verify model cache accessibility
ls /data/meta-llama/ 2>/dev/null && echo "Local models OK" || echo "WARNING: /data not mounted"
ls $HF_HOME 2>/dev/null && echo "HF cache OK" || echo "WARNING: HF_HOME not set or not mounted"

# 6. Check framework versions
python3 -c "
try:
    import vllm; print(f'vLLM {vllm.__version__}')
except: print('vLLM not installed')
try:
    import sglang; print(f'SGLang {sglang.__version__}')
except: print('SGLang not installed')
"

# 7. Run env-probe if available (see env-probe skill)
python3 /path/to/env_probe.py 2>/dev/null || echo "env_probe.py not available — run manually"
```

### Critical Environment Variables

| Variable | Expected | Purpose |
|----------|----------|---------|
| `HIP_VISIBLE_DEVICES` | `0,1,...,7` | GPU visibility |
| `HF_HOME` | `/data/huggingface` | Model cache (local, not NFS) |
| `HSA_ENABLE_SDMA` | `0` (if set) | Workaround for some transfer issues |
| `PYTORCH_TUNABLEOP_ENABLED` | `1` (if tuning) | TunableOp for GEMM |

## Step 4: Safe Workspace Management

When working on a remote node, always check the repo state before modifying:

### Option A: Use existing repo directly (read-only or exclusive access)

```bash
cd /workspace/repo
git status
# If clean, work directly
```

### Option B: Create a detached worktree (safe parallel work)

```bash
cd /workspace/repo
git worktree add /tmp/experiment-$(date +%s) HEAD
cd /tmp/experiment-*
# Work here — original repo untouched
# Clean up when done: git worktree remove /tmp/experiment-*
```

### Option C: Sync local changes to remote verification directory

```bash
# From local machine:
rsync -avz --exclude '.git' --exclude '__pycache__' \
  ./my-changes/ remote-node:/tmp/verify-$(date +%s)/
```

### Cleanup

Always clean up temporary worktrees and sync directories after verification:

```bash
# Remove worktrees
git worktree list
git worktree remove /tmp/experiment-* --force

# Remove temp sync dirs
rm -rf /tmp/verify-*
```

## Integration with Other Skills

| Skill | Relationship |
|-------|-------------|
| **env-probe** | Run env-probe inside the container after Step 3 for deep environment analysis |
| **amd-rocm-porting** | Use this skill as Phase 0 before starting any porting work |
| **amd-kernel-optimization** | Validate node health before profiling baseline |
| **gpu-profiling** | Confirm rocprofv3 availability and GPU state before trace collection |

## Model Loading Priority

Always prefer local storage over NFS:

1. **`/data/`** (local NVMe) — fastest, check here first
2. **`/mnt/dcgpuval/local_models/`** — local copies of common models
3. **`/mnt/dcgpuval/huggingface/`** (NFS) — slowest, avoid for large models

If a model is on NFS but will be used repeatedly, copy it to `/data/` first:

```bash
# Example: copy Qwen3.5 from NFS to local
cp -r /mnt/dcgpuval/huggingface/Qwen3.5-397B-A17B-FP8 /data/
```

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `rocm-smi` shows 0 GPUs | Container missing `--device` flags | Restart container with `--device=/dev/kfd --device=/dev/dri` |
| `torch.cuda.is_available()` returns False | ROCm/HIP not linked to PyTorch | Check PyTorch was built for ROCm: `torch.version.hip` |
| Very slow model loading | Loading from NFS | Copy model to `/data/` |
| Permission denied on GPU | Missing group membership | Add `--group-add video --group-add render` |
| `CUDA error: no kernel image` | Wrong GPU arch in Docker image | Use `mi35x` tagged image, not `mi30x` |
