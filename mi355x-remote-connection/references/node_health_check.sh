#!/usr/bin/env bash
# MI355X Node Health Check — run on the host (outside containers)
# Outputs a structured summary of node readiness for experiment execution.

set -euo pipefail

echo "=== MI355X Node Health Check ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Hostname: $(hostname)"
echo ""

# --- GPU Detection ---
echo "--- GPU Status ---"
GPU_ARCH=$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -1 || echo "unknown")
echo "GPU Architecture: $GPU_ARCH"

if [ "$GPU_ARCH" != "gfx950" ]; then
    echo "WARNING: Expected gfx950 (MI355X), got $GPU_ARCH"
fi

GPU_COUNT=$(rocm-smi --showid --json 2>/dev/null | python3 -c "
import json, sys, re
try:
    d = json.load(sys.stdin)
    print(sum(1 for k in d if re.search(r'card\d+', k, re.IGNORECASE)))
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
echo "GPU Count: $GPU_COUNT"

# GPU utilization summary
rocm-smi --showuse 2>/dev/null | head -20 || echo "rocm-smi --showuse failed"
echo ""

# --- ROCm Version ---
echo "--- ROCm ---"
if [ -f /opt/rocm/.info/version ]; then
    echo "ROCm Version: $(cat /opt/rocm/.info/version)"
else
    echo "ROCm Version: not found at /opt/rocm/.info/version"
fi
echo ""

# --- Disk Space ---
echo "--- Disk Space ---"
df -h / /data /mnt/dcgpuval 2>/dev/null | tail -n +2 || echo "df failed"

DATA_FREE_GB=$(df -BG /data 2>/dev/null | tail -1 | awk '{gsub("G",""); print $4}' || echo "0")
echo ""
if [ "${DATA_FREE_GB:-0}" -lt 100 ]; then
    echo "WARNING: /data has less than 100GB free ($DATA_FREE_GB GB)"
else
    echo "OK: /data has ${DATA_FREE_GB}GB free"
fi
echo ""

# --- Running Containers ---
echo "--- Docker Containers ---"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" 2>/dev/null | head -10 || echo "docker not available"
echo ""

# --- Local Models ---
echo "--- Local Models (/data/) ---"
if [ -d /data ]; then
    for d in /data/*/; do
        if [ -d "$d" ]; then
            SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
            echo "  $(basename "$d"): $SIZE"
        fi
    done
else
    echo "  /data not mounted"
fi
echo ""

# --- Summary ---
echo "=== Summary ==="
echo "GPU Arch: $GPU_ARCH"
echo "GPU Count: $GPU_COUNT"
echo "Data Disk Free: ${DATA_FREE_GB:-unknown}GB"
echo "Containers Running: $(docker ps -q 2>/dev/null | wc -l || echo unknown)"
echo "=== Done ==="
