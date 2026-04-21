---
name: gpu-recompute-headroom-check
description: >
  Check whether activation checkpointing / recompute is overspending compute on GPU training workloads.
  Use when memory usage is comfortably below device capacity, micro-batch size is already minimal,
  and backward time remains high. Guides a general verify-before-change workflow for reducing
  recompute layers or checkpoint coverage without naming workload-specific knobs.
---

# GPU Recompute Headroom Check

Use this skill when ALL of the following are true:

- The workload is training, not pure inference
- Peak memory usage is comfortably below capacity (roughly `<60%` of total device memory, or an equivalent clear headroom signal across ranks)
- Micro-batch size is already `1` or otherwise pinned by workload constraints
- Backward pass remains a large fraction of step time

## Goal

Decide whether activation checkpointing is now a compute tax rather than a memory necessity.

## 4 Rules

1. **Profile first.** Do not change recompute settings from intuition.
2. **Treat headroom as a prerequisite, not proof.** Low memory alone does not justify reducing recompute.
3. **Change checkpoint coverage gradually.** Prefer reducing recomputed layers or interval before disabling everything.
4. **Re-measure after every change.** Track both throughput and peak memory.

## Evidence To Collect

- Peak memory used vs total available per rank
- Step-time breakdown: forward / backward / optimizer / communication
- Whether micro-batch size can increase; if not, record why
- Current checkpointing mode and coverage:
  - full-layer / selective / interval / block-level
  - number of recomputed layers or equivalent coverage
- Any recent OOMs or instability near the current config

If you do not have this breakdown yet, use the `gpu-profiling` skill first.

## When Recompute Is A Good Candidate

Consider a recompute reduction experiment when:

- Peak memory stays well below capacity even after warmup
- Backward time is disproportionately high relative to forward
- The workload is already at `mbs=1` or another hard floor, so saved memory is not buying a larger batch
- Profiler evidence suggests repeated backward-side work rather than communication or dataloader stalls

## Experiment Ladder

1. Confirm the current checkpoint coverage and where it is configured.
   Megatron-style stacks often express this as a recomputed-layer count or checkpoint granularity setting; treat those names as examples, not required APIs.
2. Reduce recompute by one notch:
   - fewer recomputed layers
   - wider checkpoint interval
   - narrower checkpoint scope
3. Re-run with the same benchmark methodology.
4. Compare:
   - throughput / TFLOP/s
   - peak memory
   - OOM margin
   - numerical stability if the run is long enough to see it
5. Keep the change only if throughput rises and memory remains safe.

## Stop Conditions

- Peak memory approaches an unsafe range
- Compile / graph behavior regresses
- Throughput gain is only noise
- Convergence or stability issues appear

## Anti-Patterns

- Do not reduce recompute just because memory is below `100%`; the trigger is meaningful headroom plus backward cost.
- Do not change micro-batch size in the same experiment. That confounds the diagnosis.
- Do not confuse recompute overhead with some other dominant kernel bottleneck; if another category dominates, use the relevant profiling or kernel-optimization skill instead.
- Do not write workload-specific cookbook advice here. This skill should point to a bottleneck class, not a named benchmark knob.

## Output

Record:

- Memory headroom before/after
- Recompute coverage before/after
- Step-time breakdown before/after
- Whether the change looks generalizable or workload-specific
