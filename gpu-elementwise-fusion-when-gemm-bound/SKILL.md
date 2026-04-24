---
name: gpu-elementwise-fusion-when-gemm-bound
description: >
  Reduce AMD GPU step time when GEMM is no longer the dominant bottleneck and
  activation/normalization/other elementwise chains still consume a meaningful
  share of GPU time. Use after profiling shows memory-bound elementwise work
  remains hot despite a healthy GEMM backend. Covers generic fusion candidates
  such as activation+mul, residual+norm, bias+activation, and other multi-op
  epilogues on ROCm without naming framework-specific knobs.
---

# GPU Elementwise Fusion When GEMM-Bound Work Is Already Healthy

**Rule: do not start here without a real profile.**
This skill is for the case where the obvious GEMM/backend work has already been
done or ruled out, but the step is still paying for repeated activation,
normalization, cast, add, or multiply kernels that move the same tensors
through memory multiple times.

## When to use

Use this skill when:

- a profile already shows that GEMM/backend dispatch is no longer the dominant
  unresolved bottleneck
- aggregated activation/normalization/other elementwise kernels still account
  for roughly **15%+** of stage or step GPU time
- the hot path shows repeated memory-bound chains such as:
  - activation -> multiply
  - residual add -> normalization
  - bias -> activation
  - cast -> elementwise -> cast
  - multi-op MLP epilogues around an otherwise healthy matrix multiply
- the next likely lift is fewer launches or less memory traffic, not a new GEMM
  backend, communication fix, or topology change
- you have a reproducible before/after profiling path that can prove whether the
  chain collapsed into fewer kernels

## When NOT to use

Do **not** use this skill when:

- you have not yet profiled the workload or classified GPU time by category
- GEMM, grouped GEMM backend dispatch, attention, communication, or topology
  is still the dominant unresolved bottleneck
- compile coverage or graph stability is still broken; fix that first through
  `amd-rocm-porting` or `amd-kernel-optimization`
- the real issue is backward-memory tradeoff / activation checkpointing rather
  than forward or backward elementwise launch chains
- the candidate "fusion" is really a framework-specific config flag with no
  generic reasoning behind it

## Decision tree / routing questions

Use this routing tree before changing code:

```text
Did you profile and classify GPU time first?
└─ NO -> gpu-profiling

Is GEMM / grouped GEMM backend dispatch still the dominant unresolved hotspot?
└─ YES -> amd-kernel-optimization or grouped-gemm backend dispatch checks first

Is compile coverage / graph stability still broken?
└─ YES -> amd-rocm-porting first

Do activation / normalization / elementwise chains still take ~15%+ of GPU time
after GEMM/backend work is already healthy?
└─ YES -> stay in this skill

Is the likely next move reducing backward recomputation rather than collapsing
elementwise kernels?
└─ YES -> use the recompute-headroom lane instead of this skill
```

### What to look for

You are looking for repeated chains where the same tensor is read and written
multiple times with little arithmetic intensity between steps. Common generic
patterns:

- activation followed by elementwise combine
- add followed by normalization
- normalization followed by cast or scale
- gated or residual epilogues around an already-optimized matrix multiply

The goal is **not** "find a specific fused kernel by name." The goal is:

1. identify whether a repeated elementwise chain is actually hot
2. decide whether that chain can be collapsed at graph/backend/source level
3. verify the chain shrank in launches or memory traffic after the change

## AMD translation

Every imported fusion intuition must be translated explicitly:

| Upstream / NVIDIA-style assumption | AMD / ROCm translation | Validate with |
| --- | --- | --- |
| one fused epilogue exists behind a CUDA-only flag | prefer source-level fusion, Triton fusion, Inductor epilogue fusion, or backend-native fused path on ROCm | before/after trace shows fewer kernels or lower elementwise share |
| Nsight view is enough to identify memory-bound chains | use `torch.profiler` (Chrome trace + kernel aggregation) to recover repeated kernel chains and launch counts on AMD | trace names, source scopes, and launch counts are recoverable |
| CUDA graphs hide launch overhead but are still safe to reason about | HIP graph / compile paths can also hide chains; keep a profiling path with graphs relaxed if needed | profiling path clearly exposes the unfused chain |
| cuBLAS epilogue fusion is the only way to win | ROCm may win through Triton/Inductor fusion or source-level multi-op collapse instead | backend path and kernel count are visible in artifacts |
| one activation family implies one remedy | keep the skill generic across activation, normalization, residual, and cast chains | the trigger is based on profile share and chain shape, not model-specific names |

## Required artifacts / validation

You are not done because a chain "looks fusible." You are done when the profile
supports that conclusion and the after-state proves it helped.

### Required artifacts

Always record:

- `profiling_scope`
- `stage`: `forward` | `backward` | `mixed`
- `elementwise_share_pct`
- `top_elementwise_chain`: ordered list of the hot repeated kernels or source ops
- `before_trace_path`
- `candidate_fused_path`: source file / backend path / graph path you intend to change
- `why_not_gemm_or_attention`
- `after_trace_path` once a change exists
- `kernel_count_delta` for the targeted chain
- `next_move_grounded_in`

### Required triage tables

Produce at least these views:

| Table | Required fields |
| --- | --- |
| Chain table | `stage`, `chain`, `time share`, `launch count`, `source location` |
| Candidate fusion table | `pattern`, `current path`, `possible fused path`, `expected win` |
| Verification table | `before share`, `after share`, `kernel count delta`, `regression risk` |

### Validation cut line

Do **not** claim this skill helped until all of these are true:

1. GEMM/backend dispatch is no longer the dominant unresolved bottleneck, or you
   explicitly recorded why it was ruled out
2. the hot elementwise chain was identified from a real profile, not guessed
3. the after-state proves one of:
   - the targeted chain uses fewer launches
   - the targeted chain takes materially less GPU time
   - total step/stage time improved with no regression in correctness
4. compile coverage / graph behavior did not regress
5. if the profile instead shows grouped GEMM, recompute, topology, or
   communication as the real next bottleneck, you route away instead of forcing
   a fusion story

## References

- [`../gpu-profiling/SKILL.md`](../gpu-profiling/SKILL.md)
- [`../amd-kernel-optimization/SKILL.md`](../amd-kernel-optimization/SKILL.md)
- [`../amd-kernel-optimization/references/benchmarking-and-profiling.md`](../amd-kernel-optimization/references/benchmarking-and-profiling.md)
- [`../amd-kernel-optimization/references/triton-on-rocm.md`](../amd-kernel-optimization/references/triton-on-rocm.md)
- [`../amd-kernel-optimization/references/torch-compile-and-graphs.md`](../amd-kernel-optimization/references/torch-compile-and-graphs.md)
