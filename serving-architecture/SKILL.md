---
name: serving-architecture
description: >
  Topology and routing guide for inference stacks that cross router, gateway,
  prefill, decode, KV transfer, and compile-coverage boundaries. Use this skill
  when a serving regression is really about request flow or graph coverage
  rather than ROCm bring-up or low-level profiling.
---

# Serving Architecture

## When to use

- A regression appears after enabling prefill/decode disaggregation, KV-cache
  transfer, sticky routing, or a gateway-mediated RL serving loop.
- The main question is how requests move through router, gateway, session
  store, prefill worker, decode worker, and reward writeback.
- A custom op or wrapper is fast in isolation, but end-to-end serving regresses
  because compile coverage may have changed.
- The supervisor needs to decide whether a failure belongs to topology and
  routing rather than ROCm implementation details or kernel-level hotspots.

## When NOT to use

- The first blocker is a ROCm compile crash, HIP graph failure, allocator
  problem, or AMD environment breakage. Use `amd-rocm-porting`.
- The main need is stage-aware trace collection, two-phase profiling, or
  benchmark evidence. Use `gpu-profiling`.
- The hotspot is already known to be a specific kernel/backend path that needs
  source-level optimization. Use `amd-kernel-optimization`.
- The task is first-time SLIME path discovery rather than serving behavior.
  Use `slime-repo-navigation`.
- The real problem is rollout/reward/trajectory semantics in RL training. Use
  `training-workflow-contracts`.

## Decision tree / routing questions

1. What is the live request path?
   Recover it explicitly, for example:
   `client -> gateway -> session store -> backend -> reward writeback`
   or
   `router -> prefill -> KV transfer -> decode`.
2. Does the failure start only after P/D split, connector enablement, or
   sticky-session logic?
   If yes, this skill owns first-pass diagnosis.
3. Did the regression appear after integrating a custom op or wrapper into the
   serving path?
   If yes, ask whether graph coverage changed before touching the kernel again.
4. Is the first observable failure a ROCm compile crash, dtype/runtime error,
   illegal capture, or HIP-specific correctness issue?
   If yes, route to `amd-rocm-porting`.
5. Is the topology already stable, but the slow stage or hotspot is unclear?
   If yes, route to `gpu-profiling`.
6. Is the user asking for file ownership in SLIME or for rollout semantics?
   If yes, route to `slime-repo-navigation` or
   `training-workflow-contracts` instead.

### Minimum keep

- P/D plus KV transfer decision nodes
- sticky routing checks
- gateway/session ownership questions
- compile coverage checks

### Explicit drop

- ROCm porting checklist
- low-level profiling commands
- kernel tuning loop
- repo tour prose

### Routing canaries

| Canary prompt | Chosen skill | Rejected skill | First question | Required artifacts | Expected output |
| --- | --- | --- | --- | --- | --- |
| "Prefill is saturated, decode is underutilized, and I suspect KV transfer or sticky routing." | `serving-architecture` | `gpu-profiling` | "Is the first classification topology/routing, not kernel hotspot ownership?" | worker-role topology, request path, serving metrics, stage labels | `problem_class + chosen_skill + why_not_others + required_artifacts + escalate_to` |
| "A custom op is fast in isolation, but end-to-end serving regressed after integration; I need to know if compile coverage or gateway/session shape is the real issue." | `serving-architecture` | `amd-rocm-porting` | "Is the regression caused by compile-region or serving-shape ownership before ROCm checklist work?" | graph-break logs, serving traces, wrapper path, request flow | `problem_class + chosen_skill + why_not_others + required_artifacts + escalate_to` |

## AMD translation

This skill imports serving-topology knowledge, not CUDA-only bring-up or
profiler cookbooks. On AMD, keep topology and routing reasoning separate from
ROCm implementation details so the agent does not fall into "porting checklist"
mode for every serving regression.

| NVIDIA/CUDA assumption | AMD/ROCm equivalent | Validate with |
| --- | --- | --- |
| CUDA Graph, NCCL, or NVLink details define the architecture layer. | Treat RCCL, HIP graph, and transport choices as implementation details below the routing contract. | request-path sketch, component ownership, escalation note when failure drops to `amd-rocm-porting` |
| Nsight or kernel traces are the first proof of a serving regression. | First prove the routing and compile-coverage contract with request IDs, session IDs, graph-break evidence, and connector stats; profile only after topology is pinned down. | route logs, session binding evidence, compile-coverage report, connector/buffer evidence |
| P/D examples assume producer and consumer layouts already match. | On AMD too, require explicit agreement on TP/PP shape, KV layout, and buffer ownership before trusting the split. | prefill/decode topology config plus KV-transfer completion or mismatch evidence |
| A faster kernel benchmark guarantees a faster serving path. | On AMD, ask whether the optimized op actually stays inside the compiled serving region before any kernel-tuning loop begins. | operator benchmark, end-to-end benchmark, graph-break report, profiler or route evidence showing whether the path is compiled |

## Required artifacts / validation

Every routing decision using this skill should emit:

- `problem_class + chosen_skill + why_not_others + required_artifacts + escalate_to`
- explicit request-path or topology sketch
- component ownership list for router, gateway, session store, prefill, decode,
  and reward writeback as applicable
- sticky-routing or session-binding evidence
- connector or buffer configuration if KV transfer is involved
- compile-coverage artifact if a custom op or wrapper is implicated
- explicit escalation target if the problem drops to `amd-rocm-porting` or
  `gpu-profiling`

Validation checklist:

- The response names the architectural boundary that owns the bug.
- P/D split questions do not silently turn into profiler-command advice.
- Compile-coverage regressions ask for graph-break evidence before kernel work.
- ROCm bring-up failures are pushed out to `amd-rocm-porting` early.
- Kernel hotspot investigation is pushed out to `gpu-profiling` once topology is
  no longer the ambiguous part.

Minimal dry-run:

- `problem_class=topology and routing ownership | chosen_skill=serving-architecture | why_not_others=the first ambiguity is P/D plus KV-transfer behavior, not kernel hotspot ownership | required_artifacts=worker-role topology, request path, serving metrics, stage labels | escalate_to=gpu-profiling after topology is stable and the slow stage is still unclear`
- `problem_class=compile-coverage vs serving-shape regression | chosen_skill=serving-architecture | why_not_others=the regression is integration-level before ROCm bring-up or kernel tuning work | required_artifacts=graph-break logs, serving traces, wrapper path, request flow | escalate_to=amd-rocm-porting only if the serving path fails before topology evidence exists`

## References

- Upstream source family: `knowledge/serving/disaggregated-serving.md`
- Upstream source family: `knowledge/serving/gateway-serving-patterns.md`
- Upstream source family: `knowledge/serving/torch-compile-coverage.md`
- Adjacent local skills: `amd-rocm-porting`, `gpu-profiling`,
  `training-workflow-contracts`, `slime-repo-navigation`
