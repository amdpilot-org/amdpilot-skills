---
name: slime-repo-navigation
description: >
  Repo-native navigation guide for first-time work in SLIME. Use this skill to
  recover the right docs, examples, and source hotspots before editing code,
  especially for rollout, reward, eval, and sample-contract tasks.
---

# SLIME Repo Navigation

## When to use

- The task is in SLIME and the first blocker is "where does this live?"
- The agent needs the shortest path to the right docs, example, CLI surface, or
  source hotspot before making changes.
- A reviewer or supervisor needs a stable `task -> file` routing table for
  rollout, reward, eval, or sample-contract work.
- The task spans SLIME workflow code, but the first step is still orientation
  rather than contract reasoning or topology diagnosis.

## When NOT to use

- The issue is about workflow, rollout, reward, or trajectory semantics rather
  than file ownership. Use `training-workflow-contracts`.
- The issue is serving topology, gateway state, P/D disaggregation, KV
  transfer, or compile coverage. Use `serving-architecture`.
- The first blocker is ROCm environment, `torch.compile`, HIP graph, or AMD
  correctness debugging. Use `amd-rocm-porting`.
- The sample contract is already located and the next question is stage-aware
  trace analysis or benchmark evidence. Use `gpu-profiling`.
- The task is low-level kernel or backend optimization rather than repo
  orientation. Use `amd-kernel-optimization`.

## Decision tree / routing questions

1. Are you trying to locate the owning file, doc, or example before editing?
   If yes, start here.
2. Is the question about public CLI flags or argument meaning?
   Read `docs/en/get_started/usage.md` first, then `slime/utils/arguments.py`.
3. Is the question about the real training-sample schema?
   Read `slime/utils/types.py` before touching rollout or reward code.
4. Is the question about rollout orchestration or SGLang interaction shape?
   Read `slime/rollout/sglang_rollout.py`.
5. Is the task multi-turn tool use?
   Start with `examples/search-r1/` before backend internals.
6. Is the user actually asking why train/eval semantics or reward wiring are
   wrong?
   Escalate to `training-workflow-contracts`.
7. Is the task about live serving, routing, or compile coverage?
   Escalate to `serving-architecture`.

### Task -> file routing table

| Task shape | Open first | Then |
| --- | --- | --- |
| first training run | `docs/en/get_started/quick_start.md` | `docs/en/get_started/usage.md` |
| CLI flags or config meaning | `docs/en/get_started/usage.md` | `slime/utils/arguments.py` |
| custom rollout or reward | `docs/en/get_started/customization.md` | `slime/rollout/sglang_rollout.py` plus the matching reward or rollout module |
| sample contract confusion | `slime/utils/types.py` | `training-workflow-contracts` |
| multi-turn tool use | `examples/search-r1/` | closest workflow code that consumes its sample fields |
| eval dataset wiring | eval config helper or docs | `slime/utils/eval_config.py` if present in the target checkout |

### Minimum keep

- task-to-file routing table
- first-read order
- workflow hotspot list
- `examples/search-r1/` entrypoints

### Explicit drop

- reward-contract explanation
- serving diagnosis
- profiler command cookbook
- executor boilerplate

### Routing canaries

| Canary prompt | Chosen skill | Rejected skill | First question | Required artifacts | Expected output |
| --- | --- | --- | --- | --- | --- |
| "I am new to SLIME and need the shortest path to search-r1, sample contract, and the reward entrypoints." | `slime-repo-navigation` | `training-workflow-contracts` | "What are the first files, docs, and examples I should read?" | target task summary, likely subsystem, example or CLI clue | `problem_class + chosen_skill + why_not_others + required_artifacts + escalate_to` |
| "I need to add a custom eval or rollout path in SLIME and want the first files and docs to inspect before touching backend code." | `slime-repo-navigation` | `training-workflow-contracts` | "Which task-to-file route gets me to the right hotspot fastest?" | task description, current example/config, suspected file family | `problem_class + chosen_skill + why_not_others + required_artifacts + escalate_to` |

## AMD translation

This skill imports repo-navigation patterns, not CUDA-only execution recipes. On
AMD, keep the navigation surface repo-native and treat NVIDIA-specific examples
as hints about ownership, not as launch templates to copy verbatim.

| NVIDIA/CUDA assumption | AMD/ROCm equivalent | Validate with |
| --- | --- | --- |
| A CUDA/H100 example is the default first read for every task. | Choose the closest SLIME example by workflow shape, then verify AMD feasibility separately with `amd-rocm-porting` or `env-probe`. | first-read bundle plus explicit escalation note when backend validation is needed |
| Backend-specific scripts are the source of truth for CLI behavior. | On AMD too, the public docs and `slime/utils/arguments.py` stay the ownership source for argument meaning. | opened file list containing both `usage.md` and `slime/utils/arguments.py` |
| Nsight/profiler docs are a good onboarding surface. | Do not start from profiling docs for navigation on AMD; first recover file ownership, then escalate to `gpu-profiling` only if evidence collection is the next step. | task-to-file route plus escalation note if profiling is actually needed |
| Megatron/FSDP examples imply CUDA defaults should be copied directly. | Treat them as topology or training-shape references only; ROCm-specific launch safety belongs to `amd-rocm-porting`. | chosen example family plus separate ROCm validation owner |

## Required artifacts / validation

Every routing decision using this skill should emit:

- `problem_class + chosen_skill + why_not_others + required_artifacts + escalate_to`
- target task sentence
- first-read file list
- `task -> file` routing row
- closest example family
- one escalation target if orientation is no longer the main blocker

Validation checklist:

- The first-read order reaches a public doc before backend internals.
- The response names actual repo paths instead of generic "look around" prose.
- The sample contract path is called out explicitly when rollout or reward code
  is in scope.
- If the task is really about semantics, topology, or ROCm correctness, the
  response escalates instead of pretending navigation alone will solve it.

Minimal dry-run:

- `problem_class=repo entrypoint discovery | chosen_skill=slime-repo-navigation | why_not_others=the first blocker is ownership/path discovery, not workflow semantics | required_artifacts=target task summary, likely subsystem, example or CLI clue | escalate_to=training-workflow-contracts if the contract stays ambiguous after the first reads`
- `problem_class=task-to-file routing for eval or rollout edits | chosen_skill=slime-repo-navigation | why_not_others=the task still needs concrete docs/examples before backend edits begin | required_artifacts=task description, current example/config, suspected file family | escalate_to=serving-architecture only if the issue turns into live serving topology`

## References

- Upstream source family: `knowledge/rl/slime-repo-navigation.md`
- Upstream source family: `.claude/skills/03-design/slime-workflows.md`
- Adjacent local skills: `training-workflow-contracts`, `serving-architecture`,
  `amd-rocm-porting`, `gpu-profiling`
