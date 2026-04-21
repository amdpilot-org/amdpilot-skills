---
name: training-workflow-contracts
description: >
  Route RL training tasks that are really about workflow, rollout, reward, or
  trajectory contracts rather than repo navigation, serving topology, or ROCm
  implementation details. Use this skill when train/eval branches diverge,
  multi-turn trajectories lose token fidelity, or reward wiring is ambiguous.
---

# Training Workflow Contracts

## When to use

- A new workflow, rollout function, or reward function is being added and the
  contract between them is still implicit.
- Train and eval appear to use different assumptions about sample fields,
  reward keys, or rejection behavior.
- A multi-turn or tool-using pipeline preserves final chat text but not the
  token trajectory needed for RL training.
- The supervisor needs to decide whether a failure belongs to workflow
  semantics instead of repo navigation, serving topology, or ROCm debugging.

## When NOT to use

- The agent first needs to find the right SLIME docs, example, or source
  hotspot. Use `slime-repo-navigation`.
- The problem is prefill/decode split, KV transfer, sticky routing, gateway
  state, or compile coverage inside serving. Use `serving-architecture`.
- The issue is ROCm correctness, `torch.compile` behavior on AMD, HIP graph
  capture, allocator setup, or environment breakage. Use `amd-rocm-porting`.
- The workflow contract is already clear and the remaining question is stage
  diagnosis, trace collection, or benchmark evidence. Use `gpu-profiling`.
- The hotspot is already known to be a specific kernel or backend path. Use
  `amd-kernel-optimization`.

## Decision tree / routing questions

1. What is the trainable object?
   It should be an explicit token trajectory, not reconstructed final text.
2. Where does one episode end?
   The workflow should define prompt preparation, generation, reward
   invocation, output schema, and rejection behavior for one episode.
3. Are `train` and `eval` branches actually sharing the same sample contract?
   If not, fix the contract split before changing backend code.
4. Does a multi-turn path preserve `prompt_ids`, `response_ids`, and
   `response_mask` directly?
   If not, this skill owns the problem before any profiler or serving work.
5. Is the agent really asking "where in SLIME does this live?"
   If yes, route to `slime-repo-navigation`.
6. Is the failure really about request routing, gateway state, or compile
   coverage inside a live serving stack?
   If yes, route to `serving-architecture`.
7. Is the first blocker a ROCm crash, compile hang, or HIP graph error before
   the workflow contract is even observable?
   If yes, route to `amd-rocm-porting`.

### Minimum keep

- `workflow -> rollout -> reward` contract questions
- multi-turn trajectory shape
- train vs eval branch split
- reward-key and rejection-semantics check

### Explicit drop

- repo navigation prose
- serving topology diagnosis
- profiler procedure
- kernel tuning loop

### Routing canaries

| Canary prompt | Chosen skill | Rejected skill | First question | Required artifacts | Expected output |
| --- | --- | --- | --- | --- | --- |
| "I want to add a new reward workflow in SLIME, but I am not sure whether the change belongs in workflow, rollout, or reward." | `training-workflow-contracts` | `slime-repo-navigation` | "Which layer owns the contract change first: workflow, rollout, or reward?" | workflow entrypoint, rollout interface, reward interface, train/eval config | `problem_class + chosen_skill + why_not_others + required_artifacts + escalate_to` |
| "A multi-turn trajectory works in eval but breaks in training; I need to know which contract is wrong before editing code." | `training-workflow-contracts` | `slime-repo-navigation` | "Is the failure caused by trajectory shape or train-vs-eval branch ownership?" | sample trajectory, train path, eval path, failure symptom | `problem_class + chosen_skill + why_not_others + required_artifacts + escalate_to` |

## AMD translation

Imported workflow guidance often assumes CUDA-centric serving or profiler
surfaces. On AMD, keep the workflow contract above backend-specific execution
details and validate the contract with explicit artifacts instead of relying on
NVIDIA-only tooling.

| NVIDIA/CUDA assumption | AMD/ROCm equivalent | Validate with |
| --- | --- | --- |
| The serving edge or executor can implicitly define the training schema. | Treat backend choice as replaceable; the workflow must still own explicit trajectory and reward fields on ROCm. | one train sample dump, one eval sample dump, explicit reward key, rejection status field |
| Multi-turn agent loops can reconstruct the trainable trajectory from final messages. | Preserve token-level trajectory fields directly on AMD too; do not let tool round-trips rewrite the training object. | saved `prompt_ids`, `response_ids`, `response_mask`, and any tool metadata for one trajectory |
| Throughput or profiler data is the first debugging lens. | On AMD, first prove the workflow contract is correct; only then escalate to `gpu-profiling` or `amd-kernel-optimization`. | contract checklist completed before any trace request |
| NCCL/CUDA execution details belong inside the workflow layer. | Keep workflow semantics separate from ROCm executor and communicator details; backend setup belongs to `amd-rocm-porting` or `serving-architecture`. | workflow-level schema plus explicit escalation note when the failure moves below the contract layer |

## Required artifacts / validation

Every routing decision using this skill should emit:

- `problem_class + chosen_skill + why_not_others + required_artifacts + escalate_to`
- workflow entrypoint or class path
- rollout function path
- reward function path
- train/eval branch description
- one accepted trajectory artifact
- one rejected or edge-case trajectory artifact
- explicit list of required sample fields and reward keys

Validation checklist:

- One episode contract is explicit from prompt preparation through reward.
- The trainable object is token-level, not reconstructed from final chat text.
- Train and eval branches either share one schema or clearly document why not.
- Rejection behavior is explicit instead of silent drop-on-floor behavior.
- If the next blocker is not a contract question anymore, the draft escalates to
  exactly one of `slime-repo-navigation`, `serving-architecture`,
  `amd-rocm-porting`, or `gpu-profiling`.

Minimal dry-run:

- `problem_class=workflow contract ownership | chosen_skill=training-workflow-contracts | why_not_others=the first ambiguity is workflow vs rollout vs reward, not repo navigation or profiling | required_artifacts=workflow entrypoint, rollout interface, reward interface, train/eval config | escalate_to=slime-repo-navigation if file ownership is still unclear`
- `problem_class=train-vs-eval trajectory split | chosen_skill=training-workflow-contracts | why_not_others=the failure is trajectory semantics before serving or profiling work | required_artifacts=sample trajectory, train path, eval path, failure symptom | escalate_to=gpu-profiling only after the contract is explicit and a stage bottleneck remains`

## References

- Upstream source family: `knowledge/rl/workflow-contracts.md`
- Upstream source family: `.claude/skills/03-design/add-workflow.md`
- Adjacent local skills: `slime-repo-navigation`, `serving-architecture`,
  `amd-rocm-porting`, `gpu-profiling`
