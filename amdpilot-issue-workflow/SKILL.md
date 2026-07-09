---
name: amdpilot-issue-workflow
description: "Use one GitHub issue as a controlled AMDPilot end-to-end probe: launch through the dashboard, verify frozen artifacts, diagnose formulation/Docker/baseline/node/scheduler/serving/agent failures, route fixes to the correct repo or owner, and capture system-improvement learnings. Use for AMDPilot issue runs, dashboard launches, baseline verification, frozen artifacts, node/runtime problems, zombie cleanup, pre_hydration_guard, or turning a failed run into a durable fix."
metadata:
  priority: 10
  pathPatterns:
    - 'amdpilot-evals/**'
    - 'amdpilot/**'
    - 'amdpilot-dash/**'
    - '**/task.yaml'
    - '**/Dockerfile'
  bashPatterns:
    - '\bcurl\s+.*(/api/queue|/api/jobs|/api/experiments|/api/node/inventory|/api/runtime/allocations)'
    - '\bgh\s+issue\b'
  promptSignals:
    phrases:
      - "AMDPilot issue workflow"
      - "AMDPilot issue probe"
      - "submit one GitHub issue through AMDPilot"
      - "launch AMDPilot job"
      - "dashboard launch"
      - "Prepare Docker + Baseline Plan"
      - "baseline verification"
      - "frozen artifacts"
      - "total_trials"
      - "zombie_cleanup"
      - "pre_hydration_guard"
      - "node optimization"
      - "root cause routing"
    allOf:
      - [amdpilot, issue]
      - [dashboard, baseline]
      - [docker, baseline]
      - [frozen, artifacts]
      - [baseline, verification]
      - [node, amdpilot]
    anyOf:
      - "amdpilot"
      - "baseline"
      - "dashboard"
      - "docker"
      - "job"
      - "issue"
      - "node"
    minScore: 5
---

# AMDPilot Issue Probe Workflow

## Mission

Use **one GitHub issue as a controlled probe** of the live AMDPilot system. The goal is not only to run or fix that issue; it is to discover the first wrong boundary across issue spec, formulation, frozen artifacts, dashboard UX, queue/scheduler, node/runtime, serving/proxy, verifier, or agent behavior, then turn the evidence into a durable fix or follow-up.

Do **not** treat the current system as gold. UI/API/log/node disagreement is evidence of a contract or observability problem, not noise.

Load `references/failure-patterns.md` when you see worker SHA mismatch, repeated pending despite free GPUs, `zombie_cleanup`, heartbeat ambiguity, validator false-pass/false-skip behavior, or issue-specific patterns such as `openpi#3` / `FlyDSL#2`.

## Non-negotiable guardrails

1. One GitHub issue per workflow run; one branch/worktree per code change.
2. Start from the dashboard user flow unless the user explicitly asks for API-only debugging.
3. Capture evidence before changing code: issue URL, task/job id, repo, stage, node, frozen artifacts, logs, exact failure.
4. Change one hypothesis at a time, then rerun the smallest stage that can validate it.
5. Do not kill other users' jobs, containers, GPU leases, or node processes without explicit approval.
6. Do not commit, push, open a PR, or restart the daemon unless the current request authorizes that class of action.
7. Do not call a run **started** until a new task id is visible in the queue/jobs API.
8. Do not call it **running experiments** until baseline passed and `/api/jobs.total_trials > 0`.
9. Treat baseline failure as a diagnostic signal, not agent failure. First classify whether the issue is spec, artifact, metric contract, verifier, node/runtime, serving dependency, or real target behavior.
10. If switching fix targets, record why the previous target is not correct.

## Primary evidence sources

Use targeted API/log snapshots rather than full dumps:

- Dashboard UI: user-visible launch, preview, job detail, screenshots, network calls.
- `/api/queue/<task-id>`: phase narrative, worker/node/lease/failure context.
- `/api/jobs`: executor metrics: `total_trials`, `best_metric`, `metric_name`, `metric_direction`, `completed_at`.
- `/api/node/inventory` + `/api/runtime/allocations`: node health, roles, worker metadata, GPU leases; these are the source of truth for capacity questions.
- Frozen task artifacts for the exact task id: `task.yaml`, `Dockerfile`, harness and benchmark scripts.
- Daemon journal and remote-worker logs only after UI/API evidence is captured.

## State gates

Use this state machine to avoid false success reports. At each boundary, record the entry evidence and the first wrong boundary.

| State | Entry evidence | Common false positive |
|---|---|---|
| S0 Issue intake | issue URL, repo, issue number, title | issue text exists but lacks a benchmark or expected metric |
| S1 Preview/formulation | active preview matches repo/number/title/intent | stale chat tab from another issue |
| S2 Reviewed artifacts ready | generated `task.yaml`, `Dockerfile`, harness/scripts visible | `Prepare Docker + Baseline Plan` treated as launch |
| S3 Queue confirmed | UI submitted or new task id in `/api/queue`/`/api/jobs` | clicked launch but no task id |
| S4 Frozen artifacts verified | fetched frozen artifacts for that task id and key commands/image/paths match | reviewed draft differs from frozen canonicalized artifact |
| S5 Node assigned | queue/job shows node, worker, GPU ids/lease | `/api/queue.active` non-empty treated as no capacity |
| S6 Docker built | build status/log shows success with no stale failure signature | success event contains old failure text |
| S7 Baseline executed | exact baseline command ran in image/env | wrapper swallowed command failure |
| S8 Metric extracted | metric line matches `metric_pattern` | `0.00` extracted for the wrong metric |
| S9 Baseline gate passed | measured/expected/tolerance/direction all meaningful | verifier skipped invalid contract |
| S10 Trials started | `/api/jobs.total_trials > 0` | baseline passed but executor never started |
| S11 Terminal outcome | terminal completion or concrete executor/infra failure | “pipeline launched” reported as full success |

## Dashboard launch path

Dashboard URLs:

- Job list: `https://smci355-ccs-aus-n08-09.tailc77e3a.ts.net/#/job-list`
- Launch: `https://smci355-ccs-aus-n08-09.tailc77e3a.ts.net/#/launch`

Prefer Playwright MCP when available; otherwise use the available browser tools. Use a fresh browser context/page per issue so stale Chat Launch sessions cannot leak in.

Launch flow:

1. Open job list, inspect active/running jobs and visible node state.
2. Open launch page and click `+ New Task`.
3. Paste the GitHub issue URL.
4. Preview the task and verify repo, issue number, title, intent, and GPU/runtime assumptions.
5. Click `Prepare Docker + Baseline Plan` only to generate reviewed artifacts; this is not launch.
6. Launch only after artifacts are ready and the active preview still matches the requested issue.
7. Record preview screenshot, confirm network response/task id, job detail URL, and final screenshot/API snapshot.
8. Immediately fetch frozen artifacts for the new task id and verify they match the intended command/image/path changes.

If the dashboard hangs on preparation, preserve screenshot/network evidence, then use `/api/experiments/chat` to fetch/regenerate `draft_artifacts`. Still hard-check repo, issue number, title, `runtime.gpu_arch`, and artifact keys before confirm.

## API fallback rules

- `/api/experiments/submit` is equivalent to dashboard confirm only if it includes reviewed artifacts: `task.yaml`, `Dockerfile`, `test_harness.py`, and benchmark scripts. Direct submit without artifacts can fail remote workers with `pre_hydration_guard`.
- `/api/experiments/chat`: create a fresh session, but use the **server-returned** `session_id` for `reformulate`, `get_artifacts`, and `confirm`. Do not assume the client-supplied id was retained.
- `confirm` may re-run server-side artifact canonicalization. If testing a local daemon/formulator fix, deploy/restart the live daemon first; passing edited artifacts alone may be overwritten.

## Root-cause routing matrix

Classify by the first wrong boundary; do not default every failure to evals.

| Primary class | Signals | Typical route |
|---|---|---|
| Issue spec | no benchmark, no expected behavior, impossible metric | record issue insufficiency or request spec clarification |
| Formulation/intent | repo/issue/title/intent wrong, missing constraints | `amdpilot` formulation defaults/prompts/canonicalizers |
| Artifact/eval | Dockerfile/task/harness/script/metric contract wrong | `amdpilot-evals` or daemon artifact canonicalizer, depending on where generated |
| Dashboard UX/API | stale session, misleading status, preview/confirm mismatch | `amdpilot-dash` |
| Queue/scheduler | wrong node, no eligible worker despite capacity, lease policy bug | `amdpilot` scheduler/dispatcher |
| Node/runtime | only one node fails, buildx/ROCm/disk/network/env differs | node ops, node metadata, remote-worker config |
| Serving/proxy | model/proxy health, API reachability, tunnel/credential failures | serving infra; prove with health/models/completion smoke when relevant |
| Verifier/metric | no metric, skipped contract, false pass/fail, direction mismatch | verifier or baseline contract |
| Agent/model | baseline sound, trials ran, patch poor or tests fail legitimately | record model failure; do not mutate infra to hide it |
| Observability | cannot identify state, UI/API/logs conflict, missing artifacts/logs | dashboard/API/logging improvement |

## Edit target and worktree selection

Pick the edit target from evidence, not habit.

- On `smci355-ccs-aus-n08-09`, `/home/jinpan12/amdpilot` currently contains the daemon/formulation source. Common per-issue anchors include `_KNOWN_FORMULATION_DEFAULTS`, `_canonical_*_dockerfile`, and `_ensure_*_baseline_command` in `src/amdpilot/api/server.py`.
- Use `/home/jinpan12/amdpilot/evals` or `/home/jinpan12/amdpilot-evals` only when the frozen eval bundle or benchmark contract is wrong there and the repo exists/writable.
- Use `/home/jinpan12/amdpilot-dash` for dashboard-only state, UX, or read-side placement display bugs.

Before branching:

```bash
cd /home/jinpan12/amdpilot  # or the evidence-selected repo
git status --short
git fetch origin
git worktree add -b fix/amdpilot-<repo>-<issue-number>-<short> ../worktrees/amdpilot-<repo>-<issue-number> origin/main
```

If the shared source tree has unrelated WIP, do not mix it into the issue branch. Stash or leave it untouched and create a clean worktree from `origin/main`. A scoped formulation/canonicalizer fix is often small; a large diff across unrelated issue blocks usually means you mixed WIP.

## Stage diagnosis

### Launch/formulation/artifacts

Check issue URL, supported repo, detected source SHA, task description, artifact keys, runtime arch, and whether draft artifacts survived timeout/reformulate paths. Fix formulation defaults or canonicalizers when preview/frozen artifacts are wrong.

### Docker build

Check base image, ROCm version, Python/venv path, package mirrors, repo checkout, benchmark-script availability, and whether failure-only signatures appear in success logs. Fix artifact generation only when logs prove build-time setup is wrong.

### Node allocation and runtime

Do not treat a non-empty queue as a blocker by itself; AMDPilot can run concurrently across nodes/GPUs. Use node inventory plus GPU leases. For each assigned node, snapshot:

```markdown
Node:
- node_name, gpu_arch, gpu_count:
- selected_by: scheduler | manual | unknown
- worker_git_sha / required_sha / worker_status:
- leases and active containers owned by this task:
- Docker/buildx status:
- disk/network/ROCm visibility:
- serving/proxy reachability if needed:
- previous same-stage failures on this node:
```

Cross-node heuristic:

- Same frozen artifact fails on node A but passes on node B → suspect node/runtime/env.
- Different issues fail at the same stage on one node → suspect node/system.
- Same issue fails the same way on multiple nodes → suspect artifact/eval/common system path.
- Moving nodes may unblock the job, but still record the original node follow-up.

### Baseline and metric verification

Separate `no metric extracted` from extracted-but-failing metrics. For `no metric`, verify the script exists inside the image, the wrapper returns nonzero on failure, and stdout contains the exact metric line expected by `metric_pattern`.

Cross-check that `metric_pattern`, `metric_name`, `metric_direction`, `expected_metric`, tolerance, and issue intent describe the same metric. A lower-is-better latency contract over a generic `SCORE` line is a false-pass shape. A passing baseline is meaningful only if the measured metric is meaningful.

Check Python environment consistency: if dependencies were installed into `.venv`, the baseline command should not accidentally run `/usr/bin/python3` without that environment.

### Agent run and final verification

If baseline is sound and `/api/jobs.total_trials > 0`, inspect the agent patch, final verification logs, and test output. If the model simply failed a sound task, record it as agent/model failure rather than changing infra or weakening the harness.

### Zombie cleanup and lease loss

For `zombie_cleanup`, collect `failure_stage`, `node_name`, `lease_heartbeat_age_s`, `node_heartbeat_age_s`, remote-worker logs, container state, supervisor/nudge/executor reachability, and `total_trials`. Treat as remote-worker/container lifecycle infrastructure unless logs prove issue-specific eval failure.

Read `references/failure-patterns.md` before relaunching or cleaning resources.

## Daemon operations

Restart `amdpilot serve` only when needed and after checking `/api/queue` has no active/pending job or after user approval. Preserve Docker/buildx environment:

```bash
HOME=/home/jinpan12
DOCKER_CONFIG=/home/jinpan12/.docker
PATH=/home/jinpan12/.docker/cli-plugins:$PATH
```

After restart, verify `/health`, `Queue runner started`, `docker buildx version`, worker eligibility, and that the next frozen artifacts reflect the deployed code rather than stale daemon output.

## Intervention loop

For each attempt:

1. State the hypothesis and first wrong boundary.
2. Make the smallest relevant change.
3. Rerun the smallest reproducible stage.
4. Record result with job id, log path, artifact diff, or node snapshot.
5. Relaunch from dashboard only when the fix affects dashboard-visible behavior or full orchestration.
6. Before relaunch, record whether the prior attempt reached queued, Docker built, baseline passed, and trials started.
7. After relaunch, verify frozen artifacts plus queue/jobs state before reporting success.

Stop and ask when blocked by login, permissions, destructive cleanup, missing secrets, unavailable capacity, or ambiguous ownership.

## Progress update format

```markdown
Issue: <repo>#<issue-number> - <title>
Worktree: <path or none>
Branch: <branch or none>
Dashboard job: <task/job id or not launched yet>
Current state: <S0..S11 and label>
First wrong boundary: <state/classification or unknown>
Run state: <queued? docker built? baseline passed? total_trials?>
Node: <node, gpu ids, worker status if assigned>
Frozen artifacts: <verified? key command/image/path checks>
Evidence: <short log/API/screenshot observation; include best_metric, metric_name, metric_direction when available>
Next step: <one concrete action>
```

## End-of-run learning report

Every run should leave a learning, even without a PR:

```markdown
## AMDPilot Issue Probe Result
Issue: <url> (<repo>#<number> - <title>)
Run: <task/job id, node, reached_state, total_trials, terminal_status>
First wrong boundary: <expected vs observed, with evidence>
Root cause classification: <primary class, confidence, why>
Minimal repro: <UI/API/command/log/artifact reference>
Fix route: <repo/owner/ops follow-up/none, and why>
System improvement candidate: <dashboard | daemon | scheduler | eval template | node/runtime | serving | observability>
Skill delta: <new rule/gotcha/check to add, or none>
```

## PR or follow-up readiness

Before opening a PR:

- The diff is scoped to the single issue or single system bug.
- The failing stage has been rerun successfully, or the remaining blocker is documented.
- Job ids, frozen artifact checks, and logs are cited.
- The PR says whether it fixes eval/workflow artifacts, daemon/canonicalizer logic, dashboard UX, scheduler/dispatcher behavior, or node/runtime config.
- No unrelated worktree changes are included.

Suggested title: `fix(amdpilot): support <repo> issue <number>` or `fix(amdpilot-evals): support <repo> issue <number>` depending on evidence.

## Related skills

- `amdpilot-dispatcher`: daemon startup, submission methods, queue status, dispatcher troubleshooting.
- `amd-kernel-infra-fix`: AMD SFT Kernel Pipeline Docker/test/validator infrastructure failures.
