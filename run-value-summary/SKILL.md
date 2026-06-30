---
name: run-value-summary
description: >
  Emit a structured RUN_SUMMARY v1 block at the end of any scored run so the
  reviewer (human or LLM judge) can decide whether to invest more on this
  lineage. Use when a run produced measurable outcomes (benchmark, accuracy,
  speedup, rocprof). Skip it when the run is a trivial bugfix or port-that-
  just-compiles with no scoring signal.
---

# Run-Value Summary

## When to use

Emit a `RUN_SUMMARY v1` block at the end of a run when ANY of these apply:

- The run produced a benchmark number (latency, throughput, TFLOPS).
- The run produced an accuracy / quality number.
- The run produced a rocprof / nsys profile.
- The run claims a speedup over a baseline.

## Skip it when

Skip it when the run is one of these — there's no scoring signal to summarize:

- A bugfix that just makes a crash go away (no perf/quality measurement).
- A port that just makes a kernel compile (no benchmark run).
- A docs-only or config-only change.
- A dry-run / smoke test that proved reachability but produced no numbers.

Forcing a RUN_SUMMARY onto every task regardless of type was the cycle-80p
bug — it wasted agent turns on tasks that had nothing to measure. The skill
exists so the agent applies it *when relevant*, not as a blanket requirement.

## The block contract

```
RUN_SUMMARY v1
  metric: <name>           # e.g. "latency_ms_p50", "throughput_tok_s", "accuracy@1"
  value: <number>          # the measured value
  baseline: <number|na>    # the comparison baseline, or "na" if no baseline
  speedup: <number|na>     # value / baseline (>=1.0 is better), or "na"
  n_samples: <int>         # how many measurement iterations (must be >= 10)
  warmup: <int>            # how many warmup iterations (must be >= 3)
  std_pct: <number>        # std as % of mean; flag if > 10%
  next_bottleneck: <text>  # one sentence: what's the next thing to optimize
```

## Rules

1. **`next_bottleneck` is the load-bearing field.** The reviewer reads it to
   decide whether to invest more on this lineage. If you can't name the next
   bottleneck, you didn't profile — say so explicitly ("did not profile;
   recommend rocprof before next iteration").

2. **Never invent numbers.** If a field can't be measured from this run, use
   `na` rather than a guess. A fabricated `speedup: 2.3x` is worse than
   `speedup: na`.

3. **`std_pct > 10` is a red flag.** State it honestly — it usually means
   graph breaks, recompilation, or contention. Don't hide it by reporting
   only the best iteration.

4. **`n_samples < 10` invalidates the summary.** Re-run with more iterations
   rather than reporting a 3-sample mean as if it were stable.

## Example

```
RUN_SUMMARY v1
  metric: latency_ms_p50
  value: 42.1
  baseline: 58.3
  speedup: 1.38
  n_samples: 50
  warmup: 5
  std_pct: 2.1
  next_bottleneck: GEMM still 60% of kernel time; try torch.compile inductor max-autotune.
```
