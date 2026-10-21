# tools/

Scripts and helpers that are not part of the build. Building the native library stays in `native/`
(`build-gpu.sh`, and `setup-gpu-deps.sh` for a fresh machine); every script here sources `native/local-env.sh`
for the machine paths.

## Runners

Tests and benchmarks of the built extension. Each runs from any directory (it changes to the repository root).

| Script | What it does |
|---|---|
| `run_gpu_tests.sh` | builds libolfgpu, then runs `GpuFullNewtonTest` and `GpuLoadFlowTest` with the library loaded from the classpath (the packaging path) |
| `run_bench.sh` | the fixed-iteration batched N-1 benchmark (`GpuBatchedN1FixedIterTest`, case9241pegase, 500 contingencies, K=4); `--no-build` skips the native build |
| `sweep_ordering.sh` | the benchmark (or, with `suite`, the whole test suite) once per cuDSS reordering algorithm |
| `track-numbers.sh` | the production SA benchmark and `run_bench.sh`, then one row appended to `docs/numbers.md` |

## Investigation helpers

Temporary: delete one once the investigation it served is over.

| Helper | What it does |
|---|---|
| `cudss-swap.sh` | `source tools/cudss-swap.sh 0.7.1\|0.8.0`: builds the native stack once per cuDSS version and swaps between them (A/B tests) |
| `ol_match.py`, `ol_stats.py`, `ol_summary.py` | outer-loop statistics from CPU/GPU logs (`CPU_OL_STATS` lines), 2026-10-06/07 investigation |
| `rl_reasons.py` | why the ReactiveLimits outer loop costs what it does (`CPU_RL_CHECK` lines) |
| `pypowsybl/` | comparisons against the official pypowsybl (load flow, security analysis, a reactive-limits oscillation case) |
