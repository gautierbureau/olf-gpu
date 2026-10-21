# tools/

Temporary helpers: not part of the build, not called by it, and not needed to build, test or benchmark the
extension (those live in `native/`: `build-gpu.sh`, `run_gpu_tests.sh`, `run_bench.sh`, `sweep_ordering.sh`,
`track-numbers.sh`, `setup-gpu-deps.sh`). Delete a helper here once the investigation it served is over.

| Helper | What it does |
|---|---|
| `cudss-swap.sh` | `source tools/cudss-swap.sh 0.7.1\|0.8.0`: builds the native stack once per cuDSS version and swaps between them (A/B tests) |
| `ol_match.py`, `ol_stats.py`, `ol_summary.py` | outer-loop statistics from CPU/GPU logs (`CPU_OL_STATS` lines), 2026-10-06/07 investigation |
| `rl_reasons.py` | why the ReactiveLimits outer loop costs what it does (`CPU_RL_CHECK` lines) |
| `pypowsybl/` | comparisons against the official pypowsybl (load flow, security analysis, a reactive-limits oscillation case) |
