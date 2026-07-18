# Global number tracker

One row per bench-affecting commit — the single place to see where performance and
equivalence stand over time. Cold numbers first (the honest end-to-end story), warm second.
Append a row with `native/track-numbers.sh` after every change touching the batched GPU
path; never edit old rows (history is the point). Context/analysis for each row lives in
`codegen/gpu_benchmark.md` under the matching date; capability state in
`docs/gpu-vs-cpu-capabilities.md`.

Production bench = `GpuVsCpuProductionSaBenchmarkTest`: case9241pegase, 500 matched
contingencies, RTE JSON parameters, CPU arm 4 threads, sparse forced.
Fixed-iter bench = `native/cuda/run_bench.sh`: same case/contingencies, K=4, tol=0,
UBATCH chunk 500 (the gpusim2grid-comparable configuration).
"viol diff" = onlyCPU/onlyGPU threshold-edge re-reports (ctg = contingencies affected);
statuses have been identical on every run since 2026-07-17.

| Date | Commit | Prod COLD ms/ctg | Prod warm ms/ctg | CPU-4t ms/ctg | Warm ×CPU | Viol diff (cpu/gpu/ctg) | FxIter COLD E2E ms/ctg | FxIter warm ms/scn | @1e-6 | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-07-17 | (perf day) | — | 24.4–25.2 | 27.0 | ~×1.10 | 15/7/6 | 11.56 | 4.31 | 499/500 | straggler exit + device norms + compaction landed; violation saga closed |
| 2026-07-18 | cc41597 | 25.8 | 23.1 | 25.8 | ×1.12 | 15/7/6 | 11.71 | 4.23 | 499/500 | full-row tap override + PST/shunt actions |
| 2026-07-18 | a082092 | 26.7 | 23.6 | 24.6 | ×1.04 | 15/7/6 | — | 4.27 | 499/500 | bare-run = production defaults (harness fix) |
| 2026-07-18 | 955017a | 30.4–30.6 | 26.9–27.8 | 23.1–23.5 | ×0.85 | 13/7/5 | — | 4.25 | 499/500 | outer-loop param wiring: RTE maxOuterLoopIterations=30 on device — diff 22→20, warm +1.5 s |
