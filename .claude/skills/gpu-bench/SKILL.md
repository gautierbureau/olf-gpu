---
name: gpu-bench
description: Run OUR GPU batched N-1 benchmark (pegase9241, 500 matched contingencies, K=4) and report warm + TRUE-cold numbers against the recorded baseline
---

Run the repo's N-1 throughput benchmark and report the results.

```bash
cd ~/Projects/powsybl/powsybl-open-loadflow-gpu
native/cuda/run_bench.sh              # builds libolfgpu vs the pinned cuDSS first
# native/cuda/run_bench.sh --no-build # if the native lib is already current
```

Defaults (in the script): cuDSS 0.8.0 (`CUDSS_ROOT=~/Projects/powsybl/cuda/cudss`),
UBATCH (the only batch mode), `OLF_GPU_MAXBATCH=500` (the bench test's chunk-size hook), K=4, FP64, the 500 matched
non-islanding contingencies from `~/Projects/powsybl/cuda/matched_contingencies/`.

Report BOTH lines and compare to the recorded baseline
(`codegen/gpu_benchmark.md`, 2026-07-13 section):
- `[E2E K=4] ... | N ms/ctg` = **TRUE cold end-to-end** (CPU base LF + ONE cold GPU
  SA pass; the ~2 s one-time CUDA init is inside the GPU SA term). Baseline: **11.41 ms/ctg**.
- `K=4 : ... ms/scenario` = **warm steady-state** contingency solve. Baseline:
  **4.87 ms/scn** (205 scn/s), 499/500 @ 1e-6, med ‖F‖∞ ~3.6e-12.

A warm regression > ~10% or convergence below 499/500 is a finding — bisect against
the last change to native/, the kernels, or the cuDSS pin (see
`source native/cudss-swap.sh 0.7.1|0.8.0` for version A/B; chunk size is decisive on
0.8.0 — never bench it at MAXBATCH=32).
