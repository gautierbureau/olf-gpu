---
name: gpu-bench-compare
description: Apples-to-apples GPU N-1 comparison vs gpusim2grid (same 500 matched contingencies, same cuDSS 0.8.0, batch 500, K=4, FP64) — run both sides, build the cold-E2E table
---

Run the apples-to-apples N-1 comparison: OURS vs gpusim2grid (Benjamin Donnot's
batched GPU powerflow, ex-powerflow_gpu). Same physics on both sides: cuDSS 0.8.0,
UBATCH, FP64, K=4, batch/chunk 500, the SAME 500 matched non-islanding contingencies
on case9241pegase, same GPU.

**1. Ours:**
```bash
cd ~/Projects/powsybl/powsybl-open-loadflow-gpu && native/cuda/run_bench.sh
```
Take `[E2E K=4] base CPU loadflow B + cold GPU SA S = T total | N ms/ctg`
→ ours cold E2E = T (CUDA init sits inside S), warm = the `K=4 :` line.

**2. Theirs (gpusim2grid):**
```bash
cd ~/Projects/powsybl/cuda/gpusim2grid/benchmarks
export CUDA_HOME=$HOME/Projects/Software/cuda-toolkit
export LD_LIBRARY_PATH=$HOME/Projects/powsybl/cuda/cudss/lib:$CUDA_HOME/lib64
../.venv/bin/python matched_contingency_bench.py --grid_name case9241pegase \
  --cont_file ~/Projects/powsybl/cuda/matched_contingencies/matched_pflow_idx.txt \
  --batch_size 500 --nb_iter 4 --strategy direct_refactor_every
```
Take `one-time build: base_case BC ... | chunks C` and `total TT ms`
→ theirs cold E2E = BC + TT (their GPU base + CUDA init live in BC), warm ≈ chunks.
(If the venv is missing, rebuild per `codegen/gpu_benchmark.md` / the
gpu-n1-benchmark-howto memory: venv + vendored eigen + CMAKE_ARGS with
`cudss_DIR=$HOME/Projects/powsybl/cuda/cudss/lib/cmake/cudss`; needs grid2op.)

**3. Compare — END-TO-END TOTALS ONLY** (CUDA init lands in different phases on the
two sides; phase-by-phase comparisons lie). Build the table: cold E2E ms/ctg, warm
ms/ctg, convergence (n/500 @1e-6, med ‖F‖∞) — and diff against the recorded
2026-07-13 rerun in `codegen/gpu_benchmark.md`:
ours 11.41 vs theirs 8.81 cold (1.30×); warm 4.87 vs 3.76–4.29.
Measured contributors on our side: base placement (fixed — the SA base runs on the
GPU solver since a94135a) and host extraction; the residual chunk-speed difference is
OUR optimization backlog — do NOT attribute it to their formulation or matrix size
(they run a different formulation by design; we keep OLF's — 1:1 parity not required).
A ratio drifting meaningfully above the recorded one means WE regressed; below means
we closed ground (report which lever moved).
