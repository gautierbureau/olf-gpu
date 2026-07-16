# test6 GPU batched N-1 — throughput benchmark

Standalone throughput of the test6 GPU batched security-analysis solver on a
single-branch N-1 workload. Harness: `codegen/bench/` (`GpuBatchedN1FixedIterTest`).
Recorded 2026-06-22.

- **GPU:** NVIDIA RTX A1000 6 GB Laptop (sm_86), CUDA 13.3.
- **Grid:** `case9241pegase` (9241 buses), single-branch N-1, non-islanding subset
  (islanding outages are pre-qualified out and routed to the CPU path by
  `GpuBatchContingencyValidator`).
- **Config:** batched cuDSS, chunk/batch size 32, cuDSS 0.7.1. `-Dolf.gpu.batchMode`
  selects UBATCH (1) or the per-scenario path (0); both compiled in.

## End-to-end throughput (warm, K=4, ms/scenario)

From the converged base-case AC solution (the production hot-start), the batched
solver reaches `‖F‖∞ < 1e-6` at **K = 4** Newton iterations:

|              | cold (= solve + analysis) | of which analysis |
|--------------|---------------------------|-------------------|
| **UBATCH**    | **8.52** (8.22 + 0.30)    | 0.30              |
| **batch API** | **9.23** (6.31 + 2.92)    | 2.92              |

- "cold" = a single SA pass with cuDSS analysis included, CUDA process init excluded
  (absorbed by a throwaway warm-up solve). "solve" = the per-iteration device Newton
  loop only.
- **UBATCH cold ≈ warm** — its one shared-block analysis (~0.30 ms/ctg) is negligible,
  so a single cold pass costs essentially the per-iteration solve. This is the path to
  ship for a one-shot SA.
- The **UBATCH ↔ batch-API tradeoff**: the batch API has the faster per-iteration solve
  (6.31 vs 8.22) but a far costlier analysis (2.92 vs 0.30) because cuDSS 0.7.1's batch
  API re-analyzes all 32 scenario-blocks even though they share one sparsity pattern,
  while UBATCH analyzes the single shared block. For the primary single-cold-pass case,
  UBATCH wins end-to-end.

**One-time process init.** The first `cudaMalloc` in the JVM triggers CUDA context +
module/JIT load (~2.1 s), paid once per process — irrelevant to throughput over many SA
runs. Including it, a single cold K=4 pass is 13.04 ms/ctg (UBATCH) / 13.76 (batch API).
Every subsequent context build is ~61 ms (host pattern 40 + alloc/upload 28 + cuDSS
analysis lazy in the first solve iteration); the context cache pays the analysis once per
pass and reuses it across the pass's chunks.

## Per-phase decomposition (cuDSS 0.7.1, nS = 32, ms/system)

| mode      | n     | LU nnz/block | fill  | factor | solve |
|-----------|-------|--------------|-------|--------|-------|
| batch API | 18482 | 299587       | 2.0×  | 0.866  | 0.331 |
| UBATCH    | 18482 | 304772       | 2.0×  | 1.274  | 0.378 |

The per-iteration cost is dominated by cuDSS (factorize + fwd/bwd solve ~85% of GPU
time by `nsys`; the per-branch fused fill kernels are ~9%). Running cuDSS with
`IR_N_STEPS = 0` (the outer Newton loop self-corrects, so iterative refinement is
redundant) cut ~30% off the per-iteration solve versus the inherited `IR = 2`.

## Fixed-iteration sweep (DC-init, 926 GPU contingencies, chunk 32)

Raw device throughput at a fixed iteration count `K` (bare Newton, no per-scenario
retirement), hot-started from DC-init (needs more iterations than the AC hot-start
above):

| K  | ms/ctg | ctg/s | converged to 1e-6 | med ‖F‖∞ |
|----|--------|-------|-------------------|----------|
| 4  | 10.93  | 91.5  | 0/926             | 1.2e-1   |
| 5  | 12.98  | 77.0  | 0/926             | 3.2e-4   |
| 10 | 24.01  | 41.6  | 924/926           | 3.4e-12  |
| 20 | 46.02  | 21.7  | 924/926           | 3.4e-12  |

(2/926 never reach 1e-6 — a stubborn contingency class.) The DC-init start needs
K ≈ 10 to converge; the AC hot-start (production path) converges at K = 4, which is why
the warm headline above is ~8.5 ms/scenario. Per-iteration cost scales linearly with K,
confirming the cost is the device Newton loop, not per-pass overhead.

## 2026-07-13 rerun — cuDSS 0.8.0, batch 500, vs gpusim2grid (renamed powerflow_gpu)

Both sides on cuDSS **0.8.0**, UBATCH, FP64, K=4, **batch/chunk 500**, the SAME 500
matched non-islanding contingencies, RTX A1000. Theirs = gpusim2grid built from source
against the same 0.8.0 install, matched bench ported to it
(`gpusim2grid/benchmarks/matched_contingency_bench.py`).

|                                   | ours (this repo)                  | gpusim2grid                          |
|-----------------------------------|-----------------------------------|--------------------------------------|
| contingency phase (steady-state)  | **4.87 ms/ctg** (205 ctg/s)       | chunks **3.76** (total incl one-time preprocess/analysis 4.29, 233 ctg/s) |
| TRUE cold end-to-end              | **11.41 ms/ctg** (5 705 ms = CPU base LF 979 + cold GPU SA 4 726, CUDA init inside) | **8.81 ms/ctg** (4 403 ms = base_case 2 256, GPU base + CUDA init, + total 2 147) |
| convergence                       | 499/500 @1e-6, med ‖F‖∞ 3.6e-12   | 499/500 @1e-6, med ‖F‖∞ 8.6e-12      |

- Cold gap **1.30×** (June 2026, cuDSS 0.7.1/batch 32: ours 16.05 vs theirs 11.93 =
  1.35×) — both sides gained ~35–40% from 0.8.0 + large uniform batches. Measured
  contributors on our side: base-solve placement (our base LF on CPU ~1 s vs their
  ~0.4 s GPU base, which also absorbs CUDA init — since fixed: the SA base now runs
  on the GPU solver) and host extraction; the residual chunk-speed difference is not
  yet attributed and is our optimization backlog (fill kernels, per-iteration
  readbacks, batching geometry). NB the two tools use DIFFERENT formulations (we run
  OLF's own — that is the project; theirs is benchmark context, not a target): 1:1
  parity is not required.
- The 0.8.0 batch-size cliff is CONFIRMED ON THEIR HARNESS TOO: at batch 32 on 0.8.0
  their chunks are 16.28 ms/ctg (vs 6.56 on 0.7.1/June) — same per-launch fixed cost
  we measured on our side (17.8 at MB=32). cuDSS 0.8.0 UBATCH demands large batches,
  for everyone.

## 2026-07-13 — base-on-GPU + shared extraction (cold-E2E levers, ours)

Same setup as the rerun above (cuDSS 0.8.0, UBATCH, batch 500, K=4, 500 matched ctg):

| ours, cold E2E                     | ms/ctg | total (base + SA)      |
|------------------------------------|--------|------------------------|
| CPU base (before)                  | 11.41  | 5 705 = 979 + 4 726    |
| base on GPU (`a94135a`)            | 10.99  | 5 496 = 2 899 + 2 597  |
| + shared extraction (GpuAcDataCache)| **10.78** | 5 391 = 2 843 + 2 548 |

The base phase absorbs the ~2 s one-time CUDA init (so the SA phase is warm), and the
batch reuses the base solver's host extraction through the invalidation-guarded
GpuAcDataCache. Production (`GpuBatchedAcSecurityAnalysis`) does both automatically.
vs gpusim2grid's 8.81: 1.22×. Remaining on our backlog: the residual chunk-speed
difference (2 351 vs 1 881 ms — not yet attributed; profile fill kernels /
per-iteration readbacks / batching geometry) and the duplicated NATIVE context
(base solver vs batch contexts are separate device allocations/analyses; ~100 ms).

## 2026-07-15 — nsys attribution of the chunk-speed residual (ours 2351 vs theirs 1881 ms)

Both sides profiled with nsys on the identical workload (0.8.0, UBATCH, batch 500, K=4,
500 matched ctg). Per Newton iteration of the 500-chunk (ours 532 ms wall, theirs ~460):

| per iteration            | ours          | gpusim2grid    | delta   |
|--------------------------|---------------|----------------|---------|
| cuDSS factorize (2 sub-batches) | 272 ms | 243 ms         | +29     |
| cuDSS fwd / bwd / copy   | 78 / 48 / 20  | 77 / 48 / ~20  | ~0      |
| J fill                   | **92** (81 CB + 7.5 shunt + misc) | ~55 (fill_J ~40 + Va/Vm updates ~15) | +37 |
| host/launch gaps         | ~25           | tight          | ~+20    |

Plus once per pass: our trailing residual check runs the FULL J fill (~90 ms) where
they use a dedicated mismatch-only kernel (fill_FP, ~1.5 ms).

Reconciliation: 1881 + 4×(29+37+20) + 90 ≈ 2315 ≈ our 2351. So the residual is:
- ~120 ms/pass cuDSS factorize (their LU is slightly sparser; fwd/bwd/copy are parity),
- ~150 ms/pass our J-fill cost (fillClosedBranchBatch at 81 ms/launch is compute-bound:
  3 dynamic sincos per branch-scenario × 8M),
- ~90 ms/pass full-J fill used for an F-only residual check,
- ~100 ms/pass host gaps.

Ranked levers (all ours): (1) mismatch-only residual kernel (~90 ms, easy);
(2) fill trig reduction — precompute static sin/cos(ksi±a1) per scenario and derive
sinTheta/cosTheta from one sincos(dph) per branch via angle addition, i.e. feed the
PRECOMPUTED-TRIG form of the same OLF formulas (the Java's own signature shape —
java2cuda can emit that variant from the same source); (3) cuDSS reordering-alg knobs
for the factorize delta; (4) launch-gap batching.

## 2026-07-15 — F-only fill kernels (lever 1 from the attribution)

java2cuda now emits a values-only routine per family (`eval<X>F` — the residual
values alone, kernel locals pruned) next to the fused J+F routine, from the same
parsed OLF Java. The JNI uses F-only batch fills wherever an iteration never
consumes the Jacobian: the final residual/convergence fill of every pass, and
mode 4's solve-only iterations (residual positions in the dest stride are pinned
to the generated scatter tables by static_asserts).

- mode 3 K=4 warm chunks: 2351 → **2291 ms (4.58 ms/scn)** — the trailing full-J
  fill (~90 ms) became ~30 ms. Cold E2E ≈ 10.8–11.0 ms/ctg (CUDA-init run noise
  exceeds the delta).
- mode 4 re-checked with cheap F-iterations (K sweep 4/8/12 = 1143/1766/2393 ms,
  313/476/495 of 500 converged): still convergence-limited vs mode 3 K=4's
  499/500 @ 2291 — the June "net wash" verdict stands; frozen-J is not the path.
- Next per the attribution: the fused fill itself (81 ms — precomputed-trig
  variant via angle addition), cuDSS reordering knobs (+29 ms/iter factorize).

## 2026-07-15 — CPU-mirrored emission + precomputed-trig (lever 2, measured verdict)

java2cuda now emits ONE DEVICE FUNCTION PER OLF JAVA METHOD — same name, same
signature (proof check E locks all 63 of them to the golden manifest), same body,
in a per-family namespace mirroring the Java class scoping. The eval* routines are
thin assemblers that only call them (AcNetworkVector's structure), in four forms:
raw / F-only / Pre / PreF (precomputed-trig parameters, the CPU signature shape).

Measured verdict on feeding the batch fills precomputed trig (per-element statics
+ one sincos(dph) via angle addition, controlled-a1 fallback):
- FUSED fill: **slower** (81 → 100 ms/launch) — it is atomic/memory-bound on its
  28 scattered outputs; the saved FP64 trig was hidden latency, the extra static
  reads were not. Kept on the raw assembler.
- F-ONLY fill: **faster** (~30 → 23.8 ms) — 4 outputs, genuinely compute-bound.
  Kept on PreF.
Net: mode 3 K=4 warm chunks 2291 → **2281 ms (4.56 ms/scn)**. The remaining
per-iteration composition is cuDSS factorize 272 + fill 81 + fwd/bwd/copy 146 —
the fill's floor is its atomics, not its math.

## 2026-07-15 — Shared native structure between base LF and batched SA (lever: context unification)

The duplicated NATIVE context construction is gone: everything both solvers derive
from one extraction — row permutation, superset CSR pattern (host + device), the
four family packs with role-folded dest arrays, DISTR_Q/RHO + zero-impedance packs,
per-element static trig — now lives in ONE `SharedStructure` (built by
`createStructure`, held by `shared_ptr` in both `SolverContext` and
`BatchSolverContext`, cached per `GpuAcData` by the weak-keyed `GpuAcStructures`).
`createBatchContext`/`recommendBatchSize` take the structure handle; the chunk-size
LU probe (a full cuDSS ANALYSIS on the pattern) is cached ON the structure — one
probe per extraction instead of one per SA run. ~230 lines of duplicated native
build code deleted alongside (`computePattern` retired; net -678 lines).

Where the sharing engages: the PRODUCTION SA path (`GpuBatchedAcSecurityAnalysis`),
where the base GPU LF and the batch run on the same engine equation system — now
asserted by `GpuBatchedAcSecurityAnalysisTest#gpuBatchedEngagesAndMatchesCpuOnDevice`
(`GpuAcStructures.hitCount()` increases across the SA run). In the FIXED-ITER bench
the base phase solves a different equation-system instance, so there the gain is
"one structure build + one probe" inside the cold SA pass only.

Bench (RTX A1000, cuDSS 0.8.0, UBATCH, MAXBATCH=500, K=4, 500 matched ctg):
- CPU-base cold E2E: 11.41 → **11.08 ms/ctg** (cold SA phase 4726 → 4583 ms).
- Base-on-GPU E2E: **10.89–10.95 ms/ctg** (SA phase 2597 → 2556 ms).
- Warm steady state unchanged: **2278–2287 ms chunks (4.56 ms/scn)**, 499/500 @
  1e-6, med ‖F‖∞ 3.5e-12 — numerically identical path, structural change only.
Suite 69/0, checkstyle 0.

Note: structure sharing is per-`GpuAcData` instance; the base LF always extracts
the DS_RL flavor, so an SA run with DS or RL disabled extracts a different flavor
and builds its own structure (same cost as before — no regression, just no hit).
Cross-flavor content matching is a possible follow-up if that case ever matters.
