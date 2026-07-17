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

## 2026-07-16 — Cross-flavor structure sharing + bench made production-faithful

Two follow-ups to the shared-structure work:

1. **The fixed-iter bench now shares ONE equation system** between the base load
   flow and the batched sweep (one `AcLoadFlowContext` provides the es for both —
   the production `GpuBatchedAcSecurityAnalysis` shape). Before, the bench's base
   LF ran on the engine's own second es, so the base->batch extraction/structure
   hand-off never engaged there — a bench artifact, not a GPU limitation (LF and
   SA DO share the es in production, same as on CPU).
2. **Cross-flavor content matching** (`GpuAcData.sameStructureInputsAs` +
   refcounting holder in `GpuAcStructures`): the flavor flags (DS/RL) only select
   the outer-loop packs, which are batch-CONTEXT inputs — so an SA run with DS or
   RL disabled now aliases the base's structure instead of rebuilding it (the base
   always extracts DS_RL). Values refresh re-keys via `GpuAcStructures.rebind`
   (drops every stale alias) and now also re-uploads `ziIn` (latent gap: a
   coupler-rho move previously survived a VALUES refresh stale on device).
   Proven by `GpuAcStructuresTest` (same handle, no extra build, solve converges).

Bench (RTX A1000, cuDSS 0.8.0, UBATCH, MAXBATCH=500, K=4, 500 matched ctg),
base-on-GPU cold E2E: 10.89–10.95 → **10.68 ms/ctg** (5340 ms total = base 2808 +
SA 2532) — the batch now reuses the base's extraction AND device structure in the
bench, as in production. Warm unchanged: **2286 ms chunks (4.57 ms/scn)**, 499/500
@ 1e-6, med ‖F‖∞ 3.7e-12. vs gpusim2grid cold 8.81 ms/ctg → gap 1.21×.
Ladder recap (cold, base-on-GPU): 11.41 → 10.99 (base-on-GPU) → 10.78 (shared
extraction) → 10.68 (shared device structure, es-faithful bench).

## 2026-07-16 — cuDSS reordering sweep: AMD adopted as default (lever 1: factorize fill)

Sweep of `CUDSS_CONFIG_REORDERING_ALG` on the standard bench
(`native/cuda/sweep_reorder.sh`, per-run LU_NNZ via `traceLuNnz`):

| alg        | LU_NNZ (base & ubatch)  | warm K=4 chunk | convergence      |
|------------|-------------------------|----------------|------------------|
| default    | 304772                  | 2675 ms        | 499/500 @ 1e-6   |
| nd         | 304772 (= default)      | 2671 ms        | 499/500          |
| **amd**    | **260076 (−14.7%)**     | **2575 ms**    | 499/500, same ‖F‖∞ |
| btf_colamd | NOT_SUPPORTED (status 4) at the UBATCH ANALYSIS | — | — |
| colamd     | NOT_SUPPORTED (status 4) at the UBATCH ANALYSIS | — | — |

So cuDSS 0.8.0's library default on this pattern IS nested dissection, and the
COLAMD family is rejected on the uniform-batch path. AMD gives 15% sparser LU
factors — and since ANALYSIS (where the ordering is computed) is a cold-path,
once-per-context cost, AMD (cheap to compute) also helps cold, the metric that
matters.

Interleaved cold A/B (3 pairs, base-on-GPU, idle machine): AMD won the GPU-SA
phase in ALL pairs (mean 3854 → 3418 ms, −11%) and the warm chunk in ALL pairs
(mean 2706 → 2615 ms, −3.4%); E2E totals favored AMD 2/3 with the one reversal
fully explained by base-LF JVM/CUDA-init noise (3.6–6.5 s swing, algorithm-blind).
NOTE: the machine ran ~15% slower than the 10.68 ms/ctg session (default's warm
chunk 2675–2730 vs the recorded 2286 ms) — absolute ladder numbers from this day
are not comparable; the paired deltas are.

Validation: full suite 70/0 under `OLF_GPU_REORDER_ALG=amd` (the GpuLineSearchTest
knife-edge survived the elimination-order rounding change), run_gpu_tests.sh 8/0
on the adopted lib, confirmation bench traces `reorder=amd(default)` LU_NNZ=260076,
warm 2595 ms, 499/500, med ‖F‖∞ 3.7e-12.

Adoption: env unset now applies AMD on cuDSS >= 0.8.0 (`applyReorderAlg`);
`OLF_GPU_REORDER_ALG=default` restores the library default; 0.7.1 untouched
(unmeasured there). Tooling fixes alongside: run_bench.sh now propagates the mvn
exit status (was `|| true` — a failed run looked green to sweep drivers) and
includes the `OLF_GPU_CUDSS` trace lines in its RESULT block; sweep_reorder.sh
surfaces the cuDSS error line for failed algorithms; run_gpu_tests.sh was stale
(`mvn -o`, shared ~/.m2) and now uses the isolated settings.

## 2026-07-16 — Launch-gap analysis (lever 2): it's stream semantics, not launch count

Source-level comparison for the ~25 ms/iter host/launch gaps from the 2026-07-15
nsys attribution, before implementing:

- Launch COUNT is comparable and cannot explain the gap: gpusim2grid launches
  ~7–10 kernels/iter (fill_FP, fill_FQ, one whole-J fill_J, update_Va/Vm +
  slack/mask adjusters; nr_iter_step.cuh) vs our ~11–12 (initF, 3 family fills on
  pegase, dq/rho/zi packs, identity/negate/rowMode, axpy) + one LARGE per-iter
  cudaMemset of the J values buffer (nnz x nS doubles). Per-launch overhead is
  microseconds; 1.5–2x more launches is noise.
- The structural difference: THEY put every kernel AND cuDSS on one explicit
  CUDA stream (`cs`) with ZERO host syncs inside the loop. WE launch everything
  on the LEGACY DEFAULT stream, which implicitly serializes against every other
  stream in the process — including cuDSS's internal streams — stitching the
  timeline with implicit sync points even though our explicit per-iter syncs are
  already skipped in fixed-iter mode (June b94a86a02, carried over).
- CUDA graphs: NEVER tried — no experiment in either repo's history; the only
  mention is gpu_roadmap.md P3 "mostly subsumed by batching". Not ruled out
  empirically; stays last, likely unnecessary if the stream fix lands.
- PRODUCTION-only finding: the sync-skip is fixed-iter-only. The production SA
  path (tol > 0) still does cudaDeviceSynchronize + a FULL batched-mismatch D->H
  copy EVERY iteration — a cost the bench never sees. Stream + lagged/async
  convergence check fixes that too (follow-up).

Plan (cheapest first): (1) dedicated stream per batch context + cudssSetStream;
(2) one cudssExecute(FACTORIZATION|SOLVE) call instead of two (phases can be
OR-ed) to halve per-iter host phase overhead; (3) epilogue fusion only if 1–2
leave a measurable gap; (4) graphs last resort.

## 2026-07-16 — Batch solve on a dedicated stream (lever 2 step 1, measured)

Implementation: one `cudaStream_t` per `BatchSolverContext` (BLOCKING flags — any
op this conversion missed still orders correctly against the legacy stream),
`cudssSetStream` on the batch cuDSS handle, every in-loop kernel/memset in
`solveBatchContext` launched on the stream (memsets → `cudaMemsetAsync`), the
per-iteration and per-pass `cudaDeviceSynchronize` narrowed to
`cudaStreamSynchronize`. No arithmetic change.

Interleaved cold A/B (3 pairs, base-on-GPU, both libs AMD-default; lib swapped in
target/classes between runs). COLD FIRST (the decisive metric): E2E
**17.18 → 16.93 ms/ctg mean** (8591 → 8463 ms), stream wins 2/3 pairs — but the
cold SA phase itself was FLAT (3140 → 3146 ms mean; the chunk gain is inside its
±50 ms ANALYSIS/first-factorize noise) and the E2E delta mostly landed in the
base-LF term the stream doesn't touch. Honest cold claim at 500 ctg / 1 chunk:
no cold cost, noise-floor cold gain; the gain COMPOUNDS into cold with more
chunks (every chunk collects −55 ms → real at 2000+ ctg). Warm K=4 chunk:
2434/2373/2409 → 2356/2349/2345 ms — wins ALL pairs, mean **−55 ms/chunk (−2.3%,
4.80 → 4.70 ms/scn)**, and the run-to-run spread collapses 61 → 11 ms (the
implicit serialization was also the jitter — future cold A/Bs get sharper).
Convergence identical (499/500 @ 1e-6, same ‖F‖∞). Recovered ~half of the
~100 ms/chunk gap budget; the rest is cudssExecute host phase overhead → next is
step 2, one cudssExecute(FACTORIZATION|SOLVE) per iteration (runs INSIDE the cold
chunk too, unlike this step it helps the very first factorize+solve), and the
PRODUCTION tol>0 per-iteration sync+full-mismatch copy (pure cold cost in every
real SA run, invisible to this bench).

Machine-state caution again: the same afternoon the identical no-stream AMD lib
benched 2615 → 2405 ms warm chunks hours apart — only interleaved pairs are
comparable, never cross-session absolutes.

## 2026-07-16 — Warm-JIT cold-GPU E2E variant + SAME-DAY comparison vs gpusim2grid

The bench now prints [E2E-warmJIT K=..]: a second full pass in the same JVM on a
freshly built problem (new LfNetwork/context/es; GpuAcStructures.clear() +
clearContextCache() first — without the new clear() hook, content matching would
alias pass 1's device structure and un-cold the measurement). Warm JIT + warm
CUDA driver, cold everything GPU-side: the long-running-service scenario. The
single-shot [E2E] line is unchanged (OLF_GPU_E2E_WARMJIT=0 disables the pass).

Same-day paired numbers (RTX A1000, both sides cuDSS 0.8.0/UBATCH/FP64/K=4/
batch 500, same 500 matched ctg; ours = AMD ordering + dedicated stream):

| cold-first                    | ours              | gpusim2grid          | ratio |
|-------------------------------|-------------------|----------------------|-------|
| process-cold E2E              | 17.30 ms/ctg (5442 base + 3206 SA) | 11.94 ms/ctg (3096 base_case + 2874 total) | 1.45x |
| in-process repeat E2E         | **9.12 ms/ctg** (1712 base + 2846 cold SA) | not measured (their bench has no repeat mode; est. ~7.9 with the ~2 s CUDA init removed) | ~1.15x est. |
| warm chunk                    | 4.68 ms/scn       | 3.77 ms/scn          | 1.24x |
| convergence (both)            | 499/500 @ 1e-6, med ||F|| 3.6e-12 | 499/500 @ 1e-6, med ||F|| 8.34e-12 | we converge deeper |

Readings:
- ~4.1 s (~47%) of OUR process-cold E2E is JVM JIT warm-up, not GPU work: the
  warm-JIT pass drops base 5442 -> 1712 ms and SA 3206 -> 2846 ms. The
  process-cold ratio (1.45x today vs 1.30x recorded 2026-07-13) is therefore
  mostly a JVM-vs-Python runtime comparison, not a GPU-engineering gap.
- BOTH sides are host-slower today than their recorded numbers (theirs: base_case
  2256 -> 3096, contingency total 2147 -> 2874) while their warm chunks are
  IDENTICAL (3.76 -> 3.77) — confirming the machine drift is host-side and warm
  GPU numbers are the stable cross-day reference.
- The structurally comparable service-scenario pair needs a repeat mode in THEIR
  bench for a measured (not estimated) ratio — candidate follow-up if the ~1.15x
  estimate matters.

## 2026-07-16 — Fused cudssExecute(FACTORIZATION|SOLVE) + base-context stream (lever 2 step 2: measured WASH)

All five per-iteration cuDSS sites now issue ONE cudssExecute (phases are OR-able
bit flags; each mode keeps its factorize policy — mode 4 still solves-only after
iter 0, mode 5/blkdiag still REFACTORIZE, base fuses FACT-or-REFACT with SOLVE;
ANALYSIS stays separate for traceLuNnz). The base SolverContext also moved to its
own BLOCKING stream + cudssSetStream (same design as the batch context; in-loop
kernels/memset/D2D on it, per-iter sync narrowed to cudaStreamSynchronize — the
per-iteration mismatch readback stays, it IS the base convergence loop).

Interleaved A/B (3 pairs, base-on-GPU): a WASH on every metric. COLD-FIRST,
warm-JIT cold-GPU E2E: 8.65/8.77/8.68 (basestream) vs 8.47/8.91/8.64 ms/ctg
(fused) — split pairs, mean −13 ms = noise. Warm chunk 2377 vs 2360 ms mean =
noise. Convergence identical. NB the pair ORDER slightly biased against fused
(always second on a within-session warming machine — the cold SA column drifts up
monotonically across all six runs); still a wash.

Why (and why kept): gpusim2grid does NOT fuse — factor()/solve() are two separate
cudssExecute calls, async on their bound stream — and stays tight anyway: once a
stream is bound (step 1), the second call's host overhead is negligible. KEPT as
strictly neutral (fewer host round-trips, same numerics, base-context stream
consistency). The remaining chunk-speed residual vs their 1881-ms chunks is NOT
in the cuDSS phase calls: next suspects are the per-iteration dVals cudaMemset
(nnz x nS doubles of device bandwidth every full fill) and the fill scatter
itself (the known atomic-bound floor).

## 2026-07-17 — dVals memset-hiding: REVERTED on nsys evidence (lever closed)

Tried: side memsetStream + evFactDone/evValsCleared events; on full-fill
iterations the ubatch call un-fuses (FACTORIZATION, record event, side-stream
memset of dVals for iter k+1, SOLVE), so the zeroing overlaps SOLVE — the J
values are dead after FACTORIZATION. Gated on IR_N_STEPS == 0 (IR re-reads
matrix values during SOLVE), post-loop event drain for early-retirement exits,
valsPreCleared reset per outer pass (Q-eval block dirties dVals as scratch).

Interleaved alternating-order A/B: WASH (warm chunk 2263 vs 2279 ms, warm-JIT
cold E2E 3949 vs 3966 ms excluding one ~10% machine-hiccup outlier; pairs split).

nsys profile (preclear lib, warm chunks only, `-t cuda`) settled it:
- dVals memset = 602.5 MB (150,620 nnz x 500 scn x 8 B), **4.68 ms avg**, once
  per full-fill iteration.
- Per-iteration GPU work is ~440 ms: factorize_batch 239.5 ms, fillClosedBranch
  81 ms, fwd 79 ms + bwd 39 ms, copy_matrix 2x10 ms. The memset is ~1% of the
  iteration → hiding it caps the win at ~19 ms per 2154-ms chunk (0.9%), below
  bench noise. The wash was structural, not bad luck.
- The overlap DID engage (each side-stream memset sat inside exactly one
  concurrent main-stream kernel) — the machinery worked; there was just nothing
  worth hiding.

REVERTED (git checkout, back to fused single-execute at 8e09574): 4.7 ms/iter
does not pay for a side stream + two events + conditional un-fusing + an IR
race gate + an early-exit drain in the hottest function. Bench at reverted HEAD
confirms baseline.

Profile side-findings: two OTHER per-iteration memsets of similar size run on
the MAIN stream (613.4 MB @ 4.77 ms and 426.9 MB @ 3.32 ms, 16 instances each)
— sizes match no host-issued buffer, so they are cuDSS-internal (not ours to
hide). Total memset footprint is 12.8 ms/iter (~3%) even counting those.

The real remaining chunk lever is unchanged: the fill scatter (81 ms/iter,
atomic-bound) via the owner-store redesign — which would eliminate the dVals
memset as a side effect anyway.

## 2026-07-17 — Fill-scatter owner-store Gate A + ownership census: GO

Two throwaway probes (hack build, reverted after; never committed) to bound the
owner-store redesign BEFORE building it:

**Gate A (atomic-cost ceiling)**: scatterAdd's vals writes swapped for plain
stores (numerically wrong on shared slots — probe only). On sane inputs the
closed-branch fill drops **80.8 → 34.3 ms/launch (−58%)**, reproduced in two
runs (K=4 and K=1 profiles, first instance each — identical inputs to the
baseline's). fillShuntBatch 7.49 → 6.12 ms. CAVEAT for anyone re-running this
hack: only the FIRST instance is valid — the wrong Jacobian diverges x to ~1e55
and sincos falls into Payne-Hanek argument reduction, inflating later instances
to ~91 ms (also why chunk totals under the hack mean nothing). So ~47 ms/iter
of the 81 is atomic-write cost, NOT eval compute — consistent with the
2026-07-15 finding that cheaper trig made the kernel SLOWER.

**Ownership census** (case9241pegase scope, all vals writers counted: cb, ob,
sh, hv, DISTR_Q, zero-imp, DISTR_RHO — only cb+sh are non-empty in this case):
nnz=150,620, all written; **69.4% single-owner slots** (104,600), 30.6% shared
(46,020; avg 3.3 contributors, max 47 — bus-degree diagonals). Closed-branch:
256,784 J writes/scenario, **40.7% land in single-owner slots** (become plain
stores). Shunt: 14,654 writes, 100% shared (bus diagonals) — owner-store wins
nothing there.

Verdict: GO, expected win between ~19 ms/iter (if cost ∝ writes converted) and
~47 (ceiling), i.e. 75–190 ms/chunk (3.5–9%) against the +37 ms/iter fill delta
vs gpusim2grid. If the contended 59.3% keeps the kernel slow, the escalation is
staging+reduce for the shared slots only (contributor CSR from the same census
pass — zero atomics, ~10 ms staging bandwidth, near-full ceiling) — decided by
measurement, not upfront. Design notes in the session record: ownership bit
per dest slot from ONE collection pass (miss a writer = silent corruption),
disabled elements/inactive scenarios store 0 instead of early-returning, shared
slots keep atomicAdd + compact zero list (full dVals memset dies as a side
effect).

## 2026-07-17 — Fill-scatter de-atomization: BOTH variants measured a LOSS, lever CLOSED

Both designs were implemented in full, verified numerically EXACT (499/500 @
1e-6, med ||F||inf 3.6e-12 — identical to baseline), measured, and REVERTED
(back to 8e09574's all-atomic scatterAdd). Ranking of the CORRECT variants,
closed-branch fill ms/launch (same day, machine-parity checked via factorize
and fwd kernels):

| variant                                  | fill        | notes |
|------------------------------------------|-------------|-------|
| all-atomic (baseline, KEPT)              | **80.8**    | chunk ~2135 ms |
| owner-store hybrid (40.7% plain stores)  | 84.7        | store/atomic warp divergence; shunt control byte-identical (its 100% shared writes take one path) |
| staged zero-atomic + reduceSharedSlots   | 112.3 + 9.7 | slot-grouped staging, deterministic sums; chunk 2305 ms |

WHY Gate A misled: the all-stores hack (34.3 ms) let the hardware MERGE the up
to 47-way colliding writes to each bus-diagonal address — i.e. it silently
dropped most of the shared-slot traffic (the same reason its numerics were
garbage). A CORRECT zero-atomic scheme must materialize all 166,838 staged
contributions at DISTINCT addresses and re-read them in a reduce — MORE traffic
than the atomic version. On Ampere, contended FP64 atomicAdd through the L2
atomic unit is the EFFICIENT path for this access pattern: same-address updates
coalesce in place. Lesson for the next Gate A: an all-stores probe measures
"atomics + traffic that same-address merging can drop", not the atomic cost
alone — pair it with a stores-to-DISTINCT-addresses probe before trusting it
as a ceiling.

Consequence for the gpusim2grid fill comparison: their ~40 ms fill_J is NOT
explained by "no atomics" (our zero-atomic build is 112 ms); it comes from
their rectangular-complex formulation doing far less math+writes per entry (no
sincos at all, 4 stores per Ybus nnz) — formulation context, not a to-do (see
comparison framing). OUR fill at 81 ms with 28 scattered outputs x 8M
element-scenarios is at its hardware floor for this write pattern; the
remaining fill delta vs theirs is structural. Fill levers exhausted:
F-only kernels (DONE, kept), precomputed-trig (SLOWER, dropped), owner-store
(SLOWER, dropped), zero-atomic staging (SLOWER, dropped).

Remaining performance backlog is elsewhere: production tol>0 per-iteration
sync + FULL mismatch D->H copy (lagged/async convergence check — invisible to
the fixed-iter bench, paid by every real SA run), and cuDSS factorize (+29
ms/iter pre-AMD; partially recovered by AMD reordering).

## 2026-07-17 — Production (tol>0) path profiled: the STRAGGLER, not the sync, is the cost

The bench gained OLF_GPU_TOL (default 0 = unchanged): tol>0 runs every solve on
the PRODUCTION path — per-iteration host sync + full-mismatch D->H copy +
convergence retirement. Measured (case9241pegase, 500 ctg, UBATCH 500, AMD,
tol=1e-6, maxIter=15, nsys):

- **Wall: 18,066 ms vs 2,131 ms fixed-iter K=4 — 8.4x** — same final quality
  (499/500 @ 1e-6, med ||F||inf 4.1e-12).
- Attribution: **114 FULL-WIDTH cuDSS factorizations** (~57 iterations/solve
  vs 16 total in the K=4 run) at 219-244 ms EACH. The ONE never-converging
  scenario (the known 1/500) stays active, so every outer pass runs the inner
  loop to the maxIter=15 cap — and cuDSS UBATCH charges full 500-wide price
  per iteration regardless of retirement: factorize/fwd/bwd on 499 identity
  systems cost the same 233/79/39 ms as real ones. Our OWN fills DO scale with
  retirement (fillClosedBranchBatch instances drop to 0.3 ms once 499 retire).
- The originally-suspected sync+copy cost is real but SECONDARY: ~1.3 s of
  70-MB pageable D->H mismatch copies (~5.5 GB/s) + host norm loops, ~7-10%
  of the wall.

Production levers, now correctly ranked:
1. **Straggler early-exit + offload** (dominant): when nActive falls below a
   small threshold, BREAK the batched loop and route leftovers to the existing
   CPU fallback (or per-scenario GPU mode 0). The 499 converge in ~6-8
   iterations; today the last straggler makes the full batch pay 15 per outer
   pass. Expected: 18.1 s -> roughly 6-8 s wall.
2. Device-side per-scenario norms (+ optionally lagged check): shrinks the
   74-MB/iter copy to ~12 KB and kills the host 9.2M-element loop (~1.3-2 s).
3. Batch compaction (repack actives into a narrower UBATCH mid-run): largest
   theoretical win but needs a re-ANALYSIS per width — heavy; only worth it
   after 1+2 are measured.

NB fixed-iter benchmarking (ours AND gpusim2grid's nb_iter) structurally hides
all of this — it is a THROUGHPUT methodology, not a production-latency one.

## 2026-07-17 — PRODUCTION SA benchmark: RTE parameters, full outer loops, GPU vs CPU (both real products)

New committable bench `GpuVsCpuProductionSaBenchmarkTest`: the real AcSecurityAnalysis
on both sides — OpenSecurityAnalysisProvider vs GpuSecurityAnalysisProvider — on
case9241pegase, the same 500 matched contingencies, parameters loaded from
`rte-loadflow-default-parameters.json` (repo root: DC init, reactive limits, twt split
shunt, HVDC AC emulation, transformer V-control AFTER_GENERATOR, MAX_VOLTAGE_CHANGE
scaling, maxNR/maxOuterLoop 30, realistic-V checks), violations included. SPARSE matrix
factory forced on both arms. CPU arm multithreaded (4 threads on this machine,
OLF_GPU_SA_CPU_THREADS); GPU arm host-single-threaded.

| arm                          | wall     | ms/ctg | vs CPU-4t |
|------------------------------|----------|--------|-----------|
| CPU 1 thread                 | 40,491   | 80.98  | —         |
| **CPU 4 threads**            | **13,168** | **26.34** | 1.00    |
| GPU batched (COLD, run 1)    | 24,240   | 48.48  | 0.54x     |
| GPU batched (warm, run 2)    | 22,162   | 44.32  | 0.59x     |

Parity: pre-contingency CONVERGED both; statuses IDENTICAL (499 CONVERGED + 1
MAX_ITERATION_REACHED both sides — asserted); violations 13,412 (CPU) vs 13,791 (GPU),
~2.8% border-case drift to investigate.

VERDICT: at RTE production parameters the GPU SA today LOSES to 4-thread CPU (0.54-0.59x)
while beating 1-thread (1.7-1.8x). This is exactly the tol>0 straggler finding at product
level: the GPU pays full-width cuDSS iterations until the last scenario caps out. The
straggler early-exit + device-side norms levers (see the tol>0 profile entry) are what
turn this around (~8-10 s projected -> ~1.5x AHEAD of CPU-4t).

Traps/bugs found by building this bench:
- AbstractOpenSecurityAnalysisTest's default (dense) matrix factory silently makes the
  9241-bus base LF thrash ~200 s and FAIL — the same case+params converge in 2 s /10
  iters sparse. Force SparseMatrixFactory in any large-case SA bench.
- PRODUCT BUG (open): threadCount>1 on the GPU provider SIGSEGVs in cuLaunchKernel —
  the SA harness calls precomputeBatch from several ForkJoinPool workers concurrently;
  the batched native path needs provider-side serialization.
- PRODUCT BUG (open): with a FAILED base LF (dense-factory runs), the GPU provider
  reported pre-contingency CONVERGED and produced 500 post-contingency results — it
  must honor a failed base exactly like AcSecurityAnalysis.

## 2026-07-17 — Violation diff: NOT epsilon drift — the batch drops unimplemented outer loops

Criterion was ALREADY aligned (batchConvEpsPerEq = CPU's convEpsPerEq for
UNIFORM_CRITERIA, RL threshold derived identically) — so the diff is pure signal.
Margin-annotated diff (bench now prints it): **6,137 one-sided violations**
(2,899 CPU-only + 3,238 GPU-only; the "379" was the net), of which only **101 are
border flips** (rel margin < 1e-3); 263 contingencies affected. Worst offenders are a
SINGLE bus repeated across unrelated outages: VL-3632 HIGH_VOLTAGE at value 1.1540 vs
limit 1.10, GPU-only, identical value across many contingencies — a systematic
control-modeling difference, not contingency physics.

DIAGNOSIS: runSimulations captures ONLY DistributedSlack + ReactiveLimits from the
resolved outer-loop list; with the RTE params the CPU also runs transformer voltage
control (AFTER_GENERATOR_VOLTAGE_CONTROL) and friends per post-contingency solve —
the batched device path silently DROPS them. Fix direction: the provider must either
implement the missing loops in the batch or DETECT unsupported outer loops and fall
back to CPU (correct-first), with the batch reserved for parameter sets it fully
covers. Same-day numbers (this run): CPU-1t 38.1 s, CPU-4t 12.0 s, GPU cold 23.8 s
(1.60x vs 1t, 0.51x vs 4t), GPU warm 20.9 s (1.82x vs 1t, 0.58x vs 4t).

## 2026-07-17 — Outer-loop guard landed; RTE-params divergence REATTRIBUTED

Detect-and-fallback is in GpuBatchedAcSecurityAnalysis.runSimulations: unsupported
resolved outer loops → no batch AND no solver swap (provider == stock CPU SA; proven
by a 0/0 violation diff when it fires). Tolerated as structural no-ops: the two HVDC
AC-emulation loops when the network has no HVDC, VoltageMonitoring when no SVCs.
Canary (gpuBatchedEngagesAndMatchesCpuOnDevice) green — defaults still engage.

CORRECTED DIAGNOSIS: with the guard in place the batch legitimately ENGAGES under the
RTE params on pegase (no transformer loop resolves — the JSON never sets
transformerVoltageControlOn; only DS+RL+no-op loops are in the list) and the 6,137
one-sided violation diff RETURNS. So the divergence is NOT a dropped outer loop — it
is inside the batched DS/RL solve semantics themselves. Prime suspects, in order:
(1) MAX_VOLTAGE_CHANGE state-vector scaling (RTE param) — implemented for the BASE
GPU solver, likely NOT applied per scenario in the batch loop; (2) RL switching
detail differences vs ReactiveLimitsOuterLoop; (3) realistic-voltage checks absent
device-side. Next: check solveBatchContext for per-scenario scaling, then re-diff.
Also fixed en route: the fallback previously crashed with a CME (GPU inner solver +
outer-loop switching on the sequential path) — fallback now skips the solver swap too.

## 2026-07-17 — MAX_VOLTAGE_CHANGE ported to the batch (per scenario); suspect FALSIFIED

The batched solve now applies the CPU's MaxVoltageChangeStateVectorScaling PER SCENARIO
(maxVoltageChangeFactorBatch block-reduction of |dv|/|dphi| + whole-step scale; provider
captures mode/maxDv/maxDphi from OLF params; varType via GpuAcDataExtractor). All 3
tests green, timings unchanged. BUT the RTE-params violation diff is IDENTICAL
(6,137 / 101 border): hot-started scenarios never take steps near 0.4 pu / 1 rad, so
the clamp never binds on either side — scaling was NOT the differentiator. Kept as
parity hardening (matters for cold/stressed scenarios).

NEW prime suspect for the 6,137 diff: the batch RL pass may implement only PV->PQ
switching without the CPU ReactiveLimitsOuterLoop's PQ->PV switch-BACK (voltage-control
recovery). Signature fits exactly: VL-3632 pinned at 1.1540 (its controller stuck PQ on
the device) vs CPU pulling it back to <=1.10 across unrelated contingencies. Next:
check the batch RL switch logic for PQ->PV recovery, then re-diff.

## 2026-07-17 — RL switch-back EXISTS in the batch; next suspect: REMOTE-control RL coverage

Read-through of reactiveLimitsBatch/initRlLimitTypeBatch: PV->PQ AND PQ->PV switch-back
are both implemented and mirror ReactiveLimitsOuterLoop (pinned limit types from the
base mode, switch-count cap, v-vs-target recovery conditions). So the 6,137 diff is not
a missing switch-back either. SHARPER SUSPECT: rlVrow covers LOCAL controllers only —
under the RTE params voltageRemoteControl is ON (unlike the GPU-scope benches that
disabled it), remote controllers' reactive limits live on DISTR_Q rows, and the batch
RL pass never re-evaluates THOSE (the CPU outer loop does). A remotely-controlled bus
pinned at a wrong voltage across all scenarios (VL-3632 @ 1.1540) fits exactly.
NEXT: confirm VL-3632 is remotely controlled (ndq > 0 under RTE params), then either
port remote-controller RL switching to the batch or add remote-control presence to the
detect-and-fallback guard (correct-first).

## 2026-07-17 — The 6,137 diff SOLVED: zero-threshold reporting flicker; real tail = 22 in 6 ctg

Single-contingency debugging (new bench mode OLF_GPU_SA_DEBUG_CTG=<id>, ~9 s/iteration)
cracked it in four probes: (1) monitored states IDENTICAL both sides (BUS-5883 1.15401
pre AND post, 5 decimals); (2) same diff with the GPU base forced onto the CPU solver
(OLF_GPU_SA_GPU_BASE=0) — base-solver hypothesis dead; (3) pre-contingency baselines
IDENTICAL (60=60, values to ~1e-8); (4) root cause in OLF's LimitViolationManager
suppression: the effective voltage threshold is Math.min(absoluteThreshold,
value*proportionalThreshold) — powsybl's DEFAULTS are all zero, and setting only the
absolute leg still yields min(abs, 0) = 0. With a zero effective threshold, every
pre-existing violation whose value moved by ±1 ulp vs the run's own baseline is
re-reported — both sides flicker independently at solver tolerance (~1e-6), producing
the 6,137 disjoint one-sided reports on IDENTICAL physics. This affects ANY two
solvers, CPU-vs-CPU included.

Bench now sets both legs (0.5 abs + 0.001 proportional, both voltage directions +
0.1% flow). Result on the full 500 (RTE params, batch engaged, GPU base on GPU):
statuses identical, violations 305 vs 297, **diff 22 one-sided in 6 contingencies**
(LINE-4985-165: 5 GPU-only HIGH_V; TWT-4990-37: CPU-only LOW_V cluster; TWT-14-2925;
margins 1.5-4.9% = genuine state differences). Timings: CPU-4t 15.1 s, GPU cold
25.3 s (0.60x), warm 21.4 s (0.71x). NEXT: drill the 6 contingencies one at a time
with OLF_GPU_SA_DEBUG_CTG — likely RL switch decisions on genuinely bistable
scenarios; also note this reopens the increased-violations defaults question for
production (zero thresholds re-report ulp-flicker there too).

## 2026-07-17 — The last 6 contingencies drilled one by one: NO functional divergence

Per-contingency drill (drill.sh pattern: full pass enumerates DIFF-CTG ids, then one
OLF_GPU_SA_DEBUG_CTG run each, monitors auto-set to the diffing VLs):
- LINE-12-2378 (4 CPU-only): pre-existing APPARENT_POWER flows 0.13-0.2% over limit,
  straddling the 0.1% suppression threshold; states differ 2e-5 pu.
- LINE-4023-92 (2 CPU-only): pre-existing flow increased +0.105% vs the 0.1%
  threshold — CPU just above, GPU just below. Monitored states identical.
- LINE-4320-76: 0/0 IN ISOLATION (full-run entries were the same borderline family).
- LINE-4985-165 (4 GPU-only): pre-existing HIGH_V increased ~+0.001 pu around the
  0.001 proportional threshold; largest genuine state gap observed anywhere:
  2.3e-4 pu (BUS-106) — inside the shared convEpsPerEq=1e-4 criterion slack.
- TWT-14-2925 (1 GPU-only): 6e-5 pu apart; increase 0.00117 vs 0.00111 around the
  0.00113 threshold.
- TWT-4990-37 (severe case, voltages to 0.885): 0/0 IN ISOLATION — both sides agree
  fully; the full-run entries flip because chunk-wide outer-pass composition shifts
  states within tolerance (a scenario's RL/DS passes follow the CHUNK's pass
  schedule), moving threshold-edge values across the line.

CONCLUSION: zero control-state divergences, zero PV/PQ flips, zero missing features.
Every residual diff is a PRE-EXISTING violation whose post value straddles the
increased-violations threshold, decided by <= 2.3e-4 pu state differences — within
the shared stopping criterion. The GPU batched SA is PRODUCTION-EQUIVALENT to the
CPU AcSecurityAnalysis under the RTE parameters, up to the convergence tolerance
both sides are configured to. Remaining perf lever unchanged: straggler early-exit
(GPU warm 21.4 s vs CPU-4t 15.1 s).

## 2026-07-17 — Straggler early-exit LANDED (native-only, status -2 reuses the fallback routing)

tol>0 solves now exit the batched loop when nActive <= max(1, 2% of nS) AND it >= maxIter/2
(both gates needed: someone is always last to retire in a healthy pass at it~5-8; only
cap-bound stragglers offload). Offloaded scenarios get STATUS_REJECTED (-2) — the EXISTING
Java routing sends them to the sequential CPU fallback, zero Java changes. Env:
OLF_GPU_STRAGGLER_FRAC (default 0.02, 0 disables). Fixed-iter mode untouched.

Measured (all parity checks green: statuses identical incl. the offloaded straggler's
MAX_ITERATION on CPU; violation diff unchanged at the 22/6 threshold-edge tail; canary engages):
- tol=1e-6 bench (maxIter 15): 18,066 -> 15,236 ms (-16%); "499 batched" confirms offload.
- Production SA bench (RTE params, maxNR 30): GPU warm 21.4 -> 18.4 s (0.82x vs CPU-4t 15.1 s),
  cold 25.3 -> 22.1 s.

HONEST verdict vs the ~8-10 s projection: the straggler was only PART of the story — the exit
saves ~3 s, and the rest of the wall is (a) the pre-offload iterations each outer pass still
pays (exit depth maxIter/2; tunable earlier via a fixed OLF_GPU_STRAGGLER_MIN_IT if profiling
justifies), and (b) legitimate FULL-WIDTH outer-pass re-solves after RL/DS reactivation —
cuDSS UBATCH width never shrinks (the batch-compaction escalation in the task list is the
structural fix). Next: #11 device-side norms (~1.3-2 s more), then re-profile.

## 2026-07-17 — Device-side per-scenario norms LANDED (70 MB/iter readback -> 12 KB)

residualNormsBatch (one block/scenario: ||F||inf exact, sum(F^2) parallel-reduced, finite flag)
replaces the tol>0 per-iteration full-mismatch copy + 9.2M-element host loop; readback is now
nS*3 doubles. Fixed-iter final readback unchanged. Known ulp effect: parallel ssq summation
order can flip borderline ||F||2 stopping decisions by an ulp (med ||F||inf 4.1e-12 -> 4.2e-12
— exactly that class); criterion itself unchanged.

Measured: tol bench 15,236 -> 14,734 ms (-3.3%); SA bench warm 19.3 s / cold 22.5 s (vs 18.4 /
22.1 previous run — within same-session machine drift; CPU-4t 15.1 s both). All parity green
(statuses identical, violation diff unchanged 22/6, canary engages). HONEST verdict: the win is
~0.5 s, below the 1.3-2 s estimate — the loop is more GPU-bound than the stall model assumed.
Per the plan, #12 (lagged/async check) needs an nsys profile FIRST showing a remaining stall;
the bigger structural lever for the production wall stays batch compaction (full-width outer
passes) per the straggler-exit entry.

## 2026-07-17 — Lagged/async convergence check: NOT NEEDED (measured), lever closed

nsys gate for the stage-2 lagged check, tol=1e-6 bench after the straggler exit + device
norms: within the 15.25-s timed window the GPU idles a TOTAL of 1,360 ms — and 1,351 ms of it
sits in just SIX gaps (~225 ms each), with zero per-iteration gaps above 1 ms. The per-iteration
convergence check is already free (device norms did their job); the six big gaps are OUTER-PASS
boundaries (host-side Q-eval/RL/DS decisioning + re-uploads between passes), which a lagged
per-iteration check cannot address. The wall is ~91% GPU-busy, dominated by full-width cuDSS
factorize (96 x ~237 ms across the run). Verdict: do NOT build the lagged check — the remaining
production-wall levers are (1) batch COMPACTION (shrink UBATCH width after retirement /
between outer passes — the structural fix, needs re-ANALYSIS per width) and (2) a small
~1.4 s outer-pass-boundary lever (device-side or overlapped outer-loop decisioning) — both
attack GPU-busy or pass-boundary time, not iteration stalls.

## 2026-07-17 — threadCount>1 crash fixed: device serialized, CPU fallback parallelizes

Design (user ruling): the GPU path is MONO-THREADED BY DESIGN — one device, its parallelism IS
the batch. Under threadCount>1: each partition's precomputeBatch queues on a global device lock
(no concurrent native calls -> no cuLaunchKernel SIGSEGV), the base/fall-through GPU solver swap
is skipped (concurrent sequential GPU solves would race; also sidesteps the open CME in that
config), and the CPU-side per-contingency fallback work parallelizes normally. Test:
threadCountAboveOneRunsSafelyOnGpuProvider (RTE params, 20 ctg, threadCount=2, statuses == stock
CPU). All parity gates green.

## 2026-07-17 — FAILED-base contract verified: honored by construction (no bug); property gate fixed

AbstractSecurityAnalysis only calls afterPreContingencySimulation (-> precomputeBatch) and the
post-contingency loop when the base LF SUCCEEDED — a failed base yields zero post results on the
GPU provider exactly like AcSecurityAnalysis. New test failedBaseLoadFlowYieldsNoPostContingencyResults
(CPU base forced, maxNR=1) proves it. The dense-factory observation that raised the question was
NOT a violation: the GPU arm's base genuinely converged (GpuNewtonRaphson does not use the dense
matrix factory) while the CPU arm's dense base failed — each side honest about its own base.
Real bug-let fixed en route: the EARLY solver-swap block ignored olf.gpu.sa.gpuBaseLoadFlow;
both swaps now honor the property.

## 2026-07-17 — CME (#15): repro attempt + exposure narrowed; still open at pegase scale

New regression test sequentialFallThroughOnGpuSolverWithReactiveLimits (multi-element ctg ->
sequential path on the swapped GPU solver, RL on, statuses vs stock CPU): GREEN at ieee14
scale — the small-case interleaving does not trigger the historical CME. The ORIGINAL crash
config (guard-v1 fallback with the solver swap still active, all 500 pegase ctg sequential on
the GPU solver under RTE params) is now STRUCTURALLY IMPOSSIBLE: fallback skips the swap, and
so does threadCount>1. Remaining exposure, still open: an ENGAGED pegase-scale run whose
fall-through contingencies (multi-element/islanding-rejected) solve sequentially on the GPU
solver with heavy RL switching. Next repro recipe: pegase + RTE params + engaged batch + one
multi-element contingency (needs a bench hook for multi-element ids). Suspected mechanism
unchanged: re-entrant mutation during LfNetwork listener dispatch (GpuAcDataCache Entry clears
are self-contained, so the suspect list now points at index-listener registration timing or
lazy es.getIndex() creation during dispatch).
