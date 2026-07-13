# GPU track — scope & roadmap

Status as of 2026-06-12 (branch `claude/level-a-closed-branch-formulas`, through
commit `93dfd89b`). Companion documents: `codegen/README.md` (the equation codegen),
`codegen/gpu_assembly.md` (the rung-by-rung journal).

## 1. Where we are

One native library (`libolfgpu`, jar-bundled, cuDSS linked dynamically) runs the FULL
Newton-Raphson on the device behind one JNI call, selected from a plain `LoadFlow.run`
via the AcSolver SPI (`GPU_NEWTON_RAPHSON`):

- **Formulas**: the four fused fill kernels (closed-branch, open-branch, shunt, HVDC
  AC emulation) are generated from the same residual specs as the Java formulas —
  CPU and GPU cannot drift (proven to 1e-12, fused audits ~1e-16).
- **Structure**: fixed SUPERSET CSR pattern in STABLE row indexing (a row is its
  diagonal-paired variable row: P↔BUS_PHI, Q↔BUS_V). OLF's renumbered equation
  columns are not in the GPU structure at all; slack/PV identities are a per-solve
  device mask. cuDSS analyzes the pattern once per topology.
- **Invalidation**: event-driven (`EquationSystemIndexListener` + `LfNetworkListener`,
  implemented directly so new OLF events force a compile-time classification).
  Two levels today — see §3.
- **Score** (PEGASE, warm wall clock, default LoadFlow parameters, KLU as CPU
  baseline):

  | case | CPU | GPU | verdict |
  |---|---|---|---|
  | 1354 | 81–102 ms | ~111 ms | CPU slightly ahead |
  | 2869 | 199–208 ms | 192–243 ms | parity |
  | 9241 | 690–901 ms | 614–804 ms | GPU ahead |
  | 13659 | 1145–1180 ms | 896–915 ms | GPU ~1.3× |

  Inside the GPU wall: one context build ~250 ms, then reruns prepare in 1–5 ms and
  solve in 8–15 ms; the remainder is OLF engine machinery common to both paths.

Scope today: plain AC power flow. The extractor validates and throws on anything
uncovered; benchmarks force `REPLACE_BY_MIN_IMPEDANCE_LINE` + local-only voltage
control to stay inside that scope.

## 2. Priority 1 — batched N-1 security analysis (the payoff)

### Why this is the goal

A single load flow gives the GPU ~1.3× at 13k buses because one Newton solve has
little parallel slack left. N-1 security analysis runs **hundreds to thousands of
post-contingency load flows on the same network**, each differing by one outaged
element. The whole architecture was chosen for this:
fixed pattern + analyze-once + value-only updates means the per-case costs that
dominate a single solve amortize over the entire contingency set, and the
scenario dimension provides the parallelism a GPU actually wants. Expected gain:
one-to-two orders of magnitude over sequential SA, not 1.3×.

### Why the architecture is ready

- A branch contingency is the SAME superset pattern with that branch's
  contributions zeroed — a **per-scenario element mask**, no pattern change, no
  re-analysis (the N-1 leaves entries structurally present, numerically zero; the
  diagonal-pairing guarantee keeps every row factorizable).
- The fill kernels already read all per-element state through index packs; adding
  a scenario dimension is a second grid axis (thread = element × scenario) writing
  into per-scenario value/mismatch sets.
- cuDSS offers uniform batched execution (same structure, many factorizations /
  RHS) — the same batched-solve machinery.

### Work items

Status legend: [x] implemented (pending GPU validation — no device in the dev env),
[~] partial, [ ] not started.

1. [x] **Native batch context**: `BatchSolverContext` (in `olf_gpu_nr_jni.cu`) shares
   the rowPtr/colIdx/diagPos pattern, the family index packs and cuDSS's analysis with
   all scenarios; values, F, x, dx, targets, rowMode are `[scenario][...]` device
   arrays. A per-scenario disabled-element mask (`char[nScenarios*nElements]`, element
   order: cb, ob, sh, hv) is consumed by the batched fill kernels
   (`if (!active[s] || disabled[s*nElements+off+e]) return;`).
2. [x] **Batched cuDSS**: both paths compiled into `olf_gpu_nr_jni.cu` and selected at RUNTIME
   by `-Dolf.gpu.batchMode` (mode 0 default / mode 1 batched). The UNIFORM BATCHED API (mode 1) builds one batch CSR matrix whose
   scenarios share the rowPtr/colIdx pattern with per-scenario value sets + batched dense
   B/X (`cudssMatrixCreateBatchCsr`/`cudssMatrixCreateBatchDn`), runs ANALYSIS once on the
   shared pattern, then ONE batched FACTORIZATION + SOLVE per Newton iteration; retired
   scenarios (converged/diverged) are frozen to an identity sub-system
   (`writeIdentityForInactive`) so a singular block never fails the batch. The per-scenario
   refactorize+solve loop (mode 0, the DEFAULT) is kept as the proven fallback. VALIDATED on
   cuDSS 0.8.0 (2026-06, RTX A1000): the batched API matches
   OLF to 1.5e-15 and runs 11.2x vs 7.5x for the fallback on IEEE-118 N-1. cuDSS 0.8's batch
   API needed: int32 host size arrays + `CUDSS_R_32I` integerType; the integerType arg on
   `cudssMatrixCreateBatchDn`; DEVICE arrays of device pointers (not host); `rowEnd=NULL`.
   (NB cuDSS 0.7.1 has no general batch API; it uses the
   `CUDSS_CONFIG_UBATCH_SIZE` single-matrix uniform batch instead, which 0.8 replaced.)
   cuDSS >= 0.7.1 is required (the cudssMatrixCreate* type-args are CUDSS_VERSION-shimmed so the
   same source builds against 0.7.x and 0.8); both modes are compiled and
   chosen at runtime, default mode 0. (Was a compile-time `-DOLF_GPU_BATCHED_CUDSS` switch.)
   (A block-diagonal variant — one big `blkdiag(J_1..J_N)` solved as a single matrix — was
   prototyped and BENCHMARKED then removed: correct but cuDSS does not exploit the block
   structure, so it falls to ~1.3-1.6x on PEGASE and OOMs at 9241 buses. Don't re-attempt;
   it is a known dead end. Footnote: on a 6 GB card BOTH batched
   modes OOM at PEGASE-9241 — they hold ~N factorizations — while only the =0 fallback
   scales there, so the chunk auto-size needs a tighter memory model for the batched modes.)
3. [x] **Convergence retirement**: per-scenario `active` mask checked host-side after
   each batched fill; converged scenarios stop filling/solving. Per-scenario iteration
   counts, final mismatch and status (0 converged / 1 max-iter / -1 diverged) are
   returned; a NaN/Inf scenario retires as diverged without poisoning the batch.
4. [x] **Java seam**: `GpuAcNewtonSolver.solveBatch(x0Batch, targetBatch, rowModeBatch,
   disabledMasks, ...)` → per-scenario `[state(n), iters, ||F||inf, status]`.
   `GpuBatchedSecurityAnalysisSolver.solveBatchN1` prepares scenarios from ONE base
   extraction; `GpuBatchElementMapper` maps each contingency's `LfBranch` to its pack
   position for the disabled mask.
5. [x] **Per-scenario targets/rowMode + qualification**: `GpuBatchContingencyValidator`
   qualifies each contingency against `LfNetwork.getConnectivity()` — a temporary edge
   removal reveals whether the outage islands buses from the main component. Contingencies
   that FRAGMENT the network (bridge branches) or are absent from the GPU model are
   REJECTED loudly and returned with `STATUS_REJECTED` (the caller routes them to the
   sequential CPU path) rather than silently mis-solved. `solveBatchN1` now batches only
   the qualified subset. Per-scenario targets/rowMode stay base-case for the qualified
   class (a single-branch outage keeping the main component intact does not change PV/slack
   roles there); the batch API already carries per-scenario targetBatch/rowModeBatch for
   when device-side outer loops arrive. CPU-only validator test:
   `GpuBatchContingencyValidatorTest`.
6. [x] **Validation**: `GpuBatchedN1CrossCheckTest` proves the batched result for every
   qualified single-branch contingency equals OLF's own Newton on the equation system
   with that branch disabled (`setDisabled(true)` → `onDisableChange` deactivates the
   branch's flow terms, the same model the GPU mask produces) — cross-checked on the
   self-contained meshed IEEE-14, to 1e-6. A benchmark on IEEE-118 times the batch vs the
   sequential per-contingency CPU solves over the same contingency set. (`GpuBatchedN1Test`
   already covered batch-replication == single solve and the disabled-mask plumbing.) The
   PEGASE-scale benchmark is `GpuBatchedN1BenchmarkTest` (1354/2869/9241): loads the
   LfNetwork with the GPU-scope parameters (min-impedance + local-only voltage control, as
   `GpuBenchmarkTest`) through a SPARSE matrix factory, then times the batch vs sequential
   per-contingency CPU solves over up to 1000 single-branch contingencies. Machine-local
   (skipped without the GPU lib + cases dir); the wall-time numbers come from running it.
7. [x] **Integration**: `GpuBatchedAcSecurityAnalysis` (subclass of `AcSecurityAnalysis`)
   wires the batch into OLF's SA engine at three seams — `runSimulations` (capture
   network + contingencies), `afterPreContingencySimulation` (precompute the batch from the
   converged base state), `runPostContingencySimulation` (INJECT each precomputed state via
   `AcSolverUtil.updateNetwork` instead of running the per-contingency Newton, then reuse
   the engine's violation detection + result building unchanged). Selected by the
   `olf.sa.gpuBatched` system property in `OpenSecurityAnalysisProvider`. Anything it cannot
   batch — multi-element / islanding / out-of-scope networks / non-converged / no GPU —
   falls through to `super`, so results are never wrong, only un-accelerated.
   `GpuBatchedAcSecurityAnalysisTest` asserts CPU-fallback equivalence with the standard SA.
   SCOPE (prototype): no device-side outer loops (distributed slack / reactive limits not
   re-run on GPU; distributed active power from pre-distribution only); operator strategies
   still run CPU-side from the injected state; single-threaded. Full outer-loop parity per
   contingency remains future work.

### Open design questions

- Distributed slack / reactive limits PER SCENARIO: the prototype runs plain NR
  per scenario (no outer loops device-side); post-contingency outer-loop parity is
  a later increment. Document the difference vs sequential SA when comparing.
- Memory: 13k buses × nnz≈300k values × f64 ≈ 2.4 MB/scenario for the value/state
  arrays, but the UNIFORM batched factorization holds an LU per scenario (~10–25 MB
  /scenario with fill-in at 13k) — that is the ceiling, ~1–2k scenarios on a 24 GB
  card. CHUNKING handles this: `GpuBatchedSecurityAnalysisSolver` solves the qualified
  set in chunks of at most `olf.gpu.sa.maxBatch` (default 256), reusing ONE chunk-sized
  context across chunks so cuDSS's analysis (shared pattern) is done once and amortized;
  the last chunk is padded with base-case scenarios (converge at iteration 0, discarded).
  The per-scenario fallback loop (batch mode 0) needs far less memory (one
  LU at a time), so it tolerates larger chunks. The chunk is AUTO-SIZED from free device
  memory: `GpuAcNewtonSolver.recommendBatchSize` builds the real Jacobian pattern (native
  `computePattern`), sizes the dominant per-scenario term — cuDSS's LU numeric factors —
  from the EXACT fill-in (`luBytesPerScenario` runs a cheap single-matrix probe ANALYSIS on
  the shared pattern and reads `CUDSS_DATA_LU_NNZ`, falling back to a heuristic multiplier
  if the query is unavailable, guarded by `OLF_GPU_CUDSS_LU_NNZ`), adds the value/state
  arrays, reads `cudaMemGetInfo` (after the probe frees its buffers), and divides the free
  memory (× a 0.7 safety margin). `olf.gpu.sa.maxBatch` overrides with an explicit manual
  cap; no GPU / query failure falls back to 256.

## 3. The invalidation taxonomy (and why some events are "STRUCTURE")

The cached device context bakes in, at `createContext` time:
(a) the **element packs** — `cbIn/obIn/shIn/hvIn` (physical parameters: y, ksi,
r1, a1, g, b, p0, k, …) and the index packs (WHICH elements exist and their
variable rows); (b) the **CSR pattern + destination arrays** derived from them;
(c) **cuDSS's symbolic analysis** of that pattern.

The per-solve ACTIVATION arrays (`rowMode`, `targetMap`) encode only *which
equation occupies each stable row* and *where its target comes from*. They cannot
express a changed element parameter or a changed element set. Hence the
classification rule: **an event is ACTIVATION only if everything it changes is
expressible in rowMode/targetMap; anything touching the packs, the element set, or
the pattern is STRUCTURE.**

Concretely, the four events asked about:

- **Tap position change** (`onTapPositionChange`): moves the transformer's `r1`
  (and possibly `a1`) — those are *values inside `cbIn`*, uploaded to the device at
  create. The activation arrays have no slot for them; the cached values would be
  silently stale. → STRUCTURE.
- **Shunt susceptance change** (`onShuntSusceptanceChange`): same, for `b` inside
  `shIn`. → STRUCTURE.
- **Zero-impedance events** (spanning tree change / split / merge): these change
  *which branches are modelled as impedant* and create/remove `ZERO_PHI`/`ZERO_V`
  equations — i.e. the element SET and the equation types change, both outside the
  GPU scope. The full re-extract triggers the validator, which (correctly) throws
  on zero-impedance equations instead of computing on a wrong model. → STRUCTURE,
  for fail-loudly as much as for caching.
- **HVDC AC-emulation status change** (`onHvdcAcEmulationStatusChange`): turns the
  HVDC's P terms on/off as *elements* — the hvdc enters or leaves the `hvIn/hvIdx`
  packs, which changes the pattern rows its angles touch. Not expressible as a
  mask over the existing pattern (the entries wouldn't exist / would linger).
  → STRUCTURE.

The honest nuance: tap and shunt-b changes alter **values only — the pattern is
unchanged**. Classifying them as STRUCTURE is conservative: it costs a full
re-extract + context rebuild (re-analysis included) where a *values-only refresh*
(re-upload `cbIn`/`shIn`, keep pattern + analysis) would suffice. This is exactly
the distinction OLF's own `JacobianMatrix` draws between `VALUES_AND_ZEROS_INVALID`
and `STRUCTURE_INVALID`. We have not added the third level because, in the
currently validated scope, those events never fire between solver runs (no
transformer/shunt controls are admitted), so the conservatism is free.

**Planned refinement (with scope widening, §4): a three-level taxonomy** —
`STRUCTURE` (element set / pattern: rebuild + re-analyze) ⊃ `VALUES` (element
parameters: re-upload packs into the existing context, no re-analysis) ⊃
`ACTIVATION` (rowMode/targetMap only). Tap moves and shunt-b changes then become
`VALUES`, which is precisely what transformer-voltage-control outer loops will
need to stay fast.

## 4. Priority 2 — robustness & scope (production readiness)

Each item below either improves the SOLVE (item 1) or LIFTS one of the
`GpuAcDataExtractor.validate()` rejections (items 2–4). Today validate() throws on:
non-`BUS_V`/`BUS_PHI` variables; equation types other than `BUS_TARGET_{P,Q,V,PHI}`;
non-power / non-`VariableEquationTerm` terms; and disconnection-allowed branches. Items
2–4 are exactly "admit one more variable/equation kind, end to end."

### Shared foundation (the real cost of items 2–4)

Items 2, 3 and 4 all need the SAME generalization, which is where most of the work and
risk lives, so build it once first:

- **Pattern beyond bus rows/cols.** The fixed CSR pattern + diagonal-pairing
  (`buildRowMap`: P↔`BUS_PHI`, Q↔`BUS_V` per bus) assumes every row pairs 1:1 with a
  `BUS_V`/`BUS_PHI` column. New variable kinds (`DUMMY_P/Q`, `BRANCH_ALPHA1/RHO1`,
  `SHUNT_B`) and new equation rows (`ZERO_V/PHI`, `DISTR_Q`, the target/distr rows) must
  get their own columns/rows with a pairing that keeps the diagonal factorizable for
  cuDSS's static pivoting. Generalize `buildRowMap` + the stable-row indexing to a
  per-element-type column/row allocation (buses first, then control/dummy columns), with
  an explicit diagonal partner per new row.
- **Extractor + index packs.** `GpuAcDataExtractor` learns the new variable/equation
  types (column/row numbers per element), and the per-family index packs gain the new
  column indices (today `-1`/skip).
- **Three-level invalidation** (already designed in §3): add the `VALUES` level so element
  parameter changes (tap `r1/a1`, shunt `b`) re-upload packs without re-analysis — needed
  by item 4 and by outer loops generally.

Once this exists, each of 2/3/4 is "stamp this family's extra rows/cols + scatter."

### 1. Device-side line search — independent, do FIRST

- **Scope.** Reproduce OLF's `LineSearchStateVectorScaling`: after `dx`, if the new
  ‖F‖ ≥ the previous, backtrack `x = x_prev + dx·μ` with `μ = 1/stepFold^k` (defaults
  stepFold 2, ≤5 trials) until the norm improves. Removes the device-side caveat
  (today GpuNewtonRaphson does undamped Newton only) and rescues stressed cases.
- **Approach.** Device only, no structure/codegen. Insert a backtracking sub-loop in
  `solveContext`/`solveBatchContext` between the cuDSS solve and accepting `x`: an
  F-only refill at the trial `x` (a `fillFOnly` variant, or reuse the full fill into a
  scratch vals buffer — the fill is cheap vs the solve), per-trial ‖F‖, halve μ.
  Batched: per-scenario `μ` array + a "still backtracking" mask, folded into the
  existing per-scenario norm/retirement pass; `axpy` takes the per-scenario μ. Java:
  `GpuNewtonRaphson` passes the line-search config (currently it ignores state-vector
  scaling) through `solveContext`.
- **Effort/risk.** Medium / low–medium. Only changes step length, not the converged
  target, so it is safe and independently validatable.
- **Test.** A stressed network where undamped NR diverges but OLF `LINE_SEARCH`
  converges → GPU-with-line-search state == OLF line-search NR; verify non-stressed
  cases are bit-unchanged (μ≡1).

### 4. Transformer / shunt controls as variables — do SECOND (lowest physics risk)

- **Scope & lift.** Make `a1` (`BRANCH_ALPHA1`), `r1` (`BRANCH_RHO1`), shunt `b`
  (`SHUNT_B`) state variables with their control rows (`BRANCH_TARGET_ALPHA1/RHO1`,
  `SHUNT_TARGET_B`/`DISTR_SHUNT_B`). Lifts the "unsupported variable type" reject.
- **Approach.** The kernels ALREADY compute `da1`/`dr1` (closed-branch) and `dqdb`
  (shunt) — today scattered to role-2 SKIP. The work is: un-skip those scatter columns
  (role 2→1) once the columns exist; add the control rows (a fixed phase/ratio target is
  a constant identity row; a voltage-/flow-control target reuses the existing branch
  P/Q-derivative outputs routed to the control row); and add the `VALUES` invalidation
  level so tap/shunt moves re-upload packs cheaply. No new physics — exercises the shared
  foundation with the least numeric risk, which is why it goes first among 2–4.
- **Effort/risk.** Medium–high (structure + invalidation) / medium.
- **Test.** Transformer voltage/phase control + shunt voltage control networks
  (OLF has these) vs CPU; tap-move reruns hit the `VALUES` path, not a rebuild.

### 2. Zero-impedance branches — do THIRD

- **Scope & lift.** Support zero-impedance branches (`ZERO_V`: v-equality, `ZERO_PHI`:
  θ-equality, with `DUMMY_P/Q` flow variables) so `REPLACE_BY_MIN_IMPEDANCE_LINE` is no
  longer required. Lifts the "unsupported equation type ZERO_V/ZERO_PHI" reject.
- **Approach.** The equations are LINEAR/constant (±1 entries: `ph1−ph2`, `v1−v2`, and
  the dummy flow injected ±1 into the adjacent buses' P/Q), so it's a constant-stamp
  family, not autodiff. The real work is structural: `DUMMY_P/Q` columns + `ZERO_V/PHI`
  rows in the generalized pattern with a sound diagonal partner, and the zero-impedance
  SUBNETWORK / spanning-tree handling (a tree change is a STRUCTURE event → full rebuild,
  already classified in §3).
- **Effort/risk.** High / high — it stresses the diagonal-pairing invariant the most
  (rows not owned by a single bus).
- **Test.** Networks with zero-impedance branches / breakers / zero-Z voltage control vs
  CPU, WITHOUT the min-impedance replacement.

### 3. Remote voltage control / `DISTR_Q` — do LAST (most coupled)

- **Scope & lift.** Generators controlling a remote bus and reactive-power sharing among
  several controllers (`DISTR_Q` rows; the controlled bus holds `BUS_TARGET_V`). Lifts
  the "unsupported equation type DISTR_Q" reject and the local-control-only requirement.
- **Approach.** The distribution COEFFICIENTS are constant, but each controller's Q is
  state-dependent, so a `DISTR_Q` row reuses the existing closed-branch Q-derivative
  outputs (at the controller buses) scaled by participation and routed to the distr row —
  i.e. an activation/scatter re-routing on top of the generalized structure, plus the
  controlled bus switching to a `BUS_TARGET_V` identity row. Most entangled with the
  row-mode/activation machinery, so it lands last.
- **Effort/risk.** High / high.
- **Test.** Remote voltage control + multiple-generators-on-one-bus networks vs CPU.

**Recommended order:** 1 (independent) → shared foundation → 4 → 2 → 3.

## 5. Priority 3 — performance polish

Mostly subsumed by batching, opportunistic otherwise: CUDA Graphs for the
iteration kernel sequence (launch overhead dominates the 8–15 ms reused solves),
per-context streams instead of device-wide synchronization, pinned host memory for
the mismatch copy, sort-based CSR position lookup instead of `std::map` in
`createContext`.

## 6. Priority 4 — infrastructure

- GPU CI job (self-hosted runner) executing `GpuFullNewtonTest` + `GpuLoadFlowTest`
  (`native/cuda/run_gpu_tests.sh`); CPU CI stays GPU-free (tests skip).
- Windows `libolfgpu` build (CMake is already arch-aware).
- If upstreaming to powsybl: split the branch into reviewable units
  (codegen | native packaging | solver SPI | benchmarks) — to execute AFTER the GPU
  validation pass.
