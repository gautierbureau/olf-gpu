# DC port plan — Woodbury security analysis + time-series load flow on the GPU

Written 2026-09-14. This worktree (`~/Projects/powsybl/powsybl-open-loadflow-gpu-1`, branch `dc`)
is the DC session; the AC session keeps `~/Projects/powsybl/powsybl-open-loadflow-gpu`
(`test_inteliji` == `master` at branch time, uncommitted work in flight). Both are worktrees of
the same repo, so commits are shared; `dc` merges into `master` at milestones.

## 1. Objectives (user, 2026-09-14)

1. **Woodbury DC security analysis on the GPU** — a port of OLF's `WoodburyDcSecurityAnalysis`
   (`dcFastMode`): the fast DC N-k that never re-factorizes.
2. **Time-series DC load flow on the GPU** — many DC load flows over ONE topology with varying
   injections. The API is `com.powsybl.openloadflow.ts.TimeSeriesLoadFlow` on the user's fork,
   branch `gb/claude/olf-timeseries-loadflow-hvdc-fixed` (the most complete of five stacked
   branches; §2.6). Not on OLF main.

3. **DC sensitivity analysis on the GPU** (user, same day: "a very important feature also") —
   a port of `DcSensitivityAnalysis` (PTDF / flow sensitivities with contingencies), which is
   built on the same `WoodburyEngine` and `ConnectivityBreakAnalysis` as objective 1.

All three objectives are the SAME device primitive: **one sparse LU of the DC matrix, reused
against many right-hand sides, with the per-column post-processing (Woodbury combination,
flows, limit screen, sensitivity values) done on the device so no n×N matrix ever crosses
PCIe.** Objective 1 builds the primitive and the Woodbury kernels; objective 3 reuses both and
adds the factor-states solve and a (function, variable-group, contingency) value kernel.

Explicitly NOT objectives: fast-decoupled AC (user: "very low"), a general DC solver SPI in OLF.

## 2. What OLF does today (the CPU reference — mirror it, do not reinvent)

All in stock OLF, all public or reachable from the `com.powsybl.openloadflow.sa` split package.
No OLF patch is expected for this port (see §7).

### 2.1 The DC system

`DcLoadFlowContext` builds an `EquationSystem<DcVariableType, DcEquationType>`:

| variables | equations |
|---|---|
| `BUS_PHI` per bus | `BUS_TARGET_P` per bus (the first slack bus's is INACTIVE) |
| `BRANCH_ALPHA1` per PST (when `forcePhaseControlOffAndAddAngle1Var`, which Woodbury forces ON) | `BUS_TARGET_PHI` for the reference bus; `BRANCH_TARGET_ALPHA1` per PST (identity rows) |
| `DUMMY_P` per zero-impedance branch | `ZERO_PHI` / `DUMMY_TARGET_P` — NOT present under Woodbury (it forces `minImpedance=true`) |

Branch term (`ClosedBranchSide1DcFlowEquationTerm`): `p1 = -power · (ph2 − ph1 + A2 − a1)`,
`power = b · (r1·R2 if useTransformerRatio)`, `b = 1/x` (IGNORE_R) or `x/(r²+x²)` (IGNORE_G).
The Jacobian is CONSTANT. `DcLoadFlowEngine.solve` is `jacobianMatrix.solveTransposed(target)`
because OLF stores J transposed (rows = variables, columns = equations); its `SparseMatrix` in
CSC of Jᵀ **is exactly the CSR of the true J** — the device can take `(columnStart, rowIndices,
values)` verbatim and do a plain solve, like the AC path does ("TRUE J, no transpose games").
No transposed solve is needed anywhere in the DC port (cuDSS 0.8.0 has none — see the AC repo's
task #40).

### 2.2 `WoodburyDcSecurityAnalysis.runSimulations` (the algorithm)

1. `cleanContingencies`; parameters forced: `minImpedance=true`, `hvdcAcEmulation=false`,
   `forcePhaseControlOffAndAddAngle1Var=true`. Actions checked by `Actions.checkWoodburySupported`
   (PST tap, switch, terminals-connection except transformer closing, generator, load).
2. Pre-contingency `x0 = WoodburyEngine.runDcLoadFlowWithModifiedTargetVector(ctx, DisabledNetwork())`
   — distributes slack (`distributeSlack`, P_RESIDUE_EPS 1e-3 MW, a plain host loop), clones the
   target, ONE solve. `updateNetwork`, pre-contingency `PreContingencyNetworkResult` + `LimitViolationManager`.
3. `ConnectivityBreakAnalysis.run`:
   - `ComputedElement.calculateElementsStates`: dense RHS `n × nDistinctBranches` with `+1/−1` at
     the two bus P rows of every contingency branch (one column per distinct branch; a slack-side
     entry is dropped since that row is inactive), ONE multi-RHS `solveTransposed(DenseMatrix)` →
     `S` (`contingenciesStates`). **This is the heavy op and the device-resident matrix**:
     rte6515 ≈ 6.5k × 9k × 8 B ≈ 470 MB; pegase9241 ≈ 9k × 16k ≈ 1.2 GB (fits the 6 GB A1000,
     column-chunk otherwise).
   - Sensitivity criterion per contingency: `Σ_j |p1_j.calculateSensi(S, col_j)| > 1 − 1e-6` over
     the contingency's OWN elements ⇒ "potentially breaking" ⇒ graph confirmation
     (`GraphConnectivity` temporary changes) ⇒ `ConnectivityAnalysisResult` (disabled buses,
     partial branches, `elementsToReconnect` = the minimal set that restores connectivity, which
     are then EXCLUDED from the Woodbury element list, HVDCs without power).
4. Per contingency (`calculatePostContingencyAndOperatorStrategyStates`):
   - Woodbury elements = branch contingency elements minus `elementsToReconnect` (+ action elements).
   - `setAlphas`: k×k system `A_ij = δ_ij / power_i − (S[ph1_i, j] − S[ph2_i, j])`,
     `b_i = x[ph1_i] − x[ph2_i]` (+ new PST α for tap actions), `α = A⁻¹ b` (k=1: scalar division);
     `x_post = x + S·α` (`toPostContingencyAndOperatorStrategyStates`).
   - The base `x` is `x0` UNLESS the contingency loses generation/load/HVDC power, islands buses,
     or loses a PST: then a modified target (`processLostPowerChanges` + `distributeSlack` on the
     remaining buses, islanded P rows zeroed, lost PST α rows zeroed) gets ONE extra solve on the
     same LU, and Woodbury is applied on top of that `x'`. Bus DC states are saved/restored around it.
5. Result: `stateVector.set(x_post)`, `updateNetwork`, `lfContingency.apply`,
   `PostContingencyNetworkResult.update`, `LimitViolationManager.detectViolations` (all host),
   `NetworkState.restore` per contingency. Operator strategies: `actionsStates` (same multi-RHS
   solve for action elements), connectivity re-check, k×k with actions.

Per contingency the CPU does O(n·k) arithmetic + a full host `updateNetwork`/result/violation
pass — the SAME host cost profile the AC port measured as "detection 27%" and moved to a device
screen. The fast DC is already fast; what the GPU can remove is the n-sized per-contingency host
work and the dense S round trip, and it can make the k×k + combination + flows + screen ONE kernel
launch per chunk.

### 2.3 Where the extension hooks in

`GpuSecurityAnalysisProvider.run` (`sa/gpu`) currently sends `isDc()` to `super.run` (stock
`OpenSecurityAnalysisProvider`, which picks `WoodburyDcSecurityAnalysis` when
`OpenSecurityAnalysisParameters.isDcFastMode()`). The DC hook is the same 3-line seam: if
`isDc && dcFastMode` (and the GPU is available) run `GpuWoodburyDcSecurityAnalysis`, else `super`.
The AC session owns that file — keep the DC edit to that branch of the `if`.

There is NO solver SPI for the DC engine (`DcLoadFlowEngine.solve` is static on
`JacobianMatrix`); the only stock injection point for a plain DC LF is the `MatrixFactory`
(`OpenLoadFlowProvider(MatrixFactory)`). That is why objective 2 is designed as its OWN batched
API (the OLF branch's time-series API), not as a solver swap.

### 2.4 `DcSensitivityAnalysis.analyse` (objective 3's reference)

Same context and parameters as Woodbury (it overrides the unsupported ones itself). Flow:

1. Factors read and grouped (`SensitivityFactorGroupList`: injection / PST phase / HVDC groups),
   `cleanContingencies`, participating elements for distributed slack.
2. `calculateFactorStates`: dense RHS `n × nGroups` (`initFactorsRhs`: `+1` at the variable's
   row(s), `−participation` at every slack-participating bus row; or `−1` at the slack when
   not distributed), ONE multi-RHS `solveTransposed` → `F` (`factorStates`).
3. `calculateFlowStates`: the reference flows = `runDcLoadFlowWithModifiedTargetVector`
   (one solve, slack saved/restored) → `flowStates`.
4. `ConnectivityBreakAnalysis.run` → `S` and the breaking/non-breaking split (shared with SA).
5. Per contingency (`calculateSensitivityValuesForContingencyAndOperatorStrategy`): a
   `WoodburyEngine` on the contingency's elements; `toPostContingencyStates(F)` applies the
   Woodbury correction to EVERY factor column (setAlphas per column, `F_post = F + S·α`),
   same for the flow column; when connectivity broke and the RHS changed
   (`rhsChangedAfterConnectivityBreak`) `F` is recomputed with the new participating set; lost
   gen/load → modified flow states like SA. Values: `sensi = p1.calculateSensi(F_post, group)`,
   `function = p1.calculateSensi(flow_post, 0)` — i.e. `power·(Δph1 − Δph2 + Δa1)` on the
   factor's branch; written through `SensitivityResultWriter` after unscaling and filtering.

The value for (function branch b, group g, contingency c with k=1) is
`sensi(b,g,c) = P_b·F[:,g] + α_{c,g} · P_b·S[:,c]` with `α_{c,g} = (F[ph1_c,g] − F[ph2_c,g]) / a_c`
— an outer-product of two small matrices (`P·F` is nFunctions × nGroups, `P·S` is
nFunctions × nElements) plus a scalar per (c, g). That is GEMM-shaped and is what the device
computes; the CPU loops it per factor.

Hook: `OpenSensitivityAnalysisProvider.run` builds `new DcSensitivityAnalysis(matrixFactory,
connectivityFactory, parameters)` inside a private helper → a `GpuSensitivityAnalysisProvider`
(`@AutoService`, name `OpenLoadFlowGpu`, same pattern as the SA provider) overrides `run` and
constructs `GpuDcSensitivityAnalysis`. `DcSensitivityAnalysis`'s helpers are PRIVATE, so the
GPU class either rewrites `analyse` (~300 lines of orchestration) or OLF widens 3-4 methods to
`protected` (`calculateFactorStates`, `calculateFlowStates`,
`calculateSensitivityValuesForContingencyAndOperatorStrategy`, `processContingencyAndOperatorStrategy`)
on `olf-local-patches` — decide at phase 4 start (see §7 on OLF patches).

### 2.6 `TimeSeriesLoadFlow` (objective 2's reference — fork branch, 25 commits on 6ef8e6fc7)

Entry point: `TimeSeriesLoadFlow.run(network, List<DoubleTimeSeries> plan, TimeSeriesLoadFlowParameters,
NetworkResultWriterFactory) -> TimeSeriesLoadFlowResult(List<StepResult>)`.

- **Plan** = powsybl-core `DoubleTimeSeries`, one per element, series name = element id, values =
  active power in MW per step; a series drives a generator (`targetP` + `initialTargetP` +
  `reApplyActivePowerControlChecks`), a load (`LfLoad.setOriginalLoadP0`, q0 optionally following
  at constant power factor), an angle-droop HVDC (`AcEmulationControl.setP0`) or a fixed-set-point
  VSC HVDC (both stations as generators, inverter losses applied). LCC rejected. Absolute values
  only. Read ONCE on the main thread (`DoubleTimeSeriesValues`).
- **Contract**: step N == an independent `LoadFlow.run` on the network carrying that step's
  values, independent of step order and thread partitioning. So per step: `NetworkState.restore()`
  (snapshot taken BEFORE any solve), apply setpoints, `DcLoadFlowEngine.run()` — i.e. the DC
  engine's own slack distribution (host, pre-solve, from targets) and outer loops — then emit.
- **Output** streamed through core's `NetworkResultWriter` (`com.powsybl.loadflow.resultswriter`,
  core fork branch `claude/core-generalize-result-writer`, 7 commits, CSV + Parquet + in-memory
  impls, 3WT dataset): branches (`stateId=ISO instant, subStateId="", status, branchId,
  p1,q1,i1,p2,q2,i2,flowTransfer`) via the allocation-free `LfBranch.emitBranchResults` sink
  (zero-impedance flows via `ZeroImpedanceFlows` on host), buses (`v, angle`), generators
  (`targetP` requested, `p` after slack). Non-converged steps stream too, tagged.
- **Threads**: step range partitioned, one `LfNetworkList` + context per partition, one
  writer part-file per partition.
- **Measured CPU behaviour** (design doc §8): DC re-solves are NOT pure back-substitutions today
  because writing the state fires `onStateUpdate` → `VALUES_INVALID` → numeric LU update per
  step. Fix: **upstream PR #1475** (SylvestreSakti, OPEN since 2026-08-07, "Avoid recomputation of Jacobian Matrix in DC mode when state update" — implemented differently from the
  fork's `gb/claude/olf-dc-jacobian-freeze`; prefer upstream's, never carry both).
- Tests: `TimeSeriesLoadFlowTest` (24, incl. `dcStepsMatchIndependentDcLoadFlow`,
  `dcLoadStepsMatchIndependentDcLoadFlow` at 1e-2 MW vs `LoadFlow.run`), Parquet round-trip.

Where the CPU time goes per step (to MEASURE first, phase 3.0): restore (all buses/branches,
listener fan-out), target-vector rebuild O(n), LU update (until the upstream fix), solve,
`updateNetwork` O(n), emit O(branches) rows into CSV/Parquet. The device can take the solve and
the flows; it pays only if the host per-step work is also cut (§3.3).

### 2.7 DC prior art on the user's OLF fork (CPU — the GPU host side must not be slower than this)

None of these is in `olf-local-patches` or upstream main (checked 2026-09-14). Merge probes onto
`olf-local-patches` (git 2.34, real `merge --no-commit` in a throwaway worktree):

| branch | what | merge |
|---|---|---|
| `gb/claude/lodf-factors-dc-sa-i1ttsi` (23 c.) | `LodfCalculator`/`LodfComputer`: LODF(l,k) = PTDF(l,k)/(1−PTDF(k,k)) from ONE multi-RHS ±1 solve, column-batched at 256 MB, streaming `LodfResultWriter`, threadCount, self-PTDF>1−1e-6 ⇒ NaN column; + a **vectorized DC equation system** (`DcNetworkVector`, array evaluators) | CLEAN |
| `gb/pr/vectorized-dc` (3 c.) | vectorized DC es, opt-in, + DC LF/SA benchmark | CLEAN |
| `gb/pr/fastdc-sa-violations` (3 c.) | skip bus-voltage detection in fast DC; hoist disabled map; no per-call limits supplier | CLEAN |
| `gb/pr/fastdc-sa-skip-bus-update` (1 c.) | skip per-contingency bus-angle write when no output needs it | CLEAN |
| `gb/pr/fastdc-sa-limit-snapshot` (2 c.) | precompute branch limit groups once (−19% on 13659pegase/1000 ctg) + `limitViolationReporting` most-restrictive option | not probed (shares 433cfa81f) |
| `gb/pr/fastdc-sa-incremental-detection` (2 c.) | **τ-prefix screen**: a branch cannot violate when spread(Δθ)·\|power\| ≤ L−\|p_base\|; sort by τ once, check the prefix — bit-identical, detection −90%, SA −33% (13659pegase, 1000 ctg ≈ 7.3 s → ~5 s, i.e. ~5 ms/ctg CPU) | 1 conflict (`LimitViolationManager`) |
| `gb/pr/fastdc-sa-restore` (7 c.) | restore only contingency-impacted elements, positional bus state | 3 conflicts (`BusState`, `BusDcState`, `LfNetworkListener` — overlaps the AC session's OLF patches) |
| `gb/pr/dc-sensi-restrict` (3 c.) | Woodbury correction restricted to monitored rows in DC sensitivity | not probed |
| `gb/claude/olf-dc-jacobian-freeze` (1 c.) | `DcJacobianMatrix` skipping the LU update when only the state moved (2.1–3.8×) — SUPERSEDED by upstream PR #1475 | CLEAN |
| `gb/claude/olf-timeseries-loadflow-hvdc-fixed` (25 c.) | the time-series API (§2.6) + the SA flow-streaming seam (`monitorAllBranches`) | 1 conflict (`AbstractSecurityAnalysis`) |
| `gb/claude/dc-sensitivity-matrix-chunking-70xia7` | design doc only: chunk the factor RHS matrix | — |

Consequences: (a) the τ-screen IS the host-side form of our device limit screen — the device
version computes every flow anyway, so the screen there is a plain compare; (b) the LODF closed
form is exactly phase 1's N-1 kernel (`p_l(c) = p_l + LODF(l,c)·p_c`), and `LodfCalculator` is
the CPU oracle for the device `S`/flow tables at 1e-12; (c) the honest CPU arm for benches is
`olf-local-patches` + the fast-DC PRs (+ upstream PR #1475), not plain main — the STARTING
branch deliberately leaves `fastdc-sa-violations` and `fastdc-sa-skip-bus-update` out (user,
2026-09-15); report which PRs the CPU arm carries in every bench row.

### 2.8 Prior art already on this machine

- `powsybl-core-1` (`cudss_factory`): `CuDssMatrixFactory` / `CuDssLUDecomposition` — cuDSS LU
  with `solve(double[])`, `solve(DenseMatrix)` (direct ByteBuffer, host↔device copy per call),
  transposed variants via an explicit flag. **Phase 0 baseline needs zero new code.**
- `powsybl-math-native-1` (`cudss_solver_1`): the `libmathcudss` JNI behind it.
- The AC extension: `olf_gpu_violations.cu` (flat `GpuLimitTable` + `screenViolationsBatch`,
  "the kernel screens, the host decides", 0 differing lines at rte6515/2000), cuDSS
  handle/loader plumbing in `GpuAcNewtonSolver` (preloads libcudss from `CUDSS_ROOT`),
  `GpuTestIsolationExtension`, the vendored `CommonTestConfig`/`ServiceParameterResolver`, the
  validation runner's `--sa` mode (violation-parity gate, all-branch N-1, RTE parameters).

## 3. Device design

One new native translation unit `native/cuda/jni/olf_gpu_dc_jni.cu` added to the existing
`olfgpu` target (one line in `CMakeLists.txt`, the only shared native touchpoint with the AC
session). Java side in NEW packages only: `com.powsybl.openloadflow.dc.gpu` (solver/context) and
`GpuWoodburyDcSecurityAnalysis` in `com.powsybl.openloadflow.sa` (it needs the package-private
`sa` machinery; `GpuBatchedAcSecurityAnalysis` set the precedent of extending the protected ctor).

### 3.1 `GpuDcContext` (the primitive both objectives share)

```
createDcContext(n, csrRowPtr, csrColIdx, csrVal)      // TRUE J from JacobianMatrix's SparseMatrix, cuDSS analyze+factor ONCE
solveMany(ctx, rhs[n × m] on device or host)          // cuDSS multi-RHS solve, result stays on device (handle)
branchTables(ctx, nBranches, rowPh1[], rowPh2[], rowA1[], power[])   // the flow formula's inputs, uploaded once
flows(ctx, statesHandle, m) -> p[nBranches × m]       // p_b = power_b · (x[ph1]−x[ph2]−A2+x[a1])   (one thread per (b, col))
destroyDcContext(ctx)
```

Row indices are the equation-system's own `Variable.getRow()`/`Equation.getColumn()` (stable
for the run: the DC matrix never changes structure during a Woodbury SA, and the time-series
API keeps one topology by definition). No extractor/validator of the AC kind is needed: the
device consumes the CSR OLF already assembled. Keep a `refreshValues(ctx, csrVal)` for the
time-series case where injections change but the matrix does not (no-op) and for a later
"topology changed, refactor" path.

### 3.2 Woodbury on the device

**Where the win is, and what it must be worth (user, 2026-09-15: "DC SA on GPU with Woodbury
should pay").** The fast DC SA on the CPU costs ~5–7 ms per contingency once the fork's perf PRs
are in (13659pegase, a limit on every branch, 1000 contingencies: ~7.3 s → ~5 s), and it is
almost entirely per-contingency HOST work: the n-vector Woodbury combination, `updateNetwork`,
the result walk and the limit detection. The device does the same per contingency in
O(nBranches·k) flops — microseconds — with ONE launch per chunk, and the dense `S` (470 MB at
rte6515) never leaves the device. What stays on the host is bounded by the number of
VIOLATIONS, not by n × contingencies: candidate confirmation and `LimitViolation`/result objects.
Target for the phase-1 gate: **≤ 0.5 ms/contingency end-to-end on rte6515 × 9042 (S solve,
connectivity analysis and result building included), ≥ 10× the fork-PR CPU arm, at 0 differing
violations.** That is why the device limit screen and the device-side result assembly belong
in phase 1, not later: without them the port would only move the cheap arithmetic.


- `elementStates(ctx, k, rowPlus[], rowMinus[])` builds the ±1 RHS ON the device (no n×k host
  matrix) and solves → `S` device-resident (column chunking when `n·k·8B` exceeds a budget;
  budget from `cudaMemGetInfo`, the A1000 has 6 GB and the AC session may hold some).
- `connectivityScreen(ctx, S, contingency→elements)` → per contingency
  `Σ_j |power_j·(S[ph1_j,c_j] − S[ph2_j,c_j])|` vs `1 − 1e-6`; returns the "potentially
  breaking" mask. The host keeps OLF's graph confirmation for those (it is the exact semantics,
  and it is a small share — measure on rte6515).
- `woodburyFlows(ctx, x, S, contingency list (k ≤ K_MAX in-kernel, e.g. 8), a1 overrides)`:
  one block per contingency: gather the k×k `A` and `b`, dense solve in registers/shared
  (k=1 is a division), then for every branch `p_b(c) = p_b(x) + Σ_j α_j · power_b · (S[ph1_b,j] − S[ph2_b,j])`
  — the post-contingency FLOWS are computed directly; the n-vector `x_post` is never
  materialized unless the host asks for bus results (`readStates(c)` for monitored-bus/
  `createResultExtension` runs). k > K_MAX → host (OLF's own `WoodburyEngine`), never wrong.
- `screen(ctx, flows, limitTable)` → the AC flat-table screen reused for DC branch limits
  (permanent + temporary active-power/current limits; DC has no voltage violations; the
  current limit uses `dcPowerFactor` — mirror `ClosedBranchDcCurrent`). Candidates back to the
  host; the host builds the `LimitViolation`s with OLF's own manager on the confirmed set.
- Modified-base contingencies (lost gen/load/HVDC power, islands, lost PST): the host computes
  the target delta exactly as OLF does (`processLostPowerChanges`, `distributeSlack`, zeroed
  rows) — cheap, keeps OLF's slack semantics byte-for-byte — and ships ONE sparse delta per
  contingency; `solveMany` on the stacked deltas gives every `x'` in one call; Woodbury then
  applies on that column. Phase 1 routes these to the CPU path; phase 2 batches them.

### 3.3 Time-series load flow on the device

`GpuTimeSeriesDcEngine`, a DC `StepEngine` alternative behind `TimeSeriesLoadFlow` (the engine
is chosen in `buildEnginePlan`; the GPU one is selected by an extension-side parameter or by
the `OpenLoadFlowGpu` provider name once the ts API grows a provider seam — decide with the
user when the branch is integrated). Per partition:

1. Host, once: `DcLoadFlowContext` → CSR → `createDcContext`; `branchTables`; snapshot.
2. Host, per step, UNCHANGED semantics: `restore → applySetpoints → DC slack distribution`
   (`DcLoadFlowEngine`'s own pre-solve distribution, from targets — cheap, and it is what makes a
   step equal `LoadFlow.run`), then take `context.getTargetVector().getArray()` as column j of a
   chunk of B steps. No `DcLoadFlowEngine.run`, no `updateNetwork`.
3. Device, per chunk: `solveMany` (B columns) → `flows` (nBranches × B) → one download of the
   flow chunk (+ angles when `streamBusResults`).
4. Host: stream rows straight from the downloaded buffers through the SAME `NetworkResultWriter`
   (branches from `power·Δθ` in SI; zero-impedance branch flows from the `DUMMY_P` rows of the
   state / `ZeroImpedanceFlows` on host; buses `v=NaN or 1`, angle; generators `initialTargetP`,
   `targetP` after slack from the host arrays). Statuses: a step whose solve failed is tagged like
   OLF does (last iterate streamed).

Golden: byte-comparable to the CPU engine at 1e-12 on every streamed value, and to
`TimeSeriesLoadFlowTest`'s independent `LoadFlow.run` references. Bench: steps/s cold and warm,
CPU 1- and N-thread vs GPU, at rte6515 with an 8760-step synthetic plan (the user supplies real
plans). Expectation to state honestly: the DC solve is ~1 ms on KLU, so the win comes from
removing the per-step `updateNetwork`/`emitBranchResults` walk and batching, not from the solve.

**AC time-series is the bigger GPU win and is already ~built**: the AC session's batch context
solves nS scenarios with per-scenario target deltas (`setBatchTargetDeltas`) and in-kernel
DS/RL loops — an AC step is a scenario with no topology mask. Hand that to the AC session as a
consumer of the same ts API; not part of this DC plan.

### 3.5 DC outer loops on the device (user, 2026-09-15: "plan to port them in kernels like for AC")

Where they matter: **Woodbury SA runs NO outer loop** (linear one-shot; it forces phase control
off and AC emulation off), so objective 1 needs none. They matter for the **time-series engine**
(every step is a full `DcLoadFlowEngine.run`) and for the non-fast `DcSecurityAnalysis` (full
re-solve per contingency). OLF's DC loop list, innermost first (`DefaultDcOuterLoopConfig`):
`DcHvdcAcEmulationLimits` → `DcIncrementalPhaseControl` → `DcAreaInterchangeControl`; the
engine nests them exactly like the AC engine (re-run a loop to stability, sweep the list until
`lastUnstableOuterLoop` fixpoint, `maxOuterLoopIterations` budget). Slack distribution is NOT a
DC outer loop: `DcLoadFlowEngine.run` distributes ONCE, pre-solve, from the targets
(`P_RESIDUE_EPS` 1e-3 MW, failure behaviors), then solves.

The AC session's device driver already implements OLF's nested-loop ORDER over a batch of
scenarios (loop list, per-scenario stability flags, `maxOuterLoopIterations`, host-owned loop
slots via pause/resume, DS-carry rules). The DC port reuses that driver shape over the
time-series chunk (a column = a step) with DC loop kernels. What each DC loop needs, and what
the matrix does:

| loop | what the CPU does | matrix impact in DC | device port |
|---|---|---|---|
| pre-solve slack distribution | `getActivePowerMismatch` from targets, one `ActivePowerDistribution.run` over participating gens/loads with limits, failure behavior | RHS only | kernel per column (`dsBatch`): mismatch = Σ target column, single-pass distribution with the AC DS kernel's factor types/limits/failure modes (`GpuDsParams` exists), writes the distributed targets into the column + the per-generator `p` for the generators dataset. Host fallback first (it is cheap); kernel when the per-step host walk shows in the 3.0 profile |
| `DcIncrementalPhaseControl` | sensitivities dP/dα: one RHS per controller with `toRadians(1)` at its `BRANCH_TARGET_ALPHA1` row, `solveTransposed`; then OLF's incremental decisions (deadband, MAX_DIRECTION_CHANGE, min sensi, tap steps) → new α target | RHS only (α is a variable pinned by an identity row) | sensitivities = `solveMany` ONCE per topology (the matrix is constant — unlike AC they never go stale); decision kernel = the AC `GpuBatchIncrementalFlowControl` logic per column reading the controlled flows from `flows`; writes the α target column; re-solve only the unstable columns (active mask) |
| `DcHvdcAcEmulationLimits` | `checkAcEmulationMode`: linear ↔ saturated (CS1→CS2 / CS2→CS1) when \|p\| crosses Pmax; saturated = fixed injection ±Pmax | **matrix values change**: the droop `k` couples θ1−θ2 in J; saturating removes it | two options, pick by measurement: (a) Woodbury on the shared LU — a saturation is the "outage" of a pseudo-branch of power `k` between the two converter buses (element states ±1 like a branch, k×k α) plus an RHS shift of ±Pmax; reuses phase-1 kernels, no refactor; (b) per-column matrix values on the fixed pattern with cuDSS UBATCH (the AC session's context kind) — heavier, general. Mirror OLF's transition table verbatim (incl. the opposite-Pmax flip) |
| `DcAreaInterchangeControl` | area interchange balancing = distributions over area bus sets, then remaining slack | RHS only | host-owned loop slot via pause/resume first (AC marks AIC low priority, user 2026-09-07); kernel = the DS kernel with area grouping if a plan needs it |

Sequencing per chunk: `ds → solve → flows → hvdcCheck → (re-solve) → phaseCheck → (re-solve) →
aic` with per-column stability flags and OLF's sweep-until-fixpoint, all columns of the chunk
advancing together, converged columns masked. Statuses (`MAX_OUTER_LOOP_ITERATIONS`, DS failure)
tagged per column exactly as `DcLoadFlowResult` would.

Gate for every loop: `TimeSeriesLoadFlowTest`-style goldens vs an independent `LoadFlow.run`
per step at 1e-12, one test per loop that FAILS when the kernel is disabled (the AC lesson: a
hollow test passed for a whole session), and the OL_TRACE-style sequence diff against the CPU
engine (diff the loop CALL SEQUENCE, not the quantities).

This would also make the **non-fast `DcSecurityAnalysis`** batchable later (NOT in the first
features wanted — user, 2026-09-15; recorded only as a consequence): a contingency is a
column with a topology change, which in DC is the Woodbury element list (phase 1) on top of the
same loop driver — full DC SA with outer loops = Woodbury + §3.5, no new kernels.

## 4. Phases and gates

Every phase lands with its parity test at **1e-12** (linear algebra: the only differences vs
KLU are LU-ordering roundoff; anything above 1e-10 is a bug — never tighten a tolerance to
agree). Every phase appends a row to `docs/numbers.md` (cold first).

- **Phase 0 — baseline, no new code (day 1).** Run stock `WoodburyDcSecurityAnalysis` with
  `CuDssMatrixFactory` vs `SparseMatrixFactory`, rte6515 all-branch N-1, RTE parameters +
  `dcFastMode`; same on the fast-DC-PR OLF (§2.7) so the CPU arm is the user's fastest. Gives: (a) the CPU wall split (S solve vs per-contingency host work) via a
  profiler/`Stopwatch` probe, (b) whether the dense S round trip already pays, (c) the parity
  harness (`GpuVsCpuDcSaTest`, violation sets + statuses, 0 differing lines expected at the
  default eps since nothing is iterative). Decides how much of §3.2 is worth building first.
- **Phase 1 — `GpuWoodburyDcSecurityAnalysis`, N-1/N-k without power loss, WITH the device
  screen.** Device `S`, connectivity screen, in-kernel Woodbury + flows, the flat limit table +
  `screen` kernel (host confirms candidates and builds the violations), result assembly from
  device flows for monitored branches (bus results only when asked). Contingencies with lost
  gen/load/HVDC/islands/PST → the parent's CPU path per contingency (a `cpuRouted` census like
  the AC scope log). Gate: OLF's own DC-fast tests rerun against the GPU provider
  (`OpenSecurityAnalysisTest` ×5, `OpenSecurityAnalysisProviderTest` ×2,
  `WoodburyDcSecurityAnalysisWithActionsTest` ×2 via the `securityAnalysisProvider` field
  reassignment), `LodfCalculator` as the 1e-12 oracle for `S`/flows, rte6515 9042 violation
  parity 0 differing, and the §3.2 target (≤ 0.5 ms/ctg, ≥ 10× the fork-PR CPU arm) or an
  explanation of why not.
- **Phase 2 — the modified-base classes batched.** Target-delta columns for lost power /
  islands / lost PST (one more `solveMany`), `OLF_GPU_VIOL_CROSSCHECK`-style A/B of the screen.
  Gate: 100% batched on 9042 (verify on the FULL list, never a 2000 window), 0 differing, cold +
  warm ms/ctg vs CPU 1-thread and 4-thread fork-PR Woodbury.
- **Phase 3 — time-series LF (§3.3).** 3.0: integrate the ts branch (§7, one conflict) and
  measure the CPU per-step split on rte6515 × 8760 steps (restore / targets / LU / solve /
  update / emit) — this decides how much host work the GPU engine must bypass. 3.1: the device
  engine on `solveMany` + `flows` (built in phase 1). Gate: every streamed value == the CPU
  engine at 1e-12, `TimeSeriesLoadFlowTest` green against the GPU engine, steps/s cold and warm
  vs CPU 1- and N-thread. 3.2: the DC outer loops on the device (§3.5) in OLF's order — slack
  distribution kernel, phase control (sensitivities once + decision kernel), HVDC AC-emulation
  limits (Woodbury pseudo-branch first), AIC as a host-owned slot; each with its
  fails-when-disabled golden.
- **Phase 4 — DC sensitivity analysis (objective 3).** `F` = `solveMany` on the factor RHS
  (built on device from the group rows + participation weights), `S` and the connectivity split
  shared with phase 1, one kernel over (function branch, group, contingency) producing
  `sensi` and `function` values; the host keeps factor reading/grouping, the
  `rhsChangedAfterConnectivityBreak` re-solve (one more `solveMany`), unscaling, filtering and
  the `SensitivityResultWriter`. Gate: `DcSensitivityAnalysisTest` (58) +
  `DcSensitivityAnalysisContingenciesTest` (80) rerun against the GPU provider at 1e-12; a real
  PTDF-shaped bench (all-branch functions × many injection groups × N-1) cold and warm vs CPU.
  Can start right after phase 1 (it does not need phase 2's modified-base batching to be
  useful — lost-power contingencies CPU-route first, as in SA).
- **Phase 5 (optional) — operator strategies** in SA and sensitivity (action states = one more
  `solveMany`, k×k with actions, connectivity re-check).

## 5. Validation and benches

- Parity oracle = stock OLF Woodbury on the same `LfNetwork` (same slack, same
  `cleanContingencies`), compared as: post-contingency branch P (all branches, 1e-12 pu),
  violation sets keyed `subject|limitType|side|limitName` (the validation repo's SA gate), and
  per-contingency statuses. Remember the AC lesson: a GPU test that only compares to CPU cannot
  detect that the GPU never ran — assert an engagement counter (`lastGpuPrecomputedCount`-style).
- Real data: add `--dc` (sets `LoadFlowParameters.dc=true` + `dcFastMode=true` on both arms) to
  the validation runner's `--sa` mode; the user runs it (never read `.arc` data).
- Benches: CPU Woodbury 1-thread and 4-thread vs GPU, cold E2E first, warm second, at rte6515
  9042 and pegase9241; a DC section in `docs/numbers.md`. The competitor name stays out of
  code/tests/commits (benchmark markdown only).

## 6. Risks and open questions

- **Device memory**: `S` is `n × nBranches`; 1.2 GB at pegase9241 on a 6 GB card shared with
  the AC session. Column chunking is in the design from day 1.
- **Slack-side elements**: `fillRhs` drops the slack-bus entry (its P row is inactive) — the
  device RHS builder must reproduce that exactly, and the reference-bus `BUS_TARGET_PHI` row is
  an identity row like the AC path's.
- **Multi-slack** (`p.setActive(false)` only for the first slack per SC; others via the target
  vector) — follow OLF's rows, never assume one slack.
- **Islanding semantics**: `elementsToReconnect` (excluded from Woodbury) + zeroed injections +
  `hvdcsWithoutPower` — take OLF's `ConnectivityAnalysisResult` verbatim; only the screen is
  ours.
- **Time-series API unknown** until the branch is shown — design device-side around "a column
  is a target vector".

## 7. Session mechanics (parallel with the AC session)

- **Branch**: `dc` in this worktree (a worktree cannot check out `master`/`test_inteliji`
  while the AC checkout holds them). Merge `dc` → `master` at each phase gate; rebase on
  `master` before merging so the AC session's work is picked up. The "commit on master" rule
  is honored at merge time.
- **Shared touchpoints with the AC session, keep them tiny**: `CMakeLists.txt` (one source
  line), `GpuSecurityAnalysisProvider.run` (the DC branch of one `if`), `docs/numbers.md`
  (append-only), `docs/gpu-vs-cpu-capabilities.md` (a new DC section). Everything else is new
  files under `dc/gpu`, `sa/GpuWoodburyDc*`, `native/cuda/jni/olf_gpu_dc_jni.cu`,
  `native/cuda/dc/`.
- **No OLF patches from this session for SA and time-series.** Everything those need is public
  or reachable from the `sa` split package. Sensitivity (phase 4) may need 3-4 `private →
  protected` widenings in `DcSensitivityAnalysis` (§2.4). Any OLF patch goes on
  `olf-local-patches` in `powsybl-open-loadflow-3` (the AC session's checkout, with uncommitted
  work in it) and needs one coordinated rebuild — say so explicitly before doing it, and batch
  the widenings into ONE commit.
- **Integration DONE 2026-09-15 — the DC session's dependency worktrees:**
  - core: `~/Projects/powsybl/powsybl-core-dc`, branch `cudss_factory-dc` = `cudss_factory` +
    `gautierbureau/claude/core-generalize-result-writer` (clean). Installed as **`7.4.0-SNAPSHOT`**
    (same coordinates as the AC session's core — a strict superset: only `loadflow-api`,
    `loadflow-results-parquet`, `security-analysis-api` and the `loadflow` aggregator pom were
    rebuilt; suffixing ~100 core modules was judged disproportionate). If the AC session ever
    re-installs core from `cudss_factory`, the writer classes vanish → rebuild from this worktree.
  - OLF: `~/Projects/powsybl/powsybl-open-loadflow-dc`, branch `olf-local-patches-dc` =
    `olf-local-patches` (a143817d7) + `gb/claude/lodf-factors-dc-sa-i1ttsi` (contains
    `pr/vectorized-dc`) + upstream PR #1475 (fetched as local `pr-1475`) + the ts branch (merge
    48215a287, two conflicts resolved by hand) + a 3-line `DcNetworkVector` listener adaptation
    (upstream moved `onTapPositionChange` and added `onEquationTermConstantsChanged`). Installed
    as **`2.4.0-SNAPSHOT-gpu-dc`** (+ tests jar). Rebuild: `cd ~/Projects/powsybl/powsybl-open-loadflow-dc
    && mvn -s ~/Projects/powsybl/tmp/settings.xml install -DskipTests -Dcheckstyle.skip -Dmaven.javadoc.skip -Dlicense.skip`.
  - extension `dc` worktree pom pins `2.4.0-SNAPSHOT-gpu-dc` and math-native `1.6.0-SNAPSHOT-gpu`.
  - NOT in the starting branch (user): `pr/fastdc-sa-violations`, `pr/fastdc-sa-skip-bus-update`.
  - Upstream PR #1475 changes two listener signatures (`onTapPositionChange` gains
    `modifiedTapImpedance`; `EquationSystemIndexListener.onEquationTermConstantsChanged` is new).
    Three AC-owned extension files were adapted mechanically on `dc` (`GpuAcDataCache`,
    `GpuSolverInvalidation`, `GpuHybridBatchOuterLoopEngine`); the AC session will need the same
    the day it takes PR #1475 or an upstream containing it — expect a trivial conflict at merge.
  - Merge sanity on the OLF DC branch: `TimeSeriesLoadFlowTest`, Parquet streaming, Woodbury
    actions, `DcJacobianMatrixTest`, `LodfCalculatorTest`, DC vectorization, `DcLoadFlowTest`,
    `DcSensitivityAnalysisContingenciesTest`, SA flow streaming — all green (2026-09-15).
- **Original integration notes (kept for the rationale):**
  1. powsybl-core: `cudss_factory` + `gautierbureau/claude/core-generalize-result-writer` merge
     CLEAN (15 files, pure additions: `loadflow-api/.../resultswriter`, new module
     `loadflow-results-parquet`, `AbstractSecurityAnalysisRunParameters`). Install as
     `7.4.0-SNAPSHOT` is a strict superset of what the AC session uses — still announce before
     installing; or install under `7.4.0-SNAPSHOT-dc` and pin it in the `dc` worktree's pom.
  2. OLF: new worktree (e.g. `~/Projects/powsybl/powsybl-open-loadflow-dc`) on branch
     `olf-local-patches-dc` = `olf-local-patches` + `gb/claude/olf-timeseries-loadflow-hvdc-fixed`
     (1 conflict, `AbstractSecurityAnalysis.java`) + `vectorized-dc` + `lodf-factors-dc-sa` +
     upstream PR #1475. **NOT at the start (user, 2026-09-15): `pr/fastdc-sa-violations` and
     `pr/fastdc-sa-skip-bus-update`** — keep them in mind (they merge clean) for the CPU bench arm
     or a later integration, but the starting branch does not carry them.
     (rebase or cherry-pick; NOT the fork's `olf-dc-jacobian-freeze`). The ts branch
     also bumps OLF's core to `7.4.0-SNAPSHOT` and adds `powsybl-time-series-api`. Install as
     `2.4.0-SNAPSHOT-gpu-dc`; the `dc` worktree's pom pins that. Rebase this branch when the AC
     session moves `olf-local-patches`.
  3. The `fastdc-sa-restore` and `fastdc-sa-incremental-detection` branches conflict with the AC
     session's OLF patches (`BusState`, `LfNetworkListener`, `LimitViolationManager`) — take them
     only for the CPU BENCH arm, resolved by hand, or leave them out and say so in the bench row.
- **Dependency version**: the AC session is mid-switch (uncommitted, 2026-09-14) from
  `2.4.0-SNAPSHOT` to `2.4.0-SNAPSHOT-gpu` for OLF and `1.6.0-SNAPSHOT-gpu` for math-native;
  both jars exist in `~/Projects/powsybl/tmp/m2`. This worktree's `pom.xml` must match whatever
  is installed when it builds — check `ls ~/Projects/powsybl/tmp/m2/com/powsybl/powsybl-open-loadflow/`
  before the first build, and take the AC session's pom change at the first rebase.
- **Builds**: `mvn -s ~/Projects/powsybl/tmp/settings.xml ...` (isolated repo, never `-o`,
  never `~/.m2`); natives via `native/build-gpu.sh` (its cache `native/build/` and
  `native/local-env.sh` are per-worktree and gitignored — `local-env.sh` was copied in on
  2026-09-14; `native/cudss-cache/` is only for `cudss-swap.sh`, not needed).
- **The GPU is one 6 GB A1000 for both sessions**: do not run the big benches concurrently
  with the AC session's; unit tests on IEEE/small cases are fine in parallel.
- **The device gets the RUN's parameters** (dcPowerFactor, balanceType, useTransformerRatio,
  dcApproximationType, limit reductions) — never `new ...Parameters()` in the GPU path.
