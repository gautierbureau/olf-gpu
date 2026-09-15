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

**Do we need kernels to assemble the DC matrix on the device? (user question, 2026-09-15)**
Not for objectives 1 and 3 alone — but the user then asked for the **full DC load flow assembled
and run on the GPU (2026-09-15)**, and that is now what the device path does (§4f): the matrix
VALUES, the target vector, the slack distribution and the solve are device work; the host keeps
the equation system's STRUCTURE, which it captures once (`GpuDcAssembly`) as contribution lists
in OLF's own `initDer` / `DcTargetVector.init` order, so the device sums are OLF's sums —
bit-identical values and targets, tested against `SparseMatrix.getValues()` and
`TargetVector.getArray()`. No codegen family was needed: the DC "formulas" are `±power`,
`∓power·(A2−a1)`, `±1`, `±k` — the wiring is the risk, and the bit-identical oracle on OLF's own
assembled system is a stronger proof than a parse of the term classes.

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

### 3.5 DC outer loops on the device (user, 2026-09-15: "plan to port them in kernels like for AC") — LANDED, see §4g

Where they matter: **Woodbury SA runs NO outer loop** (linear one-shot; it forces phase control
off and AC emulation off), so objective 1 needs none. They matter for the **time-series engine**
(every step is a full `DcLoadFlowEngine.run`) and for the non-fast `DcSecurityAnalysis` (full
re-solve per contingency). OLF's DC loop list, innermost first (`DefaultDcOuterLoopConfig`):
`DcHvdcAcEmulationLimits` → `DcIncrementalPhaseControl` → `DcAreaInterchangeControl`; the
engine nests them exactly like the AC engine (re-run a loop to stability, sweep the list until
`lastUnstableOuterLoop` fixpoint, `maxOuterLoopIterations` budget). Slack distribution is NOT a
DC outer loop: `DcLoadFlowEngine.run` distributes ONCE, pre-solve, from the targets
(`P_RESIDUE_EPS` 1e-3 MW, failure behaviors) — that one is a device kernel since §4f.

What each DC loop needs, and what the matrix does:

| loop | what the CPU does | matrix impact in DC | device port |
|---|---|---|---|
| pre-solve slack distribution | `getActivePowerMismatch` from targets, one `ActivePowerDistribution.run` over participating gens with limits, failure behavior | RHS only | **DONE (§4f)**: `distributeSlack` kernel on the device generator tables, bit-identical to `ActivePowerDistribution` |
| `DcIncrementalPhaseControl` | sensitivities dP/dα: one RHS per controller with `toRadians(1)` at its `BRANCH_TARGET_ALPHA1` row, `solveTransposed`; then OLF's incremental decisions (deadband, MAX_DIRECTION_CHANGE, min sensi, closest tap) → new tap | RHS only (α is a variable pinned by an identity row) | **DONE (§4g)**: `phaseRhs` + one multi-column solve on the load flow's own factorization, then `phaseDecision` — one thread per controller, the deadband, `calculateSensi`, `da = toRadians(dp/a2p)`, the closest-tap search over the allowed range and the direction ratchet, all in-kernel; the host applies `setTapPosition` |
| `DcHvdcAcEmulationLimits` | `checkAcEmulationMode`: linear ↔ saturated (CS1→CS2 / CS2→CS1) when \|p\| crosses Pmax; saturated = fixed injection ±Pmax | **matrix values change**: the droop `k` couples θ1−θ2 in J; saturating removes it | **DONE (§4g)**: `hvdcDecision` kernel = OLF's transition table on the emulation flows read from the state; the host calls `AcEmulationControl.switchTo…`. The matrix follows through the CONTRIBUTION refresh (`setContributions`), not a Woodbury pseudo-branch and not UBATCH: OLF emits a matrix element for every variable of every term whether active or not, so **the non-zero pattern is invariant** and a saturation is a value refill on the same analysis |
| `DcAreaInterchangeControl` | area interchange balancing = distributions over area bus sets, then remaining slack | RHS only | **host-owned loop slot** (landed as such): it is not a criterion on the state but a distribution loop with its own iteration, per-area factors and reporting. The engine runs OLF's own check, interleaved with the device loops. A kernel would be the DS kernel with area grouping, if a plan ever needs it |

Sequencing: OLF's own, unchanged — the device only replaces the `check` of the two loops that
read the state. For the time-series engine the same kernels apply per column with per-column
stability flags (phase 3).

Gate for every loop: a scenario where the loop MUST fire, compared to the stock provider, plus a
device-decision counter so a kernel that decided nothing fails (the AC lesson: a hollow test
passed for a whole session), plus the A/B against `-Dolf.gpu.dc.hostOuterLoops=true`.

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

## 4b. Phase 0 log (2026-09-15)

Harness: `src/test/.../sa/gpu/dc/DcWoodburyBaselineProbe` (rte6515_full symlinked at the repo
root from the AC checkout — untracked; all-branch N-1 of the main synchronous component = 9042;
RTE parameters with `dc=true`, `dcFastMode`; knobs `OLF_DC_MAX_CTG`, `OLF_DC_THREADS`,
`OLF_DC_MONITORED`, `OLF_DC_SKIP_CUDSS`, `OLF_DC_SKIP_KLU`; `main()` for plain-JVM profiling with
`java -XX:StartFlightRecording ... -cp target/test-classes:target/classes:$(mvn dependency:build-classpath)`).

**Two bugs in the core fork's cuDSS binding (`libmathcudss`), both fixed on
`powsybl-math-native-dc` branch `cudss_solver_dc` (c36d2c4, d0e97a4), published as
`powsybl-math-native 1.6.0-SNAPSHOT-gpu-dc` (the `-gpu` jar with the rebuilt .so; the `dc`
worktree pins it):**
1. **NaN on any OLF Jacobian with an empty structural diagonal.** OLF stores J with rows =
   variables and columns = equations, numbered independently; in the DC system every bus after
   the slack (whose inactive P equation shifts the equation numbering) has NO diagonal entry —
   6530 of 6536 columns on rte6515. KLU pivots through it; cuDSS without matching returned NaN
   (silently — the SA does not check the pre-contingency solve). Fix: `CUDSS_CONFIG_MATCHING_ALG
   = AUTO` before the analysis. Standalone on the dumped matrix: cuDSS == KLU to 1.3e-12, 1 and
   300 RHS (and a 6536 × 9042 full-S solve in ONE call works, |S|max 1.74).
2. **SIGFPE inside libcudss on a dense solve with ZERO columns.** The Woodbury SA issues one on
   every connected component that owns no contingency element (`calculateElementsStates` with an
   empty list → an n × 0 RHS). Fix: `solveInto` returns when `cols == 0`. (Isolated by
   replaying the SA's solve sequence step by step — same matrix, same thread kinds, same RHS
   pattern all passed; only the empty component reproduced it.)
   The AC session's `CuDssLoadFlowTest` never saw either: AC's sorted index happens to align the
   bus P/Q rows with the φ/V columns, and a plain LF never solves a 0-column RHS.

**Results so far** (1 thread, RTX A1000 6 GB, i7-13700H; the second arm runs in the same JVM):

| run | KLU | cuDSS drop-in | parity |
|---|---|---|---|
| 300 ctg, 100 monitored | 3536 ms (11.8 ms/ctg) | 2957 ms (9.9 ms/ctg), ×1.20 | 0 differing lines; max violation value diff 3.2e-10 MW; monitored P1 max diff 2.5e-10 MW over 29897 values |
| 9042 ctg, KLU only, 0 monitored | 57.8 s (6.4 ms/ctg), 98955 violations, all CONVERGED | — | — |
| 1500 ctg, KLU only | 11.3 s (7.5 ms/ctg) | — | — |

| **9042 ctg, 100 monitored (the phase-0 result)** | **65.3 s (7.22 ms/ctg)** | 66.2 s (7.32 ms/ctg), ×0.99 | **0 differing lines** over 98955 violations; max common-violation value diff 1.16e-9 MW (1.2e-11 pu); monitored P1 max diff 2.5e-10 MW over 904093 values |

**Where the CPU time goes** (JFR, KLU arm, 9042 ctg, 0 monitored, 75.8 s under the profiler,
6327 samples on the SA worker thread; self-frame attribution):

| share | what |
|---|---|
| ~45% | `LimitViolationManager.detectViolations` (`detectBranchSideViolations` 18%, `detectBranchCurrentViolations` 16%, `getLimits1/2` 2%, the rest) — every branch, every contingency |
| ~35% | `NetworkState.restore` per contingency (`AbstractElement.setDisabled` 9%, `LfLoadImpl.get(Map)` 8%, `BranchState.restore` 6%, `BusDcState.restoreGenerators` 2.5%, `setOriginalLoadsDisablingStatus`, `BusState.restore`) |
| ~6% | the `isBranchDisabled` predicate (`getBranchesStatus().containsKey`) evaluated per branch inside detection |
| 3.8% | the modified-target re-solve (lost power / islands / PST contingencies) |
| 1.4% + 1.1% | `updateNetwork` (bus angles) + `DcTargetVector.init` |
| 0.3% | `WoodburyEngine` alphas + n-vector combination |
| ~0% | the `S` multi-RHS solve, the connectivity analysis, the LU |
| 8% | JVM/GC, no powsybl frame |

So the linear algebra is under 5% of the fast DC SA; the drop-in cuDSS LU cannot pay (measured
×0.99) and phase 1 must remove the host per-contingency work: detection → device screen (the
fork's τ-prefix PR attacks the same 45% on the CPU), restore → nothing to restore when the
device owns the state (the fork's `fastdc-sa-restore` PR attacks the same 35%), the disabled
predicate → a per-scenario mask. That is the ≥10× in §3.2.

The cuDSS-vs-KLU disagreement (~1e-9 MW) is LU-ordering roundoff on a ~4e4-conditioned DC
matrix (`max|v|/min|v|` ≈ 3.9e4/0.58) — reported, not tolerated: the phase-1 device path will be
held to 1e-10 pu against OLF's KLU results (cond·eps ≈ 4e-12 relative is the floor for two
different direct solvers; the probe asserts 1e-8 MW), and the violation gate stays 0 differing lines.
Standalone timing of the LU itself: KLU factor+solve of the 6536 matrix ≈ 4–5 ms (1 RHS), 15–31
ms (300 RHS); cuDSS ≈ 400–500 ms warm per context (analysis + factorization dominate; the AC
memory's "one context ≈ 425 ms" holds here too) — the drop-in wins only because the SA's
per-contingency host work, not the LU, is the wall (§3.2).

## 4c. Phase 1 + 2a log (2026-09-15)

**Landed** (`dc` commits 9fed038 phase 1, f0e981e phase 2a, 660867b phase 2b): `GpuDcSolver` + `olf_gpu_dc_jni.cu` (cuDSS
factorization with matching, device-resident S built from the ±1 rows, per-chunk alphas
k ≤ 8 in registers, states, flows, the relaxed limit screen, per-contingency base columns from
host-built modified targets); `GpuWoodburyDcSecurityAnalysis` (stock pre-contingency and
connectivity analysis; device for every pure branch outage INCLUDING islanding ones and lost
PSTs; host confirms candidates on OLF's evaluables and builds the stock results; no apply, no
restore); `DcSlackDistributionModel` (OLF's generation distribution on arrays — its trap:
`NetworkState.save` re-anchors `initialTargetP` to the base dispatch before the loop, so the
per-contingency water-filling starts from the base, not the original network; 240 differing
lines until that was mirrored). Stock-routed per contingency: lost power from propagation,
HVDC without power, shunt shifts, bus loss, > 8 Woodbury elements, load-based balance types,
multi-slack; whole run: operator strategies/actions.

**Comparison rule (user, 2026-09-15): GPU single-thread vs CPU with 8 threads.**

| rte6515 × 9042, RTE parameters in DC, 100 monitored | wall | ms/ctg | parity |
|---|---|---|---|
| stock Woodbury, KLU, 1 thread (phase 0) | 65.3 s | 7.22 | reference |
| stock Woodbury, KLU, **8 threads** | 17.3 s | 1.92 | reference |
| GPU Woodbury, 1 thread, phase 1 (6458 on device, 2584 breaking on stock) | 29.3 s | 3.24 | 0 differing lines |
| **GPU Woodbury, 1 thread, phase 2a (9042/9042 on device)** | **13.4 s** | **1.49** | **0 differing lines**, flows ≤ 2.5e-10 MW |

| **GPU Woodbury, 1 thread, phase 2b (device criterion, refinement off), 0 monitored** | **3.65 s** | **0.40** | 0 differing lines |
| GPU Woodbury, 1 thread, phase 2b, 100 monitored | 8.3 s (CPU 8 threads: 18.4 s) | 0.92 | 0 differing lines, flows ≤ 2.5e-10 MW over 904093 |

**Phase 2b split** (0 monitored, 3.65 s): pre-contingency 0.06 s, cuDSS factorization 0.23,
device S (9042 columns, IR off) 0.54, connectivity 0.51 (device criterion + OLF's graph
confirmation of 2648 potentially breaking), modified-target columns 0.50 (2584), device chunks
0.47, host per-contingency loop 1.12 (0.12 ms: state load, candidate confirm, stock result
objects), the rest outside (LfNetwork load, contingency list). **The §3.2 target is met on the
0-monitored run: 0.40 ms/ctg, ×17.9 the 1-thread CPU, ×4.5 the 8-thread CPU.** With 100
monitored branches the host loop grows to 5.7 s on BOTH arms: it is OLF's own
`PostContingencyNetworkResult.addResults` (a per-contingency walk with string-keyed set
lookups — the JFR of the GPU run is 60% `String.hashCode` under `HashSet.contains`), which the
stock path pays identically; an OLF-side fix (index the monitored branches once) would help
both arms and is upstream-worthy, not a GPU item.

Phase 2a split (`OLF_GPU_PHASE_TIME`): setup 4.9 s (cuDSS init + factorization, device S, OLF's
host connectivity analysis with ITS KLU S, limit table, NetworkState.save), modified-target
columns 1.07 s (2584 islanding outages, 0.41 ms each), device calls 1.12 s, host per-contingency
5.2 s (0.58 ms each: state load + candidate confirm + stock result objects). The §3.2 target
(≤ 0.5 ms/ctg end-to-end, ≥ 10× the 1-thread CPU) is NOT met yet: 1.49 ms/ctg, 4.9× the
1-thread CPU, 1.3× the 8-thread CPU. Next levers, in order of size: the host per-contingency
work (profile below), the setup (device connectivity criterion instead of the host KLU S; the
limit table and the model are already once-per-run), the column build (mismatch by a plain
loop instead of a bus-set copy; the distribution model's boxed lists).

## 4d. Phase 2c log (2026-09-15) — the remaining classes

Every contingency class the stock Woodbury handles now goes to the device except operator
strategies (phase 5) and two shapes the stock does not handle either or the kernel does not
size for (a lost SLACK bus; more than 32 Woodbury elements). Mechanism: the stock's lost-power
branch reproduced on the host as an exact replay (bus DC states saved, `processLostPowerChanges`,
`distributeSlack` on the remaining buses, the target vector re-read from the mutated network,
islanded P and disabled α rows zeroed, everything restored) for the classes the array model does
not cover — lost generation/load/HVDC, HVDC without power (stations added to the lost generators
FIRST, as the stock does, and the reported `LfContingency` built BEFORE that addition), load-based
balance types, multi-slack — while plain islanding keeps the array replay. Bus losses are OLF's own
cleanup (incident branches opened) plus the islanded bus; shunt shifts have no DC effect. A
contingency without any branch element is a modified target alone (k = 0), and a component
without any element (the far side of an HVDC link) works. Parity 1e-10 pu vs stock on nine cases
(`GpuWoodburyDcSecurityAnalysisTest`). The device count accumulates over connected components.

## 4e. Phase 5 log (2026-09-15) — operator strategies

Landed (`dc`, the phase-5 commit): strategies run on the device as a second pass. What stays on
the stock path per contingency is now only a lost SLACK bus (unsupported in DC by the stock too)
and a scenario with more than 32 Woodbury elements. Mechanism: action elements' ±1 columns appended
to S with `WoodburyEngine`'s per-element diagonal and new alpha shipped to the kernel; per scenario
the condition on the post-contingency violations, the connectivity re-check (device criterion,
OLF graph confirmation), the exact replay of the modified target with the actions, Woodbury over
contingency + action elements, the stock result with contingency and actions applied on the
network then restored (`NetworkState`, one per scenario — strategies are few). OLF fix found on the
way: the stock Woodbury NPE'd on generator/load actions (`olf-local-patches-dc` 02cdae225,
upstream-worthy). Parity 1e-10 pu on OLF's own tap scenarios, two taps, switch open/close on every
outage of the node-breaker network, generator + load set points (13 tests in
`GpuWoodburyDcSecurityAnalysisTest`).

**Base solve on the device (2026-09-15, after phase 5).** The pre-contingency DC solve is the
device's too (same target vector, cuDSS instead of KLU), so nothing of the DC linear algebra runs
on the CPU on the device path any more; KLU is only touched by a stock-routed subset. What stays on
the host by nature: LfNetwork load, contingency propagation, the base slack distribution and the
target vector, the pre-contingency result and violations, the graph confirmation of flagged
contingencies, the modified-target columns (host replay / array model) and the result objects.
**Timing caveat worth knowing**: the cuDSS context creation (CUDA init + analysis + factorization)
varies from 0.22 s to 2.6 s between otherwise identical runs while the AC session shares the GPU;
the per-contingency work is stable (~0.2 ms). Report cold numbers with that spread.

## 4f. Full DC load flow on the device (2026-09-15, user: "Full LF DC should be assembled and run on gpu")

Landed on `dc`:
- **`GpuDcAssembly`** (Java, `dc.gpu`): the structure capture. Walks the equation system exactly
  like `JacobianMatrix.initDer` (single equations and the `BUS_TARGET_P` array interleaved by
  column; `SingleEquation.der`'s TreeMap-by-variable order; `EquationArray.der`'s stable sort by
  variable, runs of equal rows closed by the additional single terms, then the leftover single-term
  rows) and records per matrix element an ordered list of contributions `(kind, element, coef)`:
  `coef·power[b]` for a closed-branch term (side 1: +,−,+ on ph1, ph2, a1; side 2 opposite),
  a constant for a variable term (±1) or an HVDC AC-emulation term (±k). Same for the target:
  a base kind per column (`BUS_P` with the multi-slack share, `ALPHA1`, `ZERO_PHI`, 0) and an
  ordered rhs list (`∓power·(A2−a1fixed)`, `±p0`). Every classification is checked at capture
  against the term's own `der`/`rhs` value (`Double.compare`), an unknown term class throws
  `UnsupportedOperationException` and the caller keeps the host path. Also the bus tables (P
  column, angle row, load target P, flags, synchronous component, generator range), the
  synchronous components' bus lists and slack counts, and the generator tables in OLF's iteration
  order (bus, then `bus.getGenerators()`).
- **Native** (`olf_gpu_dc_jni.cu`): `createAssembled` (structure only, cuDSS analysis on the
  first `fillValues`), kernels `fillValues` (one thread per non-zero, `v = 0; v += …` in OLF's
  order, `__dadd_rn`/`__dmul_rn` so nvcc cannot fuse), `busGeneration` (`getGenerationTargetP`'s
  sum from 0.0 in generator order), `scMismatches`, `fillTarget` (`DcTargetVector.init` per
  column), `activePowerMismatch` (sequential, one thread), and **`distributeSlack`** — one thread,
  sequential like the CPU: `ActivePowerDistribution.run` + `GenerationActivePowerDistributionStep`
  with the reference-generator reset and the step reset accumulated SEPARATELY and added at the
  end (as OLF does — accumulating them in one running sum was a last-ulp difference), the JDK's
  compensated `DoubleStream.sum()` for the factor norm (the `sum − comp` final, verified 20000/20000
  against the JDK on random data), Java's `Math.max/min` semantics, iterator removal by
  compaction. `setBuses/setBusLoads/setExcludedBuses/setGenerators/setGeneratorTargets/
  readGeneratorTargets/readValues/readTarget/solveTarget` round it out.
- **`GpuDcLoadFlowEngine`**: `DcLoadFlowEngine.run` line for line — `initStateVector`, the
  device mismatch, the device (or, for load-based balance types, host) distribution with OLF's
  `handleDistributionFailureBehavior` and reports, the device target + solve, the outer loops
  (host `check`, re-`prepare()` after an unstable iteration: same pattern → refill + refactorize,
  structure change → rebuild), cleanup, `buildDcLoadFlowResult`. Protocol: the LfNetwork is the
  source of truth between phases; the device distribution's generator targets are written back at
  once. `solveBase()` is the Woodbury pre-contingency solve (distribution over all buses without
  epsilon/failure handling, target, solve).
- **`GpuLoadFlowProvider`** (`"OpenLoadFlowGpu"`, `@AutoService(LoadFlowProvider)`): OLF's
  provider with the DC engine swapped through a new protected seam `runDcEngine` on
  `olf-local-patches-dc` (217214135; also `DcOuterLoopContext`'s constructor made public,
  c8e4e563f). It returns no JSON parameter serializer of its own (OLF's is keyed by extension
  name and a duplicate throws in `JsonLoadFlowParameters`). AC runs → stock.
- **The Woodbury SA** builds its device context from the assembly (no `JacobianMatrix`, no KLU
  on the device path at all) and takes its base from `solveBase()`.

Gates (all green 2026-09-15):
- `GpuDcAssemblyTest` (14 dynamic cases: ieee14 scalar and vectorized, distributed P_MAX / P /
  LOAD, three slack buses, eurostag, IGNORE_G, phase shifter fixed / regulating / alpha variable,
  HVDC AC emulation, zero-impedance line, ieee118): pattern identical, values and target
  **bit-identical** to OLF's, device solve vs KLU < 1e-10 pu, and the distribution kernel
  bit-identical to `ActivePowerDistribution` for the four generation types × limits on/off ×
  four mismatches (targets, remaining, iterations).
- `GpuDcLoadFlowProviderTest` (23 cases incl. every balance type, multi-slack, the regulating
  phase shifter outer loop, HVDC emulation, zero impedance, the three distribution failure
  behaviors on ieee57, THROW parity, ieee118/300): component results equal, angles 1e-7°, flows
  1e-8 MW, generator and load P bit-identical, tap positions equal.
- `GpuWoodburyDcSecurityAnalysisTest` 13/13 on the new base; rte6515 × 9042: 9042/9042 on the
  device, 0 differing lines, 5.3–5.6 s GPU 1-thread vs 15.2–16.7 s CPU 8 threads
  (capture 39 ms; cuDSS analysis+factorization 2.1 s in these runs — the known spread).
- `DcLoadFlowGpuProbe` (rte6515 full DC LF, RTE parameters): flows within 1.7e-10 MW, angles
  1e-11°, generators bit-identical; **cold GPU 2.5 s vs CPU 0.66 s, warm 225 vs 154 ms** — a
  single DC solve is not where the device pays (the cuDSS analysis alone outweighs KLU's whole
  run); the assembled context pays when reused: the SA above, the time-series engine next.

Left for later (noted, not blocking): the SA's per-contingency modified-target columns still
come from the host (array model / exact replay) — the device `setExcludedBuses` +
`distributeSlack` + `fillTarget` can build them too (islanding class first; lost power needs
`processLostPowerChanges` on the tables); OLF's own target vector goes stale by one ulp after
`updateNetwork` writes the solved alpha back (nothing invalidates it on `setA1`) — the device
engine re-captures the current alpha instead, a 1e-17 difference the gate cannot see.

## 4g. DC outer loops as kernels (2026-09-15, user: "Do the DC outer loops as kernels")

Landed on `dc`. The split is the AC driver's: **the kernel READS the state and produces the loop's
DECISION, the host APPLIES it with OLF's own mutators**, so the equation updater, the listeners,
the reports and the network result stay OLF's.

- **Incremental phase control** (`phaseRhs` + `phaseDecision`, `GpuDcOuterLoops.checkPhaseControl`).
  The sensitivities are `DcSensitivityContext.calculateSensitivityValues` on the device: one
  right-hand side per controller with `toRadians(1)` at its `BRANCH_TARGET_ALPHA1` column, solved
  against the load flow's own factorization (chunks of 1024 columns), recomputed per check exactly
  as OLF rebuilds its context per check. The decision kernel is one thread per controller and
  reproduces `checkActivePowerControlPhaseControls` step for step: the controlled flow from the
  state (`DcNetworkVector.updateClosedBranches`'s formula, same operation order), the half
  deadband (`max(targetDeadband, 1 MW)/2`), `calculateSensi` read off the solved column
  (side 1 `[p, -p, p]`, side 2 opposite), `da = toRadians(dp / a2p)` guarded by `SENSI_EPS`, then
  `PiModelArray.updateTapPositionToReachNewA1`'s `ClosestTapPositionFinder` over the range the
  allowed direction permits. `IncrementalContextData.ControllerContext`'s ratchet
  (`MAX_DIRECTION_CHANGE`, current direction, allowed direction) lives on the device, per branch,
  across checks. The host then calls `PiModel.setTapPosition`.
- **HVDC AC-emulation limits** (`hvdcDecision`, `GpuDcOuterLoops.checkHvdc`). The kernel evaluates
  the two emulation flows on the state (`k (ph1 - ph2) + p0` / `k (ph2 - ph1) - p0`, `k = droop·180/π`)
  and applies `checkAcEmulationMode`'s transition table verbatim, including the
  opposite-Pmax flip; the host calls `switchToSaturationFromCS1toCS2` / `…CS2toCS1` /
  `switchToLinearMode` (so the VSC target P and the equation deactivation are OLF's) and emits
  OLF's three reports.
- **The matrix follows a saturation with no new analysis.** A saturating link deactivates its two
  emulation terms, and OLF emits a matrix element for every variable of every term *whether active
  or not* (`SingleEquation.der`'s comment: "create a derivative even if all terms are not active");
  so the non-zero PATTERN is invariant and only the contribution lists change. `prepare()` now
  splits `sameStructure` into `samePattern` + a `setContributions` upload, then refills and
  refactorizes. Rebuilding the context (a new cuDSS analysis) is left for a genuine structure change.
- **Area interchange control stays on the host**, deliberately: it is a distribution loop, not a
  criterion on the state (per-area factors, its own iteration, its own reports). The engine runs
  OLF's check for it, interleaved with the device loops. Its dispatch is the one quantity that is
  NOT bit-identical between the arms — it distributes *from the solved state*, so it inherits the
  two solvers' roundoff (1e-13 MW measured); the pre-solve distribution stays bit-identical.
- `-Dolf.gpu.dc.hostOuterLoops=true` routes both checks back to OLF (the A/B).

Gates (all green 2026-09-15), `GpuDcOuterLoopTest`, 8 cases on OLF's own scenarios: the phase
shifter that must move taps (`DcLoadFlowTest.outerLoopPhaseShifterTest`, tap 1 → 2) and one with an
unreachable target (the ratchet), the HVDC that must saturate either way
(`AcLoadFlowVscTest.testAcEmuAndPMax`, DC arm), the saturation side flip
(`testAcEmulationSaturationToOppositeChange`), both loops interleaved, area interchange + phase
control (`DcLoadFlowTest.multipleOuterLoopsTest`'s network), and the host/device A/B. Each asserts
the stock result AND a device-decision counter (`GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES` /
`LAST_DEVICE_HVDC_SWITCHES`), so a kernel that decided nothing fails twice.
Regression: `GpuDcAssemblyTest` 14, `GpuDcLoadFlowProviderTest` 23, `GpuWoodburyDcSecurityAnalysisTest`
13; rte6515 × 9042 SA 4.8 s GPU 1-thread vs 15.6 s CPU 8 threads, 0 differing lines.

Note on the OLF scenario transposed to DC: `AcLoadFlowVscTest` saturates `hvdc12` at Pmax 51 MW,
but in DC that link carries 49.7 MW and never reaches it — the DC tests use 20 MW, which makes
both loops fire on the same network.

## 4h. LODF and PTDF on the device (2026-09-15, user: "Now what about ptdf and lodf on gpu?")

**Landed**: `GpuLodfCalculator`, mirroring the API and every convention of the CPU
`LodfCalculator` on the user's OLF branch, and the two kernels it needs. **Verdict: as a factor
PRODUCER the device does not pay.** The measurements are below; the reason is structural and it
tells us exactly where the sensitivity work should go.

The port itself is small because the ingredients were already device-resident: the CPU
calculator's "injection states" are the SAME ±1 element states the Woodbury SA keeps on the
device, and its `calculateSensi` is the same expression the connectivity criterion already
evaluates (and bit-for-bit so: the scalar term's `eval(dph1,dph2,da1) - eval(0,0,0)` equals
`power·((dph1-dph2)+da1)` exactly, because IEEE subtraction is antisymmetric and A2 = 0). So:
`lodfSelfPtdf` (self-PTDF per column) + `lodfBlock` (one thread per monitored branch × outage,
with OLF's NaN/-1 conventions) + a `sensiBlock` kernel for raw PTDF, streamed in outage batches
and column blocks.

Parity: `GpuLodfCalculatorTest`, 7 networks (ieee14 scalar and vectorized, ieee30, ieee118,
eurostag, a phase shifter with an alpha variable, a zero-impedance line among the monitored),
every cell against `LodfCalculator` at 1e-10 with the NaN and -1 structure identical; rte6515
9042 × 9042 = 81,757,764 factors, identical finite/NaN counts and identical sum to 9 digits.
One property does NOT hold: the factors are not bit-identical across batchings, because a cuDSS
multi-right-hand-side solve blocks its columns and the same column solved in a block of 4 and of
186 differs by an ulp or two. Within the 1e-10 bound either way.

**The numbers (rte6515, 9042 outages, CPU 8 threads vs GPU 1 thread, `DcLodfGpuProbe`)**:

| stage | CPU 8 threads | GPU |
|---|---|---|
| element states only (1 monitored branch) | 512 ms | 561 ms |
| + all 81.8 M factors computed, not consumed | 818 ms | 969 ms |
| + streamed to a writer | 880 ms | 2006 ms |
| one-off device analysis + factorization | — | 2165 ms |

Read it stage by stage. The 9042-column solve is a wash (cuDSS 561 vs KLU 512) — the same phase-0
finding, the linear algebra is not where the device wins. Evaluating the 81.8 M factors costs the
CPU 306 ms across 8 threads and the device 408 ms, and that 408 ms is mostly the **650 MB of
results crossing PCIe**. Streaming them out then costs the CPU 62 ms (8 threads) and the device
1037 ms (one thread walking a 650 MB buffer). A LODF factor is three loads and three flops: the
whole computation is bandwidth-bound, the CPU calculator is already at memory bandwidth, and the
device adds a PCIe crossing the CPU does not have.

**The consequence, which is the useful part.** A device sensitivity kernel pays only when the
factors are CONSUMED on the device instead of shipped. Two shapes qualify:
- **Screening** — "which (monitored, outage) pairs exceed a threshold", post-contingency flow
  `p + LODF·p_outaged` against limits. This is the SA's "the kernel screens, the host decides"
  pattern, and its strongest form already exists and already pays: the Woodbury DC SA at
  0.53 ms/contingency, ×3 against CPU 8 threads. A `lodfScreen` entry point is the same kernel
  with a threshold and a compacted output; worth adding when a caller wants LODF screening
  specifically rather than a security analysis.
- **Sensitivity analysis WITH contingencies (phase 4)** — there the expensive object is the
  `n × nGroups` factor-state matrix that `WoodburyEngine.toPostContingencyAndOperatorStrategyStates`
  rebuilds PER CONTINGENCY (O(n · nGroups · k), plus a small dense LU per column for the alphas).
  On the device that matrix never leaves: only the sensitivity VALUES come back, `nFactors` per
  contingency, orders of magnitude smaller than the states. That is where PTDF-on-device pays, and
  it is what phase 4 should build — not a bare PTDF matrix producer, which would land exactly where
  LODF just landed.

**Phase 4 map** (from a source survey of `DcSensitivityAnalysis`, 768 lines): `analyse` is ~95%
bookkeeping (parameters, network load, factor reading and grouping, predefined results, unscaling,
filtering, writing) around four solves — base flow (1 column), base factors (nGroups columns),
contingency states (one per distinct outaged branch), action states — and then, per contingency,
up to two re-solves plus two in-place Woodbury updates. Port targets in order of payoff:
(1) `WoodburyEngine.toPostContingencyAndOperatorStrategyStates(DenseMatrix)` and its per-column
`setAlphas` dense LU, (2) the two `calculateSensi` calls of `createBranchSensitivityValue` (already
the `sensiBlock` kernel), (3) the multi-column factor-state solves (already `setElements` /
`setSparseColumns` shaped). Everything else stays on the host. Visibility: an extension can live in
`com.powsybl.openloadflow.sensi` (no `module-info.java` in OLF), which reaches the protected
factor/group machinery; the fine-grained seams inside `DcSensitivityAnalysis` are private and
would need widening — `calculateFactorStates`, `calculateSensitivityValues`,
`createBranchSensitivityValue`, `calculateSensitivityValuesForContingencyAndOperatorStrategy`,
`processContingencyAndOperatorStrategy` — or `analyse` can be overridden wholesale (it is public,
and the constructor is public). Worth knowing: `DcSensitivityAnalysis.analyse` **ignores the
`Executor`** it is given, so the CPU DC sensitivity reference is single-threaded.

## 4i. Phase 4 — DC sensitivity analysis on the device (2026-09-15)

Landed: `GpuDcSensitivityAnalysis` (package `com.powsybl.openloadflow.sensi` in this extension, a
split package so OLF's protected factor machinery is nameable), `GpuSensitivityAnalysisProvider`
(`"OpenLoadFlowGpu"`), and the sensitivity kernels.

**The shape follows §4h's verdict**: the factor states (`equations x factor groups`) and the flow
states (`equations`) are solved into device memory and STAY there for the whole contingency loop;
per contingency the Woodbury update rebuilds them on the device and the per-factor `calculateSensi`
evaluations run in a kernel; the only thing that crosses back is the VALUES, one double per factor.

**Almost no OLF bookkeeping was duplicated.** Four seams were widened upstream-style on
`olf-local-patches-dc` (e57ea8add): `calculateFlowStates`, `calculateFactorStates` and
`calculateSensitivityValues` became protected, a `createWoodburyEngine` factory was added, and
`OpenSensitivityAnalysisProvider` got a `createDcSensitivityAnalysis` factory. The subclass overrides
those four and inherits everything else — factor reading and checking, grouping, participation
factors, the connectivity break analysis, GLSK rescaling, the per-contingency network state
save/restore, statuses, unscaling, filtering and writing.

The trick that makes it possible: the state matrices OLF passes between those seams are
PLACEHOLDERS, tiny `DenseMatrix` instances whose ROW COUNT tags which device buffer they stand for
(base/alternate x flow/factor). OLF only ever copies them (`new DenseMatrix(rows, cols)` +
`copyValuesFrom`) and hands them to the seams, so the tag survives and nothing reads the contents.
The right-hand sides are not re-derived either: OLF's own `SensitivityFactorGroup.fillRhs` writes
into a `SparseColumnRecorder`, a `Matrix` that records instead of storing, and the device fills the
dense columns from those triplets.

Device pieces: `fillSparseRhs`, `alphaFactor` (the Woodbury alpha matrix is the same for every
column — only the right-hand side changes — so it is factorized ONCE per contingency instead of
per column as the CPU does), `alphaSolve` (one thread per column), `woodburyCols` (one thread per
row x column), `sensiPairs` and `sensiFunctions`. One JNI call per contingency carries the
elements, the Woodbury update and the evaluations: split across three calls it cost more in launch
and allocation overhead than in arithmetic (13.0 s -> 4.9 s on 500 contingencies, and `Buf::upload`
grew an `uploadInto` that keeps the allocation instead of a cudaMalloc/cudaFree pair per array).

**Gate: OLF's own DC sensitivity suites, rerun against the GPU provider**, assertions and tolerances
untouched — `DcSensitivityAnalysisTest` 58, `DcSensitivityAnalysisContingenciesTest` 81 (+1 skipped:
`testDebug` needs a platform configuration directory this module does not vendor, verified to fail
identically on the STOCK provider from here), `DcSensitivityAnalysisActionsTest` 12. Plus
`GpuDcSensitivityEngagementTest`, which asserts both the values against the stock engine AND the
device engagement counter, because a comparison test alone cannot tell that the device never ran.
The OLF test jar had to start shipping the sensitivity test harness (a pom include list).

**Numbers** (rte6515, 50 injection variables x 200 monitored branches = 10 000 factors, 200
contingencies = 2 M factor evaluations; the CPU arm is single-threaded because
`DcSensitivityAnalysis.analyse` ignores the executor it is given):

| run | CPU | GPU | ratio |
|---|---|---|---|
| 1 | 2876 ms | 2030 ms | x1.42 |
| 2 | 2740 ms | 2012 ms | x1.36 |
| outlier | 2437 ms | 6664 ms | x0.37 |

The outlier is GPU contention, not the code: the card is shared with the AC session, which held
1.7 GB and was computing during that run; its device calls took 4480 ms against 65-68 ms in the
clean runs. Quote the clean pair and the caveat.

**Where the remaining wall is**, from the in-run phase split (200 states x 10 000 factors):

| | time |
|---|---|
| device calls (elements + Woodbury + 2 M evaluations + download) | 65-68 ms |
| OLF's per-factor `getPredefinedResults` scan | 171-216 ms |
| OLF's per-factor unscale + filter + write | 96-99 ms |
| everything else (OLF's per-contingency orchestration, the host flow-state solve, setup) | ~1.7 s |

So the linear algebra is no longer on the critical path at all: 2 million factor evaluations plus
200 Woodbury updates of a 6536 x 50 state matrix cost 68 ms. What is left is OLF's own host
bookkeeping, identical on both arms — the per-factor predefined-result scan and result writing, and
the per-contingency network state save/restore. That is the next thing to attack if this needs to go
faster, and it is a HOST problem, not a device one.

Not ported, deliberately: the contingency and action element states (`ConnectivityBreakAnalysis`
and `ComputedElement.calculateElementsStates`) are still solved by OLF on the host and uploaded —
one solve per distinct outaged branch, done once per run. The per-contingency flow-states re-solve
also stays on the host, because it needs OLF's modified-target build on the network; it is one
column.

## 4j. Phase 3 — the DC time-series load flow on the device (2026-09-15)

**Phase 3.0 first, as the plan required: where the CPU step actually goes.** Measured on rte6515
with `DcTimeSeriesBaselineProbe`, which switches the emitted datasets off and on and installs a
no-solve engine through the new seam, so the step splits into its parts (1000 steps, 200 planned
generators, one thread):

| phase | ms/step | share |
|---|---|---|
| restore + apply set points | 1.21 | 20% |
| solve (`DcLoadFlowEngine.run`) | 1.74 | 28% |
| emit branch rows | 2.42 | 40% |
| emit bus and generator rows | 0.75 | 12% |
| **total** | **6.12** | |

That is a different verdict from the LODF one (§4h): the part a device engine can take is the
majority, not a sliver. It also says the row emission is untouchable — 16,990 rows a step is the
writer's own cost, on any arm.

**Two seams on the ts branch**: `TimeSeriesLoadFlow.run` takes an optional `EnginePlanProvider`,
and `StepEngine` gains a batched contract (`batchSize` / `prepare` / `solveBatch` / `apply`) plus
`restoreState`. The batched loop applies each step's set points exactly once and still emits from
the network, so a batched engine changes nothing about what a step means. `EnginePlan`,
`StepEngine` and `SolveResult` became public. OLF's own 24 time-series tests pass unchanged.

**`GpuTimeSeriesDcEngine`**: a chunk of 256 steps is one factorization against 256 right-hand
sides. Per chunk the device computes, for every step in parallel, the bus generation, the slack
mismatch, `ActivePowerDistribution` (one thread per step, on that step's own copy of the generator
tables), the multi-slack share and the target column; then one multi-right-hand-side solve for the
whole chunk. The host reads a step back into the equation system and emits exactly as it does on
the CPU path.

Two things the port had to get right, both found by the row-for-row gate:
- **every generator attribute the distribution reads is per STEP**, not per run. A plan series
  writes the value into the generator's target AND its initial target and re-runs OLF's active
  power control checks, which can switch its participation off. The device therefore takes eight
  per-step generator tables, not one; uploading the run's tables gave visibly wrong flows.
- **`restoreState` must put back both** the generator targets the distribution moved and the initial
  targets the set points wrote, because the generator rows carry the initial target. Restoring only
  what the solve dirtied left the last step of the chunk's set point in every row.

**Gate**: `GpuTimeSeriesDcEngineTest`, seven cases (generator plans, load plans, distributed slack
on generation P and P_MAX, three slack buses, ieee118 with more steps than a chunk, ieee300),
comparing both engines row for row — dataset, step, status and element identical, values within
1e-8 MW — plus the per-step statuses, distributed power and slack mismatch, and the device step
counter. The signed zero of a branch carrying nothing differs between the arms and is not a
difference.

**Numbers** (rte6515, 1000 steps, 200 planned generators, one thread, 16.99 M streamed rows):

| | CPU | GPU | ratio |
|---|---|---|---|
| restore + set points + solve | 2.95 ms/step | 1.07 ms/step | **x2.76** |
| the whole run, all rows streamed | 6.12 ms/step | 3.16 ms/step | **x1.94** |

The device does restore, set points AND solve in less time than the CPU takes to restore and apply
set points alone. End to end the run is halved, and what stops it there is the row emission, which
is the same host work on both arms.

### The AC-emulation limits outer loop, driven over the chunk (2026-09-15)

A network needing a DC outer loop used to send the whole run to the stock engine — which is what
rte6515 does, since it carries AC-emulation HVDC links. That loop now runs over the chunk, so
rte6515 stays on the device with its own configuration.

The obstacle was that a step whose loop fires needs a DIFFERENT MATRIX (saturating a link removes
its droop coupling from the bus equations), and the steps of a chunk only batch because they share
one factorization. The measurement that settled the mechanism: on rte6515 a value refill plus a
cuDSS refactorization costs **11.6 ms**, against **583 ms** for the initial analysis and
factorization and 4.4 ms for a single solve. So configurations are cheap to switch between, and
grouping beats a per-step Woodbury correction, which would have meant per-step element sets and
alpha systems for a gain that 11.6 ms does not justify.

Per chunk, then: the device evaluates `DcHvdcAcEmulationLimitsOuterLoop.check` for every step at
once — one thread per (step, link), on that step's own state, its own status and its own scheduled
offset — the steps that changed are regrouped by the configuration they now ask for, each
configuration is applied to the network with OLF's own mutators, the device matrix is refilled and
refactorized for it, and that group's steps are re-solved together against it. Repeats until no step
changes, within OLF's outer-loop budget. With one loop in the list, OLF's nested sweep reduces to
exactly that: run the loop to stability.

Found on the way, and this one was silent: a plan series can schedule an AC-emulation OFFSET, and
the offset enters the target vector as the right-hand side of the emulation terms. The assembly had
captured it as a run constant, so a planned offset moved nothing. It is now a table lookup
(`RHS_HVDC`), per step in the time-series kernels and per run everywhere else — the assembly test's
bit-identical target catches any mis-mapping.

Gate: two more cases in `GpuTimeSeriesDcEngineTest` — a link driven past its limit by the generator
plan, and a plan that sweeps a link's offset across both saturation sides — compared row for row
against the stock engine like the rest.

**Numbers with the loop running** (rte6515, 1000 steps, 200 planned generators, all rows):

| | CPU | GPU | ratio |
|---|---|---|---|
| restore + set points + solve | 3.37 ms/step | 0.71 ms/step | **x4.73** |
| the whole run, all rows streamed | 6.87 ms/step | 3.36 ms/step | **x2.04** |

1000/1000 steps on the device. Stated honestly: on this plan no link ever saturates — the run
applies **zero** configuration changes — so the loop costs one check kernel per chunk, and the
ratio comes from the chunked solve and the cheap restore, not from loop work. What the loop bought
is that rte6515 runs on the device at all; the regrouping path is exercised by the two unit cases,
not by this plan.

### Both DC outer loops over the chunk (2026-09-15)

The incremental phase control loop joined the AC-emulation one, so every DC loop that decides from
the state now runs over a chunk and only area interchange control still sends a run to the stock
engine.

What it needed beyond the HVDC case:
- **A per-step ratchet.** `IncrementalContextData`'s allowed direction, direction-change count and
  current direction are per RUN, and a step IS a run, so they became per (step, controller) arrays
  carried into the decision kernel and back out of it.
- **Per-configuration sensitivities.** The `dP/dα` columns depend on the matrix, not on the step, so
  they are solved once per configuration group and shared by its steps.
- **The configuration is now (AC-emulation statuses, tap positions)**, and a tap moves the matrix —
  `power` and the phase shift — as well as the target's identity row.
- **A per-step loop state machine**, replicating `DcLoadFlowEngine.run`'s sweep: a loop that reports
  stable hands over to the next, the pass ends when the list is exhausted or the next loop is the
  one that fired last, and the run ends when a whole pass solves nothing.

Three bugs the row-for-row gate caught, all in that sweep, and all worth remembering:
- the decisions are evaluated for a whole group, but only a step whose sweep is ON that loop has
  actually run it — writing the ratchet back for the others advances a loop OLF has not reached;
- **the sweep must not stop when a round changes no configuration.** Termination is per step, on
  the state machine's own `done`: a step that merely advanced to the next loop still has to run it,
  and stopping early left the phase loop unevaluated;
- `apply(slot)` has to put the step's own configuration back before emitting, because a branch flow
  is read from the network's pi model — a controller sitting on another step's tap is emitted with
  the wrong power and shift.

Gate: `GpuTimeSeriesDcEngineTest` is now 11 cases, adding a shifter that has to chase its target
across steps and a network where the shifter and a saturating link interact, both compared row for
row. The whole DC GPU suite is 247 tests, green.

**Numbers** (rte6515, 1000 steps, 200 planned generators, all rows, both loops configured; the
machine is noisy between runs, so the range of two runs):

| | CPU | GPU | ratio |
|---|---|---|---|
| restore + set points + solve | 4.49 / 6.8 ms/step | 0.98 / 1.12 ms/step | **x4.6 to x6.1** |
| the whole run, all rows streamed | 14.69 / 6.79 ms/step | 5.54 / 2.94 ms/step | **x2.3 to x2.65** |

Same honest caveat as before: this plan never saturates a link and never moves a tap, so the loops
cost one decision kernel per chunk here. What they buy is that rte6515 runs on the device with its
own configuration instead of falling back; the regrouping and the sweep are exercised by the four
loop cases in the gate.

Left for later: the emission is now the whole remaining cost, and it parallelizes — running it on
several host threads while the device solves the next chunk is the next multiplier (OLF already
partitions by step range, but each partition would build its own device context today). And a
configuration switch re-captures the assembly (39 ms on rte6515) before the 12 ms refactorization;
for a tap-only change the powers and shifts could be patched in place instead, which would matter
on a plan that moves taps often.

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
- **OLF patches live on `olf-local-patches-dc`** (our own branch/jar, no coordination needed
  until the merge): so far ONE, `ConnectivityBreakAnalysis.computeConnectivityAnalysisResult` and
  `createContingencyElementsIndexByBranchId` widened to public (upstream-friendly) so the device
  criterion can hand only the flagged contingencies to OLF's graph confirmation. Sensitivity (phase 4) may need 3-4 `private →
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

### Row emission over host threads, and how the chunk is sized (2026-09-15)

Once the device owns restore, set points and solve, what is left of a time-series step is the row
emission, and that is host work identical on both arms. The mechanism to parallelize it already
exists: OLF partitions the step range over `threadCount` threads, each with its own writer. The GPU
engine now runs under that unchanged, with one addition.

`GpuDcSolver.DEVICE_LOCK` is a process-wide `ReentrantLock` held around engine creation, `solveBatch`
and `close`. cuDSS's analysis phase is not thread-safe across host threads: four partition threads
each creating an engine segfaults the JVM inside `libcudss`. Under the lock the device work
serializes while the set-point walking and the row emission - the parts that actually parallelize -
run concurrently. `GpuTimeSeriesDcEngineTest.matchesStockEngineOverPartitions` is the gate: the same
plan at `threadCount = 4`, rows compared as a sorted multiset because partitions interleave.

Chunk sizing has a different answer here than on either other path, and the difference is
instructive:

- **AC batch path** - the chunk IS the cuDSS `UBATCH` batch. It decides whether the batched solve is
  used at all, and it is worth x2.
- **DC Woodbury path** - no batch. One factorization, the chunk only sizes kernel launches and the
  state download. Flat: 64/256/1024 land within run-to-run noise.
- **DC time-series path** - the chunk is a number of STEPS sharing one factorization, so a larger
  chunk amortizes per-call overhead and widens the multi-right-hand-side solve. It pays up to about
  1024-2048 steps and is flat after; `DEFAULT_CHUNK` is 1024.

The time-series chunk has a second role that argues the same way: it is the outer-loop grouping unit.
Steps are grouped by configuration within a chunk, so a chunk large enough to contain repeated
configurations shares one refactorization between them. The cap is memory, one right-hand side and
one state column per step.

Measured tables in `docs/numbers.md`. The headline under the standing comparison rule: mono-thread
GPU 2.61 ms/step against 8-thread CPU 1.97 ms/step on rte6515_full at 4096 steps, so the GPU arm
needs about two host threads to match a fully threaded CPU and four to beat it. The device section
itself is down to 0.62 ms/step; everything above that is emission.
