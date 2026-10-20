# GPU environment variables

Every one of these is read at runtime. The project's standing goal is to **match the CPU at machine
precision on every difference**, so the dangerous category is the third table: a switch that selects a
different computation can preserve a wrong answer behind a default, and nothing in CI explores it.

Three were removed on 2026-09-17 for exactly that reason:

| removed | why |
|---|---|
| `OLF_GPU_CURVE_GENCUM` | re-enabled the share-reconstructed curve P after the per-generator read was proven better (rte6515 worst generator error 3.790e-02 -> 4.837e-05 kV) |
| `OLF_GPU_NO_CURVES` | a bisection lever that made curve buses wrong by design; it had served its purpose |
| `OLF_GPU_SA_NO_BATCH` | duplicated the pre-existing `-Dolf.gpu.sa.disableBatch=true` |

Classification below is by NAME and call site, not an individual audit of each one's semantics.
Anything in the third table should be treated as unverified until someone reads it.

## System properties

| property | effect |
|---|---|

## Diagnostics (58) — print only, cannot change a result

Safe to add. These are the instruments; several are the only reason a defect was found at all.

| variable | read at |
|---|---|
| `OLF_EQ_DUMP` | OLF patch, NewtonRaphson — writes the equation system OLF is about to solve as text (variable rows, column->equation map, `writeToString(true)`). The one question a residual cannot answer is WHICH equation sits on a row, and OLF will simply print it |
| `OLF_GPU_BATCH_TRACE` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchPrecomputer.java:71 +35 |
| `OLF_GPU_CENSUS_BASE_VIOLATIONS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:300 |
| `OLF_GPU_CENSUS_BUS_DUMP` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:309 |
| `OLF_GPU_CENSUS_CASE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:69 |
| `OLF_GPU_CENSUS_CHUNK` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:242 |
| `OLF_GPU_CENSUS_CONNECTIVITY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:342 |
| `OLF_GPU_CENSUS_CONV_EPS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:111 |
| `OLF_GPU_CENSUS_CPU` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:80 |
| `OLF_GPU_CENSUS_MONITOR_ALL_VL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:253 |
| `OLF_GPU_CENSUS_MONITOR_BRANCH` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:257 |
| `OLF_GPU_CENSUS_MONITOR_VL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:256 |
| `OLF_GPU_CENSUS_NO_ACTIVE_LIMITS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:108 |
| `OLF_GPU_CENSUS_NO_DS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:100 |
| `OLF_GPU_CENSUS_NO_RL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:97 |
| `OLF_GPU_CENSUS_NOFREEZE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:137 |
| `OLF_GPU_CENSUS_ONLY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:152 |
| `OLF_GPU_CENSUS_SVC` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:229 |
| `OLF_GPU_CENSUS_VIOL_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:122 |
| `OLF_GPU_CENSUS_VIOLATION_DUMP` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:354 |
| `OLF_GPU_CPU_ROUTED` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java:1383 |
| `OLF_GPU_DBG_BUSDUMP` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuStateInjector.java:139 +6 |
| `OLF_GPU_DBG_CENSUS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysisTest.java:3112 |
| `OLF_GPU_DBG_REPORT` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysisTest.java:3100 +2 |
| `OLF_GPU_DBG_RL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:313 +2 |
| `OLF_GPU_DQ_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu:2850 |
| `OLF_GPU_DX_DUMP_ON_REJECT` | native/cuda/jni/olf_gpu_nr_jni.cu — `<dir>`: the same dump (J, F, B, dx, modes) of every slot the pass-0 linear-solve guard rejects (status -3), as the guard scored it, at most 8 per process: `<dir>/reject-<k>-slot<s>.dump`. Written after the verdict, so it cannot perturb what it catches; for a reject that shows only in a full run and never alone (the rte6515 busbar one-per-run flap) | native/cuda/jni/olf_gpu_nr_jni.cu (linear-solve guard) — the slot's contingency is the `CHUNK_REJECT slot=<s>` line of the same chunk |
| `OLF_GPU_E2E` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedN1FixedIterTest.java:176 +1 |
| `OLF_GPU_E2E_WARMJIT` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedN1FixedIterTest.java:176 |
| `OLF_GPU_ITER_TRACE_CTG` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java — `<id>[,<id>]`: the main-solve iteration count of NAMED scenarios at any chunk size; line it up against OLF_LF_STEPS_TRACE on the CPU arm; also prints `DAP_CTG` (GpuBatchPrecomputer): the replayed loss pre-distribution and the device's own slack passes of the named scenario in MW, so a differing total can be split into its two halves |
| `OLF_GPU_CHUNK_DUMP` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java — `<file>`: appends one line per chunk, `<pairPhase> <chunkStart> <chunkSize> <id>...` in slot order; the composition a chunk-mate-dependent scenario has to be reproduced in |
| `OLF_GPU_REJECT_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu (`traceReject`, every status −2 gate) + GpuBatchPrecomputer (now the MAPPER rejects too, one `GPU_REJECT_CTG ctg= reason=` per contingency, which is what tells a scope report whether a non-batched contingency is one OLF itself does not answer) — `GPU_REJECT slot= reason=` (host-loop-outer-budget, straggler-exit, unrealistic-state, device-loop-routed, tvd-kernel-reject, svc-kernel-reject, ds-residual-saturated, outer-loop-budget) and `GPU_REJECT_CTG ctg=` so the census's one RUNTIME tag gets a reason per contingency |
| `OLF_GPU_LOOP_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu (driver) — `<slot>`: `LOOP_PASS` per pass, the device side of OLF's `OLF_OL_TRACE` `OL_CHECK`: which loop ran, whether the slot was found UNSTABLE, whether the phase is still unstable; since 2026-09-28 each line also carries the DS gate that slot last evaluated (`dsRem=`, `dsFire=` — the device side of OLF's DS_PASS mismatch, read from a per-scenario record the water-fill kernel writes, no printf) and the slot's iteration count for the pass (`nrIters=`); the traced slot also prints `SLOT_TRACE` per Newton iteration (host-side: inf/2-norms, the 2-norm threshold, the convergence decision, the MAX_VOLTAGE_CHANGE factor, the worst residual row with its mode/target/x, and the largest raw step entry) — OLF's NR_EXIT prints the same norms per iteration |
| `OLF_GPU_LOOP_TRACE_CTG` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java + native (setBatchLoopTraceSlot) — `<contingency id>`: the same trace addressed by CONTINGENCY; the host resolves the slot per chunk (`LOOP_TRACE_CTG ctg= slot=`). A fixed slot number follows the chunk composition, which moves with the resolved chunk size (IZUZ sat at slot 170 in one run and 18 in the next) |
| `OLF_GPU_SA_SELF_DUMP` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — `<file>`: with a warm run (no `OLF_GPU_SA_COLD_ONLY`), every contingency's status and distributed active power cold and warm; the `POST GPU-vs-GPU` line prints the same comparison summarized |
| `OLF_GPU_SLOT_DUMP` | native/cuda/jni/olf_gpu_nr_jni.cu (driver start + first outer pass) — `<prefix>`: the `OLF_GPU_LOOP_TRACE_CTG` scenario's row targets, initial state and row modes when the driver starts (`<prefix>.<k>`) and its state and bus balances (qBuf) after the first solve (`<prefix>.<k>.pass0`); diff two runs of one scenario row by row |
| (no switch) | the replay is APPLIED, not optional: setBatchZiTreeOverrides carries OLF's tree per scenario, the pattern reserves the ZERO_V/ZERO_PHI entries for every coupler, and each scenario starts with the coupler flows at zero as AcSolverUtil.initStateVector leaves them |
| `OLF_GPU_SW_LOG` | native/cuda/jni/olf_gpu_nr_jni.cu (reactive-limit kernel) — `<slot>` records EVERY check of that scenario, `all` records only the checks that DECIDED something, for every scenario. The log is written ON THE DEVICE (one atomicAdd per record) and copied out once per chunk as `SW_LOG s= pass= entry= act= lt= sc= q= min= max= v= cmpV= vt=`. The host-side traces copy hundreds of MB per pass, which serializes the stream and changes the kernel scheduling whose effect is being traced — a run-to-run flip disappears while you watch it. This one found the zero-range pilot entries: two runs of one window, aligned by (chunk, slot, pass, entry), differed on 430 decisions and every one was such an entry |
| `OLF_GPU_ZITREE_TRACE` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuAcRowIndexing.java (+ the batch path) — `ZITREE root=<bus> couplers=<n>` per relocated zero-impedance tree, `ZITREE_SUMMARY treeCouplers= relocated= stock=` per extraction (a STOCK tree coupler keeps two zero diagonals the numeric repair fills; network.xiidm busbar scope: 71 before the 2026-10-11 root order, 58 after) and `ZITREE_OPEN ctg= coupler= childSide=` where a scenario opens a relocated coupler | Also `ZITREE_STOCK coupler= dummyCols= <end>{root|child,phiId,vId,hostsOtherQ,ownQ|remoteCtl|qElsewhere,imp|noImp} <end>{...}` per stock tree coupler (why neither end is its child), `ZITREE_REPLAY_WIDE ctg= nonTree=` (GpuBatchPrecomputer: a contingency the tree replay reports more than 40 couplers out of the tree for) and `ZITREE_LEAVE_WIDE slot= ctg= nonTree=` (GpuBatchChunkSolver: the same at the chunk) - the 2026-10-11 connectivity-anchor leak showed as nonTree=366 on one random section per run
| `OLF_GPU_RL_AUDIT` | `OLF_GPU_SW_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu (reactive-limits phase) — `<slot>`: `SW_PASS`, every controller that slot's RL check CHANGED this pass with the quantities the decision turned on (q, live min/max, its own V, the pilot's V and the group target). `OLF_GPU_LOOP_TRACE` says a pass found the slot unstable; this says WHICH controller and by how much |
| `OLF_GPU_HVDC_TRACE` | GpuBatchHvdcAcEmulation / GpuBatchChunkSolver — `<phiRow>` or `all`: `HVDC_BUILD` (every AC-emulation hvdc the pack carries: id, element, OLF status, station targetP), `HVDC_SLOT` (slot -> contingency), `HVDC_PREOV` (pre-freeze override entries already on a converter-bus P row — the freeze composes on top of them), `HVDC_CTL` (per slot and hvdc: base status/deltas, mask bit, warm-point flows, base target), `HVDC_FREEZE` (the flows each scenario's freeze fixed an hvdc at, vs the base deltas) and `HVDC_END` (the status each scenario ended with). Found the doubled-corridor unfreeze race (two links on one bus pair, 2026-09-22) |
| `OLF_GPU_PHASE_TIME` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuStateInjector.java:86 +9 |
| `OLF_GPU_PRELOSS_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java:697 +2 |
| `OLF_GPU_QUALIFY_TRACE` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuContingencyMapper.java:219 +9 |
| `OLF_GPU_RL_TRACE` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuAcOuterLoopPacks.java — `GPU_RL_ENTRY idx=<n> vRow=<row> bus=<id>` for every reactive-limit entry, LOCAL and REMOTE (a remote member also prints its class and its pilot's V row): the map that lines the device's per-entry `RL_TRACE` pin states up against OLF's `RL_SWITCH` bus names |
| `OLF_GPU_SA_BUSDIFF` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:348 |
| `OLF_GPU_SA_BUSDIFF_CTG` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — `<id>[,<id>]`: bus-diff those contingencies INSIDE the full scope (one scoped StateMonitor per id, so the heap holds a handful of results, not 10540 x 13202). The list decides the retained switches on a node-breaker network, so a contingency that differs in the full scope cannot be studied alone (CTG_ONLY) |
| `OLF_GPU_SA_DEBUG_CTG` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:164 |
| `OLF_GPU_SA_INVENTORY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:205 |
| `OLF_GPU_SA_MON_BRANCH` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:353 +1 |
| `OLF_GPU_SA_MON_BUS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:523 +1 |
| `OLF_GPU_SA_RESID` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysis.java:791 +3 |
| `OLF_GPU_SCOPE_CENSUS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:77 |
| `OLF_GPU_SHUNT_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchIncrementalShuntVc.java:98 |
| `OLF_GPU_STATE_DUMP` | native/cuda/jni/olf_gpu_nr_jni.cu — for slot 0 and iterations 0..&lt;value&gt;, EVERY row's mode/target/state/residual (systems up to 64 rows). On a reproducer it shows at a glance which variable ran away and which equation let it |
| `OLF_GPU_SVC_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchSecondaryVc.java:464 +2 |
| `OLF_GPU_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu:289 +2 |
| `OLF_GPU_UNREALISTIC_TRACE` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysis.java:808 +1 |
| `OLF_RL_PROBE_BUS` | (OLF repo, branch olf-local-patches) src/main/java/com/powsybl/openloadflow/ac/outerloop/ReactiveLimitsOuterLoop.java |

## Test scope selectors (8)

Choose which contingencies a validation run covers. They change what is measured, not how it is computed.

| variable | read at |
|---|---|
| `OLF_GPU_SA_CPU_ARM_DIR` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — with `OLF_GPU_SA_CPU_ARM_MODE=save`: run the CPU arm only and write it (cpu.json, cpu-busv.bin/.idx, cpu-ms.txt) here; `=load`: rebuild the CPU arm from these files and run the GPU arms only. The two arms of the full busbar scope do not fit one JVM. |
| `OLF_GPU_SA_CPU_ARM_MODE` | see `OLF_GPU_SA_CPU_ARM_DIR` (`save` / `load`) |
| `OLF_GPU_SA_GPU_ONLY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — `1`: run the CPU arm on an EMPTY contingency list (debug lever: the GPU arm's own behaviour on a large scope in minutes; every comparison is then meaningless and the CPU base sits on a different bus decomposition). |
| `OLF_GPU_CONT_FILE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:77 +4 |
| `OLF_GPU_SA_CTG_ONLY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:218 +1 |
| `OLF_GPU_SA_MAX_CTG` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:233 |
| `OLF_GPU_SA_CASE_FILE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — path of the case to run instead of rte6515 (e.g. `network.xiidm` at the repo root); the other selectors apply unchanged |
| `OLF_GPU_SA_CTG_RANGE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — `from:to`: keep only that index window of the selected contingency list (after `OLF_GPU_SA_CTG_KINDS`); a window is ONE or a few chunks — always finish on the full width |

## Build / toolchain (4)

Paths, set by `native/local-env.sh`.

| variable | read at |
|---|---|

## Highest risk: tolerances that apply to the GPU ARM ONLY

These set a tolerance for the GPU arm that the CPU arm does not get:

| variable |
|---|
| `OLF_GPU_SA_GPU_CONV_EPS` |
| `OLF_GPU_SA_GPU_RL_TOL` |
| `OLF_GPU_SA_GPU_SLACK_TOL` |

A one-sided tolerance is the mechanism by which a GPU/CPU difference can be made to disappear without
being fixed, which the project forbids: **a disagreement at the default tolerance is an orchestration
bug to fix, not a tolerance to loosen.** Contrast `OLF_GPU_DBG_CONV_EPS`, which tightens BOTH arms —
that is the legitimate instrument, because squeezing the slack and watching what survives is how a real
state difference is told from convergence slack.

Also in the list and wrong by construction, useful only as momentary bisection levers:
`OLF_GPU_NO_DS`, `OLF_GPU_NO_RL`, `OLF_GPU_NO_DIST_SLACK`, `OLF_GPU_SA_NO_DS`, `OLF_GPU_SA_NO_RL`,
`OLF_GPU_HOST_LOOP_NOOP`, `OLF_GPU_SVC_INERT`. Each disables an outer loop or makes it inert, so any
run with one set cannot match the CPU. They should be deleted once the question that motivated them is
answered — the three removed above were removed on exactly that ground.

## BEHAVIOUR-CHANGING (23) — REVIEW

Each selects a different computation. Several disable an outer loop or loosen a tolerance, which is wrong by construction (and note the standing rule: never loosen a tolerance to make the GPU agree with the CPU). These should be justified individually or deleted.

| variable | read at |
|---|---|
| `OLF_GPU_BASE_ON_GPU` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedN1FixedIterTest.java:149 +2 |
| `OLF_GPU_DBG_CONV_EPS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:295 +2 |
| `OLF_GPU_DBG_SLACK_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysisTest.java:3084 +1 |
| `OLF_GPU_DISC_EPS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuSingleSolveRejectReasonProbe.java:60 |
| `OLF_GPU_DISC_SLACKTOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuSingleSolveRejectReasonProbe.java:64 |
| `OLF_GPU_FIXED_ITERS` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedN1FixedIterTest.java:79 |
| `OLF_GPU_MAXBATCH` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:83 +5 |
| `OLF_GPU_NO_DIST_SLACK` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuCpuOuterLoopTrajectoryProbe.java:55 |
| `OLF_GPU_NO_DS` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedPegaseOutlierDiagnosticTest.java:69 |
| `OLF_GPU_NO_RL` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedPegaseOutlierDiagnosticTest.java:70 |
| `OLF_GPU_SA_CONV_EPS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:197 |
| `OLF_GPU_SA_CPU_SINGLE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:276 |
| `OLF_GPU_SA_GPU_CONV_EPS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:242 |
| `OLF_GPU_SA_GPU_RL_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:244 |
| `OLF_GPU_SA_GPU_SLACK_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:243 |
| `OLF_GPU_SA_MAX_OUTER` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:219 |
| `OLF_GPU_SA_NO_DS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:229 |
| `OLF_GPU_SA_NO_RL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:226 |
| `OLF_GPU_SA_RL_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:211 |
| `OLF_GPU_SA_SLACK_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:205 |
| `OLF_GPU_SVC` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:310 |
| `OLF_GPU_SVC_PILOT_SHIFT` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:317 |
| `OLF_GPU_VIOL_CROSSCHECK` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysis.java:305 +1 |
