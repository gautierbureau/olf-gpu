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
| `olf.gpu.sa.disableBatch` | batch nothing; OLF's engine drives every contingency sequentially |
| `olf.gpu.sa.gpuSequentialLoadFlow` | keep the GPU solver for sequential fall-through solves (needs the run configured with `GPU_NEWTON_RAPHSON`; on its own it does nothing) |
| `olf.gpu.sa.outerLoopMode` | `DEVICE` (default) or `JAVA` |
| `olf.gpu.driverOuterLoops` | `false` puts OLF's Java engine back in charge of the base load flow |
| `olf.cuda.path`, `olf.cudss.path`, `olf.gpu.lib` | native library locations |

### The manual single-contingency debug path

    -Dolf.gpu.sa.disableBatch=true  OLF_GPU_SA_SEQ_SOLVER=1  OLF_GPU_RESIDUAL_TRACE=1

Solves one contingency as an ordinary load flow on the device, which is where the load-flow
instruments live. A contingency wrong batched and right there is a BATCH defect.


## Diagnostics (94) — print only, cannot change a result

Safe to add. These are the instruments; several are the only reason a defect was found at all.

| variable | read at |
|---|---|
| `OLF_DUMP_DIR` | src/test/java/com/powsybl/openloadflow/ac/equations/vector/gpu/JacobianDumpCompareTest.java — directory holding `cpu.step0` / `gpu.step0` |
| `OLF_EQ_DUMP` | OLF patch, NewtonRaphson — writes the equation system OLF is about to solve as text (variable rows, column->equation map, `writeToString(true)`). The one question a residual cannot answer is WHICH equation sits on a row, and OLF will simply print it |
| `OLF_GPU_BATCH_TRACE` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchPrecomputer.java:71 +35 |
| `OLF_GPU_CB_PROBE` | native/cuda/jni/olf_gpu_nr_jni.cu — the pi row the device ACTUALLY evaluates for one closed-branch element in every slot, after findTapOverride; says what the kernel read, not what the host meant to send |
| `OLF_GPU_CENSUS_BASE_VIOLATIONS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:300 |
| `OLF_GPU_CENSUS_BUS_DUMP` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:309 |
| `OLF_GPU_CENSUS_CASE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:69 |
| `OLF_GPU_CENSUS_CHUNK` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:242 |
| `OLF_GPU_CENSUS_CONNECTIVITY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:342 |
| `OLF_GPU_CENSUS_CONV_EPS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:111 |
| `OLF_GPU_CENSUS_CPU` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:80 |
| `OLF_GPU_CENSUS_CPU_BASE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:188 |
| `OLF_GPU_CENSUS_MONITOR_ALL_VL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:253 |
| `OLF_GPU_CENSUS_MONITOR_BRANCH` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:257 |
| `OLF_GPU_CENSUS_MONITOR_VL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:256 |
| `OLF_GPU_CENSUS_NO_ACTIVE_LIMITS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:108 |
| `OLF_GPU_CENSUS_NO_DS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:100 |
| `OLF_GPU_CENSUS_NO_RL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:97 |
| `OLF_GPU_CENSUS_NO_SCREEN_VIOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:221 |
| `OLF_GPU_CENSUS_NO_TABLE_VIOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:214 |
| `OLF_GPU_CENSUS_NOFREEZE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:137 |
| `OLF_GPU_CENSUS_ONLY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:152 |
| `OLF_GPU_CENSUS_OUTER_MODE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:195 |
| `OLF_GPU_CENSUS_SEQ_GPU` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:204 |
| `OLF_GPU_CENSUS_SVC` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:229 |
| `OLF_GPU_CENSUS_VIOL_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:122 |
| `OLF_GPU_CENSUS_VIOLATION_DUMP` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:354 |
| `OLF_GPU_CPU_ROUTED` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java:1383 |
| `OLF_GPU_CURVE_DUMP` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchRlCurves.java:127 |
| `OLF_GPU_DBG_BUSDUMP` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuStateInjector.java:139 +6 |
| `OLF_GPU_DBG_CENSUS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysisTest.java:3112 |
| `OLF_GPU_DBG_DS` | native/cuda/jni/olf_gpu_nr_jni.cu:8428 |
| `OLF_GPU_DBG_PAIR_OV` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java:817 |
| `OLF_GPU_DBG_PAIR_RL` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java:945 |
| `OLF_GPU_DBG_REPORT` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysisTest.java:3100 +2 |
| `OLF_GPU_DBG_RL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:313 +2 |
| `OLF_GPU_DBG_ROW` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuAcFamilyPacker.java — pack-time dump of the zero-impedance branch and both buses' vCol/qRow/qRowAlt behind a given row |
| `OLF_GPU_DIRTY_PROBE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuHybridBatchOuterLoopEngine.java:441 |
| `OLF_GPU_DISTRQ_DUMP` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuAcActivationExtractor.java:35 |
| `OLF_GPU_DQ3_TRACE` | GpuAcOuterLoopPacks — every DISTR_Q kind-3 (zero-impedance DUMMY_Q) contribution for one dummy-Q column, with gk/gj/qpk/side/row and the SIGNED weight, so the pack can be checked against what the matrix ends up holding |
| `OLF_GPU_DQ_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu:2850 |
| `OLF_GPU_DS_BUSMAP` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchPrecomputer.java:204 |
| `OLF_GPU_DS_DEBUG` | native/cuda/jni/olf_gpu_nr_jni.cu:8110 +2 |
| `OLF_GPU_DS_GEN_TRACE` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuAcOuterLoopPacks.java:558 +2 |
| `OLF_GPU_DS_PACK_TRACE` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuAcOuterLoopPacks.java:598 |
| `OLF_GPU_DS_TARGET_DUMP` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuAcOuterLoopPacks.java:607 +2 |
| `OLF_GPU_DX_DUMP` | native/cuda/jni/olf_gpu_nr_jni.cu — `<path>`: the first solve’s assembled Jacobian (CSR), the mismatch F, the step dx and, since 2026-09-21, `B`, the right-hand side as solved after the zero-diagonal row repair (so J·dx = B is checkable offline; F stays the pre-repair mismatch); slot 0 only | native/cuda/jni/olf_gpu_nr_jni.cu (first batch solve, it=0, slot 0) — writes J/F/dx as text; pair with `OLF_NR_DX_DUMP` on the CPU arm and read both with `JacobianDumpCompareTest` |
| `OLF_GPU_E2E` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedN1FixedIterTest.java:176 +1 |
| `OLF_GPU_E2E_WARMJIT` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedN1FixedIterTest.java:176 |
| `OLF_GPU_FINAL_TARGET_ROW` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java:1294 +1 |
| `OLF_GPU_FREECTL_TRACE` | GpuAcOuterLoopPacks — per remote-VC group, the members, the FREE controller the device picked, and whether OLF actually has a DISTR_Q equation for each |
| `OLF_GPU_GENOUT_REJECT` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchInjectionOutages.java:66 +1 |
| `OLF_GPU_GENOUT_SHAPE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchInjectionOutages.java:231 |
| `OLF_GPU_GENOUT_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchInjectionOutages.java:396 |
| `OLF_GPU_ITER_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java:1136 |
| `OLF_GPU_ITER_TRACE_CTG` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java — `<id>[,<id>]`: with OLF_GPU_ITER_TRACE, the main-solve iteration count of NAMED scenarios at any chunk size (the per-scenario print is otherwise gated on chunks of <= 4); line it up against OLF_LF_STEPS_TRACE on the CPU arm; also prints `DAP_CTG` (GpuBatchPrecomputer): the replayed loss pre-distribution and the device's own slack passes of the named scenario in MW, so a differing total can be split into its two halves |
| `OLF_GPU_LIN_RESID` | native/cuda/jni/olf_gpu_nr_jni.cu (both the sequential and the batched solve) — prints `\|\|J.dx-F\|\|inf` per iteration, i.e. how exactly cuDSS solved the system it was handed |
| `OLF_GPU_CONV_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu (batched solve) — `<slot>` or `all`: the stopping DECISION per iteration (norm2, its threshold, the active row count, inf-norm, per-row flag) plus `CONV_TOP`, the eight worst rows by index; OLF's `OLF_NR_ITER_TRACE` prints `NR_TOP` with the same shape, so the two lists name the rows where a step differs |
| `OLF_GPU_HAZ_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu (`hazRepairVals`) — device printf `HAZ_NODONOR s= row= donors=`: a zero-diagonal row none of its candidate donors could repair this scenario (lands in the surefire dumpstream) |
| `OLF_GPU_REJECT_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu (`traceReject`, every status −2 gate) + GpuBatchPrecomputer (now the MAPPER rejects too, one `GPU_REJECT_CTG ctg= reason=` per contingency, which is what tells a scope report whether a non-batched contingency is one OLF itself does not answer) — `GPU_REJECT slot= reason=` (host-loop-outer-budget, straggler-exit, unrealistic-state, device-loop-routed, tvd-kernel-reject, svc-kernel-reject, ds-residual-saturated, outer-loop-budget) and `GPU_REJECT_CTG ctg=` so the census's one RUNTIME tag gets a reason per contingency |
| `OLF_GPU_LOOP_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu (driver) — `<slot>`: `LOOP_PASS` per pass, the device side of OLF's `OLF_OL_TRACE` `OL_CHECK`: which loop ran, whether the slot was found UNSTABLE, whether the phase is still unstable; since 2026-09-28 each line also carries the DS gate that slot last evaluated (`dsRem=`, `dsFire=` — the device side of OLF's DS_PASS mismatch, read from a per-scenario record the water-fill kernel writes, no printf) and the slot's iteration count for the pass (`nrIters=`); the traced slot also prints `SLOT_TRACE` per Newton iteration (host-side: inf/2-norms, the 2-norm threshold, the convergence decision, the MAX_VOLTAGE_CHANGE factor, the worst residual row with its mode/target/x, and the largest raw step entry) — OLF's NR_EXIT prints the same norms per iteration, and `OLF_GPU_LIN_RESID` now carries ‖F‖₂ as well |
| `OLF_GPU_LOOP_TRACE_CTG` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java + native (setBatchLoopTraceSlot) — `<contingency id>`: the same trace addressed by CONTINGENCY; the host resolves the slot per chunk (`LOOP_TRACE_CTG ctg= slot=`). A fixed slot number follows the chunk composition, which moves with the resolved chunk size (IZUZ sat at slot 170 in one run and 18 in the next) |
| `OLF_GPU_ZI_TREE_CTG` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysis.java — the zero-impedance branches OLF has OUT of its spanning tree for each contingency, printed after lfContingency.apply(). Ground truth for the replay below: OLF's tree is history-dependent within one security analysis |
| `OLF_GPU_ZI_TREE_REPLAY` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchPrecomputer.java — the same set as the precomputer REPLAYS it (the non-tree coupler of each network is the last cycle edge a contingency disabled, and it sticks; a network disabled entirely is re-elected back to base by the merge). Diff against OLF_GPU_ZI_TREE_CTG: 10486 of 10540 exact on network.xiidm, the 54 differing only in the contingency's OWN transient coupler, all of them CPU-routed |
| `OLF_GPU_CV_PROBE` | native/cuda/jni/olf_gpu_nr_jni.cu (`rlCurveLimitsBatch`) — `<rl entry>` or `all`: the ACTIVE POWER the device evaluates each reactive-capability CURVE at and where it came from (base P, the generator's own water-fill cumulative, the share reconstruction and its repairs), plus the resulting min/max delta. OLF re-reads the limit at the generator's current targetP, so this is the number to line up against it (device printf -> surefire dumpstream) |
| `OLF_GPU_DS_FINAL_DUMP` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java — every distributed-slack participant's FINAL delta for slot 0 against the anchor the water-fill resets to (`DS_FINAL_ROW`), the device counterpart of OLF's per-generator `DS_GEN` trace: two water-fills that move the same TOTAL can still split it differently |
| `OLF_GPU_SW_LOG` | native/cuda/jni/olf_gpu_nr_jni.cu (reactive-limit kernel) — `<slot>` records EVERY check of that scenario, `all` records only the checks that DECIDED something, for every scenario. The log is written ON THE DEVICE (one atomicAdd per record) and copied out once per chunk as `SW_LOG s= pass= entry= act= lt= sc= q= min= max= v= cmpV= vt=`. The host-side traces copy hundreds of MB per pass, which serializes the stream and changes the kernel scheduling whose effect is being traced — a run-to-run flip disappears while you watch it. This one found the zero-range pilot entries: two runs of one window, aligned by (chunk, slot, pass, entry), differed on 430 decisions and every one was such an entry |
| `OLF_GPU_RL_SKIP` | native/cuda/jni/olf_gpu_nr_jni.cu (driver) — `<slot>:<outer>`: make ONE scenario's reactive-limit check not happen at ONE pass, reproducing "the trajectory one check short" on demand. With `431:8` on the 2000-branch window of network.xiidm it reproduced the 8.178e-05 kV signature exactly (on whichever contingency held that slot), which is how that failure shape was confirmed without waiting for the rare event |
| `OLF_GPU_RL_AUDIT` | native/cuda/jni/olf_gpu_nr_jni.cu (end of the outer-loop driver) — `<pu>`: audits the FINAL state of every converged scenario against the rule that ENDS OLF's reactive-limits loop. OLF returns only after a check that found nothing to switch, so in its answer every UNPINNED controller is inside its reactive limits; the audit re-fills the diagnostic q at the final state and prints `RL_AUDIT s= entry= vrow= q= min= max= over=` for each one that is not. A defect detector that needs NO CPU arm. Each hit also dumps that scenario's whole recorded decision history (`AUDIT_HIST` / `AUDIT_PASS` / `SW_PASS`), because the failing scenario is not known until the solve is over. Negative-tested with `OLF_GPU_RL_NOPIN_ROW` (1168 hits naming exactly the blocked entry) |
| `OLF_GPU_SW_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu (reactive-limits phase) — `<slot>`: `SW_PASS`, every controller that slot's RL check CHANGED this pass with the quantities the decision turned on (q, live min/max, its own V, the pilot's V and the group target). `OLF_GPU_LOOP_TRACE` says a pass found the slot unstable; this says WHICH controller and by how much |
| `OLF_GPU_RL_TRACE_VROW` | native/cuda/jni/olf_gpu_nr_jni.cu — `<v row>`: `RL_VROW`, one LOCAL reactive-limits entry's q / live limits / row target / mode at every check (the remote-only RL_TRACE never names it) |
| `OLF_GPU_DS_DEBUG_SLOT` | native/cuda/jni/olf_gpu_nr_jni.cu — with `OLF_GPU_DS_DEBUG`: which slot the kernel-level water-fill prints (`[DS-GATE]`/`[DS-WF]`/`[DS-SAT]`, device printf → surefire dumpstream) follow; default 0 |
| `OLF_GPU_HVDC_TRACE` | GpuBatchHvdcAcEmulation / GpuBatchChunkSolver — `<phiRow>` or `all`: `HVDC_BUILD` (every AC-emulation hvdc the pack carries: id, element, OLF status, station targetP), `HVDC_SLOT` (slot -> contingency), `HVDC_PREOV` (pre-freeze override entries already on a converter-bus P row — the freeze composes on top of them), `HVDC_CTL` (per slot and hvdc: base status/deltas, mask bit, warm-point flows, base target), `HVDC_FREEZE` (the flows each scenario's freeze fixed an hvdc at, vs the base deltas) and `HVDC_END` (the status each scenario ended with). Found the doubled-corridor unfreeze race (two links on one bus pair, 2026-09-22) |
| `OLF_GPU_MON_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchVoltageMonitoring.java:59 |
| `OLF_GPU_NAME_ROW` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchPrecomputer.java — turns a DEVICE row index (from `OLF_GPU_WORST_ROW` / `OLF_GPU_PROBE_ROWS`) back into its variable and element; row indices are NOT portable between runs with different contingency sets |
| `OLF_GPU_PHASE_TIME` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuStateInjector.java:86 +9 |
| `OLF_GPU_PRELOSS_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java:697 +2 |
| `OLF_GPU_Q_PROBE` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysis.java:497 +2 |
| `OLF_GPU_QUALIFY_TRACE` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuContingencyMapper.java:219 +9 |
| `OLF_GPU_RESIDUAL_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/GpuNewtonRaphson.java:1065 |
| `OLF_GPU_RL_NOPIN_ROW` | native/cuda/jni/olf_gpu_nr_jni.cu:8850 |
| `OLF_GPU_RL_PACK_DUMP` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuAcOuterLoopPacks.java:484 |
| `OLF_GPU_RL_TRACE` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuAcOuterLoopPacks.java — `GPU_RL_ENTRY idx=<n> vRow=<row> bus=<id>` for every reactive-limit entry, LOCAL and REMOTE (a remote member also prints its class and its pilot's V row): the map that lines the device's per-entry `RL_TRACE` pin states up against OLF's `RL_SWITCH` bus names |
| `OLF_GPU_RMQ_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu:8658 |
| `OLF_GPU_ROW_PROBE` | native/cuda/jni/olf_gpu_nr_jni.cu — `<deviceRow>`: that row in every scenario at every inner iteration (residual, state, target, non-zero Jacobian entries) AND, with `OLF_GPU_BATCH_TRACE`, the slot-0 target AND residual of that row on every `pass done` line (`probeRow=... target=... F=...`): a target a loop PHASE mutates between solves (HVDC freeze/unfreeze, DS anchor reset) is invisible to the per-iteration print. Found the doubled-corridor unfreeze race (2026-09-22) |
| `OLF_GPU_SA_BUSDIFF` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:348 |
| `OLF_GPU_SA_BUSDIFF_CTG` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — `<id>[,<id>]`: bus-diff those contingencies INSIDE the full scope (one scoped StateMonitor per id, so the heap holds a handful of results, not 10540 x 13202). The list decides the retained switches on a node-breaker network, so a contingency that differs in the full scope cannot be studied alone (CTG_ONLY) |
| `OLF_GPU_SA_DEBUG_CTG` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:164 |
| `OLF_GPU_SA_INVENTORY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:205 |
| `OLF_GPU_SA_MON_BRANCH` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:353 +1 |
| `OLF_GPU_SA_MON_BUS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:523 +1 |
| `OLF_GPU_SA_RESID` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysis.java:791 +3 |
| `OLF_GPU_SCOPE_CENSUS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:77 |
| `OLF_GPU_SEED_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchLossPreDistribution.java:181 |
| `OLF_GPU_SEQ_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/GpuNewtonRaphson.java:1165 |
| `OLF_GPU_SHUNT_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchIncrementalShuntVc.java:98 |
| `OLF_GPU_STATE_DUMP` | native/cuda/jni/olf_gpu_nr_jni.cu — for slot 0 and iterations 0..&lt;value&gt;, EVERY row's mode/target/state/residual (systems up to 64 rows). On a reproducer it shows at a glance which variable ran away and which equation let it |
| `OLF_GPU_SVC_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchSecondaryVc.java:464 +2 |
| `OLF_GPU_SVC_WB_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchSecondaryVc.java:223 |
| `OLF_GPU_TARGET_ROW` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java:797 |
| `OLF_GPU_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu:289 +2 |
| `OLF_GPU_UNREALISTIC_TRACE` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysis.java:808 +1 |
| `OLF_GPU_ZI_DEST_TRACE` | native/cuda/jni/olf_gpu_nr_jni.cu — one zero-impedance element's Q destinations and alternates (b1Q/b2Q/alt0/alt1/dd2/dd3/dd6/dd7); a suppressed Q row shows as -1 and the alternate is what must carry the coupling |
| `OLF_OL_TRACE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuHybridOuterLoopBridge.java:51 |
| `OLF_RL_PROBE_BUS` | (OLF repo, branch olf-local-patches) src/main/java/com/powsybl/openloadflow/ac/outerloop/ReactiveLimitsOuterLoop.java |

## Test scope selectors (9)

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
| `OLF_GPU_SA_BASE_ONLY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java + GpuBatchedAcSecurityAnalysis — `1`: both arms run the BASE load flow only (the GPU arm hands an empty contingency list to the SA after its base) and the test stops after the base comparison; the lever that isolates a base-on-device failure from a 20-minute contingency sweep |

## Build / toolchain (4)

Paths, set by `native/local-env.sh`.

| variable | read at |
|---|---|
| `CUDA_HOME` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuNativeLoader.java:75 +1 |
| `CUDA_PATH` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuNativeLoader.java:76 +1 |
| `CUDA_TOOLKIT` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuNativeLoader.java:77 +1 |
| `CUDSS_ROOT` | src/main/java/com/powsybl/openloadflow/ac/equations/vector/gpu/GpuNativeLoader.java:80 |

## Highest risk: tolerances that apply to the GPU ARM ONLY

These set a tolerance for the GPU arm that the CPU arm does not get:

| variable |
|---|
| `OLF_GPU_SA_GPU_CONV_EPS` |
| `OLF_GPU_SA_GPU_RL_TOL` |
| `OLF_GPU_SA_GPU_SLACK_TOL` |
| `OLF_GPU_SA_GPU_BASE` |

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

## BEHAVIOUR-CHANGING (39) — REVIEW

Each selects a different computation. Several disable an outer loop or loosen a tolerance, which is wrong by construction (and note the standing rule: never loosen a tolerance to make the GPU agree with the CPU). These should be justified individually or deleted.

| variable | read at |
|---|---|
| `OLF_BUSSTATE_PROFILE` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuHybridBatchOuterLoopEngine.java:1447 |
| `OLF_GPU_BASE_ON_GPU` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedN1FixedIterTest.java:149 +2 |
| `OLF_GPU_BATCHMODE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:87 +5 |
| `OLF_GPU_DBG_CONV_EPS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:295 +2 |
| `OLF_GPU_DBG_SLACK_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysisTest.java:3084 +1 |
| `OLF_GPU_DISC_EPS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuSingleSolveRejectReasonProbe.java:60 |
| `OLF_GPU_DISC_SLACKTOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuSingleSolveRejectReasonProbe.java:64 |
| `OLF_GPU_DRIVER_OUTER_LOOPS` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysis.java:319 +2 |
| `OLF_GPU_FIXED_ITERS` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedN1FixedIterTest.java:79 |
| `OLF_GPU_FORCE_FIRST_STEP` | native/cuda/jni/olf_gpu_nr_jni.cu:7630 |
| `OLF_GPU_HOST_LOOP_NOOP` | native/cuda/jni/olf_gpu_nr_jni.cu:8137 |
| `OLF_GPU_HYBRID_N` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuHybridBatchValidationTest.java:132 |
| `OLF_GPU_IR_STEPS` | native/cuda/jni/olf_gpu_nr_jni.cu (`irSteps`) — cuDSS iterative-refinement steps; default 4 when the structure has a zero-impedance branch (2 until 2026-09-28: the batched factorization's accuracy on the repaired zero-diagonal rows depends on which other matrices share the batch — network.xiidm/LLAX solved to 2.5e-06 in one chunk composition and 2.2e-08 in another; 4 steps give 1.1e-11), 0 otherwise. Re-measured 2026-09-21 after the zero-diagonal row repair and STILL needed: IR=0 leaves ‖J.dx−F‖∞ 3.8e-4 on the repaired coupler/controller rows (2.6e-8 kV vs 2.6e-9 with IR=2 on rte6515's 500 busbars) |
| `OLF_GPU_MATCHING_ALG` | native/cuda/jni/olf_gpu_nr_jni.cu:250 |
| `OLF_GPU_MAXBATCH` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:83 +5 |
| `OLF_GPU_NO_DIST_SLACK` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuCpuOuterLoopTrajectoryProbe.java:55 |
| `OLF_GPU_NO_DS` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedPegaseOutlierDiagnosticTest.java:69 |
| `OLF_GPU_NO_RL` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedPegaseOutlierDiagnosticTest.java:70 |
| `OLF_GPU_REORDER_ALG` | native/cuda/jni/olf_gpu_nr_jni.cu (`applyReorderAlg`, `applyRobustSingleMatrixConfig`) — cuDSS reordering for the batched contexts (default per structure: AMD, ND when zero-impedance branches are present); the ROBUST single-matrix mode 6 (the batch-of-one retry after a linear-solve reject, status -3) keeps AMD unless set, and enables matching + pivoting only there (cuDSS 0.8 refuses matching under the COLAMD family) |
| `OLF_GPU_SA_CONV_EPS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:197 |
| `OLF_GPU_SA_CPU_SINGLE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:276 |
| `OLF_GPU_SA_GPU_BASE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:92 |
| `OLF_GPU_SA_GPU_CONV_EPS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:242 |
| `OLF_GPU_SA_GPU_RL_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:244 |
| `OLF_GPU_SA_GPU_SLACK_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:243 |
| `OLF_GPU_SA_MAX_OUTER` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:219 |
| `OLF_GPU_SA_NO_DS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:229 |
| `OLF_GPU_SA_NO_RL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:226 |
| `OLF_GPU_SA_RL_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:211 |
| `OLF_GPU_SA_SEQ_SOLVER` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:387 |
| `OLF_GPU_SA_SLACK_TOL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:205 |
| `OLF_GPU_STRAGGLER_FRAC` | native/cuda/jni/olf_gpu_nr_jni.cu (`stragglerFrac`) — perf switch, default 0 = OFF since 2026-09-22: `<frac>` ejects the last `frac*nS` unconverged scenarios at maxIter/2 to the CPU replay; off, the device carries them to maxIter and reports MAX_ITERATION_REACHED itself (the precomputer's `exhausted` set). |
| `OLF_GPU_SVC` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:310 |
| `OLF_GPU_SVC_DBG` | native/cuda/jni/olf_gpu_nr_jni.cu:7527 |
| `OLF_GPU_SVC_HOST_ORCHESTRATOR` | src/main/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchChunkSolver.java:998 +1 |
| `OLF_GPU_SVC_INERT` | native/cuda/jni/olf_gpu_nr_jni.cu:7515 |
| `OLF_GPU_SVC_PILOT_SHIFT` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:317 |
| `OLF_GPU_TVD_REJECT_CARRIER_CLAMP` | native/cuda/jni/olf_gpu_nr_jni.cu:6208 |
| `OLF_GPU_VIOL_CROSSCHECK` | src/main/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysis.java:305 +1 |
