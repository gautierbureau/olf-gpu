# GPU environment variables

Every one of these is read at runtime. The project's standing goal is to **match the CPU at machine
precision on every difference**, so the dangerous category is a switch that selects a different computation:
it can preserve a wrong answer behind a default, and nothing in CI explores it. None is left (see the removed
table below and the last section).

Three were removed on 2026-09-17 for exactly that reason, five on 2026-11-10, the first two with the
`olf.gpu.test.convEps` / `olf.gpu.test.slackTol` system properties that were read before them, and the
forty behaviour-changing ones on 2026-11-17: the eighteen filed as such, and twenty-two more read by tests but
filed elsewhere or not documented (every test now does what it did with the variable unset, except where noted):

| removed | why |
|---|---|
| `OLF_GPU_CURVE_GENCUM` | re-enabled the share-reconstructed curve P after the per-generator read was proven better (rte6515 worst generator error 3.790e-02 -> 4.837e-05 kV) |
| `OLF_GPU_NO_CURVES` | a bisection lever that made curve buses wrong by design; it had served its purpose |
| `OLF_GPU_SA_NO_BATCH` | duplicated the pre-existing `-Dolf.gpu.sa.disableBatch=true` |
| `OLF_GPU_DBG_CONV_EPS` | tightened OLF's Newton criterion on both arms (the SA test helpers, the rte6515 harness). Tightening hides where the arms' trajectories part instead of answering it |
| `OLF_GPU_DBG_SLACK_TOL` | tightened the slack band on both arms (the SA test helpers). The two tests that set it match the CPU at 1e-12 kV at OLF's default criteria |
| `OLF_GPU_CENSUS_CONV_EPS` | tightened OLF's Newton criterion on both arms of the scope census (`GpuRteScopeCensusTest`) |
| `OLF_GPU_DISC_EPS` | tightened OLF's Newton criterion on both arms of `GpuSingleSolveRejectReasonProbe` |
| `OLF_GPU_DISC_SLACKTOL` | tightened the slack band on both arms of `GpuSingleSolveRejectReasonProbe` |
| `OLF_GPU_FIXED_ITERS` | the K sweep of the fixed-iteration bench: it runs K=4, its default and `tools/run_bench.sh`'s |
| `OLF_GPU_MAXBATCH` | forced the chunk size of three benches: the fixed-iteration bench now fixes it at 500 (`tools/run_bench.sh` always set 500: the like-for-like comparison with gpusim2grid); the production-SA and Pegase load-flow benches take the production auto-sized chunk, their unset behaviour |
| `OLF_GPU_BASE_ON_GPU` | ran the fixed-iteration bench's base load flow on the GPU; it runs on the CPU, as unset |
| `OLF_GPU_SA_CONV_EPS`, `OLF_GPU_SA_SLACK_TOL`, `OLF_GPU_SA_RL_TOL`, `OLF_GPU_SA_MAX_OUTER` | overrode the RTE parameters' criteria on both arms of the production-SA bench |
| `OLF_GPU_SA_GPU_CONV_EPS`, `OLF_GPU_SA_GPU_SLACK_TOL`, `OLF_GPU_SA_GPU_RL_TOL` | one-sided tolerances on the production-SA bench's GPU arm: both arms now run the same parameters |
| `OLF_GPU_SA_CPU_SINGLE` | added a 1-thread CPU arm to the production-SA bench |
| `OLF_GPU_SA_NO_RL`, `OLF_GPU_SA_NO_DS` | disabled an outer loop on both arms of the production-SA bench and the rte6515 harness |
| `OLF_GPU_NO_DS`, `OLF_GPU_NO_RL` | disabled an outer loop in `GpuBatchedPegaseOutlierDiagnosticTest` |
| `OLF_GPU_NO_DIST_SLACK` | disabled the distributed slack in `GpuCpuOuterLoopTrajectoryProbe` |
| `OLF_GPU_SVC`, `OLF_GPU_SVC_PILOT_SHIFT` | secondary voltage control in the rte6515 harness: now its own test, `gpuSaMatchesCpuOnRte6515FullWithSecondaryVoltageControl`, with every pilot target shifted by 1.0 kV (the run recorded in docs/tasks.md) |
| `OLF_GPU_TOL`, `OLF_GPU_E2E`, `OLF_GPU_E2E_K`, `OLF_GPU_E2E_WARMJIT` | the fixed-iteration bench's tolerance and end-to-end passes: it runs at tol 0 (fixed iterations) with both end-to-end passes at K=4, as `tools/run_bench.sh` always set them |
| `OLF_GPU_AGREE_TOL`, `OLF_GPU_SLACK_TOL` | the Pegase load-flow test's agreement and slack tolerances (1e-6, 1e-8) and the outlier diagnostic's slack tolerance (1e-2), their defaults |
| `OLF_GPU_SA_CPU_THREADS` | the CPU arm's thread count: 4 in the production-SA bench, 1 (bus-breaker) or 8 in the rte6515 harness, their defaults |
| `OLF_GPU_CENSUS_CPU`, `_NO_RL`, `_NO_DS`, `_NO_ACTIVE_LIMITS`, `_VIOL_TOL`, `_THREADS`, `_NOFREEZE`, `_SVC`, `_CHUNK` | the scope census's control arm, outer-loop, tolerance, thread, freeze, SVC and chunk-size switches (filed as print-only diagnostics, they changed the computation): the census runs the GPU provider on the RTE parameters with OpenLoadFlow's SA defaults |
| `OLF_GPU_SA_BANK`, `OLF_GPU_SA_CPU_WARM`, `OLF_GPU_SA_COLD_ONLY`, `OLF_GPU_SA_DEFAULT_INCREASED` | the rte6515 harness's forced bank size, extra warm CPU arm, skipped warm GPU run and powsybl-default violation thresholds: it runs the device-sized bank, one CPU arm, both GPU runs and the 0.5 kV / 0.1 % thresholds |

Classification below is by NAME and call site, not an individual audit of each one's semantics.
Anything in the third table should be treated as unverified until someone reads it.

## System properties

| property | effect |
|---|---|

## Trace channels (38) — print only, cannot change a result

Each one is read once: on the Java side by the enum `com.powsybl.openloadflow.util.gpu.GpuTrace`, on the native
side by `native/cuda/common/trace.cuh`. A variable both sides read is listed on both. This table is generated
from those two lists; add a channel there, then here.

| variable | side | prints |
|---|---|---|
| `OLF_GPU_BATCH_TRACE` | java + native | Per-pass and per-iteration convergence of the batched solves, and what every outer loop decided. Native: per-iteration convergence, and what every outer loop decided |
| `OLF_GPU_CHUNK_DUMP` | java | `<file>`: every chunk's contingencies in slot order, one line per chunk. |
| `OLF_GPU_CPU_ROUTED` | java | The contingencies the device hands back to the CPU, and why. |
| `OLF_GPU_DBG_BUSDUMP` | java | `1`: the buses, rows and islands a branch outage touches, and the bus values injected back. |
| `OLF_GPU_DQ_TRACE` | native | `<row>`: every DISTR_Q residual contribution landing in that row |
| `OLF_GPU_DS_ACCOUNT` | java | The distributed active power booked when a load flow's distributed slack check falls back to OLF's own. |
| `OLF_GPU_DX_DUMP_ON_REJECT` | native | `<dir>`: J, F, B, dx and modes of the slots the linear-solve guard rejects |
| `OLF_GPU_EQ_DUMP` | java | `<file>`: the GPU equation system of the load flow as text, once (the first pass with a bus switched to PQ). |
| `OLF_GPU_EQ_DUMP_CTG` | java | `<contingency id>`: OLF's post-contingency equation system of that contingency, as text. |
| `OLF_GPU_EQ_DUMP_CTG_FILE` | java | `<file>`: where `OLF_GPU_EQ_DUMP_CTG` writes (default: the contingency id + `.olf.post`). |
| `OLF_GPU_EQ_DUMP_WHEN` | java | `any`: `OLF_GPU_EQ_DUMP` writes the first pass, with a bus switched to PQ or not. |
| `OLF_GPU_HVDC_TRACE` | java | `<phiRow>` or `all`: the AC emulation HVDC links the pack carries and their loop decisions. |
| `OLF_GPU_ITER_TRACE_CTG` | java | `<id>[,<id>]`: iteration count, final mismatch and distributed active power of those contingencies. |
| `OLF_GPU_LOOP_TRACE` | native | `<slot>`: that slot's outer-loop passes (or the OLF_GPU_LOOP_TRACE_CTG slot) |
| `OLF_GPU_LU_COMPARE` | native | Every Newton step of our batched LU against cuDSS's uniform batch on the same systems: worst difference, backward errors, clamped pivots |
| `OLF_GPU_LU_PATTERN_DUMP` | native | `<file>`: our batched LU's pattern, elimination order and switch dummy rows, one binary file per structure size (`<file>-n<n>`, integers only) |
| `OLF_GPU_LOOP_TRACE_CTG` | java | `<contingency id>`: the native outer-loop trace of that contingency's slot, in every chunk. |
| `OLF_GPU_PHASE_TIME` | java + native | Where the run's time goes: phases of the security analysis, chunk sizing, context reuse. Native: where the time goes (analysis, outer-loop passes, solves); in the DC library, each cuDSS phase |
| `OLF_GPU_PRELOSS_TRACE` | java | The replayed loss pre-distribution: what each contingency's generators take before the first solve. |
| `OLF_GPU_QUALIFY_TRACE` | java | Why each contingency is batched, or which rule sends it to the CPU. |
| `OLF_GPU_REJECT_TRACE` | java + native | The scenarios the device rejects, with the reason. |
| `OLF_GPU_RL_AUDIT` | native | `<tol>`: the final state audited against the rule that ends OLF's reactive-limit loop |
| `OLF_GPU_RL_TRACE` | java + native | The reactive-limit entries the pack builds, and the device's switches to PQ and back to PV per scenario. Native: the reactive-limit tables a context is built with |
| `OLF_GPU_SA_EQ_DUMP` | java | `<file>`: the security analysis extraction's equation system as text. |
| `OLF_GPU_SA_RESID` | java | `<n>`: the n worst rows of OLF's residual at the device's post-contingency state, with their equation. |
| `OLF_GPU_SA_VAR_DUMP` | java | `<file>`: the device's post-contingency state per OLF variable (scoped by `OLF_TRACE_CTG`). |
| `OLF_GPU_SEED_DUMP` | java | `<file>`: the per-generator seed of the loss pre-distribution of every contingency. |
| `OLF_GPU_SHUNT_TRACE` | java | The incremental shunt voltage control's sensitivities and decisions. |
| `OLF_GPU_SLOT_DUMP` | native | `<prefix>`: the traced slot's row targets, state and modes at driver start and pass 0 |
| `OLF_GPU_STATE_DUMP` | native | `<n>`: slot 0's rows (mode, target, state, residual) for iterations 0..n |
| `OLF_GPU_STATE_VIEW_CHECK` | java | Every device-answered contingency's detection inputs recomputed from the device state array and compared, bit for bit, with what OLF's objects return after the write-back; a contingency answered from its state, or without the network, also answered the usual way and the decisions compared; every screened detection compared with a full scan, a retapped branch's screen row with its pi model, and the side-opened branches' screen values measured against the exact ones. Every full state is read back too, and a contingency answered from its gathered rows is checked against it. A summary and the first differences, by number. |
| `OLF_GPU_STRUCTURE_FINGERPRINT` | native | `<file>` (appended to) or `1` (stderr): one line per shared structure built: its size and a hash of each part (pattern, each element family's indices and destinations, distributions, zero-impedance, reactive relocation, zero-diagonal repair), to check that a change leaves the structure as it was |
| `OLF_GPU_SW_LOG` | native | `<slot>` or `all`: every reactive-limit check of that slot, logged on the device |
| `OLF_GPU_SW_TRACE` | native | `<slot>`: every controller that switches in that slot's reactive-limit passes |
| `OLF_GPU_TRACE` | native | cuDSS analysis timing and the zero-diagonal repair's tables |
| `OLF_GPU_UNREALISTIC_TRACE` | java + native | The post-contingency realistic-voltage check: which buses make a state unrealistic. Native: the buses that make a post-contingency state unrealistic |
| `OLF_GPU_VIOL_CROSSCHECK` | java | Every contingency's violations compared with OLF's stock detection over every branch. |
| `OLF_GPU_ZITREE_TRACE` | java | The zero-impedance coupler trees relocated, and the seeds of their couplers. |
| `OLF_OL_STATS` | java | Each contingency's outer-loop passes and Newton iterations, in the shape of OLF's own `CPU_OL_STATS`. |
| `OLF_TRACE_CTG` | java | `<contingency id>`: OLF's own filter for its debug dumps, which `OLF_GPU_SA_VAR_DUMP` follows. |

## Test diagnostics (20) — print only, read by the tests

| variable | read at |
|---|---|
| `OLF_GPU_CENSUS_BASE_VIOLATIONS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:300 |
| `OLF_GPU_CENSUS_BUS_DUMP` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:309 |
| `OLF_GPU_CENSUS_CASE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:69 |
| `OLF_GPU_CENSUS_CONNECTIVITY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:342 |
| `OLF_GPU_CENSUS_MONITOR_ALL_VL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:253 |
| `OLF_GPU_CENSUS_MONITOR_BRANCH` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:257 |
| `OLF_GPU_CENSUS_MONITOR_VL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:256 |
| `OLF_GPU_CENSUS_ONLY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:152 |
| `OLF_GPU_CENSUS_VIOLATION_DUMP` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:354 |
| `OLF_GPU_DBG_CENSUS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysisTest.java:3112 |
| `OLF_GPU_DBG_REPORT` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysisTest.java:3100 +2 |
| `OLF_GPU_DBG_RL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:313 +2 |
| `OLF_GPU_SA_SELF_DUMP` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — `<file>`: with the warm run, every contingency's status and distributed active power cold and warm; the `POST GPU-vs-GPU` line prints the same comparison summarized |
| `OLF_GPU_SA_BUSDIFF` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:348 |
| `OLF_GPU_SA_BUSDIFF_CTG` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — `<id>[,<id>]`: bus-diff those contingencies INSIDE the full scope (one scoped StateMonitor per id, so the heap holds a handful of results, not 10540 x 13202). The list decides the retained switches on a node-breaker network, so a contingency that differs in the full scope cannot be studied alone (CTG_ONLY) |
| `OLF_GPU_SA_DEBUG_CTG` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:164 |
| `OLF_GPU_SA_INVENTORY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:205 |
| `OLF_GPU_SA_MON_BRANCH` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:353 +1 |
| `OLF_GPU_SA_MON_BUS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:523 +1 |
| `OLF_GPU_SCOPE_CENSUS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:77 |

## Read by the patched OLF (2)

| variable | read at |
|---|---|
| `OLF_EQ_DUMP` | OLF patch, NewtonRaphson — writes the equation system OLF is about to solve as text (variable rows, column->equation map, `writeToString(true)`). The one question a residual cannot answer is WHICH equation sits on a row, and OLF will simply print it |
| `OLF_RL_PROBE_BUS` | (OLF repo, branch olf-local-patches) src/main/java/com/powsybl/openloadflow/ac/outerloop/ReactiveLimitsOuterLoop.java |

## Test scope selectors (11)

Choose which contingencies a validation run covers. They change what is measured, not how it is computed.

| variable | read at |
|---|---|
| `OLF_GPU_SA_CPU_ARM_DIR` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — with `OLF_GPU_SA_CPU_ARM_MODE=save`: run the CPU arm only and write it (cpu.json, cpu-busv.bin/.idx, cpu-ms.txt) here; `=load`: rebuild the CPU arm from these files and run the GPU arms only. The two arms of the full busbar scope do not fit one JVM. |
| `OLF_GPU_SA_CPU_ARM_MODE` | see `OLF_GPU_SA_CPU_ARM_DIR` (`save` / `load`) |
| `OLF_GPU_SA_GPU_ONLY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — `1`: run the CPU arm on an EMPTY contingency list (debug lever: the GPU arm's own behaviour on a large scope in minutes; every comparison is then meaningless and the CPU base sits on a different bus decomposition). |
| `OLF_GPU_CONT_FILE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuProductionSaBenchmarkTest.java:77 +4 |
| `OLF_GPU_SENSI_BENCH` | src/test/java/com/powsybl/openloadflow/sensi/gpu/GpuVsCpuSensitivityBenchmarkTest.java — set: run the GPU against CPU sensitivity benchmark (with `OLF_GPU_CONT_FILE`); unset, it is skipped |
| `OLF_GPU_SENSI_CASE` | src/test/java/com/powsybl/openloadflow/sensi/gpu/GpuVsCpuSensitivityBenchmarkTest.java — the case under `OLF_GPU_CASE_DIR` to run instead of case9241pegase |
| `OLF_GPU_SENSI_NCTG` | src/test/java/com/powsybl/openloadflow/sensi/gpu/GpuVsCpuSensitivityBenchmarkTest.java — keep only the first N contingencies of `OLF_GPU_CONT_FILE` |
| `OLF_GPU_SA_CTG_ONLY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:218 +1 |
| `OLF_GPU_SA_MAX_CTG` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:233 |
| `OLF_GPU_SA_CASE_FILE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — path of the case to run instead of rte6515 (e.g. `network.xiidm` at the repo root); the other selectors apply unchanged |
| `OLF_GPU_SA_CTG_RANGE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — `from:to`: keep only that index window of the selected contingency list (after `OLF_GPU_SA_CTG_KINDS`); a window is ONE or a few chunks — always finish on the full width |

## Build / toolchain (4)

Paths, set by `native/local-env.sh`.

| variable | read at |
|---|---|

## No one-sided tolerance, no disabled outer loop

None of the variables left sets a tolerance for the GPU arm only, or disables or empties an outer loop: the last
ones were removed on 2026-11-17 (`OLF_GPU_SA_GPU_*`, `OLF_GPU_*NO_DS`, `OLF_GPU_*NO_RL`, `OLF_GPU_NO_DIST_SLACK`;
`OLF_GPU_HOST_LOOP_NOOP` and `OLF_GPU_SVC_INERT` were already gone from the code). A one-sided tolerance is the
mechanism by which a GPU/CPU difference can be made to disappear without being fixed, which the project forbids:
**a disagreement at the default tolerance is an orchestration bug to fix, not a tolerance to loosen.** Tightening
both arms does not answer it either: the arms must take the same iterates at the default criterion, and the
rte6515 harness's `OLF_GPU_DBG_NR` prints the CPU arm's trajectory to put beside the device's.
