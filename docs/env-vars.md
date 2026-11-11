# GPU environment variables

Every one of these is read at runtime. The project's standing goal is to **match the CPU at machine
precision on every difference**, so the dangerous category is the third table: a switch that selects a
different computation can preserve a wrong answer behind a default, and nothing in CI explores it.

Three were removed on 2026-09-17 for exactly that reason, and five on 2026-11-10, the first two with the
`olf.gpu.test.convEps` / `olf.gpu.test.slackTol` system properties that were read before them:

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
| `OLF_GPU_PHASE_TIME` | java + native | Where the run's time goes: phases of the security analysis, chunk sizing, context reuse. Native: where the time goes (analysis, outer-loop passes, solves) |
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
| `OLF_GPU_STATE_VIEW_CHECK` | java | Every device-answered contingency's detection inputs recomputed from the device state array and compared, bit for bit, with what OLF's objects return after the write-back; a summary and the first mismatches, by number. |
| `OLF_GPU_SW_LOG` | native | `<slot>` or `all`: every reactive-limit check of that slot, logged on the device |
| `OLF_GPU_SW_TRACE` | native | `<slot>`: every controller that switches in that slot's reactive-limit passes |
| `OLF_GPU_TRACE` | native | cuDSS analysis timing and the zero-diagonal repair's tables |
| `OLF_GPU_UNREALISTIC_TRACE` | java + native | The post-contingency realistic-voltage check: which buses make a state unrealistic. Native: the buses that make a post-contingency state unrealistic |
| `OLF_GPU_VIOL_CROSSCHECK` | java | Every contingency's violations compared with OLF's stock detection over every branch. |
| `OLF_GPU_ZITREE_TRACE` | java | The zero-impedance coupler trees relocated, and the seeds of their couplers. |
| `OLF_OL_STATS` | java | Each contingency's outer-loop passes and Newton iterations, in the shape of OLF's own `CPU_OL_STATS`. |
| `OLF_TRACE_CTG` | java | `<contingency id>`: OLF's own filter for its debug dumps, which `OLF_GPU_SA_VAR_DUMP` follows. |

## Test diagnostics (28) — print only, read by the tests

| variable | read at |
|---|---|
| `OLF_GPU_CENSUS_BASE_VIOLATIONS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:300 |
| `OLF_GPU_CENSUS_BUS_DUMP` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:309 |
| `OLF_GPU_CENSUS_CASE` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:69 |
| `OLF_GPU_CENSUS_CHUNK` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:242 |
| `OLF_GPU_CENSUS_CONNECTIVITY` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuRteScopeCensusTest.java:342 |
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
| `OLF_GPU_DBG_CENSUS` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysisTest.java:3112 |
| `OLF_GPU_DBG_REPORT` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuBatchedAcSecurityAnalysisTest.java:3100 +2 |
| `OLF_GPU_DBG_RL` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java:313 +2 |
| `OLF_GPU_SA_SELF_DUMP` | src/test/java/com/powsybl/openloadflow/sa/gpu/GpuVsCpuRte6515SaTest.java — `<file>`: with a warm run (no `OLF_GPU_SA_COLD_ONLY`), every contingency's status and distributed active power cold and warm; the `POST GPU-vs-GPU` line prints the same comparison summarized |
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
bug to fix, not a tolerance to loosen.** Tightening both arms does not answer it
either: the arms must take the same iterates at the default criterion, and the rte6515 harness's `OLF_GPU_DBG_NR`
prints the CPU arm's trajectory to put beside the device's.

Also in the list and wrong by construction, useful only as momentary bisection levers:
`OLF_GPU_NO_DS`, `OLF_GPU_NO_RL`, `OLF_GPU_NO_DIST_SLACK`, `OLF_GPU_SA_NO_DS`, `OLF_GPU_SA_NO_RL`,
`OLF_GPU_HOST_LOOP_NOOP`, `OLF_GPU_SVC_INERT`. Each disables an outer loop or makes it inert, so any
run with one set cannot match the CPU. They should be deleted once the question that motivated them is
answered — the three removed above were removed on exactly that ground.

## BEHAVIOUR-CHANGING (18) — REVIEW

Each selects a different computation. Several disable an outer loop or loosen a tolerance, which is wrong by construction (and note the standing rule: never loosen a tolerance to make the GPU agree with the CPU). These should be justified individually or deleted.

| variable | read at |
|---|---|
| `OLF_GPU_BASE_ON_GPU` | src/test/java/com/powsybl/openloadflow/ac/solver/gpu/batch/GpuBatchedN1FixedIterTest.java:149 +2 |
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
