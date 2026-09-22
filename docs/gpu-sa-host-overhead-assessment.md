# GPU security analysis — where the time goes, and which leads are worth pursuing

Assessment of 2026-09-21, written BEFORE any modification, to trace the leads and their evidence.
Every "gain" is an estimate unless it says "measured"; a lead marked **DONE** records its measured outcome.

## 1. Setup and method

| | |
|---|---|
| Machine | NVIDIA H100 PCIe 80 GB (sm_90), 30 cores, 364 GB RAM |
| Case | `rte6515_full.xiidm.gz`, RTE production parameters (`rte-loadflow-default-parameters.json`) |
| Contingencies | **14,260** = every kind except busbar sections: 9042 branches, 1389 generators (all voltages), 3673 loads (all), 122 shunts, 23 batteries, 6 SVCs, 5 HVDC lines |
| Test | `GpuVsCpuRte6515SaTest`, `OLF_GPU_SA_CTG_KINDS=branch,generator,load,hvdc,svc,shunt,battery OLF_GPU_SA_GEN_MIN_KV=0 OLF_GPU_SA_LOAD_TOP=1000000` |
| GPU arm | `GpuSecurityAnalysisProvider`, one host thread (threadCount > 1 crashes, see L9) |
| Code | GPU repo `b963f0c`, OLF `olf-local-patches` `b85b1fae8` |

Instruments, and how far each can be trusted:

- **Wall-clock timers of the test** (`CPU (n threads)`, `GPU (COLD/warm)`): exact.
- **`OLF_SA_PROFILE=1`** (OLF): per-contingency timers around apply / loss distribution / simulation / restore,
  summed over threads, cumulative over the JVM (per-arm values = differences between successive prints).
  Cheap (a few `nanoTime` per contingency): the run with it on is within 1% of the run without.
- **`OLF_GPU_PHASE_TIME=1`** (GPU repo): phase timers of the GPU path, JNI marshal/solve/pack split.
- **JFR** (`OLF_GPU_SA_JFR=<file>`, profile of the cold GPU run only): each sample is weighted by the wall
  time until that thread's next sample (the thread runs continuously: 3 gaps > 0.2 s, 1.3 s total).
  Agrees with the direct timers where both exist (post-contingency call 13.0 s JFR vs 11.5 s timer).
- **`OLF_BUSSTATE_PROFILE=1`: do NOT use for magnitudes on this case.** It times each of the ~93 M bus
  restores and inflated the GPU arm from 47 s to 77 s (restore 14.9 s -> 42.8 s).

## 2. Baseline (measured)

| CPU threads | CPU cold | CPU warm | GPU cold | GPU warm | GPU vs CPU warm |
|---|---|---|---|---|---|
| 8 | 124.1 s | 118.6 s | 50.4 s | 46.9 s | x2.35 cold / x2.53 warm |
| 16 | 68.3 s | 68.5 s | 50.8 s | 49.2 s | x1.35 / x1.39 |
| 30 (CPU cold only) | ~58–60 s | — | ~47–49 s | ~46–48 s | ~x1.2–1.3 |

Every run: statuses identical (14255 converged, 3 failed, 2 max-iterations), 906,447 violations on both
arms, 0 contingencies with a differing violation count, 14258/14260 solved on the device (2 device
rejections, status −2, fall back to the CPU). The CPU scales 8 -> 16 threads x1.73, 16 -> 30 only ~x1.15.
**The GPU arm is flat at ~47–50 s whatever the CPU thread count or the chunk size.**

## 3. Where the GPU arm's ~47–51 s go (cold run, 51.7 s under JFR)

| Phase | Wall | Source |
|---|---|---|
| Base load flow + setup (base LF **on the GPU**: batch of one, 11 Newton iterations / 13 outer passes, 96 ms device) | ~1.7 s | JFR + JNI counter |
| **Precompute** (`GpuBatchPrecomputer.precomputeLocked`) | **16.3 s** | JFR |
| — `solveBatchContext` (JNI): device solve 7.8 s, marshal 1.2 s, pack 0.4 s | 10.3 s | JFR + JNI timers |
| — contingency qualification (`qualifyOne` -> connectivity) | 2.55 s | JFR |
| — override composition + HVDC `indexOf` | 1.8 s | JFR |
| — scenario planning, extraction | 0.3 s | JFR |
| **Per-contingency loop** (OLF `AbstractSecurityAnalysis.processContingency`, host, 1 thread) | **33.0 s** | JFR |
| — **`NetworkState.restore`** | **14.3–14.9 s** | JFR / SA_PROFILE |
| — GPU post-contingency simulation (inject, update, checks, violations, results) | 11.1–13.0 s | SA_PROFILE / JFR |
| — **contingency lost-power redistribution** | **4.6–5.7 s** | SA_PROFILE / JFR |

Precompute-side device work is ~8 s of ~50: **the GPU is no longer the bottleneck of the GPU arm; single-
threaded host code is.**

## 4. Leads

Each lead: measured cost, mechanism, what depends on it (the correctness constraints), the change, where it
lives, gain, risk, and how to verify. Ordered by expected gain.

### L1 — `NetworkState.restore` restores the WHOLE network after every contingency — 14.9 s

- **Mechanism.** `NetworkState.save` captures every bus, branch, HVDC and area; `restore()` rewrites all of
  them after each contingency (`AbstractSecurityAnalysis.processContingency`, "restore base state"), plus
  `setExcludedSlackBuses` on each synchronous network. 14,260 x ~6,500 buses ≈ 93 M bus restores.
- **JFR breakdown (14.3 s):** `BusState.restore` 9.6 s — of which `AbstractLfBus.setDisabled` 4.1 s (the
  guard makes it a no-op compare, paid 93 M times), `restoreGenerators` 2.7 s, `restoreLoads` 0.9 s —;
  `BranchState.restore` 2.8 s; `setExcludedSlackBuses` 1.2 s (rebuilds the bus list by stream each call).
- **What the GPU path actually mutates per device-solved contingency:** `lfContingency.apply()` (disabled
  branches/buses/HVDCs, side disconnections, shunt G/B shifts, lost loads and generators); the loss
  redistribution (participating generators' target P — see L2); the injected state (bus V/angle via
  `AcSolverUtil.updateNetwork`). Everything else the restore walks is unchanged.
- **Constraints.** (a) `apply()`'s changes MUST be undone. (b) Bus V/angle must be the base ones before a
  CPU-routed contingency: OLF's post-contingency Newton starts from the current bus values. The next
  device-solved contingency overwrites them, so they do not matter there. (c) Listener side effects:
  `setDisabled` / `setTargetP` notify the equation-system updaters; a partial restore must trigger the
  same notifications for exactly the elements it reverts.
- **Prior art (read before changing anything).** OLF `2223c8e56` (2026-09-10) measured, on the hybrid
  path, that a bus restore's cost is almost entirely `generator.setTargetP` on CHANGED targets (~4 µs, the
  listener fan-out), and that the hybrid's dirty-tracking "did not pay despite restoring 8x fewer buses"
  because revert-then-apply set each target twice. On THIS path the picture differs: 93 M mostly-no-op
  restores (the O(network) floor) dominate, so restoring fewer elements should pay here. The coupling
  with L2 is real: most CHANGED generator targets come from the redistribution.
- **Change.** OLF: a protected hook for how `processContingency` saves/restores the network state
  (default: today's full `NetworkState`). GPU repo: a restore of the contingency's own elements +
  whatever L2 still changes + bus V/angle before a CPU-routed contingency.
- **Upstream value: real.** The CPU arm pays it too: 43.2 s of thread time at 8 threads, **4.9%**, and
  3.0 ms per contingency vs the GPU arm's 1.05 ms (the CPU load flow leaves more state changed). A cheaper
  full restore (e.g. `setExcludedSlackBuses` without the per-call stream) is upstreamable on its own.
- **Gain (estimate):** ~12–14 s of the 14.9 s.
- **Risk: highest of all leads.** A missed mutation silently corrupts every later contingency's starting
  point. Violation counts may not catch it (they are a threshold test).
- **Verify:** violation diff 0 + statuses on the 14,260 set, AND `OLF_GPU_SA_BUSDIFF=1` on slices (bus-level,
  1e-12 kV bar) including CPU-routed contingencies placed right after device-solved ones; OLF's own SA
  tests for the hook.
- **DONE (2026-09-22).** OLF (`olf-local-patches` and `upstream-candidates`): the protected `restoreBaseState` hook,
  `NetworkState.restore(buses, branches)` (listed elements + every HVDC, area, excluded slack bus and bus voltage),
  and the empty-exclusion fast path of `setExcludedSlackBuses`. GPU: after a contingency the device solved AND
  whose loss distribution the replay answered, restore the contingency's own buses (disabled, lost loads' /
  generators' / shifted shunts' buses, closed over voltage and reactive-power control groups) and branches; any
  other contingency gets the full restore. `-Dolf.gpu.sa.targetedRestore=false` for the A/B.
  - **Gate** (`OLF_GPU_RESTORE_CHECK=1`, a field-level fingerprint of everything NetworkState restores against the
    state a full restore gives): 14,256 targeted restores, 14,258 / 14,258 identical, cold and warm.
  - **The gate's first run found an upstream OLF bug in the FULL restore:** `LoadDcState.restore` handed the saved
    original-load disabling map itself to the load, so the next contingency wrote into the saved state — 3,828
    restores left lost loads flagged, accumulating; `LfLoadAction` then skips a strategy's action on such a load.
    Fixed with a regression test (OLF `529369aa8`, upstream-candidates).
  - **Measured:** restore 14.9 s -> 1.37 s (the remaining floor is every bus's voltage and angle). Full run: GPU
    25.5 s cold / 24.3 s warm vs CPU-8t 121.7 s (x4.78 / x5.01); statuses identical, 906,447 violations 0 differing,
    distributed active power within 8e-7 MW; suite 273/0.

### L2 — host-side lost-power redistribution, already done on the device — 4.6 s

- **Mechanism.** `DefaultContingencyActivePowerLossDistribution.run` water-fills each contingency's lost
  active power over the participating generators (`ActivePowerDistribution.run`: 5.0 s, mostly
  `getParticipatingElements` rescanning every bus per contingency). Branch outages have no loss and skip it.
- **Why it is redundant for device-solved contingencies:** `GpuBatchLossPreDistribution` replays the same
  water-fill host-side during precompute and feeds it to the device as target overrides; the device state
  that is then injected already embodies it.
- **Constraints (checked in code).** (a) The returned scalar `preDistributedActivePower` IS used: it goes
  into every `PostContingencyResult` — the precompute must supply it. (b) Generator target P IS read later
  by operator-strategy conditions: `ThresholdConditionEvaluator` on an `InjectionThresholdCondition` with
  `ACTIVE_POWER` — **skip only for contingencies without an operator strategy.** (First written as "and by
  `GpuStrategyComposer`'s generator-action deltas": WRONG — the composer reads the IIDM base target at
  precompute time and never sees the host distribution; a mutation test proved relative AND absolute actions
  insensitive to it.) (c) CPU-routed contingencies need the real distribution. (d) The benchmark case has no
  operator strategies, so it cannot exercise (b) — `GpuBatchedAcSecurityAnalysisTest#replayedLossDistribution
  KeepsResultsAndSparesStrategyContingencies` does, and fails on its RESULTS (the strategy stops firing) when
  the precondition is removed.
- **Change.** GPU repo only: `runSimulations` receives the distribution as an interface
  (`ContingencyActivePowerLossDistribution`); wrap it to return the precomputed scalar for device-solved,
  strategy-free contingencies and delegate otherwise. No OLF change.
- **Gain (estimate):** ~4.5 s, plus part of L1's `restoreGenerators` (targets no longer changed).
- **Risk: low–medium** (the scalar's exact value; the strategy precondition).
- **Verify:** as L1, plus a test with an operator strategy on a generator/load contingency.
- **DONE (2026-09-21) — measured outcome.**
  - Replay vs OLF's return value (`OLF_GPU_PRELOSS_CHECK=1`, no behaviour change): all 14,258 device-solved
    contingencies carry a replayed value, worst |difference| 6.2e-14 pu (6e-12 MW) — rounding, and on the
    samples OLF's own value is the one carrying it.
  - With the switch (`olf.gpu.sa.replayLossDistribution`, default true): statuses identical, 906,447
    violations with 0 differing, reported GPU `distributedActivePower` within 9.3e-12 MW of before (the
    run-to-run floor is ~2e-11 MW, see below). Host loss distribution in the GPU arm 4.6 s -> 25 ms.
  - **Wall gain smaller than the 4.6 s removed:** GPU cold 47.0 -> 45.8 s, warm 45.9 -> 43.6 s (CPU-8t
    123.1 s: x2.69 cold / x2.82 warm), because `NetworkState.restore` got ~2.2 s SLOWER (14.9 -> 17.1 s).
    Most likely memory traffic: the distribution's scan over every bus used to warm the caches for the
    restore's walk right after it. Part of L2's apparent cost was really the O(network) walk — which only
    goes away if nothing walks the network: **L1 matters more than the table above says.**
  - Reproducibility floors met on the way: the GPU arm's reported value moves by up to 1.6e-11 MW between
    two runs (the base load flow on the device sums with atomicAdd, moving the base's distributed targets by
    an ulp); the CPU arm's by up to 5.3e-9 MW between 30 and 8 threads (4 contingencies — per-partition
    network copies).

### L3 — state injection recomputes EVERY branch — 6.1 s (+0.7 s index update)

- **Mechanism.** `GpuStateInjector.injectState` -> `StateVector.setValuesOnly` -> `notifyStateUpdate` ->
  `AcNetworkVector.updateClosedBranches`: all closed-branch flow vectors recomputed per contingency.
  `EquationSystemIndex.update` (0.67 s) re-sorts the variables when `apply()` changed the active set
  (3,935 remap rebuilds in 14,258 — the "remap rebuilds (count)" line of PHASE_TIME is a count, not ms).
- **Constraint.** Branch results and violation detection read those flows: they must be right for every
  branch that is reported or monitored.
- **Change options.** Recompute only the branches the device violation screen flagged + monitored ones;
  or return flows from the device. GPU repo.
- **Gain (estimate):** ~4–5 s. **Risk: medium** (lazy/partial vectors vs everything that reads them).
- **DONE (2026-09-22).** OLF: `StateVector.setValuesOnly(array, closedBranchNums)` + `AcNetworkVector` refreshing
  only the listed branches (`olf-local-patches` 2cd493c79; on `upstream-candidates` after the values-only commit
  it builds on). GPU: the injection lists the branches the violation detection reads — those of the limits the
  device screen flagged for the contingency, plus the screen's uncovered branches; a branch/3WT state monitor, an
  operator strategy on the contingency, the first contingency (no screen yet) or a flow-reading diagnostic gets
  the full refresh. `-Dolf.gpu.sa.partialFlows=false` for the A/B.
  - **Gate** (`OLF_GPU_PARTIAL_FLOW_CHECK=1`: detect again after a full refresh of the SAME state): 14,254
    contingencies, violations bit-identical in all; 47.3 branches read per contingency instead of ~9,000.
  - **Measured:** post-contingency simulation 10.8 s -> 6.5 s (cold); GPU 21.6 s cold / 20.4 s warm vs CPU-8t
    131.9 s this run (121.7 s the previous one: x5.6–6.5); statuses identical, 906,447 violations 0 differing.

### L4 — per-contingency full-bus scans after injection — 2.9 s

- `AcSolverUtil.updateNetwork` 1.5 s (writes every bus V/angle; needed for results) and
  `isStateUnrealistic` 1.4 s (walks every variable with a bus lookup). Both could be one pass, or the
  realistic-voltage check could run on the device beside the violation screen.
- **Gain (estimate):** ~1.5–2 s. **Risk: low.** GPU repo.
- **DONE (2026-09-22, `5663190`):** one walk writes V/angle and checks the band (BUS_V/BUS_PHI only, other
  kinds fall back to the two calls). JFR, cold GPU run: 2.85 s -> 2.02 s — the walk itself is the floor.

### L5 — violation detection — 1.8 s

- `GpuViolationScreen.screenAll` 0.9 s (0.7 s of it `Arrays.sort`), `GpuLimitScanner` 0.85 s.
- **Gain:** < 1 s. Low priority.

### L6 — contingency qualification copies the main component per contingency — 2.55 s

- `GpuBatchContingencyValidator.qualifyOne` -> `EvenShiloachGraphDecrementalConnectivity.getLargestConnectedComponent`
  -> `computeMainConnectedComponent`: builds a `HashSet` of the whole main component every time (2.52 s of the
  2.55 s). The question asked is only "does this outage island anything".
- **Change.** Ask the connectivity for component count/size instead of materializing the set (GPU repo),
  or cache/avoid the copy in OLF's graph class (possible upstream).
- **Gain (estimate):** ~2.4 s. **Risk: low.**
- **DONE (2026-09-22, `15b0a67`):** `getComponentNumber(slack) != 0` (component 0 is the largest in both
  OLF implementations). JFR, cold GPU run: 2.55 s -> 0.09 s.

### L7 — override composition and HVDC `indexOf` — 1.8 s

- `GpuBatchOverrides.compose` 0.94 s (HashMap traffic); `GpuBatchHvdcAcEmulation.indexOf` 0.88 s rebuilds a
  `HashMap<Integer,Integer>` on every call (`applyStatusOverrides`, `applyFreeze`).
- **Gain:** ~1.5 s. **Risk: low.** GPU repo.
- **DONE (2026-09-22, `824f873`):** both indexes are trove `TIntIntHashMap`s (already on the classpath via
  OLF), bit-identical override lists. JFR, cold GPU run: 1.82 s -> 1.01 s; the boxed `ArrayList` storage behind
  the orchestrators' `List` API is what remains.

### L8 — the device solve itself — 7.8 s + 1.6 s JNI

- Throughput-bound: 42,870 scenario-iterations (~3 per contingency), ~0.15–0.17 ms each; waiting on each
  chunk's slowest scenario adds 7–11% (`widthIterSum` vs `activeIterSum`, compaction active).
- **Chunk size: CLOSED, no gain.** Sweep (`-Dolf.gpu.sa.maxBatch`): 951 -> 55.1 s cold, 1783 -> 50.6,
  3565 (auto, the default) -> 48.9, 7130 -> 49.1, 14260 -> 47.2 — flat within ±1.5 s above ~2000. The
  `AUTO_MAX_CHUNK = 4096` cap has no recorded rationale, is not needed for correctness (64-bit offsets,
  a 14,260 chunk allocated fine) and costs nothing measurable.
- Further device gains need fewer iterations or cheaper factorizations — not a lead at today's share.

### L9 — the host loop is single-threaded (known bug) — multiplies everything above

- GPU provider + threadCount > 1 -> concurrent `precomputeBatch` -> SIGSEGV in `cuLaunchKernel` (open bug of
  2026-07-17; `hs_err_pid2651455.log` of 2026-09-15 shows the same crash from `solveBatchSensitivities`).
- **Change.** Serialize the device part (precompute) and let OLF's per-contingency loop run on N threads.
- **Gain:** the 33 s host loop spread over N threads, whatever L1–L3 leave of it. **Risk: medium**
  (thread-safety of the shared device structure / context caches). GPU repo.

### L10 — RESULT DEFECT found while preparing L2: `getDistributedActivePower()` differs between the arms

- **Measured (before any L2 change):** 5,095 of 14,260 contingencies differ by > 1e-9 MW, worst 124.5 MW
  (`LINE-4678-3180`: CPU 124.54, GPU 0; `GEN-6353`: CPU 1,430.06, GPU 1,307.61 = 122.45 MW apart).
  Statuses and violations are identical, so no existing check saw it (`GpuVsCpuRte6515SaTest` now prints it;
  `OLF_GPU_SA_DAP_DUMP=<file>` dumps both values per contingency).
- **Mechanism (code):** the CPU reports `preDistributedActivePower + result.getDistributedActivePower()` — the
  contingency's pre-distributed loss plus what its own load flow's distributed slack spread
  (`AbstractSecurityAnalysis.java:727`); the GPU path reports `preDistributedActivePower` only
  (`GpuBatchedAcSecurityAnalysis.java:805`). The device's own DS pass does distribute; the amount is never
  brought back.
- **Change:** return the per-scenario amount the device DS pass distributed and add it. GPU repo.
- **Not a speed lead — a correctness one.** Independent of L2 (which must leave the GPU values unchanged).
- **DONE (2026-09-22).** Two defects, one cause (the device's own distributed-slack passes never reported):
  - *Post-contingency:* each device pass re-derives the whole distribution from the anchor, so after the last
    pass that fired the total is Σ genCum + the slack's retained share (the base write-back's own sum), or the
    slack's absorbed total when it is the only participant; no pass fired = the pre-distribution alone. Read
    per chunk and reported instead of the pre-distribution. The read-backs are reset at every solve, so a new
    native counter (`readBatchSolveCount`) keeps it to chunks ONE solve covered (the default driver mode);
    anything else (SVC pause/resume) reports the pre-distribution as before.
  - *Base case, found by the new unit test:* OLF's engine collects the amount only from outer loops that are
    `AcActivePowerDistributionOuterLoop`s; the device-driven wrapper was not one, so every GPU-driven SA base
    reported 0 (IEEE 14: CPU -60.29 MW, GPU 0.0). `DeviceDrivenDistributionLoop` now reports the device's
    amount (OLF's real loop's when a solve fell back to the CPU). A plain `LoadFlow.run` keeps OLF's loop and
    was never affected (`GpuLoadFlowTest` now asserts it).
  - rte6515, 14,260 contingencies: worst difference 124.5 MW -> 2.2e-6 MW; base 14.359916550 vs 14.359916552
    MW. `GpuVsCpuRte6515SaTest` asserts < 1e-3 MW; `distributedActivePowerMatchesCpu` (branch, generator, load,
    base) asserts < 1e-6 MW and fails when the post-contingency fix is reverted (verified).

### L11 — UPSTREAM OLF inconsistency found while testing L2: injection threshold conditions compare per-unit

- `ThresholdConditionEvaluator.evaluateGeneratorCondition` compares `gen.getTargetP()` / `getInitialTargetP()`
  — **per-unit** — with the condition's threshold as given, while branch and three-winding-transformer
  conditions convert their value to SI first (`* PerUnit.SB`, `* PerUnit.ib(...)`). A user's "generator above
  150 MW" is compared with ~1.5: it fires almost always.
- OLF's own test (`OpenSecurityAnalysisWithActionsTest`, lines 213–217) builds the threshold from
  `lfNetwork.getGeneratorById(...).getTargetP()` — per-unit — so it mirrors the implementation and cannot see it.
- **DONE (2026-09-22), OLF `e97bb4e57` on `olf-local-patches`, upstream candidate** (the patch applies cleanly
  to `origin/main`, whose evaluator is identical): target P / initial target P times `PerUnit.SB`; the unit test
  builds injection thresholds in MW like the branch/3WT ones and checks that half a generator's MW is below its
  value — which fails on the old evaluator (verified). OLF `sa` tests 311/0, checkstyle clean. The GPU project's
  L2 test now states its threshold in MW.

### L12 — EXPERIMENT: the device decides the violations (`olf.gpu.sa.deviceViolations`, off by default)

- **What:** the screen kernel gains a DECIDE mode — OLF's strict `>` against each threshold, the first exceeded
  limit per group in table order, emitted with the value it reports (signed P); buses compare the state's own V.
  The host only builds the violation objects (and applies the increased-violation filter, voltage-angle limits
  and the branches the screen does not cover); the injection refreshes no branch flow.
- **Accuracy** (`OLF_GPU_DEVICE_VIOLATIONS_CHECK=1`: the host's exact detection on the same state, every
  contingency): 14,255 / 14,255 identical violation SETS (906,434 violations); values never bit-identical, worst
  relative difference **2.1e-10** (currents: nearly cancelling terms amplify last-bit differences — not ~1e-16).
  Against the CPU arm: statuses identical, 906,447 violations both sides, 0 differing.
- **Speed:** default vs device, two runs each: cold 24.2 / 26.7 vs 23.1 / 23.4 s, warm 21.1 / 21.9 vs 19.0 / 20.2 s
  — ~1–2 s, within run-to-run noise (two identical default runs differ by 2.5 s). What detection still costs is
  BUILDING ~906 k violation objects, not the confirm the device replaced.
- **To make it pay:** decide inside the batch solve context (the screen re-uploads every state, ~1.5 GB per run,
  that was on the device a moment before) and make building the violation objects cheaper.

## 5b. Device-side leads (profile of 2026-09-22, Nsight Systems, per run)

The batched solve is ~10.8 s of ~21 s, but the linear algebra proper (cuDSS factorization + forward/backward
solves) is ~3.7 s of it.

- **D1 — compaction re-analyses (~1.2 s).** The main context is analyzed once per chunk size; the straggler
  compaction path has its own cuDSS context and destroys + re-analyzes it at every width tier change, uncached —
  ~35 analyses per run instead of 2 (73 `define_superpanel_ker` launches over two runs). A data object cannot
  change batch size (illegal memory access, 2026-09-16), but one analyzed data object PER WIDTH can be cached and
  reused across chunks and runs: ~9 analyses per run at most, ~0 warm.
  **DONE (2026-09-22).** Counted directly (`OLF_GPU_PHASE_TIME` now prints cuDSS analyses by path): per run 3 main
  (base load flow, chunk-size probe, chunk context) + 34 compaction — 73 over cold + warm. With one analyzed data
  object per compact width (`OLF_GPU_COMPACT_CACHE=0` restores the old path): 12 compaction analyses per run (the
  widths 1..1024 and the cap), JNI solve −1.0 s per run in four A/B runs; statuses, 906,447 violations 0 differing.
  The warm run still pays its 12 + 2: the batch context is rebuilt every run. **Not pursued:** the COLD run is
  the primary case (decision 2026-09-22), so cross-run context reuse is warm-only gain.
- **D2 — `fillClosedBranchBatch` (~1.06 s, 410 launches, ~2.6 ms each):** heavy for a fill; atomicAdd scatter is
  the prime suspect. **PARKED (2026-09-22):** already tried — a fused fill was slower, "the fill's floor is its
  atomics" (`codegen/gpu_benchmark.md` ~L135-182).
- **D3 — host round trips:** device-to-host `cudaMemcpyAsync` 3,942 calls over two runs, 2.93 s of API time of which
  2.91 s waits for the queued kernels (52 MB copied in 0.007 s). **Corrected:** these are NOT our per-iteration
  syncs — attributed by the kernel preceding each copy, they are cuDSS ANALYSIS internals (34 scalar readbacks per
  analysis during the forward/backward ordering, 3 reorder-sized ones, …; our solve loop has 5 explicit
  device-to-host copies, all in rarely-used host-paused controls). They shrink with every analysis D1 avoids.
  Pinned memory would not help (bulk copies are only ~0.3 s of DMA per run).
- **D4 — `cudaMalloc`/`cudaFree` (~0.35 s):** a memory pool / cuDSS device-memory handler instead of per-compaction
  allocations.
- **D5 — `writeIdentityForInactive` (~0.31 s)** for retired scenarios.

## 5e. Against CPU OLF at 16 and 30 threads (2026-09-22, after the bus-array update; CPU cold AND warm)

| CPU threads | CPU cold | CPU warm | GPU cold | GPU warm | GPU vs CPU warm |
|---|---|---|---|---|---|
| 16 | 72.0 s | 69.8 s | 18.9 s | 17.9 s | x3.68 cold / x3.90 warm |
| 30 (all cores) | 60.1 s | 53.7 s | 19.0 s | 18.1 s | x2.83 cold / x2.96 warm |

GPU arm on ONE host thread. Both runs: statuses identical, 906,447 violations, 0 differing; distributed active
power within 9e-7 / 2.6e-6 MW. At the start of this assessment the GPU arm was ~47–51 s (x1.4 vs 16 threads,
x1.2–1.3 vs 30).

## 5d. Cold-run profile after L1–L4, L6, L7, L10, D1 (2026-09-22; the cold run is the primary case)

`OLF_GPU_SA_COLD_ONLY=1` (GPU arm cold only), JFR + Nsight Systems; cold GPU arm 23.3 s (JFR) / 24.0 s (Nsight).

| Part | Time | Detail |
|---|---|---|
| Setup before the SA loop | ~1.2 s | network loading |
| Base load flow | 0.9 s | |
| Precompute | 12.7 s | JNI solve 9.6 s (GPU kernels 6.2 s, marshal 0.9, pack 0.6, syncs/gaps ~1.6); override lists 0.6; planning, qualification and the rest ~2.5 |
| Per-contingency loop | 8.4 s | detection 2.3 (host `Arrays.sort` of the screen's candidates 1.05, confirm 0.9); injection 1.9 (OLF variable-index re-sort 1.0); bus V/angle update + realism check 1.9; restore 1.2; the 2 CPU-routed contingencies 1.0 |

GPU kernels (6.2 s): cuDSS factorization 2.6, forward/backward solve 1.1, analysis 0.61 (1.2 before D1), Jacobian
fill 1.1 (D2, parked), identity rows 0.3, outer-loop passes 0.3, violation screen 0.02, other 0.3.

Next cold-path candidates: the candidate sort (~1 s: sort on the device, or do without the order); bus V/angle
from arrays instead of OLF's variable objects (up to ~1.5 s — **DONE**: per-bus V/angle rows of the device state,
disabled buses skipped; `OLF_GPU_BUS_ARRAY_CHECK=1` against the variable walk: 14,258 / 14,258 bit-identical;
1.88 s -> 0.80 s cold; GPU 20.1 s cold vs CPU-8t 134.8 s this run, 906,447 violations 0 differing); OLF's variable-index re-sort after `apply()` (~1 s);
the 2 CPU-routed contingencies (~1 s for 2); JNI marshal/pack (1.5 s) and the boxed override lists (0.6 s).

## 5c. How gpusim2grid handles the same parts (`github.com/Grid2op/gpusim2grid` @ 957ad32, read 2026-09-22)

File:line references are under `gpusim2grid/src/_cpp`.
- **Analysis:** uniform batch (UBATCH) like ours, ANALYSIS once per `run()` (`batch_pf_driver.cu:279-290`), one
  fixed batch size per run with the last chunk padded (`contingency_batch.cuh:152-153`, `driver.cuh:58-59`); NOT
  reused across runs (`contingency_analysis_session.cu:186-187` rebuilds the driver). Base case: its own context,
  once per session.
- **Stragglers:** none by construction — a FIXED iteration count (`nb_iter`, default 4), no convergence check, no
  masking, no compaction (`driver.cuh:60,85`); convergence judged after the loop. Their "one analysis" comes from
  there, not from a technique we lack: iterating to convergence with outer loops is what makes compaction (and
  its batch sizes) necessary for us.
- **Factorization:** default FACTORIZATION once then REFACTORIZATION + SOLVE (`policy_refactor_every.cuh:40-58`);
  also chord (iter 0 only) and frozen base-case factors.
- **Round trips:** an always-on `CudaTimer` synchronizes ~5 times per iteration (`cuda_utils.h:650-655`); no pinned
  memory, CUDA graphs, memory pools or cuDSS memory handler anywhere.
- **Jacobian fill: no atomics** — one thread per (slot, Ybus nonzero), precomputed nonzero -> J-slot maps, plain
  stores (`acpf_nr_kernels.cu:408-500`). The idea for D2.
- **Violations:** fully on the GPU, returned as arrays (type, element, side, value, limit, count, truncated;
  `violation_kernels.cuh:10-60`); no exactness check against a reference — our L12 experiment, minus the Java
  violation objects we must build.

## 5. Interactions and projection

- **L1 x L2 coupled:** skipping the redistribution removes most CHANGED generator targets, the expensive
  part of a partial restore (OLF `2223c8e56`). Do L2 first, then measure L1 again before designing it.
- **L9 multiplies:** with the host loop parallelized, L1–L7's remaining cost is divided by the thread count.
- **Projection (estimate, 1 host thread):** L1 + L2 + L3 + L6 + L7 ≈ −25 s -> GPU arm ~22–25 s, i.e.
  x4.7–5.4 vs CPU-8t warm (118.6 s) and x2.7–3.1 vs CPU-16t (68.5 s). To be replaced by measurements lead by lead.

## 6. Proposed order (one commit per step, measured after each)

1. **L2** (GPU repo) — lowest risk, no OLF change, and it changes L1's cost profile. **DONE**: see L2.
2. **L6, L7, L4** (GPU repo) — small, independent, low risk. **DONE**: −4.1 s of the cold GPU run by JFR
   (precompute 16.3 -> 13.9 s). Full run after L2 + L10 + L6/L7/L4: GPU 38.7 s cold / 39.3 s warm vs CPU-8t
   125.9 s (x3.25; wall times move by several seconds between identical runs — read the per-method numbers).
3. **L1** — OLF hook (upstreamable patch) + GPU-side partial restore; the most verification. **DONE**: see L1.
4. **L3** (GPU repo). **DONE**: see L3.
5. **L9** (GPU repo) — last: it multiplies whatever is left, and profits from a smaller host loop.

Verification protocol for every step: the 14,260 run with 0 differing violations and identical statuses;
bus-level diff on slices for steps touching state (L1, L2, L3); CPU arm time unchanged for OLF patches
(and OLF's own SA tests); GPU repo suite green (`mvn test -Dcheckstyle.skip`).

## 7. Open questions

- The ~0.84 s of CPU fallback for the 2 device-rejected contingencies (status −2): which rejection, and is
  it the known pair?
- Does the precompute already hold the per-contingency `preDistributedActivePower` scalar, or must it be
  derived from `GpuBatchLossPreDistribution`'s remaining mismatch?
- CPU arm: 3.0 ms per restore vs 1.05 ms on the GPU arm — confirm it is the larger changed set (CPU load
  flow outer loops) before quoting it upstream.
