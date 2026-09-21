# GPU security analysis — where the time goes, and which leads are worth pursuing

Assessment of 2026-09-21, written BEFORE any modification, to trace the leads and their evidence.
Nothing below has been implemented yet; every "gain" is an estimate unless it says "measured".

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

### L2 — host-side lost-power redistribution, already done on the device — 4.6 s

- **Mechanism.** `DefaultContingencyActivePowerLossDistribution.run` water-fills each contingency's lost
  active power over the participating generators (`ActivePowerDistribution.run`: 5.0 s, mostly
  `getParticipatingElements` rescanning every bus per contingency). Branch outages have no loss and skip it.
- **Why it is redundant for device-solved contingencies:** `GpuBatchLossPreDistribution` replays the same
  water-fill host-side during precompute and feeds it to the device as target overrides; the device state
  that is then injected already embodies it.
- **Constraints (checked in code).** (a) The returned scalar `preDistributedActivePower` IS used: it goes
  into every `PostContingencyResult` — the precompute must supply it. (b) Generator target P IS read later
  by operator-strategy conditions (`ThresholdConditionEvaluator`) and by the GPU strategy composer's
  generator-action deltas (`GpuStrategyComposer`): **skip only for contingencies without an operator
  strategy.** (c) CPU-routed contingencies need the real distribution. (d) The benchmark case has no
  operator strategies, so it cannot exercise (b) — a dedicated test must.
- **Change.** GPU repo only: `runSimulations` receives the distribution as an interface
  (`ContingencyActivePowerLossDistribution`); wrap it to return the precomputed scalar for device-solved,
  strategy-free contingencies and delegate otherwise. No OLF change.
- **Gain (estimate):** ~4.5 s, plus part of L1's `restoreGenerators` (targets no longer changed).
- **Risk: low–medium** (the scalar's exact value; the strategy precondition).
- **Verify:** as L1, plus a test with an operator strategy on a generator/load contingency.

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

### L4 — per-contingency full-bus scans after injection — 2.9 s

- `AcSolverUtil.updateNetwork` 1.5 s (writes every bus V/angle; needed for results) and
  `isStateUnrealistic` 1.4 s (walks every variable with a bus lookup). Both could be one pass, or the
  realistic-voltage check could run on the device beside the violation screen.
- **Gain (estimate):** ~1.5–2 s. **Risk: low.** GPU repo.

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

### L7 — override composition and HVDC `indexOf` — 1.8 s

- `GpuBatchOverrides.compose` 0.94 s (HashMap traffic); `GpuBatchHvdcAcEmulation.indexOf` 0.88 s rebuilds a
  `HashMap<Integer,Integer>` on every call (`applyStatusOverrides`, `applyFreeze`).
- **Gain:** ~1.5 s. **Risk: low.** GPU repo.

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

## 5. Interactions and projection

- **L1 x L2 coupled:** skipping the redistribution removes most CHANGED generator targets, the expensive
  part of a partial restore (OLF `2223c8e56`). Do L2 first, then measure L1 again before designing it.
- **L9 multiplies:** with the host loop parallelized, L1–L7's remaining cost is divided by the thread count.
- **Projection (estimate, 1 host thread):** L1 + L2 + L3 + L6 + L7 ≈ −25 s -> GPU arm ~22–25 s, i.e.
  x4.7–5.4 vs CPU-8t warm (118.6 s) and x2.7–3.1 vs CPU-16t (68.5 s). To be replaced by measurements lead by lead.

## 6. Proposed order (one commit per step, measured after each)

1. **L2** (GPU repo) — lowest risk, no OLF change, and it changes L1's cost profile.
2. **L6, L7, L4** (GPU repo) — small, independent, low risk.
3. **L1** — OLF hook (upstreamable patch) + GPU-side partial restore; the most verification.
4. **L3** (GPU repo).
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
