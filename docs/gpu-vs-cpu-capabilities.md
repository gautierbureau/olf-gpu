# GPU vs CPU capability tracking — path to full CPU replacement

Living document. One row per modelling / solver capability, tracked across the four run
types: plain load flow, security analysis, operator strategies, remedial actions. The goal
is an honest inventory of what the GPU extension covers today vs stock CPU OLF, so the
remaining gap to "replace all CPU runs with GPU" is a checklist, not a guess.

Last full review: 2026-07-18 (commits `cc41597`/`a082092`).

## Legend

| Mark | Meaning |
|------|---------|
| ✅ | Supported on the GPU path and validated by a test against CPU |
| 🔶 | Supported with caveats (scope conditions listed in the row) |
| ⏩ | Not on the GPU path — **detected and routed to CPU** (safe: results identical to stock OLF, only slower) |
| ❌ | Rejected loudly (PowsyblException) — run fails rather than silently degrading |
| — | Not applicable |

The provider's safety contract everywhere: **enabling the GPU must never change results.**
Anything unsupported either falls back (⏩) or fails loudly (❌); nothing is silently dropped.

## Architecture: the two GPU execution modes

The support envelope differs fundamentally between them — this is the single most important
distinction in this document:

1. **Sequential inner-solver swap** (`GpuNewtonRaphson`, plain LF and the SA base solve):
   only the inner Newton iteration runs on the device. **All of OLF's outer loops keep
   running unchanged on the CPU** between inner solves — the GPU context refreshes values
   (tap/shunt moves → `refreshContextValues`, analysis reused) or activation (PV↔PQ → row
   mode/target remap) between outer iterations. Outer-loop coverage here is therefore
   *by construction complete*; the envelope is set by what the **extractor can pack**
   (see "Plain LF" below).
2. **Batched device path** (`GpuBatchedSecurityAnalysisSolver`, SA contingencies +
   strategy pairs): thousands of scenarios iterate on the device without CPU round-trips,
   so **every outer loop must be re-implemented device-side per scenario**. Outer-loop
   coverage here is an explicit, short list; anything else triggers the detect-and-fallback
   guard (whole run → stock CPU sequential path).

## 1. Plain load flow (sequential GPU inner solver)

Engagement: `GpuNewtonRaphsonFactory` as solver factory (the SA provider also swaps it in
for the base solve when the batch qualifies). Unsupported networks degrade per the
`olf.gpu.cpuFallback` property (default: fall back to CPU NR inside the solver; the
validation project sets it false to fail loudly instead).

### Network / equation-system packing (the extractor envelope)

| Capability | GPU | Notes |
|---|---|---|
| Buses PV/PQ, slack | ✅ | activation refresh handles PV↔PQ switching mid-run |
| Lines, 2W transformers (fixed taps) | ✅ | closed-branch pack (y g1 g2 b1 b2 ksi r1 a1) |
| Open / partially-connected branches | ✅ | open-branch pack; disconnection between solves is a contingency, not a reject |
| Transformer ratio control (local, own terminal) | 🔶 | r1 as state variable; remote / multi-transformer DISTR_RHO shapes ❌ |
| Transformer phase control | ❌ | unsupported variable type — extractor throws |
| Shunts (fixed + controlled b) | 🔶 | controlled-shunt DISTR_Q contribution shapes covered by `buildDistrQContributions`; shunt/open-branch/remote contributor shapes ❌ |
| Zero-impedance branches | 🔶 | only shapes with DUMMY_P/DUMMY_Q variables; other non-impedant shapes ❌ |
| ZIP / exponential load models | ❌ | unsupported equation terms |
| Static VAR compensators (slope) | ❌ | unsupported equation terms |
| HVDC (AC emulation) | ❌ | unsupported equation terms (untested beyond the guard) |
| Remote voltage control (DISTR_Q) | 🔶 | sequential LF: covered where `buildDistrQContributions` can wire it; batch: rejected per-contingency |
| Vectorized equation-array systems | ❌ | array term types rejected explicitly |

### Outer loops in plain LF

All run on CPU around the GPU inner solve — supported **by construction**, including:
distributed slack, reactive limits, phase control (incl. incremental), transformer voltage
control (all three modes), transformer reactive-power control, shunt voltage control
(incl. incremental), secondary voltage control, voltage monitoring, area interchange
control, HVDC AC-emulation loops, automation systems. The constraint is only that the
*network* packs (table above). The CME-under-outer-loop-switching bug on this path was
fixed 2026-07-17 (dead-flag + synchronous listener purge).

### Solver / convergence in plain LF

| Capability | GPU | Notes |
|---|---|---|
| Newton-Raphson | ✅ | the only ported solver |
| Fast decoupled / DC | ⏩ | explicit non-default solver choice is respected → CPU |
| Stopping: `DefaultNewtonRaphsonStoppingCriteria` (‖F‖₂ < √(ε²·n)) | ✅ | `convEpsPerEq` wired through — byte-comparable criterion vs CPU |
| Stopping: uniform ‖F‖∞ < tol | ✅ | GPU-native mode (bench/fast paths) |
| Stopping: `PerEquationTypeStoppingCriteria` | ⏩ | not ported |
| State scaling: NONE | ✅ | lsMode 0 |
| State scaling: LINE_SEARCH | ✅ | lsMode 1 (device line search) |
| State scaling: MAX_VOLTAGE_CHANGE | ✅ | lsMode 2 (per-type Δv/Δφ clamps, device) |
| maxIterations, per-run report | ✅ | |
| Linear solver | ✅ | cuDSS LU (KLU-equivalent role); no per-solver ordering choice exposed |

## 2. Security analysis (batched device path)

Engagement: `GpuSecurityAnalysisProvider`. The guard resolves the run's outer-loop list
up front; if anything outside the batchable set is active, the **whole run** executes on
the stock CPU path (base + contingencies + strategies), logged.

### Outer loops in batched SA

| Outer loop (CPU OLF name) | Batched | Notes |
|---|---|---|
| DistributedSlack | ✅ | per-scenario participation reweight on gen loss; caveat: contingency losing a gen at a DS bus when the SLACK ITSELF participates → that contingency ⏩ |
| ReactiveLimits | ✅ | device PV↔PQ pinning; semantic note: device pins AT the limit, CPU freezes at switch-time evaluated Q — states match ≤1e-4 pu except through GeneratorAction perturbations (see strategies) |
| TransformerVoltageControl | 🔶 | AFTER_GENERATOR mode only, and only when no generator voltage control sits under the transformer-controlled level (`transformerControlSupported`); orchestrated Java-side over warm restarts |
| AcHvdcAcEmulationLimits, FreezingHvdcACEmulation | 🔶 | tolerated as structural no-ops when the network has **no HVDC**; otherwise ⏩ |
| VoltageMonitoring | 🔶 | tolerated when the network has **no SVC**; otherwise ⏩ |
| PhaseControl / IncrementalPhaseControl | ⏩ | |
| IncrementalTransformerVoltageControl / Simple | ⏩ | |
| TransformerReactivePowerControl (incremental) | ⏩ | |
| ShuntVoltageControl (incl. incremental) | ⏩ | |
| SecondaryVoltageControl | ⏩ | |
| AreaInterchangeControl | ⏩ | |
| AutomationSystem | ⏩ | |

### Contingency types

Per-contingency granularity: a non-qualifying contingency routes to the sequential
CPU fallback individually (option `olf.gpu.sa.gpuSequentialLoadFlow` keeps it on the GPU
sequential solver instead, default false); the rest of the batch stays on the device.

| Contingency element | Batched | Notes |
|---|---|---|
| Branch outage (N-1, N-2) | ✅ | ≤2 branches, both-sides opening; per-scenario element mask |
| Fragmenting branch outage (islanding) | ✅ | islanded buses frozen to clean flat state via row-mode + target override |
| Generator outage (single / multiple) | ✅ | per-BUS merged injection override; PV→PQ flip, Q-target shed, DS reweight; ❌→⏩ slack-bus loss, >1 DS-participant bus, remote-controlling gens |
| Load outage | ✅ | PowerShift injection deltas (Q-carrying load delta at a PV bus → ⏩) |
| Shunt shift / reconnection | ✅ | per-scenario admittance override; DISTR_Q-contributing or unpacked shunt → ⏩ |
| Fixed-shunt disconnection | ✅ | element mask |
| Branch+generator+load+shunt combos | ✅ | any mix of the above (branch part ≤2; gen on islanded bus → ⏩) |
| ≥3 branch outage | ⏩ | superset-pattern cost — extendable if needed |
| Single-side branch opening | ⏩ | masks model both-sides only |
| Bus / busbar-section loss | ⏩ | |
| HVDC line contingency | ⏩ | |
| Outage of a DISTR_RHO controller branch | ⏩ | |

### SA machinery parity

| Capability | GPU | Notes |
|---|---|---|
| Base LF on GPU (absorbs CUDA init) | ✅ | `olf.gpu.sa.gpuBaseLoadFlow`, default on; FAILED base → no post-contingency results, same as stock |
| Limit-violation detection / IncreasedViolations filtering | ✅ | shared CPU code — identical by construction (see `docs/increased-violations-thresholds.md` for the zero-default flicker trap) |
| State monitors (bus results) | ✅ | shared CPU code on the restored per-scenario states |
| threadCount > 1 | 🔶 | GPU batch is mono-device **by design**: device work serialized under a global lock, no solver swap; CPU-fallback contingencies parallelize normally |
| Production-parameter equivalence | ✅ | case9241pegase × 500 ctg × RTE JSON params: statuses identical, violation diff = 22 threshold-edge re-reports in 6 ctg (states ≤2.3e-4 pu apart), drilled one-by-one 2026-07-17 |

## 3. Operator strategies & remedial actions

Architecture: a (contingency, strategy) pair is an EXTRA batch scenario — actions composed
onto the contingency's deltas, solved as a **warm continuation** of the parent contingency
(two-phase: parent chunks export their reactive-limits pinning state; pair chunks seed the
parent's solution + RL state, reconciled device-side). A pair with any non-batchable
element falls back to the stock per-strategy CPU path individually.

### Conditions

| Condition | Batched | Notes |
|---|---|---|
| TrueCondition (single-stage) | ✅ | contexts ALL / ONLY_CONTINGENCIES / SPECIFIC |
| AnyViolationCondition / AtLeastOneViolationCondition / AllViolationCondition | ⏩ | needs the engine-loop collection → deferred warm batch (plumbing exists — next in #24) |
| Multi-stage conditional actions | ⏩ | |

### Action types (powsybl action-api 7.3.0, complete list)

| Action | Batched | Notes |
|---|---|---|
| TerminalsConnectionAction (open) | ✅ | adds the branch to the pair's outage mask |
| TerminalsConnectionAction (close) | 🔶 | only re-closing a branch opened by the contingency (line reconnection); closing a base-open branch ⏩ (extractor packs disconnected elements — shared deferral with shunt reconnection of base-open) |
| SwitchAction | 🔶 | same open/close semantics as above, switch-mapped |
| LoadAction | ✅ | absolute / relative P shed as injection delta |
| RatioTapChangerTapPositionAction | ✅ | full static-row override [y g1 g2 b1 b2 ksi r1 a1] from the target tap's pi model — per-tap impedance changes covered; controlled transformer (r1/a1 state var) → ⏩ |
| PhaseTapChangerTapPositionAction | ✅ | same compiler as RTC (validated incl. ±100% per-step r/x/g/b swings); PST under active phase control → ⏩ |
| ShuntCompensatorPositionAction | ✅ | linear model only (Δsections × per-section admittance); non-linear model → ⏩ |
| GeneratorAction | ⏩ | root cause understood, not modelling: CPU freezes a switched bus's Q at its SWITCH-TIME value, device pins AT the limit; redispatch perturbs a pinned controller (~0.17 kV). Needs device freeze-at-evaluated-q or pinned-target export/import |
| PercentChangeLoadAction | ⏩ | trivially compilable to LoadLoss — not yet wired |
| RatioTapChangerRegulationAction / PhaseTapChangerRegulationAction | ⏩ | changes control MODE, not position |
| HvdcAction | ⏩ | HVDC not packed |
| StaticVarCompensatorAction | ⏩ | SVC not packed |
| AreaInterchangeTargetAction | ⏩ | AIC outer loop not batched |
| BoundaryLineAction (dangling line) | ⏩ | |
| MultipleActionsAction | ⏩ | composite — could compile recursively over supported members |

## 4. Solver capabilities (batched path specifics)

| Capability | Status | Notes |
|---|---|---|
| cuDSS UBATCH (mode 3) shared-pattern LU | ✅ default | modes 0/1/2 compiled in, runtime-selected; chunk memory auto-sized (fallback 256) |
| Per-scenario row-mode masks + disabled-element masks | ✅ | fixed superset CSR pattern |
| Per-scenario target / injection overrides | ✅ | flat index/value lists |
| Per-scenario admittance (shunt) + full-row (tap) overrides | ✅ | |
| Hot start from base + per-scenario x0 (warm continuation) | ✅ | |
| RL pinning state export/import between solves | ✅ | packed limitType + switch counts; device reconcile kernel |
| Device-side convergence norms | ✅ | ‖F‖∞ / ‖F‖₂ / finite per scenario (12 KB vs 70 MB readback) |
| Straggler early-exit | ✅ | ≤2% actives past half the budget → status −2 → CPU fallback (`OLF_GPU_STRAGGLER_FRAC`) |
| Solve-only batch compaction | ✅ | gather actives → power-of-2 narrow ubatch on a dedicated context |
| Per-scenario step scaling (MAX_VOLTAGE_CHANGE) | ✅ | lsMode 2 batched |
| Reference numbers (RTX A1000 6 GB) | — | production RTE params: cold 12.9–13.3 s vs CPU-4t ≈12.5–13.5 s, warm 11.5–12.1 s; fixed-iter K=4: cold E2E 11.7 ms/ctg, warm 4.2–4.4 ms/scn |

## 5. The gap list to full CPU replacement (ranked)

What actually stands between "GPU covers our production runs" and "GPU covers everything",
in rough order of blocking impact for RTE-style networks:

1. **Violation-conditioned operator strategies** — the dominant real-world strategy shape;
   engine-loop collection + deferred warm batch, plumbing already in place (#24).
2. **GeneratorAction** — RL freeze-at-evaluated-q semantics on the device (#24; root cause
   isolated, not a modelling gap).
3. **Remaining SA outer loops** — shunt voltage control and incremental transformer /
   phase controls are the ones present in RTE parameter sets; secondary voltage control
   and AIC behind them. Each is a device port (the CPU plain-LF path already has them all).
4. **HVDC + SVC packing** — extractor equation-term work; unblocks both plain-LF coverage
   on those networks and their SA/action rows.
5. **Base-open element closes** (branch reconnection / shunt reconnection of base-open
   elements) — extractor packs disconnected elements into the superset pattern.
6. **Bus/busbar contingencies, ≥3-branch outages, single-side openings** — mask/topology
   generalizations, needed for exhaustive contingency lists.
7. **PerEquationTypeStoppingCriteria** — only if a production parameter set uses it.
8. **ZIP/exponential loads, zero-impedance shapes without dummies, vectorized systems** —
   long-tail extractor envelope.

## Maintenance

Update this file whenever a row changes state (the PR that lands a capability flips its
mark), and re-run the two benches plus `GpuBatchedAcSecurityAnalysisTest` before flipping
anything to ✅. Numbers and dated evidence live in `codegen/gpu_benchmark.md`; this file
holds only the current truth.
