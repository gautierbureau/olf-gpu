# GPU vs CPU capability tracking — path to full CPU replacement

Living document. Part I is the overview matrix (one row per capability, four run types:
plain LF, SA, operator strategies, remedial actions). Part II goes to corner-case
granularity: per outer loop, every semantic axis of the CPU implementation (modes,
deadbands, back-switching, saturation, failure behaviors, shared controls) against what
the GPU ports actually do; plus the exhaustive extractor envelope (every accept/reject
branch, DISTR_Q contributor shapes, DISTR_RHO shapes) and the full batch disqualification
predicate list. The goal is that "replace all CPU runs with GPU" is a checklist, not a
guess — including the corner cases.

Last full review: 2026-07-18 (commits `cc41597`/`a082092`; Part II from a line-level
mine of `GpuAcDataExtractor`, `GpuBatchContingencyValidator`, `olf_gpu_nr_jni.cu`,
`GpuBatchedSecurityAnalysisSolver`, and the stock OLF `ac/outerloop` sources).

## Legend

| Mark | Meaning |
|------|---------|
| ✅ | Supported on the GPU path and validated by a test against CPU |
| 🔶 | Supported with caveats (scope conditions listed in the row) |
| ⏩ | Not on the GPU path — **detected and routed to CPU** (safe: results identical to stock OLF, only slower) |
| ❌ | Rejected loudly (PowsyblException) — run fails rather than silently degrading |
| ⚠️ | Semantic delta vs CPU inside supported scope — behavior differs in a corner case |
| — | Not applicable |

The provider's safety contract everywhere: **enabling the GPU must never change results.**
Anything unsupported either falls back (⏩) or fails loudly (❌); nothing is silently dropped.

## Architecture: the two GPU execution modes

The support envelope differs fundamentally between them — the single most important
distinction in this document:

1. **Sequential inner-solver swap** (`GpuNewtonRaphson`, plain LF and the SA base solve):
   only the inner Newton iteration runs on the device. **All of OLF's outer loops keep
   running unchanged on the CPU** between inner solves — the GPU context refreshes values
   (tap/shunt moves → `refreshContextValues`, analysis reused) or activation (PV↔PQ → row
   mode/target remap) between outer iterations. Outer-loop coverage here is therefore
   *by construction complete*; the envelope is set by what the **extractor can pack**
   (Part II §E).
2. **Batched device path** (`GpuBatchedSecurityAnalysisSolver`, SA contingencies +
   strategy pairs): thousands of scenarios iterate on the device without CPU round-trips,
   so **every outer loop must be re-implemented device-side per scenario**. Outer-loop
   coverage here is an explicit, short list; anything else triggers the detect-and-fallback
   guard (whole run → stock CPU sequential path).

---

# Part I — Overview matrix

## 1. Plain load flow (sequential GPU inner solver)

Engagement: `GpuNewtonRaphsonFactory` as solver factory (the SA provider also swaps it in
for the base solve when the batch qualifies). Unsupported networks degrade per the
`olf.gpu.cpuFallback` property (default: fall back to CPU NR inside the solver; the
validation project sets it false to fail loudly instead).

### Network / equation-system packing (summary — exhaustive version in Part II §E)

| Capability | GPU | Notes |
|---|---|---|
| Buses PV/PQ, slack | ✅ | activation refresh handles PV↔PQ switching mid-run |
| Lines, 2W transformers (fixed taps) | ✅ | closed-branch pack (y g1 g2 b1 b2 ksi r1 a1) |
| Open / partially-connected branches | ✅ | open-branch pack; disconnection between solves is a contingency, not a reject |
| Transformer ratio control (local, own terminal) | ✅ | r1 as state variable, qRow moved onto the rho column ("item 4" derivation) |
| Transformer ratio control (remote pilot / multi-transformer groups) | 🔶 | DISTR_RHO linear coupling derived incl. free-controller mechanism; remote controller zero-diagonal left to cuDSS pivoting; see Part II §E.4 |
| Remote generator voltage control (DISTR_Q), incl. **multiple controllers on one pilot** | 🔶 | fully wired in sequential LF (weights, free controller, shunt / open-branch / zero-impedance contributors); voltage-controlling-shunt contributor ❌; batch path rejects all DISTR_Q. Part II §E.3 |
| Transformer phase control (BRANCH_ALPHA1 as variable) | 🔶 | ✅ **2026-07-20** for the INCREMENTAL mode row: a1 is packed as a state variable and `BRANCH_TARGET_ALPHA1` (the a1−target identity that mode keeps active all run) is accepted, validated GPU-vs-CPU at ~1e-14 with the fallback counter asserted (`GpuPhaseControlTest`). Note `isDeriveA1` keys on `isPhaseController()` alone, so BOTH modes create the column. ⏩ CONTINUOUS mode with control ENABLED: that activates `BRANCH_TARGET_P` on the CONTROLLED branch — a branch-owned power row with no GPU family — so the solver falls back for those solves and resumes on GPU once the loop switches control off at iteration 0. GATE: all of it hangs off `phaseShifterRegulationOn` (core default **false**, not set by the RTE params file) |
| Controllable shunt (SHUNT_B as variable) | ❌ | unsupported variable type |
| Fixed shunts | ✅ | per-bus G,B pack |
| Zero-impedance branches | 🔶 | only shapes with DUMMY_P/DUMMY_Q; one-side-open ZI ❌; both-ends-controlled / 3-winding star ❌. Part II §E.5 |
| ZIP / exponential load models | ❌ | terms outside the 12 supported classes |
| SVC (slope, standby automaton) | ❌ | SHUNT_B / non-supported terms |
| HVDC AC emulation | 🔶 | packed (droop, P0, losses) when both buses present and AC-emulation mode; otherwise represented only through terminal P injections |
| Batteries / LCC / VSC / dangling lines / tie lines / areas | 🔶 | no first-class handling: fine while they only contribute plain P/Q bus targets; anything introducing an unsupported term/variable throws. Part II §E.7 |
| Vectorized equation-array systems | 🔶 | only BUS_TARGET_P/Q arrays with the 4 closed-branch array evaluators; other array types/evaluators ❌ |

### Outer loops in plain LF

All run on CPU around the GPU inner solve — supported **by construction**, including every
corner case of Part II §A–§D (deadbands, back-switching, shared controls, failure
behaviors). The constraint is only that the *network* packs. The
CME-under-outer-loop-switching bug on this path was fixed 2026-07-17.

### Solver / convergence in plain LF

| Capability | GPU | Notes |
|---|---|---|
| Newton-Raphson | ✅ | the only ported solver |
| Fast decoupled / DC | ⏩ | explicit non-default solver choice is respected → CPU |
| Stopping: `DefaultNewtonRaphsonStoppingCriteria` (‖F‖₂ < √(ε²·n)) | ✅ | `convEpsPerEq` wired through — byte-comparable criterion vs CPU |
| Stopping: uniform ‖F‖∞ < tol | ✅ | GPU-native mode (bench/fast paths) |
| Stopping: `PerEquationTypeStoppingCriteria` | ⏩ | not ported |
| State scaling: NONE / LINE_SEARCH / MAX_VOLTAGE_CHANGE | ✅ | lsMode 0/1/2 on device |
| maxIterations, per-run report | ✅ | |
| Linear solver | ✅ | cuDSS LU; no per-solver ordering choice exposed |

## 2. Security analysis (batched device path)

Engagement: `GpuSecurityAnalysisProvider`. The guard resolves the run's outer-loop list
up front; if anything outside the batchable set is active, the **whole run** executes on
the stock CPU path (base + contingencies + strategies), logged.

### Outer loops in batched SA (summary — corner-case comparison in Part II §A–§D)

| Outer loop (CPU OLF name) | Batched | Notes |
|---|---|---|
| DistributedSlack | ✅ **2026-09-03** | every balance type ± useActiveLimits batched (GpuDsParams); water-filling with saturation + re-normalization matches CPU; failure behavior ≠ LEAVE_ON_SLACK_BUS handled via scenario rejection → CPU. SA parity detail: OLF's `DefaultContingencyActivePowerLossDistribution` distributes each contingency's SCHEDULED lost P BEFORE the post-contingency solve — the batch replays it host-side (`GpuBatchLossPreDistribution`) as anchor shifts + slack share + target overrides (a custom SPI name → whole-run CPU); without the replay, injection-loss scenarios land elsewhere INSIDE the 1 MW band (measured 4.2e-4 kV). §A |
| ReactiveLimits | 🔶 | PV→PQ + PQ→PV back-switch + run's maxPqPvSwitch cap match CPU; deltas: no keep-strongest-PV guard, curve limits frozen at base P, no robust-mode realistic-V forcing. §B |
| TransformerVoltageControl (all 3 flavors) | ✅ **2026-08** | AFTER_GENERATOR (GpuBatchTransformerControl), WITH_GENERATOR/Simple (GpuBatchSimpleTvc: VC-ON first solve + round-to-closest-tap re-solve) and INCREMENTAL (GpuBatchIncrementalTvc: in-kernel dV/drho sensitivities) all DEVICE-batched at 1e-12. Stage-1 scope: no generator VC under the transformer-controlled level (`transformerControlSupported`), per-controller (no shared-group ratio manager). §C |
| AcHvdcAcEmulationLimits, FreezingHvdcACEmulation | ✅ **2026-08** | batched WITH HVDCs (GpuBatchHvdcAcEmulation: freeze-at-warm-flows pre-solve + limit/saturation rounds); every AC-emu HVDC must start LINEAR (saturated/frozen base → CPU); frozen-HVDC base LF stays on CPU (cpuBaseOnly) |
| VoltageMonitoring | ✅ **2026-08** | batched WITH monitoring SVCs (GpuBatchVoltageMonitoring, two-phase pre-RL/full-solve; svcShunt b0 packed); monitoring-SVC base LF stays on CPU (cpuBaseOnly) |
| PhaseControl / IncrementalPhaseControl | ✅ **2026-09-03** | BOTH batched on DEVICE. IncrementalPhaseControl (ACTIVE_POWER_CONTROL + CURRENT_LIMITER): decision loop over the pinned α1 identity with es-loaded flow terms (`GpuBatchIncrementalFlowControl`). Continuous PhaseControl (CONTINUOUS_WITH_DISCRETISATION): row mode 3 re-emits the controlled side's P block at the α1-paired row (cbPcSide/pcDest in the structure) so α1 solves free in the first pass; `GpuBatchContinuousPhase` then rounds to the closest tap (CONTROLLER) / one-tap I-capping steps (LIMITER) with r1Ov pi-row pins. LOCAL control only (remote → CPU via the extraction guard); the SA base LF stays on CPU for CONTROLLER mode (BRANCH_TARGET_P activates mid-run, sequential solver guards row modes ≥ 2). Both `MatchesCpu` at 1e-12. Semantic axes in §D |
| IncrementalTransformerVoltageControl / Simple | ✅ **2026-08** | folded into the TransformerVoltageControl row above |
| TransformerReactivePowerControl (incremental) | ✅ **2026-08** | GpuBatchIncrementalFlowControl KIND_RPC (rho → branch Q, es-loaded flow terms), 1e-12 |
| ShuntVoltageControl (incl. incremental) | ✅ **2026-08** | continuous (SHUNT_B in-Newton) + INCREMENTAL (GpuBatchIncrementalShuntVc, in-kernel dV/dB), 1e-12 |
| SecondaryVoltageControl | ⏩ (batched only) | **plain LF: ✅ validated 2026-07-20** — this loop owns NO solver variable and NO equation type. It builds its own dense side matrices, solves a small LU OUTSIDE the Newton system, and writes back only `GeneratorVoltageControl.setTargetValue()` — i.e. it mutates the RHS of existing BUS_TARGET_V rows and their active flags. On the sequential path (CPU outer loops around a GPU inner solve) it therefore needs NO packing support: `GpuSecondaryVoltageControlTest` drives the ieee14 z1 pilot point to 13.0000 kV with max dV 2.8e-14 vs CPU and ZERO CPU fallbacks. BATCHED path ✅ **2026-08** (GpuBatchSecondaryVc: batched device sensitivities + host zone-system replay, 1e-11 — user-accepted machine precision on the pilot-point knife-edge); stage-1 scope LOCAL generator VC only. §D |
| AreaInterchangeControl | ⏩ | §D |
| AutomationSystem | ⏩ | §D |

### Contingency types

Per-contingency granularity: a non-qualifying contingency routes to the sequential
CPU fallback individually (option `olf.gpu.sa.gpuSequentialLoadFlow` keeps it on the GPU
sequential solver instead, default false); the rest of the batch stays on the device.
Full predicate list in Part II §F.

| Contingency element | Batched | Notes |
|---|---|---|
| Branch outage (N-1, N-2) | ✅ | ≤2 branches, both-sides opening; per-scenario element mask |
| Fragmenting branch outage (islanding) | ✅ | islanded buses frozen to clean flat state; islanding the slack → ⏩; islanding a REMOTE/DISTR_Q VC group member (pilot or any controller) → ⏩ **2026-09-04** (stale qPct weights solved 0.47 kV wrong before the gate — `islandsRemoteVcMember`). **2026-09-06 DISTR_Q audit**: remote groups whose members can CHANGE PIN STATE (finite Q limits) + the RL loop → was whole-run ⏩ in BOTH modes (`remoteGroupPinDynamics`). **Part B LANDED 2026-09-06 (DEVICE mode)**: NON-free member pin/unpin dynamics run in-kernel at 1e-12 — canonical suppressed-dest/mode-4 member encoding (B2), rlRemote RL entries with PILOT-referenced switch conditions (B1/B3), per-scenario DISTR_Q weights recomputed from RAW reactive keys + limitType (an RL pin renormalizes qPct — updateReactiveKeys zeroes controller-disabled members) with the frozen-member TARGET compensation kept in step by updateDistrQTargetsBatch, and each member's own q accumulated into its qBuf row (fillRemoteMemberQBatch). Only a group whose FREE member is pinnable stays whole-run CPU in DEVICE mode (`remoteGroupFreeMemberPinnable`, plan-B4 — the free role would re-pick, A6 territory); the JAVA hybrid keeps the full gate — **until 2026-09-07: the JAVA hybrid now supports remote pins too** (OLF's own RL loop decides; live qPct weights per scenario via dqOv; same B4 free-member pre-gate) and shared TVC groups (member outages CPU-routed per scenario: the element disable SHRINKS the live es variable layout the hybrid's stable-row alignment needs). Tests: post-contingency pin, base-pinned relief (CPU stays pinned — the remote V-based switch-back is knife-edge while the group keeps control), the formerly-gated network now batching, the B4 gate, + the 5 JAVA twins. **B4 LANDED 2026-09-07 (DEVICE)**: the FREE member pins/unpins on-device too — the A6 `qrPtr/qrRows` candidate groups extended to DISTR_Q pilots (candidates = pilot vCol fallback + member vCols; in-kernel mode-scan re-picks the pilot-Q carrier when the free member goes mode 4), `fillDistrQBatch` suppresses the CURRENT carrier's DISTR_Q row, `capturePilotQBatch`+`updateDistrQTargetsBatch` move the pilot's Q target with the carrier, and the qBuf diagnostic's frozen group dests are re-pointed at the pilot's own row (`patchFrozenGroupDests` — the pilot balance was corrupting the free member's own-q RL check). Gate narrowed to `remoteGroupPinDynamicsUnrepresentable`: whole-run CPU only when ALL enabled members of a group are pinnable (all-pinned = pilot vEq deactivation, not modeled) or a pinnable free member has no relocation table (<2 members, or a pilot with half-open/zero-impedance q terms). **JAVA lifted same day** (`setBatchDqQrSuppression` installs just the carrier-suppression tables; OLF's RL loop pins, the pushed mode-4 row re-picks the carrier, the re-extracted activation/target push lands the pilot Q on the new carrier's row — a pin changes activation only, never the variable layout, so the hybrid's stable-row alignment holds); the ALL-PINNED corner (group control lost) batches in BOTH modes since later 2026-09-07: on-device `updatePilotVeqBatch` flips the pilot's V identity to a power row carrying the captured pilot-Q target when the mode-scan finds no live member (and restores it on the first unpin, flag-guarded per scenario/group); in JAVA OLF deactivates the vEq itself and the pushed activation carries it. `device/javaRemoteAllMembersPinnedMatchesCpu` verify genuine in-flight control loss at 1e-12 (base at target, post-outage 0.23 kV off) — the scenario needs a DISTANT keeper PV because the all-pinned system is near-singular without any voltage reference. The remote pin-dynamics gate is now identical in both modes (unclean-pinnable member / table-less pinnable free member only). **FORCED-PV detector, found under the old all-pinnable gate**: OLF's GLOBAL \"all PV buses should switch — strongest stays PV\" rule (switchPvPq at remainingPvBusCount 0) is NOT implemented by the device RL kernel (a network-wide election; measured 1.9 kV batching a network whose only PV buses are the three limited members — the device pinned the bus OLF forces to stay PV at q=1742 > maxQ). Instead of a whole-run gate (which would kill pegase — every gen limited): a per-pass host-side detector rejects a SCENARIO the moment its RL state has zero unpinned non-monitored entries (its trajectory has already diverged from the CPU's), precomputed-guarded by `allPvPinnable` — one unlimited PV bus anywhere skips the check entirely. `deviceRemoteForcedPvRejectsToCpuAndMatches` (count-0 contract). JAVA needs nothing — OLF's own loop applies the rule and the hybrid follows (javaRemoteForcedPvMatchesCpu, 1e-12). **2026-09-07 audit closure, then LIFTED the same day**: a PINNABLE member with HALF-OPEN or ZERO-IMPEDANCE q terms was silently wrong when pinned (measured 3.3e-4 kV with one open line at the member), first whole-run guarded, then FIXED by porting the alternate-dest channels to the ob and zero-impedance families (`obQAlt`/`ziQAlt` per-side alt rows through createStructure; the ob family reuses the generic Family alt/reloc machinery incl. the frozen-group-dest patch, the zi family gets 4 dynamic q slots {F,J}×side with `ziDynDest`; pattern union + checkRange for both). The unclean-member/pilot guards are GONE — members and pilots with half-open or zero-impedance q terms pin and relocate on-device. Tests: free post-ctg pin (carrier g1→g2), base-pinned free, chained g1+g2 pins (carrier → g3), gate fallbacks (all-pinnable + unclean-pinnable), unclean-unpinnable-still-batches — all 1e-12, DEVICE + JAVA twins |
| Generator outage (single / multiple) | ✅ | per-BUS merged injection override; PV→PQ flip, Q-target shed, DS reweight. **2026-09-06**: a gen outage AT a DISTR_Q controller bus that KEEPS another controller is batched — the CPU's updateReactiveKeys renormalization is replayed host-side (toggle-OFF + re-extract + diff) and installed as per-scenario weight overrides (`setBatchDqWeightOverrides`); losing the bus's LAST controller (group restructure) / slack-bus loss / >1 DS bus / remote single-controller shapes → ⏩ |
| Load outage | ✅ | PowerShift deltas (Q-carrying load delta at a PV bus → ⏩) |
| Shunt shift / reconnection | ✅ | admittance override; DISTR_Q-contributing or unpacked shunt → ⏩ |
| Fixed-shunt disconnection | ✅ | element mask |
| Branch+generator+load+shunt combos | ✅ | branch part ≤2; gen on islanded bus → ⏩ |
| HVDC contingency (AC-emulation) | ✅ **2026-09-04** | the hv element's per-scenario disabled bit removes the coupling; the VSC stations' losses ride the generator-outage channel (PropagatedContingency puts them there); solo + branch combos at 1e-12. Needs the olf-local-patches getHvdcIdsToOpen accessor (cbdb9bd90). Still ⏩: setpoint-mode/LCC hvdc outage (not an hv element), a station on the slack bus (slack-loss guard), islanded hv endpoint (hvdcsWithoutPower unmodeled) |
| ≥3 branch outage, single-side opening, bus/busbar loss | ⏩ | |
| ~~live-hvdc first-step divergence~~ | ✅ **RESOLVED 2026-09-05** | NOT numerics (sequential trajectories are bit-identical once iteration counts align — the probe had an off-by-one, OLF's NR runs maxIterations+1). Root cause: the CPU SA freezes AC emulation for EVERY contingency solve by DEFAULT (SA-level `OpenSecurityAnalysisParameters.startWithFrozenACEmulation` = true, injected by `applyGenericContingencyParameters` — NOT visible in the base outer-loop list), and `GpuSaRunScope` keyed the batch's freeze phase on the base list only, skipping it. Scope now honors the SA-level flag; the once-drifting live-hvdc branch outage guards it at 1e-12 (`deviceHvdcContingencyBatchedMatchesCpu`) |
| Outage of a DISTR_RHO controller branch | ⏩ | outaging one member reforms the 1/count coupling — mask cannot represent it. **2026-09-06 audit**: the guard was SILENTLY BYPASSED at an AFTER_GENERATOR base (`isTransformerVoltageControlled()` false with control off emptied both the DISTR_RHO pack and the guard set — a member outage batched 3.7e-3 kV wrong); predicates now key on the equations EXISTING, and SHARED groups route the whole TVC run to CPU (`transformerControlSupported`) because the per-branch alone-toggle deltas cannot compose the group activation. **A1–A4 LANDED 2026-09-06**: existence-keyed DISTR_RHO packing (all-enabled configuration, free = first member), group-level toggle-diff TransformerController (per-member off-deltas; a clamp that would re-pick the FREE role routes the scenario to CPU via STATUS_REJECTED), AFTER_GENERATOR + WITH_GENERATOR shared groups batch at 1e-12 (engagement verified); **A5 LANDED 2026-09-06**: NON-FREE member outages batch on-device at 1e-12 (deviceSharedTransformerControlMatchesCpu: 3-member group, T2wT2 outage batched, free-member T2wT outage CPU-routed by the guard). Three coupled pieces: the drOv channel (`setBatchDrCoefOverrides`: analytic 1/c′ renorm — survivors self 1/c′−1, cross 1/c′, disabled member's cross term dropped — exactly OLF's live DoubleSupplier over the enabled count with the disabled variable's term deactivated); per-scenario nActive (`setBatchActiveRowCounts`) for the sqrt(eps²·n) L2 threshold (CPU removes the outaged member's row); rounded taps applied as FULL pi rows via r1Ov (CPU rounding switches per-step R/X/G/B, not just rho); plus the machine-level DS target carry (captureDsCarry: pair-phase anchors/slack-share/seeds after every solve — without it each machine step re-derived the whole slack distribution from re-uploaded base targets and took extra Newton steps, 1.7e-4 kV). **A6 LANDED 2026-09-07**: FREE-member outages batch on-device at 1e-12 — per-group pilot-Q candidate table (`qrPtr/qrRows`: fallback vCol + member rho cols) with IN-KERNEL mode-scan carrier select (`relocChoose`: first candidate row with mode ∉ {identity, pinned} = OLF's first-enabled free role), role-BUILT machine group rows per step (pairwise offDeltas don't compose for clamp+outage combos) + per-step carrier drOv zeroing, and analytic re-picked-carrier drOv in qualification (deviceSharedTransformerControlMatchesCpu now batches all 3 outages incl. the free member). Documented residual: clamping the CURRENT CARRIER while others stay continuous re-picks the role mid-flight on the CPU's 1-MW DS-threshold knife edge (1.7e-9 offset flips a 1.0045-MW distribution decision) → that SCENARIO routes to CPU (STATUS_REJECTED, `deviceSharedCarrierClampRoutesToCpuAndMatches`); non-carrier clamps stay on-device. Separate pre-existing gap seen on that network: the GPU sequential BASE LF degrades when the base itself clamps → run base on CPU (`olf.gpu.sa.gpuBaseLoadFlow=false`). **JAVA member outages LANDED 2026-09-07** via the hybrid's live/base row map (`liveToBaseMap`: the outaged member's BRANCH_RHO1 leaves the live es, so pushes expand to base indexing — missing var = identity pin at base state — device states compact to live before the bridge, and `setBatchActiveRowCounts` follows nLive; the free member's outage re-picks the carrier through the A6 mode-scan off the pushed mode-0 row; the activation extractor's free picks now skip DISABLED members). INCREMENTAL / useInitialTapPosition stay CPU-routed |
| Node-breaker outage under contingency PROPAGATION | ✅ | 2026-07-20: the propagated retained-breaker openings are per-scenario ZERO-IMPEDANCE element outages (`ziOff` — identity row-mode flip, dummies pinned 0), the impedant part keeps the ≤2 mask rule, the dead stub is frozen by the islanding machinery. Switch outage leaving BOTH sides energized (zi spanning tree would change) → ⏩; outage ISOLATING THE SLACK → ⏩ (CPU relocates the slack, the batched pack anchors on the base slack row) |

### SA machinery parity

| Capability | GPU | Notes |
|---|---|---|
| Base LF on GPU (absorbs CUDA init) | ✅ | `olf.gpu.sa.gpuBaseLoadFlow`, default on; FAILED base → no post-contingency results, same as stock |
| Limit-violation detection / IncreasedViolations filtering | ✅ | shared CPU code — identical by construction (`docs/increased-violations-thresholds.md`) |
| State monitors (bus results) | ✅ | shared CPU code on the restored per-scenario states |
| threadCount > 1 | 🔶 | GPU batch mono-device **by design**: device work serialized, no solver swap; CPU-fallback contingencies parallelize |
| Production-parameter equivalence | ✅ | case9241pegase × 500 ctg × RTE JSON params: **ZERO violation diff at convEpsPerEq=1e-7** (statuses identical incl. the one MAX_ITERATION, 305=305 violations). At the production 1e-4 criterion a ~5-report residual remains whose mechanism is fully characterized: pre-existing violations whose post-ctg increase straddles the increased-violations threshold within the ~2e-4 pu state band two different linear engines legitimately occupy at that criterion. Root causes closed 2026-07-18: per-scenario outer budget (was global per-chunk) + criterion slack |

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
| Any/AtLeastOne/AllViolationCondition (single-stage) | ✅ | speculative precompute: the ENGINE evaluates the condition on CPU-detected violations before our runActionSimulation override is invoked — unused pair states are simply dropped (2026-07-19) |
| Threshold conditions (Branch/Injection/AcDcConverter/T3wt, single-stage) | ✅ | same mechanism — any single-stage condition qualifies |
| Multi-stage conditional actions | ⏩ | stage-on-stage semantics need intermediate-state evaluation |

### Action types (powsybl action-api 7.3.0, complete list)

> **Read this before marking any action ⏩.** A row is only meaningful if the CPU side DOES something.
> Four rows here were once marked ⏩ ("a GPU gap we could close") for actions OLF does not implement at
> all — they were written by reading core's action-api and asking "could our composer express this?",
> never checking `LfActionUtils.createLfAction`, whose `default` throws `UnsupportedOperationException`.
> Compiling such an action on the GPU would make the batched path SUCCEED where the stock CPU path
> throws, violating the provider's safety contract. Check the switch first; mark — , not ⏩.

| Action | Batched | Notes |
|---|---|---|
| TerminalsConnectionAction (open) | ✅ | adds the branch to the pair's outage mask |
| TerminalsConnectionAction (close) | ✅ | re-closing a ctg-opened branch AND closing a BASE-OPEN branch (2026-07-19: opt-in base-open pack, default-disabled mask, reclose clears the bit); dead-island re-energization (disabled endpoint) → ⏩ |
| SwitchAction | ✅ | open, reclose-of-ctg-opened, AND closing a base-open switch (2026-07-20: per-scenario zi-element enable + identity→coupling row flip — the retained switch's dummy variables keep their rows while open). Scope: spanning-tree MERGES with energized endpoints; loop closes composed as the CPU-matching no-op; ≥2 switch closes per scenario → ⏩ |
| Base-open SHUNT reconnection (TerminalsConnectionAction on a shunt) | ✅ | enabled by the olf-local-patches upstream patch (59565edb6: shuntIdsToClose retention + section-controller reconnection, LINEAR model); batched as a per-scenario ShuntShift of the full section admittance; open-of-connected also covered. REQUIRES the patched OLF 2.4.0-SNAPSHOT |
| LoadAction | ✅ | absolute / relative P shed as injection delta |
| RatioTapChangerTapPositionAction | ✅ | full static-row override [y g1 g2 b1 b2 ksi r1 a1] from the target tap — per-tap impedance changes covered; controlled transformer → ⏩ |
| PhaseTapChangerTapPositionAction | ✅ | same compiler (validated incl. ±100% per-step r/x/g/b swings); PST under active phase control → ⏩ |
| ShuntCompensatorPositionAction | ✅ | linear model only; non-linear → ⏩ |
| GeneratorAction (active-power redispatch) | ✅ | composed as a net-injection delta on the warm-continued pair (2026-07-19; DS base-shift accounting fixed — see §A); targetV/targetQ/voltageRegulatorOn changes → ⏩. Validated across every ieee14 contingency incl. islanding |
| PercentChangeLoadAction | — | **not a GPU gap**: OLF itself does not implement it. `LfActionUtils.createLfAction` has no `PCT_LOAD_CHANGE` case and its `default` throws `UnsupportedOperationException`; zero references in the whole OLF tree (the class only exists in core's action-api). Compiling it on the GPU would make the batched path SUCCEED where the CPU path throws — a safety-contract violation. Needs an upstream `LfPercentChangeLoadAction` first (see the #32 shunt patch for the pattern), then the composer is ~5 lines: p0PercentChange/100 × P0 → LoadLoss, with CONSTANT_PQ_RATIO also shifting q |
| RatioTapChangerRegulationAction / PhaseTapChangerRegulationAction | — | **not a GPU gap**: OLF does not implement them. No `createLfAction` case, zero references in the OLF tree; the classes exist only in core's action-api, so OLF throws. Batching them would make the GPU SUCCEED where CPU throws |
| HvdcAction, AreaInterchangeTargetAction | ⏩ | GENUINE gaps, but blocked UPSTREAM of the strategy machinery, not by it: HVDC is not packed by the extractor (§E), and area interchange control is an unported outer loop (§D). Batching the action is meaningless until the element/loop exists device-side |
| StaticVarCompensatorAction, BoundaryLineAction | — | **not a GPU gap**: no `createLfAction` case in OLF, so it throws — same as the regulation actions above |
| MultipleActionsAction | — | **not a GPU gap**: same as above — no `MULTIPLE_ACTIONS` case in `LfActionUtils.createLfAction`, so OLF throws on it. Once upstream supports it, the composer compiles recursively over its members |

## 4. Solver capabilities (batched path specifics)

| Capability | Status | Notes |
|---|---|---|
| cuDSS UBATCH (mode 3) shared-pattern LU | ✅ default | modes 0/1/2 compiled in, runtime-selected; chunk memory auto-sized (fallback 256) |
| Per-scenario row-mode masks + disabled-element masks | ✅ | fixed superset CSR pattern |
| Per-scenario target / injection / admittance / full-row-tap overrides | ✅ | |
| Hot start from base + per-scenario x0 (warm continuation) | ✅ | |
| RL pinning state export/import between solves | ✅ | packed (limitType & 3) \| (switchCount << 2); device reconcile kernel |
| Device-side convergence norms | ✅ | ‖F‖∞ / ‖F‖₂ / finite per scenario |
| Straggler early-exit | ✅ | ≤2% actives past half budget → status −2 → CPU fallback |
| Solve-only batch compaction | ✅ | gather actives → power-of-2 narrow ubatch |
| Reference numbers (RTX A1000 6 GB) | — | live table: `docs/numbers.md` (global number tracker, one row per bench-affecting commit) |

## 5. The gap list to full CPU replacement (ranked)

1. **Remaining SA outer loops (2026-09-03: only TWO left, both modes)** — **AreaInterchangeControl**
   (multi-area DS generalization; the anchor-shift/factor channels + the host water-fill of
   `GpuBatchLossPreDistribution` are its building blocks) and **AutomationSystem** (overload
   automatons: per-scenario branch-tripping decisions off monitored flows — the current-limiter
   machinery is the closest template). Everything else is batched at 1e-12 (SecondaryVC 1e-11):
   DS all balance types, RL, TVC ×3, incremental shunt/phase/RPC, Simple TVC, HVDC AC-emulation,
   VoltageMonitoring, slope-SVC, continuous PhaseControl (both DEVICE and JAVA modes).
   Stage-1 scope carve-outs (per-shape CPU fallback, not per-loop): remote phase/transformer
   control, generator VC under a transformer-controlled level — but NOTE **2026-09-04**: plain
   remote/distributed GENERATOR voltage control (DISTR_Q) is fully batched in SA (validated 1e-12
   incl. an RL-pinned remote controller; deviceRemoteGeneratorControl* tests); "local only" now
   applies ONLY to the SecondaryVC zone replay, SVC voltage monitoring, and continuous
   PhaseControl orchestrators (OLF's own SecondaryVC loop is general — the port collapsed the
   controller/controlled index sets, so remote there is orchestrator restructuring, no kernel
   work). REMOTE continuous PhaseControl (CONTROLLER mode): ✅ **2026-09-05** — cbPcSide now encodes
   the CONTROLLER's a1 column, so the controlled element emits its P block at the right row; LIMITER
   stays local (OLF itself is local-only there); DEVICE + JAVA at 1e-12 — strategy pairs under
   anchor-dependent DS factor types (TARGET / load-based / REMAINING_MARGIN), an islanded DS
   participant in the loss pre-distribution, a custom ContingencyActivePowerLossDistribution SPI,
   loadPowerFactorConstant with load-based balance, ZIP loads for load losses.
1b. ~~generator outage at a DISTR_Q bus~~ / ~~remote continuous phase~~ /
   ~~PerEquationTypeStoppingCriteria~~ / ~~curve limits at runtime P~~ / ~~ZIP loads~~ — the whole
   "time plus new kernels/channels" tier ✅ **landed 2026-09-04..06** (see the rows above and §B).
2. **HVDC packing remainder** — (audited 2026-09-04: the former "HVDC + SVC packing" item is
   mostly landed — AC-emulation HVDC is packed with its batched loops, slope/monitoring SVCs
   landed with modes 2/3.) Still open: **HVDC contingency** (likely cheap: the per-scenario hv
   disabled bit exists — needs the precomputer mapping + non-AC-emu terminal-injection removal),
   **HvdcAction** (a per-scenario hv-parameter override channel: droop/P0/status), LCC shapes,
   and an HVDC with a missing bus end.
3. **Node-breaker contingency propagation** — ✅ **landed 2026-07-20.** A propagated contingency
   opens the target's retained breakers alongside it (switches are LfBranches in breaker mode), so
   the naive shape was ">2 branches" and NOTHING batched on a node-breaker network. Now split: the
   impedant part still obeys the ≤2-branch element mask, the switch part is a per-scenario
   **zero-impedance element outage** (`ziOff`), and the dead stub it leaves behind is frozen by the
   existing islanding machinery. Remaining ⏩ CPU cases: (a) a switch outage that leaves BOTH sides
   energized (the zero-impedance spanning tree would have to be re-picked per scenario); (b) an
   outage that **isolates the slack** — CPU RELOCATES the slack into the largest connected component
   (`PropagatedContingency`: "leads to an isolated slack bus"), and the batched pack anchors its
   targets and distributed-slack participation on the BASE slack row, so a relocated slack is not
   expressible as a per-scenario mask (`ISLANDS_SLACK`). (b) is the next scope-widening step and is
   common on node-breaker networks — measure its share before building it. (Switch CLOSES landed
   2026-07-20; shunt reconnection 2026-07-19.)
4. **Bus/busbar contingencies, ≥3-branch outages, single-side openings.**
4b. **Operator strategies** — the machinery is complete for its scope; what is left is NOT strategy
   work. Only **multi-stage conditional actions** (#30) remain in the machinery itself: stage-on-stage
   needs the stage-1 state evaluated before stage 2 is composed, i.e. a second warm continuation off
   the pair's own result rather than one extra scenario. The two remaining action gaps —
   `HvdcAction` and `AreaInterchangeTargetAction` — are blocked upstream (extractor packing §E and an
   unported outer loop §D). Narrow carve-outs inside supported actions: tap action on a transformer
   under ACTIVE voltage control (worth re-examining now that transformer voltage control is batched
   and the control-row pairing is understood), PST tap action under active phase control, non-linear
   shunt model, GeneratorAction touching targetV/targetQ/regulator rather than P.
5. **⚠️ corner-case deltas inside supported scope** (Part II): keep-strongest-PV guard,
   ~~curve-limits-at-runtime-P~~ (✅ 2026-09-05, §B), shared-control transformer groups — now a
   HARD GATE instead of a silent delta (2026-09-06: whole-run CPU; see the DISTR_RHO contingency
   row for the audit and the 5-step on-device plan) (`useInitialTapPosition`),
   multi-slack components, realistic-voltage post-check on batched states. (DS balance
   types, useActiveLimits, failure behaviors, maxPqPvSwitch, maxOuterLoopIterations —
   closed 2026-07-18 by guard + parameter wiring.)
6. ~~PerEquationTypeStoppingCriteria~~ ✅ **2026-09-05** — per-row threshold arrays on the batch
   context (`setBatchRowCriteria`; the norms kernel reduces max |F_i|/thr_i, thresholds follow the
   live row mode; base LF stays CPU — the sequential solver still only implements the uniform
   criterion); DEVICE + JAVA at 1e-12. ZI shapes without dummies still open. (ZIP/exponential loads:
   ✅ **2026-09-04** — the vdl family adds c_n·target·V^n to the bus P/Q rows with the dJ/dV
   slot, plain LF at ~1e-13 kV and batched SA at 1e-12 (`GpuLoadModelTest`,
   `deviceZipLoadModelMatchesCpu`). Still CPU: load LOSSES of modeled loads (not a target
   override), load-based DS with a load model (per-round coef rescale), a modeled load on a
   controller-shunt bus (dynamic Q-relocation moves its Q row).)

---

# Part II — Corner-case matrices

Line references: `EXT` = `src/main/java/.../gpu/GpuAcDataExtractor.java`, `VAL` =
`.../solver/GpuBatchContingencyValidator.java`, `CU` = `native/cuda/jni/olf_gpu_nr_jni.cu`,
`SOLVER` = `.../solver/GpuBatchedSecurityAnalysisSolver.java`; OLF refs are stock
open-loadflow sources.

## §A. Distributed slack — CPU semantics vs device port

CPU reference: `DistributedSlackOuterLoop` + `ActivePowerDistribution` +
`GenerationActivePowerDistributionStep` / `LoadActivePowerDistributionStep`.
Device: extraction `EXT:299-380`, kernels `computeSlackRetain` `CU:1831-1843` and
`distributeSlackBatch` (full water-filling) `CU:1897-1952`.

| Corner case | CPU behavior | GPU batched | Verdict |
|---|---|---|---|
| Balance type P_MAX (`PROPORTIONAL_TO_GENERATION_P_MAX`) | factor = maxP/droop, eligibility `isParticipating() && droop != 0` | identical formula and eligibility (`EXT:333-347`) | ✅ |
| Balance types P / PARTICIPATION_FACTOR / REMAINING_MARGIN | factor = \|targetP\| / participationFactor>0 / sign-aware margin | not implemented — **guarded**: non-P_MAX balance type → whole-run ⏩ (2026-07-18) | ⏩ |
| Balance types LOAD / CONFORM_LOAD | distribute over loads (`getAbsVariableTargetP`), Q kept at constant power factor if configured | not implemented — same guard | ⏩ |
| Distribution trigger | \|mismatch\| > slackBusPMaxMismatch/SB AND > P_RESIDUE_EPS | `fabs(rem) <= slackDistTol → return` (`CU:1911`); `slackDistTol` = per-unitized slackBusPMaxMismatch | ✅ |
| Iterative re-normalization on generator saturation | clamp at min/maxTargetP, remove from list, re-normalize, repeat until residue < 1e-5 | water-filling loop up to nds iterations, re-normalizes over unsaturated buses, clamps cumulative delta, `P_RESIDUE_EPS = 1e-5` (`CU:1916-1948, 2565`) | ✅ (bus-granular vs CPU's gen-granular within a bus — equivalent because per-bus headroom/footroom sums per-gen limits, `EXT:348-357`) |
| Sign-preservation rule | targetP<0 ⇒ maxTargetP=min(maxTargetP,0); else minTargetP=max(minTargetP,0) | same rule at extraction (`EXT:351-358`) | ✅ |
| useActiveLimits=false (±∞ bounds) | limits replaced by ±MAX_VALUE | **guarded**: useActiveLimits=false with DS → whole-run ⏩ (2026-07-18) | ⏩ |
| Slack bus itself participates | slack's own gens absorb their share (stay in denominator) | `dsSlackFactor` in denominator only; retain share `ρ=1−Σf` recomputed EVERY outer pass, tracking AC-loss change (`CU:1825-1843`) | ✅ |
| All participants saturated | remaining left per failure behavior | `fsumActive ≤ 0 → break`, residual stays at slack (`CU:1927-1929`) | 🔶 equals LEAVE_ON_SLACK_BUS |
| Failure behaviors THROW / FAIL / DISTRIBUTE_ON_REFERENCE_GENERATOR | exception / FAILED result (reverted power) / dump on reference gen, no limits | **wired** (2026-07-18): behavior ≠ LEAVE_ON_SLACK_BUS ⇒ a saturated waterfill flags the scenario, host rejects it (status −2) → CPU fallback replays with the real behavior (RTE production uses FAIL) | ✅ |
| Multiple synchronous components / multiple slack buses | per-SC mismatch map, per-SC sub-reports; maxSlackBusCount>1 | single slack per pack (active BUS_TARGET_PHI); batch runs per LfNetwork component | 🔶 one slack per component only; multi-slack not representable |
| movedBuses epsilon | Σ\|targetP moves\| > 0.9·P_RESIDUE_EPS aggregated | `switched[s]` set if any bus moved at all (`CU:1949-1951`) | ✅ (stricter; only affects when an extra no-op re-solve happens) |
| Per-scenario generator loss at a DS bus | N/A (CPU rebuilds network per contingency) | dsOv override shrinks that bus's factor/base/bounds (`CU:1914,1920,1933`); ≤1 DS bus per contingency; slack-participates now rejects only actual FACTOR changes | ✅ within scope |
| Injection override (load delta / redispatch action) at a DS-participant bus | targetP change is a real target move, not distributed power | dsOvBaseDelta shifts the effective dsBaseTargetP in BOTH retain and waterfill (2026-07-19 — before this, the accounting misread the override as absorbed slack and CANCELLED it) | ✅ |
| DS stopping-band ambiguity | both engines stop anywhere inside slackBusPMaxMismatch (1 MW default) | same band; on ill-conditioned all-pinned networks ~0.1 kV of legitimate between-engines state ambiguity (scales linearly with the band; match at 0.01 MW) — can seed RL back-switch bifurcations in strategy replays | 🔶 inherent to the parameterization, not a defect |
| AreaInterchangeControl replacing DistributedSlack | `filterInconsistentOuterLoops` drops DS when AIC present | AIC in the loop list → whole-run ⏩ by the guard | ✅ safe |

## §B. Reactive limits — CPU semantics vs device port

CPU reference: `ReactiveLimitsOuterLoop`. Device: pack `EXT:281-297`, kernels
`initRlLimitTypeBatch` `CU:1741-1759`, `reactiveLimitsBatch` `CU:1771-1810`,
`seedRlModeFromState` `CU:1972-1985`.

| Corner case | CPU behavior | GPU batched | Verdict |
|---|---|---|---|
| PV→PQ switch criterion | q < minQ − maxReactivePowerMismatch → pin MIN; q > maxQ + tol → pin MAX; q includes loadTargetQ | identical: `q = qBuf + loadQ`, `target = limit − loadQ` (`CU:1784-1797`); `reactiveLimitTol` = per-unitized maxReactivePowerMismatch | ✅ (freeze-at-LIMIT matches CPU `switchPvPq` which freezes at qLimit) |
| PQ→PV back-switch criterion | MIN-pinned & V < targetV → back to PV; MAX-pinned & V > targetV | identical voltage criterion (`CU:1799-1808`) | ✅ |
| Switch-count cap (oscillation guard) | maxPqPvSwitch, default 3; blocks the PQ→PV direction | run's `reactiveLimitsMaxPqPvSwitch` **wired** to the device (2026-07-18; 0 ⇒ default 3) | ✅ |
| Keep-strongest-bus-PV when ALL PV buses would switch | if remainingPvBusCount==0, the strongest bus (nominalV, then targetP, then id) stays PV | **no equivalent** — all buses can switch PQ in a scenario | ⚠️ pathological-network corner (all-PV-breach); never observed in the 9241 equivalence run |
| Q-limit UPDATE while pinned (`busesWithUpdatedQLimits`) | if pinned and \|limit − q\| > tol because the LIMIT moved (curve at new P), re-freeze at new limit | ✅ **2026-09-05** — with the curve channel active, the pinned branch re-freezes at the moved limit (gated on cvStride; unreachable with static limits) | closed with the curve row below |
| Reactive capability CURVES | per-gen min/maxQ evaluated at CURRENT targetP each check — a DS redistribution that moves targetP moves the curve limits | ✅ **2026-09-05** — curve channel (`setBatchRlCurves` / `rlCurveLimitsBatch`): per participating curve gen, points + p_now = baseP + share·(rowTarget − base); dynamic limits re-evaluated before every RL decision + the update-while-pinned re-freeze. Validated on a sloped ieee14 curve (1.8 kV wrong with the channel off) at 1e-11 (association ulps of the share reconstruction — machine precision). Scope: OLF patch b8e813a15 (public getReactiveLimits); extrapolateReactiveLimits+curves → whole-run CPU; injection outage at a curve bus's P row / strategy pairs → per-contingency CPU (the linkage models the DS delta only); non-participating curve gens stay static (exact) | closed (was the ⚠️ curve-vs-DS coupling) |
| Multi-generator buses | bus-aggregated Σ minQ/maxQ over voltage-capable gens; whole bus flips at once | same aggregation, snapshotted (`LfBus.getMinQ/getMaxQ` summed at pack) | ✅ |
| Base-PQ-but-capable buses (back-switchable under a relieving contingency) | checkPqBus path | packed explicitly (`EXT:284-296` second kind) | ✅ |
| Robust mode (`voltageRemoteControlRobustMode`): unrealistic-V forcing, MIN/MAX_REALISTIC_V pin at initial targetQ, setV(1) | remote-controller-only extra criteria | not implemented — but remote control (DISTR_Q) is rejected from the batch entirely | — unreachable in batch scope |
| Buses with generator slope (SVC) skip back-switch | `hasGeneratorsWithSlope()` guard | SVC never packs | — unreachable |
| Remote reactive-power controller buses | switchReactiveControllerBusPq | not packed | — unreachable (⏩ via guard/extractor) |
| Freeze semantics divergence (observed) | manager/robust paths freeze Q at the EVALUATED switch-time value (`freezeGenerationTargetQAndDisableGeneratorVoltageControl(getQ().eval())`); RL switchPvPq freezes at the limit | device always pins at the limit | ⚠️ the GeneratorAction ~0.17 kV delta traces here; blocking GeneratorAction batching, not contingencies |
| `isFixVoltageTargets` (VoltageTargetChecker.fix at initialize) | pre-pass mutating implausible targets | runs on CPU before extraction (base solve) → inherited | ✅ by construction |
| Which scenarios get checked | every outer sweep on the current state | only scenarios (re)solved AND converged this pass (`rlActive = wasActive && status==0`, `CU:2944-2945`); qBuf rebuilt from scratch each pass, disabled branches skipped | ✅ equivalent fixpoint |
| Warm continuation seeding (strategy pairs) | N/A | parent's (limitType, switchCount) imported; `seedRlModeFromState` re-pins target = limit − loadQ; base seed derived from base row mode (`CU:1741-1759, 1972-1985`) | ✅ validated (B2 drill) |

## §C. Transformer voltage control — CPU semantics vs device port

CPU reference: `TransformerVoltageControlOuterLoop` (AFTER_GENERATOR) + `TransformerRatioManager`
+ `GeneratorVoltageControlManager`. Device: Java orchestrator `SOLVER:1000-1130`,
support guard `SOLVER:104-119`, controllers `SOLVER:124-177`.

| Corner case | CPU behavior | GPU batched | Verdict |
|---|---|---|---|
| Mode WITH_GENERATOR (Simple: continuous in-system + round at iter 0) | `SimpleTransformerVoltageControlOuterLoop` | ⏩ (guard) | — |
| Mode INCREMENTAL (sensitivity-driven tap steps, MAX_DIRECTION_CHANGE=3, MIN_SENSI 0.05, maxTapShift) | `IncrementalTransformerVoltageControlOuterLoop` | ⏩ (guard) | — |
| Mode AFTER_GENERATOR: 3-phase INITIAL→CONTROL→COMPLETE | enable out-of-deadband controllers, freeze generators under the controlled level, solve continuous, clamp to extremes, round, unfreeze | orchestrator: deadband-check on first converged solve → ON-deltas (row-mode+target diff from toggling the control) → ≤8 steps of clamp(min/maxR1)-then-round(closestTap) with warm restarts (`SOLVER:1019-1127`) | ✅ within the guard's scope |
| Generator freezing under max controlled nominal V (stage 2), step-up-transformer / VSC exemptions | `GeneratorVoltageControlManager.disable/enable`, freeze at evaluated Q | **not ported** — `transformerControlSupported` rejects any network where a generator-controlled bus sits at nominalV ≤ max transformer-controlled nominalV (`SOLVER:112-118`) → whole-run ⏩ | 🔶 scope guard instead of port; the freeze-at-evaluated-q semantics is the same blocker as GeneratorAction |
| Deadband | `targetDeadband.orElse(0.1 kV /nominalV)/2`, tested at INITIAL | identical formula (`SOLVER:151`), tested at INITIAL only | ✅ (CPU also only gates enablement at initStep; no re-deadband after rounding) |
| Shared control groups (`useInitialTapPosition`: group-averaged min/max/initial R1, spread-preserving update) | `TransformerRatioManager` shared mode | per-controller independent min/maxR1 + closestTap — equals `useInitialTapPosition=false` individual behavior | ⚠️ `useInitialTapPosition=true` group semantics not ported; parallel-transformer groups will round independently |
| Controller not connected both sides / missing r1Col/vCol | excluded from control | skipped at controller build (`SOLVER:140-150`) | ✅ |
| Controller transformer outaged in the scenario | N/A per-contingency rebuild | controller skipped when its element is masked (`SOLVER:1016`) | ✅ |
| Tap table unavailable (non-PiModelArray) | rounds via piModel.roundR1ToClosestTap | reflection snapshot; empty table ⇒ ratio left continuous (`SOLVER:87-88, 211-212`) | ⚠️ silent continuous ratio if reflection breaks (logged) |
| Equal-nominal-V transformers excluded from the freeze-limit computation | `GeneratorVoltageControlManager` validity filter | N/A (stage 2 not ported) | — |

## §D. Outer loops not ported to the batch — semantic-axis catalogue (port checklists)

Each of these is ⏩ in batched SA (whole-run fallback) and ✅-by-construction in plain LF.
The axes below are what a device port must reproduce — captured now so a future port
doesn't rediscover them.

- **PhaseControl** (`PhaseControl`, CONTINUOUS_WITH_DISCRETISATION — the OLF default mode):
  CONTROLLER mode = continuous solve then switch-off + round A1 at iteration 0; LIMITER mode =
  from iteration 1, one-tap steps to bring current under limit, direction from sign of dI/dA1;
  a PST necessary for connectivity (its removal changes component count) gets its control
  switched off. Row structure: `BRANCH_TARGET_P` on the CONTROLLED branch (one closed-branch
  p1/p2 term, side per `getControlledSide()`) active while control is on; `BRANCH_TARGET_ALPHA1`
  (identity row on the α1 variable, target = `piModel.getA1()`) active while it is off — exactly
  one of the two at a time, so the system stays square (`AcEquationSystemCreator:528-587`).
  Unit A throws — current limiter is never solved continuously.
- **IncrementalPhaseControl** (`IncrementalPhaseControl`): `setPhaseControlEnabled` is NEVER
  called, so `BRANCH_TARGET_ALPHA1` stays active for the whole run — α1 is a variable PINNED to
  its tap value, and the column exists only so the loop can solve the transposed Jacobian
  (RHS = `toRadians(1)` at the α1 equation row, `solveTransposed`, one column per controller).
  CONTROLLER acts outside half-deadband (floor 0.5 MW), then `da = toRadians(dp/a2p)` and
  `updateTapPositionToReachNewA1` → CLOSEST tap; LIMITER acts only when over limit and uses
  `updateTapPositionToExceedNewA1` → FIRST tap ABOVE, deliberately overshooting so the limit
  clears. SENSI_EPS=1e-6, MAX_DIRECTION_CHANGE=3 (after 3 reversals the PST is locked monotone),
  MAX_TAP_SHIFT unbounded (tap range + direction guard are the only limits; a saturated PST
  contributes 0 moves and the loop simply reports STABLE), cross-PST impact warning at 0.75
  margin (log only, no algorithmic effect). DC variant ignores LIMITER PSTs entirely.

  **Port state — much of the structure already exists and is unreachable today:** `CB_SCATTER`
  (`native/cuda/closed_branch_scatter.h`) already carries `dp1da1[5] dq1da1[12] dp2da1[19]
  dq2da1[26]` scattering to the branch-owned α1 column (endpoint 2, colVar 0), and
  `GpuAcDataExtractor:203` already packs `branchAlphaCol` into `cbIdx[8]` — both dead because
  `validate()` throws first. Staged port: (1) allowlist α1 + pack the `BRANCH_TARGET_ALPHA1`
  identity row — structurally the same as the ZI dummy-target rows, and the CONTROLLER on/off
  swap is a row-mode identity flip, i.e. the #33 `ziOff` machinery; this alone unblocks plain LF
  in INCREMENTAL mode (outer loops run on CPU there by construction). (2) `BRANCH_TARGET_P` —
  the genuinely new piece, a BRANCH-owned row fed by one closed-branch p1/p2 term; precedent is
  the DISTR_RHO family, which already scatters into a branch-owned extra row
  (`olf_gpu_nr_jni.cu:519`). `computeQRow` also throws on phase-control shapes today.
  (3) Batched SA is the expensive tier: per-scenario transposed solve + the discrete tap finders
  + the direction-change guard.
- **IncrementalTransformerVoltageControl**: BRANCH_TARGET_RHO1 sensitivities, insensitivity
  filter 0.05 (reset when another loop ran in between), one-tap-at-a-time distribution
  across shared controllers, maxTapShift per sweep, MAX_DIRECTION_CHANGE=3.
- **TransformerReactivePowerControl** (incremental): Q-deadband (min 0.1 MVar), single
  controller only (multi-controller is an OLF TODO), generator reactive control has
  priority for the target.
- **ShuntVoltageControl**: continuous SHUNT_B in-system, dispatchB rounding at iteration 0;
  **IncrementalShuntVoltageControl**: SHUNT_TARGET_B sensitivities, shunts sorted by B
  magnitude, maxSectionShift per sweep, deadband 0.1 kV default, MAX_DIRECTION_CHANGE=3.
- **SecondaryVoltageControl**: control zones with pilot buses; adjusts controller-bus
  target Vs via a cross-zone LU sensitivity system; re-enables helpful PQ-switched
  controllers; DV_EPS=1e-4 pu, alignment eps 1e-3; ANY implausible new targetV ⇒ zone-wide
  skip with FAILED status.
  **Structural footprint: NONE** — no variable, no equation type; the sensitivity RHS is
  built on existing BUS_TARGET_V rows and solved via `j.solveTransposed`, the side matrices
  (`createA`/`createK0`/`createJk`/`createJvpp`) are dense and standalone. Hence plain LF
  works unmodified (validated 2026-07-20). For a BATCHED port the pieces are: (a) a
  transposed solve per scenario — reuse the per-scenario cuDSS factorization rather than
  shipping Jacobians to the host (incremental phase control needs the same primitive);
  (b) per-scenario target mutation, which the override toolkit already does; (c) the
  generator-control enable/disable propagation. ⚠️ Note it FIGHTS the reactive-limits loop:
  `tryToReEnableHelpfulControllerBuses` flips Q-limited buses back from BUS_TARGET_Q to
  BUS_TARGET_V when SOME but not all controllers in a zone are limited — a mid-run
  activation change the device RL port would have to honor. Gate: `secondaryVoltageControl`
  (default false, absent from the RTE params file) + the iidm SecondaryVoltageControl
  extension with a resolvable pilot bus per zone.
- **AreaInterchangeControl**: replaces DistributedSlack (config filter); per-area
  interchange mismatch vs areaInterchangePMaxMismatch; buses-without-area absorb slack
  first, then margin-proportional cross-area distribution; throws on >1 synchronous
  component; delegates to plain DistributedSlack when no areas.
- **VoltageMonitoring** (SVC standby automaton): PQ buses with MONITORING_VOLTAGE SVCs;
  V beyond high/low threshold ⇒ PQ→PV at the automaton's high/low target V, control type
  mutated to VOLTAGE.
- **AutomationSystem** (overload management): monitored-branch current vs tripping
  thresholds ⇒ open/close branches with real connectivity mutation (one tripping per
  system per check).
- **AcHvdcAcEmulationLimits**: LINEAR ↔ SATURATION_CS1→CS2 / CS2→CS1 transitions on Pmax
  crossings, incl. side-switch on flow reversal. **FreezingHvdcACEmulation**: freeze AC
  emulation at start (unless angle undefined — HVDC reconnected by action), one UNFREEZE
  step, then COMPLETE.

### Execution-model comparison (the harness itself)

| Aspect | CPU engine | Device batch | Verdict |
|---|---|---|---|
| Loop ordering | fixed nesting order (innermost DistributedSlack → … → AutomationSystem outermost); each loop re-run to stability before moving on; full-list sweeps until `lastUnstableOuterLoop` fixpoint | two-phase machine: DS phase stabilizes first, then RL, restart at DS if RL changed anything (`CU:2609-3014`); TVC orchestrated outside around whole batched solves | ✅ same DS-before-RL relative order; equivalence validated empirically at 9241×500 |
| Global iteration budget | maxOuterLoopIterations (default 20), reached ⇒ result forced UNSTABLE | run's `maxOuterLoopIterations` **wired** to the device (2026-07-18; RTE production = 30); exhausted ⇒ scenario unconverged → CPU fallback. Wiring 30 shrank the production violation diff 22→20 (6→5 ctg) for ~1.5 s warm | ✅ |
| Warm restart between outer iterations | PreviousValueVoltageInitializer | device state persists in place; only changed scenarios re-solved (`active[s]=rlSwitched[s]`) | ✅ |
| State injection under contingency-mutated es (islanding) | CPU solves in the mutated es directly | device states use the BASE row layout; injection remaps per-variable through a base row snapshot (2026-07-19 — raw-index injection shifted every bus after a disabled island) | ✅ |
| Unrealistic-state check (min/maxRealisticVoltage, deferred in robust mode until the fixing loop stabilizes) | engine-level check → UNREALISTIC_STATE | **not implemented on device**; a wildly unrealistic scenario either diverges (status −1) or converges and is accepted | ⚠️ no realistic-voltage post-check on batched states — worth a cheap device-side min/max V scan |
| No-generator-voltage-control network ⇒ immediate SOLVER_FAILED | engine pre-condition | base solve is CPU-orchestrated → inherited | ✅ |
| Per-loop reports / distributedActivePower accounting | reportNode tree + context data | not reproduced per scenario (batch reports totals; fallback scenarios get full CPU reports) | 🔶 reporting granularity, not results |

## §E. Extractor envelope — exhaustive accept/reject inventory

### E.1 Equation types (`validate()`, EXT:516-555; activation re-walk EXT:422-452)

| Equation type | Disposition |
|---|---|
| BUS_TARGET_P / BUS_TARGET_Q | accepted (power rows; terms re-checked) |
| DISTR_Q | accepted structurally; real guard in `buildDistrQContributions` (§E.3) |
| ZERO_V / ZERO_PHI / DUMMY_TARGET_P / DUMMY_TARGET_Q | accepted (zero-impedance family) |
| DISTR_RHO | accepted (linear coupling, §E.4) |
| BUS_TARGET_V / BUS_TARGET_PHI / BRANCH_TARGET_RHO1 | accepted as identity rows; non-`VariableEquationTerm` active term ❌ |
| anything else (incl. activated at runtime) | ❌ `unsupported equation type` (EXT:553, 444-445) |
| equation ARRAYS | only BUS_TARGET_P/Q array types; others ❌ (EXT:494-496) |

### E.2 Terms and variables

- Exactly **12 supported P/Q term classes** (EXT:74-80): Closed/Open branch side-1/2
  active+reactive flows, shunt active+reactive, HVDC AC-emulation side-1/2 active. Plus
  dummy-coupling terms (all-DUMMY variables, EXT:567-570). Any other active term ❌ —
  this is the branch that rejects ZIP/exponential loads, SVC slope, LCC/VSC terms.
- Array evaluators: only `ac_{p,q}_array_closed_{1,2}` (EXT:84-85); others ❌.
- Variables: BUS_V, BUS_PHI, BRANCH_RHO1, DUMMY_P, DUMMY_Q accepted; **BRANCH_ALPHA1
  (phase control) and SHUNT_B (controllable shunt) ❌** (EXT:484-488).

### E.3 DISTR_Q — remote generator voltage control (the multi-controller case)

Entered for MAIN-merge pilot buses with remote generator voltage control (EXT:674-678).
**Multiple controllers on one pilot bus: fully supported in sequential LF** — every
non-free controller emits its own DISTR_Q row, weighted by
`remoteControlReactivePercent` (EXT:681-689); the pilot's Q row rides on the FREE
controller (the one with inactive DISTR_Q); **no free controller (all DISTR_Q active) ❌**.
Contributor shapes per controller:

| Contributor shape | Disposition |
|---|---|
| fixed shunt at controller bus | ✅ wired (kind 1) |
| voltage-CONTROLLING shunt (SHUNT_B) at controller bus | ❌ throw |
| incident closed branch | ✅ (kind 0, side-aware) |
| incident half-open branch | ✅ (kind 2, connected-side q) |
| incident zero-impedance spanning-tree edge with DUMMY_Q | ✅ (kind 3, ±DUMMY_Q linear) |
| incident zero-impedance spanning-tree edge without DUMMY_Q | ❌ throw |
| incident zero-impedance NON-tree edge | skipped (contributes nothing — matches OLF) |
| disabled incident branch | skipped |
| any other incident shape | ❌ throw |

Remote-controller (non-pilot) bus: own Q scatter suppressed, qRow −1 (EXT:609-614).
**Batch path: all of §E.3 is out of scope** — the RL pack only admits local control
(EXT:275); a contingency touching remote control falls back per the provider mapping.

### E.4 DISTR_RHO — transformer ratio distribution

- Single LOCAL transformer controlling its own terminal bus: derived (qRow onto the rho
  column) — mask-safe, batchable.
- Pilot controlled by transformers: qRow onto the free controller's rho row; ❌ when no
  supported voltage control (shunt VC on the pilot), no free controller, or free
  controller without a BRANCH_RHO1 column (EXT:639-653). Remote transformer controller
  yields a zero pilot-Q diagonal left to cuDSS pivoting (documented, not thrown).
- Multi-transformer groups: 1/count − δjk coupling derived over enabled controllers
  (EXT:747-774) — works in sequential LF; **batch rejects any contingency outaging a
  member of an active DISTR_RHO group** (`distrRhoControllerBranches`, EXT:783-801),
  because the outage reforms the 1/count coefficients.

### E.5 Zero-impedance shapes (EXT:153-174)

| Shape | Disposition |
|---|---|
| ZI branch, both buses, DUMMY_P+DUMMY_Q present | ✅ (spanning-tree edge adds ZERO_V/ZERO_PHI coupling; non-tree edge holds dummies at identity) |
| ZI branch open on one side | ❌ throw |
| ZI branch without dummies (both ends voltage-controlled, 3-winding star) | ❌ throw |

### E.6 Branch/shunt/HVDC packing (EXT:146-232)

Disabled branches skipped by default; closed/side-1-open/side-2-open packed. Base-open
CLOSABLE branches (disabled, both buses present and energized) are packed opt-in for the
batch (appended after the normal closed branches, default-disabled per scenario) — the
plain-LF path never packs them.
Shunts packed per bus. HVDC packed only with both buses present AND AC-emulation mode;
otherwise the converters appear only as bus P injections.

### E.7 Elements with no first-class handling (grep-verified absent from the extractor)

SVC/standby automaton, batteries, LCC/VSC converters, dangling lines, tie lines,
three-winding-transformer legs (star bus → §E.5 throw), areas, asymmetrical models.
They pass through only insofar as they reduce to plain bus P/Q targets; any term/variable
they introduce outside §E.2 throws.

## §F. Batch disqualification predicates (complete list)

Validator level (`VAL:110-161`):
1. `NOT_IN_GPU_MODEL` — any outaged branch absent from the GPU element map.
2. `FRAGMENTS_NETWORK` — islanding while `handleIslanding=false` (fixed-iter bench mode).
3. `ISLANDS_SLACK` — islanding that removes a slack bus (production mode keeps other
   islanding via frozen rows).

Solver qualification level (`SOLVER:642-701`):
4. Tap override on an unpacked branch, or on a branch whose ratio or phase is a state
   variable (r1col/a1col ≥ 0).
5. Shunt shift/outage on an unpacked shunt or a DISTR_Q-contributing shunt.
6. Injection outage unresolvable: slack-bus generator loss, >1 DS-participant bus in one
   contingency, remote-controlling generator, or (when the slack itself participates) any
   DS-participant-bus generator loss.
7. Combo with a branch part outaging a DISTR_RHO controller (§E.4).
8. Combo where a lost generator sits on an islanded bus.
9. Reclose of a branch not in the base-open pack (unknown id, zero-impedance, or a disabled
   endpoint bus — dead-island re-energization).

Provider mapping level (before qualification): >2 branches, single-side openings,
bus/busbar losses, HVDC contingencies, Q-carrying load delta at a PV bus, non-linear
shunt models — all route the contingency to CPU (Part I §2 table).

---

## Maintenance

Update this file whenever a row changes state (the PR that lands a capability flips its
mark), and re-run the two benches plus `GpuBatchedAcSecurityAnalysisTest` before flipping
anything to ✅. ⚠️ rows are standing corner-case debts INSIDE supported scope — each
needs either a device fix or a guard extension (they are enumerated in gap-list item 5).
Numbers and dated evidence live in `codegen/gpu_benchmark.md`; this file holds only the
current truth.
