# Task list (recovered 2026-07-21)

The task tracker (an MCP server) disconnected mid-session, so this file is the durable
record. It was reconstructed from the 2026-07-21 session transcript: every entry below is
backed by a commit, a measurement, or an explicit decision recorded in that session.

**Fidelity boundary.** Only tasks referenced during that session are here. Numbers below
#24 from earlier sessions are NOT recoverable from it (except #18, which came up
explicitly) — if the tracker ever comes back, reconcile rather than assume this file is
complete. Keep the numbering: it is cited in commit messages.

## Open

### #93 — DONE: the tests' tolerance switches are gone (2026-11-10)

`AbstractGpuSaTest`'s two comparison helpers read `olf.gpu.test.convEps` / `olf.gpu.test.slackTol` (falling back to
`OLF_GPU_DBG_CONV_EPS` / `OLF_GPU_DBG_SLACK_TOL`) and tightened OLF's Newton criterion and slack band on both arms; the
rte6515 harness read `OLF_GPU_DBG_CONV_EPS` too. A switch that selects a computation has no place in the final design,
tests included, and tightening hides where the arms' trajectories part instead of answering it. The reads are gone.
- Two tests set the properties in code (slack 0.01 MW, Newton 1e-9). Each read called `OpenLoadFlowParameters.create`,
  which replaces the extension, so the slack read wiped the Newton one: both tests ran at OLF's default Newton
  criterion and a 0.01 MW band.
- Both pass at OLF's default criteria at 1e-12 kV, on cuDSS and on our batched LU, and now run there:
  `GpuSaDistributedSlackTest.l121SevereContingencyMatchesCpu` (was `l121SevereContingencyMatchesCpuAtTightSlackTol`,
  whose comment called the 0.105 kV it once measured at the default band "legitimate ambiguity, NOT a defect") and
  `GpuSaOperatorStrategyTest.generatorActionStressAcrossContingencies`.
- `docs/env-vars.md` lists the two variables as removed; §A of `docs/gpu-vs-cpu-capabilities.md` no longer calls the
  stopping band's 0.1 kV inherent.
- The same kind of switch went in a second commit: `OLF_GPU_CENSUS_CONV_EPS` (`GpuRteScopeCensusTest`) and
  `OLF_GPU_DISC_EPS` / `OLF_GPU_DISC_SLACKTOL` (`GpuSingleSolveRejectReasonProbe`). The probe's GPU arm selected the
  GPU solver with `OpenLoadFlowParameters.create`, which dropped the RTE parameters' OLF extension: it ran OLF's
  defaults against the CPU arm's RTE parameters. It now sets the solver on the loaded extension. On its default case
  (PEGASE 9241, LINE-4985-165 out) the worst bus gap is 2.425e-12 pu before and 2.430e-12 pu after.
- `OLF_GPU_CENSUS_NO_ACTIVE_LIMITS` has the same `create` flaw (both census arms lose the RTE extension); it is a
  switch of another kind, still read.

Gates: `GpuSa*Test` 299/0/2 on cuDSS and on our batched LU; full suite 523/0/6 with checkstyle. After the second
commit, on `main` at f5f04199 with the native library rebuilt: `GpuSa*Test` 299/0/2 on both, full suite 525/0/6 with
checkstyle (the census skips without its RTE case; the probe ran by hand, numbers above).

### #92 — DONE: the violation screen's band, existence tests, scan order and doc (h–k) (2026-11-10)

Issues h–k of `docs/device-final-results-plan.md` §7 (branch `device-final-results`).
- **h.** `GpuViolationScreen` (and the kernel's comment) put the device-vs-host gap at ~1e-16 relative and the 1e-9
  band ~7 orders above it. L12 measured 2.1e-10 at worst (network.xiidm, 906,434 violations, on currents: nearly
  cancelling terms amplify last-bit differences), a 5x margin. A candidate the screen misses is a violation lost and
  one it adds costs one exact re-test, so the band is now 1e-7 (about 500x); the comments give the measurement.
- **i.** `pairDistributedOnDevice` and the post-contingency path tested a state's existence with `states().get`,
  which with the state store reads the whole state from its file: for every candidate pair of every strategy, three
  times per contingency (`everyStrategyAnsweredOnDevice` runs three times). They use `containsKey`, an index lookup.
- **j.** `GpuLimitScanner.detectFromCandidates` scanned the branches the screen does not cover after the buses, while
  OLF reports every branch before the buses. They are scanned in place with the side-opened ones.
  `GpuLimitTableTest.scansAnUncoveredBranchInPlaceBeforeTheBuses` (an uncovered overloaded line beside a voltage
  violation, the screen forged) fails without it. No case of ours shows it: the screen leaves out zero-impedance
  branches (OLF reports no violation on them: measured on IEEE 14 with a zero-impedance line given an active power
  limit under its flow), base-disabled ones (OLF skips them) and branches missing a state column (none on one
  connected component).
- **k.** `docs/on-device-violation-detection.md`: status, the screen that "would not help" (what landed is a screen),
  the group cut (stops at the first limit beyond the band since 9725fdfd), the band's measurement, zero-impedance
  branches settled, and "What NOT to do" now names deciding on the device instead of screening.

Gates: `GpuSa*Test` 299/0/2 on cuDSS and on our batched LU; full suite 523/0/6 with checkstyle.

### #91 — FIXED: what a strategy's threshold condition reads after a device contingency (d, e) (2026-11-10)

Issues d and e of `docs/device-final-results-plan.md` §7 (branch `device-final-results`). OLF evaluates a strategy's
threshold condition on the network the contingency's load flow left (`ThresholdConditionEvaluator`): a branch's or a
three-windings transformer's flow from the state vector, a generator's ACTIVE_POWER from its target P, TARGET_P from its
initial target P (which OLF sets to the target P before a single strategy).

**e — FIXED on the device.** After a device-answered contingency the network held the loss pre-distribution only (OLF's
own `ContingencyActivePowerLossDistribution` runs for a contingency a strategy targets), not the device's slack passes.
Measured on IEEE 14 with B3/B6/B8 at 20 MW and B1-G out of the distribution: B6-G's target is 4.929 MW in the base and
9.865 MW after L1-2-1, a line outage that loses no injection, so the passes alone move it; a condition at 7.397 MW held
on the CPU and not on the GPU. The chunk solver already read each contingency's final generator moves for its pairs'
seeding (`genMovesByIndex`); the solver now hands them to the host as absolute targets (`BatchScenarioResult
.GeneratorTargets`: the anchors read before the solve plus the moves, for the contingencies with strategy pairs), and
`runPostContingencySimulation` gives them to the participating generators when a strategy on the contingency has an
`InjectionThresholdCondition` (a lost generator keeps OLF's 0). OLF's full restore (a contingency a strategy targets
never takes the targeted one) and `BusDcState` put the targets and initial targets back.
`GpuSaOperatorStrategyTest.injectionConditionReadsTheDevicesSlackDistribution` fails without it (0 strategy results
against OLF's 1).

**d — routed to OLF's path.** A contingency that runs out of Newton iterations (MAX_ITERATION_REACHED) still has its
strategies evaluated, and NewtonRaphson leaves the state vector at its last iterate: a branch threshold reads that
iterate's flows. The device answered such a contingency without injecting anything, so the condition read the flows
the network held before. Measured on IEEE 14 with the budget one step short of the 4 steps L1-2-1 needs: L1-5-1 at
260.95 MW in the outage's solution against 75.51 MW in the base, a condition at 168.2 MW held on the CPU and not on the
GPU. The last iterate of a run that does not converge is not something the device reproduces to the bit (a plateau's
last sample is a lottery, 4c2b48cb9), so a contingency the device exhausted runs on the CPU when a strategy of it has a
threshold condition (injection conditions included: that combination has no fixture); its pairs stay on the device.
`GpuSaOperatorStrategyTest.thresholdConditionAfterAnExhaustedContingencyReadsOlfsLastIterate` fails without it.

Not covered: the area interchange control moves the same participating generators, and nothing tests that its moves
reach the targets handed back. The RTE runs carry no operator strategy, so nothing changes there.
Gates: `GpuSa*Test` 299/0/2 on cuDSS and on our batched LU; full suite 522/0/6 with checkstyle.

### #90 — FIXED (in OLF): the network restore left the shunt compensators' sections, so results depended on the order (2026-11-10)

A shunt with sections (a voltage-controlling shunt, or shunt compensators an action operates) keeps the section of each
of its compensators, and its susceptance and conductance are their sum: every section change sums them again
(`Controller.updateSectionB`, `dispatchB`). OLF's `BusState` saved and restored the sums but not the sections (upstream
main 144a54c9d too), so after a contingency or a strategy that changed a section, the next one started from the base
susceptance with the changed section, and its first section change summed it in. Measured with OLF alone:
- **Incremental shunt voltage control** (ShuntNetworkFactory plus `l1_BIS`, SHUNT's deadband at 0.1 kV, contingencies
  `l1` and the loss of `ld1`), b3 against the same contingency run alone: target 393 kV with `l1_BIS` at twice `l1`'s
  impedance, `ld1` after `l1` +4.2e-8 kV and `l1` after `ld1` +1.3e-6 kV; target 395 kV at ten times, `ld1` after `l1`
  +4.0e-5 kV. Every other case of the sweep was bit-identical, and so was the continuous mode, whose rounding starts from
  the section only to break an exact tie.
- **Position actions on one bus** (`createWithShuntSharedRemoteControl` plus SHUNT3B next to SHUNT3, both on section 25,
  contingency `tr2`): strategy1 puts SHUNT3 on 50, strategy2 SHUNT3B on 0; strategy2 kept SHUNT3 on 50 after strategy1,
  b4 at 392.407 kV instead of 399.617 kV.

`BusState` now saves and restores the section of each compensator of the controller shunt and of the shunt. Fork commit
3d9e6d64b on `olf-local-patches` (suite 1531/0/3); upstream branch `restore-shunt-sections` 2927936b9 on upstream main
144a54c9d, signed off (suite 1617/0/4, 0 checkstyle violations). Both are pushed to `gb` (2026-11-10). The tests
`OpenSecurityAnalysisTest.testIncrementalShuntVoltageControlAfterASectionChange` and
`OpenSecurityAnalysisWithActionsTest.testShuntActionAfterAnotherOnTheSameBus` fail without the fix.

The device answers every contingency from the base sections and every pair from the post-contingency ones, which is
what the fixed OLF computes. `GpuSaShuntControlTest.deviceIncrementalShuntSectionsAfterAContingencyThatSwitchedOne` and
`deviceShuntActionAfterAnotherOnTheSameBus` hold it, and fail against OLF without the fix (b2 393.2433645 against the
device's 393.2433456 kV; b3 19.393 against 20.133 kV). With the fixed OLF: `GpuSa*Test` 291/0/2 on cuDSS and on our
LU, full suite 514/0/6 with checkstyle. The RTE parameters leave the shunt voltage control off and the RTE runs carry
no action, so no RTE bus has a shunt with sections.

### #89 — a contingency's own load flow parameters, on the device (2026-11-10; step 1 done)

OLF applies a contingency's `ContingencyLoadFlowParameters` (distributed slack, area interchange control, balance type,
outer loop names) to the context before its simulation (`AbstractSecurityAnalysis`, `applySpecificContingencyParameters`).
The batch precomputes every contingency with the run's parameters, and nothing in this repo reads the extension except
`ReplayedLossDistribution`, which then falls back to OLF's distribution. Measured on ShuntNetworkFactory with `l1_BIS`
(continuous shunt control), `l1_BIS` carrying outer loops `[DistributedSlack, ReactiveLimits]`: the device answers it
with the shunt loop, b3 390.929 kV against OLF's 395.710 kV, and reports it. The RTE contingency list carries no such
extension, so network.xiidm is not affected.

**Decided (2026-11-10): on the device, not the CPU.** What OLF changes for such a contingency is its outer-loop list
only (`AcSecurityAnalysis.applySpecificContingencyParameters`: the loops rebuilt from the run's parameters and the
overrides), for the contingency and, with the default scope `CONTINGENCY_AND_OPERATOR_STRATEGY`, for its strategies;
`CONTINGENCY_ONLY` puts the run's loops back before the strategies. The device's whole loop configuration is a pure
function of a loop list (`GpuSaRunScope.evaluate`), the run-wide batch settings are read nowhere but at the precompute
call, and the loss replay reads its participation factors from the extraction, which the distributed-slack parameters
key. Plan:
1. Group the contingencies by their overrides; a group's loops are OLF's own (`applySpecificContingencyParameters` on
   the context's parameters, put back after, as OLF's resetter does), with the generic contingency parameters
   (`startWithFrozenACEmulation`) applied the way `applyGenericContingencyParameters` does. A group's scope keys the HVDC
   freeze on its loops alone: an explicit list without `FreezingHvdcACEmulation` does not freeze, where the run's scope
   adds the freeze from the analysis flag because the run's list is still the base one. Its balance type is the
   override's.
2. Precompute every group in the same pass (`GpuBatchPrecomputer`): one walk over the contingencies in OLF's order (the
   zero-impedance tree replay depends on it), one batch and one solve per group with its own settings, one state sink
   and one set of result maps. A group whose loops the device does not run stays on the CPU, as the whole run would.
3. A pair takes its contingency's group (default scope). `CONTINGENCY_ONLY` with strategies: the pairs run with the
   run's loops but start from the parent's device run (state, reactive-limit states, distribution anchors), captured in
   another solve; first step keeps those contingencies on the CPU, second step carries the parent captures across.
4. Result time: `pairDistributedOnDevice` reads the pair's group distribution, not the run's flag.

**Step 1 (2026-11-10).** Done as planned: `GpuSaRunScope.evaluate` takes a group's balance type and keys its HVDC freeze
on its own loops (`contingencyLoops`); `GpuBatchPrecomputer` takes `SolveGroup`s (one `SolveSettings` each, the run's
first) and solves them one after the other into one sink and one set of results; `GpuBatchedAcSecurityAnalysis.solveGroups`
builds them through OLF's own `applySpecificContingencyParameters`; `pairDistributedOnDevice` reads
`GpuSaRunState.ownDistributedSlack`. `GpuSaContingencyParametersTest`, each at 1e-12 with the contingency on the device,
each failing without the change: own loop list (ShuntNetworkFactory, OLF holds SHUNT where the run's shunt loop moves
it), distributed slack off and load-based balance on B2-G's 40 MW (42.67 MW distributed by the old device run against
0 and 37.87), a strategy under the default scope (pair in the contingency's solve), a strategy under `CONTINGENCY_ONLY`
(on the CPU, below), an explicit list without `FreezingHvdcACEmulation` on an HVDC network (branch flows compared too:
both buses hold their voltage). Gates: `GpuSa*Test` 297/0/2 on cuDSS and on our LU, full suite 520/0/6 with checkstyle.

That last case found a device bug of its own: the HVDC controller tables were built only when the freeze or limits loop
ran, and the slack check counts a link's flow at the slack bus from them (`hvdcSlackBalanceBatch`). With neither loop
the link's 2.4 MW stayed out of the check, and the device distributed 2.6 MW where OLF distributes none. OLF's default
loop list always holds the limits loop with AC emulation on, so it takes an explicit list (per contingency, or the run's
`outerLoopNames`). The tables are now built whenever the pack holds an AC emulation link, in the batch
(`GpuBatchedSecurityAnalysisSolver`) and the base load flow (`GpuNewtonRaphson`); without the freeze loop nothing is
frozen and the device's freeze phase finds nothing to release.

**Step 2, not started: `CONTINGENCY_ONLY` with strategies.** The contingency runs with its own loops, its strategies with
the run's from the contingency's state. Today such a contingency and its strategies run on the CPU (correct, measured).
The pair phase seeds a pair from its parent's result row and six captures (`ParentCaptures`: reactive-limit states and
target Q, distribution targets, generator moves, exhausted start, absorbed slack), all keyed by the parent's index in
the same solve. Carrying them across solves is enough when the override leaves the distribution alone (an outer loop
list with the run's distribution). An override that changes the distribution (distributed slack, balance type, area
interchange) gives the parent another extraction with other participants, so its captures do not apply to the pair's:
the pair would have to start from the parent's full targets, which the pair phase has no path for.

### #88 — FIXED: the targeted restore left the device's control state variables in the network (2026-11-10)

Issue a of `docs/device-final-results-plan.md` §7 (branch `device-final-results`). After a contingency the device
answered, `restoreBaseState` restores only the elements the contingency touched, and the bus voltages lazily. When the
base system holds a SHUNT_B, BRANCH_ALPHA1 or BRANCH_RHO1 variable (continuous shunt, phase or transformer voltage
control), `GpuPostContingencyState` writes the state back through `AcSolverUtil.updateNetwork`, which also writes the
controller shunt's susceptance and the branch's phase or ratio override, and nothing restored them. A later contingency
left to the CPU reads them in `initStateVector`, and holds them as targets where its control stays off. OLF's full
restore puts them back.

Measured on ShuntNetworkFactory with `l1_BIS` (twice `l1`'s impedance), continuous shunt control: `l1`'s outage
dispatches SHUNT on section 1, the base on section 2.
- `l1_BIS` on the CPU with the shunt loop: the leak changes only Newton's starting point, and the continuous solve lands
  on the same bits.
- `l1_BIS` on the CPU without the shunt loop (its own outer loops, #89): Newton holds the leaked susceptance, and the
  GPU arm distributes 1.65 MW of slack where OLF distributes none.
- The ratio and phase overrides equal the tap's value while the device leaves the tap where it was, and a branch whose
  tap the device moved was already restored with it. No effect found; they are restored anyway.

The targeted restore now also restores the buses of the controller shunts with a SHUNT_B variable and the branches with
a BRANCH_ALPHA1 or BRANCH_RHO1 one, taken once per extraction from the base system's variables
(`controlStateElements`). `GpuSaShuntControlTest.cpuContingencyAfterADeviceOneHoldsTheBaseSusceptance` fails without it
(distributed active power of `l1_BIS`: 0 against -1.65 MW). The RTE parameters switch none of these controls on, so
rte6515 and network.xiidm hold no such variable and nothing changes there.

Two things seen on the way, not acted on:
- With both contingencies on the CPU route, the GPU provider still differs from OLF's provider by one ulp on `l1_BIS`
  (b3, 5.7e-14 kV), with the same network state at the start of `l1_BIS` in both GPU runs. Not the leak; probably the
  base case, which the GPU provider solves on the device (not verified).
- OLF's restore puts back a controller shunt's susceptance but not its controllers' section positions: in INCREMENTAL
  mode its results depend on the order of the contingencies (#90).

### #87 — operator strategies: an unrealistic pair is never answered (b); a flow condition after a device strategy (c, FIXED) (2026-11-09)

Issues b and c of `docs/device-final-results-plan.md` §7 (branch `device-final-results`).

**b — not reachable, no change.** The host builds a device-answered pair's result as CONVERGED without the realistic
voltage check the contingency path applies (`updateBusesFromState`). No unrealistic pair ever gets there. The driver
checks the band where `AcloadFlowEngine` does: at the end of the last loop able to fix an unrealistic state
(`i == realisticFixingIdx`), after every later solve, and after every solve when there is no fixing loop. Those points
always include the final state. `DeviceLoopTables.install` switches the check on for every context the device drives,
the robust retry's too. Such a pair is rejected and the CPU replays it. Measured on IEEE 14, with 20 MVar more at B14
after L2-5-1 and a minimum realistic voltage of 1.0 pu (B14 at 0.974 pu, the contingency alone at 1.01 pu or above):
both attempts are rejected with `unrealistic-state`, and the CPU reports FAILED like OLF.
`GpuSaOperatorStrategyTest.strategyWithAnUnrealisticStateFailsLikeCpu` keeps it that way, with the contingency alone
on the device.

**c — FIXED.** OLF evaluates strategy k's condition on the network strategy k−1's load flow left: its restore puts back
the network's elements, not the state vector. A branch or three-windings transformer threshold therefore reads every
flow of k−1's state. A device-answered k−1 refreshed only the flows its own detection reads (`strategyFlowReadSet`),
so k's condition read the post-contingency flow of any other branch. Now a pair refreshes every flow when another
strategy on its contingency has a stage conditioned on a flow (`anotherStrategyReadsFlows`). The contingency itself
already refreshed them all when a strategy targets it. The test is
`GpuSaOperatorStrategyTest.thresholdConditionAfterADeviceStrategyReadsThatStrategysFlows`. PS1's tap raises L1 over the
second strategy's threshold, and L1 carries a limit far above its flow, so the screen leaves it out. Before the fix the
CPU reported 2 strategy results and the GPU 1, with all three scenarios on the device.

**f — no visible effect, left as is.** The HVDC action (`LfHvdcAction`, AC emulation off) sets the converter target P
from `getP1().eval()`. On the device path that reads the state vector as the previous injection left it, where OLF
re-initializes it from the restored network first. The device composes the pair itself, nothing in the result reads the
converter target P, and the full restore that follows a non-tap strategy puts it back.

### #86 — FIXED (in OLF): the continuous phase control's target P kept the initial tap's admittance (2026-11-09)

Found while proving the device tap write-back (branch `fix-device-tap-moves`): the existing phase-control fixtures
never move a tap after a contingency (PS1 has three taps; OLF keeps tap 1 for every regulation target from 40 to
120 MW), so PS1's phase tap changer was rebuilt with 21 steps of one degree (alpha -10..10, R = 0.3·(t+11) %,
X = 3·(t+11) %, rho 1), ACTIVE_POWER_CONTROL at 83 MW with a 1 MW deadband, plus the `L1_BIS` clone of `L1`.
In CONTINUOUS_WITH_DISCRETISATION mode, the device ended elsewhere than OLF after `L1_BIS` whenever PS1 STARTED on a tap
other than the one the base load flow rounds to:

| initial tap | base rounds to | OLF's continuous a1 after `L1_BIS` | the device's |
|---|---|---|---|
| 7 | 7 | 0.1155236777090457 (tap 7) | the same, every digit |
| 6 | 7 | 0.11068574169440427 (tap 6) | 0.1155236777090457 (tap 7) |
| 0 | 6 | 0.0844781497739782 (tap 5) | 0.11384603866416136 (tap 7) — bus B2 383.1265 kV against OLF's 383.6706 |

**Cause, in OLF.** The phase control's `BRANCH_TARGET_P` equation holds a flow term of its own
(`AcEquationSystemCreator.createTransformerPhaseControlEquations` builds a new `ClosedBranchSide1ActiveFlowEquationTerm`),
and `AbstractBranchAcFlowEquationTerm` copies y, ksi, g and b from the pi model at construction.
`AcEquationSystemUpdater.onTapPositionChange` refreshed only the branch's six closed terms (`getClosedP1..I2`, which the
vectorized creator does not even need). A security analysis runs every post-contingency load flow on the base load flow's
equation system, so after the base rounding each continuous solve held the controlled flow at its target with the
INITIAL tap's admittance while the bus equations used the rounded tap's. The device uses the current tap's in both. OLF
alone shows it: PS1's flow after `L1_BIS` was 79.92 MW from initial tap 6 against 85.23 MW from initial tap 7, though
the base load flow rounds both to tap 7. The same staleness hits every other flow term created beside the six: a
generator remote reactive power control's `BRANCH_TARGET_Q`, and the additional closed P/Q terms (the reactive keys).

**Fix:** fork commit `61476b592` on `olf-local-patches` (pushed to `gb`). For upstream, the same fix on upstream main is
branch `refresh-flow-terms-on-tap-change` (`c567a6597`, pushed to `gb`, signed off): upstream main keeps a
`modifiedTapImpedance` guard and a `notifyEquationTermConstantsChange()` call there, and the new loop sits before the
notification. On upstream main the test fails the same way without the fix, and the suite runs 1616 tests with 0
failures and 0 checkstyle violations.
`onTapPositionChange` also refreshes every closed flow term the equation system attaches to the branch
(`equationSystem.getEquationTerms(BRANCH, num)`). The fork's new
`OpenSecurityAnalysisTest.testContinuousPhaseControlAfterBaseTapRounding` fails without it (79.916 against 85.231 MW).
The fork suite runs 1529 tests with 0 failures and 3 skipped.

**Here:** `GpuSaPhaseControlTest.devicePhaseTapMoveIsInTheReportedBranchFlows` runs INCREMENTAL from tap 0 and
CONTINUOUS_WITH_DISCRETISATION from taps 0, 6 and 7. Every bus voltage and branch flow matches OLF at 1e-12 kV / 1e-9,
with the device's tap moves written back. The RTE parameters leave phase-shifter regulation and transformer voltage
control off, so rte6515 and network.xiidm never move a tap: their CPU references are unchanged. (A trap met on the way:
`@EnumSource` runs the modes in the enum's declaration order, CONTINUOUS_WITH_DISCRETISATION first, whatever order
`names` lists them in.)

### #85 — GPU sensitivity analysis provider: step 1, the AC load flows on the device (2026-11-09)

**Done (branch `gpu-sensitivity`), not yet run on a device.** `GpuSensitivityAnalysisProvider` (`sensi.gpu`,
`@AutoService`, name `OpenLoadFlowGpu` like the load flow and security analysis providers) extends OLF's
`OpenSensitivityAnalysisProvider`. An AC run takes a JSON copy of the caller's parameters, its load flow parameters
those of a GPU load flow (`GpuLoadFlowProvider.withGpuSolver`, made public for it: `GPU_NEWTON_RAPHSON`, plus OLF's
parameters set by name), and `threadCount` 1 (concurrent native solves on one device are not supported, open bug (1)
in CLAUDE.md); a DC run is OLF's. Its load flow provider is `OpenLoadFlowGpu`, so pypowsybl's `provider_parameters`
for the run's load flow set the GPU solver's and OLF's.
- **Why it is sound without new device code:** OLF's sensitivity analysis runs every load flow through
  `getSolverFactory().createLoadFlowEngine(context)` (the fork's 7a795ca), then computes the sensitivities on
  `context.getJacobianMatrix()`. The GPU solver writes its state back with the full `StateVector.set` (the Jacobian is
  notified, unlike `setValuesOnly`), and its reactive-limit decisions with OLF's own bus calls
  (`freezeGenerationTargetQAndDisableGeneratorVoltageControl`, `setGeneratorVoltageControlEnabledAndRecomputeTargetQ`),
  so OLF's equation system, which the Jacobian is built from, swaps the voltage-target and Q equations as OLF's own
  ReactiveLimits loop would. `GpuSensitivityAnalysisProviderTest#acSensitivitiesAreOlfsOnTheCpu` (GPU-gated) checks it:
  IEEE 14 with bus 8's condenser held at +-6 MVar (PQ at its limit), two contingencies, every sensitivity and function
  reference against OLF on the CPU, and the condenser's target-voltage sensitivities exactly 0 on both (OLF fills that
  right-hand side only while the voltage-target equation is active).
- **Found on the way, fixed in the security analysis provider too:** powsybl collects the parameter serializers of
  every provider by extension name (`Collectors.toMap`, no merge), so a provider inheriting OLF's serializer makes the
  extension name a duplicate key. With the olf-gpu jar on the classpath, every JSON write and read of
  `SecurityAnalysisParameters` failed (`Duplicate key open-security-analysis-parameters`), and the new provider would
  have done the same for sensitivity parameters. Both GPU providers now register no serializer
  (`getSpecificParametersSerializer` empty: their specific parameters are OLF's, which OLF's provider serializes);
  regression tests in `GpuSecurityAnalysisProviderTest` and `GpuSensitivityAnalysisProviderTest`.
- **Next steps:**
  1. Run the gated test on a device and time a run against OLF's on the CPU (each load flow is a batch-of-one device
     solve, with a re-extraction after each contingency's topology change).
  2. The sensitivity solves on the device. OLF's forward sensitivities are `solveTransposed` on its stored `M = Jᵀ`,
     i.e. `J⁻¹` on the many right-hand sides of the factor groups: a plain solve with our true-CSR `J`, available with
     cuDSS (several right-hand sides on one factorization) and with our batched LU. The adjoint (`runAdjoint`) is the
     other direction, which the transposed solve of #40 / PR 25 provides on our LU.
  3. Batched contingencies: the security analysis's batch machinery solves the post-contingency load flows together;
     the sensitivity solves then run per scenario on its factors.

### #64 — FIXED: a re-homed member's reactive limit read from the wrong row (2026-09-30)

Worst distributed-active-power difference **1.328 MW -> 7.5e-09 MW** on the contingency, and
the scope's worst **1.328 MW -> 1.398e-03 MW**. One missed reactive pin was worth 1.328 MW.

`BUS-6113_BBS_1` islands VL-6113_0 and re-homes its voltage control to VL-6113_1, which becomes
a plain local PV bus (`reHomePv bus=VL-6113_1 rlEntry=969 baseLimitType=0`). OLF pins it MIN_Q
at `q=-3.231385934` against a `-0.774` limit. The device never pinned it.

Because VL-6113_1's reactive injection is accumulated by `fillRemoteMemberQBatch` at `rlQrow` —
for a relocated child its parent coupler's `DUMMY_Q` row, NOT its own V row — and that pack is
STRUCTURAL: it keeps writing there even in a scenario that pins the dummy to a `DUMMY_TARGET`
identity. `reactiveLimitsBatch` used ONE index for both the q READ and the pin WRITE, and its
`mode[relocIdx] == 0 ? ridx : relocIdx` fallback sent the READ to the bus's own row exactly when
the coupler is outaged — where nothing accumulates q. So it read q ~ 0, inside any limit.

`qBuf` is an accumulator SEPARATE from the equation system, so reading it at a row whose
equation is an identity is perfectly valid; only the pin WRITE belongs on the live row. The fix
splits the two (`qIdx` for the read, `idx` for the writes).

**`GpuAcOuterLoopPacks`'s own `rlQrow` comment had already recorded this failure for the base** —
"filling the own row left the check reading ZERO for every relocated member, so it could never
pin one ... the device's base stopped after its first ReactiveLimits pass where OLF takes three"
— and the kernel reintroduced it PER SCENARIO. When a comment names a failure mode, check the
other places that make the same choice.

**How a missed pin became 1.328 MW.** Both arms agree to 9 digits through two slack checks
(`-0.001506610` vs `-0.001506609759`). The missing pin then leaves the THIRD check at
`0.013280180` where OLF measures `0.006846832` — straddling the 1 MW gate — so the device runs a
whole extra distribution OLF never runs: `dsPlaced` 2.178895055 then 2.192175235 against OLF's
single 2.178895055. The DAP difference was the SYMPTOM; the pin was the cause. A DAP gap that is
an exact multiple of a gate crossing is worth reading as "one arm took a pass the other did not"
before looking at the water-fill arithmetic.

| | before | after | OLF |
| --- | --- | --- | --- |
| RL switches / pins | 4 / 19 | **5 / 20** | 5 / 20 |
| DS passes fired | 2 | **1** | 1 |
| contingency DAP diff | 1.328019 MW | **7.5e-09 MW** | — |

Suite 177/177 + GpuLoadFlow/GpuFullNewton/GpuAcStructures/GpuLineSearch green; checkstyle
unchanged at 220.

### #66 — The scope's last disagreement: the two arms solve slightly DIFFERENT systems (2026-09-30)

OPEN, narrowed to one sentence and with the instrument that makes it chaseable now in place.
Worst 1.398e-03 MW at default criteria, over the `assertTrue(worstDistributed < 1e-3)` assertion.

**The tail is two components** (`OLF_GPU_DBG_CONV_EPS`, tightening BOTH arms; three eps points,
because two cannot tell slack from a floor):

| | eps=1e-4 | eps=1e-8 | eps=1e-10 |
| --- | --- | --- | --- |
| BUS-4467_BBS | 9.564e-04 | 8.464e-04 | **8.464e-04** |
| BUS-5250_BBS | 1.398e-03 | 4.749e-05 | **4.749e-05** |

Above eps=1e-8 it is convergence slack (cuDSS and KLU satisfy the same `‖F‖₂` test at different
iterates). **Below it there is a hard floor that does not move**, and the floor is the real bug.

**What the floor is NOT** — three hypotheses of mine, each killed by a measurement:

- *the [[#64]] gate signature*: no — both arms fire exactly ONE pass, so the AMOUNT differs.
- *chunk composition*: no — byte-identical at `-Dolf.gpu.sa.maxBatch=512`.
- *a differing saturation CLAMP* (my story for eps-independence, since a clamped generator
  contributes its exact limit): **no** — the per-generator join shows the saturation sets AGREE
  and 620 of 670 generators differ by 3-6e-09 pu each, summing to 9.553e-06 pu = the whole gap.
  A PROPORTIONAL spread, not a clamp.
- *a differing pre-distribution seed* (host-side, hence eps-independent — a good fit on paper):
  **no** — the seeds match, device `absorbed=17.248640` vs OLF `previousMismatch=17.248639569`.

**What it IS.** The device's MEASURED slack mismatch after the solve is `0.018464` where OLF's is
`0.018454123` — larger by ~9.6e-06 pu, which is exactly the gap, and it is then spread
proportionally over every participant. Converged states that differ by ~1e-05 pu and do NOT move
when the criterion tightens 10^4x are not converging differently: **the two arms are solving
slightly different systems.** Small, genuine, and localized somewhere in this contingency's
equation set.

**The STRUCTURES agree exactly — checked, and it removes the last structural suspicion.** Device
first fill of slot 585: `n=14504` rows, 985 identity (mode 0) + 13499 power (mode 1) + 20 own-Q
(mode 4). OLF: 14494 ACTIVE equations = BUS_TARGET_P 6880 + BUS_TARGET_Q 5578 + BUS_TARGET_V 980
+ DISTR_Q 323 + ZERO_V 366 + ZERO_PHI 366 + BUS_TARGET_PHI 1. The difference is
`14504 - 14494 = 10 = 2 x 5`, and the contingency disables exactly 5 buses — the device keeps a
fixed row set and freezes an islanded bus's two rows as identities where OLF removes the
equations. `islandedDsParticipants=2` is consistent too (only 2 of the 5 carry participating
generators). So the row sets are right and the participant sets agree.

**What `EQ_DUMP_CTG`'s F0 section actually is** (I first called it "stale by construction" and
that was WRONG — corrected in a92dcc8's follow-up): the POST-contingency system evaluated at the
PRE-contingency state. The dump runs before the solver touches x, but `lfContingency.apply()` has
already deactivated equations, which invalidates the vector, so `AbstractVector.getArray()`
RECREATES it against the current index. Proof from the dump itself: `fx0.length` = 14494 =
exactly the active-equation count of its own post-contingency system section (independently
counted), not the base's. Not stale.

**It is still not comparable with a device fill's norm**, for a different reason: the device's
first fill is taken after ITS seeding (frozen islanded rows at flat V=1/phi=0, the pre-loss
target seeding) and after `applyRowMode` turns identity rows into `x - target`. OLF `norm2=774.03`
vs device `35.95`, and OLF's is dominated almost entirely by TWO equations — `distr_q1952/1953`
at 519.00 and 518.98 — so those norms describe two equations, not the network.

**Instrumentation done, and it answered:** F0 now writes `f` and `target` separately. The re-dump
says **519.0048 is `f`, with `target = 0.0`** — same for 518.98, and for the next three
(118.32 / 104.66 / -100.04, all DISTR_Q, all targets zero). A genuine evaluation, not a target
artefact, so `dummy_q9161`/`9162` hold magnitudes around 780 pu — not a physical coupler flow.

**Why, and it is the third and correct reading of this instrument.** `BusState` saves angle,
voltage, `generationTargetQ`, shunt B/G … and NOT `dummy_p`/`dummy_q`. Those are equation-only
variables of the zero-impedance couplers, so `NetworkState.restore()` between contingencies does
NOT restore them: they carry whatever the PREVIOUS contingency's solve left. At the dump point the
vector is fresh and the index is current, but the STATE is partly stale — precisely in the
variables these DISTR_Q equations reference. (First reading: "stale equation vector" — wrong,
mechanism. Second: "legitimately large" — wrong, it is not the base state. Third: stale
equation-only STATE variables.) Harmless for OLF, whose solver re-converges them; it only means
F0's dummy-referencing entries are not a "starting residual".

**HYPOTHESIS for this bug, UNTESTED, but it fits every measured property.** The device tiles the
BASE state to every scenario, so it starts each one from the base `dummy_q`; OLF starts from the
previous contingency's. If the coupler dummies are UNDERDETERMINED — a null space, which a
coupler loop gives — both arms converge, but to different points in that null space. Physical
quantities would still agree (and they do: BUSDIFF 0 scope-wide, bus voltages exact), while a
DISTR_Q equation that REFERENCES `dummy_q` enforces a slightly different reactive split, moving
the controllers' q, the losses, and hence the slack mismatch by ~1e-5 pu. That would be:
eps-independent (it is not a convergence depth), proportional across participants (it moves the
sharing, not one unit), invisible in bus voltages, and identical under a chunk-size change — which
is the full list of properties measured for BUS-4467_BBS.

**TESTED 2026-10-01 — the hypothesis is DEAD.** `OLF_DUMMY_DUMP` (new, fork 86607e525) writes the
CONVERGED equation-only variables after each post-contingency solve. For BUS-4467_BBS: 732
entries, mean |v| = **0.3506 pu**, max **11.63** (`DUMMY_P 9384`). Nothing near 780 — so that
magnitude was ENTIRELY the stale pre-solve state, and the converged dummies are ordinary. Two
independent reasons finish it off: determined variables cannot differ beyond convergence
tolerance, and cuDSS factors the uniform batch WITHOUT pivoting, so a rank-deficient dummy
subsystem would fail to factor rather than settle on an arbitrary null-space point — these
scenarios factor cleanly, so `ZERO_V`/`ZERO_PHI` plus the balances determine the dummies.

(Run limitation, for the next reader: only ONE dump section appeared, the CPU arm's. For a
device-solved contingency the GPU provider returns from the precomputed state and never calls
OLF's `runPostContingencySimulation`, so the device's converged dummies do NOT come out of this
dump. Not worth another run given the two reasons above.)

**Eliminated so far, each by measurement, not argument:** the [[#64]] slack-gate signature; chunk
composition; a differing saturation clamp; the pre-distribution seed; row/equation structure; the
participant set; the dummy null space. SEVEN.

**ANSWERED 2026-10-01: the two arms converge to DIFFERENT EQUATION SETS on a coupled group.**

**The principle that should have been applied on day one** (user, 2026-10-01): *"if the two
systems do not agree at machine precision then the two equation systems are not the same — it is
as simple as that."* A gap above machine precision is NOT a property of the solve. Every
"same system, different solution" story is excluded a priori, and this entry burned most of a
session on four of them — convergence slack, a coupler-dummy null space, a saturation-clamp
quantum, a `P_RESIDUE_EPS` floor — each costing a ~20-minute scope run to kill. The first
eps-squeeze that did not collapse the gap was already the proof. Go find the equation.

**What it is.** The paired dump (`OLF_VAR_DUMP` + `OLF_GPU_SA_VAR_DUMP`, same point, keyed by
element id, filtered by `OLF_TRACE_CTG`) confines the whole difference to the coupler dummies:

| variable | differing | worst abs diff |
| --- | --- | --- |
| **DUMMY_Q** | 331 of 366 | **8.410878e-03** |
| DUMMY_P | 271 of 366 | 5.637924e-07 |
| BUS_V | 5349 of 6881 | 2.655704e-07 |
| BUS_PHI | 6880 of 6881 | 8.231543e-08 |

OLF is properly converged there — `‖f − target‖₂ = 5.822350e-07`, worst residual 4.309392e-07 —
which also confirms the retraction below. And the governing equations at CONVERGENCE name the
defect:

```
EQ BUS_TARGET_V VL-2669_0   resid 0
EQ DISTR_Q      VL-2669_1   resid -1.8e-16
EQ DISTR_Q      VL-2669_2   resid -1.8e-16
EQ DISTR_Q      VL-2669_3   resid -1.8e-16
EQ DISTR_Q      VL-2669_4   resid -1.8e-16
```

In OLF's converged system those buses are on **DISTR_Q** (reactive SHARING). Pre-solve they were on
`BUS_TARGET_Q` with targets 0.0257 / 0.0257 / 0.02 / 0.0257, so OLF's solve SWITCHED them:
DISTR_Q 323 -> 332 (+9), BUS_TARGET_Q 5578 -> 5571 (-7), BUS_TARGET_V 980 -> 978 (-2).

Both arms' numbers then follow exactly:
- OLF's dummy increments are all EQUAL (-0.0241962788) — enforced equal sharing under DISTR_Q;
- the device's are -0.0257 / -0.0257 / -0.0200 — the individual BUS_TARGET_Q targets.

**So the device does not reproduce OLF's post-contingency re-activation of DISTR_Q on that coupled
group.** The sharing is never enforced, each member sits at its own reactive target, the dummies
differ by 8.4e-03 pu, and DISTR_Q's dependence on `dummy_q` carries it into the reactive split,
the losses and the 9.553e-06 pu slack mismatch. That accounts for every measured property:
eps-INDEPENDENT (a different equation set, not a convergence depth), PROPORTIONAL across
participants, INVISIBLE to BUSDIFF, chunk-size invariant.

**CHECKED, and it refuted the mechanism I had inferred.** The device's rows for the chain buses
hold, at slot 585:

```
BUS_V VL-2669_0  row 6048  mode 0  target 0.62     (matches OLF's BUS_TARGET_V)
BUS_V VL-2669_1  row 6050  mode 1  +1*v(_0) -1*v(_1)   <- ZERO_V, F=0
BUS_V VL-2669_2  row 6052  mode 1  +1*v(_1) -1*v(_2)   <- ZERO_V, F=0
BUS_V VL-2669_3  row 6054  mode 1  +1*v(_2) -1*v(_3)   <- ZERO_V, F=0
BUS_V VL-2669_4  row 6056  mode 1  +1*v(_3) -1*v(_4)   <- ZERO_V, F=0
DUMMY_Q COUPLER_1 row 14113 mode 4 target +0.0257  -dq1 + dq2
DUMMY_Q COUPLER_2 row 14115 mode 4 target +0.0257  -dq2 + dq3
DUMMY_Q COUPLER_3 row 14117 mode 4 target +0.0200  -dq3 + dq4
DUMMY_Q COUPLER_4 row 14119 mode 4 target +0.0257  -dq4
```

The V rows hold ZERO_V (the relocation's own design: couplers' ZERO_V/ZERO_PHI move to the child's
rows), and the children's reactive balances sit on the coupler DUMMY_Q rows in **mode 4 — own Q
balance, i.e. voltage control OFF** — against their individual targets. OLF at convergence has
**DISTR_Q** there (target 0, residual -1.8e-16): control ON, reactive power SHARED. **So the two
arms solve different equations for that group.** That part is settled and needs no more
measurement; it is the 8.4e-03 pu dummy gap and the 9.553e-06 pu slack mismatch.

**RETRACTED (3): "a missed PQ->PV switch-back".** OLF's whole RL activity for BUS-4467_BBS is
`PV->PQ` on VL-6490_0 / VL-6489_0 / VL-6113_1 / VL-6231_0 / VL-6231_1 and one `PQ->PV` on
VL-6515_0. **Nothing touches VL-2669.** So the group's DISTR_Q does not come from the
reactive-limits loop and there is no switch-back to restore. (681 `PQ->PV` switches occur
elsewhere in the run, so the trace works.)

**RETRACTED (4): "OLF's solve switched 9 DISTR_Q on (323 -> 332)".** Those counts come from
DIFFERENT ARMS. `eq2.post` was written by the GPU provider at its own entry, in a run with
`OLF_GPU_SA_GPU_ONLY=1` (CPU arm given an empty contingency list); the converged `EQ` section fires
in `AbstractSecurityAnalysis`, which the GPU provider OVERRIDES and returns from early for a
device-solved contingency — so that section is the CPU arm's. Pre-solve on one arm against
post-solve on the other says nothing about what either solve did. Fifth point-in-time/cross-arm
error of the session, and the first to manufacture a MECHANISM rather than just a wrong number.

**ROOT CAUSE, and it is in the BASE, not in any contingency.** `OLF_GPU_TARGET_ROW` says the
device's mode-4 own-Q state on those coupler DUMMY_Q rows is **STRUCTURAL** — `baseMode=4`,
`base=0.0257` — decided at extraction, with no per-scenario override. And OLF's OWN base dump
(`OLF_EQ_DUMP`) has **DISTR_Q** on the same buses, a five-member group sharing with 1/5 keys:

```
eq col=646 type=DISTR_Q elementNum=3025 id=VL-2669_1          (…_2, _3, _4 likewise)
distr_q3025 = -0.8*(-1.0*dummy_q9212) + -0.8*dummy_q9213
            + 0.2*ac_q_closed_1(v3024,v3029,…) + 0.2*dummy_q9212 + 0.2*(-1.0*dummy_q9213) + …
```

So **the device never builds the DISTR_Q sharing equation for a coupler-relocated group; extraction
puts each child's OWN Q balance on that row instead.** The two arms' systems differ in the BASE.

**Extent — this is not a corner case:**

| | |
| --- | --- |
| OLF base DISTR_Q equations | **345** |
| of those, referencing a coupler dummy | **341** |
| distinct coupler dummies involved | 344 |
| DUMMY_Q differing in the one-contingency join | **331 of 366** |

Essentially EVERY DISTR_Q member in rte6515 is a coupler-relocated child. (Verified row by row for
the VL-2669 group; the counts and the 331/366 corroborate the generality.)

**Why it was invisible for the whole project.** `ZERO_V` pins every bus of a coupler chain to the
head's voltage whatever the reactive SPLIT among them, so the voltages agree; the group's total
reactive output is fixed by the voltage target, so external flows barely move (base active power
agrees to 7e-09 MW). And `BUSDIFF` — the basis of every "bus voltages exact scope-wide" claim in
this file, base included — compares BUS VOLTAGES ONLY. The reactive split was never checked by
anything until the paired variable dump existed.

**This answers the design question asked earlier in the session** ("why don't we have DISTR_Q
equation alternates for relocation?"). The 2-way stock twin (`dqAlt`/`dqSel`, [[#65]] and the
2026-09-29 stock-placement work) covers "the coupler died, move the equation back to the bus's own
row". It does NOT cover "this member's row IS a relocated dummy that must host a GROUP equation" —
so the sharing has nowhere to live and extraction falls back to the own-Q balance. That is very
likely also what draws the `PLAN:VC_REMERGE_MULTI` boundary (47 of the 55 plan refusals).

### Implementation plan (drafted 2026-10-01, NOT started)

**A. What OLF does — verified in `AcEquationSystemCreator`.**
`createGeneratorReactivePowerDistributionEquations` walks `getMergedControllerElements()` and for
each controller i creates `DISTR_Q` **on bus i**: `(qPct_i − 1)·q_i + Σ_{j≠i} qPct_i·q_j`, where
`q_x = createReactiveTerms(bus x)`. For a coupler-connected bus those reactive terms ARE the
`DUMMY_Q` variables, which is why `distr_q3025` references `dummy_q9212..9215`. So the equation is
owned by the controller bus, pairs with that bus's `BUS_V`, and spans EVERY member's dummies.
`recreateReactivePowerDistributionEquations` rebuilds the set when membership changes, and skips it
entirely for a local control.

**B. What the device does — verified.** Extraction DOES see the four DISTR_Q equations and assigns
them to the coupler dummy rows (`OLF_GPU_DISTRQ_DUMP`):

```
controller=VL-2669_1 eqCol=646 assignedRow=14113 ownVRow=6050   (…_2/_3/_4 -> 14115/14117/14119)
```

and those same rows carry `baseMode=4` with targets 0.0257/0.0257/0.0200/0.0257 — an own-Q
balance. Two equations, one row; mode 4 wins and the sharing is dropped, exactly as the extractor
comment states ("row mode 4 … the DISTR_Q emissions at this row gate off"). The child's own `BUS_V`
row is unavailable because the relocation puts `ZERO_V` there (verified from the matrix: row 6050 is
`+1·v(_0) −1·v(_1)`).

**C. CLOSED 2026-10-01 — both equations are ACTIVE at the device's extraction.** The new
`OLF_GPU_QROW_TRACE` says, in the SA run:

```
QROW_CLAIM row=14113 bus=VL-2669_1 busTargetQcol=11308 remoteCtl=true pilotSelf=false
           freeMember=false enrolled=false -> mode4=true
DISTRQ_ROW pass=13 controller=VL-2669_1 assignedRow=14113 enabled=true
```

`busTargetQcol >= 0` means `BUS_TARGET_Q` is genuinely ACTIVE for that bus (not a stale column:
`EquationSystemIndex.removeEquation` does `setColumn(-1)` immediately on deactivation), and the
same pass assigns that bus's `DISTR_Q` to the SAME row. In OLF's converged base only `DISTR_Q` is
active (`bus_target_q[3025..3028]` bracketed). So the device extracts from a state where the pair
is inconsistent with OLF's, and its mode-4 rule then picks the own-Q balance and drops the sharing.
(`assign()` does not throw, so only one of the two ever claims the row in `targetMap` — the other
is simply never built. The earlier phrasing "two equations, one row" was loose.)

**CANDIDATE FIX KILLED by the measurement that was meant to justify it (2026-10-01).** Extending
`OLF_GPU_QROW_TRACE` to print the merged control gives, at the extraction the batched SA uses:

```
row=14113 bus=VL-2669_1 busTargetQcol=11308 distrQcol=-1
          mainVc=VL-2669_0 mainMembers=5 enabled=FALSE
          ownVc=VL-2669_1 ownMembers=1 ownMerge=DEPENDENT -> mode4=true
```

Three readings, two of which retract earlier claims:

- **`distrQcol = -1`: DISTR_Q is INACTIVE there.** "Both equations are active at the device's
  extraction" (claim C, d8886c2) is WRONG — it compared the `DISTRQ_DUMP` of a BASE-ONLY run with
  the `QROW` trace of the SA run. Two runs again, the same error as F0/targets/dummies.
- **`mainVc=VL-2669_0`, `mainMembers=5`: the merge IS resolved.** `mainGenVc` is fine, so that lead
  (the DISTR_Q dump showing a 1-member dependent) was a dump artefact, not a defect.
- **`enabled=FALSE`: the control is genuinely DISABLED at extraction.** So BUS_TARGET_Q active +
  DISTR_Q inactive + control off is a CONSISTENT pinned state, and `mode[qr] = 4` is the CORRECT
  reading of it. The proposed `&& !mainVc.isControllerEnabled(bus)` guard is already satisfied —
  the change would have been a NO-OP, shipped on momentum, with the gap unexplained afterwards.

**So the question moves one step earlier, and the DISTR_Q-hosting gap is NOT what bites here.**
Why is that control DISABLED at the device's extraction when OLF's converged base has it ENABLED
(DISTR_Q active, `bus_target_q[3025..3028]` bracketed)? That is a BASE-level pin-state divergence
between the arms — the [[#64]] family (a wrong pin decision), not an equation-hosting problem.

**RESOLVED: there is NO base pin divergence, and `enabled=false` is not a pin.** Mapping the
device's 15 `anyBaseLt` entry indices to bus ids gives VL-6439_0, VL-6452_0, VL-4244_0, VL-6515_0,
VL-6003_0, VL-6027_0, VL-6095_0, VL-6106_0, VL-6130_0, VL-6157_0, VL-6173_0, VL-6194_0, VL-6226_0,
VL-6229_0, VL-6239_0 — **identical** to OLF's base `PV->PQ` set (14 in its first pass plus
VL-6106_0 in its second). The arms agree exactly on which controllers pin.

And VL-2669 has ONE RL entry — `idx=486 bus=VL-2669_0 gens=5 vcEnabled=true basePv=true`,
UNPINNED, at the pilot, covering all five generators. VL-2669_1..4 have NO RL entries. So the
`enabled=false` the QROW trace reports cannot be a reactive-limit pin: there is nothing there to
pin. (It is most likely the ordinary state of a merged DEPENDENT member, whose own control is off
while the MAIN owns the group — OLF still lists it in `getMergedControllerElements()` and still
gives it `DISTR_Q`.)

**What remains is purely the SNAPSHOT MOMENT.** In a BASE-ONLY run, extraction passes 13/14/15 see
`DISTR_Q` ACTIVE for those buses (`OLF_GPU_DISTRQ_DUMP`, `enabled=true`). In the SA run, the
extraction the batch uses sees `DISTR_Q` INACTIVE and `BUS_TARGET_Q` ACTIVE (`distrQcol=-1`,
`busTargetQcol=11308`). Same network, different moments: the batched SA's cached extraction is
taken at a state where the merged dependents' controls are off, so extraction legitimately records
an own-Q balance (mode 4) and the sharing equation is never built — and the structure then holds
that for every scenario.

**Fix direction (narrowed, still not coded):** make the batched SA's extraction come from the
SETTLED base state rather than an unsettled one — i.e. establish WHEN `GpuAcDataCache` captures it
relative to the base load flow's voltage-control update, and capture (or refresh) after. Note that
`refreshContextValues` already re-pushes VALUES into a shared structure; row MODES and the
equation set are structural, so a value refresh cannot repair this.

**THE FLIP, MEASURED IN ONE RUN (2026-10-01).** Both traces now carry the extraction pass number,
taken once at the top of `extractActivation`, and `OLF_GPU_QROW_TRACE` accepts a BUS-ID PREFIX so
the measurement no longer needs the exact full-scope row numbers. Full busbar scope, one run,
twelve extractions in one contiguous burst (no other output between them):

| | passes 1-6 (`DISTRQ_ROW`) | passes 7-12 (`QROW_CLAIM`) |
|---|---|---|
| `DISTR_Q` | EXISTS, `eqCol=646..650` | INACTIVE, `distrQcol=-1` |
| `BUS_TARGET_Q` | inactive (no claim line) | ACTIVE, `busTargetQcol=11308..11312` |
| the bus's `isGeneratorVoltageControlEnabled()` | **true** | **false** |

The two traces are MUTUALLY EXCLUSIVE across the 6/7 boundary, and that is itself the result:
`QROW_CLAIM` only fires inside `if (q >= 0)` (an ACTIVE `BUS_TARGET_Q`) and `DISTRQ_ROW` only for an
ACTIVE `DISTR_Q`. Both read the SAME per-bus flag — `GeneratorVoltageControl.isControllerEnabled(b)`
is literally `b.isGeneratorVoltageControlEnabled()` — so this is ONE flag flipping, not two
different questions.

**So passes 1-6 match OLF's converged base EXACTLY** (group controlling, sharing active) and the
device's own later extractions disable it. The snapshot-timing reading is confirmed, and sharpened:
the device is not extracting "too early", it extracts the RIGHT state six times and then something
turns the control off.

Also settled: `controlledBus=VL-2669_1 members=[VL-2669_1(on)]` printed with `qPct=0.200000000`
(= 1/5, a FIVE-member group's share) confirms the one-member listing is the bus's own DEPENDENT
control object — the dump artefact noted above, not a merge defect.

**ROOT CAUSE, MEASURED (2026-10-01). The disable is a TRANSIENT `MIN_REALISTIC_V` freeze, and
the device bakes the frozen window into the STRUCTURE.**

`QROW_CLAIM` now also prints the three disable paths' signatures, and the answer is unambiguous:

```
frozenQ=true  monitoringV=false  qLimit=MIN_REALISTIC_V  gens=1
```

So `ReactiveLimitsOuterLoop` froze these buses under its ROBUST-MODE UNREALISTIC-LOW-VOLTAGE rule
(`ReactiveLimitsOuterLoop:296-304`), not under a reactive-limit rule:

```java
if (robustMode && remainsPV && generatorRemoteController) {   // Q IS within bounds
    if (isUnrealisticLowVoltage(controllerBus)) {             // V < minRealisticVoltage * 1.02
        controllerBus.setV(1);
        buses.add(new ControllerBusToPqBus(..., QLimitType.MIN_REALISTIC_V));
```

With the RTE params (`minRealisticVoltage = 0.8`, `REALISTIC_VOLTAGE_MARGIN = 1.02`) the trip point
is **0.816 pu** — far tighter than the 0.5 default, and easy to touch at a 20 kV generator section
mid-iterate. `VL-2669` is a **20 kV** voltage level whose sections are coupler-joined, so each
section is a distinct `LfBus` and `isGeneratorRemoteController` is TRUE for `_1.._4`.

**This is why every pin comparison said the arms agreed.** A `MIN_REALISTIC_V` freeze is NOT a
Q-limit pin: it has `qLimitType=MIN_REALISTIC_V`, never `MIN_Q`/`MAX_Q`, and it is TRANSIENT —
`switchPqPv` clears it (`setQLimitType(null)`) when the voltage recovers. Comparing converged pin
SETS therefore could not see it, and neither could `RL_SWITCH`-vs-`anyBaseLt`.

**Ruled out by direct check, so these need not be re-explored:** `MONITORING_VOLTAGE` (the case has
6 SVCs with `standbyAutomaton`, all at ~63 kV, NONE at VL-2669 — and `monitoringV=false`);
`GeneratorVoltageControlManager.disableGeneratorVoltageControlsUnderMaxControlledNominalVoltage`
(it would freeze a 20 kV group, since `generatorVoltageControlMinNominalVoltage = 120`, but it is
built ONLY by `TransformerVoltageControlOuterLoop` and `transformerVoltageControlOn` is absent from
`rte-loadflow-default-parameters.json`, hence false, so that outer loop never runs); and
`minNominalVoltageRealisticVoltageCheck = 180` exempting the bus (it gates only
`AcloadFlowEngine.isStateUnrealistic`, the whole-state check — NOT the outer loop's per-controller
test, so a 20 kV bus is legitimately subject to the freeze).

**The cycle, and where the structure is captured.** One GPU-only full-busbar run, pass numbers on
both traces, phase markers interleaved by log line:

```
lines  67..114   passes  1-12   base LF: 1-6 CONTROLLING, RL freezes at 7, 7-12 FROZEN
lines 115..116   GPU CENSUS (cold) / "GPU precomputed on device: 6550 / 6667"
lines 117..      passes 13-24   the SA run: 13-18 controlling, 19-24 frozen -- the SAME cycle
```

Two complete 6-on/6-off cycles, and **the precompute's census line lands immediately after pass 12,
the end of a FROZEN window**. Whether `DISTR_Q` exists at all is STRUCTURAL — it is fixed when
`createStructure` runs — so a structure captured there carries own-Q balances for all 6550
scenarios, and the per-solve `extractRowMode` / `extractTargets` refresh cannot put the sharing
equation back. That is the whole of [[#66]].

**Fix, and why it is not a timing patch.** Chasing the snapshot moment (re-extract after the base
settles) is fragile: the cycle repeats inside the SA too (passes 13-24), so "settled" is not a
single instant. The principled fix is the one the code ALREADY implements for exactly this problem:
the CANONICAL, PIN-STATE-INDEPENDENT extraction behind `GpuAcRowIndexing.hasCandidateTable` —
"extracted CANONICALLY (pin-state independent: every controller's frozen q row suppressed...)" per
the comment in `GpuAcActivationExtractor`. A group with a candidate table already gets a structure
that holds BOTH shapes and lets the in-kernel mode scan choose per scenario.

**THE DEFECT, LOCATED (2026-10-01). The group HAS a candidate table; the canonical path is gated
off for RELOCATED members.** `candidateTableReason` (new, printed as `candTable=` in `QROW_CLAIM`)
says `YES` on all 24 claim lines — so the "does not qualify" guess above was WRONG, and the
`hasCandidateTable` clauses are all satisfied. The leak is one gate in
`GpuAcActivationExtractor.extractActivation`:

```java
int qr = stubBr >= 0 ? dummyQCol[stubBr] : qRow[bus.getNum()];
GeneratorVoltageControl mainVc = GpuAcRowIndexing.mainGenVc(bus);
if (stubBr < 0 && mainVc != null && GpuAcRowIndexing.hasCandidateTable(mainVc, es)) {
    qr = GpuAcRowIndexing.stateQRow(bus, vCol, stubBranchOfBus, dummyQCol);   // CANONICAL path
}
assign(es, targetMap, qr, q, AcEquationType.BUS_TARGET_Q, bus.getNum());
...
if (mode4) { mode[qr] = 4; }                                                  // pin-state dependent
```

`VL-2669_1..4` are coupler-relocated, so `stubBr >= 0`: the row is the parent coupler's `DUMMY_Q`
column (correct — relocation is structural) but the **`stubBr < 0` gate skips the canonical
pin-state-independent treatment for exactly these buses**, and `mode4` is then stamped from the
instantaneous pin state anyway. `mode4` is pin-state dependent for a non-pilot member because it
reads `!isFreeRemoteGenVcMember(bus)`, and "free" is *the first ENABLED member* — with all four
transiently frozen none is free, so all four get mode 4. That is precisely the leak
`hasCandidateTable`'s own javadoc forbids: "the pin state at extraction time must not leak into the
structure (see computeQRow)".

**TWO WRONG LEADS, KILLED BY MEASUREMENT — the row and the pack are both FINE.**

1. *The `stubBr < 0` gate is not the leak.* `stateQRow(bus, vCol, ziParentEdge, dummyQCol)` ALREADY
   applies the relocation to the carrier (`return parent >= 0 && dummyQCol[parent] >= 0 ?
   dummyQCol[parent] : vCol[carrier]`), and for a NON-PILOT member `stateQCarrier` returns the bus
   itself — so the canonical branch would compute the SAME row the gated branch already does
   (`dummyQCol[stubBr]`). The gate changes nothing for these buses.
2. *The DISTR_Q pack is not missing them.* `buildDistrQContributions` is EXISTENCE-keyed
   (`es.getEquation(gk, DISTR_Q).isEmpty()`) on the equation OBJECT, not the active index, and
   `OLF_GPU_FREECTL_TRACE` shows the objects SURVIVE the freeze:

   ```
   FREECTL pilot=VL-2669_0 members=[VL-2669_0, VL-2669_1, VL-2669_2, VL-2669_3, VL-2669_4]
           freeCtl=VL-2669_0
           olfHasDistrQ=[VL-2669_0=true, VL-2669_1=true, VL-2669_2=true, VL-2669_3=true, VL-2669_4=true]
   ```

   So the union pattern DOES hold the sharing slots, and `freeCtl` is already canonical
   (`controllers.get(0)`, the candidate-table branch). I had inferred OLF's
   `recreateReactivePowerDistributionEquations` would DELETE a frozen member's equation; measured
   false. (Also settled: the pilot `_0` IS one of its own controllers — a pilot-self group.)

**So the leak is ONLY the row MODE.** The sharing terms are packed and the row is right; what turns
them off is `mode[qr] = 4`, which the extractor comment describes as gating "the DISTR_Q emissions
at this row" off. And mode 4 is written into `baseRowMode` from the frozen instant, while
`GpuBatchChunkSolver` reads `rowModePerScenario[s] != null ? rowModePerScenario[s][r] :
baseRowMode[r]` — so every scenario without an explicit per-scenario mode inherits the frozen 4.

The canonical suppression ALREADY EXISTS but covers only the pilot:

```java
boolean enrolled = pilotSelf && pilotSelfRotationEnabled() && members.size() >= 2 && <all vCol >= 0>;
boolean mode4 = isRemoteGenVcControllerElement(bus)
        && (pilotSelf || !isFreeRemoteGenVcMember(bus)) && !enrolled;
```

`enrolled` requires `pilotSelf`, true only for `_0`. For `_1.._4`, `pilotSelf=false` so
`enrolled=false` and `mode4 = !isFreeRemoteGenVcMember(bus)` — and "free" is *the first ENABLED
member*, so with all four transiently frozen none is free and all four get mode 4.

**THE PLAN (the per-scenario machinery already exists).** The kernel does NOT need the row mode to
know a member's pin — it reads the REACTIVE-LIMIT state per (scenario, entry). From
`olf_gpu_nr_jni.cu` around the pilot-Q mode scan:

> OLF's own test for "this group has no controller": its ENABLED controllers
> (`VoltageControl.isControllerEnabled` -> `isGeneratorVoltageControlEnabled`). On the device that
> fact is the members' REACTIVE-LIMIT state, which is per (scenario, entry) and current from the
> seeding onwards — **unlike the row modes this kernel used to infer it from**, which at seed time
> are still ambiguous.

i.e. `dqMemberQrGroup[m]` / `dqMemberRlEntry[m]` + `limitType[s * nrl + re]`. That is the template,
and it already covers MEMBERS, not just the pilot.

**And for this group it is not merely a workaround — it is the CORRECT model.** `VL-2669_1..4` have
NO RL ENTRIES (measured: VL-2669's only entry is `idx=486 bus=VL-2669_0 gens=5`, at the pilot). A
`MIN_REALISTIC_V` freeze is NOT a Q-limit pin, so the device's RL pack does not and should not
represent it. Canonically-unpinned is exactly what OLF's CONVERGED base has.

**Steps:**

1. Give a candidate-table group's MEMBER the same canonical suppression the pilot gets, so
   `mode4` is false and `baseRowMode` stays 1 (sharing live). Today `enrolled` requires
   `pilotSelf`, which is true only for the pilot bus.
2. Put it behind `-Dolf.gpu.canonicalMemberMode` (default OFF at first), the convention this repo
   already uses for `olf.gpu.ziTreeRelocation` / `olf.gpu.pilotSelfRotation`, so both sides are
   measurable in one build and the default flips only on evidence.
3. Gate: the 12 base-pinned fixtures named in the extractor comment
   (`deviceRemoteFreeMemberBasePinnedMatchesCpu` & co.) are the tripwire — they pass TODAY because
   `baseRowMode` carries a genuine base pin. A member that is genuinely Q-limit pinned HAS an RL
   entry, so the kernel can still see it; that is the thing to verify, not assume.
4. Then the full gate: 177-test `GpuBatchedAcSecurityAnalysisTest`, `GpuLoadFlowTest`,
   `GpuFullNewtonTest`, `GpuAcStructuresTest`, `GpuLineSearchTest`, and the full busbar scope at
   violations diff 0 / base BUSDIFF 0 / worst DAP <= 1.4e-03 MW / coverage 6550+ — plus the thing
   this whole entry is about: `DUMMY_Q` on `VL-2669_1..4` agreeing instead of differing by
   8.410878e-03 pu.

**!! VOID — STEPS 3 AND 4 BELOW WERE RUN WITH THE FLAG OFF (caught 2026-10-01). !!** I passed the
property as `-DargLine="-Dolf.gpu.canonicalMemberMode=true -Xmx30g"`, and `QROW_CLAIM` proves it
never reached the extractor: `canonMember=false -> mode4=true` on all 24 lines while
`candTable=YES`. With a plain `mvn -Dolf.gpu.canonicalMemberMode=true` the SAME run gives
`canonMember=true -> mode4=false` on all 24. So "177/177 with the flag ON" was 177/177 with it OFF
and proves NOTHING about the tripwire, and the neutral scope table measured the unchanged code.

This is a LESSON ALREADY IN THIS FILE at the `pilotSelfRotation` entry: surefire forks, so a
behaviour property must go through `mvn -Dolf.gpu.X=true` (it propagates `-D` to the fork), not
`MAVEN_OPTS` and not `-DargLine` (which REPLACES surefire's own argLine). **Read that entry before
overriding a property.** The corroborating detail I misread: "JVM max heap: 30688 MB" appears in
runs with NO `argLine` at all — the pom sets the heap — so that line was never evidence the
override took.

**STEPS 1-3 (the VOID run's text, kept for the record):** `GpuAcRowIndexing.canonicalMemberModeEnabled()` +
`canonicalMember` in the extractor's `mode4`; `canonMember=` added to `QROW_CLAIM`. With
`-Dolf.gpu.canonicalMemberMode=true`:

- `GpuBatchedAcSecurityAnalysisTest` **177 run, 0 failures, 1 skipped**
- `GpuLoadFlowTest` + `GpuFullNewtonTest` + `GpuAcStructuresTest` + `GpuLineSearchTest`
  **20 run, 0 failures, 0 skipped**

**The tripwire is CLEAR**, and that is the substantive result: the 12 base-pinned fixtures
(`deviceRemoteFreeMemberBasePinnedMatchesCpu` & co.) pass with `baseRowMode` no longer carrying the
pin, which CONFIRMS the kernel's per-(scenario, entry) reactive-limit read covers MEMBERS and not
just the pilot — the premise the whole plan rests on, now measured instead of assumed.

**STEP 4: the scope run is NEUTRAL, and the fix is therefore NOT yet demonstrated.** Full busbar
scope, both arms, `-Dolf.gpu.canonicalMemberMode=true`:

| | flag ON | recorded baseline |
|---|---|---|
| BASE BUSDIFF | 0 | 0 |
| violations | CPU=503923 GPU=503923 diff=0 | diff=0 |
| worst DAP | 1.397828e-03 MW | 1.398e-03 MW |
| coverage | 6550 / 6667 | 6550 / 6667 |
| census | identical (`VC_REMERGE_MULTI=47`, `PINNED_MAIN=3`, `RELOC_VROW=3`, …) | same |

No regression anywhere, and no improvement either: `1.397828e-03` IS the baseline's `1.398e-03`.
The test still fails its DAP assertion at that value, as it did before.

**So the honest status: the code path is implemented and harmless, but NOTHING here shows it fixed
#66.** The scope headline cannot show it — `BUSDIFF` compares bus VOLTAGES only and is blind to the
reactive split, which is the whole reason #66 stayed hidden. Two possibilities, not yet separated:

1. `canonicalMember` did not actually fire (the property reached the JVM — "JVM max heap: 30688 MB"
   confirms the `argLine` override took — and `candTable=YES` was measured, so it SHOULD have, but
   that is an inference, not a measurement).
2. It fired and `DUMMY_Q` now agrees, but the residual 1.4e-03 MW DAP gap has a DIFFERENT cause,
   which would mean #66 was never what drove the DAP difference.

**Next, in this order — do NOT claim the fix before both:**
(a) one run with the flag ON and `OLF_GPU_QROW_TRACE=VL-2669`, to see `canonMember=true -> mode4=false`
    instead of `mode4=true`. Cheap, and it settles (1).
(b) the `DUMMY_Q` comparison the entry is actually about: device `OLF_GPU_SA_VAR_DUMP` against the
    fork's `OLF_VAR_DUMP` at the SAME point, filtered by `OLF_TRACE_CTG`, flag ON vs OFF — does the
    8.410878e-03 pu gap on `VL-2669_1..4` (331 of 366) close?

Default stays `false` until (a) and (b) report.

**(a) ANSWERED, and it VOIDED the two runs above** — see the VOID banner. With the property passed
correctly the extractor reads `canonMember=true -> mode4=false` on all 24 lines.

**THE REAL GATE RESULT, flag genuinely ON: 20 FAILURES — the plan's premise is FALSE.**

```
deviceBasePinnedMemberKeepsItsQWhenASiblingIsIslandedMatchesCpu   javaRemoteAllMembersPinnedMatchesCpu
deviceRemoteForcedPvElectionMatchesCpu                            javaRemoteForcedPvMatchesCpu
deviceRemoteFreeMemberBasePinnedMatchesCpu                        javaRemoteFreeAndSiblingPinMatchesCpu
deviceRemoteGeneratorControlReactiveLimitsMatchesCpu              javaRemoteFreeMemberBasePinnedMatchesCpu
deviceRemoteGroupAllPinnedInBaseMatchesCpu                        javaRemoteFreeMemberPostContingencyPinMatchesCpu
deviceRemoteGroupLostInBaseRegainsControlInSamePassMatchesCpu     javaRemoteMemberBasePinnedMatchesCpu
deviceRemoteGroupUnpinnedByContingencyRecomputesTargetQMatchesCpu javaRemoteMemberPostContingencyPinMatchesCpu
deviceRemoteMemberBasePinnedReliefMatchesCpu                      javaRemoteOpenBranchMemberPinMatchesCpu
deviceRemoteRobustRealisticVoltageMatchesCpu                      javaRemoteSingletonMemberPinMatchesCpu
deviceRemoteSingletonMemberPinMatchesCpu                          deviceRemoteZeroImpMemberPinMatchesCpu
```

So `baseRowMode = 4` IS load-bearing for a genuine base pin: **the per-(scenario, entry)
reactive-limit read reconstructs a pin for the PILOT but NOT for a MEMBER.** The kernel comment I
built the plan on is true of the pilot path only. Premise retracted.

**Refinement tried: suppress only for a REALISTIC-V freeze.** `MIN_Q`/`MAX_Q` is modelled by the
device's RL pack and must stay in the structure; `MIN_REALISTIC_V`/`MAX_REALISTIC_V` has NO RL entry
(set with Q inside its bounds) and is cleared by `switchPqPv`. That took **20 failures down to 1** —
strong evidence the discriminator is the right axis.

**But the ONE that remains disproves the rule: `deviceRemoteRobustRealisticVoltageMatchesCpu`**, the
fixture built for exactly this rule: `bus voltage b1 for l23b expected 20.122797375361753 but was
20.123052662204003` (2.55e-04 kV). **A realistic-V freeze CAN PERSIST into OLF's converged state**,
so "ignore every realistic-V freeze" is wrong, and whether a freeze is transient is NOT knowable at
the extraction instant. The axis is right; the test on it cannot be local.

**Shipped path unaffected:** with the default (`false`), `GpuBatchedAcSecurityAnalysisTest` +
`GpuLoadFlowTest` + `GpuFullNewtonTest` + `GpuAcStructuresTest` + `GpuLineSearchTest` = **197 run,
0 failures, 1 skipped**.

**CAPTURE TIMING IS ALSO WRONG — measured at the boundary (2026-10-01).** A new
`Q_PROBE_BASE_VC` print in `afterPreContingencySimulation` (the exact instant the precompute
captures, i.e. after the base load flow converged) says the base LF genuinely ENDS frozen:

```
Q_PROBE_BASE_VC bus=VL-2669_0 vcEnabled=true  frozenQ=false qLimit=-               v=0.620000000
Q_PROBE_BASE_VC bus=VL-2669_1 vcEnabled=false frozenQ=true  qLimit=MIN_REALISTIC_V v=0.620000000
Q_PROBE_BASE_VC bus=VL-2669_2 vcEnabled=false frozenQ=true  qLimit=MIN_REALISTIC_V v=0.620000000
Q_PROBE_BASE_VC bus=VL-2669_3 vcEnabled=false frozenQ=true  qLimit=MIN_REALISTIC_V v=0.620000000
Q_PROBE_BASE_VC bus=VL-2669_4 vcEnabled=false frozenQ=true  qLimit=MIN_REALISTIC_V v=0.620000000
```

So a fresh extraction here would read the SAME frozen state — re-extracting after convergence
cannot fix anything. Retracted before being coded.

**And the voltage is the real story: v = 0.620000000 pu.** 62% of nominal on a 20 kV level, far
below the 0.816 trip point — which is WHY the freeze sticks rather than being cleared by
`switchPqPv`. The pilot `_0` still controls (`vcEnabled=true`), so V is pinned at its target and the
couplers tie all five sections to exactly that value, which is why all five read an identical
0.620000000.

**THE HARNESS'S BASE COMPARISON HAS BEEN VACUOUS, AND THAT IS HOW A 0.62 pu BUS STAYED HIDDEN.**
Every scope run in this file reporting `BASE BUSDIFF buses>1e-12kV=0 of 0` compared **zero buses**
— `of 0` is the count COMPARED. The test's own comment already says so ("was not a clean base - it
was no base compared at all, in every run this test has ever done") and adds an explicit N-state
monitor, **but the monitors are only installed when `busDiff` is true**, i.e. when
`OLF_GPU_SA_CTG_ONLY`, `OLF_GPU_SA_BUSDIFF_CTG` or `OLF_GPU_SA_BUSDIFF=1` is set. A plain full-scope
run sets none of them, so it still prints `0 of 0`. **Treat "base BUSDIFF 0" from any run without
one of those variables as NO EVIDENCE** — including the ones quoted earlier in this entry and the
"base BUSDIFF 0" in the [[#62]]..[[#65]] gate descriptions.

The cheap way to get a REAL base comparison (keeps the full list, the full network, and a small
heap — the all-contingency monitor dies at the 14 GB ceiling): `OLF_GPU_SA_BUSDIFF_CTG=<one id>`,
which installs the scoped monitor plus the `ContingencyContext.none()` base monitor, with
`OLF_GPU_SA_MON_BUS=<bus ids>` to print the buses of interest on both arms.

**ROOT CAUSE FOUND (2026-10-01) — the device's robust-mode exemption is NARROWER than OLF's.**
An equation-level predicate mismatch, no tolerance anywhere. Chain, all measured:

1. **The freeze is the DEVICE's own decision, written back into the `LfBus`.**
   `GpuNewtonRaphson:403` does `bus.freezeGenerationTargetQAndDisableGeneratorVoltageControl(qLimit)`
   then `bus.setQLimitType(... case 3 -> MIN_REALISTIC_V ...)` from the device's reactive-limits
   result. So `qLimit=MIN_REALISTIC_V` at VL-2669 is NOT OLF's outer loop — it is the device's.
2. **OLF never freezes this group.** CPU arm (`OLF_GPU_DBG_RL`, single-threaded): the robust rule
   IS live — `CPU_RL Switch bus 'VL-1251_1' PV -> PQ, q set to -0.35 = targetQ - because V < 16.0kV
   when remote voltage target is maintained`, likewise VL-1459_1/_2 — but `grep -c 2669` is 3 and
   NONE is a VL-2669 bus (one is a digit coincidence inside `q=102.10822669930263`, two are
   VL-6226_0). So the rule fires elsewhere and skips VL-2669.
3. **WHY OLF skips it.** `ReactiveLimitsOuterLoop.isGeneratorRemoteController` reads the bus's OWN
   control object: `controllerBus.getGeneratorVoltageControl().map(c -> c.getControlledBus() !=
   controllerBus).orElse(false)`. For a merged DEPENDENT member that own control is LOCAL
   (`controlledBus == bus` — measured: `ownVc=VL-2669_1 ownMerge=DEPENDENT`), so the bus is NOT a
   remote controller and the whole `robustMode && generatorRemoteController && …` block is skipped.
4. **WHY the device freezes it.** The kernel gate is
   `bool unrealisticLo = robust && rlRemote[i] != 2 && x[ridx] < vLoRobust;` and its own comment
   says it exempts only "a pilot that controls its own bus (class 2)". `rlRemote` is assigned at
   `GpuAcOuterLoopPacks:676` as `(byte) (pilotSelf ? 2 : 1)` with `pilotSelf = isPilotSelfEntry(bus)`,
   computed from the **MAIN** control (`mainGenVc(bus).getControlledBus() == bus`). For VL-2669_1 the
   MAIN is at VL-2669_0, so the member is class 1 = "remote" and the rule applies.

**So the two predicates differ exactly for a merged dependent member:** OLF asks the bus's OWN
control (local ⇒ exempt), the device asks the MERGED group (member ⇒ remote ⇒ subject). The rest of
#66 follows: the device freezes, mode 4 own-Q balances replace the sharing in the structure,
`DUMMY_Q` ends 8.410878e-03 pu off, and OLF's converged base keeps `DISTR_Q` active.

**Also now explained without any tie-breaking story:** `v = 0.620000000` pu is simply the group's
own target — `targetV = 12.4` kV on a 20 kV nominal level = 0.62 exactly — so the bus sits ON
target, and `minRealisticVoltage = 0.8` (threshold 0.816 pu) declares that target unrealistic. The
voltage is identical in both arms; only the EXEMPTION differs.

**RETRACTED along the way, and not to be revisited:** (i) "the arms converge to different tightness
so the strict `<` in the switch-back test falls differently" — a same-system-different-solution
story, and I had ONE value from ONE arm when I made it; the switch-back test's other conjunct
`canSwitchPqToPv = !bus.hasGeneratorsWithSlope()` is a static network property, identical in both.
(ii) "the 4-thread CPU arm retains different switches, so the networks differ" — the test already
sets `defaultThreads = kindsEnv.contains("busbar") ? "1" : "8"`, so every busbar run was ALREADY
single-threaded on both arms (its comment: "the base alone differs by 8e-6 kV at 8 threads,
bit-identical at 1"). Correctness runs must stay at 1 thread; CLAUDE.md's "default 4" is the
BENCHMARK test.

**FIXED 2026-10-01 — `rlRemote` class 3.** No new array and no JNI change were needed: an AUDIT of
every `rlRemote` reader showed a third class slots in for free. The pack now emits
`2` = pilot-self, `1` = a remote member the robust rule APPLIES to, `3` = a remote member it
EXEMPTS, from a verbatim copy of OLF's predicate (`GpuAcOuterLoopPacks.isOlfRobustRemoteController`,
`bus.getGeneratorVoltageControl().map(c -> c.getControlledBus() != bus).orElse(false)` — note the
`orElse(false)`: no control at all is also exempt). The kernel gate became
`bool robustRemote = rlRemote[i] != 2 && rlRemote[i] != 3;`.

Why class 3 is free (the audit, worth not redoing): every other reader treats it exactly like class
1 — the TRUTHINESS tests (`rlRemote[i]` at .cu 4712/5560/5860/5883/6457/6500, Java `!= 0` at
`GpuBatchChunkSolver:432`) see a remote member; the PILOT tests (`== 2` at .cu 5591/5592/5874/5882/
6454, Java `!= 2` at `GpuNewtonRaphson:435` and `== 2` at `GpuBatchChunkSolver:771`) correctly see a
non-pilot; and `rlRobustVResetBatch`'s `!= 1` early-return correctly SKIPS an exempt bus, since a
bus the gate never pins to 3/4 must not get the robust V reset either.

**Measured, same full busbar scope:**

| | before | after |
|---|---|---|
| base control state of `VL-2669_1..4` | frozen, `qLimit=MIN_REALISTIC_V` | **`vcEnabled=true frozenQ=false qLimit=-`** |
| `QROW_CLAIM` (an own-Q/frozen member) | 24 | **0** |
| `DISTRQ_ROW` (sharing ACTIVE) | 24, alternating 6-on/6-off | **48, every one of 12 passes** |
| BASE BUSDIFF | `0 of 0` (vacuous) | **worst 2.731e-11 kV over 6886 buses** |
| violations | CPU=503923 GPU=503923 diff=0 | CPU=503923 GPU=503923 **diff=0** |
| coverage / census | 6550/6667, census X | 6550/6667, **census X identical** |
| unit gates | — | **200 run, 0 failures, 1 skipped** (no property gate) |

The 6-on/6-off extraction cycle is GONE and no member ever claims an own-Q row, so the structure
carries the sharing for all 6550 scenarios. `Q_PROBE_BASE_VC` also confirms the predicate directly:
for bus `_1` it prints `ctlBus=VL-2669_1`, i.e. its own control really is LOCAL.

**Checkstyle: the violation SETS are identical before and after** (diff of the sorted lists is
empty — the branch's 220-odd are pre-existing, 74 of them an old indentation block in
`GpuAcOuterLoopPacks`). Do not read a raw count as a delta; line numbers shift.

**What this did NOT fix, and is now known to be a SEPARATE item:** worst `distributedActivePower`
is `1.397830e-03` MW, versus `1.397828e-03` before — unchanged, and the test still fails that
assertion. So #66 was never the cause of the DAP gap. That gap is its own open question.

**Dead experiment removed:** `-Dolf.gpu.canonicalMemberMode` and
`GpuAcRowIndexing.canonicalMemberModeEnabled()` are gone — its premise was retracted above and the
real cause is fixed at the SOURCE. The `mode4` comment now records that a member inheriting mode 4
from a genuine Q-LIMIT pin is CORRECT (the 20 base-pinned fixtures rely on it); what used to reach
there wrongly was a freeze the device should never have made.

**Also now explained, with no tie-breaking story:** `Q_PROBE_BASE_VC` prints
`ctlV=0.62000000000000000 tgtV=0.62000000000000000 ctlV<tgtV=false d=0.000e+00` — the switch-back's
strict `<` is an EXACT tie in both arms, so it could never have fired either way. The retracted
"one arm lands a hair below" story was not just unevidenced, it was impossible.

**Superseded reasoning (kept): "the fix IS a capture-timing fix after all".** The earlier
objection ("the cycle repeats inside the SA too, so settled is not one instant") was wrong in an
important way: the repeats at passes 13-24 are PER-CONTINGENCY solves, whereas the BASE load flow's
CONVERGED state is a well-defined instant — and it is the reference, because OLF's converged base is
what the arms must agree on. The pass map shows the precompute capturing right after pass 12, the
end of a FROZEN window, while OLF's converged base has those controls ENABLED. **Next step: build
the structure from an extraction taken AFTER the base load flow converges** (or re-extract then),
rather than from whatever instant the base LF last happened to extract at. That keeps a PERSISTENT
realistic-V pin (it is still there at convergence, which is why `deviceRemoteRobustRealisticVoltage`
must keep passing) and drops a TRANSIENT one, with no local "is this transient?" test needed.

**Superseded (kept for the reasoning):** `QROW_CLAIM` also prints
`frozenQ` / `monitoringV` / `qLimit` / `gens`, which separate OLF's THREE disable paths — they need
opposite fixes and no pin dump can tell them apart:

- `frozenQ=true` -> `freezeGenerationTargetQAndDisableGeneratorVoltageControl`, i.e. a PV->PQ pin
  (`ReactiveLimitsOuterLoop`, or the tap manager's
  `disableGeneratorVoltageControlsUnderMaxControlledNominalVoltage`). `qLimit` then names which.
- `monitoringV=true` -> a `MONITORING_VOLTAGE` generator (an SVC with a stand-by automaton):
  `LfNetworkLoaderImpl.createGeneratorVoltageControl` creates the control DISABLED AT LOAD and
  `MonitoringVoltageOuterLoop` enables it during the solve. **This path is invisible to
  `RL_SWITCH`** — which is exactly how the arms' pin SETS matched while `enabled` did not.
- neither -> `LfContingency:234` / `NetworkCache:451`, or a `BusState` restore.

The full writer list, so this is not re-derived: `setGeneratorVoltageControlEnabled` is private to
`AbstractLfBus` and reached only from `freezeGenerationTargetQAndDisableGeneratorVoltageControl`
(false) and `setGeneratorVoltageControlEnabledAndRecomputeTargetQ` (either); the latter's callers
that pass FALSE are `LfNetworkLoaderImpl` 212 (MONITORING_VOLTAGE at load) and 222
(`discardGeneratorVoltageControl`), `LfContingency` 234, `NetworkCache` 451 and `BusState` 127/161
(restore); those that pass TRUE are `ReactiveLimitsOuterLoop` 227 (PQ->PV switch BACK),
`MonitoringVoltageOuterLoop` 63, `GeneratorVoltageControlManager` 84 and
`LfSecondaryVoltageControl` 141.

**Superseded candidate (kept for the reasoning):** The comment at that site
says "ACTIVE own-Q balance at a (non-free) remote gen-VC member = the RL-PINNED state", but the
condition never tests whether the controller is actually DISABLED:

```java
if (GpuAcRowIndexing.isRemoteGenVcControllerElement(bus)
        && (pilotSelf || !GpuAcRowIndexing.isFreeRemoteGenVcMember(bus)) && !enrolled) {
    mode[qr] = 4;
}
```

Adding `&& mainVc != null && !mainVc.isControllerEnabled(bus)` makes mode 4 fire only for a
genuinely pinned (control-off) member, which is what the comment claims it means, and leaves the
row to the `DISTR_Q` sharing when the control is on. `isControllerEnabled` is the same predicate
the DISTR_Q dump prints. **Risk to check:** several fixtures were built AROUND the current
behaviour (the comment cites `deviceRemoteFreeMemberBasePinnedMatchesCpu` and ~12 tests, the
all-pinned-group fixture, and a 976x LINEAR_SOLVE_FAILED regression on the enrolled-pilot path), so
this must be judged on the full gate, not on BUS-4467_BBS alone.

**Still unexplained and worth one measurement before coding:** WHY both are active at extraction
while OLF's converged base has only DISTR_Q. If the device extracts before OLF's
`updateRemoteVoltageControlEquations` has settled the pair, the better fix may be WHERE it
extracts, not the gate. The DISTR_Q dump shows the bus's OWN control (1 member, qPct 0.2, so a
merged DEPENDENT), while the row logic uses `mainGenVc` — check whether the two disagree here.

**D-old. (superseded) THE REMAINING UNKNOWN as first written.** In OLF's BASE,
`bus_target_q[3025..3028]` are **INACTIVE** (bracketed; only DISTR_Q cols 646-649 are active), and
the mode-4 gate fires only when `row(es, bus, BUS_TARGET_Q) >= 0`, i.e. ACTIVE — `row()` reports
activity faithfully (`SingleEquation.column` is -1 when inactive and the index resets it on
deactivation). **So the gate should not have fired at all.** Two candidates:
  (C1) the structure the batched SA uses comes from an extraction pass taken at a state where those
       controllers WERE pinned (the base LF's own RL loop switches during the solve; the DISTR_Q
       dump shows passes 13 and 14 in a base-only run, so there are several), and the row modes are
       that snapshot; or
  (C2) a DIFFERENT bus whose `qRow[bn]` ALIASES the same dummy row has an active BUS_TARGET_Q and
       assigned both the target and mode 4 there.
These have different fixes, so decide first. **Cheapest test:** a one-line trace at the
`mode[qr] = 4` site printing `bus`, `qr`, `q` and the target, gated on a row env, run on the SA
(not base-only) and filtered to rows 14113-14119. That names the bus and the pass.

**D. Then the fix, by branch.**
  - If C2: the aliasing is the bug — one row cannot host two buses' reactive equations. Fix where
    `qRow`/`relocatedQRow` assigns it, the same family as [[#62]]-[[#64]].
  - If C1: the device's base must reach OLF's final pin state. Either extract after the base's outer
    loops have settled, or let the device's own RL kernel unpin (mode 4 -> the DISTR_Q row) as OLF
    does. Note the trap: OLF's RL trace shows only ONE base PQ->PV switch in the whole run
    (VL-5997_1), so if those buses were never pinned by OLF, C1 reduces to "the snapshot was taken
    before the VC merge was applied" — check `getMergedControllerElements()` size at extraction
    (the DISTR_Q dump prints the bus's OWN control, which shows 1 member; `mainGenVc` resolves the
    MAIN control and is what the row logic should use).
  - Either way the device needs somewhere to put a relocated member's DISTR_Q that is not its own
    `BUS_V` row (taken by `ZERO_V`) and not shared with an own-Q balance. The 2-way stock twin
    (`dqAlt`/`dqSel`) is the existing mechanism and is NOT enough: it answers "the coupler died,
    move it back", not "this row must host a GROUP equation".

**E. Gate for any change.** 177-test `GpuBatchedAcSecurityAnalysisTest`; `GpuLoadFlowTest` /
`GpuFullNewtonTest` / `GpuAcStructuresTest` / `GpuLineSearchTest`; full busbar scope with
violations diff 0, base BUSDIFF 0, worst DAP <= 1.4e-03 MW and no loss of device coverage
(6550/6667). It changes the BASE for ~341 members, so also re-check the base's own
`distributedActivePower` (currently cpu 14.359916545 vs gpu 14.359916552 MW).

**F. Harness gap to close regardless.** `BUSDIFF` compares BUS VOLTAGES ONLY. Add the dummy columns
(or all state variables) to the comparison, or this class of difference stays invisible — it is how
an 8.4e-03 pu reactive-split error survived every "bus voltages exact" claim in this file.

**The only valid comparison, and the discipline broken five times here:** both arms at the SAME
stage, from the SAME run. `OLF_GPU_EQ_DUMP` does exactly that for the BASE (it writes the device's
system and OLF's own dump of the same state), and `OLF_GPU_TARGET_ROW` separates a STRUCTURAL
mode (`baseMode`) from a per-scenario one. If the two arms already disagree on this group in the
BASE, the defect is in extraction and the base is where to fix it; if they agree there, it is
per-contingency.

**Two retractions, kept visible.** (1) "the device's dummies do not satisfy the chain" — backwards.
(2) "OLF's converged state leaves 1.5e-03..4.2e-03 on its own BUS_TARGET_Q equations" — invalid:
computed from the PRE-solve targets in F0 against the POST-solve state, and a BUS_TARGET_Q target
is not constant through a solve (a pin rewrites it with `setGenerationTargetQ(limit)`). OLF is the
reference; a computation showing it violating its own equations is evidence the computation is
wrong. Third point-in-time error of that session after `F0` and the dummy state — dump state and
targets AT THE SAME POINT.

**Why it hid for weeks:** `BUSDIFF` compares BUS VOLTAGES ONLY, so an 8.4e-03 dummy difference is
invisible to it, in the base as much as in a contingency. Worth closing on its own.

### #65 — Duplicate target overrides: measured, then fixed at the producer (2026-09-30)

RESOLVED. Kept because the METHOD is the reusable part, and because ~200 benign duplicates
per run remain.

`GpuBatchOverrides`'s own javadoc states the invariant: "A row must carry at most ONE
override. The device scatter has no ordering, so two entries for the same row race."
`applyTargetOverrideBatch` is one thread per entry doing `target[ovIndex[k]] = ovValue[k]`,
so a second entry makes the surviving target UNSPECIFIED. `add()` permits duplicates and
documents "last one wins", which the device does not honour — two fixtures were passing
*because* the race favoured the earlier entry ([[#62]]).

**Measured before changing anything** (`OLF_GPU_OV_DUP_TRACE`, full busbar scope, per run):

| | |
| --- | --- |
| override entries handed to the device | ~3.4 M |
| distinct rows with >1 entry | 211 |
| rows whose entries DISAGREE | **26**, over 24 contingencies |
| producers | **one** — `GpuBatchChunkSolver` DISTR_Q target seeding |
| row mode of every conflicting row | **4** (own-Q), baseMode 1 |
| shape | 25 of 26 are `0.0` shadowed by a small reactive target; 1 is 0.06 -> 0.092 |

Mode 4 means the row hosts its OWN reactive balance — an RL pin, or a control the contingency
killed whose `BUS_TARGET_Q` the injection override owns. `updateDistrQTargetsBatch` already
REFUSES to maintain a DISTR_Q target there ("A DISTR_Q target has no business here", measured
on BUS-1055_BBS_1 — which is one of the 24). So the host was uploading exactly what its own
kernel declines to maintain. Fix: don't seed a DISTR_Q target on a mode-4 row. That is a rule,
not a choice of winner — which is why forcing `put` (last-wins) earlier the same day was wrong
and turned two fixtures red.

**Guard only what was measured.** The first attempt also skipped mode 0, the voltage identity
the same kernel refuses, on symmetry grounds. No measured conflict was on a mode-0 row, and
skipping it moved `deviceMultiChunkMatchesCpu`'s B3 by 1.06e-12 kV — past its 1e-12 assertion.
Restricted to mode 4: suite 177/177. The comment at the site says why mode 0 is deliberately
NOT guarded, so nobody "completes" it.

**What it bought, full scope** (conflictRows 26 -> **0**; 800/4-chunk benign duplicates remain):

| | before | after |
| --- | --- | --- |
| BUS-1599_BBS_1 DAP | 1.737e-02 MW | **3.711e-05** |
| scope total sum abs DAP | 1.499204 MW | 1.481873 MW |
| worst DAP (BUS-6113_BBS_1) | 1.328019 MW | unchanged |
| violations diff | 71 | unchanged |

**A prediction that was wrong, recorded so it is not repeated:** the +-2 violation pair
BUS-1359_BBS_1 / BUS-1869_BBS_1 has 2 conflicts each and did NOT move. Their DAP differences
are ~1e-12 MW — those states agree to machine precision, and the +-2 is the threshold flicker
of `docs/increased-violations-thresholds.md`, not a state difference. A conflicting override
does not imply a visible difference, and the converse held too: neither headline failure
([[#63]], [[#64]]) has any conflict.

**Method notes.** Device row indices are NOT portable across contingency sets —
BUS-1055_BBS_1's conflicting row is 2460 in the full set and 2290 run alone; a row number in a
note needs its scope stated. And BUS-1055_BBS_1 PASSES run alone while still racing, so this
defect class hides completely from single-contingency reproduction. The remaining ~200
same-value duplicates per run are harmless by construction (either write gives the same
target) but they are still invariant violations: `OLF_GPU_OV_DUP_TRACE` reports them, and a
future `conflictRows > 0` is a regression.


### #63b — superseded by #63 above (kept for the measurement recipe)

The writer named in #63 step 3. Was open, It is the WHOLE violation gap of the full busbar scope: 71 of 71.

`BUS-6194_BBS_2` loses the middle of three coupled sections (VL-6194_0/_1/_2, both couplers
opened, VL-6194_1 islanded) and the VC group re-homes: `reHome main=VL-6194_0
cluster=[VL-6194_2] newMain=VL-6194_2 target=1.123 local=true`. The device SOLVES it —
`status=0 iters=1 norm=2.17e-9` — but the run is reported `FAILED` where OLF reports
`CONVERGED`, so the GPU arm yields an EMPTY result and loses every violation on it. The
status comes from OLF's realistic-voltage band, CPU-side:

    UNREALISTIC bus=VL-6194_2 v=0.625100000 nominalV=380.000 band=[0.800, 1.200]

**What is proven.** VL-6194_2's own V row is 12938. The HOST uploads it correctly — mode 0
with target **1.123** (`OLF_GPU_TARGET_ROW=12938`). The DEVICE ends it at **0.6251**
(`OLF_GPU_FINAL_TARGET_ROW=12938` -> `final=0.625100000 setupMode=0 devMode=0`). So a device
outer-loop kernel overwrote the row's target between upload and convergence, the mode stayed
0, and a mode-0 identity makes x = target: the bus voltage BECOMES the reactive number.
0.6251 is this group's `mKey` == `mTargetQ` for all three members (they coincide because all
derive from the same 62.51 MVAr reactive limit), so what landed there is a DISTR_Q/pilot-Q
target.

**Where it is not.** `updateDistrQTargetsBatch` is the obvious suspect and it already has the
right guard — "The row is a VOLTAGE IDENTITY this scenario ... A DISTR_Q target written here
replaces the voltage target" (measured on BUS-6029_BBS_1) — and it runs AFTER the
stock-placement diversion, so a member diverted onto 12938 should return. Its `qg >= 0`
carrier branch writes only when `mrow == carrier`, i.e. only to a row that already passed
that guard. The group has NO live keys here (`DQISL keySum=0.0`, `keeps=false` for all three
members), so no DISTR_Q target is even owed. Which kernel writes it is NOT yet established —
do not guess it; instrument the writes. Device `printf` does NOT reach the surefire log
(measured: `OLF_GPU_DS_DEBUG` printfs never appeared), so this needs a host-readable record,
the way the DS gate record works.

Reproduce in ~40 s:

    OLF_GPU_SA_CTG_KINDS=busbar OLF_GPU_SA_BUSBAR_TOP=100000 \
    OLF_GPU_SA_CTG_EXCLUDE=docs/rte6515-impactless-busbar-sections.txt \
    OLF_GPU_SA_CTG_ONLY=BUS-6194_BBS_2 OLF_GPU_FINAL_TARGET_ROW=12934,12936,12938,13045,13047 \
    mvn -Dcheckstyle.skip -Dsurefire.failIfNoSpecifiedTests=false \
      -Dtest='GpuVsCpuRte6515SaTest#gpuSaMatchesCpuOnRte6515Full' test

**Fixed on the way** (same contingency, a different bug, suite still 177/177): the coupler
block's re-homed V identity — `rowZi[childVRow] = 0` plus the VC target — ran ABOVE the
`childFrozen` guard, so an ISLANDED child got a voltage identity as well. On this contingency
VL-6194_1's V row 12936 carried TWO overrides, 1.000 from the island freeze and 1.123 from
that identity, resolved by the `applyTargetOverrideBatch` race (see [[#62]]). The guard's own
comment already said the freeze must not be overwritten; it simply sat below the write it was
guarding. Now one override, 1.000. It did not change this contingency's outcome — the 0.6251
is a separate writer — but it is the second duplicate-override race found in a day, which
raises the priority of deduplicating the override list to its documented last-wins.

### #64c — superseded by #64 above (fixed)

The full busbar scope's only material DAP difference:
`cpu=217.889505513 gpu=219.217524713 MW`, **1.328 MW**. Every other contingency is at or
below 1.7e-02 MW and the bulk at ~3.7e-05. It also carries a violation delta of -1. Not yet
investigated. Worth doing right after [[#63]]: both are busbar-section contingencies on
coupled sections, the population [[#62]]'s fixes moved most.


### #62 — Coupler relocation: a structural row a contingency un-relocates (2026-09-30)

FIXED (54a866a) for the two middle-section fixtures; the pattern is the thing to remember.

**Attribution, corrected.** The per-scenario STOCK PLACEMENT this builds on (`dqAlt`/`dqSel`, the
three-destination stock pair on the reloc pool, `stockPairOut`/`stockPairPick`) was the
**2026-09-29** session's work (ec422f1, 53c8035, 5353094), not this one's. Do NOT read the
`Claude-Session:` trailer as provenance — this environment stamps the SAME session URL on every
commit made from it, so several distinct sessions share one URL. The AUTHOR DATE is the signal
(and the `Co-Authored-By` model line: 1a0bf35 is Claude Fable 5.1).

**That session left a handover naming all three suspects, and it was not read first.** Its entry
ends: "So one consumer still reads the dead row. Next: ... check which of the member's tables
still points at c12's `DUMMY_Q`: the reactive-limit entry rows, the DISTR_Q member row (`mRow`),
or the injection-override rows." Two of the three were then rediscovered independently over a
full day — the DISTR_Q member row (787985b) and the reactive-limit rows (9b957d3, the q read at
`rlQrow`) — plus the DS participant row. **The third, the INJECTION-OVERRIDE rows, is still
unverified: no measurement either way.** Read the newest `docs/tasks.md` entry before starting;
it is a handover from whoever ran last, not a diary.

`dsPhiRow` and `dqMemberRow` are **structural** — decided once per extraction. Under the
coupler-tree relocation a child bus's P/Q balances live on its parent coupler's
`DUMMY_P`/`DUMMY_Q` rows, so those arrays point there. A contingency that **opens** that
coupler pins the dummies to `DUMMY_TARGET` identities and the stock-placement fallback
returns the balances to the bus's **own** rows. Every consumer still holding the structural
row then writes to a dead row. Both middle-section fixtures had been red since the
relocation went on by default, for exactly this reason on the two sides:

| | OLF | device (before) |
| --- | --- | --- |
| bbs2 DS share | +0.173869 pu on `bus_target_p[5]` | on the pinned dummy; kernels read `(0 − anchor)` = −1.043213 as "absorbed" |
| device slack passes | 1 (the pre-distribution) | **4**, over-placing 0.0687 pu |
| `distr_q5` target | −0.2 (keys renormalized 1/3 → 1/2) | −0.133333333 (base keys) |
| l2 Q1 | 239.0484 MVAr | 225.7150 (−13.33) |
| vl3_b | 401.6903 kV | 401.3695 (−0.32) |

**Method that worked, after two wrong readings.** `OLF_GPU_EQ_DUMP_CTG=<id>` prints OLF's
own post-contingency system; its docstring already warned that reasoning about where a
relocated row *could* go had produced two wrong fixes, because opening a coupler changes
the GROUP and OLF may DEACTIVATE an equation rather than move it. Reading it settled both
sides in one shot (bbs2 alive, `bus_target_p[5]` + `distr_q5` active on its own rows, keys
0.5/0.5) — and settled the second fixture too, where only ONE `distr_q` row is active and
the seeding writes two.

**The collision that hid the Q half.** The coupler-opening block in `GpuBatchChunkSolver`
also moves that row's target, using `baseTargets[dqRow]` — the base-key value. Both writes
landed as **two override entries on one row**, and `applyTargetOverrideBatch` is one thread
per entry doing `target[ovIndex[k]] = ovValue[k]`: which survives is a **race**. The base
value was winning. `GpuBatchOverrides.add` documents "last one wins", which the device does
not honour — so the ordering intent of any duplicate is fiction. Fixed by making the
coupler block YIELD the row when the plan seeds a renormalized target for it, which keeps
every scenario whose plan seeds none bit-identical. Two fixtures
(`deviceBusLossIslanding{BaseCarrierMember,MergedPilotSurvivor}`) were *relying* on the
race: forcing "renormalized wins" everywhere (`put`) turned them red, which is how the
duplicate was found. **Any remaining duplicate override on one row is a latent bug of this
kind** — deduplicating `indexArray()`/`valueArray()` to the documented last-wins is the
real repair, not yet done.

**Instruments were lying, and that cost measurements.**
- The DS gate record is 8 doubles per scenario but the water-fill wrote `gateLast[6*s+5]`,
  the stride of the 6-slot record it replaced: for `s=0` it overwrote `dsRetain` with the
  placed amount (a FIRING pass printed the wrong column, `dsPlaced` stayed 0) and for every
  `s>=1` it scribbled into ANOTHER scenario's slots. **Every `LOOP_PASS` number read off a
  firing pass, in this file and elsewhere, was the wrong quantity.** Fixed to `8*s+7`; the
  base pass now reads exactly (`retain 0.172854 + placed 0.129640 = mismatch`).
- `PRELOSS_AFTER` compared the participants' total against the WHOLE mismatch (wrong
  whenever the slack participates) and counted an islanded row's frozen target as a
  distribution: it read the fixture as −1.669969 pu off OLF when the real gap was −0.0687.
  Now prints `seededOnParticipants` and skips islanded rows.
- `OLF_GPU_TARGET_ROW` takes a list or `lo-hi` range; one row per JVM run made "which
  device row carries this equation" a five-minute question.
- `islandedDistrQEntries` had no trace. `DQSEED` is the *generator* path's — its silence
  says nothing about the islanding path, and reading it as "the target is never seeded"
  sent one fix at an inert code path. `OLF_GPU_DQISL_TRACE` now prints that path.
- **A compile error in a trace made four runs silently use stale classes** and produced a
  "bit-identical, so my fix did nothing" reading that was simply false. Grep the mvn log
  for `BUILD FAILURE` / `cannot find symbol` before believing any measurement.

Result: suite 177/177 (was 175/177), and the fixtures match the CPU **at** the 1e-12 kV
assertion rather than within it. GpuLoadFlow / GpuFullNewton / GpuAcStructures /
GpuLineSearch green. Checkstyle unchanged at its 220 pre-existing violations (this branch
is not at zero — CLAUDE.md's "must be 0" does not hold here).


### #61 — GPU SA host overhead: the GPU is no longer the bottleneck (2026-09-21)

H100, rte6515, 14,260 contingencies (all kinds but busbar sections), RTE params: GPU arm ~47–51 s on one
host thread vs CPU 118.6 s (8 threads, warm) / 68.5 s (16), 0 differing violations. Of the GPU arm, only
~8 s is device solve: the rest is single-threaded host code — full `NetworkState.restore` per contingency
(14.9 s), the per-contingency host loss redistribution the device already did (4.6 s), full branch
recompute on state injection (6.1 s), connectivity copies in qualification (2.55 s), … Chunk size is
closed (flat above ~2000). Leads L1–L9, their constraints, gains, risks and the order to do them in:
**`docs/gpu-sa-host-overhead-assessment.md`** — read it before touching any of them. It also records L10, a
RESULT defect — `getDistributedActivePower()` differed on 5,095 contingencies (up to 124.5 MW) and the base
reported 0: the device's own DS passes were never reported — FIXED 2026-09-22 (worst now 2.2e-6 MW); and
L11, an upstream OLF unit bug in injection threshold conditions — FIXED on olf-local-patches (e97bb4e57).

### #60 — Generator and load contingencies on rte6515: first fold (2026-09-16)

Both kinds have been batchable and unit-tested for a while; what had never been run is a REAL set.
`GpuVsCpuRte6515SaTest` can now pick its kinds — `OLF_GPU_SA_CTG_KINDS=branch,generator,load`, with
`OLF_GPU_SA_GEN_MIN_KV` (default 400) and `OLF_GPU_SA_LOAD_TOP` (default 50) — so a first pass is a
small deliberate slice, not every injection. The all-branch 9042 set stays the default and the reference.

**The transmission level in this case is 380 kV, not 400.** Generators by nominal voltage:
3 kV … 63 kV=550, 90 kV=232, 150 kV=18, 225 kV=63, **380 kV=106**. A 400 kV filter selects nothing.

**How it folds** (106 generators at 380 kV + the 50 biggest loads, 156 contingencies):

| | value |
| --- | --- |
| batched | 83 / 156 at first fold, **101 / 156** after the two fixes below |
| violations | CPU 14073, GPU 14070 at first fold, **14073 / 14073, 0 differing** now |
| warm vs CPU-8t | x0.69 — still slower, because a third of the set is CPU-routed |

Loads alone: 41 / 50 batched and **0 differing violations**. The two kinds fold differently and the
generator side is where the work is.

**Three things to correct before going full throttle.**

1. **`PLAN:GEN_OUTAGE_SHAPE` rejected 64 of the 106 generator contingencies** — by far the biggest item.
   The reactive-capability CURVE channel's `p_now` linkage models the DISTRIBUTED-SLACK delta only
   (`pNow = basePpu + share * (target[pRow] - baseRowTargetP)`), so an injection override at a curve bus's
   P row would charge every generator on that bus a share of the lost generator's own output. Losing a
   generator IS such an override, which is why it hit most of them.

   **Half of it is now batched, on an argument rather than a model change.** When the outage takes the
   bus OUT of voltage control (`GenOutage.toPq()` — losing the only generator at a bus does exactly
   that), its reactive-limit entry is inert for the rest of the solve: the RL pass's unpinned branch
   wants a V row and finds a power row, its pinned branch wants a limit type and finds none. The curve
   channel still writes a dynamic limit for that entry and nobody reads it, and a skew that cannot reach
   an answer is not a reason to leave the batch. **64 rejections -> 46, batched 42 -> 60 of 106**, with
   the branch reference set untouched at 9040/9042 and the same two lines.

   What stays on the CPU is the case the argument does not cover: a bus with SEVERAL generators where the
   survivors still control voltage. Those need the model change — `pNow` corrected by the outage step so
   only the DS part is shared, the dead generator's own curve entry dropped, and the bus's aggregate
   `rlMinQ`/`rlMaxQ` reduced by its limits. That needs per-GENERATOR outage identity in the curve channel,
   which the plan currently aggregates per bus.
2. **`PLAN:INJECTION_OUTAGE_SHAPE` rejects 9 of the 50 loads** — the shape computation returns null for a
   slack bus, more than one distributed-slack bus, or a remote generator. Tagged `PLAN:SHUNT_SHAPE` until
   now, a copy-paste from the block above it; the census is how we pick what to fix, so a tag that lies
   about the reason sends the next person at the wrong gate. Renamed.
3. **Three differing violations, all on generator contingencies** (GEN-5997, GEN-5997#0, GEN-6130, each
   −1). **Two of the three are now fixed, and the cause was a real modelling gap**, not a knife edge.

   A probe settled it in one run: routing out every generator outage at a bus that KEEPS voltage control
   took the difference from 3 to 0. The device's `rlMinQ`/`rlMaxQ` are RUN-CONSTANT — the bus aggregate as
   extracted — while OLF's `bus.getMinQ()`/`getMaxQ()` sum the generators still there. Lose one generator
   at a two-generator bus and OLF's reactive limits shrink while the device's do not, so the two arms
   disagree about when that bus pins, and every equation still converges on both.

   `GenOutage` now carries the reactive-limit entry and the capability the lost generators take with them,
   and it reaches the kernel as a per-(scenario, entry) delta on the same dynamic-limit arrays the CURVE
   channel writes. Those arrays existed only when curves did; the selection is now `useDynRlLimits`, true
   for either channel, and they are allocated lazily when only the new one is live. **3 differing -> 1**,
   the branch reference set untouched at 9040/9042 with the same two lines, suite 261/0.

   **GEN-6130 is fixed too, and all three hypotheses above were wrong.** The instrument that found it:
   `OLF_GPU_SA_CTG_ONLY=<id>` runs one contingency, describes its bus, and monitors every voltage level.
   It said the difference was LOCAL — 0.995 kV at that node against 2.2e-4 kV everywhere else — and
   `OLF_GPU_GENOUT_TRACE` then said `busPv=false`, `busGens=1`.

   So the bus was ALREADY PQ: the base loadflow's reactive-limits loop had pinned it. The RL pack keeps
   such a bus as a `basePqCapable` entry with a seeded limit type, exactly so a RELIEVING contingency can
   switch it back to PV — and with its only generator gone there is nothing to switch back to, while OLF
   has no controller there at all. `GenOutage` now reports when a bus keeps NO generator and the entry is
   masked for that scenario through the same channel a contingency-stranded controller uses.

   **The whole set now agrees**: 106 generators at 380 kV plus the 50 biggest loads, 14073 violations on
   both arms, 0 differing. On GEN-6130 alone, every one of the 29832 buses agrees to 3.6e-13 RELATIVE
   (worst absolute 5.7e-11 kV, which moves run to run — round-off, not an offset). Note that an absolute
   1e-12 kV bar is below one ulp of a 400 kV bus (~5.7e-14), so the instrument reports both.

### #59 — Both routes build their phase tables with one function (2026-09-16)

The base load flow and the batched security analysis called the SAME nine builders with the SAME
arguments from two places, 150 lines against 60. `GpuBatchPhases.build` is now the only one, taking a
`Request` of which loops the run configures (the analysis fills it from its `BatchConfig`, the base from
the configured outer loops) and returning every phase's controllers.

Building is shared; INSTALLING is not and cannot be — the base uploads straight onto its batch-of-one
context, the analysis builds before its chunk context exists and uploads per chunk. HVDC stays inline on
the base path for the same reason: it applies its freeze into that solve's own target overrides.

Two accidental divergences fell out of merging them. The base constructed a fresh
`GpuBatchElementMapper` per builder, three times over, where the analysis built one and shared it. And a
parameter: the base fed the flow builder the incremental transformer REACTIVE POWER loop's
`getMaxTapShift()` while I merged it onto a `Request` field I had named after the transformer VOLTAGE
loop, because the `BatchConfig` field it comes from on the analysis side is named
`incTvcMaxTapShift` and holds the ratio-tap-control value. The test caught it at 1.3e-4 kV on BUS_2 —
`deviceIncrementalTransformerReactivePowerControlMatchesCpu` — and the field is named
`rpcMaxTapShift` now.

Behaviour-neutral otherwise: suite 261/0, rte6515 unchanged at 9040/9042 with the same two lines.

**Second half, same day:** the tables now TRAVEL as that record too. `solveQualifiedInChunks` took ten
positional phase arguments, several of the same type and several nullable, which is precisely how a new
phase gets mis-wired — a swapped pair compiles. They are one `GpuBatchPhases.Built` parameter now, the
equation-system context folded in with them, and the chunk solver's body reads them off the record.

### #58 — PLAN: what "every builder is total" would take (scoped 2026-09-16)

Totality would let the four host-pause slots go. Since #57 it is an OPTIMISATION, not a correctness
requirement: a builder that cannot represent a controller now hands the whole loop back loudly instead
of dropping it silently. So this is about how often a loop leaves the device, not about whether the
answer is right.

The nine skip sites fall into four classes, and only one of them is real work.

**1. Already impossible — turn the skip into an assertion.** `tapRows.length == 0` on a tap controller.
Since the tap table moved off reflection onto `PiModelArray.getModels()` (#52), an empty table means the
branch carries a continuous `SimplePiModel` and has no taps, which a voltage or phase controller never
does. Five sites. Cost: an afternoon, and it converts five "can't happen" branches into proof.

**2. Not skips at all — stop counting them.** `branch.isDisabled()`, `!isConnectedAtBothSides()`,
`shunt.getControllers().isEmpty()`. OLF's own loops skip these too, so representing them is not owed.
They share a `continue` with the real refusals today; splitting the two is mechanical.

**3. The real work: a variable that is not in the equation system.** `rhoRow` / `a1Row` / `vRow` /
`shuntBRow` / `rlIdx` null. One root cause: the row index is taken at extraction time, and several loops
run with their control DISABLED at the base, so the variable does not exist yet. The project already has
the fix pattern and uses it twice — `GpuBatchContinuousPhase` toggles a CONTROLLER-mode control on and
diffs the activation, `GpuBatchTransformerControl` does the ON/OFF toggle diff — and the standing lesson
says it outright: never key a pack or a guard on control-ENABLED state, key it on the equations
EXISTING. Generalising the toggle-and-diff to every builder is the piece that would move the needle.

**4. Genuinely unmodelled, and honest about it.** A control element that is zero-impedance in AC (the
element mapper packs those through the zero-impedance family, not as a branch), and a controller or SVC
shunt whose conductance varies with section (`elemIndex < 0 && gVaries`) — the device cannot carry the g
change for a shunt it does not pack. These want extraction work or a whole-run gate, not a per-controller
skip.

**Do not chase this blind.** Zero skips on rte6515 and across the whole 261-test suite, so there is no
measured case to fix. `GpuBatchSkips.summary()` is on every run and both routes log when it is non-empty:
let a production network name a class-3 or class-4 case, then fix that one.

### #57 — One ownership rule for both routes, and the partial-representation hole (2026-09-16)

Asked why the four host-pause slots are needed when every loop has a kernel. Answering it found the
hole they were only half covering.

Every phase builder walks OLF's controllers and SKIPS the ones it cannot express — a control whose
variable is not in the equation system, a branch the extraction does not pack. Nine such sites, all of
which only LOGGED. That gives three cases, and only two were handled:

| builder result | what happened | correct? |
| --- | --- | --- |
| complete | kernel drives the loop | yes |
| EMPTY | kernel declines, the base pauses and the SA refuses | yes |
| PARTIAL | kernel takes the loop, the pause slot is CLEARED, the skipped controllers are dropped | NO |

The partial case is the dangerous one precisely because nothing flags it: the device drives a loop on a
subset of what OLF drives and every equation still converges. So the slots were the all-or-nothing net,
covering only the all-or-nothing case.

`GpuBatchSkips` now counts the skips per thread (the SA builds on OLF's partition threads) and
`GpuBatchPhaseOwnership` turns the count into the one decision both routes take: a loop is device-owned
only when its representation is COMPLETE. Incomplete hands it back — the base load flow clears the
kernel tables and restores the host-pause bit so OLF's own loop runs at its own slot, and the batched
SA, which has no pause slot for anything but secondary voltage control, routes the run to the CPU.

**Measured before enforcing**: zero skips on rte6515 and across the whole 261-test suite. So the rule
is behaviour-neutral on everything we validate against, which is also why the hole had never shown up —
and why it would have been found the hard way, on a customer network, without this.

The slots stay. Deleting them needs every builder to be provably total, which is a different piece of
work from porting AreaInterchangeControl and AutomationSystem: those two never used the pause
mechanism at all.

### #56 — The base load flow takes the same route as the SA (2026-09-16)

`SecondaryVoltageControl` was the one in-kernel phase the base load flow never handed to the driver.
The kernel landed on the chunk path, where it is the default; `GpuNewtonRaphson` builds every other
phase (monitoring, incremental TVC, incremental shunt, the flow-observed family, continuous phase,
discrete TVC with the generator freeze, simple shunt) and never built this one. So
`buildHostLoopMask` set its bit and nothing cleared it: every network with a zone PAUSED the driver
there, by construction rather than by refusal — while the same network ran it on device under the
security analysis. Two routes through the same loop, which is the divergence class this project keeps
paying for.

Now built and uploaded the same way, from the same builder, with the same inputs. Everything it needs
was already on the base path: the reactive-limit rows, the state vector, the plausible-target-V bounds
off `getNetworkParameters()`, an element mapper.

**What the chunk path did not need and the base path does: a write-back.** The batched SA restores the
network per scenario, so the moved zone targets never have to reach it; a load flow has to leave them
on the network exactly as OLF's own loop does. The first attempt read the moved target out of the
CONTROLLED bus's row, which looks right and is wrong — that row carries a Q target the moment the bus
is pinned or is a shared-distribution member, and writing 0.165 pu in as a target voltage puts the
control outside the plausible band, so OLF drops it and the run stops batching. Two JAVA-mode tests
caught it. The targets now come from the phase's own state (`dSvcTargetV`, read back per
(scenario, controller)), which is the thing the kernel actually settled on.

This also closes the most reachable route into the restart gap: a paused loop whose mutation dirties
the structure is what cancels the pause and forces a restart, and moving generator VC targets is
exactly that kind of mutation.

### #55 — Freeze a REMOTE generator voltage controller in-kernel (landed 2026-09-16)

The last whole-run CPU gate in the AFTER_GENERATOR phase. `freezeCandidates` returns null the moment
one candidate is a REMOTE controller, because the kernel's freeze writes a V identity row
(`mode 0/2 -> 1, target = qBuf - loadQ`) and a remote controller's own row is a DISTR_Q SHARE, not a V
identity. OLF itself does not care: `disableGeneratorVoltageControlsUnderMaxControlledNominalVoltage`
walks `getMergedControllerElements()` and freezes every enabled one at `getQ().eval()`, local or
remote alike. So this is a device representation gap, not a semantic one.

**The primitive already exists.** A frozen remote member is the reactive-limit remote PIN with a
different target: row mode 4 ("the row hosts its own Q balance"), `target = qBuf - loadQ`, saved mode
and target for the restore — which is exactly what `rlCheckBatch`'s remote branch writes, with a Q
LIMIT as the target instead of the current injection. `qBuf` at a member's own row is already the
member's own reactive injection (`fillRemoteMemberQBatch`), and OLF's freeze target is
`getQ().eval() - loadTargetQ`, the same expression the local freeze uses.

**What comes for free, and what does not.** Checked against the kernels:

- carrier relocation and the all-members-gone pilot flip key on `mode` (`relocChoose` counts members
  at mode 1, `updatePilotVeqBatch` flips the pilot's V identity when none is left), so a frozen member
  at mode 4 is handled like a pinned one — nothing to do;
- the DISTR_Q key renormalization keys on `limitType[(scenario, entry)] != 0`, NOT on the mode. A
  freeze that only wrote mode 4 would leave the survivors on their old shares while the frozen member
  stopped participating. It needs its own limit type.

**Shape of the change.** Add limit type 5, "frozen by the generator voltage control manager", beside
1 MIN_Q / 2 MAX_Q / 3 MIN_REALISTIC_V / 4 MAX_REALISTIC_V — the renormalization's `!= 0` test then
drops the member with no change, while the type stays distinguishable from a Q-limit pin.
`freezeCandidates` keeps remote candidates and records each one's RL entry index (−1 when local);
`FreezeTables` carries it; the kernel freeze branches on it; `rlCheckBatch` skips an entry at type 5 in
BOTH branches, since OLF's reactive-limits loop never sees a bus whose voltage control is off; the
COMPLETE restore puts mode, target and limit type back. The packed RL state already has three bits for
the type, so no JNI width change.

**Landed, and the test earned its keep.** Built as scoped, plus one thing the plan missed: freezing a
remote controller can take the LAST live member of its group, and then the group controls nothing —
OLF's freeze disables the voltage control, so the CONTROLLED bus stops carrying a V identity. That
flip is `updatePilotVeqBatch`, which the reactive-limits phase runs after its own pins; the freeze
happens two loops later (OLF order 8 against 6), so the pilot held a target for a control that no
longer existed until the next sweep came back round to RL. Measured 0.6 kV on BUS_2. The same kernel
now runs after the discrete transformer phase too, and it restores the identity at COMPLETE on the
same rule.

The test builds a remote controller whose CONTROLLED bus is under the transformer-controlled level
WITHOUT being the bus the transformers control — putting it there takes their control away and the
whole thing passes vacuously, which is what the first attempt did (`controllers=0`). Two settings: the
automatic limit freezes only the remote controller, the not-controlled side keeps a PV bus, the #53
bracket decides locally and the scenario BATCHES at 1e-12 with the freeze in-kernel; forcing the limit
to 150 kV freezes the local generator too, empties the min mask, and the bracket sends the scenario to
the CPU, where the two arms still have to agree.

rte6515 unchanged at 9040/9042 with the same two lines.

### #54 — A second SA run in one JVM re-pays the cuDSS ANALYSIS (found 2026-09-16)

Measured, not suspected. `OLF_GPU_TRACE=1` now tags every ANALYSIS with the context's batch size, and
the production bench (three GPU runs: a 20-scenario qualification pass, then cold and warm at 250)
prints six:

```
ubatch nScenarios=1   LU_NNZ=260076      <- base load flow, run 1
ubatch nScenarios=20  LU_NNZ=260076
ubatch nScenarios=1   LU_NNZ=260076      <- base load flow, run 2
CONTEXT_EVICT cached chunk=20 -> requested chunk=250
ubatch nScenarios=250 LU_NNZ=260076
ubatch nScenarios=1   LU_NNZ=260076      <- base load flow, run 3
CONTEXT_EVICT cached chunk=250 -> requested chunk=250
ubatch nScenarios=250 LU_NNZ=260076
```

Within ONE analysis the contingencies are fine: the chunk context is acquired before both the chunk
loop and the strategy-pair group, `ctx->analyzed` latches, and every chunk of every group shares that
one ANALYSIS. The base load flow pays its own, for a batch of one.

The third eviction is the finding: `chunk=250 -> chunk=250`, the same batch size and the same LU_NNZ, so the
PATTERN is identical and the symbolic analysis would have been reusable. `acquireBatchContext` misses
because it keys on `GpuAcData.sameAs`, which compares element PARAMETER VALUES (`cbIn` and friends) —
and the previous run left the network at different tap positions. The base load flow already has the
right pattern for this: `sameStructureInputsAs` + `refreshStructureValues`, reuse the device structure
and push the new values.

What makes it more than a one-line change: `createBatchContext` takes the reactive-limit and
distributed-slack packs as CONSTRUCTOR arguments (`rlMinQ`, `rlVtarget`, `dsFactor`, …) and
`GpuBatchOuterLoopTables.install` does not re-upload them, so reusing a context across runs needs
those packs to become setters like the channels installed beside them.

**Size it against the measured cost.** The trace now times the ANALYSIS: ~64 ms at the chunk's batch
size, ~80 ms for the base load flow's batch of one, and the cost is INDEPENDENT of the batch size
(63.4 ms at 20, 63.6 ms at 250) because UBATCH analyzes a single block and carries the batch size as
`CUDSS_CONFIG_UBATCH_SIZE`. So ~144 ms per run, about 1.8% of a 7.9 s warm bench and 0.2% of the
82 s rte6515 run. Worth doing, not urgent.

**And one analysis CANNOT serve two batch sizes — tested, not assumed.** A COST that does not move with
the batch size invites analyzing once and re-setting `CUDSS_CONFIG_UBATCH_SIZE`, which is what the
compaction path's tier change would avoid. Tried: cuDSS ACCEPTS the resize without a new analysis, the
production bench ran 11% faster warm (7073 ms vs 7917 ms) with EXACT violation parity, and then
pegase9241 died with `CUDA error: an illegal memory access was encountered` — the data object's
buffers are sized for the batch size it was analyzed at. The analysis does not cost more for a bigger
batch, but its STATE belongs to that batch size. The compaction path's destroy-and-re-analyze is right; the note there now says
why, so nobody re-derives the same false economy. A bench that gets faster while corrupting memory is
the reason to run the whole suite on a perf idea, not just the bench.

### #30 — Multi-stage conditional actions
Pending. The only item left **in** the operator-strategy machinery; everything else in #24
is done. Single-stage conditions (True / violation / threshold) all qualify today via
speculative precompute. Stage-on-stage needs the stage-1 state evaluated before stage 2 is
composed — i.e. a second warm continuation off the pair's own result, not one extra
scenario. That does not fit the current "a (ctg, strategy) pair is one extra BatchContingency"
model, which is why it is the remaining architectural piece.

### #34 — Slack-relocating contingencies (`ISLANDS_SLACK`)
Pending, and **measure before building**. #33 left two node-breaker shapes on CPU. The big
one: an outage that isolates the slack. CPU (`PropagatedContingency`) relocates the slack
into the largest connected component and re-anchors there; the batched pack anchors its
targets and distributed-slack participation on the BASE slack row, so a relocated slack is
not expressible as a per-scenario mask. FIRST instrument the `ISLANDS_SLACK` rejection
counter on the production 500-ctg bench to learn what share of a real node-breaker run this
costs — build relocation support only if the share justifies it. Second, smaller shape: a
switch outage leaving BOTH sides energized (the zero-impedance spanning tree would have to
be re-picked per scenario).

### #39 — Tap action on a transformer under ACTIVE voltage control
Pending; likely cheaper than when it was first carved out. `RatioTapChangerTapPositionAction`
is batched, but a transformer under active voltage control falls back to CPU, because a
controlled transformer's rho is a solver VARIABLE whose row moves (`computeQRow` relocates
the controlled bus's Q balance onto the `BRANCH_RHO1` column), so a static full-row override
cannot express the tap change. Two things changed since: transformer voltage control is now
batched (#16), and the 2026-07-20 phase-control work established exactly how control rows
pair (`BRANCH_TARGET_RHO1` / `BRANCH_TARGET_ALPHA1` identity rows vs the power row they swap
against, and how a row-mode identity flip expresses "control off, value pinned"). Plausibly
just: pin rho via its identity row at the action's tap value.
**Check CPU semantics first** — does `LfRatioTapChangerAction` on a regulating transformer
disable the control or keep regulating from the new position? Mirror it; do not invent.

### #27 — Batched-SA perf at the RTE 1e-4 criterion — RESOLVED by operating point
Kept open only for optional work. The original framing ("GPU loses at 1e-4, recover it") is
moot operationally: the per-arm eps split (`7ac0627`) measured, in ONE controlled run,
**GPU@1e-7 warm 23.52 ms/ctg vs CPU@1e-4 33.45 ms/ctg → ×1.42**, violations 305=305,
statuses identical. GPU@1e-7 is a production-or-BETTER (tighter) answer delivered faster
than the CPU delivers its production answer.
Decisive supporting finding: the CPU is nearly **eps-insensitive** (1e-4 → 33.45 ≈ 1e-7 →
33.32), so it cannot recover by loosening; the historical "CPU@1e-4 ≈ 25" figures were
cross-run machine-state noise, not an eps effect — which is why an earlier paper estimate of
~×1.1 was wrong and the same-run measurement mattered.
Remaining OPTIONAL: make GPU@1e-4 itself fast by killing the outer-loop churn (levers:
decision hysteresis — only re-check RL on scenarios whose q moved > tol; narrow fills/norms
when few scenarios are active). Worth it only if a consumer insists on eps=1e-4 on the GPU.
Keep the fixed-iter bench and the SA tests at baseline for any such work.

### #40 — Batched transposed solve (adjoint/VJP) — ONE primitive, THREE consumers
Not started. The single highest-leverage missing capability on the batched device path,
because three separate roadmap items all need the *same* thing: **a transposed solve against
the per-scenario cuDSS factorization we have already computed**, without shipping Jacobians
back to the host.

**Consumer 1 — GNN training with an implicit load-flow layer** (the driver). A converged AC
LF is an implicit function `F(x, p) = 0`; by the implicit function theorem
`dx*/dp = −J⁻¹ ∂F/∂p`. Training a GNN whose forward pass contains an LF layer means
backpropagating through it, which in JAX is a `custom_vjp` whose backward is the adjoint

```
Jᵀ λ = (∂y/∂x)ᵀ ȳ          ← ONE transposed solve, reusing the factorization
p̄    = (∂y/∂p)ᵀ ȳ − (∂F/∂p)ᵀ λ
```

Reverse mode is what training needs (many parameters → one scalar loss), so this is one
transposed solve total, versus one solve per variable in the forward-mode sensitivity
analysis that exists today. Design already worked out in
`~/Projects/energnn/project-tertiary-voltage-control/docs/`:
`differentiable_power_flow.md` (the `custom_vjp` boundary, the phased plan, and §4's list of
what OLF must expose), `adjoint_block_and_vjp_math.md`, `energnn_vjp_training_integration.md`.
The reverse-mode derivation — specifically *why* each transpose is forced rather than a
layout convention — is in `~/Projects/powsybl/powerflow/pyloadflow/docs/reverse_mode_vjp_explained.md`.

**Consumer 2 — SecondaryVoltageControl, batched.** ~~Its sensitivity RHS is built on existing
`BUS_TARGET_V` rows and solved with `j.solveTransposed`~~ — NOT a consumer (corrected
2026-09-15): OLF's `solveTransposed` is the Newton direction, and the batched port already
solves it forward on the driver's factors (see the loop table below).

**Consumer 3 — IncrementalPhaseControl, batched.** Same primitive: a transposed solve with
`toRadians(1)` injected at the `BRANCH_TARGET_ALPHA1` row, one column per controller.

**PRIOR ART — do not start from scratch. Three layers already exist:**

1. **A working CPU adjoint in OLF.** `~/Projects/powsybl/powsybl-open-loadflow-5`, branch
   **`adjoint-vjp`** (~12 commits, head `a9de15a3d`). `AcSensitivityAnalysis.runAdjoint(...)`
   is exposed as a public SPI entry on `OpenSensitivityAnalysisProvider`. It contracts
   `θ̄ = Sᵀ·ȳ` **without materialising S**, using a single
   `context.getJacobianMatrix().solveTransposed(factorsStates)` and an **O(functions+variables)
   minimal factor set** (validated against the full cross product). The public surface is
   deliberately small: `runAdjoint`, `AdjointBlock`/`AdjointVariable`, `functionCotangentKey`.
   Returns the UNSCALED sensitivity (physical dual of `get_sensitivity_matrix`).
   10 tests in `AcSensitivityAnalysisAdjointTest`, including **finite-difference gates**
   (`runAdjointMatchesForwardSensitivityAndFiniteDifferenceOnIeee14`), distributed slack, the
   SVC pilot lever, line admittance, linearity in the cotangents, and silent-failure guards
   (rejects a cotangent keying no block function; rejects empty declarations) — the commit
   log shows those guards were added after real silent-zero/silent-drop bugs.
   **This is the semantics the GPU path must reproduce, not reinvent.**
2. **The lifetime mechanism, already solved on CPU.** `runAdjoint` reuses the AC load flow
   retained in `NetworkCache` (`networkCacheEnabled`) — the cached LF *is* how the
   factorization survives between the forward and the adjoint call. The GPU equivalent is the
   open question in item 2 below.
3. **The Python prototype that preceded this OLF GPU work** —
   `~/Projects/powsybl/powerflow/pyloadflow`. `docs/reverse_mode_vjp_explained.md` derives why
   each transpose is *forced* vs a layout convention; `docs/jacobians_and_implicit_diff.md`;
   and `docs/batched_solves_cpu_gpu.tex` works the batching design explicitly, including
   §"Scenario B — the forward *with* a gradient (VJP)" and §"Four ways to lay out B systems".

**What the batched GPU path already has:** UBATCH cuDSS with a per-scenario LU over a fixed
superset CSR pattern — exactly the "uniform batch" layout that the tex note identifies as
correct when members share a sparsity pattern (as N-1 scenarios do).

**What is missing — i.e. the GPU/batched delta over the working CPU adjoint:**
1. **Batched transposed solve against the existing per-scenario factorization** — cuDSS
   solve-transpose on the factors already computed, no refactorization, no host round-trip.
   CPU does this with one `solveTransposed` on a retained `JacobianMatrix`; the device
   equivalent over a UBATCH of factors is the core new piece.
2. **Factorization LIFETIME across the fwd/bwd boundary.** The requirement that differs in
   kind from security analysis, and the easiest to underestimate. SA frees its factors as soon
   as the forward converges; a VJP must keep the per-sample factors alive *between* `f_fwd`
   and `f_bwd`, and JAX may separate, reorder under `jit`, or repeat those calls. CPU solved
   this via `NetworkCache`; the device needs a context with `create → adjoint_solve → free`,
   an opaque handle in the `custom_vjp` residual, matrices staying in device memory, and
   thread-safe re-entrancy for parallel training. NB our batched solver already had a
   JVM-global-state incident (#38) — do not repeat it; make context state per-call.
3. **`(∂y/∂x)ᵀ` RHS extraction and `(∂F/∂p)ᵀ` products** on device for a declared
   function/variable set (CPU's `buildAdjointFactors` + the scatter of `der()` into equation
   rows is the reference implementation).
4. **Batching for different networks — cuDSS's NON-UNIFORM batch is the primary answer, and
   pyloadflow already validated it.** Training runs on **different networks**, so the uniform
   UBATCH the entire SA path rests on (one shared sparsity pattern, B sets of values) **does
   not apply to a training batch**. Of the four layouts in
   `pyloadflow/docs/batched_solves_cpu_gpu.tex` §"Four ways to lay out B systems":
   - **(3) non-uniform batch — the fit.** cuDSS accepts a *list* of differently-patterned
     matrices as one batched call: each member keeps its own pattern, the library overlaps
     their work; no shared symbolic analysis, but one launch instead of B. Available since
     **cuDSS ≥ 0.4.0** (we pin 0.8.0). pyloadflow states the batched *"non-uniform / uniform
     cuDSS solve confirmed for the VJP"* (`docs/gpu_security_analysis.md`), and its SA design
     already **buckets** work into pattern-preserving → uniform vs topology-changing →
     non-uniform, with a classifier in `jacobian/pattern.py`. Port that bucketing idea rather
     than reinventing: different networks simply always land in the non-uniform bucket.
   - **(4) block-diagonal / fused** — glue the B grids onto one diagonal: one analysis + one
     factorization, no fill between independent blocks, and the transpose of a block-diagonal
     is block-diagonal with each block transposed, so the orientation fix composes unchanged.
     Viable alternative, but its symbolic analysis is per batch *composition*, so it only
     amortizes if membership is stable across steps; reshuffled minibatches could make
     re-analysis dominate.
   Decide by measurement (non-uniform's per-member analysis vs block-diagonal's per-composition
   analysis), not by argument. **Do not treat block-diagonal as the only option** — the
   non-uniform path is the one with existing evidence behind it.
   Note consumers 2 and 3 (SVC, incremental phase control) run INSIDE the SA batch where the
   pattern IS shared, so they stay on the uniform path and never hit this — another reason to
   do them first.
5. **Non-smoothness.** `x*(p)` is piecewise-smooth: PV↔PQ switching on Q-limits, tap limits,
   distributed-slack participation changes. The correct derivative is that of the locally
   active KKT system, so the active set must be exposed and freezable while linearizing.
   `differentiable_power_flow.md` §6 calls this the correctness frontier — prototype on a
   network with known binding Q-limits before trusting it broadly. (CPU's adjoint already
   handles distributed slack and the SVC pilot lever, so check what it does there first.)

**THE ORIENTATION PROBLEM — the crux, and why the CPU adjoint does not port.**
CPU and GPU store the Jacobian in **opposite conventions**, deliberately:

- **GPU** (`olf_gpu_nr_jni.cu` header): *"J is true CSR with row = equation column, col =
  variable row, so cuDSS solves `J dx = -F` directly — no transposition games (contrast
  powsybl-math's CSC-as-CSR convention, which exists to serve solveTransposed)."*
- **CPU** (powsybl-math): CSC-as-CSR, i.e. the stored matrix **is `M = Jᵀ`**.

The consequence is the opposite of what one first assumes, and the CPU adjoint says so in
code: `// λ = J⁻ᵀ x̄: the stored matrix is M = Jᵀ, so plain solve() IS the adjoint solve. One
solve.` → `context.getJacobianMatrix().solve(lambda)`. On CPU the **backward is a plain
`solve()`**; `solveTransposed()` is what the **forward** sensitivity uses (`J (dx/dp) = −∂F/∂p`).

**On CPU BOTH directions are free, and the convention is not what makes them so.** KLU
implements `klu_solve` *and* `klu_tsolve` against the **same numeric factorization** — a
transpose solve is just the triangular solves taken in the other order, at the same cost and
with no refactorization. So one factorization serves both directions. The storage convention
only decides which method *name* maps to which mathematical direction; it costs nothing
either way:

| | stored | one factorization gives | |
|---|---|---|---|
| CPU (powsybl-math + KLU) | `Jᵀ` | `solve()` → `J⁻ᵀ` (**backward**) | `solveTransposed()` → `J⁻¹` (**forward**) |
| GPU (ours, true CSR + cuDSS) | `J` | `solve()` → `J⁻¹` (**forward**) | **transpose solve: not available** |

**So the problem is not our convention — it is a missing cuDSS capability.** cuDSS (pinned
0.8.0) offers only the non-transposed solve against its factors, so a GPU factorization yields
**one** direction instead of two. Flipping our storage convention would not help: it would
merely move the unavailable direction from the adjoint to the Newton, which is strictly worse
(the Newton does many solves per sample, the adjoint one at the end).

**The precise ask of cuDSS is parity with KLU: a transpose solve against an existing
factorization.** The cuDSS team has said this is in progress. The moment it ships, our
true-CSR convention costs nothing, the workaround below deletes, and the adjoint becomes one
extra solve rather than an extra factorization.

**Resolution while cuDSS lacks it (works today):** keep the forward Newton in its current
orientation and, at convergence, build `Jᵀ` explicitly, factorize it **once**, then the adjoint
is a plain non-transposed cuDSS solve. This buys the second direction with a second
factorization — precisely the cost KLU does not pay.
- The value transpose is a **gather with a precomputed permutation**: for a given sparsity
  pattern the `CSR(J) → CSR(Jᵀ)` index map is symbolic, computed once per distinct pattern,
  then each sample's values are permuted by one kernel. cuDSS's symbolic analysis for `Jᵀ` is
  likewise once per pattern.
- Cost: **+1 factorization per sample per training step**, against the ~4–8 refactorizations
  the forward Newton already performs — roughly 12–25% overhead. That is the price of no
  solve-transpose, and it is acceptable.

**Build the workaround to be DELETED.** When cuDSS ships the transpose solve, the explicit
`Jᵀ` build *and* its factorization both disappear and the GPU reaches KLU parity: one
factorization, both directions. So put the adjoint behind a small seam (one function: "solve
with `Jᵀ` given the forward's factors") whose first implementation is transpose-and-
refactorize, so the swap is local rather than a rewrite. Re-check on every cuDSS bump —
`native/cudss-swap.sh` already does the A/B, and cuDSS version choice is already tracked as a
decisive perf lever.
(If cuDSS instead exposes the raw `L`/`U` factors, the same seam takes the other classical
route: `J = LU` ⇒ `Jᵀ = UᵀLᵀ`, so an adjoint solve is two transposed triangular solves against
the existing factors — again one factorization, which is exactly how KLU's `klu_tsolve`
works.)

**Active-set / control handling in the CPU adjoint (checked 2026-07-21).** The reassuring
finding for the GPU port: **both distributed slack and the SVC pilot are handled entirely on
the RHS (`∂F/∂p`) side — neither touches `J`, its factorization, or the transpose machinery.**

- **Distributed slack** enters only as `slackParticipationByBus`
  (`AcSensitivityAnalysis:447-456`), folded into the RHS by
  `initFactorsRhs → fillRhsSensitivityVariable → factorGroup.fillRhs(rhs, participationByBus)`
  (`AbstractSensitivityAnalysis:772-794`). With `isDistributedSlack()` it is
  `getParticipatingElements(...)` mapped to `-factor` per bus; otherwise the single entry
  `{slack bus: -1.0}`. So a parameter perturbation carries its slack redistribution inside the
  `∂F/∂p` column. Note this takes the participation **at the converged state** and treats it as
  fixed — i.e. it IS the "freeze the locally-active branch" linearization the differentiability
  design calls for, already in practice.
- **SVC pilot closed loop** is `fillSvcPilotFactorsRhs` (`AcSensitivityAnalysis:250-296`) →
  `SvcPilotPointClosedLoopSensitivity.buildCoordination(context)` →
  `coordination.weightsForPilot(pilotBus)`, which fills that variable group's RHS column with
  per-pilot weights over the controller buses. The coordination system is a **separate small
  matrix with its own `LUDecomposition`** (`SvcPilotPointClosedLoopSensitivity.Coordination`,
  field `luB`), NOT a modification of the LF Jacobian — and it is built + factorized **once**
  because it is pilot-independent, then reused for every queried pilot.
- The adjoint reuses the forward's RHS builder verbatim: `analyseAdjoint` calls the same
  `initFactorsRhs` + `fillSvcPilotFactorsRhs`, commented *"∂F/∂p columns (one per variable
  group), including the SVC pilot closed-loop — identical to forward."*

**DECIDED — the SVC coordination matrix stays on CPU in the end-goal design, for the adjoint.**
It is small, and per converged state it costs one LU plus one back-substitution per pilot. It
is pilot-independent, so it is built once per state and reused for every pilot. Porting it to
device would be substantial work (a second solver path for a small system, exactly the regime
where a GPU is latency-bound and loses) for no measurable gain. For the adjoint the data
crossing the boundary is small — per-pilot weights used to fill an RHS column — and the host
boundary is already being crossed anyway, since the cotangents come from the host. So: keep it
on CPU, deliberately, and do not revisit.

**ALSO SETTLED — the SVC OUTER LOOP may stay host-side in batched SA too. Investigated
2026-07-21; my earlier "the synchronization could dominate" worry was overstated.**
The batched architecture is ALREADY hybrid, and transformer voltage control (#16) is the
proven production precedent:
- **On device, inside the native solve loop:** distributed slack + reactive limits — the HOT
  loops, iterating many times per solve, where host round-trips would be fatal.
- **On host, orchestrated around whole batched solves:** transformer voltage control.
  `orchestrateTransformerControl` reads `chunkResults[slot]` (the converged state per
  scenario), loops `for (int step = 0; step < 8; step++)`, and each step rebuilds per-scenario
  `x0` / row-modes / targets and calls `solveBatchContext` again, copying `r2[slot]` back for
  the affected slots.

The decisive fact: **`solveBatchContext` already returns the full per-scenario state to the
host after every batched solve.** A host-orchestrated outer loop therefore costs **no extra
transfer** — that data crosses regardless. And the device fills honour an `active[s]` mask
(15 sites in the JNI), so an orchestration step only does work for the scenarios still moving.
So the state changing per iteration is not an objection: re-solving on changed state is exactly
what the orchestrator does, and TVC already does it in production under RTE parameters.

What this means for SVC: **follow the TVC pattern** — host-side orchestration around batched
solves, with the coordination matrix built on host per converged state (see above). The real
cost driver is **iteration count, not data volume**: each orchestration step is one batched
re-solve. TVC bounds itself at 8 steps; SVC must be similarly bounded, so check its
convergence behaviour before wiring it. The residual host cost is B small coordination
builds+LUs per outer iteration — at B ≈ 500 these are tiny matrices and parallelizable across
host threads, so measure it but do not pre-optimize. Only if that measurement surprises does
batching the B small systems (block-diagonal / non-uniform cuDSS) become worth it.

**Consequences for the GPU work:** (a) the transpose/factorization design above is orthogonal
to both controls — they do not complicate `Jᵀ`; (b) what must be ported is the **RHS
construction** (participation-weighted for slack, coordination-weighted for the SVC pilot),
not new solver machinery; (c) the SVC coordination's build-once/reuse-for-all-pilots structure
is the pattern to keep on device; (d) the "freeze the active set at the converged point"
question is, for distributed slack at least, already answered by existing practice — mirror
it rather than inventing a convention.

**Acceptance gate, inherited from the CPU work:** three-way agreement — finite difference
(independent ground truth) vs forward sensitivity vs the adjoint. **The orientation mismatch
makes this gate non-optional**: an inverted convention produces a plausible-looking but wrong
gradient, and FD is what catches it. The two analytic paths
should agree to solver tolerance (same kernels, opposite contraction order); FD agrees to
truncation error and catches a systematic error common to both. `AcSensitivityAnalysisAdjointTest`
already encodes this pattern; mirror it rather than inventing a new gate.

**Sequencing note:** consumers 2 and 3 are self-contained inside our batch and would prove
the primitive on ground we control, before the JAX/pypowsybl context plumbing of consumer 1.

**FIRST STEPS (ordered).** Principle: prove the risky primitive on ground we control, before
any JAX or batching complexity.

**Prerequisite — how the CPU adjoint reaches our build. DONE 2026-07-21** (rebased onto main,
`git merge --squash adjoint-vjp` as `ec2063108` "Bring in the CPU adjoint (VJP) from adjoint-vjp",
pushed to `gb`). Plan and reasoning kept below because it documents *why* squash rather than
cherry-pick. The GPU extension builds against OLF `2.4.0-SNAPSHOT` from
`powsybl-open-loadflow-3` (`olf-local-patches`); `runAdjoint` lives on `adjoint-vjp` in
`powsybl-open-loadflow-5`. Facts checked:
- **All the OLF checkouts are WORKTREES OF ONE REPOSITORY** (`git worktree list`), so every
  object is already shared: `adjoint-vjp`'s tip `a9de15a3d` is directly reachable from OLF-3.
  **No remote, no fetch needed** — cherry-pick/merge works across the worktrees as-is.
- **`olf-local-patches` is 2 ahead / 2 BEHIND main** (merge base `6ef8e6fc7` "Bump to
  v2.4.0-SNAPSHOT"), while `adjoint-vjp` is based on current main. **So rebase
  `olf-local-patches` onto main FIRST**, then take the adjoint work. (Main's `fe0bd3dfb`
  = #1451, which adds outer-loop names to status text, touches ~60 files — that, not our work,
  is what makes a raw two-tip diff look large.)
- **`olf-local-patches` does NOT contain shunt or SVC sensitivity** — its only 2 commits are the
  shunt *reconnection action* work (#32: `TerminalsConnectionAction` on a base-open LINEAR
  shunt + the `getShuntIdsToShift()` accessor). **It does not need to:** both
  `SvcPilotPointClosedLoopSensitivity` (+ its test) and `SHUNT_COMPENSATOR_SUSCEPTANCE`
  sensitivity are **already on main**, so the rebase brings them in.
- **What to take from `adjoint-vjp`:** 21 commits vs main, but not all are wanted —
  `e7ed4632b` is already upstream as main's `fe0bd3dfb` (#1451), and four prototype commits
  are later deleted wholesale by `168999764` ("Remove the superseded
  LoadFlowAdjoint/ClassicVoltageControlVjp prototype", −608 lines). Cherry-picking the range
  verbatim would add-then-delete a prototype. **Prefer `git merge --squash adjoint-vjp` after
  the rebase** — it takes the tip state in one commit with no prototype churn. Collision risk
  is low: the adjoint touches only 9 files, almost all under `sensi/`, disjoint from the
  shunt-action work.
- After the rebase: rebuild + reinstall OLF into the isolated m2 and re-run the GPU suite, since
  the extension builds against it (this is also the moment to re-check the pinned
  `powsybl-open-loadflow.version`).

The CPU adjoint is the ORACLE for every step below, and an exact reference is far stronger than
finite differences alone — which is why it is worth doing this properly rather than comparing
against recorded numbers.

**Steps 1a and 1b are DONE (2026-07-22).** `GpuTransposedSolveTest` proves the transpose machinery
on synthetic non-symmetric matrices (~1e-16, with a guard test that transposing changes the answer),
and `GpuAdjointOnRealJacobianTest` now passes on a real ieee14 Jacobian:
`vs solve abs=5.3e-15 rel=7.4e-16 | vs solveTransposed abs=6.06e+00 | residual=5.6e-15`.
**The orientation claim above is CONFIRMED by measurement, not inverted:** powsybl's CSC arrays read
as CSR are already one transpose, that is the matrix KLU factorizes, so OLF's plain `solve()` IS the
adjoint direction and our single GPU transpose lands on it.

> **Pivoting is a hard prerequisite for any GPU solve on a RAW Jacobian.** 1b failed at 7.164e-03 for
> a week on a misdiagnosis ("stale extraction"). The extraction was always correct — a dense solve
> built from those exact arrays matches `solve()` to 6.2e-15. The real cause: the Jacobian has
> **5 exactly-zero diagonal entries and 0 of 28 diagonally dominant rows**, while `applyReorderAlg`
> forces AMD with no pivoting configured; cuDSS substitutes a static perturbation for the zero pivot
> and returns a wrong `x` **with a matching residual**, which is why it read as a correctness bug.
> Measured: AMD 7.16e-03, nested dissection 7.18e-02, natural order 6.2e-15, **AMD + MC64-style
> matching 3.6e-15** (the fix, `applyAdjointMatching`). Iterative refinement does NOT help (IR=0 and
> IR=3 bit-identical). The batched NR never hit this only because it builds a superset pattern with a
> guaranteed nonzero diagonal — **steps 4–5 must keep that guarantee or set matching explicitly.**
> Natural ordering is not an option at scale (fill explodes on grid patterns).

1. **Single-scenario GPU adjoint, no batching.** Run one network to convergence on the GPU,
   build `Jᵀ` via the permutation, factorize, adjoint-solve, contract to `θ̄`. Compare against
   `runAdjoint` on the same network with the same cotangents, plus an FD check. This isolates
   the highest bug risk — orientation/permutation — from everything else. **Deliberately pick
   a case with NO binding Q-limits**, so any mismatch is the transpose and not the active-set
   question. No batching, no JAX, no context lifetime.
2. **Measure the cost assumption early.** The "+1 factorization ≈ 12–25% overhead" above is an
   ESTIMATE, not a measurement. Time transpose-build + factorize against the forward's existing
   factorizations on a real case. If it is much worse, the sequencing changes — it may be
   better to wait for cuDSS's transpose solve than to build around the workaround. Cheap to
   check, and exactly the kind of assumption that has bitten this project before.
3. **Put in the seam** (one function: "solve with `Jᵀ` given the forward's factors", with
   transpose-and-refactorize as its only implementation) once step 1 proves correctness, so the
   cuDSS upgrade is a local swap rather than a rewrite.
4. **Extend to the uniform batch** — per-scenario `Jᵀ`, batched factorize, batched adjoint
   solve; still the shared-pattern UBATCH, still no JAX.
5. **Prove it through a real consumer: batched SVC or incremental phase control.** Needs no
   JAX, no non-uniform batching, no context-lifetime plumbing, and both are independently on
   the roadmap. If the primitive works here, it is real.

Only then: non-uniform batching (port pyloadflow's pattern bucketing) and the JAX/pypowsybl
context for training.

Verify cheaply alongside step 1: (a) that cuDSS 0.8.0 genuinely lacks a transpose-solve entry
point — confirm the API surface rather than assume; it also establishes the re-check procedure
for each bump; (b) what the CPU adjoint does for distributed slack and the SVC pilot, since it
already handles both (see "Active-set / control handling in the CPU adjoint" below).

### #18 — GPU-vs-CPU LF validation project
Blocked on RTE Nexus access.
**Standing constraint: never read or load `.arc` files or any real network data.**

### #67 — the non-batched tail is a CONVERGENCE problem, not a factorization one (2026-10-01)

**Why it was opened.** 117 of 6667 contingencies do not batch on rte6515's full busbar scope
(`GPU precomputed on device: 6550 / 6667`), and the no-CPU-fallback goal wants them on device.

**Where they actually go** (`OLF_GPU_REJECT_TRACE=1`, GPU-only run):

```
108  reason=max-iter-bad-solve
  5  reason=outer-loop-budget      (108 + 5 = 113 = the RUNTIME:status=-2 census exactly)
```

So the PLAN refusals in the census (`VC_REMERGE_MULTI=47`, `PINNED_MAIN=3`, `RELOC_VROW=3`,
`ISLANDS_SLACK=1`) are refusals of a plan VARIANT that later succeeds — they do NOT own the 117.
`max-iter-bad-solve` does: the scenario spent its whole iteration budget on Newton steps the
factorization got wrong (`lastSolveRatio > LIN_SOLVE_FAIL_RATIO = 1e-2`), so the device refuses to
call that OLF's answer. How bad, over the 184 bad-solve reject lines:

| last-solve ratio (gate 1e-2) | count | worst ‖F‖ |
|---|---|---|
| 1e-2 … 1e-1 (within one decade of the gate) | 126 | 5.52 |
| 1e-1 … 1 | 44 | 1.82 |
| > 1 | 14 | 1.44 |

No NaN/Inf anywhere — inaccurate solves, not divergences.

**The obvious fix was tried and it DOES NOT WORK.** The device already recovers exactly this on the
BASE load flow by rebuilding in mode 6 (`MODE_ROBUST`: mode 0's per-scenario loop with matching and
global column pivoting on cuDSS's single-matrix API) after a linear-solve reject; a CHUNK scenario
had no route to it and went to the CPU. So a second device pass was built —
`GpuBatchPrecomputer.retryRejectedOnDevice` + `BatchConfig.withRobustSolveMode()` + the mode choice
at `GpuBatchChunkSolver`'s `acquireBatchContext`. Result, full busbar scope:

```
GPU_ROBUST_RETRY attempted=113 recovered=0 statuses={-2=74, 1=39}
```

**Zero recovered.** Mode 6 turns 39 of the 113 from a bad solve into `STATUS_MAX_ITERATION` with a
SOUND last solve — so pivoting genuinely fixes those factorizations — and leaves 74 still rejecting
on a bad solve. Not one converges. **So "the unpivoted batch factorization is why they fail" is
FALSE for this structure.** These scenarios do not converge on the device within the budget, and
the CPU replay does converge them: that is the question to answer, and it is an equation/trajectory
question, not a linear-algebra one.

**A trap caught on the way.** The first version accepted `!needsCpuFallback()`, which also accepts
`STATUS_MAX_ITERATION`, and reported `recovered=43` — but device coverage stayed at 6550/6667 while
`DEVICE:MAX_ITERATION_REACHED` went 4 -> 47. Those 43 were not recovered; they were CPU answers
replaced by "MAX_ITERATION_REACHED, no network result" claims from the device. A retry may only
accept `STATUS_CONVERGED`.

**Kept, default OFF:** `-Dolf.gpu.sa.robustRetry=true` re-enables the pass so it can be re-tested on
another grid (same spirit as the EXPERIMENTAL batch modes 4/5). It costs a cuDSS analysis (switching
mode evicts the single-entry context cache) plus a full extra pass, and buys nothing here.

**Census artefact to know:** the retry re-runs qualification, so with the pass ON the `PLAN:` counts
DOUBLE (`VC_REMERGE_MULTI` 47 -> 94 etc.). They are per-attempt, not per-contingency.

**THE EQUATION SYSTEMS AGREE — checked on one of the 108 (2026-10-01).** The standing rule says a
non-converging GPU means a different equation system, so that was checked FIRST, with the two
dumps the code already pairs for it: `OLF_GPU_EQ_DUMP_CTG` (OLF's post-contingency equations,
inactive ones bracketed) against `OLF_GPU_ROWMODE_DUMP_CTG` (the device's per-scenario row modes)
plus `OLF_GPU_SA_EQ_DUMP` (the device's row map). Contingency `BUS-1039_BBS`, full busbar list kept
so the network is the real one:

| check | result |
|---|---|
| PV set: OLF `bus_target_v` ACTIVE vs device `BUS_V` mode 0 | 982 vs 982, EXACT |
| per-bus V-row occupant (target_v / distr_q / target_q / none vs mode 0/1/4) | **0 mismatches over 6237 non-relocated buses** |
| islanded set: OLF `disabled buses` vs device rows frozen to identity | 9 vs 9, and the SETS are equal (no bus on either side only) |
| device island freeze | 18 rows = 9 buses x (V->1.0, PHI->0.0) |

**And the iteration policy agrees too:** the RTE parameters set
`stateVectorScalingMode = MAX_VOLTAGE_CHANGE`, which `GpuSaRunScope` mirrors as `lsMode=2` with the
same `maxDv`/`maxDphi`, and `maxNewtonRaphsonIterations = 30` matches the reject's `it=30`. So same
system, same step clamp, same cap — and the device still needs more than 30 steps where OLF needs
fewer. The reject is at `outer=0 it=30`: it cannot finish even the FIRST Newton solve.

**THE TARGETS AGREE TOO.** `OLF_GPU_LOSS_TRACE=<contingency id>` (it takes an ID, not `1`):

```
device: deviceScheduledLoss=-1.243000000 pu islanded=[the same 9 buses] lostGens=[] lostLoads=[]
OLF:    activePowerLoss    =-1.243000000 pu (disconnectedGeneration=0, disconnectedLoad=1.243)
```

Identical to all nine decimals, same islanded set, and OLF's zero disconnected generation matches
the device's empty `lostGens`.

**ONE REAL EQUATION DIFFERENCE FOUND, AND IT IS EXACTLY 2 ROWS.** Resolving every dq-carrying device
row to its member bus — 40 on a member's own `BUS_V` row, 305 on a parent coupler's `DUMMY_Q` row
via the dump's `zi#K … dummyQ=ROW childSide=` mapping — and diffing against OLF's active set:

```
OLF active distr_q buses : 334
device dq rows LIVE      : 336     GATED (mode 4): 9     unmapped: 0
LIVE on device, INACTIVE in OLF (2): ['VL-5997_1', 'VL-6112_1']
ACTIVE in OLF, not live on device  (0): []
```

**Those two buses are already on record as the delicate ones** — `GpuAcActivationExtractor`'s own
comment: "the carriers of VL-5997 and VL-6112 came in at mode 4, the scan fell back to the identity
pilot rows, the raw q and the couplers' DUMMY_Q vanished (dead columns 13171/13207) and **every
scenario failed its first linear solve**". And [[#67]]'s rejects are at `outer=0`, the first solve.

Why OLF has them inactive, from the dump's own group lines:

```
gvc pilot=VL-5997_0 controlled=VL-5997_0 local=false pilotIsController=true members=2:
    VL-5997_0[vCol=13154 qRow=-1 qRowAlt=-244  enabled=FALSE]   <- the pilot, PINNED
    VL-5997_1[vCol=13156 qRow=-1 qRowAlt=14403 enabled=true ]
```

A pilot-self group of TWO whose pilot is pinned has exactly ONE enabled controller, so by OLF's rule
("DISTR_Q for every enabled controller except the FIRST, BUS_TARGET_Q for every pinned one") that
member IS the first enabled one and gets NO sharing equation. The device encodes the group
CANONICALLY (as if all controllers were enabled), which gives the member a DISTR_Q, and the
in-kernel mode scan is what must re-pick the carrier per scenario and gate it off. For this scenario
it did not: the row stayed mode 1 with live sharing, and the per-scenario row-mode diff contained
only the 18 island rows.

**CAUSALITY TESTED AND NOT ESTABLISHED — the intervention made it WORSE.** Rather than reason about
it, `OLF_GPU_FORCE_MODE4=<row>[,<row>…]` (new, diagnostic-only, stamps mode 4 on the named rows at
the end of `extractActivation`) forced the two rows and re-counted. They are `14403`
(`DUMMY_Q(BUS-5997_COUPLER_1)`) and `14439` (`DUMMY_Q(BUS-6112_COUPLER_1)`), which match the
`qRowAlt` in those groups' dump lines. Full busbar scope:

| | baseline | rows forced to mode 4 |
|---|---|---|
| coverage | 6550 / 6667 | **6548 / 6667** |
| `RUNTIME:status=-2` | 113 | **118** |
| `max-iter-bad-solve` | 108 | **116** |
| `DEVICE:MAX_ITERATION_REACHED` | 4 | 1 |
| `PLAN:VC_REMERGE_PINNED_MAIN` | 3 | 4 |

So gating the sharing there costs 2 contingencies and adds 5 rejects.

**And the probe was NOT a faithful emulation of OLF, which is why this is inconclusive rather than
exculpating.** For a pilot-self group whose pilot is pinned, OLF puts the PILOT'S REACTIVE BALANCE
on the single enabled controller's row (that member is the carrier). Mode 4 is an OWN-Q balance
against the member's OWN target — a THIRD, different equation. So the measurement shows mode 4 is
the wrong content for those rows; it does not show the rows are innocent. To test the hypothesis
properly the row must be made to carry the PILOT-CARRIER form, which is what `stateQRow` /
`stateQCarrier` already compute for a candidate-table pilot — i.e. the question is whether the
in-kernel mode scan picks that carrier for these 2-member pilot-self groups, and if not, why.

**RETRACTED 2026-10-02 — the 2-row difference is a FALSE POSITIVE. It compared the STRUCTURE against
OLF's RUNTIME.** Task-1 follow-up, reading the kernel rather than inferring:

- **Does the scan select the pilot-carrier for a 2-member pilot-self group? YES.** The pilot's
  `qRowAlt = -244` decodes as `-2 - g` (`ExtractionColumns`), i.e. candidate-table group 242, so the
  pilot's Q is routed by the in-kernel scan. `relocChoose` picks "the FIRST member row whose
  per-scenario mode is neither identity (0) nor pinned (4)"; `VL-5997_0`'s own V row is mode 0
  (it holds `bus_target_v`), so the scan picks the member's row — **14403**, exactly where OLF puts
  the pilot's pinned `bus_target_q`.
- **Is the sharing gated at that row? YES, and the gate already exists** — the plan-B4 branch in
  `fillDistrQBatch`: `if (g >= 0 && !psGateOff && d0 <= -2 && -2 - d0 == qrRows[qrPtr[g] +
  relocChoose(...)]) return;`, whose own comment is "the CURRENT pilot-Q carrier's row … its DISTR_Q
  row is inactive this scenario, exactly like OLF deactivating the new free member's DISTR_Q".
- **And `psGateOff` is false exactly when it must be.** `pilotSelfGateBatch` sets
  `skip = limitType == 0 ? 1 : 0` — an UNPINNED pilot skips the suppression (its q feeds those very
  sharing rows, so suppressing would leave the system one equation short: the comment records that
  measurement), a PINNED pilot does not. `VL-5997_0`/`VL-6112_0` are pinned (`enabled=false`), so the
  gate FIRES and the `dq` emission at 14403/14439 stops for the scenario.

So the runtime equation set at those rows is the pilot's balance alone, which is what OLF has. The
`336 live vs 334` count read dq-carrying rows and their BASE MODE out of the structure — which holds
the terms CANONICALLY by design — and compared that against OLF's active set. Wrong quantity. It
also explains the forced-mode-4 result above: that probe broke a correctly-gated arrangement, which
is why coverage FELL.

**Net effect on [[#67]]: there is NO identified equation-system difference.** Everything the dumps
can compare agrees — PV set, per-bus V-row occupant over 6237 buses, islanded sets, scheduled loss,
step clamp, iteration cap, and now the two rotation-carrier rows too. The remaining blind spot is
the one the next item attacks: the **649 relocated buses**, where a per-bus V-row comparison is
invalid by construction.

**THE BLIND SPOT IS CLOSED — the relocated region agrees too (2026-10-02).** The 649 relocated
buses could not be checked by a per-bus V-row comparison, so they were checked per COUPLER instead,
using the dump's `zi#K branch= b1= b2= dummyP= dummyQ= childSide=` lines to pair each coupler's rows
with its child's OLF equations. All 366 zi branches, of which 37 are `childSide=0` (unrelocated):

| check, over the 329 relocated couplers | result |
|---|---|
| child's P balance: OLF `bus_target_p[child]` ACTIVE ⟺ device `DUMMY_P` row is a power row | **0 mismatches** |
| child's Q: OLF `distr_q[child]` ⟺ device `DUMMY_Q` row mode 1 WITH dq terms; OLF `bus_target_q[child]` ⟺ mode 1 or 4; neither ⟺ the row hosts the pilot balance via the scan | **0 mismatches** |
| child's V row hosts the coupler's `ZERO_V` | 329 / 329 |
| child's PHI row hosts the coupler's `ZERO_PHI` | 329 / 329 |

Consistent with OLF having all 366 `zero_v` and all 366 `zero_phi` ACTIVE and all 366
`dummy_target_p` / `dummy_target_q` INACTIVE: the device hosts the child's balances on the dummy
rows and the coupler's zero equations on the child's own rows, which is exactly that arrangement.

**So the equation-system hypothesis for [[#67]] is EXHAUSTED on this contingency.** Everything the
two dumps can express agrees: the PV set (982), the per-bus V-row occupant over 6237 non-relocated
buses, the islanded sets (9, equal), the scheduled loss (-1.243000000 pu to nine decimals), the step
clamp (`MAX_VOLTAGE_CHANGE` -> `lsMode=2`), the iteration cap (30 = the reject's `it=30`), the two
rotation-carrier rows, and now all 329 relocated couplers on P, Q, V and PHI.

**What is still NOT compared, and is the only dimension left:** the per-row NUMERIC TARGETS and the
Jacobian VALUES. Equation IDENTITY matching does not mean the two sides solve the same numbers — a
single wrong target on one row is a different system by the standing rule, and only the AGGREGATE
loss has been checked so far, not the per-row target vector. The device exposes
`GpuAcNewtonSolver.readBatchTargets(handle)` (already used by `OLF_GPU_ROWMODE_DUMP_CTG`, which
prints targets only for rows DIFFERING from base) and OLF has its own `TargetVector`. **Next:
dump BOTH target vectors for a rejecting scenario, keyed by element, and diff per row** — the same
same-point, keyed-by-id discipline that cracked [[#66]].

### #69/#68 — BUS-1219_BBS's outer-loop cycle IS #68, and #68's premise was right (2026-10-05)

`OLF_GPU_LOOP_TRACE_CTG=BUS-1219_BBS` named the cycling loop: `kind=2` = **AcHvdcAcEmulationLimits**, flipping
every pass (`switched=1 nChanged=1`) to the 32-pass budget. A host-side trace of the loop's transitions
(`OLF_GPU_HVDC_LOOP_TRACE=<slot>`, the fork's twin `OLF_HVDC_TRACE=1`) shows SYN_HVDC_4 going LINEAR →
SAT_2_TO_1 (as OLF does: p2 = 2.75 > pmax 2.0) and then **SAT_2_TO_1 ↔ SAT_1_TO_2 forever**, i.e. the emulated flow
REVERSES once the saturation injections are applied; OLF's stays at p2 = 3.1 and settles. Alone on the list the
device converges and matches OLF to 2e-10 kV — the full list has every feeder breaker retained, so the converter
bus is a coupler CHILD and its P balance lives on the parent's `DUMMY_P` row while its own PHI row hosts
`ZERO_PHI`: the loop's ±2 pu saturation deltas landed on the `ZERO_PHI` row (#68's finding exactly) and forced
the angles apart instead of injecting power.

Why #68's fix "was catastrophic": the kernel read the bus ANGLE at the same index it wrote the target to
(`ph1 = x[hvBus1Row]`), so routing the write to the `DUMMY_P` row made it read a dummy variable as an angle. The
two roles are now separate arrays (`bus{1,2}PhiCol` read, `bus{1,2}PRow` = `GpuAcData.relocatedPRow` written),
with the dead-parent-coupler fallback to the own row on both the kernel (`mode == 0 && relocOwn`) and the host
freeze/status writers (`Ctl.bus1WriteRow(rowMode)`). The `GpuAcData.relocatedPRow` "UNUSED, kept as a warning"
javadoc said the dumps' `terms=` listing was misleading; it was not.

**Proof (`gpu-load-68b.log`, full busbar scope, GPU arm vs the saved CPU arm):** BUS-1219_BBS now converges on
the device — SYN_HVDC_4 goes LINEAR → SAT_2_TO_1 once (`HVDC_LOOP s=241 outer=6 … 0 -> 2`) and stays, worst bus
8.2e-10 kV against OLF — so **`DEVICE FAILURES on a system OLF converges: 0`**. Statuses identical
`{CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}`, `violations: CPU=503923 GPU=503923 diff=0`, coverage
6584 / 6667, DAP worst 9.6e-04 MW (was 1.398e-03: the HVDC targets on relocated converter rows were part of the
tail). The 21 plan refusals are now listed with their reason: 11 `VC_REMERGE_MULTI`, 5 `VC_REMERGE_PINNED_MAIN`,
4 `VC_REMERGE_RELOC_VROW`, 1 `ISLANDS_SLACK` (validator). Full suite 308 tests, the one failure the pre-existing
coupler-chain first-step test.

### #71 — the base main SURVIVES a middle-section loss and a second cluster forms (2026-10-05, DESIGN)

The largest of the 21 plan refusals left after #70 (`PLAN:VC_REMERGE_MULTI`, ~9 contingencies: BUS-6002_BBS_2/3/4,
BUS-6139_BBS_2/3, BUS-6218_BBS_2/3, BUS-6134_BBS_2, BUS-689_BBS_2, BUS-710_BBS_2): the lost section is the one
NEXT to the base main, so the main keeps its own section alone and the far sections form a second cluster that
OLF re-merges into a NEW group (elected main, DISTR_Q over the cluster only). Both groups live on ONE candidate
table whose pilot is the base main. What the device needs per scenario — a **detached pilot** channel, one char
per (scenario, group):
- keys: the pilot out of every denominator (the existing inactive-member entry does it);
- terms: the pilot's q contributions LEAVE the members' DISTR_Q rows (`fillDistrQBatch`: skip `dqJMember ==
  pilot` — unlike an islanded member, whose elements are disabled, the detached pilot's branches are live);
- the pilot's own row keeps its BUS_TARGET_V identity (OLF: the main's group still has its own controller), and a
  pin on it is a LOCAL pin — mode 4 on its own row with its raw q routed to candidate 0 of the group pool
  (`scatterAddDyn` / `ziDynDest`: `pool[off + 0]` when detached and `mode[pilotRow] == 4`, never the carrier);
- `pilotSelfGateBatch` skip = 1 (no balance rides a carrier), `rlPilotSelfMixedGate` ignores it,
  `updatePilotVeqBatch` never flips it (the members' pins belong to the other group now), and
  `updateDistrQTargetsBatch` never writes `pilotQ` to the carrier for it;
- the surviving cluster's elected main gets its identity from the killed-coupler block (`blockCovers`) or an
  explicit ROW_V_IDENTITY, exactly as #70 does — `reHomedControls` accepts the split when the main's own cluster
  is {main} with a single controller element and ONE other non-local cluster forms.
Host side: a GenOutage marker (`pqRowMode = ROW_PILOT_DETACHED`, `inactiveDqMember` = the pilot's slot,
`rlEntry` = the pilot's RL entry → `rlLocalOv` at its own target) → `setBatchDetachedPilots(slot, group)`.
Fixture: four sections, main at an end, lose the second — `lostPilotSectionNetwork` with the loss moved.

**Implemented differently, as SUBGROUP LABELS (2026-10-05):** rather than detaching the pilot, every surviving
cluster that is not the base main's gets a per-scenario label (1, 2, …) on its dq members — dense
`char[nS x nDqMembers]` via `setBatchDqSubgroups` + a `qr group -> first dq member` map — and the kernels pair
within a label: `fillDistrQBatch` (key sums and a contribution of another label leaves the row),
`updateDistrQTargetsBatch` (key sums), `relocChooseSub` for every carrier pick (the base pilot's balance rides
label-0 members only; `RelocTables` carries the labels), `updatePilotVeqBatch` (the "no enabled controller" scan)
and `rlPilotSelfMixedGate`. The host seeds a member's renormalized DISTR_Q target over its own label's keys and
drops its replayed weight overrides for a split group (the kernel's dynamic keys own those rows). The base main's
cluster needs no special treatment at all — with two controllers on its side its DISTR_Q row stays a label-0
sharing row, with one it is the pilot alone and the existing pilot-self logic covers it. `reHomedControls` labels
instead of refusing when every labelled member is enrolled in a candidate table whose member order the kernels'
`qrMemberStart[g] + c - 1` mapping reproduces (`qrMemberOrderConsistent`, else `PLAN:VC_REMERGE_SUBGROUP_TABLE`).
Fixtures `deviceSplitGroupBehindSurvivingMainMatchesCpu` (4 sections, every loss listed) and
`deviceSplitGroupWithTwoControllersOnMainSideMatchesCpu` (5 sections) pass at 1e-12 kV with every loss batched.

**Proof (`gpu-load-71.log`, full busbar scope, GPU arm vs the saved CPU arm):** `PLAN:VC_REMERGE_MULTI` is gone
from the census (was 11 contingencies), device coverage **6595 / 6667** (was 6584), statuses identical
`{CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}`, `violations: CPU=503923 GPU=503923 diff=0`, device failures on
OLF-converging systems 0, and the newly batched splits agree at machine precision (BUS-6002_BBS_2 5.3e-11 kV,
BUS-6139_BBS_2 2.4e-11, BUS-6218_BBS_3 3.2e-11, BUS-689_BBS_2 3.2e-11). Full suite 310 tests, the one failure
the pre-existing coupler-chain first-step test. **Left: 10 plan refusals on OLF-converging systems** — 5
`VC_REMERGE_PINNED_MAIN` (a base-pinned elected main: needs the per-scenario pilot pin treatment, i.e. the
pinned main's balance riding a carrier of ITS subgroup with its identity kept), 4 `VC_REMERGE_RELOC_VROW` (the
elected main's own V row hosts a LIVE coupler's ZERO_V), 1 `ISLANDS_SLACK` (#34).

### #72 — pinned re-homed mains and re-homed mains under a live coupler (2026-10-05, IN PROGRESS)

**PINNED_MAIN (5).** OLF's post-contingency switches (saved CPU arm, `OLF_RL_TRACE`) settle it: the pinned
bus is PQ at its limit and a cluster with NO enabled controller keeps `BUS_TARGET_V` off (BUS-6194_BBS_1 and
BUS-6003_BBS_1: both survivors pinned, never unpinned); a pinned LOCAL re-home can UNPIN (BUS-6157_BBS_1 unpins
VL-6157_1 then re-pins it at MAX_Q, BUS-6194_BBS_2 unpins VL-6194_2, BUS-6229_BBS_1 stays). So:
- local re-home of a base-pinned bus → `ROW_PINNED_V`: the row starts as the reactive balance at the pinned
  value (`LfBus.getTargetQ()`, the frozen generation minus the load) and the entry is a LOCAL PV bus at the
  re-homed voltage target (`rlLocalOv`), so the kernel's existing local unpin restores the identity;
- a remote cluster whose controllers are ALL base-pinned → no identity (the killed-coupler block keeps a pinned
  child's balance at home, mode 4) and one `ROW_PINNED_NO_UNPIN` entry per controller: an unpin would need
  `BUS_TARGET_V` on a lost pilot's table, so the kernel rejects the scenario (`unpin-unrepresentable`, CPU
  replay) instead of switching — never fired on rte6515; a pinned main beside an ENABLED sibling stays refused.
Fixtures `deviceLostPilotLocalPinnedSurvivorMatchesCpu` (2 sections, survivor pinned) and
`deviceLostPilotAllPinnedClusterMatchesCpu` (3 sections, both survivors pinned) pass at 1e-12 kV.

**RELOC_VROW (4).** The elected main's own V row hosts a LIVE coupler's ZERO_V (it is a child of an alive
coupler: BUS-6029_BBS_1/2's VL-6029_2 two couplers from the island, BUS-6014_BBS_1/2's VL-6014_3, BUS-6029_BBS_4).
Its reactive equation lives on the parent coupler's `DUMMY_Q` row, whose pattern already holds the child's own
dq/dV slot — so `BUS_TARGET_V` goes THERE as a **cross identity**: device row mode 6 = "identity on column
`relocOwn[row]`" (`applyRowModeBatch`), treated as an identity by every carrier pick and target maintainer, with
the reactive-limit kernel pinning/unpinning that relocated row (`pinIdx`, the voltage still read at the bus's
own column) and the pinned form riding the existing stock pair (dummy row on mode 4). No new array: `relocOwn`
is structural. Entries `ROW_CROSS_V_IDENTITY` (`vRow` = the dummy row, `qTarget` = the voltage target), or
`ROW_PINNED_V` on the dummy row when the bus is base-pinned. Fixtures: `deviceLostPilotReHomedUnderLiveCouplerMatchesCpu`
(3 sections, the middle uncontrolled), `deviceLostPilotRemoteClusterUnderLiveCouplerMatchesCpu` (4 sections) and
`deviceLostPilotPinnedUnderLiveCouplerMatchesCpu`.

**Two more defects the fixtures and the first proof exposed (2026-10-05):**
- `reHomedControls`' cluster BFS walked CONTROLLED buses only, so an uncontrolled section between two controlled
  ones split the group and re-homed a controller OLF keeps merged (the 4-section fixture with bbs1 uncontrolled,
  bus-3: vl12_2 "re-homed" 1.4 kV off, bus-2 a singular solve; rte6515 BUS-6029_BBS_4's VL-6029_2 behind the
  uncontrolled VL-6029_1). It walks every live bus now and collects the controlled ones.
- a LOCALLY re-homed bus beside a SURVIVING main was not labelled, so the base pilot's "enabled controller" scan
  (`updatePilotVeqBatch`) still counted it: BUS-6194_BBS_2 — VL-6194_2 re-homed and unpinned, the kernel read it as
  a live controller of the base-pinned main VL-6194_0 and restored the main's identity at 1.123 where OLF has it
  PQ at its limit (row 13634 `mode=0 target=1.123`, 4.775 kV on BUS-6194_BBS_1). A local re-home now gets its own
  label too (keys, q terms and the pilot's scan all leave it alone). 17 fixtures green.

Diagnostic full-scope run before those two (`gpu-load-73.log`): plan refusals down to **1** (`ISLANDS_SLACK`),
statuses identical, violations 503923 = 503923, device failures 0, coverage 6604 / 6667; BUS-6014_BBS_1/2 (cross
identity) at 3e-11 / 4e-11 kV; BUS-6029_BBS_1/2/4 at 5.7e-05 / 1.4e-05 / 2.0e-05 kV — to be read after the BFS fix
(BUS-6029_BBS_4 was mis-split by it).

**Proof (`gpu-load-75.log`, full busbar scope, GPU arm vs the saved CPU arm):** plan refusals on OLF-converging
systems **1** (BUS-4714_BBS, `ISLANDS_SLACK`), device failures on OLF-converging systems 0, coverage
**6604 / 6667**, statuses identical `{CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}`, `violations: CPU=503923
GPU=503923 diff=0`, `conflictRows=0`. The pinned mains agree at machine precision (BUS-6003_BBS_1 1.4e-10,
BUS-6157_BBS_1 3.6e-11, BUS-6194_BBS_2 2.6e-11 kV; the latter with OLF's unpin reproduced). One
`unpin-unrepresentable` reject fired (slot 2297, an all-pinned cluster the device wanted to unpin where OLF did
not — replayed, hence the equal statuses; the pre-unpin state differed, to be looked at). Full suite 315 tests,
the one failure the pre-existing coupler-chain first-step test. **Left in this class:** the BUS-6029 family —
BUS-6029_BBS_1 5.7e-05, BUS-6029_BBS_2 1.4e-05, BUS-6029_BBS_4 2.0e-05 kV (DAP ~4e-09 MW): the cross-identity
scenarios with an uncontrolled section between the island and the elected main; OLF switches five more buses in
BBS_1 (pins VL-6490_0, VL-5967_1, VL-6055_0, VL-6231_0/_1, unpins VL-6095_0), so the first question is whether the
device's decision sequence is the same.

**Read off (same day):** it is the same sequence, and NOT a placement defect. The device's final rows for
BUS-6029_BBS_1 (slot 2330) carry OLF's layout exactly — VL-6490_0 / VL-6055_0 / VL-6231_0 pinned (mode 1 ← 0),
VL-6095_0 unpinned (mode 0 ← 1), the cross identity on COUPLER_2's dummy row (`row=14427 mode=6 target=1.039`),
VL-6029_3's sharing row live — and the worst buses are BUS-6231_BBS_1/_2 (245.000612 vs 245.000555 kV). The
device's pinned target there is 0.57075810 where OLF pins at qLimit 0.570780403: a **reactive-capability CURVE
limit evaluated at a slightly different generator P** (2.2e-5 pu). On the 4-contingency list the same contingency
agrees to 2.1e-10 kV with the pinned target equal to OLF's to 1e-13 (1.1415608051 — there VL-6231 is ONE section,
in the full list its two sections carry one generator each), so the difference is the per-generator
distributed-slack P reconstruction the curve channel reads (`genCum`) for generators split across the sections of
a merged zero-impedance group — the same channel as the remaining DAP tail (9.6e-4 MW), not #72. Opened as **#73**.

### #73 — a pinned pilot-self PILOT re-stamped at a sub-threshold curve drift (2026-10-05, DONE)

Symptom: BUS-6029_BBS_1/2/4 at 1.4e-05 … 5.7e-05 kV, the pinned target of VL-6231_0 off OLF's by 2.2e-5 pu while the
same contingency on a 4-contingency list (VL-6231 un-split) matches to 1e-13. Suspect: `rlCurveLimitsBatch`'s p_now
for a generator whose bus is one SECTION of a merged group — the per-generator water-fill cumulative (`dGenCum`) or
its share fallback — versus OLF's targetP after the distributed-slack loop. Measure with `OLF_GPU_CV_PROBE=<entry>`
(VL-6231_0's entry) on the full list against OLF's generator targetP (`OLF_VAR_DUMP`), before touching the channel.

**Measured (full list, slot 2330).** The suspect was wrong: the P channel is exact. Device row targets for
VL-6231_0's P (13721) and VL-6231_1's relocated P (14498) = 1.1469127793758390 vs OLF's 1.1469127793758054
(3e-14); OLF's `OLF_DS_SPLIT` shows GEN-6231 and GEN-6231#0 moving identically. The pinned Q targets split:
VL-6231_1 (member, mode 4 on the coupler dummy) 0.5707804025575266 = OLF's 0.5707804025575688; VL-6231_0 (the
PILOT-SELF pilot, mode 0→1) 0.5707581042516531. `OLF_GPU_CV_PROBE=1340 OLF_GPU_CV_PROBE_SLOT=ctg` (new: the
chunk-local slot the host resolves for `OLF_GPU_LOOP_TRACE_CTG`, with a host readback `[CV-HOST]` because device
printf is not captured by surefire) gives the device's own curve limit pass by pass: 0.570780402558 at the pin
(RL outer 35–41), then the second slack round (outer 42, mismatch 0.038224 = OLF's second DS_SPLIT) moves P by
+5.25e-5 pu and the limit to 0.570758104 (outer 47+). OLF re-freezes a pinned bus at its moved limit only past
`maxReactivePowerMismatch` (`checkPqBusWithQLimitType`: |maxQ − q| > 1e-2 MVAr = 1e-4 pu), so it keeps the
pin-time limit; the device's MEMBER path (`reactiveLimitsBatch`, `tol`) does the same — which is why VL-6231_1 was
exact — but the enrolled pilot's re-stamp in `updatePilotVeqBatch` fired above 1e-12.

**Fix.** `updatePilotVeqBatch` takes `tol` (the run's `REACTIVE_LIMIT_TOL`) and gates the moved-limit re-stamp
on it. Full busbar scope: BUS-6029_BBS_1 5.68e-05 → 6.5e-11 kV, _2 1.4e-05 → 4.9e-11, _4 2.0e-05 → 3.1e-11;
statuses `{CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}` both arms, violations 503923 = 503923, device
failures on OLF-converging systems 0, coverage 6604/6667, DAP worst unchanged (9.564e-04 MW). Fixture
`devicePinnedPilotSelfKeepsItsLimitUnderThresholdMatchesCpu` (two merged sections, curve on the main's generator,
a listed section loss retains the coupler, losing a P-less reactive-support generator pins both sections and a
second slack round drifts the curve by ~2e-8 pu): fails on the old gate (1.75e-8 kV at vl3), passes with the fix;
the batched SA class 189/189. The split-section wording of the title was a misdiagnosis — nothing about the split
mattered beyond VL-6231_0 being a pilot-self pilot.

### #74 — the DAP tail's head: a relocated child's curve read its own PHI row (2026-10-05, DONE)

The worst remaining `distributedActivePower` difference, BUS-4467_BBS at 9.564e-04 MW, carried the same bus
signature as #73 (BUS-6231_BBS_1/2 worst, then 6171/5976/6225/6005) with a 1.85 MW second slack round. Measured
against OLF pass by pass (`OLF_OL_TRACE` + `OLF_DS_TRACE` on the CPU arm, `OLF_GPU_LOOP_TRACE_CTG` on the device):
the SEQUENCE is identical — first slack check 0.002820383 on both arms (stable), the same six switches, then a
slack round — and the only number apart is that second check, OLF 0.018454123 vs device 0.018463687: 9.6e-06 pu,
the whole 9.56e-04 MW. The paired converged dump (`OLF_VAR_DUMP` / `OLF_GPU_SA_VAR_DUMP`) puts the worst variable
at `DUMMY_Q BUS-6231_COUPLER_1` (3.1e-06 pu) — the reactive split of the two VL-6231 sections.

`[CV-HOST]` on both entries (1340 = VL-6231_0, the pilot-self pilot; 1341 = VL-6231_1, its coupler-relocated
child) at the pin pass: 1340 maxQ 0.566570618018 = OLF's qLimit to 1e-12; **1341 maxQ 0.576013011409** — the same
unit halves (GEN-6231 / GEN-6231#0, same P, same curve) evaluated at DIFFERENT P. After the slack round (outer 47)
both read 0.566558432, so the per-generator water-fill path is right and only the pre-slack SHARE FALLBACK is not:
it reconstructs p_now from `target[pRow] - baseRowTarget`, with pRow the bus's OWN `BUS_PHI` row — which for a
relocated child hosts `ZERO_PHI` (target 0, never moved by the pre-distribution), while the bus's active balance,
and the 0.0223 pu the security analysis pre-distributed onto it, live on its parent coupler's `DUMMY_P` row. So
the child pinned at the BASE-P limit; the 9.4e-3 pu that put on the coupler changed the losses the next slack
check read. OLF then keeps the pin-time limit (drift under the threshold, #73) while the device re-froze the child
at its moved limit, leaving the 1.2e-5 pu it ended with. The curve dump (`OLF_GPU_CURVE_DUMP`) confirms both halves
have DS slots (666/667) and identical packs; genCumValid is simply 0 until the first slack round runs.

**Fix.** The host packs the slack pack's row for the bus (`GpuAcData.dsPhiRow()` via `dsBusNum`) and the kernel
resolves it PER SCENARIO through `dsRowOf(dsPhiRow, dsRowOv, s, nds, dgBus[slot])` — the row the slack pack itself
retargets, moved back to the child's own row when the coupler dies. Full busbar scope, GPU arm vs the saved CPU arm:
BUS-4467_BBS 2.870e-05 → **4.087e-11 kV**; DAP worst **9.564e-04 → 1.353e-04 MW** (BUS-6115_BBS_1, 9.6e-05, left
the top five too); statuses `{CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}` both arms; violations
503923 = 503923; coverage 6604/6667; device failures on OLF-converging systems 0. Fixture
`deviceRelocatedChildCurvePinnedAtPreDistributedPMatchesCpu`: a curve on the relocated child's generator, a listed
section loss to retain the coupler, a 100 MW support generator at vl3 whose loss the SA pre-distributes (slack
non-participating — a lost participating generator with a participating slack is `PARTICIPATING_SLACK`, a CPU
shape), both sections pinned at the pre-distributed P (qLimit 0.606601878 against a base-P 0.604334), DEFAULT slack
threshold so no slack round precedes the pin. The fixture fails on the old code (2.8e-3 kV at vl3) and passes with
the fix; it exercises the HOST half (the packed row) — the loss of the pilot section (bbs0, listed too) kills the
coupler but pins nothing, so the kernel's per-scenario `dsRowOv` resolution is reasoned, not fixture-proven:
a surviving child whose curve must be read on its own PHI row after the coupler dies still wants a fixture that
pins it there.

**Remaining tail after this** (`OLF_GPU_SA_DAP_DUMP`): BUS-716_BBS 1.353e-04, BUS-2414_BBS_1 1.24e-05,
BUS-4603_BBS 1.07e-05, BUS-1599_BBS_2 5.3e-06, BUS-349_BBS 4.5e-06 MW. Method that worked here, to repeat on
BUS-716_BBS: busdiff → the worst buses name the controller; `[CV-HOST]`/`SW_PASS` on its entries against OLF's
`RL_SWITCH` qLimit; the per-pass slack checks against `DS_PASS`.

**Side observation, not pursued:** with the support generator AT its maxP (no headroom), the same fixture's
bbs1 loss differs by 2.5e-09 kV at vl4 with no switch and no slack round on either arm, and the two arms' slack
mismatch readings differ (OLF 6.13e-04 vs device 1.45e-04 pu) — the retain share `rho` counts a saturated
participant that OLF's water-fill drops. Harmless here (neither arm distributes) but it is a candidate for the
sub-1e-4 MW tail.

### #75 — the DAP tail below 1.4e-4 MW is KLU's forward error read by OLF's stopping criterion (2026-10-05, MEASURED)

BUS-716_BBS, the head of the tail after #74 (1.353e-04 MW), differs at the FIRST slack check: OLF `DS_PASS
mismatch=0.057195631`, device `dsRem=0.057196984834` — 1.354e-06 pu = the gap — with no switch and no
pre-distribution before it (`previousMismatch=0`). Both arms' first post-contingency Newton solve, traced
(`OLF_NR_TRACE=1 OLF_NR_ITER_TRACE=1` on the CPU arm, `OLF_GPU_CONV_TRACE=2994` on the device, slot from
`LOOP_TRACE_CTG`):

| | OLF (KLU) | device, IR=0 | device, IR=4 (default) | device, IR=8 |
|---|---|---|---|---|
| ‖F‖₂ at entry | 35.97019031573278 | 35.97019031573512 | 35.97019031573509 | 35.97019031573524 |
| after step 1 | **5.05632848098e-02** | 5.05848962592e-02 | **5.05632838301e-02** | 5.05632838302e-02 |
| after step 2 (both stop) | 2.12192626685e-04 | 2.53843895e-04 | 2.12131551e-04 | 2.12131551e-04 |

The two arms enter at the same residual to 14 digits and both stop after two steps (OLF's ‖F‖₂ < 1.204e-02),
so the equations, the targets and the stopping decision are the same; the FIRST STEP is not. The device's step is
converged under refinement — IR=4 and IR=8 agree to 1e-13 and two independent runs to 3e-14 — and its backward
residual ratio is 1.6e-15 (`OLF_GPU_LIN_RESID=1`, slot 2994). OLF's step sits 9.8e-10 above it (2e-08 relative),
which on a Jacobian of condition ~5e8 (the `irSteps` note) is KLU's unrefined forward error (κ·ε ≈ 5e-08). The
loose criterion then freezes that deviation into the state the slack check reads: 1.35e-06 pu at the slack row,
1.19e-07 kV at the worst bus. With IR=0 the device's own forward error is 4e-05 relative and the scope's DAP worst
is 1.07 MW (the run fails), so the refinement is load-bearing and the default stays 4.

**What this means.** The residual tail (BUS-716_BBS 1.35e-04, BUS-2414_BBS_1 1.2e-05, BUS-4603_BBS 1.1e-05 MW
and the ~3600 contingencies above 1e-9 MW) is the CPU arm's linear-solve error at a two-step stopping point, not
a device defect: the device computes the exact Newton step, OLF does not, and nothing on the device can make its
exact step equal KLU's inexact one. The two ways to close it are both on the CPU side or in the comparison:
refine KLU's solve in OLF (an upstream change — one IR step would take its forward error to ~1e-11), or compare
at a criterion tight enough that both arms iterate the deviation away (the `#66` eps-squeeze already showed the
BUS-4467 class floor; this class does collapse, by construction). Not re-measured here on purpose: the only
confirmation left is KLU's own forward error on this matrix (dump slot 2994's it=0 system, KLU solve vs one
refined solve), which would put a number on OLF's side instead of inferring it from κ·ε.

**The 9042-branch reference scope, same code (2026-10-06, GPU arm cold vs a 1-thread CPU arm saved the same
morning):** 9040/9042 on device (`DEVICE:MAX_ITERATION_REACHED=2`, both OLF's own), statuses identical
`{CONVERGED=9037, FAILED=3, MAX_ITERATION_REACHED=2}`, violations 583458 = 583458, device failures on
OLF-converging systems 0, plan refusals 0, DAP worst 7.33e-07 MW (LINE-3760-3113) with 745 of 9037 above
1e-9 MW — the test PASSES, GPU arm 25 s. The generator kind (1389 contingencies at every voltage) is being
measured separately for GEN-1522, the 2026-09-22 worst there (~1e-6 MW; note the default `branch` kinds hold no
generator contingency, and a CPU-arm save writes `cpu-busv.bin` only with `OLF_GPU_SA_BUSDIFF_CTG` set).

Also corrected today: CLAUDE.md said "no state-vector scaling / line search on device"; the batch applies OLF's
MAX_VOLTAGE_CHANGE clamp per scenario and OLF's exact stopping test, and refines cuDSS solves 4× on networks with
zero-impedance branches.

### network.xiidm, full non-busbar scope: wall time vs a 20-thread CPU (2026-10-06, injection-scope)

`OLF_GPU_SA_CASE_FILE=network.xiidm OLF_GPU_SA_CTG_KINDS=branch,generator,load,hvdc,svc,shunt,battery
OLF_GPU_SA_GEN_MIN_KV=0 OLF_GPU_SA_LOAD_TOP=100000` = 24379 contingencies (10540 branches, 6071 generators, 7495
loads, 6 HVDC, 6 SVC, 240 shunts, 21 batteries), RTE parameters, this branch at cf2fd96 (base-outer7-divergence
merged through #76). Timing only — a multi-threaded CPU arm is not comparable on results (its partitioning retains
other switches), so the exact comparison is run separately against a 1-thread arm:

| arm | wall | per contingency |
|---|---|---|
| CPU, 20 threads (`OLF_GPU_SA_CPU_THREADS=20`) | 417.7 s | 17.13 ms |
| GPU, cold (`OLF_GPU_SA_COLD_ONLY=1`) | 198.3 s | 8.13 ms |

**GPU 2.11x the 20-thread CPU**, 24374 / 24379 on device (3 OLF-own MAX_ITERATION_REACHED, 2 device
`max-iter-bad-solve` rejects — FUCZ and FUDC, to be read against the 1-thread arm). Method notes: the CPU-arm
`save` mode runs the CPU arm ONLY (the GPU wall time comes from a `load` run); the 1-thread arm on this scope
needs more than the 3 h first budgeted (exit 124 at 10800 s), re-run with 6 h.

### #84 — why ReactiveLimits costs so much: three buses oscillating to the switch cap (2026-10-07)

Read from the existing 20-thread CPU trace (`OLF_OL_TRACE=1 OLF_RL_TRACE=1`, contingency-labelled; no new run):
170,898 unstable ReactiveLimits checks (= re-solves) over 24,296 contingencies. Causes: PV->PQ only 67.8 %, PQ->PV
only 26.4 %, both 5.7 %; a limit update alone **12** — the re-solves are real switches, not curve refreshes.
452,282 switches for 130,465 distinct (contingency, bus) pairs; **52,583 pairs switch 6+ times**, and three buses —
**IZTH_0, IZTH_2, IZSV_0** — switch in 19,588 contingencies. The cycle (the base load flow shows it too): IZTH_0 and
IZTH_2 pin MIN_Q, IZSV_0 pins MIN_Q, all three switch back PQ->PV in the same check, repeat until maxPqPvSwitch (3)
stops them. **152,664 of the 170,898 ReactiveLimits re-solves (89 %) are passes where ONLY these three switched** —
~65 % of all the CPU's Newton iterations, and the reason 17,480 contingencies take 10–12 solve passes; the device
reproduces the sequence exactly, so its 104 RL passes are mostly this too. OLF is the reference, so the device cannot
skip them: the lever is the data (a shared voltage control whose controllers all pin and then all release on the
controlled bus's voltage?) or OLF's switch-back rule — to be confirmed by a trace of the three buses' voltage control,
targets and voltages per check.

**Confirmed by a trace** (`OLF_RL_BUS_TRACE=IZTH_0,IZTH_2,IZSV_0`, fork; base load flow): the three are the merged
controllers of ONE remote voltage control on **IZUM_0** (target 1.051298218 pu). All PV, they absorb equal shares,
−0.2476 pu each, below IZTH_0's minimum (−0.2322) and IZTH_2's (−0.2226): both pin MIN_Q. IZSV_0 alone then needs
−0.28929 against its −0.28900 minimum: it pins. With all three pinned, IZUM_0 sits at 1.051298113 — **1.05e-7 pu under
its target** — and OLF's switch-back (`checkPqBusWithQLimitType`: MIN_Q pinned and controlled V < target, a strict
comparison with no tolerance) releases all three: back to the first state. Repeat until maxPqPvSwitch. The final state
is "all pinned", 1e-7 pu short; a tolerance on that comparison would reach it in two passes instead of ~8 per
contingency — an upstream OLF question (OLF stays the reference here), and most of the ReactiveLimits cost on both arms.

**Reproduced with the OFFICIAL pypowsybl 1.16.1 (OLF 2.3.0), no fork code** (`native/pypowsybl/pp_rl_oscillation.py`,
RTE parameters, 1 thread): OLF's own report shows it in the base load flow (two cycles from iteration 1; IZUM_0 at
236.542075 kV against 236.542099) and in a one-contingency SA (LIRN: PQ->PV of all three at iterations 2, 5, 8 —
"q=minQ and v=236.542081kV < targetV=236.542099kV" —, PV->PQ of IZTH_0/IZTH_2 then IZSV_0 in between, and at
iteration 11 "3 buses blocked PQ due to the max number of switches (3)"). A stock-OLF behaviour, ready for an
upstream issue (network + RTE parameters + this one contingency).

**Device side** (`OLF_GPU_RL_REASONS=1`): causes per checked scenario match the CPU's (pin 114,611, unpin 45,150,
pin+unpin 9,718, limit refresh alone 12); 82 of the 104 RL re-solve passes have ≥ 50 % of the chunk active (~2,000
scenarios, 527 ms/pass, 43.2 s of the 46.1 s): the oscillation runs in lockstep across most contingencies, so these
passes are full, not padding — the device's RL cost is the number of oscillation passes.

Instrumentation (now built and run): fork
`ReactiveLimitsOuterLoop` prints `CPU_RL_CHECK` per check under `OLF_OL_STATS` (candidates per cause, which switched);
device `OLF_GPU_RL_REASONS=1` tallies each checked scenario's causes from the per-entry switch codes and prints
`GPU_RL_PASS` per RL re-solve pass (active scenarios, ms, batched iterations); `native/cuda/rl_reasons.py cpu|gpu`.

### #83 — what the outer loops cost on the device (2026-10-07)

`OLF_GPU_PHASE_TIME=1` now splits the driver's time per outer pass: the FIRST solve of a call, each RE-SOLVE charged to
the loop whose check triggered it, and each CHECK charged to the loop checked (`GPU_OL_TIME` lines; stream synced at
the pass boundaries, only under the switch). `native/cuda/ol_summary.py cpu|gpu <log>` prints the run's outer-loop
summary (reportedContingencies, contingenciesTriggeringOuterLoops, totalOuterLoopIterations, by type, histogram,
Newton split) from the per-contingency records. network.xiidm, GPU-only cold, nobody else on the card:

| part | passes | time | scenario-solves | per pass |
|---|---|---|---|---|
| first inner solve | 11 | 8.6 s | 24,387 | 780 ms |
| re-solves after a ReactiveLimits check | 104 | 44.1 s | 170,787 | 424 ms |
| re-solves after a DistributedSlack check | 27 | 10.4 s | 12,252 | 386 ms |
| re-solves after the FreezingHvdcACEmulation check | 10 | 4.2 s | 24,378 | 418 ms |
| checks (RL 2.0, DS 0.6, HVDC freeze 0.6, AC-emulation limits 0.5) | | 3.7 s | | |

Of the 75.6 s device solve, **the outer loops are ~62 s**: the same share as their Newton iterations on the CPU
(89 %). The summaries of both arms are identical bin by bin (GPU per-type counts are batch-level, so they include
the handful of contingencies not converged on the device). **A re-solve pass costs ~400 ms whether it solves 450
scenarios (DS) or 2,400 (HVDC freeze)**: the cost follows the number of passes (152 solving passes over 9 chunks), not
the number of contingencies that re-solve — a pass runs as soon as ONE scenario of its chunk needs it, and pays a
largely fixed amount. That per-pass fixed cost is the next thing to break down (full-width kernels, batched
iterations per pass, host syncs).

### #82 — network.xiidm profile, round 3: the host side of the per-contingency loop (2026-10-07)

GPU-only cold run (`OLF_GPU_SA_GPU_ONLY=1`), 24,379 contingencies, code at 60c3cc7 (refinement step back to 1 on
coupler systems): **123.0 s**. Split (`OLF_GPU_PHASE_TIME`, `OLF_SA_PROFILE`, JFR at 10 ms — JFR samples Java only,
so samples x 10 ms is host CPU; `scratchpad/jfr_lines.py`-style counting, not the time-to-next-sample weighting,
which inflates frames that precede a JNI call):

| part | time | what |
|---|---|---|
| precompute: device solve (native) | ~75 s | Nsight: GPU busy almost all of it — 69.4 s kernels + 2.2 s copies; cuDSS 52.5 s (fwd 17.9, factorize 15.9, bwd 11.5, spmv of the internal refinement step 7.2), closed-branch fill 5.4 |
| precompute: marshal + pack | 4.6 s | inputs to the native side, states back |
| precompute: host planning | ~4 s | qualification, override tables, the zero-impedance tree replay 1.8 s, the chunk solver's own loops |
| per-contingency loop (OLF's SA loop) | ~20 s host CPU | state injection 13 s (8.3 s of it OLF's equation-system index rebuild: variables 6, equations 2.3), base-state restore 3.6, bus update 2.1, CPU replay of 3 contingencies 1.3, contingency apply 0.6 |
| base load flow + setup | 2 s | |

Changes (all exact, see gates):
1. **Chunk solver**: the sticky non-tree loop scanned all 2653 zero-impedance elements per scenario against a
   boxed set; it now walks the base out-of-tree elements, collected once, with a primitive still-out test.
2. **Fork, `EquationSystemIndex.getSortedVariablesToFindOnly()`**: the variable side alone. Used by the injector
   and by `AcNetworkVector.updateVariables` (which only reads variable rows and was forcing the equations-to-solve
   rebuild on every state update). The equation side stays lazy for the next solver.
3. **Fork, incremental variables index**: `EquationSystemIndex` records the variables whose membership changed
   since the last update and patches the previous sorted list by binary search (≤ 256 changes, no separation),
   re-assigning rows from the first touched position — same list, same order, same rows as the full rebuild
   (`OLF_VARIDX_CHECK=1` recomputes and asserts after every update: no mismatch over the network.xiidm run and
   OLF's own 1508 tests). The TreeMap walk disappeared from the profile; per-contingency host CPU 20.3 → 15.5 s.

Result: **GPU-only cold 123.0 → 115.7 s.** Gates: network.xiidm against the 1-thread CPU 24,376/24,376 identical
per contingency (passes, iterations), violations 6190 = 6190, DAP 5.56e-05 MW; rte6515 generator 1389/1389 and
branch 9040/9042 exact (85,876 and 583,458 violations, diff 0); GPU suites 227/227; OLF suite 1508/1508.

**Measured and rejected — straggler compaction policy** (`OLF_GPU_COMPACT_FRAC`, `OLF_GPU_COMPACT_TIER`, inert at
their defaults 0.5 / powers of two): compacting at 75 % active 118.2 s (+2.5); width tiers of 256/128 instead of
powers of two 137.8–140.6 s (+22: every new width is a new analysed data object). The padded lanes (278,144
lane-iterations for 234,088 useful) are cheaper to factorize than to gather out.

**Round 3b (same day)** — what was left, taken down:
- **Which variables churn** (`OLF_GPU_INJECT_DIAG=1`): no contingency ever ADDS a variable; 11,939 remove buses' V
  and PHI (8,533 one bus, the rest 2–32), 4,061 come back to the base set (removed by the previous contingency,
  re-added by the restore). The current sorted list is therefore always a SUBSEQUENCE of the base one, and the
  injector's remap is now a two-pointer merge on REFERENCES (no field read of the 32k variables), with the keyed
  walk as fallback; the diagnostic checks merge == keyed (16,000 / 16,000).
- **Fork, incremental index part 2**: the touched record is a bounded list (an IdentityHashMap kept the 32k
  capacity of the first full build, and every clear() cost it), and the index publishes what its last update
  changed (generation, first changed row, removed variables). `AcNetworkVector.updateVariables`, when exactly one
  incremental update behind, resets the removed rows and re-walks only the suffix (checked against the full walk
  under `OLF_VARIDX_CHECK=1`).
- **Lazy voltage restore** (`olf.gpu.sa.lazyVoltageRestore`, default true; fork `NetworkState.restore(buses,
  branches, voltages)` and `restoreVoltages(...)`): after a device-answered contingency the targeted restore no
  longer walks every bus to put its voltage back — the next device-answered contingency's injection writes every
  live bus, and only its DISABLED buses get their base voltage back; a CPU replay, an exhausted contingency or an
  action simulation restores all of them first, and so does the end of the loop. Off under the restore check
  (whose reference is a full restore). Restore 3.6 → 1.0 s.

Per-contingency host CPU 15.5 → ~10 s; **GPU-only cold 115.7 → ~109–113 s** (run-to-run spread ±2 s). Gates:
network.xiidm 24,376 / 24,376 identical per contingency, violations 6190 = 6190; rte6515 generator and branch exact;
GPU suites 227/227; OLF suite 1508/1508 with both index checks on. The generator kind's worst DAP moved between
0.9e-6 and 4.3e-6 MW across runs with lazy restore on AND off — DAP comes from the precompute, before the loop, so
this is the device's run-to-run spread, not the restore.

**What is left on the host** (15.5 s CPU in the loop): the injector's remap walk over all 32k variables on every
contingency whose variable set changed (3.6 s) and `AcNetworkVector.updateVariables` doing the same inside the
state update (3.2 s); the base-state restore's voltage loop over every bus (2.6 s — redundant for a device-answered
contingency whose injection overwrites every voltage, but the CPU replay starts from those voltages); the bus
update (1.7 s). The device solve is cuDSS-bound and its count is OLF's.

### #81 — the fork's CPU arm against the OFFICIAL pypowsybl (2026-10-07)

Same scope as the harness (all kinds but busbar), same parameters: the harness's list exported with the new
`OLF_GPU_SA_CTG_DUMP=<file.json>` (powsybl contingency-list JSON, then stops), run by `native/pypowsybl/pp_sa.py`
in a venv with pypowsybl 1.16.1 (OLF 2.3.0, core 7.3.0): RTE JSON load-flow parameters, the harness's increased
violations thresholds (0.5 kV / 0.001 both voltage sides, 0.001 flow), `threadCount`, run_ac timed alone.
`native/pypowsybl/pp_compare.py` matches per contingency (statuses, violation-key multisets, values as SORTED
lists per key: a voltage level with several buses carries several violations under one key — pairing them
blindly produced fake 1.7 kV "differences", and a fake run-to-run spread between two fork arms).

| scope | pypowsybl | fork (harness CPU arm) |
|---|---|---|
| rte6515, 14,260 contingencies, 20 threads | 58.7 s | 66.7 s |
| network.xiidm, 24,379 contingencies, 20 threads | 470.1 s | 417.7 s |
| rte6515, 14,260, 1 thread | 1063.5 s | (generator and branch arms only) |

Results: **statuses identical, violation sets identical contingency by contingency** on all four comparisons
(rte6515 20 threads 906,447 = 906,447; network.xiidm 20 threads 4,579 = 4,579; rte6515 1 thread vs the fork's
1-thread generator arm 85,876 = 85,876 and branch arm 583,458 = 583,458). Values: ~0.2 % differ by 1e-9 to
1.4e-7 (line currents in A, ~1e-10 relative), consistent with the version gap (the fork's upstream base is 3
commits past 2.3.0, including "Fix reproducibility issue introduced by EquationTerm sorting"; core 7.4.0-SNAPSHOT
vs 7.3.0). Two 20-thread fork arms of network.xiidm agree with each other to 1.9e-10. The wall times are single
runs in two runtimes (pypowsybl's native image vs the surefire JVM), ±12 % either way; the fork's OLF changes do
not move the CPU's results.

### #80 — the one contingency whose Newton sequence differed: FBQV, a dead bus OLF still checks (2026-10-06, DONE)

Per-contingency match of the device against the 1-thread CPU arm (`OLF_OL_STATS=1` on both: fork `CPU_OL_STATS`,
device `GPU_OL_STATS` from the new `readBatchOlStats`, matched by `native/cuda/ol_match.py`): **24,375 of 24,376
converged contingencies identical in (solve passes, first-solve iterations, total iterations); one differed,
FBQV** (CPU 5/1/5, device 6/1/6). FUCZ and FUDC (the two device failures) match the CPU's 6/1/6 exactly: their
rejection is a post-solve check, not the sequence.

**Trace** (fork `OL_CHECK` now labelled by contingency, scoped through `OlfTraceScope`; device
`OLF_GPU_LOOP_TRACE_CTG` now also aims the switch trace, `OLF_GPU_SW_TRACE_ENTRIES=<e,…>|all` prints chosen entries
every reactive-limit pass with slope / switch count / re-home / monitored / islanded / active read back): FBQV is
the outage of FBQF_0's only generator. FBQF_0 was pinned MIN_Q in the base. OLF's `processLostGenerators` disables
its voltage control but keeps its qLimitType and its place among the controller elements, so the
reactive-limits loop runs `checkPqBus`: V 1.009793 < target 1.010221 → back to PV at the first pass, re-pinned
MAX_Q at its now-empty [0, 0] range at the second. The device had put the entry on the ISLANDED channel
(`GenOutage.silencesReactiveLimitEntry` → `rlIslandedList`), so it was never checked; FBQF_0 stayed pinned, and
CQEM_0 (excess 1.0226e-4 pu on the CPU against the 1e-4 threshold, 9.64e-5 on the device) pinned one pass late.

**Fix**: a LOCAL entry whose bus lost every generator but was PINNED in the base (base limit type 1..4) stays
live — its limits already collapse with the outage. Also
found on the way and fixed: the degenerate-range guard of the local reactive-limit path (min ≥ max → skip) now
skips only entries that are NOT controller elements of their voltage control (`rlMonitored` bit 2, set at
pack time; the two native readers of bit 1 are masked) — OLF walks controller elements, and a generator bus
whose range collapses is still one.

**Verified**: network.xiidm per-contingency match **24,376 / 24,376 identical**, statuses identical, violations
6190 = 6190, DAP worst 5.56e-05 MW; rte6515 generator kind exact (1389/1389, 85,876 diff 0, DAP 1.03e-06 — it
holds GEN-6130, the measurement behind the old silencing rule, so that rule's case is covered by the collapsed
limits); branch kind exact (9040/9042, 583,458 diff 0); suites 226/226. Fixture
`deviceBasePinnedBusLosingItsOnlyGeneratorIsStillCheckedMatchesCpu` (4-bus 400 kV mesh, g1 pinned MIN_Q at 0 in
the base, lost): OLF switches vlb back to PV then pins it MAX_Q at 0; it passes at 1e-12 kV and FAILED with the
old silencing (vlb 5.7e-05 kV off), checked before the old behaviour was removed.

### #79 — where the Newton work goes: outer loops, refinement, analyses, CPU replay (2026-10-06, injection-scope)

Asked: how many symbolic analyses run; whether fewer triangular solves is a lever at all when the GPU must do
the CPU's Newton iterations; the CPU replay's cost; how many contingencies really need outer loops versus being
right after the first inner Newton. Network.xiidm, full non-busbar scope (24,379 contingencies).

**Instruments (new).** Native, under `OLF_GPU_PHASE_TIME`: `CUDSS_PHASE_TIME` / `CUDSS_PHASE_SITE` time every
cuDSS analysis (A), factorization (F, the compact path's factorization+solve), full-width factorization+solve
(G) and solve-only (S) call with a device sync on both sides (so the run's wall time is not a reference);
`GPU_OL_HIST` = per device-converged scenario, the solve passes it took part in, its first pass's Newton
iterations and its total. Fork (`olf-local-patches`), under `OLF_OL_STATS`: `AcloadFlowEngine` keeps the first
inner solve's iterations and, per outer loop, the re-solves it triggered and their iterations;
`AbstractSecurityAnalysis` prints `CPU_OL_STATS ctg=… firstNr= totalNr= outer= status= loops=Name:resolves/iters`
per contingency; `native/cuda/ol_stats.py <log>` turns it into the histogram. Java: the GPU provider's CPU
branch is timed (`CPU replay: …` in SA_PHASE_TIME).

**1. Symbolic analyses are not a cost**: 16 in the whole run (1 base, 2 batch main, 12 compaction widths,
1 robust), **1.2 s** together.

**2. The GPU does exactly the CPU's Newton work** (CPU arm at 20 threads; the 1-thread run tonight repeats it):

| | CPU | GPU |
|---|---|---|
| Newton iterations, total | 234,153 | 233,909 |
| of which in the first inner solve | 26,697 | 26,352 |
| of which outer-loop re-solves | **207,456 (88.6 %)** | 207,557 (88.7 %) |

The solve-pass histograms match bin by bin (passes 4: 2653 / 2649; 5: 2606 / 2607; 11: 11,403 / 11,348;
12: 6077 / 6133). **Almost no contingency is right after the first inner Newton**: the minimum is 2 passes
(6 contingencies), and **17,480 contingencies take 11 or 12 passes** of about one Newton iteration each. Per
outer loop on the CPU:

| outer loop | contingencies it re-solved | re-solves | Newton iterations |
|---|---|---|---|
| ReactiveLimits | 24,296 | 170,757 | 170,891 (73 % of all) |
| FreezingHvdcACEmulation | 24,376 (every one) | 24,376 | 24,395 |
| DistributedSlack | 11,796 | 12,170 | 12,170 |

So the Newton count is OLF's and the device cannot go below it. OLF's Java is the REFERENCE: its outer-loop
sequence (7 reactive-limit re-solves of one iteration per contingency on average) and the HVDC freeze are what
the device reproduces, not levers. The freeze fires exactly once per contingency by design: the SA's
`startWithFrozenACEmulation` defaults to TRUE (`OpenSecurityAnalysisParameters`, while the LF default is false),
`AcSecurityAnalysis` turns it on for every contingency load flow, `FreezingHvdcACEmulationOuterloop.initialize`
freezes each connected AC-emulation HVDC, and its first check unfreezes them and returns UNSTABLE — one forced
re-solve, about 10 % of the Newton iterations on this network.

**3. Refinement is solve work that does NOT come from the Newton count**:

| cuDSS call | count | time (synced) |
|---|---|---|
| full-width factorization + solve | 106 | 21.3 s |
| compacted factorization + solve | 150 | 6.8 s |
| **refinement solves (solve only)** | **543** | **22.4 s** |
| analyses | 16 | 1.2 s |

543 refinement solves for 256 Newton linear solves, 2.1 per factorization, 43 % of the cuDSS time. That is the
lever "fewer triangular solves" really refers to — but NOT by removing refinement (measured: off → DAP 1.07 MW;
fixed 2 → two rte6515 busbar scenarios at a linear-solve failure). The adaptive check already stops when a lane
is accurate; the lever is a first solve accurate enough that it stops at zero, i.e. the conditioning of the
repaired zero-diagonal rows of the zero-impedance (coupler) branches (scaling them, the open note above the
adaptive policy).

**The 1-thread exact run (2026-10-06 evening, rebased on 478d38f's zero-diagonal repair):** CPU 1 thread
11,337 s (465 ms/contingency), GPU cold 110.7 s (x102, with the phase-time syncs on); statuses identical
`{CONVERGED=24376, MAX_ITERATION_REACHED=3}`, **violations 6190 = 6190**, distributed active power worst
**5.56e-05 MW** (11,458 contingencies above 1e-9 MW), 24,374 / 24,379 on device, device failures FUCZ, FUDC.
Outer-loop histograms, 1-thread CPU against GPU: Newton total 233,865 vs 233,909, first inner solve 26,345 vs
26,352; bins 3, 4, 7, 8, 11, 12, 13 identical to the contingency AND the iteration (11,348 / 125,289 both
sides); bins 5, 6, 9, 10 differ by 1–3 contingencies (2608/2607, 1047/1050, 4/5, 7/9) — a handful whose
outer-loop sequence differs from OLF's while the results still agree; the GPU histogram has no ids yet, so
naming them needs a per-scenario GPU record. Refinement solves fell 543 → **353** (cuDSS solve-only 22.4 → 15.4 s) between the
pre-rebase run and this one; the only device change in between is 478d38f's zero-diagonal repair on
re-coupled loop couplers, the conditioning lever — likely the cause, not yet isolated by an A/B.

**4. CPU replay is small on this scope**: 3 contingencies (FUCZ, FUDC — the two device failures — and DQCQ,
MAX_ITERATION_REACHED after 31 iterations), **1.5 s** of the run. The robust device retries (5) are inside
the precompute. On scopes with more plan refusals this number is what to watch.

### #78 — the in-suite flake was a device read-before-write (2026-10-06, injection-scope)

`devicePilotLosingItsOwnGeneratorWithPinningSurvivorMatchesCpu` (#76) failed in 3 of 7 runs of its class and
never alone, two shapes: vlp_b 2.2e-6 kV off (two DIFFERENT wrong values across runs) or the device failing
outright on a 4-bus batch of one (`robust-attempted=1 recovered=0`, "the batch must ENGAGE"); once a second
fixture went with it (`deviceTwoMemberPilotControllerLosingAControllerMatchesCpu`, b2 0.8 kV off). Independent
of the refinement default (1/3 with D-C, 2/3 without).

**Method** (new switches in `DeviceBuf::ensure`): `OLF_GPU_ZERO_ALLOC` zeroes every fresh `cudaMalloc`;
`OLF_GPU_POISON_ALLOC=all|<k>|<a>-<b>` fills the k-th allocation(s) of the process with `OLF_GPU_POISON_BYTE`
(255: NaN doubles, −1 ints); `OLF_GPU_ALLOC_TRACE=<k>` prints every allocation's index and size and a host
backtrace at k (resolve `+0x…` frames with `addr2line -f -C -e libolfgpu.so`, the symbol table is kept).
Zeroing: 5 of 5 class runs clean against 3 of 5 without — a read-before-write, not rounding. Poisoning every
allocation made the single fixture fail deterministically (235 allocations); bisecting the range took 12 runs
and landed on allocation #131, 1 byte, inside `createBatchContext`: **`dRlActive`**, the per-scenario
"RL-check this pass" flag. The RL pass is what first writes it (a host memcpy), but the SEED launch of
`updatePilotVeqBatch` / `updateDistrQTargetsBatch` before the first solve reads it — the kernel's own comment
says "at the post-seed launch too, when rlActive is still 0". It is 0 in a fresh process (first-use device
pages) and whatever the previous owner left after other solves in the same one, which is exactly in-suite-only.

**Fix**: fresh allocations are zeroed by default (`OLF_GPU_ZERO_ALLOC=0` restores the old behaviour for A/B) —
the codebase already assumed it, so it is now guaranteed, one memset per cudaMalloc at context creation — and
`dRlActive`/`dRlSwitched` are cleared explicitly at creation with the reason. The poison and trace switches
stay for the next one.

**Verified**: with the flag fixed, poisoning the remaining range 118–234 fails nothing (the status −1 of the
118–234 probe was the poisoned flag's "active" reading other not-yet-written RL buffers, gone with it); the
fixture alone AND the whole class (194 tests) pass with EVERY allocation poisoned — the loud regression check;
the fixture suites 226/226; the network scope cold 121.3 s with zeroing on (116.2 s before, solve 70.8 vs
69.4 s, within the run-to-run spread), identical statuses and the same FUCZ/FUDC device failures.

### #77 — network.xiidm cold GPU run, where the 207 s go (2026-10-06, injection-scope)

Same scope and arms as the wall-time entry above, instrumented the rte6515 way (`OLF_SA_PROFILE=1
OLF_GPU_PHASE_TIME=1 OLF_GPU_SA_JFR=<file>`, then a separate run under Nsight Systems, `nsys profile --trace=cuda`
around the mvn, `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum`). JFR does NOT see time inside the JNI
solve (the thread is in native code), so its 74 s weighted total IS the Java share; the device share comes from
the JNI timers. Cold GPU arm 207 s instrumented (198 s plain):

| part | wall | detail |
|---|---|---|
| base load flow + setup | 2.4 s | |
| **precompute** | **166.8 s** | JNI solve **133.4 s** (marshal 3.4, pack 1.3); Java 33 s: `planScenarios` → `distrQWeightOverrides` **18.8 s** (a whole-network `GpuAcDataExtractor.extract` per generator contingency at a DISTR_Q member: `ExtractionColumns.of` 13.3 s, `ziTreeRelocation` 10.9 s), chunk solver host 10 s |
| per-contingency loop | 28.5 s | `injectState` 14.5 s (**16,105 remap rebuilds** of 24,379: OLF publishes a new sorted-variable list after every restore, the injector rebuilt its permutation on list identity), `updateBusesFromState` 9.7 s (the per-bus array path was DISABLED by the couplers' DUMMY_P/DUMMY_Q variables, so every contingency took the exact walk: `AcSolverUtil.updateNetwork` + `isStateUnrealistic`), violations 5.5 s, restore 4.6 s |

Unlike rte6515 (#61, where single-threaded host code was the bottleneck), **the device solve is 64 % here**. Its
kernels (126.5 s of the 133 s; 9 chunks of ≤ 2709, 261 outer passes, 443 batched Newton iterations,
activeIterSum 234,089 scenario-iterations = 9.6 per scenario, widthIterSum 278,144 so ~16 % padded lanes):

| kernel | time | launches | per launch |
|---|---|---|---|
| **`fillZeroImpBatch`** | **32.2 s (25 %)** | 704 | 45.7 ms |
| cuDSS `fwd_batch_ker` | 28.1 s | 660 | 42.6 ms |
| cuDSS `spmv_ker` | 18.7 s | 1432 | 13.0 ms |
| cuDSS `bwd_batch_ker` | 17.9 s | 660 | 27.1 ms |
| cuDSS `factorize_batch_ker` | 16.2 s | 132 | 122.6 ms |
| `fillClosedBranchBatch` | 5.3 s | 701 | 7.6 ms |
| everything else | < 2 s each | | |

Two device leads: (D-A) the zero-impedance fill — each thread (element × scenario) ran three LINEAR SCANS over the
per-scenario (slot, element) lists (switch outages, base-open closures, tree overrides), O(nzi·nS·(lists)) per
launch; dense per-scenario masks built at upload replace them. (D-B) iterative refinement: 5 forward/backward
solves per factorization (1 + `OLF_GPU_IR_STEPS`, 4 on a network with couplers) — the fwd/bwd/spmv trio is
64.7 s, four fifths of it refinement; cuDSS 0.7.1 has no refinement tolerance (`CUDSS_CONFIG_IR_TOL` "not
supported right now"), so the lever is the step count (IR=0 is known to break parity: DAP worst 1.07 MW; IR=2 under
test on the exact rte6515 generator kind). Host leads H1 (analytic DISTR_Q reweight instead of a re-extraction),
H2 (element-wise reference compare of the variable list instead of a rebuild), H3 (let DUMMY_P/DUMMY_Q keep the
per-bus array path) are coded; measurements below.

**Measured, same scope, GPU arm cold against the same 20-thread CPU arm (417.7 s), statuses identical
`{CONVERGED=24376, MAX_ITERATION_REACHED=3}` on both arms, 24374/24379 on device; exact parity kept on rte6515's
generator kind (1389/1389, 85876 violations, DAP worst 2.4e-06) and branch kind (9040/9042, 583458, 7.3e-07),
fixture suites 226/226:**

| state | GPU cold | device solve | vs CPU-20t |
|---|---|---|---|
| start (profiled run) | 207 s | 133.4 s | x2.02 |
| + H1 + H3 (host) | 177.5 s | 133 s | x2.35 |
| + D-A zero-impedance masks | 146 s | 102.1 s | x2.86 |
| + D-B adaptive refinement (default) | **139.7 s** | **91.3 s** | **x2.93** |

- **D-A**: `fillZeroImpBatch` 32 s → the masks replace three per-thread list scans; on this scope the lists are
  144–1335 switch outages and ~11,000 tree overrides per 2709-scenario chunk, times 2653 elements × 2709 lanes
  of scanning. `OLF_GPU_ZI_MASKS=0` restores the scans (A/B). A first version gated the tree-override mask on
  the off/on setter having run first; the setters' order is the caller's, so chunks whose tree list came first
  lost every override and a coupler child (LLJ_2, zi#22 LMQ) lost its ZERO_V: 2,308 scenarios FAILED as
  unrealistic (V 0.787) under BOTH refinement policies. Found by switching each change off in turn
  (`OLF_GPU_BUS_ARRAY=0`, `OLF_GPU_ZI_MASKS=0`, `OLF_GPU_DQ_REWEIGHT=extract`): only the masks switch cleared it.
- **D-B**: cuDSS 0.7.1 has no refinement tolerance, so the loop is ours: cuDSS runs `OLF_GPU_IR_BASE` (1) step,
  then per batched iteration the kernels `residualBatch` / `irScoreBatch` measure every lane's
  ‖b − J·dx‖∞/‖b‖∞ and the batch refines (J·δ = r, dx += δ) while a lane sits above `OLF_GPU_IR_TOL` (1e-12),
  at most `OLF_GPU_IR_MAX` (4) in all, undoing a step that does not lower the worst ratio. On this scope
  541 checks, **285 extra solves for 443 batched iterations (0.64 per iteration against 3 fixed)**.
  `OLF_GPU_IR_STEPS=<n>` keeps a fixed count (A/B: 4 → 146.6 s, same statuses). Fixed 2 was tried first:
  73 s of solve but two busbar scenarios of rte6515 at a linear-solve failure, which is why the count is
  decided from the residual. It hooks the UBATCH full-width and compacted solves (the production path) and
  the batch-matrix API path; the sequential robust retry keeps 4.
- **H1**: `distrQWeightOverrides` recomputes the touched groups' weights analytically (`reweightedDqWeights`:
  `qPct_k − (j == k)`, with the coupler side's sign on kind-3 terms — the first version dropped that sign and
  seven busbar fixtures caught it; `OLF_GPU_DQ_REWEIGHT=check` prints every difference against the extraction,
  `=extract` restores it). Groups without a candidate table still re-extract (the free role may move).
- **H3**: `updateBusesFromState` keeps its per-bus array path when the base holds DUMMY_P/DUMMY_Q (equation-only
  variables the network update never writes); `AcSolverUtil.updateNetwork` 8.45 s → 1.5 s. `OLF_GPU_BUS_ARRAY=0`
  restores the exact walk.
- **H2 is NOT in**: an element-wise compare of the sorted variable list against the base order did not reduce
  the 16,105 remap rebuilds (12,006 with a weaker variant) — OLF's rebuilt list does not hold the same Variable
  instances in the same order, so the identity test fails even when the set is the base's. `injectState` stays
  at ~12.7 s; the lever is a cheaper permutation build or an OLF-side fact ("the index is back to the base").

**Where the remaining 139.7 s go** (same instruments): device solve 91.3 s (factorization 16 s, the forward/
backward solves now ~1.6 per factorization, the closed-branch fill 5.3 s, the zero-impedance fill after the
masks to be re-profiled), precompute Java ~16 s, per-contingency loop 20 s (inject 12.7, violations 5.4, restore
4.6), marshal + pack 5 s, base and setup 2.4 s.

The rte6515 busbar scope was used as a coupler-heavy regression gate during this work and showed device failures
that moved from run to run (BUS-5011, BUS-570_1/2, BUS-4913/6308/900, BUS-6218_3 — a different set each run,
with every change switched off as well); busbar sections are outside this scope and that is recorded as an
open question for the busbar track, not gated here.

**Second profile (2026-10-06, after D-A/D-B/H1/H3, `nsys stats` re-exported with `--force-export=true` — the
first attempt failed on a stale `.sqlite` beside the recording):** 86.3 s of kernels for the 91 s solve:

| kernel | time | launches | per launch |
|---|---|---|---|
| cuDSS `fwd_batch_ker` | 26.6 s (31 %) | 634 | 41.9 ms |
| cuDSS `bwd_batch_ker` | 16.9 s | 634 | 26.7 ms |
| cuDSS `factorize_batch_ker` | 16.1 s | 132 | 121.9 ms |
| cuDSS `spmv_ker` | 10.4 s | 647 | 16.1 ms |
| `fillClosedBranchBatch` | 5.3 s | 701 | 7.6 ms |
| `irScoreBatch` + `residualBatch` | 2.1 s | 545 + 545 | 2.7 + 1.2 ms |
| `fillZeroImpBatch` | 1.0 s (was 32.2) | 704 | 1.4 ms |
| everything else | < 1 s each | | |

The masks took the zero-impedance fill out of the picture (32.2 → 1.0 s). What is left is cuDSS: the batched
triangular solves are HALF the kernel time at 69 ms per forward+backward pair, 57 % of a factorization — and
`spmv_ker` is cuDSS's OWN residual for its one internal refinement step (`OLF_GPU_IR_BASE` = 1), 16 ms per
solve where our `residualBatch` does the same product in 1.2 ms (the 647 ≈ 634 launch counts say that step
never went on to a second solve: the library checked and stopped).

- **D-C, cuDSS's internal refinement OFF (`OLF_GPU_IR_BASE=0`), the adaptive loop alone**: solve 93.2 s →
  **69.4 s**, GPU cold 139.7 s → **116.2 s (x3.60 vs CPU-20t)**, identical statuses, identical Newton
  trajectory counts (443 batched iterations, activeIterSum 234,089), the same two device failures as the
  reference (FUCZ, FUDC) and the same 20-thread-arm DAP mismatch (1.93 MW, retained switches). The loop now
  checks 794 times and refines 538 (1.2 per iteration): more of our cheap checks and solves, none of the
  library's expensive check. Parity gates (rte6515 generator and branch kinds exact) below.
- **H2 exact hint, measured and REVERTED**: a caller-side proof that the scenario disabled nothing carrying a
  variable (no bus, no branch with RHO1/ALPHA1, no SHUNT_B shunt, no HVDC) plus a size match let the injector
  keep the identity permutation without a rebuild — rebuild count unchanged at 12,006 and `injectState`
  14.3 s. So those 12,006 rebuilds are scenarios whose variable COUNT changes (a bus loses its variables), not
  OLF republishing an unchanged list; and the rebuild is already array-keyed (`baseRowKeyed`), so the 14 s is
  the `setValuesOnly` + per-rebuild allocation, not hash lookups. Not pursued further this round.
- `OLF_GPU_SOLVE_ALG` (0.8.0: default|general) and `OLF_GPU_FACT_ALG` (default|multiblock|general) reach every
  cuDSS config this file builds, for the sweep below.
- **D-C was the default until 4b2c7a4** (`OLF_GPU_IR_BASE` 0): that commit set it back to 1 on coupler systems
  after the full busbar scope, which this measurement did not cover, lost most of its device share and turned
  rounding-dependent with 0 (5121 / 5422 of 12918 on device against 12860 / 12858 with the step). Coupler-free
  networks keep 0. The network.xiidm non-busbar numbers below were taken with 0. Gates: rte6515 generator kind
  1389/1389, 85,876 violations diff 0, DAP worst 1.04e-06 MW; branch kind 9040/9042, 583,458 diff 0, 7.3e-07 MW;
  the fixture suites 226/226 but for the in-suite flake below.
- **cuDSS algorithm sweep, no gain** (network scope, D-C on, cold): solve-phase `general` 69.6 s solve /
  119.3 s cold; factorization `multiblock` 72.5 s / 118.5 s (it changes the LU probe: 10 chunks of ≤ 2438,
  479 iterations); factorization `general` 70.3 s / 118.8 s; the defaults 69.4 s / 116.2 s. Defaults kept.

**Where the 116 s go now**: device solve 69 s (forward/backward solves ~45 s at ~1.2 extra solves per
iteration, factorization 16 s, closed-branch fill 5 s), precompute Java ~16 s, per-contingency loop 20 s,
marshal + pack 5 s, base and setup 2 s. The next device lever is the triangular-solve pair itself (69 ms per
batched pair, 57 % of a factorization — a cuDSS property, not ours), i.e. fewer solves per iteration: a
looser-but-exact refinement decision (the check is 4 ms, a refinement 69 ms) or an initial solve the library
gets right more often (scaling the repaired zero-diagonal rows, the open lever noted above the adaptive policy).

### #76 — GEN-1522: a pilot that loses its own generator kept its voltage identity (2026-10-06, DONE)

Asked whether GEN-1522, the 2026-09-22 worst DAP on the rte network (~1e-6 MW then), was fixed. Measured with
today's code on the GENERATOR kind (1389 contingencies at every voltage, CPU arm 1 thread; note the default
`branch` kind holds no generator contingency and a CPU-arm save writes `cpu-busv.bin` only with
`OLF_GPU_SA_BUSDIFF_CTG` set): **3.200 kV at VL-1525 and 1.98e-02 MW** — much worse than in September, with
identical switch sets on both arms (VL-1522_1, VL-6490_0, VL-6055_0, VL-6231_0 all MAX_Q, same q and qLimit to 9
digits) and the device's VL-1522_0 sitting at 1.122 = its target against OLF's 1.0811.

**The shape** (the composer's own comment already names GEN-1522): the lost generator's bus VL-1522_0 is the
PILOT of its group and one of its own controllers; the one survivor VL-1522_1 pins at 0.7. OLF then drops the
pilot's BUS_TARGET_V and solves its BUS_TARGET_Q = 0 (a dead bus's own balance). The device arms the in-solve
row flip for exactly this (`GenOutage.flipsPilotRowWhenAllPinned`, mode 4 on the pilot's row) — and three
things in a row kept the row from ever carrying that balance, each found with a per-pass probe
(`OLF_GPU_ROW_TRACE`, `OLF_GPU_ROW_TRACE_GROUP`, `OLF_GPU_ROW_PROBE`, `OLF_GPU_GENLOST_TRACE`, all new or
reused today):
1. **Kernel order.** `updatePilotVeqBatch` (the pilot-self "control lost" layout) ran before
   `pilotRowAllPinnedBatch` and put the row in mode 1 — an EMPTY row for a pilot-that-is-its-own-controller —
   so the outage flip found the row off its identity and did nothing. Reordered: the outage flip first.
2. **The row was still empty in mode 4** (`ROW_PROBE`: F = 0, no Jacobian entry): the pilot's own reactive
   terms go through the candidate table, gated by `psSkip`, and the gate was CLOSED (`GROUP_TRACE psSkip=1`).
3. **Why the gate was closed:** the host put the dead bus's reactive-limit entry on the ISLANDED channel
   (`silencesReactiveLimitEntry` → `rlIslandedList`, "the same channel a stranded controller uses"), and
   `pilotSelfGateBatch` reads islanded as "the bus is gone, no balance to emit". A bus that lost its generator
   is alive. New channel `setBatchRlGenLost` (per-scenario (slot, entry) mask → `initRlLimitTypeBatch` seeds
   the entry `RL_LIMIT_FROZEN`): no pin, no switch back to PV, "not enabled" to the pilot-self layout, and the
   gate OPENS so the balance is emitted — on the carrier while the survivor controls, on the pilot's own row
   (mode 4, candidate 0 of its table) once every survivor pins. Restricted to the outaged PILOT: a dead MEMBER
   stays on the islanded channel (its balance is already expressed by the inactive-member weights on the
   survivors' rows; frozen, it was also emitted raw on the carrier — deviceGenOutageAtDistrQBus and
   deviceTwoMemberPilotControllerLosingAController went 0.37 kV off).

**Measured, generator kind, GPU arm cold vs the saved 1-thread CPU arm:** GEN-1522 **3.200 kV → 4.1e-09 kV**,
DAP **1.976e-02 → 2.6e-06 MW** (6.8e-07 on one run); scope worst DAP 1.976e-02 → 2.55e-06 MW; 1389/1389 on
device; statuses `{CONVERGED=1389}` both arms; violations 85876 = 85876. Fixture
`devicePilotLosingItsOwnGeneratorWithPinningSurvivorMatchesCpu`: a 400 kV mesh, the pilot's own generator plus
one remote controller capped under the post-outage need (slack non-participating, as PARTICIPATING_SLACK is a CPU
shape). On the islanded channel the device spends its iteration budget on the empty row (MAX_ITERATION_REACHED,
CPU fallback); with the frozen channel it batches and matches at 1e-12. Suite 207 tests, busbar scope
(6604/6667, 503923 violations, DAP worst 1.353e-04) and branch scope (9040/9042, 583458, 7.33e-07) unchanged.

### #70 — host the surviving REMOTE cluster of a split voltage-control group (2026-10-04, IN PROGRESS)

**What `VC_REMERGE_MULTI` actually refused.** `OLF_GPU_QUALIFY_TRACE` on the full busbar scope, with
the contingency id now on the `reHome` line: **46 distinct refused re-homes**, 31 of them exactly
"pilot section lost, two coupler-joined sections survive" (`main=VL-2341_0 cluster=[VL-2341_1,
VL-2341_2] newMain=VL-2341_1`), the rest 3–6 sections. In **45 of 46** OLF's elected new main is
coupler-adjacent to the island — i.e. it is the CHILD of the coupler the contingency opens, which is
exactly where `GpuBatchChunkSolver`'s killed-coupler block already installs a voltage identity
(`rowZi[childVRow] = 0` + the child's own target). The 46th (VL-6029) has a control-less section
between the lost pilot and the elected main.

**OLF's rule for the new group** (`LfZeroImpedanceNetwork.updateVoltageControlMergeStatus` +
`recreateReactivePowerDistributionEquations` + `updateRemoteVoltageControlEquations`): re-elect the
main (highest target, then controlled-bus id), rebuild `DISTR_Q` over the CLUSTER's controllers with
keys renormalized by `updateReactiveKeys` (islanded -> 0), `BUS_TARGET_V` on the new main, the
dependents' V targets off, `DISTR_Q` active for every enabled controller except the FIRST.

**Why the device can host it without a new table** — every piece already exists:
- the lost pilot is an INACTIVE dq member (`islandedDistrQEntries` emits one per islanded packed
  member; the kernel's `updateDistrQTargetsBatch` drops its key from the denominator);
- the survivors' weights renormalize through `distrQWeightOverrides` with the lost controllers'
  generators as "gone" (the RENORMALIZE channel) — once `leftControllers` names them;
- the new main's identity comes from the killed-coupler block (45/46), or an explicit
  `ROW_V_IDENTITY` when the elected main is not a killed coupler's child;
- the lost pilot's raw q and carrier override are already suppressed for a pilot-self group with an
  RL entry (`psSkip = limitType == 0`, islanded entries keep their base limit type 0); the one hole —
  `updateDistrQTargetsBatch` writing the DEAD pilot's `pilotQ` onto `relocChoose`'s pick when
  `psEntry < 0` (`PS_NO_ENTRY`) — is closed by a `rowIsFrozen(pilot row)` guard mirroring
  `updatePilotVeqBatch`.

**Implemented:** `reHomedControls` accepts a non-local cluster when it is EVERY survivor of the base
group (one surviving cluster), returning `remoteClusterBuses` (removed from `deadControlBuses`, so no
own-Q entries) and `remoteMainTargets` (the elected main); the lost sections' controllers go to
`leftControllers` (-> weight overrides + inactive entries). `remoteMainIdentityEntries` adds the
explicit identity only when the killed-coupler block does not cover the main, refusing
`VC_REMERGE_RELOC_VROW` when the main's own V row hosts a LIVE coupler's `ZERO_V` (VL-6029: row 13248
= `zi#327.ZERO_V`) and `VC_REMERGE_PINNED_MAIN` when it is pinned. Kernel guard in
`updateDistrQTargetsBatch`. New fixture `deviceLostPilotSectionReHomesSurvivingClusterMatchesCpu`.

**Measured so far (GPU-only full busbar scope; census counts are per-attempt, doubled by the retry):**

| | before | after |
|---|---|---|
| device coverage | 6550 / 6667 | **6585 / 6667** (+35) |
| `PLAN:VC_REMERGE_MULTI` | 47 | 22 |
| `RUNTIME:status=-2` (leaving the device) | 74 | **36** |
| unit gates | 200 / 0 / 1 | **201 / 0 / 1** |

37 of the 47 re-homes accepted. The 10 still refused are MIDDLE-section losses where the base main
SURVIVES in its own cluster and a SECOND non-local cluster forms (`BUS-6002_BBS_2`: main VL-6002_0
alive, cluster [_2.._5]) — two groups on one candidate table; refused by design for now, next step.

**The both-arms proof of that state FAILED (2026-10-04, `remerge-both.log`, full busbar scope, CPU 1 thread):**
`statuses: CPU={CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60} GPU={CONVERGED=6601, FAILED=6,
MAX_ITERATION_REACHED=60}` — `STATUS MISMATCHES (1): [BUS-6194_BBS_1]` (CPU CONVERGED with 60 violations,
GPU FAILED with 0); `violations: CPU=503923 GPU=504060 diff=137` (BUS-2640_BBS_1 +197, BUS-6194_BBS_1 −60,
BUS-1359_BBS_1 −2, BUS-1869_BBS_1 +2); DAP worst **193.85 MW at BUS-2640_BBS_1**, 0.473 MW at BUS-6003_BBS_1,
both CONVERGED on both arms. All three are newly accepted re-homes. The device's system was NOT OLF's.

**Root cause — a target-override RACE, visible only in the full batch.** BUS-2640_BBS_1 in a
3-contingency list (`OLF_GPU_SA_CTG_ONLY=BUS-2640_BBS_{1,2,3}`, so its couplers stay retained) is
bit-identical to OLF (statuses 3/3, violations 164=164, DAP 1e-13, BUSDIFF 2.8e-11 kV). The device
row-mode dumps of the two runs for the SAME buses differ in ONE line: the elected main VL-2640_1's V
row (`OLF_GPU_ROWMODE_DUMP_CTG` + `_ALL`, rows mapped through `OLF_GPU_SA_EQ_DUMP`):

    full list, slot 1844:  row=5956 mode=0<-1 target=0.0000000000000000
    3 ctgs,   slot 0:      row=5638 mode=0<-1 target=1.0340000000000000

Two producers wrote that row: `remoteMainIdentityEntries`' `ROW_V_IDENTITY` (1.034) and
`islandedDistrQEntries`' renormalized DISTR_Q target seed (0.0 — VL-2640_1 is a packed member whose
DISTR_Q row IS its own V row). `GpuBatchChunkSolver`'s seed loop skipped mode-4 rows only (its comment
kept mode 0 because skipping it had once moved a test by 1.06e-12 kV — the race's own noise), and
`GpuBatchOverrides.add` appended a second entry, which `applyTargetOverrideBatch` (one thread per entry)
resolves by scheduling: 0.0 won at slot 1844, 1.034 at slot 0. A V identity at 0 drives the bus to 0 pu
in clamped steps — 193 MW of slack, 197 violations, and a "CONVERGED" status. The killed-coupler block's
identity (the `blockCovers=true` cases, BUS-6003_BBS_1 and BUS-6194_BBS_1) raced with the same seed on
the child's own row.

**Fixed (same day):**
- the seed loop skips mode-0 rows too (an identity row's target belongs to the identity);
- `GpuBatchOverrides.add` REPLACES in place — the device never receives one row twice; a conflicting
  duplicate is counted on every run and the chunk solver warns + records `HOST:OVERRIDE_CONFLICT`
  (`OLF_GPU_OV_DUP_TRACE=1` names the rows). A sound plan has zero;
- the TAIL (no-island) site's local re-home loop lacked the PINNED_MAIN refusal the island site has:
  BUS-6157_BBS_1 (VL-6157_1 baseLimitType=1), BUS-6194_BBS_2 (VL-6194_2, 2), BUS-6229_BBS_1 (VL-6229_1,
  2) batched with an identity on a base-pinned bus and failed on the device. Now `PLAN:VC_REMERGE_PINNED_MAIN`;
- the harness's "tried and failed" split now works by ID (`GpuBatchScopeCensus.planRefusedIds`,
  recorded where a never-batched scenario gets `STATUS_REJECTED`): the census counts are per attempt,
  and subtracting them printed "-20 of them tried and failed" and called 12 plan refusals device failures
  (BUS-6002_BBS_2/3/4, BUS-6029_BBS_1/2, BUS-6134_BBS_2, BUS-6139_BBS_2/3, BUS-6218_BBS_2/3, BUS-689_BBS_2,
  BUS-710_BBS_2 — all `iters=0 norm=NaN`, i.e. never batched; `QUALIFY_TRACE result` now says so).

**The fixture defect, resolved:** OLF's `Networks.retainAndCloseNecessarySwitches` sets `retained=false`
on EVERY node-breaker switch and re-retains only the switches some listed contingency opens or closes —
a builder's `setRetained(true)` is inert, and with `bbs0` alone in the list `c12` collapsed `vl12_1`+`vl12_2`
into one bus. The test now lists `bbs2`'s loss as well (so both couplers exist) and expects BOTH to batch;
its trace reads `cluster=[vl12_1, vl12_2] … local=false`, `remoteMain bus=vl12_1 … blockCovers=true`. A
second fixture, `deviceLostPilotSectionRootMainExplicitIdentityMatchesCpu` (bbs0 without a line, so it
cannot root the coupler tree and vl12_1 keeps its own row), is BUS-2640_BBS_1's shape — explicit
`ROW_V_IDENTITY` on the very row the DISTR_Q seed targets — and asserts `HOST:OVERRIDE_CONFLICT == 0`.
Same mechanism, general lesson: with the full busbar list every FEEDER breaker is a retained LfSwitch, so
the base coupler flows, the coupler trees and the row numbers all differ from any narrowed list — a bug
that only shows in the full run is a slot-dependent or topology-dependent one, and the first move is to
diff the two scopes' per-scenario row modes/targets for the same buses.

**Proof after the fix (`fix70-both.log`, both arms, full busbar scope, `OLF_GPU_OV_DUP_TRACE=1`):**
`OV_DUP chunk … conflictRows=0` in every chunk (170/229 same-value duplicates, none conflicting) — the race is
gone by construction. BUS-2640_BBS_1: violations 251 → 54 = CPU, DAP 193.85 MW → 1.0e-13 MW; total
`violations: CPU=503923 GPU=503863 diff=60` (all of it BUS-6194_BBS_1). Device coverage unchanged at 6585 / 6667.
Crossing the GPU-only `QUALIFY_TRACE result` lines with OLF's statuses: of the 63 scenarios the device TRIED
and rejected, 62 are OLF's own 65 non-convergences; **the only device failure on a system OLF converges is
BUS-1219_BBS** (the outer-loop cycle, #69). The 19 others in the old "20 tried and failed" list never ran
(`iters=0 norm=NaN`).

**Still open after the fix — three re-homes disagree with OLF:**

| contingency | after the fix | 3-ctg list |
|---|---|---|
| BUS-6194_BBS_1 | GPU FAILED, CPU CONVERGED (60 violations lost) | — |
| BUS-6003_BBS_1 | DAP 0.473 MW, both CONVERGED | — |
| BUS-2640_BBS_1 | bus worst 2.231e-06 kV (DAP 1e-13) | 2.75e-11 kV |

plus the pre-existing BUS-1359_BBS_1 / BUS-1869_BBS_1 ±2 violations (DAP 1e-12) and the #69 DAP tail
(BUS-5250_BBS 1.398e-03, BUS-5525_BBS 1.060e-03).

The first hypothesis — the elected main pins during the scenario and the device drops its identity — is
REFUTED by OLF's own switches (`OLF_RL_TRACE=1` on the saved CPU arm): BUS-6194_BBS_1 switches only VL-6189_0
(MAX_Q), BUS-6003_BBS_1 pins VL-6490_0 and toggles VL-6229_0/_1 (MAX_Q → unpinned → MIN_Q), BUS-2640_BBS_1
switches nothing. No re-homed main pins. What the two broken ones share instead is in OLF's BASE switches:
`RL_SWITCH ctg=base PV->PQ bus=VL-6194_0 … MAX_Q` and `… bus=VL-6003_0 … MAX_Q` — **the lost pilot was PINNED
in the base** (VL-2640_0 was not). Mechanism, read off the kernels: `initRlLimitTypeBatch` seeds an ENROLLED
pilot-self pilot's limit type from its base kind (`olfKind`, there to keep a base-pinned pilot pinned), with no
notion of the pilot being islanded this scenario; `pilotSelfGateBatch` then reads `limitType != 0` as "pinned:
its balance rides the carrier" (`psSkip = 0`), so `fillDistrQBatch` SUPPRESSES the carrier member's own DISTR_Q
(the `-2 - d0 == carrier` test) for a balance OLF removed with the bus — VL-6194_2's row lost its equation.
`updateDistrQTargetsBatch` was already guarded (`!pilotFrozen`, the #70 guard); the gate and
`rlPilotSelfMixedGate` were not. Fixed: both take the per-scenario `rlIslanded` mask — an islanded pilot reads
as "no balance" (skip = 1) and never counts as a pinned pilot or a controlling member. New fixture
`deviceLostBasePinnedPilotSectionMatchesCpu` (g0 capped at 5 MVar so the base pins it, then its section is
lost). BUS-2640_BBS_1's 2.2e-6 kV (DAP 1e-13, no switches on either arm) is a different class — the
stopping-point slack the DAP tail already carries (#67/#69), to be read with the iteration counts, not #70.

**Second proof (`gpu-load-1359.log`, GPU arm against the saved CPU arm, all of the above in):** `statuses:
CPU={CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60} GPU={CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}`,
`violations: CPU=503923 GPU=503923 diff=0`, `conflictRows=0` everywhere, device coverage 6583 / 6667
(BUS-6194_BBS_1 and BUS-6003_BBS_1 now `PLAN:VC_REMERGE_PINNED_MAIN`, their CPU answers match to 3e-11 /
4e-10 kV), the only device failure on an OLF-converging system BUS-1219_BBS (#69's cycle), 21 plan refusals on
OLF-converging systems listed by id. The full test still fails on the pre-existing DAP tail (BUS-5250_BBS
1.398e-03 MW, the 1e-3 assertion), and two batched contingencies still disagree with OLF in the REACTIVE state
only: **BUS-1359_BBS_1 (0.465 kV, violations −2) and BUS-1869_BBS_1 (0.338 kV, +2)**, DAP 1e-12 — both
`split=CONTROL_DIES` with TWO surviving remote controllers on a coupler chain (`deadControl=[VL-1359_1,
VL-1359_2]`), present in every run since before #70. Device rows agree with OLF's equation SET
(`bus_target_q[1569]`/`[1570]` active ↔ rows 3138 / 13977 mode 4; `bus_target_v`, `distr_q` inactive), so the
difference is in a TARGET or a term — being read off a 3-contingency run with both arms' dumps
(`OLF_VAR_DUMP` + `OLF_GPU_ROWMODE_DUMP_CTG`).

**Read off (2026-10-05): a TERM.** The 3-contingency list reproduces it (0.4648 kV on BUS-1359_BBS_2/3 = the
survivors themselves: OLF 65.863 kV, device 66.328). OLF's converged `EQ BUS_TARGET_Q VL-1359_1 … target 0.0`
and the device's `row=2936 mode=4 target=0.0` agree; the device's first-fill matrix (`OLF_GPU_VALS_SCAN=1
OLF_GPU_VALS_DUMP=… _ANY=1 _NTH=2 _SCAN_SLOT=0`, columns named through the SA eq dump) shows row 2936
(VL-1359_1's own reactive balance) with its three branch terms but **no `DUMMY_Q(BUS-1359_COUPLER_2)` column**,
while OLF's `bus_target_q[1468] = … + -1.0 * dummy_q9042 + dummy_q9043` carries the live coupler's dummy
(the P row 2937 does carry `+DUMMY_P(COUPLER_2)`). Cause: VL-1359_1 is a DISTR_Q controller (own Q balance
SUPPRESSED, `b1Q = -1`) relocated onto COUPLER_1's dummy row; the zi dest builder gave COUPLER_2's bus1
dummy-Q term a plain two-way alternate "emit at the dummy row 13045 when that row is mode 0/4" — mode 4 in
the base (pinned/dead there) is right, but mode 0 (COUPLER_1 dead this scenario) sent the term into the
identity row, where applyRowMode wipes it, instead of the bus's OWN row (relocOwn[13045] = 2936). The cb/sh
raw-q families already encode this as a STOCK PAIR (`stockPairOut` / `stockPairPick`: dummy row on mode 4,
own row on mode 0 + own mode 4, nothing while sharing); the zi family now does the same for a suppressed
side whose alternate is a relocated home, and the pattern gets the (own row, dummyQ) slot. Fixture
`deviceLostPilotWithRemoteSurvivorsOnCouplerChainMatchesCpu` (g1/g2 regulate bbs0 remotely, bbs0 lost, the
survivors chained by c12) — measured against the OLD native lib first: `vl3_b expected 396.486 but was 394.741`
(1.75 kV), `split=CONTROL_DIES deadControl=[vl12_1, vl12_2]` — the shape reproduces. With the fix: 20 targeted
tests green, and the 3-contingency rte6515 list gives `BUSDIFF ctg=BUS-1359_BBS_1 … worst abs=5.369e-11 kV`,
`violations: CPU=214 GPU=214` (was 0.4648 kV, 214 vs 212).

**Final proof of the day (`gpu-load-final.log`, full busbar scope, GPU arm vs the saved CPU arm):** statuses
identical `{CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}`, `violations: CPU=503923 GPU=503923 diff=0`
with NO per-contingency violation difference left, `conflictRows=0`, device coverage 6583 / 6667, BUS-1359_BBS_1
4.654e-09 kV and BUS-1869_BBS_1 6.320e-06 kV (were 0.465 / 0.338). Every remaining bus difference is in the
1e-6 kV class with DAP ≤ 1.4e-03 MW (BUS-2640_BBS_1 2.2e-06, BUS-5250_BBS 1.6e-06, BUS-5525_BBS 1.2e-06 kV) —
the stopping-point class the DAP tail is, which is the next item, not #70. The full test's only failing
assertion is that DAP bar (BUS-5250_BBS 1.398e-03 MW > 1e-3). Unit suite 308 tests, the single failure being
the pre-existing coupler-chain first-step test noted above. #70 is DONE for the single-cluster re-home; the
21 plan refusals (two surviving clusters, pinned mains, relocated main rows) stay honest CPU answers.

**A second race the closure exposed (same day).** With duplicates resolved deterministically to the LATER writer,
`deviceBusLossIslandingBaseCarrierMemberMatchesCpu` failed by 0.45 kV: `OV_CONFLICT ctg=bus-b1 row=2 mode=1
shadowed=-2.000000000 new=0.000000000 by=GpuBatchChunkSolver.java:1076`. The frozen base carrier hands the
pilot's reactive balance to the next member's row and stamps the pilot's target there (−2.0, b4's load); the
islanded-member seed loop then wrote that member's own renormalized DISTR_Q target (0.0) on the same row. On
the device the pilot's write had been winning the race, so the test passed. Now the chunk solver keeps a
per-scenario `pilotBalanceRows` set and the seed loop skips those rows; any other pre-existing override on a
seed row is kept and counted (`HOST:DQSEED_SHADOWED`, 0 so far). 15 targeted tests green with
`OV_DUP … dupRows=0 conflictRows=0` in every chunk. The CPU arm of the full busbar scope is now SAVED
(`OLF_GPU_SA_CPU_ARM_MODE=save`, 787 s, `{CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}`) so every
further proof runs the GPU arm alone against it in ~8 min (`scratchpad/gpu_load.sh`).

**Pre-existing, not #70 (found by the full 306-test suite, which the 201-test gate did not run):**
`GpuZiCouplerTreeRelocationTest.relocatedCouplerChainStepsWithOlf` fails at 15 couplers — the device's FIRST
Newton step differs from OLF's dense one by 1.37e-6 (passed on 2026-09-28 logs). Base-LF path
(`GpuAcNewtonSolver.solve`, maxIter 1), untouched today; to bisect separately (a first-step difference IS a
fill/Jacobian difference on a relocated coupler chain).

### #69 — the non-batched tail SPLIT: 55 are genuinely lost, and one class is an OUTER-loop cycle (2026-10-02)

**The split, measured.** New `OLF_GPU_SA_STATUS_LIST=1` lists every NON-CONVERGED contingency of each
arm BY ID (the aggregate counts only say the arms agree; the ids say WHICH contingencies OLF itself
fails). Both-arms full busbar scope:

```
CPU non-converged: 65  {MAX_ITERATION_REACHED: 60, FAILED: 5}
GPU non-converged: 65  {MAX_ITERATION_REACHED: 60, FAILED: 5}
same id sets: True    status disagreements: 0
```

The arms' non-converged ID SETS are IDENTICAL with zero status disagreements — a stronger check than
the count equality the test asserts. Intersecting those with the device's rejects:

| device rejects (`status=-2`) | 84 |
|---|---|
| OLF ALSO fails -> the device may answer directly | **29** |
| OLF CONVERGES -> genuinely lost to the CPU | **55** |

So after [[#67]]'s retry the remaining tail is two populations, and only the 55 are a real loss.
(`BUS-1039_BBS`, the contingency [[#67]] was diagnosed on, is in OLF's own
`MAX_ITERATION_REACHED` set and is no longer rejected — the retry answers it. That conclusion holds.)

**One of the 55 characterised, and it is an OUTER-LOOP LIMIT CYCLE.** `BUS-1219_BBS` (slot 241 via
`OLF_GPU_ROWMODE_DUMP_CTG`), device `OLF_GPU_CONV_TRACE=241` against OLF's `OLF_STEP_TRACE`:

```
OLF : 11 steps, f=1.00000000 EVERY ONE (no clamping at all), maxDv 0.033 -> 0.0011 -> 8.1e-05,
      quadratic, across 2 NR solves.
device, per pass:  37.88 -> 1.45 -> 0.0118   converged
                    6.43 -> 0.0597 -> 4.1e-05 converged
                    5.5952083543771 -> 4.181 -> 2.767 -> 1.353 -> 2.7e-11 converged
                    5.5952083543765 -> 4.181 -> 2.767 -> 1.353 -> 2.5e-11 converged
                    5.5952083543765 -> ... BIT-IDENTICAL ... converged
                    (140 convergence checks over 32 passes)
GPU_REJECT slot=241 reason=outer-loop-budget outer=32 it=0
```

**The inner Newton CONVERGES EVERY PASS; the OUTER loop re-poses a BIT-IDENTICAL problem forever**
and the scenario dies on the outer budget, not the inner one. Within a pass the `inf` residual falls
by EXACTLY 1.0 per step (3.95640976953125 -> 2.95640977 -> 1.95640977 -> 0.95640977), which is
exactly `maxDphi = 1`: a ~4-unit correction on a unit-diagonal (DUMMY) row delivered 1.0 at a time.
Then the outer loop re-applies the same ~4-unit offset. Some outer loop is not registering its own
effect — it reads a state the inner solve has already fixed and re-applies the same delta.

**AND THE GENERALISATION WOULD HAVE BEEN WRONG — the 55 are almost all QUALIFICATION REFUSALS, not
solver failures.** Tracing a second member, `BUS-2341_BBS_1`, exposed it: **OLF converges it in 2
STEPS** (trivial), and the device produces `status=-2` with **NO kernel `GPU_REJECT slot=` line at
all** and no slot from `OLF_GPU_ROWMODE_DUMP_CTG`. The reason is
`GpuBatchedSecurityAnalysisSolver:431`: a scenario whose result row is NULL — one never batched —
is marked `STATUS_REJECTED`. **So `RUNTIME:status=-2` CONFLATES kernel rejects with PLAN-TIME
refusals**, and the census's `PLAN:` entries carry no contingency id (only `MAPPER:` reasons do),
which is why the plan population was invisible.

The accounting then closes EXACTLY (arithmetic, not id-level proof — the `PLAN:` trace prints no
ids):

| population | count | OLF | nature |
|---|---|---|---|
| `PLAN:VC_REMERGE_MULTI` | 47 | CONVERGES | device DECLINES to batch |
| `PLAN:VC_REMERGE_PINNED_MAIN` | 3 | CONVERGES | same |
| `PLAN:VC_REMERGE_RELOC_VROW` | 3 | CONVERGES | same |
| `PLAN`/`VALIDATOR:ISLANDS_SLACK` | 1 | CONVERGES | same |
| runtime reject, OUTER-LOOP CYCLE (`BUS-1219_BBS`) | 1 | CONVERGES | genuine SOLVER BUG |
| runtime rejects, `max-iter-bad-solve` | 29 | ALSO FAILS | normal — answer directly |
| **leaving the device** | **84** | | |

54 plan + 1 solver bug = the 55 lost; + 29 answerable = 84. **So the dominant gap is NOT a solver
bug**: it is `VC_REMERGE_MULTI`, where a coupler outage re-splits a merged generator-voltage-control
group and the re-homed cluster is not LOCAL (`if (!local) return new ReHomed(pv, left,
"VC_REMERGE_MULTI")` in `GpuBatchQualification`), so the device refuses the layout. Supporting it
means laying out a RE-HOMED REMOTE group per scenario — real work, not a bug fix.

**Priority follows the user's rule** ("if the CPU fails it is normal that the GPU fails and it should
answer directly; if the CPU converges and the GPU does not, it is a bug"): the 29 cannot be answered
directly UNTIL the 55 are fixed, because the device cannot tell the populations apart in advance —
"device fails ⇒ report it" is only safe once a device failure implies an OLF failure. So: 47
`VC_REMERGE_MULTI` first (the bulk), then the 7 other plan refusals, then the 1 outer-loop cycle,
and the 29 come along for free.

**Superseded caution (kept):** `outer-loop-budget` is only ~5 of the kernel rejects against 108
`max-iter-bad-solve`, so `BUS-1219_BBS` looked like a minority class of the 55. The next step is to pick a `max-iter-bad-solve` member OF THE 55 (not of the 29 OLF also
fails, which is the trap that made [[#67]] look like an equation bug) and trace it the same way:
`OLF_GPU_ROWMODE_DUMP_CTG=<id>` for the slot, then `OLF_GPU_CONV_TRACE=<slot>` against
`OLF_STEP_TRACE=1 OLF_TRACE_CTG=<id>`.

**Which outer loop cycles is NOT yet identified** for the `BUS-1219_BBS` class — that needs the
device's outer-loop driver trace, not another convergence trace.

### #68 — the HVDC AC-emulation target lands on a relocated converter bus's ZERO_PHI row (2026-10-02)

Found by doing what the equation-identity comparison could not: diffing the NUMBERS. Added
`OLF_GPU_ROWMODE_DUMP_ALL=1` (the existing `OLF_GPU_ROWMODE_DUMP_CTG` prints only rows DIFFERING
from base, so a wrong target that happens to equal the base's is invisible) at `%.17g`, and diffed
the device's per-row target vector against OLF's `F0` section (`F0 <column> <type> <elementNum> <id>
<residual> <f> <target>`) for `BUS-1039_BBS`: 14504 device rows vs 14486 OLF active equations, the
18 difference being the 9 islanded buses x 2.

**Result: 14407 rows compared, exactly 10 target mismatches, every one a `ZERO_PHI` on a VSC-station
breaker — all 5 HVDC links x 2 stations, in near-equal-and-opposite PAIRS:**

| breaker | OLF target | device target |
|---|---|---|
| `SYN_VSC_4_2` / `4_1` | 0 | -3.9271053690e-05 / +3.8489558261e-05 |
| `SYN_VSC_1_1` / `1_2` | 0 | -1.0302481809e-05 / +1.0097462320e-05 |
| `SYN_VSC_3_1` / `3_2` | 0 | -5.6748343971e-06 / +5.5619051620e-06 |
| `SYN_VSC_2_1` / `2_2` | 0 | -1.0234645009e-06 / +1.0030975564e-06 |
| `SYN_VSC_0_2` / `0_1` | 0 | -8.0468953445e-08 / +7.8867621265e-08 |

A `ZERO_PHI` is `phi_parent - phi_child = 0`; its target MUST be 0. The magnitudes identify
themselves: the CPU log's "Freezing HVDC SYN_HVDC_4 at previous active setPoint:
0.003927105143741727 MW" / 100 = 3.9271051e-05 pu is the `SYN_VSC_4_2` value, and the +/- pairing is
the link's two ends with its losses as the small asymmetry.

**The targets are SWAPPED, verified on one coupler:**

```
row=14478 DUMMY_P(SYN_VSC_4_2_BREAKER) mode=1 terms=[hv#4.p2, zi#353.p1(+dP)]  target=0
row=13679 BUS_PHI(VL-6215_5)           mode=1 terms=[zi#353.ZERO_PHI]          target=-3.927e-05
```

The row that actually carries the HVDC `p2` term has target 0, and the frozen setpoint sits on the
`ZERO_PHI` row instead.

**Cause.** `GpuBatchHvdcAcEmulation.buildControllers` resolves the converter buses' rows as the raw
`BUS_PHI` VARIABLE row:

```java
Map<Integer, Integer> phiRows = …;                       // BUS_PHI variable row per bus
int r1 = phiRows.get(hvdc.getBus1().getNum());
int r2 = phiRows.get(hvdc.getBus2().getNum());
```

and stores them in `Ctl(… int bus1PRow, int bus2PRow …)`. That single number does DOUBLE DUTY: it is
the row the P target is written to AND the column the angle is read from (`x[bus1PhiRow()]`). For a
bus NOT behind a coupler those coincide, because the stable indexing makes row == paired variable
column. **Under the coupler-tree relocation they do not:** a relocated child's P balance moves to its
parent coupler's `DUMMY_P` row while its own `BUS_PHI` row hosts the coupler's `ZERO_PHI`. So the
target write lands on the `ZERO_PHI` row. This is the bug class recorded in
[[coupler-relocation-default]] — every consumer of a bus's row must ask for the RELOCATED row — and
there is a `GpuAcRowIndexing.relocatedQRow` helper for the Q analogue but none for P. The code even
documents a previous instance of the same family: "A converter bus whose BUS_TARGET_P is INACTIVE
(the slack) is packed as `~phiRow` … Writing there pinned the base LF's slack angle at the freeze
delta (BUS_TARGET_PHI residual 0.014, **base MAX_ITER**)".

**THE FIX WAS IMPLEMENTED, MEASURED, AND REVERTED — it is CATASTROPHICALLY WRONG (2026-10-02).**
The two roles were split (`Ctl` gained `bus1PhiCol`/`bus2PhiCol` for the angle read, `bus{1,2}PRow`
kept for the target) and the target row resolved through a new `GpuAcData.relocatedPRow(phiCol)`, the
P analogue of the reactive twin, derived from the same zero-impedance pack. It worked mechanically —
the ten `ZERO_PHI` targets became exactly 0 — and the solve collapsed:

| | baseline | HVDC target on the relocated `DUMMY_P` row |
|---|---|---|
| `max-iter-bad-solve` | 108 | **13194** |

Reverted; the baseline re-measures at exactly 108, and the gates are 200 run / 0 failures / 1 skipped.

**So the PREMISE was wrong, and the "10 mismatches" are most likely a MAPPING ARTEFACT — the third
in this comparison.** `GpuAcData`'s own class contract says a spanning-tree edge carries
**`ZERO_V` on the `DUMMY_P` row and `ZERO_PHI` on the `DUMMY_Q` row** — NOT on the buses' own rows.
My diff script mapped OLF's `ZERO_PHI(branch)` onto the child's own `BUS_PHI` row because the
equation dump's `terms=` listing shows `zi#353.ZERO_PHI` scattering there, and inferred from that
that the row's target must be 0. The device's arrangement is evidently not that, so the nonzero
target on the bus's own PHI row is where the device WANTS the station's P target, and OLF's
`ZERO_PHI` lives elsewhere in the device's layout.

**What this costs and what it leaves.** The HVDC observation is retracted as a defect. What survives
is: (i) `OLF_GPU_ROWMODE_DUMP_ALL=1` at `%.17g`, which is how any future target diff gets done;
(ii) `GpuAcData.relocatedPRow`, kept UNUSED and javadoc'd as a warning with the 108 -> 13194 number,
so the same wiring is not attempted again; (iii) the measured fact that **14407 of 14407 compared
rows agree on target once the mapping artefacts are removed** — i.e. the target vectors agree too.

**Method lesson, now three times over in this entry.** Every "difference" this comparison produced
turned out to be my own mapping: `bus_target_q` mapped onto a pilot's V row (5 phantom mismatches),
the structure's canonical dq rows read as a runtime set (2 phantom rows), and now `ZERO_PHI` mapped
onto the child's own PHI row (10 phantom mismatches). **Before believing a dump-to-dump mismatch,
confirm the row's occupant from the device side** (`row=… TYPE(id) … terms=[…]`) AND from the
owning class's contract — the `terms=` listing alone is not the row's equation.

**CLOSED as not-a-defect.** It was never shown to cause [[#67]]'s 108 rejects, and the attempted fix
made them 13194, so the rows are doing what the device intends.

**THE JACOBIAN IS STRUCTURALLY SOUND — no rank defect of the obvious kinds (2026-10-02).** Slot 39
of the contingency chunk IS `BUS-1039_BBS` (the `ROWMODE_DUMP` names it, and the kernel's reject
trace lists slot 39 among the 54 distinct rejecting slots), so the device's own matrix dump was
taken for the very contingency whose equations and targets were verified above:
`OLF_GPU_VALS_SCAN=1 OLF_GPU_VALS_SCAN_SLOT=39 OLF_GPU_VALS_DUMP=<path> OLF_GPU_VALS_DUMP_ANY=1
OLF_GPU_VALS_DUMP_NTH=2` (the `VALS_SCAN` master switch is required — without it `VALS_DUMP` writes
nothing). Header: `n 14504 nnz 101690 outer 0 seq 13 slot 0 nS 3307` — `outer 0`, the pass the reject
happens in, and `slot 0` because `SCAN_SLOT` clones the traced slot there by design.

| check on the CSR at `outer 0` | result |
|---|---|
| non-finite values | **0** |
| zero ROWS | **0** |
| zero COLUMNS | **0** (the "dead columns 13171/13207" failure mode is absent) |
| zero DIAGONAL entries | **0** |
| row modes | 13494 power / 1001 identity / 9 own-Q |

**Where the residual lives:** on coupler `DUMMY_P` rows paired with their parent bus's `BUS_PHI`
row, in near-identical pairs — `DUMMY_P(BUS-6174_COUPLER_1)` 7.49986 against `BUS_PHI(VL-6174_0)`
7.50028, and `DUMMY_P(BUS-5967_COUPLER_1)` / `BUS_PHI(VL-5967_0)` both exactly 6.1324. That pairing
is just the ±1 dummy coupling (the dummy column enters one bus row with +1 and the dummy row with
-1), so a mismatch in a coupler's flow appears equally on both. `||F||2 = 36.33`, `||F||inf = 11.02`
at row `DUMMY_P(BUS-6139_COUPLER_3)`, 99 rows above 1.0, all of them `BUS_PHI` / `BUS_V` /
`DUMMY_P` / `DUMMY_Q`.

**NOT A COMPARISON, and deliberately not drawn as one:** OLF's `F0` section for the same contingency
gives `||f-target||2 = 623.0`, `||f-target||inf = 401.3` (263 rows > 1.0, the largest being
`DISTR_Q VL-1697_2` at 401.3). It is TEMPTING to read that as "OLF starts from a far worse point and
still converges" — but the two are NOT at the same point, and this file already records why: the
device's `outer 0` fill is taken AFTER its island freeze, pre-loss target seeding and `applyRowMode`,
while OLF's `F0` is its raw post-contingency start. An earlier session hit exactly this (OLF
norm2=774.03 against the device's, not comparable). To compare starting points the two must be
sampled at the same stage, which neither dump currently does.

**So after equations, targets and now the matrix, [[#67]] still has no located cause.** What is ruled
out: a different equation set, different targets, a structurally singular or non-finite matrix, a
factorization problem (mode 6 fixes the factorization for 39 of the 113 and none converge). What is
NOT yet done: sampling both arms' residual AT THE SAME STAGE, which is the only way the "device
starts better / worse" question can be answered, and the per-iteration trajectory comparison
(`OLF_GPU_DBG_NR` against OLF's own Newton) that was the original plan for this item.

**MECHANISM FOUND: the step is CLAMPED to ~18% and stays there (2026-10-02).** New
`OLF_GPU_STEP_TRACE=1` prints the MAX_VOLTAGE_CHANGE factor actually applied, per (iteration, slot),
plus which variable binds it — nothing printed this before, and `||dx||inf` from `LIN_RESID` is NOT a
proxy for it (the largest step component is typically a `DUMMY_P`/`DUMMY_Q` FLOW correction, and
those carry `varType 2`, which the clamp IGNORES: `extractVarType` maps only `BUS_PHI`->0 and
`BUS_V`->1). Slot 39 = `BUS-1039_BBS`:

```
STEP_FACTOR it=0 slot=39 f=0.180879102 maxDv=2.21142 atVrow=5706 maxDphi=2.27604 atPhirow=7179
STEP_FACTOR it=1 slot=39 f=0.185022873 maxDv=2.16189 atVrow=5706 maxDphi=4.18842 atPhirow=3203
```

`f = 0.4 / 2.21 = 0.181` — the V cap binds exactly, **the SAME row 5706 binds both iterations, and
`maxDv` barely moves (2.21 -> 2.16)**. So Newton takes 18% of its step, repeatedly, and 30 clamped
iterations cannot cover the distance. That is a mechanism for the whole 108.

**And the binding row is NOT a weak row — which kills the obvious explanation.** Row 5706 is
`BUS_V(VL-2525_0)`, a mode-1 PQ bus with 7 nonzeros, and in the dumped matrix:

```
row=5706 mode=1 x=0.997945118 target=-0.077 F=-2.24674446e-13 diag=417.021265 max|row|=417.021265
```

Its residual is ZERO and its diagonal is the LARGEST entry in the row (417). So the 2.21 pu move is
not `F_i/J_ii` at a weak diagonal — it is the SYSTEM's answer, forced by coupling. The residual
actually lives on the coupler `DUMMY_P` rows (11.02 pu at `DUMMY_P(BUS-6139_COUPLER_3)`), whose
diagonal is exactly -1 for all 366 of them (the dummy-flow coefficient), so a 1100 MW coupler
mismatch propagates into large voltage/angle corrections network-wide.

**Two more explanations ruled out:**

- *A weak/ill-conditioned local row:* no — diagonal 417, F = 0 (above).
- *The device pre-distributing the contingency loss while OLF lets the slack absorb it:* no —
  `BatchConfig.contingencyLossPreDistribution` is set from powsybl's OWN
  `CONTINGENCY_ACTIVE_POWER_LOSS_DISTRIBUTION_DEFAULT_VALUE`, so OLF pre-distributes too and the
  device mirrors it. (`GpuSaRunScope` even refuses the batch when the parameter is non-default.)
- *The clamp rule or its magnitudes differing:* no — the kernel computes
  `f = min(1, maxDv/max|dv|, maxDphi/max|dphi|)`, identical to OLF's
  `MaxVoltageChangeStateVectorScaling`, with the run's own `maxDv = 0.4` / `maxDphi = 1` and the same
  `maxNewtonRaphsonIterations = 30`.

**ANSWERED: THE TWO ARMS TAKE THE SAME NEWTON PATH. [[#67]] IS AN ITERATION-BUDGET PROBLEM, NOT AN
EQUATION PROBLEM (2026-10-02).** `MaxVoltageChangeStateVectorScaling` in the fork now prints its own
`stepSize` under `OLF_STEP_TRACE=1`, scoped by `OlfTraceScope` (so `OLF_TRACE_CTG=<id>` isolates one
contingency). Run against `OLF_GPU_STEP_TRACE=1` on the same GPU-only run — the rejected contingency
is CPU-routed INSIDE the GPU provider, so one run produces both sides:

| step | device `f` | OLF `f` |
|---|---|---|
| 0 | 0.180879102 | 0.180878732 |
| 1 | 0.185022873 | 0.185022597 |
| 2 | 0.0503344306 | 0.0503343347 |
| 3 | 0.121389299 | 0.121389102 |
| 4 | 0.119122665 | 0.119122490 |
| 5 | 0.105761096 | 0.105760896 |

**Over steps 0..29 the maximum RELATIVE difference is 7.098e-06 (median 1.891e-06), and the binding
variable is the same every time** — device rows 5706 / 7178 / 7418 against OLF's 5704 / 7168 / 7408,
the same variables under each arm's own numbering (`elem2853` IS `VL-2525_0`). By the standing rule,
identical trajectories mean identical equation systems: that question is now closed for this
contingency, from both ends.

**What both arms actually do: they OSCILLATE.** `f` alternates ~0.19 / ~0.018 with `maxDv` swinging
2.05 <-> 22.4 — a limit cycle under the MAX_VOLTAGE_CHANGE clamp, on both sides. The contingency is
simply HARD: 9 islanded buses, 17 branches out, 124.3 MW of load to redistribute, and a Newton that
cannot settle at `maxDv = 0.4`.

**The ONLY difference is what each side does when the cycle outlasts the budget.** At step 30 the
device's factor goes to 1 — it has abandoned the scenario (30 inner iterations in `outer 0`
exhausted, `status -2`, `max-iter-bad-solve`) — while OLF is still oscillating at `f = 0.193713` and
keeps going, because its `maxNewtonRaphsonIterations = 30` is a budget PER NR SOLVE that each
outer-loop pass refreshes, whereas the device spends all 30 inside its first outer pass and rejects.

**So the 108 are not a device defect in the equations.** The CPU fallback then solves them and the
run's status counts match, which is why no result was ever wrong — the cost is purely that 108
contingencies leave the device, against the no-CPU-fallback goal. The fix is therefore about BUDGET
and the stall, not about finding a wrong equation:

1. ~~Give the device the same budget semantics OLF has — 30 inner iterations PER OUTER PASS.~~
   **DISPROVED BY ONE VARIABLE (2026-10-02).** New `OLF_GPU_MAXITER=<n>` (diagnostic) overrides the
   inner budget; at **120, four times the budget, NOTHING changes**: coverage stays 6550 / 6667 and
   `max-iter-bad-solve` goes 108 -> 112. More iterations converge not one extra contingency, so the
   tail is NOT iteration-starved. (Also: the premise was already shaky — `for (int it = 0; it <=
   maxIter && nActive > 0; it++)` lives INSIDE the per-pass function, so `it` already resets per
   outer pass; what the reject does is set `live[s] = 0`, permanently retiring the scenario so it
   never reaches pass 1.)
2. **THE LIMIT CYCLE IS THE CAUSE, and what breaks OLF's is the OUTER LOOP.** Both arms oscillate on
   the identical path; OLF escapes because its outer loops CHANGE the problem between NR solves
   (slack distribution, PV->PQ), which is why its 31 scaled steps span at least two NR solves. The
   device inside `outer 0` has a fixed problem and oscillates indefinitely, then retires the
   scenario. So the fix is NOT more inner iterations but: **on inner-budget exhaustion, let the OUTER
   pass run instead of retiring the scenario** — keep `live[s]`, stop only `active[s]`, and reject
   only if it still stalls after the outer-loop budget. That is a different change from raising
   `maxIter`, and it is what OLF actually does.
   **IMPLEMENTED AND IT IS A NO-OP (2026-10-02).** `OLF_GPU_STALL_TO_OUTER=1` keeps `live[s]` at the
   `it == maxIter` branch (only `active[s]` stops). Result: coverage 6550 / 6667, 108
   `max-iter-bad-solve`, census BIT-IDENTICAL to baseline. The flag does reactivate the scenario —
   `active[s] = live[s]` runs for the next phase — so the conclusion is that **the outer loops find
   nothing to do with an UNCONVERGED state**: the next phase enters with `solvePass = false` ("re-CHECK
   every live scenario against the SAME state: STABLE never re-solves (OLF)"), so nothing perturbs the
   problem and the cycle is never broken. Kept behind the flag, default off, as a recorded dead end.

3. **THE ANSWER, AND IT RETRACTS MY OWN DISMISSAL OF THE MODE-6 RETRY (2026-10-02).**

   `OLF_GPU_CONV_TRACE=39` gives the device's full stopping decision, iteration by iteration:

   ```
   it=0 norm2=36.33  it=5 norm2=17.81  it=10 norm2=10.19  it=15 norm2=5.31
   it=19 norm2=3.61  it=21 norm2=3.07  it=25 norm2=2.42  it=29 norm2=2.12  it=30 norm2=2.26
   thr2 = 1.20357799913425e-02  (= 1e-4 * sqrt(14486), OLF's OWN formula, and nActive=14486
                                  is EXACTLY OLF's active-equation count)
   ```

   It converges steadily to ~2.1 and then STALLS in a 2-cycle, **176x above threshold**. At that rate
   it needs roughly 250 iterations — which is also why `OLF_GPU_MAXITER=120` converted nothing.
   **OLF's own cap is 30 per NR solve, so OLF cannot have converged this contingency either.**

   And it does not. The harness prints both arms' statuses, and they are IDENTICAL:

   ```
   statuses: CPU={CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}
             GPU={CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}
   ```

   **OLF itself reports MAX_ITERATION_REACHED for 60 contingencies and FAILED for 5** — 65 not
   converged on either arm, matching the run's own "65 contingencies not CONVERGED on both arms". So
   for a large part of the 108, `MAX_ITERATION_REACHED` IS the correct answer, and the device can
   report it DIRECTLY instead of paying a CPU replay to arrive at the same thing.

   **Which means the mode-6 retry was working and I judged it by the wrong metric.** It moved
   `RUNTIME:status=-2` from **113 to 70** and `DEVICE:MAX_ITERATION_REACHED` from **4 to 47** — 43
   contingencies that stopped leaving the device. I called that a regression because COVERAGE stayed
   at 6550 / 6667. **Coverage cannot move for these by construction:** it counts precomputed device
   STATES, and a `MAX_ITERATION_REACHED` result carries no state. The metric that matters is how many
   contingencies leave the device for a CPU replay, and that improved by 43.

   The guard against claiming `MAX_ITERATION` for a scenario OLF actually converges is the harness's
   STATUS-COUNT EQUALITY, not coverage — and that check was never run on the mode-6 experiment,
   because it was a GPU-only run.

   **VALIDATED ON BOTH ARMS, AND NOW ON BY DEFAULT (2026-10-02).** Full busbar scope with
   `-Dolf.gpu.sa.robustRetry=true -Dolf.gpu.sa.robustRetryAcceptMaxIter=true`:

   ```
   GPU_ROBUST_RETRY attempted=113 recovered=39 statuses={-2=74, 1=39}
   statuses: CPU={CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}
             GPU={CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}
   violations: CPU=503923 GPU=503923 diff=0
   distributedActivePower worst 1.397828e-03 MW  (unchanged)
   CENSUS: RUNTIME:status=-2 = 74 (was 113), DEVICE:MAX_ITERATION_REACHED = 43 (was 4)
   ```

   | | before | with the retry |
   |---|---|---|
   | CPU replays (`status=-2`) | 113 | **74** |
   | `DEVICE:MAX_ITERATION_REACHED` | 4 | 43 |
   | CPU-vs-GPU status counts | match | **match** |
   | violations | diff 0 | **diff 0** |
   | unit gates | 200 / 0 / 1 | **200 / 0 / 1** |

   So **39 of the 108 stop leaving the device**, answering with the same status OLF gives, and both
   defaults are flipped ON. Remaining tail: 74.

   **COST MEASURED, and it is NEGATIVE — the pass PAYS FOR ITSELF (2026-10-02).** The worry was that
   switching mode evicts the single-entry context cache, so the pass costs a cuDSS analysis plus a
   solve over the rejected scenarios on every run. Both-arms full busbar scope, retry off against
   retry on:

   | | retry OFF | retry ON | |
   |---|---|---|---|
   | GPU COLD | 174751 ms (26.21 ms/ctg) | **150575 ms (22.59 ms/ctg)** | **-14%** |
   | GPU warm | 174073 ms (26.11 ms/ctg) | **147439 ms (22.11 ms/ctg)** | **-15%** |
   | vs CPU | x4.03 / x4.05 | **x4.93 / x5.04** | |

   Removing 39 CPU replays of HARD, non-converging contingencies saves far more than the extra device
   pass costs — a CPU replay of one of these runs OLF's full outer-loop machinery to a
   MAX_ITERATION_REACHED verdict. (Caveat: the two runs are not perfectly matched — their CPU arms
   differ by 5%, 704487 ms against 742567 ms — but the GPU gain is 14%, the opposite sign to a
   slowdown and larger than that noise.) So the default-on decision stands on both correctness and
   speed; `-Dolf.gpu.sa.robustRetry=false` still turns it off.

4. **Superseded lead (kept): the STOPPING CRITERION at the boundary.** OLF's 31
   scaled steps must span AT LEAST TWO NR solves (its own cap is 30 per solve), which means **OLF's
   first NR FINISHED** — while the device's first 30 steps, agreeing with OLF's to 7.1e-06, did NOT.
   If OLF's criterion is met at step ~30 and the device's is not, by a hair, then the device rejects
   exactly where OLF proceeds to its outer loop, and everything else measured (same equations, same
   targets, same path, same clamp, more iterations useless, outer loops inert on a stalled state)
   follows. The device iterates on `||F||2 < sqrt(convEpsPerEq^2 * nActive)` with
   `convEpsPerEq = 1e-4` under `UNIFORM_CRITERIA`; OLF applies its configured
   `NewtonRaphsonStoppingCriteria`. **Next: print the device's `||F||2` / `||F||inf` per iteration for
   slot 39 against its threshold, and OLF's stopping decision for the same contingency, and compare
   at the step-30 boundary.** This is the one place a hair's difference flips a reject into a
   continue, and it is not yet measured.

**Superseded (kept): the decisive measurement not yet made was OLF's own clamp factor.** Both
arms have the same equations, the same targets, comparable initial residuals at the same equations
(OLF 13.37 / 7.25 / 7.02 / 10.60 / 7.88 / 6.12 pu against the device's 11.02 / 9.79 / 7.50 / 6.70 /
6.70 / 6.13 at the matching coupler children) and the same clamp rule — yet OLF converges inside 30
iterations. So either OLF's factor is much closer to 1 (which would mean its STEP DIRECTION differs,
sending us back to Jacobian VALUES, the one thing still uncompared), or OLF's factor is also ~0.18
and its iterates still get there. **Instrument `MaxVoltageChangeStateVectorScaling` in the fork to
print its `stepSize` per iteration, scoped by `OlfTraceScope`, and compare.** That single number
splits the remaining hypotheses.

**Still open, and the honest state of [[#67]]:** the equation systems agree everywhere the dumps can
compare them; the targets, islanded sets, step clamp and iteration cap all
agree; the device nonetheless cannot finish the first Newton solve on 108 contingencies; and the one
difference found has not been shown to cause it. The 649 relocated buses remain the region a per-bus
comparison cannot reach, and the reconciliation above (OLF 334 active vs device 336 live + 9 gated)
is now fully explained by exactly those 2 rows — so that count is no longer an open discrepancy.

**Superseded (kept): the probe the dump named.** this contingency must absorb
`activePowerLoss = -1.243 pu` (124.3 MW of disconnected load), and the dump prints that value as
"the reference for the device's scheduledActivePowerLoss (`OLF_GPU_LOSS_TRACE` prints that one)". A
target that large being wrong would wreck the first solve, and TARGETS are part of the system. Run
`OLF_GPU_LOSS_TRACE=1` and compare. The still-unchecked region is the **649 relocated buses** (a
per-bus V-row comparison is invalid there, since a relocated child's `DISTR_Q` lives on its parent
coupler's `DUMMY_Q` row while its own V row hosts `ZERO_V`); the aggregate is OLF 334 active
`distr_q` against 345 device dq-carrying rows, 336 at mode 1 and 9 at mode 4, which is NOT yet
reconciled and is the next thing to nail down after the loss.

**TWO DIAGNOSTIC DEFECTS FIXED ON THE WAY — both had been silently inventing differences:**

1. `OLF_GPU_SA_EQ_DUMP` passed `new int[baseState.length]` as the row modes, so EVERY row printed
   `mode=0` (identity), including the coupler `DUMMY_Q` rows that carry the `dq` sharing terms. Any
   comparison against it reads a system difference wherever OLF has a power row. Now passes
   `baseActivation.rowMode()`; the dump shows modes 0/1/4.
2. My own parse of `OLF_GPU_EQ_DUMP_CTG` matched only the SCALAR equation names
   (`bus_target_v3 = v3`) and silently dropped every VECTORIZED one
   (`bus_target_q[0] = ac_q_array_closed_2(...)` — the element number is in BRACKETS). That hid all
   6886 `bus_target_p` and all 6886 `bus_target_q` equations, i.e. the entire P/Q balance side, so a
   first "0 mismatches" result had checked only the PV classification. The dump has BOTH shapes;
   match `([a-z_\u03c6]+?)(?:\[(\d+)\]|(\d+))\s*=`. Correct inventory for `BUS-1039_BBS`:
   `bus_target_p` 6876 active, `bus_target_q` 5561, `bus_target_v` 982, `distr_q` 334, `zero_v` 366,
   `zero_phi` 366, `bus_target_phi` 1.


## Completed 2026-07-21

| # | Task | Evidence |
|---|---|---|
| 38 | Batched solver outer-loop config per-call, not JVM-global static | `aaaaa14` |
| 37 | Pegase full-suite failure (`checked 0`) — root-caused to the config leak | `5177543` |
| 36 | Emit the CSR diagonal for the `BRANCH_ALPHA1` identity row (phase control INCREMENTAL now solves on GPU) | `e8cf120` |
| 35 | Re-time benches on an idle box (the 2026-07-20 row was contention) | `d24e38b` |
| 33 | Batch node-breaker contingencies under propagation | `f7b7226` |
| 24 | Batched operator strategies (scope complete; remainder re-scoped into #30 + upstream-blocked actions) | `856767c` |

## Not tracked, but outstanding

- **Split-bench re-run after the #38 refactor.** Interrupted before it ran. The refactor is
  behaviour-neutral in principle (same values reach the JNI call) and the full suite passes,
  but this bench is what would expose a mis-wired `dsRejectResidual` — a wrong value flips
  scenarios to CPU fallback and tanks the GPU time. Baseline to beat: GPU warm ~23.5 ms/ctg,
  ×1.42, from `7ac0627`. Run with `OLF_GPU_SA_GPU_CONV_EPS=1e-7` on an idle box.

### #41 — Outer-loop ordering & composition in the DEVICE batch path (investigated 2026-09-11)

**Problem.** The GPU DEVICE batch path runs host orchestrators in a sequence that differs from
OLF's `DefaultAcOuterLoopConfig` ordering, AND does not implement OLF's nested stabilization
model (each loop to stability; repeat whole sweep on change). Root cause is architectural: the
in-kernel driver phases (DS, HVDC, Monitoring, RL) complete before host orchestrators run, so
loops that OLF positions BEFORE RL (SecondaryVc order 4, VoltageMonitoring order 5) necessarily
run AFTER RL on the GPU.

**Current GPU host-orchestrator order** (`GpuBatchChunkSolver` lines 1137-1291):
```
[In-kernel: DS → HVDC freeze → HVDC limits → (HOST pause) → Monitoring → RL]
Host orchestrators (sequential, after main solve):
1. IncrementalTvc        ← OLF order 8
2. VoltageMonitoring     ← OLF order 5 (if not in-kernel)
3. SecondaryVc           ← OLF order 4
4. ContinuousPhase       ← OLF order 7
5. SimpleShuntVc         ← OLF order 10
6. SimpleTvc             ← OLF order 8
7. IncrementalFlowControl ← OLF orders 7+9
8. IncrementalShuntVc    ← OLF order 10
9. TransformerControl    ← OLF order 8
```

vs OLF CPU: DS(0) → HVDC(1,2) → SVC(4) → Monitoring(5) → RL(6) → Phase(7) → TVC(8) → TRPC(9) → Shunt(10).

**Three distinct issues:**

1. **Pre-RL loops running post-RL.** SecondaryVc and VoltageMonitoring are pre-RL on CPU but
   post-RL on GPU (because RL is in-kernel). VoltageMonitoring can run in-kernel (already
   implemented). SecondaryVc needs a transposed sensitivity solve and a host-side dense zone
   system — it cannot be a pure kernel. The `LOOP_HOST` pause/resume mechanism
   (`olf_gpu_nr_jni.cu:69-75, 4852-4853, 5396-5407`) was built for exactly this: a host-owned
   loop sitting in its correct OLF position between HVDC_LIMITS and MONITORING, with the driver
   pausing and resuming around it. Currently used only in base LF, not in batched SA.

2. **Post-RL orchestrators out of OLF's relative order.** IncrementalTvc runs before SecondaryVc
   and ContinuousPhase, but on CPU TVC (8) is after Phase (7). All three run after RL on both
   paths, so this is just a sequencing bug — reorder them:
   ```
   ContinuousPhase (7) → IncrementalTvc / SimpleTvc / TransformerControl (8) →
   IncrementalFlowControl/TRPC (9) → IncrementalShuntVc / SimpleShuntVc (10)
   ```

3. **No nested stabilization.** OLF's engine has a double loop: inner (each loop to stability),
   outer (repeat sequence on change). GPU orchestrators run once. This means a TVC tap change
   that destabilizes DS is caught only at the next orchestrator's re-solve (where DS runs
   in-kernel), not by restarting the sequence from DS. The JAVA hybrid mode
   (`GpuHybridOuterLoopBridge`) implements the correct nested model.

**Data flow findings** (who needs what — determines porting approach):

| Orchestrator | Sensitivity solve? | RL state R/W? | Discrete tables? |
|---|---|---|---|
| SecondaryVc | YES | YES | No |
| VoltageMonitoring | No | YES | No |
| IncrementalTvc | YES | No | Tap positions + pi params |
| ContinuousPhase | No | No | Tap positions + pi params |
| IncrementalFlowControl | YES | No | Tap positions + pi params |
| IncrementalShuntVc | YES | No | Section B values |
| SimpleShuntVc | No | No | Section B values |
| TransformerControl | No | No | Tap positions + pi params |
| SimpleTvc | No | No | Tap positions + pi params |

**In-kernel feasibility per loop** (ascending difficulty):

- **VoltageMonitoring**: Already in-kernel (LOOP_MONITORING). Done.
- **ContinuousPhase / TransformerControl / SimpleTvc**: No sensitivity. Need discrete
  tap tables on device (per-branch: list of (r1, x, g, b, a1, ksi, y) per tap position).
  Decision logic (deadband, closest-tap search, ratchet) is simple enough for CUDA.
  Medium difficulty — blocked on uploading the discrete tables.
- **IncrementalTvc / IncrementalShuntVc / IncrementalFlowControl**: Need
  `solveBatchSensitivities` which requires a separate solve step — a transposed solve
  against the factored J. Cannot be folded into the Newton iteration. Must remain as
  host orchestrators or use task #40's batched transposed solve to run in-driver.
- **SecondaryVc**: Needs sensitivity + host-side dense zone system (small matrix LU).
  Must remain host-orchestrated. The `LOOP_HOST` mechanism is the path: pause the driver,
  run SVC on host, resume. Not yet wired for batched SA.
- **SimpleShuntVc**: Rounds continuous B to section — simple, but needs section B tables
  on device. Low-medium difficulty.

**Actionable next steps (ordered):**

A. **Reorder host orchestrators** in `GpuBatchChunkSolver` to match OLF's relative post-RL
   ordering. Low risk, immediate fidelity improvement. Validate with the production SA bench.

B. **Wire `LOOP_HOST` for SecondaryVc in the batched SA.** The mechanism exists in the native
   code; the Java side needs to detect the pause, run the orchestrator, and resume. This puts
   SVC in its correct pre-RL position (between HVDC and Monitoring in the driver sequence).

C. **Upload discrete tap/section tables to device.** Prerequisite for any in-kernel phase or
   transformer control. Once available: ContinuousPhase, SimpleTvc, TransformerControl,
   SimpleShuntVc can become driver phases.

D. **Nested outer-loop sweep.** The extended driver's own nesting (`do { for each loop }
   while(changed)`) provides OLF's repeat-until-stable semantics — no separate wrapper needed.

E. **Base LF on GPU.** Currently the base LF uses `GpuNewtonRaphson.run()` with only DS+RL
   in-kernel. All other outer loops run as OLF Java loops on CPU. Solution: enable the
   `LOOP_HOST_*` slots, enter a `solveBatchContext → readBatchDriverPaused → run OLF loop
   → setBatchDriverResume` cycle. `GpuDriverOuterLoopEngine` suppresses ALL loops from the
   Java engine.

### Sensitivity-dependent loops — kernel port assessment (2026-09-12)

`solveBatchSensitivities` is a FORWARD solve (`J dx = e_i`), NOT transposed. This means
IncrementalTvc and IncrementalShuntVc are NOT blocked on task #40.

| Loop | Solve | Observable | Blocker | Verdict |
|------|-------|-----------|---------|---------|
| IncrementalTvc | Forward ✓ | variable column | Upload tap tables | **LANDED 2026-09-14/15** (`LOOP_TVC_KERNEL`; shared groups too since 2026-09-15 — `tvcGroupDecisionBatch`, OLF's adjustWithSeveralControllers per (scenario, group)) |
| IncrementalShuntVc | Forward ✓ | variable column | Upload section tables | **LANDED 2026-09-15** (`LOOP_SHUNT_KERNEL`, OLF adjustB round-robin per controlled-bus group in-kernel — several controller shunts and sub-controllers per group) |
| IncrementalFlowControl | Forward ✓ | J-row · dx on device | — | **LANDED 2026-09-15** (`LOOP_PHASE_KERNEL` order 7 / `LOOP_TRPC_KERNEL` order 9: the controlled branch's P/Q gradient from `updateClosedBranch`, I from OLF's reI/imI decomposition incl. the I = 0 one-sided rule, dotted with the unit-RHS solution at the branch's six variable columns) |
| SecondaryVc | Forward ✓ (**corrected 2026-09-15**: OLF's `j.solveTransposed` IS the Newton direction — NewtonRaphson.java:62 takes its step with the same call; `GpuBatchSecondaryVc` already solves it with `solveBatchSensitivities`) | branch Q + shunt Q gradients · dx, pilot dx (all on device since the flow-observed port) | none — the observables were kept on OLF's EquationTerms by CHOICE (device-target-writers risk), not by necessity | **Not blocked; deferred** (host-paused slot; SVC off by default, 0 occurrences on rte6515) |

To port IncrementalTvc to kernel: upload per-branch tap model tables (y,g1,g2,b1,b2,ksi,r1,a1
per tap position) + implement deadband/closest-tap/ratchet as a CUDA kernel + the forward
sensitivity solve is already available via the existing factorization.

To port IncrementalShuntVc: upload per-sub-controller section B/G tables + one-section-at-a-time
kernel with magnitude-descending ordering.

IncrementalFlowControl's observable (`calculateSensi`) is `Σ (∂Flow/∂x_j) · (dx_j/dControl)` —
a dot product of the term's Jacobian row entries with the sensitivity solution. BOTH pieces are
already on the device: `∂Flow/∂x_j` = the J entries the fill kernels scatter, `dx_j/dControl` =
the sensitivity solution from `solveBatchSensitivities`. So the observable CAN be computed
on device as a gather-and-dot from J at the flow term's variable columns. Needs: (a) a map from
each flow term to its J column positions (the scatter table already has this), (b) a kernel that
reads J[term_cols] · dx[term_cols] → flow sensitivity. Medium effort, NOT blocked.

### SecondaryVoltageControl — the in-kernel phase (landed 2026-09-15, DEFAULT)

The earlier "blocked on task #40" verdict was WRONG. OLF's `SecondaryVoltageControlOuterLoop` builds
its sensitivities with `j.solveTransposed(rhs)` — the very call `NewtonRaphson.run` uses for the
Newton step (OLF stores the Jacobian variable-major, so "transposed" is the forward system in our
layout). Task #40 is needed by the GNN adjoint (reverse mode) only.

`LOOP_SVC_KERNEL` (OLF order 4) is the whole loop on the device. Nothing is hand-derived: a
controller bus's reactive injection is its CLOSED-BRANCH side flows, whose gradient comes from the
codegen'd `updateClosedBranch` exactly as `flcGatherSensiBatch` takes it, plus its shunts' q = -b*v^2 —
and that one OLF exposes as three static lines (`ShuntCompensatorReactiveFlowEquationTerm`:
dq/dv = -2bv, dq/db = -v^2). The pilot's calculated V is the BUS_V variable itself, so its
sensitivity is dx at the pilot row. The pieces, in `check()`'s order:

- `svcTriggerBatch` — `tryToReEnableHelpfulControllerBuses` first (a zone with SOME but not all
  controllers at a reactive limit gives the limited ones back their control, restoring the PV form
  from `dRlSavedMode`), then OLF's zone filter (a zone with NO enabled controller is not looked at
  at all), then the pilot-deadband / K-spread trigger with K from the pass's own `qBuf`.
- one unit-RHS solve per controlled bus on the driver's own factors, then `svcGatherBatch` for
  `calculateSensiK` and `calculateSensiVpp`.
- `svcAssembleBatch` + **cuBLAS batched dense LU** (`cublasDgetrfBatched` / `cublasDgetrsBatched`) +
  `svcCommitBatch` — A (per-zone block), K0, `b = A.Jk^T`, the zone's LAST controlled bus's row
  replaced by the pilot sensitivities and its rhs by the pilot dv. NO hand-written linear algebra: the
  systems are assembled column-major into device memory and handed to the library, one matrix per
  scenario. A controller at a reactive limit is padded out with an identity row and a zero rhs, which
  leaves the enabled block exactly OLF's system and gives the disabled one dv = 0. A singular system
  (cuBLAS info != 0) or an implausible target rejects the scenario to the CPU, where OLF's own
  MatrixException / FAILED verdict would land it.
- `svcApplyBatch` — the moved target on an unpinned controlled bus's V identity, and the switch-back
  comparison target for a pinned one (`dSvcRlVt`, which `rlTargetOf` consults).

Two layers remain: the in-kernel phase, and the driver PAUSE at the same slot where the kernel
refuses the tables (an observable on an unpacked element, or a zone system whose per-scenario
workspace exceeds `SVC_MAT_BUDGET_BYTES`).

**Sizing, for real networks.** OLF solves ONE system over every enabled controller of every zone —
A is block diagonal by zone but Jk is full, since a controller in one zone moves a controlled bus in
another through the network — so N is the TOTAL controller count, not a per-zone one. Production
cases are quoted at 30 to 60 zones of 4 to 20 controllers, i.e. N = 120 to 1200. Two consequences
the original 16-controller cap hid: the batched LU holds an N*N matrix per scenario, so the workspace
is scenarios * N^2 * 8 bytes (58 MB at N=120 with 500-scenario chunks, 5.8 GB at N=1200 — hence the
budget, and chunk sizing will have to follow N); and Jk needs one unit-RHS solve per controlled bus,
so N solves per scenario per check, which is inherent (the matrix is genuinely N functions by N
parameters, so neither forward nor adjoint avoids it) and is what will dominate at the top of that
range. NOT YET VALIDATED above N = 4: rte6515 carries two zones of two.

**The bug that cost the most, and the wrong turn it caused.** The phase first landed 1.04e-10 kV off
on the SecondaryVc unit tests. `OLF_GPU_SVC_INERT=1` (slot in the sweep, body empty) reproduced the
shift exactly, and I read that as "an extra sweep slot perturbs the trajectory" and wrote it up as
such. It was not. The inert probe isolates everything `nSvc > 0` enables, and that includes the new
`dSvcRlVt` buffer — which `initRlLimitTypeBatch` consults through `rlTargetOf` and which was
initialised AFTER that seeding. The reactive-limit switch-back was comparing against uninitialised
memory: ieee14's z1 went 4 pins -> 2 (un-pinning two controllers) where it should go 4 -> 6. Filling
it with NaN at upload AND before the seeding fixes it, and all three SecondaryVc tests pass at 1e-11.
LESSON: an inert-slot probe proves "something enabled by this table", NOT "the slot itself"; enumerate
what else that flag switches on before naming a cause.

**Every OLF outer loop is now a device phase.** rte6515, 9042 branch contingencies, both zones
firing (`OLF_GPU_SVC=1 OLF_GPU_SVC_PILOT_SHIFT=1.0`), all 4 differing violations identical. Cold and
warm are two identical SA runs in the same JVM, so COLD carries the CUDA context, the cuDSS analysis
and JIT; WARM is the fair comparator when asking what a LOOP costs:

| chunk path for SVC | cold | warm |
|---|---|---|
| host orchestrator (old) | 208 s | 172 s |
| driver pause | 150 s | 142 s |
| in-kernel | 119 s | 91 s |
| in-kernel + base load flow on the driver | 83 s | 85 s |
| (same case with SVC switched off) | 80 s | 83 s |

So the port is worth 172 s -> 91 s of steady-state time, and SecondaryVoltageControl now costs about
3% over the SVC-off baseline (85 vs 83 s) where the host orchestrator cost more than double (172 vs
83 s). **Do NOT read the 119 -> 83 cold pair as the driver flag's doing**: that run's cold carried a
28.7 s first-run penalty (119 cold vs 91 warm) which the other run did not (83 cold vs 85 warm), and
on warm terms the flag is worth about 6%. Calibration for any single-run comparison here: the CPU arm
does identical work in every one of these runs and ranged 96 s to 109 s, so differences under ~10%
are not attributable.

### #42 — Base load flow on the FULL device driver + in-kernel incremental TVC (landed 2026-09-14)

**Outcome.** With `OLF_GPU_DRIVER_OUTER_LOOPS=1` the SA's base load flow is a batch-of-one on the SAME
driver as the contingency chunks: DistributedSlack, HVDC freeze/limits, MonitoringVoltage,
ReactiveLimits (and incremental TVC in-kernel when every controlled bus has a single controller) —
`GpuDriverOuterLoopEngine` keeps NOTHING on the Java engine (`GPU_DRIVER_ENGINE ... keptOnHost=[]`).
rte6515 / RTE params: base CONVERGED under OLF's own criterion (inf-norm 9.4e-6), 9040/9042 on device.

**What actually blocked it** (the "write-back" was only half of it): the base context received
`createBatchContext` + row criteria and nothing else, while the chunk solver also installs the RL
curve channel, the SVC quadratic limits, the remote-member RL kinds (`setBatchRlRemote`) and the
per-generator DS participants. Without the remote kinds `initRlLimitTypeBatch` read every remote
member's DISTR_Q share row (base mode 1) as a base PQ pin — limit type seeded, never re-checked —
and three rte6515 remote controllers came out pinned where OLF's loop never touched them
(BUS_TARGET_Q residuals of 1.46 pu). `GpuBatchOuterLoopTables.install` is now the ONE installer for
both paths; `GpuBatchHvdcAcEmulation.installDriverLoops` / `GpuBatchVoltageMonitoring.installDriverLoop`
likewise. A batch of one drives exactly what a batch of five hundred drives.

**Write-back** (`GpuNewtonRaphson.writeBackDriverMutations`, order matters): HVDC statuses through
`AcEmulationControl.switchTo*`; monitoring activations (VC on, SVC → VOLTAGE, automaton target);
DistributedSlack per GENERATOR from the water-fill's own `genCum` (`readBatchDsGenCum`), the slack
bus's generators splitting the retained share (`readBatchDsRetain`) by factor; ReactiveLimits pins
via `freezeGenerationTargetQAndDisableGeneratorVoltageControl` at the DEVICE's pinned target (the live
SVC limit — the packed base-V aggregate was 1.3e-4 pu off on VL-6226_0); taps (`readBatchTvcTaps`).
The extraction uses the run's own loops and DS params (`DistributedSlackOuterLoop.
getActivePowerDistribution()` — OLF getter added), never `GpuDsParams.DEFAULT`.

**Measured 2026-09-14 (rte6515, 9042 branch contingencies, `GpuVsCpuRte6515SaTest`):** driver ON
80.8 s cold (×1.20 vs CPU 8 threads 97 s), driver OFF 82.1 s — the base on the driver costs nothing.
Both arms' base violations agree to 1e-11 (one value differs in the 10th decimal). The 2 differing
post-contingency lines are the test's own documented residual ("2 out of 583k", identical with the
driver off): both sit exactly at the SA's 0.1 % increased-violation proportional threshold
(post/base = 1.00100 on LINE-2533-1468 under LINE-1550-228 and LINE-2219-439 under LINE-5466-2443).

**Three base-on-driver corners found and fixed the same day** (all invisible from the chunk path,
which extracts AFTER a converged base): (1) `rlV2Base` is NaN on a never-solved network and
`0 × NaN` poisoned every RL entry's limit — the quadratic SVC delta is now applied to SVC entries
only (`svcLim`); (2) a SINGLETON remote group has members but no DISTR_Q row, and
`setBatchRlRemote` keyed `hasDqQrGroups` on the DISTR_Q rows, so the pilot's control-loss flip
never armed — keyed on the member list now; (3) a device REJECTION (status −2) on the base is a CPU fallback like on the chunks — and since
2026-09-15 the forced-PV case no longer rejects at all: OLF's "strongest stays PV" election runs
in-kernel (`rlForcedPvElectBatch`, keys via `setBatchRlElection`), so the forced-PV network solves
on the device in the chunks AND as the base; `GpuDriverOuterLoopEngine`
now wraps every device-owned loop in `DeviceDrivenLoop`: STABLE while the solver ran on the device,
OLF's real loop (initialized late) the moment a solve fell back to the CPU — dropping the loops left
the fallback base without DS/RL (1.3e-4 kV on the zero-impedance member network).

**2026-09-15 — host-paused loops on the base, second pass.** A host loop that changes the equation
LAYOUT, the row modes or the element values (a tap/section move, a control it enables or disables:
every TVC/shunt/phase loop does) voids the paused sequence — the resume channel carries targets
only. The driver now RESTARTS: the driver's own decisions are written back to the network first,
the pause is cancelled, the extraction re-runs, the RL state is carried by bus, OLF's loop contexts
persist across the restart (`hostLoopPairs`), and the restarted sweep resumes AT the loop that went
UNSTABLE (`setBatchDriverStartLoop`) before restarting from the first loop — OLF's `runOuterLoop`
order. Two other fixes on the way: the driver owns loops only when `GpuDriverOuterLoopEngine` created
the solver (`AcloadFlowEngine.createSolver` hook; a plain `LoadFlow.run` keeps OLF's engine and the
legacy pure-Newton mode, or DS/RL ran twice), and `DeviceDrivenLoop` defers to OLF's loop whenever
the engine's current solver is not the GPU driver (CPU-routed contingencies had lost DS/RL). The
per-generator DS accumulator is zeroed per solve (a restarted sweep that never reaches DS handed
garbage to the write-back → NaN targetP).

**Same day, fourth pass — 39 → 0: `GpuBatchedAcSecurityAnalysisTest` 136/136 under the flag.** The
"DS re-anchoring class" was exactly that, and it was the base integration, not the loop: OLF's
`ActivePowerDistribution.run` resets every participant to its INITIAL target P before each
distribution (`resetToInitialState`, adding back what that undid as `previousMismatch`) and
re-derives everything — factors, sign rule, limits — from there. The device kernel has done that
since 2026-09-08 (`absorbedNow`), but the base path fed it the batched-SA extraction, whose contract
anchors at the row targets the solve starts from (right for the SA: converged base, per-scenario
channels relative to it), and every restart re-anchored at the written-back targets. Now
`GpuDsParams.anchorOnInitialTargetP` (base path only): the pack anchors the row targets, the
per-generator limits, the sign rule and the P-proportional factors at `initialTargetP`
(`anchorOffset` = Σ(tp − tp₀) over the participants — `absorbedNow` then IS OLF's
`previousMismatch`); the write-back sets `tp = tp₀ + genCum` and, for the slack bus's own
generators, `tp₀ + retain` with the retain captured at the last check that FIRED
(`dDsRetainApplied`, NaN when none did — a STABLE check's ρ·mismatch is not a distribution);
slack-only bases write back `readBatchSlackAbsorbed`. Two more: a restart starts from the network's
last state through `PreviousValueVoltageInitializer` exactly as `runOuterLoop` does (the old state
vector was not even the right length after a control was disabled); and the driver's decisions so
far (P, RL pins, HVDC statuses, activations) are handed to the network BEFORE each host-loop check
— SecondaryVoltageControl skips a zone's RL-pinned controllers, and a check that still saw them
enabled went UNSTABLE where OLF's said STABLE (one extra adjust round, 2.5e-8 kV). The SA contract
is untouched: with static factors, anchoring at the converged base is linearly equivalent
(caps are absolute, the sign never flips); PROPORTIONAL_TO_GENERATION_P (|tp₀| vs |tp|) and
REMAINING_MARGIN margins are the only balance types where it is not — latent for the SA, noted.

**Same day, third pass — 44 → 39, no ENGAGE and nothing above 1e-3 kV left.** Five more base-only
defects: (1) a converter bus without an active BUS_TARGET_P row (the slack) received the HVDC
freeze/saturation delta on its PHI identity row (base MAX_ITER, 3 ENGAGE) — packed as ~row now, the
angle still reads there, no target is ever written; (2) after a host loop's STABLE resume the RL
check was silently skipped (the resumed call's status vector said "not converged" → rlActive 0):
the pause saves and the resume restores the per-scenario statuses (SecondaryVoltageControl
7.3e-2 → 2.5e-8); (3) an SVC's limits are Bmin/Bmax·V² and a never-solved network has NaN V, so
the packed limits were NaN and the slope SVC never pinned (0.33 kV): the pack evaluates at a flat
1 pu and restores the NaN; (4) the slack's retained DS share was written back at every check,
STABLE ones included (ρ·sub-tolerance mismatch, 3.4 kW), now only when a distribution happened;
(5) the split of that share used LfGenerator.getParticipationFactor(), which OLF fills only inside
its own distribution step (0 here) — the run's GpuDsParams factors, as the pack uses (l121 severe
0.71 → 4.2e-4). Remaining 39: 17 at ≤1e-9 kV, 7 at ≤1e-6 kV, 15 between 2e-6 and 4e-4 kV
(AFTER_GENERATOR/shared TVC, TRPC, node-breaker, l121) — the DS re-anchoring class: OLF resets
every participant to its INITIAL target and re-derives the whole distribution from the anchor at
each check, the restarted driver distributes from the written-back base; and the driver re-enters
the sweep once more than OLF after a STABLE host check (checks only, no solve).

Result (second pass): 61 → 44 failing under the flag. By size: 13 at ≤1e-9 kV and 7 at ≤1e-6 kV — the base's
converged state differs from OLF's at solver-tolerance level (a different linear solver), and
AFTER_GENERATOR transformer control FREEZES state-dependent quantities (generator Q under the
controlled level, the tap rounded from the solved ratio), so the frozen targets differ by ~1e-5 pu;
inherent to a base on another solver, not a sequence bug (sequences now match call for call).
Real remaining: 3 HVDC ENGAGE (freeze/saturation/DS carry), 10 medium (shunt VC 9.6e-4, shared
TVC 4e-4, ...), 11 large (SecondaryVoltageControl 7.3e-2 ×4, slope-SVC pinned 0.33, l121 severe
0.71, TRPC 1.7e-3, shared incremental TVC 1.2e-3).

**Earlier inventory (2026-09-14):** `GpuBatchedAcSecurityAnalysisTest` with
`OLF_GPU_DRIVER_OUTER_LOOPS=1` = 61/136 failing (default path: all green). By shape: 16 "batch must
ENGAGE" (the base's write-back leaves the network in a state the plan or the JAVA guards reject —
continuous PhaseControl after the host-paused loop, shunt/TVC/SVC host loops), 12 BUS_2 voltage
mismatches (transformer/shunt voltage-control networks — the host-paused TVC/shunt slots on the
base), 3 strategy-pair precompute counts, 3 HVDC (freeze/saturation, DS carry), REMAINING_MARGIN DS.
Every remote gen-VC test (24/26) passes; the rte6515 production case passes. Next: take the failing
shapes one by one with `OLF_GPU_BATCH_TRACE=1` (HOST_LOOP lines show the pause/resume sequence),
starting with the host-paused loops (order 7–10) on the base.

### #45 — Flow-observed incremental loops in-kernel: RPC + incremental phase (landed 2026-09-15)

`LOOP_PHASE_KERNEL` (OLF order 7, phase CONTROLLER + LIMITER) and `LOOP_TRPC_KERNEL` (order 9, RPC)
share one controller table and one step (`flcStep(kindMask)`). Per (scenario, controller):
`flcCheckBatch` evaluates the CONTROLLED branch's flow at the current state — P/Q through the
codegen'd `updateClosedBranch` (the very functions the fill scatters, with the branch's tap override
applied), I through OLF's `ClosedBranchSide{1,2}CurrentMagnitudeEquationTerm` decomposition
(reI/imI and their eight partials) — and OLF's deadband test (LIMITER one-sided). One unit-RHS
solve per needed controller at its rho/a1 identity row on the driver's own factors;
`flcGatherSensiBatch` dots the flow gradient with dx at the branch's six variable columns — this IS
`EquationTerm.calculateSensi`, including its dr1-less current sensitivity and its `hypot(dre·dx,
dim·dx)` one-sided rule at I = 0. `flcDecisionBatch`: RPC → closest r1 within the run's
maxTapShift (no gate); CONTROLLER → `|sensi·toRadians(1)| > SENSI_EPS`, closest a1 unbounded;
LIMITER → `FirstTapPositionAboveFinder` (walk tap by tap while the remaining shift reaches past the
next a1) — ratchet as everywhere. `flcApplyBatch` pins the identity row; the r1 override rebuild is
shared with the TVC phase (`tvcRebuildOverrides` now merges both tables' moved taps). Base path:
`getMaxTapShift()` (OLF getter) for RPC, `readBatchFlowTaps` → `PiModel.setTapPosition`. The host
orchestrator (`GpuBatchIncrementalFlowControl.orchestrate`) stays for a controlled branch that is
not a closed branch. With this every sensitivity-dependent incremental loop runs in-kernel; the
chunk path's host orchestrators left: SecondaryVoltageControl (transposed solve, task #40); the
non-incremental TVC / shunt VC flows followed the same day (#47).

### #53 — fixTransformerVoltageControls is a per-SCENARIO check (2026-09-16)

The last of the four gaps in the ported phases: the freeze approximation. The generator freeze itself
was already OLF's (`GeneratorVoltageControlManager` frozen at `q.eval()`, restored at COMPLETE); what
was approximate was the connectivity check that runs beside it.

`LfNetwork.fixTransformerVoltageControls` disables a just-enabled transformer voltage control when the
not-controlled side's connected component holds no enabled generator voltage control. It computes that
component on the POST-CONTINGENCY graph, cutting the contingency's branches AND the ENABLED controller
branches. The device built ONE table, at base topology, cutting EVERY controller branch, and per
scenario asked only which of those buses was still PV. Two errors in opposite directions: a contingency
that split the component was invisible, and cutting non-enabled controller branches made the component
smaller than OLF's.

The dynamic half stays in the kernel — which buses are PV depends on the freeze and on the reactive
limits, and only the kernel knows that. The topological half moves to the host, per scenario. The
obstacle is that OLF's cut set is not host-knowable: the contingency is, but the ENABLED controllers
are decided inside the solve. So the host brackets it with two masks over a fixed universe (the
{@code BUS_V} row of every generator-voltage-controlled bus, one bit each):

| mask | cut set | relation to OLF's component |
| --- | --- | --- |
| min | contingency + EVERY controller | subset |
| max | contingency only | superset |

A PV bus in min means OLF sees one too, so the control stays enabled. No PV bus in max means OLF sees
none either, so it is disabled. Between the two the answer genuinely depends on which controllers the
scenario enabled, and the scenario goes to the CPU rather than to a guess. The freeze-driven cases —
where the check fires because everything under the level is frozen, which is what the mode is for —
land outside the bracket and still batch; it is the connectivity-driven ones that are ambiguous.

Cost: two connectivity sessions per scenario, each O(PV buses + members) — the component NUMBER of
every PV bus is read once and members look up their own component's mask, instead of walking a
component per member. Masks upload per chunk, not per run.

`GpuBatchTvcFixMaskTest` pins the builder directly rather than through a security analysis, where the
same scenario can be routed to the CPU for unrelated reasons and hide the difference: BUS_2 reaches the
only generator over LINE_12 alone, a 33 kV loop through a non-regulating transformer keeps the pocket
energized when that line opens, and the base mask and the scenario's differ exactly as designed.

### #52 — Remote LIMITER phase control, and the tap table without reflection (2026-09-16)

Two of the four gaps left in the ported phases. Both turned out to be self-imposed.

**Remote LIMITER phase control.** A CONTINUOUS phase control in LIMITER mode whose regulation
terminal sits on another branch routed the whole run to the CPU, in DEVICE and JAVA alike, on the
reading that "OLF's own loop is local-only". It is local BY CONSTRUCTION, not by restriction:
`PhaseControlOuterLoop.changeTapPositions` caps `controllerBranch.getI1()`/`getI2()` against the
target and `isSensitivityCurrentPerA1Positive` differentiates those same terms, so the phase
control's controlled branch never enters the decision at all. LIMITER mode is also never enabled as
an equation control — `initialize` enables CONTROLLER mode only — so there is no row-mode or
`cbPcSide` question either. The device already built its I term from the controller branch, so the
gate was refusing a computation it performs identically. Lifted in all three places; both modes now
batch local and remote. New tests build PS1 in CURRENT_LIMITER mode regulating the parallel line L2,
DEVICE and JAVA, both at 1e-12 with the batch engaged.

**Transformer control without a tap table.** `tapModels` read `PiModelArray.models` by REFLECTION and
returned an empty table on any failure, which meant a JVM configuration could silently turn into
unrounded continuous ratios and into whole-run CPU fallbacks (`generatorFreezeSupported`) that no
gate would explain. `PiModelArray.getModels()` already existed, package-private; it is public on the
patched OLF now (`olf-local-patches`), and the reflection is gone. An empty table means one thing
after this: the branch carries a continuous `SimplePiModel` and has no taps, which a voltage or phase
controller never does.

### #51 — Gaps inside the ported phases: robust mode, forced-PV, distributed slack (2026-09-16)

Three of the gaps left inside the phases that are already on the device. All three were
verified against the current code first — two of the three turned out not to be what the
earlier summary said they were.

**(a) Reactive limits: `voltageRemoteControlRobustMode`.** OLF's default is TRUE, so these
criteria were active in every default run and the RL kernel did not implement any of them —
a silent divergence, not a fallback. `ReactiveLimitsOuterLoop.checkControllerBus` applies two
things to a bus whose generator voltage control is REMOTE (`controlledBus != bus`):

  1. a bus that just switched PV -> PQ on a reactive limit, and whose OWN voltage is outside
     `[minRealisticVoltage * 1.02, maxRealisticVoltage / 1.02]`, is reseeded at 1 pu;
  2. a bus whose Q is still INSIDE its limits but whose own voltage is already outside that
     band stops maintaining the remote voltage: it is frozen at its INITIAL generation target
     Q (the sum of its generators' `targetQ`) with a `MIN_REALISTIC_V` / `MAX_REALISTIC_V`
     limit type, and is likewise reseeded at 1 pu.

Ported into the RL phase. The freeze is the existing remote pin (row mode 4) with a different
target, so only three things are new: a per-entry `rlInitTargetQ` table, limit types 3 and 4
alongside 1 (MIN_Q) and 2 (MAX_Q), and the 1 pu reseed. Two details matter. The switch-back
reads 3 like a min limit and 4 like a max limit (`LfBus.QLimitType.isMinLimit()`), but the
"Q limit moved while pinned" re-freeze is guarded to MIN_Q/MAX_Q in OLF and so skips 3/4. And
the reseed is a WRITE to the state vector the check kernel is reading, so it runs as its own
pass (`rlRobustVResetBatch`) against the pre-pass state rather than racing inside the check.

The packed per-(controller, scenario) RL state widened from two limit-type bits to three
(`(limitType & 7) | (switchCount << 3)`), on both sides of the JNI.

The other half of that gap was the realistic-voltage check. `AcloadFlowEngine.isStateUnrealistic`
runs after every solver run once the last `canFixUnrealisticState` loop has run, and an unrealistic
state ends the outer-loop cycle with `UNREALISTIC_STATE`: the run FAILS and reports no violations.
The device only ever checked the FINAL restored state, so a scenario that passed through an
unrealistic state and recovered came back CONVERGED where OLF reports FAILED. Which solve the check
belongs to depends on the configured loop order and on robust mode, so rather than reproduce that
ordering, the device flags a scenario that is unrealistic inside the sweep and routes it to the CPU
replay, which decides the status itself.

The ordering could NOT be skipped, and measuring said so. Checking after every solve cost four
rte6515 contingencies (9040 -> 9036 batched), and naming them showed every hit landing at the FIRST
solve, before any outer loop had run — exactly where OLF defers. Under robust mode
`AcloadFlowEngine` keeps `checkUnrealisticStates` false up to and including the last loop that
declares `canFixUnrealisticState()` (ReactiveLimits, or the discrete TransformerVoltageControl),
turns it on for the rest of the sweep, and resets it at the top of the next one — the whole point
being to let those loops repair the state instead of failing the run on it. The device now takes
that loop's identity as a hint from the Java side (−1 none, 0 ReactiveLimits, 1 the discrete
TransformerVoltageControl), finds its slot in its own loop order and checks only past it. rte6515 is
back to 9040 of 9042 with no sweep rejections at all, same two differing lines.
`OLF_GPU_UNREALISTIC_TRACE` names the slot, the pass and the loop that rejected, which is what made
the four visible. The temporary switch that turned the intermediate check off is gone with them: the
deferred check is what OLF does, so a switch selecting the other behaviour would only preserve a known
wrong answer — the same reason `olf.gpu.sa.hvdcLoopsInKernel` and `olf.gpu.sa.batchStrandedDs` were
deleted rather than defaulted.

**(b) Forced-PV: the per-scenario rejection detector.** The stated gap was that the in-kernel
election had landed but a silent per-scenario rejection to the CPU remained as a safety net.
The first reading — that the detector was dead code because the election keys were always
installed — was wrong, and the contract check that replaced it failed five tests. The keys were
installed by the driver's table install, and the RL phase runs on ANY context carrying RL
entries, including a full load flow where OLF (not the device) owns the outer loops. Those
contexts had no keys, so the detector was live there and every such scenario went to the CPU.

Fixed at the root: the keys are now installed where the context is CREATED (all four creators,
including the one-shot `solveBatch` entry, which has no network behind it and so gets neutral
keys that reduce the rule to OLF's last tie-break, lowest id stays PV). The detector is deleted
and `solveBatchContext` refuses a context that reaches it with RL entries and no keys.

**(c) Distributed slack.** Re-verified against the current code. `dsRejectResidual` is ON by
default (OLF's `slackDistributionFailureBehavior` default is FAIL, not LEAVE_ON_SLACK_BUS), and
it flags a saturated distribution on the ENTRY band (`fsGlobal * slackBusPMaxMismatch`) where OLF
decided the distribution failed on `ActivePowerDistribution.P_RESIDUE_EPS` (1e-5 pu, 1 kW). A
scenario that saturated with, say, 0.3 MW still unplaced therefore came back CONVERGED from the
device where the CPU reports FAILED. That was the one live DIVERGENCE among the slack gaps, and
it now flags on P_RESIDUE_EPS. The flag only routes the scenario to the CPU fallback, so the
tighter test can cost batching share and can never make the device answer where OLF refuses to;
on rte6515 it costs nothing, the batched share stays at 9040 of 9042.

The rest are fallbacks rather than divergences, with what each one needs:

- **`slackDistributionFailureBehavior` on-device.** FAIL (the default) could be answered
  in-kernel — a FAILED post-contingency result reports no violations — instead of replaying the
  scenario on the CPU. What stops it being a pure win is `distributedActivePower`: OLF reverts
  its reporting on a FAILED distribution, and the batched result carries that field.
  DISTRIBUTE_ON_REFERENCE_GENERATOR needs the reference generator's row; THROW must reach the
  host for its message.
- **Strategy-pair legs take the aggregate water-fill.** A re-anchored scenario (`dsAnchorShift`
  non-zero) skips the per-GENERATOR water-fill, because the anchor shift is a BUS-level
  displacement and nothing in the pack says how to split it across a bus's generators. Closing
  it means capturing the parent's per-generator cumulative deltas (`genCum`, already per
  (scenario, generator) on the device) and feeding them back as a per-generator anchor shift.
- **`loadPowerFactorConstant` + load-based balance, whole-run CPU.** `ensurePowerFactorConstant`
  moves each load's targetQ with its targetP, per round. The DS pack is per BUS with a
  per-generator sub-list and has no per-LOAD granularity, and the device never writes a Q target
  during distribution. Needs a per-load pack (Q row, participation factor, initial P and Q, the
  `ensurePowerFactorConstantByLoad` flag) plus a Q-target write in the water-fill.
- **`useLoadModel` + load-based balance, whole-run CPU.** Rescaling a load's targetP rescales its
  voltage-dependent coefficients, which are static on the device.
- **Custom `ContingencyActivePowerLossDistribution` SPI, whole-run CPU.** Only the "Default" impl
  is replayed host-side; an arbitrary SPI cannot be.

### #50 — PLAN: fold the JAVA hybrid into the driver's host-slot mechanism (reviewed 2026-09-16)

Not started. A review, not a design commitment — the measurement in stage 0 decides whether it is worth
doing at all.

**Why now.** The hybrid (`GpuHybridBatchOuterLoopEngine`, 1484 lines, plus `GpuHybridOuterLoopBridge`
270 and `HybridScenarioState` 85) exists because OLF's own Java loops make decisions no port can get
subtly wrong. It pays for that with a per-scenario, per-round save/restore of the compact network
state, which measured **54% snapshots / 44% Java loops / 2% device**. It is also an all-or-nothing
MODE: choosing JAVA replays EVERY loop per scenario in Java, even the ones the device now owns.

Since 2026-09-15/16 that trade is obsolete. Every loop in OLF's order is a device phase except two,
and the driver already knows how to stop at a named loop's OLF position, hand control to Java, and
resume with its sweep state intact (`LOOP_HOST_*`, `readBatchDriverPaused` / `setBatchDriverResume`,
flat per-scenario `setBatchTargetDeltas`, the RL v-target and un-pin channels). SecondaryVoltageControl
used exactly that before its kernel phase landed, at chunk scale, and it held parity.

OLF's `DefaultAcOuterLoopConfig` order, with what owns each slot today:

| order | loop | owner |
|---|---|---|
| 0 | DistributedSlack | kernel |
| 1 | FreezingHvdcACEmulation | kernel |
| 2 | AcHvdcAcEmulationLimits | kernel |
| **3** | **AreaInterchangeControl** | **nobody — whole-run CPU** |
| 4 | SecondaryVoltageControl | kernel (pause as fallback) |
| 5 | MonitoringVoltage | kernel |
| 6 | ReactiveLimits | kernel |
| 7 | PhaseControl / Incremental | kernel |
| 8 | TransformerVoltageControl, all 3 flavors | kernel |
| 9 | TransformerReactivePowerControl | kernel |
| 10 | ShuntVoltageControl, both modes | kernel |
| **11** | **AutomationSystem** | **nobody — whole-run CPU** |

So the hybrid's remaining reason to exist is two loops, and the proposal is to serve them with a host
SLOT at their own position instead of a whole parallel engine.

**What it would buy.** The per-scenario snapshot/restore disappears for every loop the device owns,
which is where most of that 54% goes. The duplicated outer-loop sequencer goes too: the bridge
re-implements OLF's nested `runOuterLoop` so it can drive one scenario at a time, while the device
sweep already sequences in OLF's order with OLF's restart-on-UNSTABLE semantics — one sequencer
instead of two. And the two coverage gates (`JAVA_SUPPORTED_LOOPS` versus the DEVICE filter) collapse
into one question: does the kernel own this loop, or does it have a host slot.

**Stages.**

0. **Measure first.** Re-run the hybrid's cost split now that every loop is in-kernel, and confirm the
   claim above. If a real case never enables AreaInterchangeControl or AutomationSystem, the honest
   answer is to delete the hybrid outright and route those two loops to the CPU, which is cheaper than
   any of the work below.
1. **Generalise the slot mechanism.** `hostLoopMask` is five bits keyed to `HOST_LOOP_SVC..SHUNT`.
   Make it a registry of (OLF order, loop name, kind) so a new host-owned loop is a table entry rather
   than a new constant in three files. Add `LOOP_HOST_AIC` at 3 and `LOOP_HOST_AUTOMATION` at 11.
2. **AreaInterchangeControl first.** It adjusts generator targetP to meet area interchange targets, so
   like DistributedSlack and SecondaryVoltageControl its mutations should be TARGET-only, which the
   existing `setBatchTargetDeltas` channel already carries. That makes it a near-copy of the SVC pause,
   including the budget rule (a host-UNSTABLE verdict must consume `prodIters`, or the pause cycle can
   spin — see #49). CONFIRM the target-only claim against OLF before building; its interaction with
   DistributedSlack at order 0 is the part to read carefully.
3. **AutomationSystem is the hard one and should be scoped separately.** It switches topology per
   round, so it changes the equation LAYOUT, not just targets. The base path handles that by
   re-extracting and restarting the sweep, which does not generalise to a 500-scenario chunk where one
   scenario changed layout. Two candidate answers: reject the scenarios it actually fires on to the
   CPU (cheap and honest if it fires rarely), or express the switching through the per-scenario
   element-disable and zero-impedance on/off channels that already exist. Decide with a measurement of
   how often it fires, not a priori.
4. **Retire the hybrid.** Once both loops have slots, delete `GpuHybridBatchOuterLoopEngine`, the
   bridge, `HybridScenarioState`, the `OuterLoopMode` enum and `javaUnsupportedOuterLoops`. That is
   ~1,800 lines and one of the two coverage gates.

**Risks to weigh before starting.**

- A paused Java loop still reads the `LfNetwork`, not just the state vector, so running it for one
  scenario still needs that scenario's network state. The slot NARROWS this to the loops and scenarios
  that need it; it does not remove it. If the unported loop fires on most scenarios, the snapshot cost
  comes straight back and the hybrid's structure was not the problem.
- The per-scenario Java work at a pause is serial. With 500-scenario chunks that is the thing to
  measure in stage 0, not to assume.
- Two host slots means two places the sweep can stop per pass. The SVC experience says an added slot is
  cheap, but it also says a new context buffer read by an existing kernel must be initialised at UPLOAD
  time, not only in the per-solve reset (#49's `dSvcRlVt` bug).

### #49 — Host orchestrators deleted; the device driver is the default (2026-09-16)

Every decision loop that ran AFTER the device sweep is gone: `GpuBatchSecondaryVc.orchestrate`,
`GpuBatchVoltageMonitoring.orchestrate`, `GpuBatchContinuousPhase.orchestrate`,
`GpuBatchIncrementalTvc/ShuntVc/FlowControl.orchestrate`, `GpuBatchSimpleTvc` and
`GpuBatchSimpleShuntVc.roundAndResolve`, and `GpuBatchTransformerControl.
orchestrateTransformerControl` — with the helpers that only served them (the clamp/round machine, the
per-step delta builder, the tap-row emitter, the per-loop ratchet movers, and the DS capture/seed
carry triple). About 1,500 lines.

They were never ports of OLF's loops, they were COMPENSATIONS for running outside the sweep: an
orchestrator re-solves after everything, so it needed the DS carry triple to undo its own re-upload of
base targets, and two of them in a chain discard each other's target moves (measured 2.4e-1 kV on
SecondaryVc + AFTER_GENERATOR TVC). Their correctness was conditional on being the only decision loop
in the run.

A loop the kernel REFUSES now routes the whole group to the CPU (`refusedLoops`, leaving
`resultsByIndex` null, which is already how the caller sends a contingency to the stock path).
Un-accelerated is the only honest alternative to a different algorithm. The one remaining fallback is
SecondaryVoltageControl's driver PAUSE at its own OLF slot — that is IN OLF's order, so it is not an
orchestrator.

**The device driver is also the DEFAULT for the base load flow** (`driverOuterLoopsEnabled`, formerly
`OLF_GPU_DRIVER_OUTER_LOOPS=1`). A batch of one is still a batch, and running the base through the
same driver as the chunks is what keeps the two on one trajectory. rte6515 parity is identical either
way (9040/9042, the same 2 knife-edge lines) and the driver is never slower.
`OLF_GPU_DRIVER_OUTER_LOOPS=0` or `-Dolf.gpu.driverOuterLoops=false` puts OLF's Java engine back in
charge of the base.

### #48 — Obsolete gates lifted, and the SecondaryVoltageControl ORDER gate (2026-09-15)

Three gates written for the host-orchestrator world were re-examined once every control loop became a
driver phase.

**Lifted.** (a) INCREMENTAL and WITH_GENERATOR transformer voltage control no longer require
`transformerControlSupported`: neither loop has a GeneratorVoltageControlManager, so the
generator-under-the-controlled-level condition never applied to them — it was a blanket copied from
AFTER_GENERATOR (`deviceIncremental/SimpleTvcGeneratorUnderLevelMatchesCpu`, OLF's four-bus network
with 90 kV generators under a 230 kV controlled bus). (b) The WITH_GENERATOR shunt VC co-presence set
shrank from seven loop names to nothing: it existed because the host orchestrator CHAIN could not
model OLF interleaving two simultaneously-unstable loops, and the driver sweeps in OLF's order with
OLF's restart-on-UNSTABLE semantics (`deviceSimpleShuntWithTransformerControlMatchesCpu`, a shunt at
the 132 kV bus and a transformer at the 33 kV bus, AFTER_GENERATOR and INCREMENTAL).

**Added, for correctness.** SecondaryVoltageControl co-present with ANY control loop is now a
whole-run CPU gate. SVC is OLF order 4, before phase (7), transformer (8), RPC (9) and shunt (10), and
it is the last HOST orchestrator on the chunk path: it runs AFTER the entire device sweep. So an
in-kernel control loop decides on a state SVC has not corrected yet, and SVC then decides on a state
whose taps have already moved. Measured on a zone whose pilot bus feeds an AFTER_GENERATOR
transformer: **1.7e-3 kV** with the control loops in-kernel and **2.4e-1 kV** with the
pre-2026-09-15 host-orchestrator chain — that combination was SILENTLY WRONG before this gate, and
routing the host chain back was measured worse, not better (the chain's re-solves do not compose).
`deviceSecondaryVcWithAfterGeneratorTvcRoutesToCpu` pins it; flip it to `assertGpuMatchesCpu` when
the gate lifts. This also answers the "restart with a machine mid-flight" question: the shape is
unreachable while the gate stands, because SVC is the only loop that can force a restart.

**LIFTED the same day — SecondaryVoltageControl runs as a DRIVER PAUSE on the chunk path.** The kernel
already had the `LOOP_HOST_SVC` slot, `hostLoopMask`, `readBatchDriverPaused`/`setBatchDriverResume` and
flat per-scenario `setBatchTargetDeltas`; only the base path used them. The chunk solver now arms the
mask when the run has SVC controllers and, after the main solve, loops: decide on the paused states →
push the moved V targets as flat deltas → resume. The decision is the SAME code as the host
orchestrator (`decideRound`, extracted so the two cannot drift), and the distributed-slack carry triple
is not needed at all because the driver owns the slack. `OLF_GPU_SVC_HOST_ORCHESTRATOR=1` restores the
old after-the-sweep orchestrator for A/B.

Getting there took two real defects, both found by BISECTING the machinery away from the decision
rather than by reading the code — the technique is worth keeping:

1. Arm the pause but resume every pause as STABLE, leaving the host orchestrator to decide. Still
   wrong (361 differing violations vs 4) ⇒ the machinery, not the decision.
2. Keep the slot in the sweep but make it an in-kernel no-op, so no call split happens
   (`OLF_GPU_HOST_LOOP_NOOP=1`). Correct again (1 differing on the slice, like the baseline) ⇒ the
   SPLIT of `solveBatchContext`, not the extra slot.

That pinned it to state a resumed call re-uploads over the sweep's own work:

- **The disabled-element mask.** A resume re-uploaded it from the host, undoing the element bits the
  HVDC AC-emulation freeze and limits phases had set during the paused sweep. The host mirror is the
  pre-solve one. Guarded with `!resume`: contingency outages do not change during a sweep, so the
  device copy is authoritative. This is why only rte6515 broke — ieee14 has no HVDC.
- **The outer-iteration budget.** A host-loop UNSTABLE verdict did not consume it, while every device
  loop's change does. Each resume then began a fresh `outer` budget, so a loop that kept moving
  targets could never retire a scenario: a 2000-contingency slice did not finish in 28 minutes. The
  resume path now increments `prodIters` for the changed scenarios and retires one that exhausts
  `RL_MAX_OUTER` to the CPU, exactly as the device loops do.

Result on rte6515 with both zones firing, 9042 contingencies:

| chunk path for SVC | differing violations | batched | GPU cold | vs CPU 8 threads |
|---|---|---|---|---|
| host orchestrator | 4 | 9040/9042 | 204 s | x0.47 |
| driver pause (default) | 4 | 9037/9042 | 150 s | x0.66 |
| no SVC, same case | 2 | 9040/9042 | 81 s | x1.20 |

Identical parity, 27% faster, and 3 scenarios conservatively retired to the CPU by the budget rule.
The remaining 4 differing lines are a pre-existing SVC scale gap shared by both paths (2 of them are
the baseline knife-edge lines) — that is the next SVC question, not this one. SVC still makes the case
about 2x slower than without it on either path, so it needs a cost pass of its own.

Still on the host inside the pause: the observables (OLF's own `EquationTerm.calculateSensi`) and the
small zone LU. Moving those into the kernel is a later, optional step — see the stage list below; it
buys wall time, not correctness.

### #47 — Discrete TransformerVoltageControl and WITH_GENERATOR shunt VC in-kernel (landed 2026-09-15)

`LOOP_TVD_KERNEL` (OLF order 8) runs OLF's non-incremental transformer loops per scenario, one thread
per scenario (OLF keeps ONE step for all controls): AFTER_GENERATOR (`tvdMode 0`) — INITIAL enables
every group whose pilot is out of its deadband (TransformerRatioManager bounds computed over the
alive members), CONTROL clamps each continuous ratio outside its (group-averaged when
useInitialTapPosition) range to the extreme, rounds it (`tvdRoundToTap`: from the CURRENT tap, strictly
closer only — `ClosestTapPositionFinder`) and fixes it, else remaps (`updateContinuousRatio`) + rounds
everything and goes COMPLETE, UNSTABLE in any case (the COMPLETE polish solve); WITH_GENERATOR
(`tvdMode 1`, SimpleTransformerVoltageControl) — the control is ON at entry (`tvdEntryBatch` applies the
ON deltas for a chunk, or reads them off the base activation OLF's initialize produced, `entryOn`),
the first check rounds and fixes every live member. Rows are laid out by ROLE (`tvdLayoutGroup`): the
pilot's V identity and the other non-member deltas as captured (the scenario's OFF target is SAVED on
the way on — a per-scenario override on the pilot's Q row comes back), the first alive continuous member
hosts the relocated pilot Q (the fills' mode scan — `relocChoose` / the DISTR_RHO carrier skip — picks
the same row), the others DISTR_RHO (0), fixed/outaged members identity pins; a group with no continuous
member left is OFF. Fixed taps ride the shared r1 override rebuild. The plan-A6 carrier-clamp rejection
is OFF by default: the in-kernel sequence IS OLF's, and `deviceSharedCarrierClampBatchesAndMatches`
now batches the clamp scenario (and drives the clamping base) at 1e-12 — `OLF_GPU_TVD_REJECT_CARRIER_CLAMP`
restores the CPU route for diagnosis. Base path: `readBatchDiscreteTvcState` → `setR1(roundFrom);
roundR1ToClosestTap(); setVoltageControlEnabled(false)` per fixed member (OLF's own calls).

`LOOP_SVD_KERNEL` (OLF order 10) is the shunt twin (`svdEntryBatch` / `svdCheckBatch`): ON at entry,
the first check runs `LfShuntImpl.dispatchB` verbatim per controller shunt (residue / remaining to the
closest section from the CURRENT position over [minPosition, size), strictly closer only), pins the
SHUNT_B identity at the section total and restores the pilot; OLF's dispatchB leaves the shunt's g
alone, so no (dG, dB) override is emitted (the host machine's dg was an OLF deviation, harmless on
constant-G sections). Base path: `readBatchDiscreteShuntState/Positions` → `setB(contB); dispatchB()`.

A solve that FALLS BACK to the CPU Newton-Raphson (a rejected shape) now turns the driver off for the
REST of that engine run (`driverFallbackForRun`): OLF's real loops took over there and hold the loop
state, so the device must not restart its machines from INITIAL against contexts sitting at
CONTROL/COMPLETE (measured 2.2e-3 kV on the carrier-clamp base before the rejection was dropped).
The host machines (`GpuBatchTransformerControl.orchestrate…`, `GpuBatchSimpleTvc`,
`GpuBatchSimpleShuntVc.roundAndResolve`) stay for a control the upload refuses (a member without a tap
table).

**GeneratorVoltageControlManager + fixTransformerVoltageControls (same day, second commit).** The
last AFTER_GENERATOR gate is gone: generator voltage controls under the transformer-controlled
level are handled in-kernel. Java (`buildFreezeTables`) replicates the manager's static part —
`minNominalVoltageLimit` (the `generatorVoltageControlMinNominalVoltage` override, else the max
nominal voltage over the transformer-controlled buses whose controllers all join two different
levels, `Double.MIN_VALUE` when none), the candidate controller buses (at or under the limit, no
`hasStepUpTransformers` exemption; a REMOTE candidate keeps the CPU route, its row is a DISTR_Q
share) with their controlled-bus V row and load Q, and per member the V rows of the generator-
controlled buses in the not-controlled side's component with every transformer-controller edge
removed (`LfNetwork.getConnectivity()` temporary changes). The kernel's INITIAL step, once needRun
is known: every candidate whose row is still a V identity (mode 0/2 — an RL-pinned one is NOT frozen,
like OLF's `isGeneratorVoltageControlEnabled` test) saves its mode/target and becomes a Q row at
`qBuf − loadQ` (OLF freezes the generation target at `q.eval()`, the NET injection, so the row
target is that minus the load); then each just-enabled member whose PV list holds no bus still in
mode 0/2 is disabled again (frozen AND pinned buses count as off, exactly `fixTransformerVoltage
Controls`); then the groups lay out. The COMPLETE transition restores the frozen rows
(`enableGeneratorVoltageControlsUnderMaxControlledNominalVoltage`). RL ignores a frozen row (mode 1,
limit type 0) as OLF ignores a bus without QLimitType. Base write-back:
`freezeGenerationTargetQAndDisableGeneratorVoltageControl(target + loadQ)` while frozen,
`setGeneratorVoltageControlEnabledAndRecomputeTargetQ(true)` once unfrozen (only a bus whose network
state shows OUR freeze: control off, target frozen, no QLimitType — the RL write-back keys on the
QLimitType and never touches it). `getMaxControlledNominalVoltageOverride()` is the second OLF
getter of the day. Approximations, documented: the component lists are the BASE topology with ALL
controller edges removed (OLF removes the ENABLED ones plus the contingency's branches per solve),
so a contingency splitting a not-controlled-side component, or an in-band group whose branch is
the only path to the PV buses, can differ. Tests: `deviceTransformerControlGeneratorFreezeMatchesCpu`
(2T2wt at 150 kV: G1 frozen and the controls disabled by the fix; 90 kV / auto: no freeze) and
`deviceSeveralTvcGeneratorFreezeMatchesCpu` (OLF's four-bus network: auto freezes the 90 kV
generators; 230 kV freezes all and the fix disables the three controls), 1e-12, count asserted,
driver off and on.

### #46 — Continuous PhaseControl in the phase phase (landed 2026-09-15)

`cphCheckBatch` shares `LOOP_PHASE_KERNEL` with the incremental kinds (`nCph > 0` selects it).
Per (scenario, controller) at the phase slot's FIRST check (`dCphIter[s] == 0`): a CONTROLLER
switches off — its a1 row leaves mode 3 (free a1, P block re-emitted) for the identity pinned at
the closest tap's a1 (`x` seeded there) — and the scenario is UNSTABLE whenever a live controller
exists (OLF: `status = UNSTABLE` for every controller at iteration 0, even one whose rounding
changes nothing — that still costs the CPU a Newton pass, so it costs us one too). From the second
check on, a LIMITER whose controlled I exceeds its limit moves one tap against sign(dI/da1) —
`flcFlowAndGrad(kind 2)` at the branch's current state, local control only (remote LIMITERs stay
CPU-routed at extraction). Taps ride the shared r1 override rebuild (`tvcRebuildOverrides` merges
TVC + phase + flow tables + the caller's list). Base path: `readBatchContinuousPhaseTaps` →
`PiModel.setTapPosition` + `setPhaseControlEnabled(false)`; the `DeviceDrivenLoop` proxy now
forwards `initialize()` to the real loop for every driven loop except FreezingHvdcACEmulation
(PhaseControl's initialize enables CONTROLLER control, which the extraction must see). The host
orchestrator (`GpuBatchContinuousPhase.orchestrate`) stays only for a chunk whose tables did not
upload. 10/10 phase tests at 1e-12, driver off and on.

### #44 — Shared-group incremental TVC in the TVC phase (landed 2026-09-15)

`tvcGroupDecisionBatch` replaces the per-controller decision kernel: one thread per (scenario,
controlled-bus group) runs OLF's `checkAndAdjustControlledBus` — `adjustWithOneController` (full
diffV, maxTapShift) for a singleton, `adjustWithSeveralControllers` (one tap per controller per pass
in merged order, remainingDiffV through each sensitivity, MIN_SENSI_FILTER per controller, until a
pass moves nothing) otherwise, both through `tvcMoveTap` (closest r1 within the shift bound and the
ratchet's range, strictly closer than the current one). The JNI upload no longer refuses shared
buses, so the host TVC orchestrator is off for every incremental group; `deviceIncrementalSharedTvc
MatchesCpu` runs its 3-member group in-kernel (`nCtrl=3 nGroups=1 kernel=true`) at 1e-12, driver
off and on.

### #43 — In-kernel IncrementalShuntVoltageControl (landed 2026-09-15)

`LOOP_SHUNT_KERNEL` at OLF order 10, same shape as the TVC phase: `shcDeadbandCheckBatch` per
(scenario, controlled-bus group) → one unit-RHS solve per needed controller at its SHUNT_B identity
row on the driver's own factors (`shcGatherSensiBatch` reads dV at the controlled bus) →
`shcAdjustBatch`, one thread per (scenario, group), OLF's `adjustB` verbatim: round-robin over the
|b|-sorted controller shunts, one section per sub-controller per pass, the per-round maxSectionShift
cap, `updateSectionB` (closest section within one step and the ratchet's allowed range, strict <),
`updateAllowedDirection` (MAX_DIRECTION_CHANGE 3), remainingDiffV through the sensitivity →
`shcApplyBatch` pins the section total at the identity row (target + state seed) →
`shcRebuildOverrides` appends the static (dG, dB) per moved controller to the caller's shunt
overrides (`applyShuntOverride` SUMS matching entries, so the loop's delta composes additively) —
redone on a driver resume. Unlike the TVC phase it takes shared groups (the round-robin runs in the
kernel), so the host orchestrator is now off whenever the tables upload. Base path: the loop's
`getMaxSectionShift()` (OLF getter), `readBatchShuntPositions` → `LfShunt.Controller.updateSectionB`.
Tests: the 7 `*Shunt*` SA tests pass driver off and on with a section actually moving
(`SHUNT_KERNEL ... subsOffBase=1`); no shared-group incremental shunt test exists yet — worth adding.

## Method notes worth keeping

- **A GPU test that only compares against CPU cannot detect that the GPU never ran.**
  CPU-identical results are the fallback CONTRACT. Assert
  `GpuNewtonRaphson.CPU_FALLBACK_COUNT` (see `GpuTestIsolationExtension`), not logs (the test
  logback config attaches no appender) and not `olf.gpu.cpuFallback=false` alone (it only
  covers throws reaching that one catch). `GpuPhaseControlTest` passed for a whole session
  while the GPU never executed.
- **Before marking a capability ⏩ ("a GPU gap"), check the CPU side does anything at all** —
  `LfActionUtils.createLfAction`'s `default` throws. Four action rows were once ⏩ for actions
  OLF does not implement; batching them would make the GPU SUCCEED where CPU throws.
- **Never trust bench timings from a loaded box.** Sample load over several seconds first; the
  2026-07-20 row read ×0.74 purely from contention + thermal throttle.

## Coupler-tree relocation: the remaining step is PER-SCENARIO, on the device (2026-09-23)

`-Dolf.gpu.ziTreeRelocation` (off by default, `…MinCouplers` threshold) lets the BASE solve
rte6515's busbar windows that the numeric row repair cannot hold — window [5000,6000) converges on
device with the trees relocated. What it does NOT yet do is carry the SCENARIOS: a contingency that
opens a relocated coupler deactivates that coupler's `ZERO_PHI`/`ZERO_V` (which the relocation moved
onto the child's `PHI`/`V` rows) and pins its dummies with `DUMMY_TARGET` identities (which those
rows no longer host), so the scenario's system is mis-placed.

Rejected (measured, 2026-09-23): routing coupler-opening scenarios to the CPU. Busbar-section
contingencies ARE coupler openings — the gate left **0 of 1000** scenarios on the device for that
window, i.e. it deletes the workload it was meant to protect. The CPU is not the answer here.

The answer is the mechanism the code already has twice: **both placements in the CSR pattern, the
destination chosen per scenario.**

| scenario has the coupler | child bus P/Q row | coupler `ZERO_PHI`/`ZERO_V` | coupler dummy rows |
| --- | --- | --- | --- |
| closed (the base) | `DUMMY_P/Q(coupler)` | child's `PHI`/`V` | — |
| **opened** | the child's own `PHI`/`V` | deactivated | `DUMMY_TARGET` identity |

One selector bit per (relocated coupler, scenario) — "is this coupler open here" — which the batch
already knows. Precedents to copy, not to reinvent: dynamic Q-relocation (§11 — `qRowAlt`,
`cbAltRow`, `hasRelocatable`: two candidate rows unioned into the pattern, the fill picking by row
mode) and continuous phase control (`cbPcSide`, `collectCbPhasePattern`, `buildCbPhaseDest`: extra
positions unioned, emission gated by a per-scenario mode). Both placements are permutations of the
same equations, so mirroring OLF exactly is preserved.

Open, noticed while measuring the rejected gate: in that experimental config (relocation at
threshold 1, every scenario CPU-routed) the window failed the distributed-active-power assertion
with a worst diff of 14.3599 MW — exactly the base's own distributed total, so one arm reported the
base value where the other reported ~0. CPU-routed scenarios go through
`super.runPostContingencySimulation` (`GpuBatchedAcSecurityAnalysis:1305`), and the default suite is
green, so this is not a shipped defect — but it wants explaining before the fallback path is trusted
for anything but the handful of scenarios it carries today.

## The busbar windows: what actually blocks them (corrected, 2026-09-23)

**A window's BASE network depends on its contingency set.** OLF retains the switches a contingency
needs, so more busbar-section contingencies ⇒ more retained switches ⇒ more zero-impedance couplers
in the base LfNetwork. Measured on window [5000,6000) with `OLF_GPU_SA_MAX_CTG`:

| slice | on device |
| --- | --- |
| 200 | 193 / 200 |
| 600 | 578 / 600 |
| 1000 | **0 / 1000** — the BASE LF itself falls back |

So "window [5000,6000) fails" is not a property of those contingencies: it is the switch count their
union retains. Every windowed measurement has to name its slice, and a slice that passes says nothing
about the full window. This also corrects an earlier attribution here: the coupler-tree relocation was
credited with fixing this window's base. It does not — the N=1000 base fails IDENTICALLY with
`-Dolf.gpu.ziTreeRelocation` on and off (same two fallbacks, same order). The relocation's real result
stands on its own (the coupler chain's first Newton step, exact at 7/15 couplers where the numeric
repair is wrong), but it is not this window's blocker.

The two real blockers, both in the base LF, both before any contingency runs:

1. **`row mode 4 (pinned remote gen-VC member) uses stale static qPct on-device`** — an explicit
   `UnsupportedOperationException`, i.e. a deliberate CPU route, not a numeric failure. A reactive-limit
   pin makes OLF renormalize the group's reactive keys (`updateReactiveKeys`) and fire NO
   equation-system event, so the cached extraction kept its pre-pin `qPct`.
   FIXED (this commit): `GpuAcActivationExtractor.pinnedRemoteMembers` exposes the pin SET, and
   `GpuNewtonRaphson` re-extracts when it changes. The fresh pack's `dqWeight` differs, which
   `GpuAcData.sameStructureAs` already compares, so the structure is REBUILT with OLF's renormalized
   keys instead of being value-refreshed in place. The throw now only covers modes above 4.
   Still open: the BATCHED path pins on-device (the RL kernels set mode 4 themselves) and cannot
   renormalize keys there — the same staleness may exist per scenario. Not yet investigated.
2. **A NaN** on the first base attempt once the reactive-limits loop engages on a switch-heavy base.
   Next up.

### The remaining blocker, characterized (2026-09-23)

`OLF_GPU_LIN_RESID=1` on the N=1000 window separates the two candidate explanations, and it is NOT a
diverging Newton. Three base solves run in sequence (all `slot=0`, n=13228):

| solve | first-iteration `‖J·dx−F‖/‖F‖` | `‖dx‖inf` | `‖F‖inf` at entry |
| --- | --- | --- | --- |
| A | 8.6e-09 → 1.2e-13 over 3 iters | 11 → 1e-2 | 40.6 (flat start) |
| B | 1.2e-11 | 3.2e-03 | 3.3e-03 |
| **C** | **6.7e+08** | **7.7e+13** | **1.73 (a sound state)** |

Solve C enters from a reasonable state and its FIRST linear solve is wrong by eight orders of
magnitude; every later iteration stays at 1e7–1e9 with steps of 7.7e13, and the Newton then walks to
NaN. So the matrix of that outer-loop pass does not factor unpivoted — the divergence and the NaN are
CONSEQUENCES, and `MAX_VOLTAGE_CHANGE` clamping (which the RTE params set, and which both the
sequential and batched paths do pass — `lsMode=2`, `maxVoltageChangeFactorBatch`) cannot save a step
computed from a broken factorization.

The batch's own divergence in the same log is also a consequence, not a second defect: with the base
fallen back, every scenario starts from a flat state (`‖F‖inf = 5.555`, identical to 7 digits across
all slots) instead of the converged base.

Leads for the next session:
- The worst rows repeat — 12582, 12914, 12933, 13005, 13181, all high indices of 13228, i.e. the
  non-bus variable region (DUMMY_P/DUMMY_Q, BRANCH_RHO1, SHUNT_B). Resolve them to equations first:
  a zero/tiny diagonal there is the unpivoted-LU failure this smells like, and the same family as the
  coupler-tree work even though the tree relocation does not change this outcome.
- What changes between the sound solve B and the broken solve C is one OUTER-LOOP pass. Identify
  which loop acts there (PV→PQ, a reactive-limit pin, a tap or shunt move) — that names the row whose
  mode/placement flips into the unfactorable shape.
- The single-solve linear-residual guard runs only at `it == 1`
  (`olf_gpu_nr_jni.cu:2678`), so a factorization that degrades later is unguarded on that path; the
  batched path's `lastSolveRatio` check is what actually caught this one, reported as a NaN.

#### Hypotheses eliminated by measurement (same window, one run each)

- **Fill-reducing ordering.** `OLF_GPU_REORDER_ALG=nd` → still NaN. Worth testing because the robust
  config hard-codes AMD when the variable is unset, and this very file documents that "AMD IS NOT SAFE
  WHEN THE SYSTEM HAS ZERO-IMPEDANCE BRANCHES" (AMD is symmetric and assumes a non-zero diagonal;
  ZERO_V/ZERO_PHI sit on the DUMMY rows and carry v1−v2 / phi1−phi2, never their own dummy) — on a base
  whose whole difficulty is retained switches. The reorder choice is NOT the cause here, but the robust
  path bypassing `applyReorderAlg(config, hasZeroImpedance)` still looks wrong on its own terms.
- **Robust mode engaging mid-run.** `-Dolf.gpu.batchMode=6` from the first context → still NaN. Tested
  because cuDSS permits ONE analysis per data object and matching's permutation is value-dependent, so
  a robust context built after a reject analyses mid-run values; the comment credits the full-scope
  base's convergence to analysing the run's FIRST values. Not the cause either.

So the robust single-matrix path (matching + pivoting, AMD or ND, from the start or after a reject)
returns NaN on this structure, while the uniform batch returns a finite-but-wrong solve (ratio 1e7).
Both orderings and both modes failing IDENTICALLY points away from the factorization and toward the
matrix itself.

**Next probe (needs a native rebuild, small).** The device checks `F` for finiteness in both paths
(`olf_gpu_nr_jni.cu:2578`, `:5192`) but NOTHING checks `dVals` — the Jacobian — before handing it to
cuDSS. An env-gated scan of `dVals` for non-finite entries, reporting the first offending (row, col)
and the element that fills it, decides between "the factorization cannot handle this matrix" and "a
fill wrote NaN into the matrix". Do that before any more ordering/mode sweeps: a NaN entering from a
fill would explain every result above at once.

### ROOT CAUSE LOCALIZED: the device's own RL pin assembles a rank-deficient system (2026-09-23)

The base solve that fails is the FIRST fill carrying **mode-4 rows** — an RL-pinned remote gen-VC member.
The per-fill census makes it unambiguous (`OLF_GPU_VALS_SCAN=1`):

| fill | mode 0 | mode 1 | **mode 4** | dupRows | deadCols | tinyDiag |
| --- | --- | --- | --- | --- | --- | --- |
| healthy (outer=0) | 998 | 12230 | **0** | none | 0 | 0 |
| **broken (outer=6)** | 982 | 12243 | **3** | none | 0 | 0 |

The matrix is finite, well-scaled (maxAbs 4.07e+04), every diagonal present and above 1e-12, no duplicate
row, no dead column — and unsolvable. **KLU, the CPU's own solver, cannot solve it either**
(`GpuDumpedMatrixRankTest` on an `OLF_GPU_VALS_DUMP`: `||dx||inf = 3.97e+16`, residual ratio **7.59**
where a real solve is ~1e-12). So cuDSS is NOT at fault and no solver configuration can help: the system
we assemble is rank-deficient. Since OLF converges on this network on the CPU, OLF's matrix at the
equivalent pass is not this matrix — **the defect is ours**.

KLU's blown-up solution is a null-space direction, and its largest components name where the rank is
lost: columns 13171, 13207, 13201, 13185, 13177, 13179, 13175, 13199, 13170, 13181 — all with
`diag = ±1` and `F = 0`, i.e. DUMMY_P / DUMMY_Q columns — clustered around mode-4 row 12560 (the three
pinned rows are 12560, 12588, 12834). That is the failure class `GpuAcActivationExtractor:129-146`
already describes: the existence-keyed DISTR_Q kind-3 terms staying live ALONGSIDE the zero-impedance
family's direct ±DUMMY_Q into the same row, the two cancelling and "leaving the coupler's DUMMY_Q in NO
equation. 45 such columns at 2000 busbar sections, which is what collapsed batching above ~1000" — and
N=1000 is exactly where this window breaks. The `pilotSelf` fix recorded there evidently does not cover
this instance.

Two further facts that matter for the fix:
- The pin happens **on the device**, inside the RL kernels, so no host-side check sees it. The
  pre-change Java produces a byte-identical broken fill (`modes=982,12243,0,0,3`), which is how we know
  the `row mode 4` de-fallback did not cause this — it only stopped hiding it.
- Ruled out by measurement, so do not re-sweep them: orderings AMD / ND / BTF-COLAMD, pivot types
  auto / global column / global row with threshold 1.0, matching auto and off, NaN entering from a fill,
  zero and tiny diagonals, duplicate rows, dead columns.

Next: compare the device's assembled system at that pass against OLF's pinned system for the same
group — specifically whether a pinned member's own Q row and its DISTR_Q row both carry, or both drop,
the coupler's DUMMY_Q term. Fix the assembly; the factorization is innocent.

Tools left in place for it: `OLF_GPU_VALS_SCAN=1` (per-fill census: finiteness, maxAbs, diagonals,
duplicate rows, dead columns, mode histogram), `OLF_GPU_VALS_DUMP=<path>` (one-shot CSR+rhs+modes at the
first mode-4 fill), `GpuDumpedMatrixRankTest` (factor it with KLU offline, print the null direction), and
`OLF_GPU_PIVOT_TYPE` / `OLF_GPU_PIVOT_THRESHOLD` / a now-honoured `OLF_GPU_MATCHING_ALG` in robust mode.

#### The wrong equation, named

The null direction's dummy columns show the defect structurally:

```
col=13171  touchedBy=3  (row=12560 mode=4 v=+1) (row=12562 mode=1 v=+1) (row=13171 v=-1)
col=13207  touchedBy=3  (row=12834 mode=4 v=+1) (row=12836 mode=1 v=+1) (row=13207 v=-1)
```

A zero-impedance branch's `DUMMY_Q` is determined by an OPPOSING pair: `+1` in one endpoint's Q row and
`-1` in the other's (`fillZeroImpBatch`: `addAt(d2, dummyQ)` / `addAt(d3, -dummyQ)`). Here both real
entries are `+1` (the `-1` on the diagonal is the numeric repair's donor, `nHaz=242`), so the
opposite-side coupling is MISSING. That endpoint is a remote gen-VC member whose own Q row is canonically
SUPPRESSED (`qRow = -1`, `ExtractionColumns:164-172`) because its reactive balance moved into the group's
DISTR_Q row — so the `-dummyQ` dest resolved to `-1` (skip) and the term was dropped. The dummy is then
undeterminable, and across the pinned cluster the matrix loses rank. The identical pattern at both pinned
members (12560/12562/13171 and 12834/12836/13207) is a systematic assembly bug, not a numerical accident.

OLF does not lose this: the suppressed member's reactive balance still exists inside the group's DISTR_Q
equation. Our layout drops the coupling instead of ROUTING it to wherever that member's balance now lives.

Fix direction: the zi family already carries a per-side alternate Q row for exactly this
(`ziQAlt`, `ziDynDest`), so the question to answer first is whether `ziQAlt` is `-1` for such a bus (the
alternate was never computed for a suppressed remote member) or whether it is set but not SELECTED when
the other side is at mode 4. Fix that, then re-measure the N=1000 window: the base should factor, and
976 of 976 scenarios are already planned onto the device behind it.

### THE DEFECT, read off our own equation dump (2026-09-23)

`GpuEquationDump` (new, the counterpart of OLF's `EquationSystem#write`) prints one line per device row
plus a per-coupler section with every row slot, `-1`s included. Run it with `OLF_GPU_EQ_DUMP=<path>`; it
also writes OLF's own dump of the same state to `<path>.olf`. On the failing window it names the defect:

```
zi#18 branch=BUS-5997_COUPLER_1 b1=VL-5997_0 b2=VL-5997_1
      pRow1=12561 qRow1=12562 pRow2=12563 qRow2=-1 dummyP=13170 dummyQ=13171 qAlt=[-1,-1]
zi#36 branch=BUS-6112_COUPLER_1 ... qRow2=-1 dummyQ=13207 qAlt=[-1,-1]
```

Those are EXACTLY the two columns KLU's null direction pointed at. The `-dQ` coupling has no row
(`qRow2 = -1`) and no alternate (`qAlt = [-1,-1]` — so it was never computed, not merely unselected),
because that endpoint is a remote gen-VC member whose own Q row is canonically suppressed: its reactive
balance lives in the group's DISTR_Q row.

Why the base is healthy until the pin, which is what makes this conclusive:

- BEFORE the pin the dummy's missing second coupling is supplied by the DISTR_Q **kind-3** term
  (`dq[elem=13171 kind=3 w=qPct-1]`), so the column IS determined and the base converges.
- AT the pin `updateReactiveKeys` zeroes that member's own key, so every DISTR_Q term on its row becomes
  the exact negation of the member's own terms (`w = 0 - 1 = -1`): the dump shows `dq[k=106 elem=8699
  w=-1]` against `cb#8699.q2` and `dq[k=107 elem=13171 kind=3 w=-1]` against `zi#18.q1(+dQ)`. The
  compensation disappears, `DUMMY_Q` keeps a single `+1`, the column is free and the matrix loses rank.

So the row layout is over-subscribed for such a group: one row (12562, `BUS_V(VL-5997_1)`) carries the
relocated pilot reactive balance AND has to become the member's OWN Q balance when the member is pinned.
OLF has no such problem because its rows are EQUATIONS — `BUS_TARGET_Q` and `DISTR_Q` each get their own.

Fix options, in the order worth trying:
1. Give the zi family the missing alternate: compute `ziQAlt` for a suppressed remote gen-VC member (its
   reactive balance row), so `ziDynDest` can route `-dQ` there when the member is pinned. Smallest change,
   uses machinery that already exists.
2. Keep the DISTR_Q kind-3 compensation alive for the dummy when the member is pinned (i.e. gate only the
   terms that duplicate the member's own q, not the coupler's).
Either way the invariant to assert afterwards is the one the dump now checks: a coupler's dummy coupling
is complete, or the missing side has an alternate that is live in every row mode.

### THE FIX: 0 -> 900 of 1000 scenarios on the device (2026-09-23)

The RL bookkeeping keys every pin on the entry's own V row, which is only valid when that bus's reactive
balance actually lives there. On a PILOT that is one of its own controllers, pinned while a SIBLING
controller still holds the voltage, it does not: the pilot's V row must keep `BUS_TARGET_V`, so
`computeQRow` relocates its reactive balance to the free member's V row. Two entries then key off one row
— the pilot's (whose pin marker is elsewhere) and the receiving member's (which reads a marker that is not
its own) — and the device pinned one row twice while unpinning the row the balance lived on. Exactly 2
buses on rte6515 have this shape (`VL-5997_0`, `VL-6112_0`); the other 3 relocated pilots have candidate
tables (`qRowAlt = -2-group`) and their re-pick machinery is fine.

Fix: the host marks such entries with `GpuAcOuterLoopPacks.RL_BASE_RELOCATED` (a base limit-type code, 6),
and the device turns that into `RL_LIMIT_FROZEN` so the reactive-limits pass, the switch-back AND
`seedRlModeFromState` all leave the row alone. The last one mattered: `RL_LIMIT_FROZEN` is non-zero, so
without an explicit skip that seeder stamped mode 4 back onto the wrong row and undid the fix.

| window [5000,6000), N=1000 | before | after |
| --- | --- | --- |
| scenarios on device | **0 / 1000** | **900 / 1000** |
| statuses vs CPU | — | identical (974 CONVERGED, 2 MAX_ITERATION both arms) |
| violations | 71538 vs 71533 (diff 5) | 71538 vs 71534 (diff 4) |
| fallbacks | 2 | 1 |

Suite: 288 tests, 0 failures. Checkstyle unchanged at 214.

Still open on this window, in order:
1. The base LF still falls back once (NaN). Understood now: the ALL-PINNED corner legitimately flips the
   pilot's V identity to mode 4 (`updatePilotVeqBatch`), but in this TABLE-LESS group neither the pilot nor
   the free member has an alternate dest (`qRowAlt = -1`), so the flipped row assembles EMPTY. Giving them
   `qRowAlt = vCol` in `ExtractionColumns` was TRIED and REVERTED: it changes the alternate-dest selection
   and the unioned pattern for every scenario, and took the window back to 0 / 1000. The alternate has to
   be introduced without disturbing the batch — most likely by making the all-pinned flip route the q terms
   itself, rather than by widening the static canonical encoding.
2. `distributedActivePower`: 525 of 974 contingencies differ, worst 1.057 MW. Unchanged by this fix and
   identical in the all-CPU-routed run, so it belongs to the base-fallback path, not to the batch.
3. Violation diff 4 of 71538.

### Why the base still falls back: a DEGENERATE DISTR_Q, not a missing row (2026-09-24)

Printing the host's pinned rows next to the device's settles what the remaining NaN is. At the failing
pass the two disagree outright:

```
host  (from OLF's state) hostMode4Rows=[12562 12588 12590 12592 12836]
device (ctx->dMode)      mode4Rows=[12560 12588 12834]        <- at outer=6, the first pinned fill
```

The device is not mis-pinning: it runs its own RL decisions against the FIRST extraction's layout, and in
THAT layout its choice is routable and right. `OLF_GPU_EQ_DUMP_WHEN=any` shows it:

```
VL-5997_0 (pilot)  qRow=-1 qRowAlt=12560 enabled=true   row=12560 mode=0 terms=[cb#8699.q2@alt]
VL-5997_1 (member) qRow=-1 qRowAlt=-1    enabled=true   row=12562 mode=1 terms=[dq ... w=0.500000 ...]
```

Both q scatters are suppressed into the group's DISTR_Q row with weight 0.5 while both controllers are
enabled — correct. The device then pins the pilot at 12560, where its `@alt` dest routes its q terms
properly, and the key renormalization gives the SURVIVOR a share of 1.0. At that point
`DISTR_Q_member = 1·Σq − q_member = q_pilot`, which is the SAME equation as the pilot's own Q balance now
hosted at 12560. Two identical equations, which is precisely the two `+1` entries measured on the
`DUMMY_Q` column and the lost rank. Nothing is missing from the assembly; one equation is DUPLICATED.

OLF cannot hit this: when a controller pins, the survivor becomes the FREE controller, its `DISTR_Q`
ceases to exist and `BUS_TARGET_V` takes that row. That ROLE ROTATION is what the candidate table
(`qRowAlt = -2-group`, `relocChoose`, `updatePilotVeqBatch`) implements — and this group has none, because
the qr-group loop deliberately excludes "a pilot that is ALSO one of its own controllers" (leaving it in
the table double-counted its q). Exactly the 2 rte6515 buses in that shape are the ones that fail.

So step 1 is NOT a small patch. Two ways to close it, both real work:
1. Extend the candidate table to pilot-self groups: give the group a table whose member list drives the
   re-pick while keeping the pilot's own q out of it (the double-count the exclusion was protecting
   against). Then the survivor's row gets the V identity and its DISTR_Q is dropped, as on the CPU.
2. A targeted rotation: when the device pins the pilot of a pilot-self group, set the surviving free
   member's row to mode 0 with the group's V target and gate off its DISTR_Q — the same effect, driven by
   a small host-provided pairing (pilot row, survivor row, V target) instead of a general table.

Tried and rejected on the way here, so do not repeat them:
- Static alternates for both buses (`qRowAlt = vCol`): took the window from 900 back to 0 / 1000. The
  selector reads the FROZEN row's mode, and mode 4 there is the pilot's own relocated pin, so the pilot's
  q got diverted off the row that must host it.
- Excluding relocated entries from the all-pinned flip test (`pilotRowAllPinnedBatch`): kept, it is
  correct on its own terms (the flip cannot route q in a table-less group), but it did NOT clear the NaN —
  the flip was never the trigger.

#### Parity check: option 1 IS the CPU behavior (2026-09-24)

`AcEquationSystemCreator.updateRemoteVoltageControlEquations` is the spec, and it is a ROTATION:

```java
vEq.setActive(!enabledControllerElements.isEmpty());       // BUS_TARGET_V iff >= 1 controller enabled
for (disabled)   { distr OFF; ctrl /*BUS_TARGET_Q*/ ON; }  // every PINNED controller gets its own Q balance
for (i, enabled) { distr.setActive(i != 0); ctrl OFF; }    // the FREE one is the FIRST ENABLED - no DISTR_Q
```

`i != 0` over the LIVE ENABLED list is the free-role rotation, and `relocChoose` ("pick the first member row
whose per-scenario mode is neither identity nor pinned") is its device-side evaluation. The candidate table
is therefore not a convenience for a corner case — it is the model of OLF's rule, so **option 1 is the
parity behavior**.

The divergence at the failing pass, in those terms:

| | active reactive equations |
| --- | --- |
| OLF | `BUS_TARGET_V(pilot)`, `BUS_TARGET_Q(pilot)` |
| device | `BUS_TARGET_Q(pilot)` at 12560, PLUS `DISTR_Q(member)` at 12562 with its key renormalized to 1.0, which is algebraically `q_pilot = target` |

i.e. `BUS_TARGET_V` is MISSING and `BUS_TARGET_Q` is DUPLICATED. The rank loss is the symptom of that, not
a numerical accident — which is also why no solver setting could ever have fixed it.

Option 2 (a targeted 2-row rotation) is parity-correct ONLY for two-member groups. Pilot-self groups on
rte6515, counted from the dump: 974 with 1 member, 19 with 2, 10 with 3, and one each with 4, 5 and 6. The
13 groups with >= 3 members need the ORDERED candidate list, because after one pin several survivors remain
and only the first goes free while the others keep their DISTR_Q. So option 2 is a stopgap the data already
outgrows; do option 1.

The hook already exists: `qrGroupOfPilot` is deliberately a SEPARATE channel from `qRowAlt = -2-group`
(`GpuAcOuterLoopPacks:241`, "Read from qrGroupOfPilot, NOT decoded from qRowAlt") so that table membership
and Q-routing are independent. That is exactly what a pilot-self group needs — a table that drives the
re-pick WITHOUT re-routing its own q, which is the double-count the exclusion in the qr-group loop was
protecting against.

### Option 1, slice 1: the pilot-self group table (2026-09-24)

`GpuAcOuterLoopPacks.pilotSelfGroups` packs every MAIN remote generator voltage control whose merged
controllers CONTAIN the controlled bus, with >= 2 controllers: member rows AND member RL ENTRIES in merged
order, the pilot's entry and row, and the group's V target. On rte6515: **27 groups — 17 of size 2, 7 of
size 3, one each of 4, 5 and 6** (`OLF_GPU_RL_TRACE` prints `GPU_PILOTSELF`).

Two design facts decided the shape, both read off the dumps rather than assumed:

- **`BUS_TARGET_V` cannot change rows.** A mode-0 identity constrains its OWN row's variable, so on a
  member's row it would pin that member's v to the pilot's target. It stays on the controlled bus's row.
  What rotates is a PINNED pilot's reactive balance, onto the first ENABLED member's row —
  `computeQRow` already puts it there, but FROZEN at extraction while the device's RL loop moves pins.
- **The choice cannot be read off row modes.** The row to pick is mode 4 BECAUSE it carries the relocated
  balance, which is circular. `limitType` per RL entry is the ground truth OLF's own rule uses, so the
  table is keyed by ENTRY.

Slice 1 is additive — nothing consumes the pack yet, so the 900/1000 window and every existing group are
untouched (suite 290/0, checkstyle 214). `GpuPilotSelfGroupTest` pins the semantics on a purpose-built
3-controller group (one generator controlling its own bus, two more controlling it remotely) instead of
inside rte6515's 13k-variable system, and asserts the merged ORDER, since OLF's free role is index 0 of the
live enabled list.

Slice 2 (next): a device kernel over the pack that reproduces
`updateRemoteVoltageControlEquations` from the pin states — for each group, per scenario: the first enabled
member is free (its `DISTR_Q` gated off), every pinned controller's row hosts its own Q balance, and a
pinned pilot's balance rides the free member's row. Then re-measure the window; the base's remaining NaN is
exactly this rotation being missing.

Trap worth remembering, cost an hour: `ExtractionColumns.of` returns `vCol = -1` for a bus whose variables
the LAZY equation index has not sorted yet. The real pipeline extracts after the solver materialises the
index; a test must call `AcSolverUtil.initStateVector` first or it silently packs nothing.

### METHOD CORRECTION: MAVEN_OPTS does not reach the surefire fork (2026-09-24)

Measured today, unambiguously, on the same run shape: with
`MAVEN_OPTS="-Dolf.gpu.pilotSelfRotation=true"` the property arrives as FALSE in the test JVM (0 enrolment
traces); with `mvn -Dolf.gpu.pilotSelfRotation=true` it arrives as TRUE (1917 traces). Surefire forks its
own JVM and propagates `-D` given on the MAVEN COMMAND LINE, not `MAVEN_OPTS`, which only configures the
Maven JVM. **Always pass a GPU property as `mvn -D<prop>=<value>`.**

This invalidates earlier entries here that switched a property through `MAVEN_OPTS`. Specifically:

- **VOID**: "window [5000,6000)'s base fails identically with `-Dolf.gpu.ziTreeRelocation` on and off" — the
  "on" run had it OFF, so both arms were the same configuration. The conclusion drawn from it, that the
  coupler-tree relocation is NOT this window's blocker, is therefore UNPROVEN and must be re-measured.
- **VOID**: the coupler-tree threshold experiments run through `MAVEN_OPTS` (the 0/1000 and base-failure
  numbers attributed to "relocation at threshold 1").
- **STILL VALID** (no property involved, or set in-JVM): the coupler-chain first-step results
  (`GpuZiCouplerTreeRelocationTest` sets the property with `System.setProperty`); the RL_BASE_RELOCATED fix
  and its 0 -> 900 of 1000 measurement; the KLU rank analysis; every equation-dump finding; the
  DISTR_Q-duplication diagnosis, which was read off dumps rather than a property switch.

### Option 1, slice 2: the pilot-self gate, property-gated and NOT yet working (2026-09-24)

Landed behind `-Dolf.gpu.pilotSelfRotation` (default false; suite 290/0 with it off, so the shipped path is
untouched). What works: pilot-self groups now enrol in the candidate table, and the device DERIVES both
facts the gate needs with no new host table — a group is pilot-self exactly when its fallback row (candidate
0, the pilot's own row) reappears among its member rows, and the pilot's RL entry is the one sitting on that
row. `PILOTSELF_GATE groups=27 of 30` on rte6515, the other 3 being the pre-existing table-ful groups: the
derivation is right.

`pilotSelfGateBatch` recomputes, per (scenario, group) before every fill, whether the pilot is unpinned, and
that gate now suppresses BOTH halves — the pilot's raw-q emission at the carrier row (`RelocTables.psSkip`
in all three pick sites) and the carrier's own `DISTR_Q` suppression in `fillDistrQBatch`. The second half
was missing at first and cost a run: for a table-ful group the carrier's `DISTR_Q` is always inactive
because the pilot always has a balance to place, but a pilot-self group's pilot has none while it controls,
so suppressing it dropped an equation without gaining one.

With both halves gated the window is still **0 / 1000 with 2 fallbacks** (vs 900 / 1000 with the property
off), so something in the enrolment is still wrong. Do NOT enable it. Next suspects, in order:
1. All 27 groups enrol, including 17 of size 2 where the pilot and one member are the whole group; check a
   size-2 group by hand against OLF's rule before trusting the general path.
2. `qRowAlt[pilot] = -2-g` REPLACES the own-row alternate the second canonicalization loop gave these
   pilots. The all-pinned corner then relies on candidate 0 falling back to that row — verify it does.
3. The pattern union (`collectCbAltPattern` and friends) now unions every candidate row for these groups;
   confirm the extra positions exist for the rows the gate can select, or a pick lands on a slot that was
   never patterned.

#### The re-measurement the MAVEN_OPTS error forced (2026-09-24)

Re-ran window [5000,6000) at N=1000 with `mvn -Dolf.gpu.ziTreeRelocation=true
-Dolf.gpu.ziTreeRelocationMinCouplers=1`, confirmed live by 2565 `ZITREE` traces:

| | relocation OFF | relocation ON (threshold 1) |
| --- | --- | --- |
| scenarios on device | 900 / 1000 | 900 / 1000 |
| base fallbacks | 1 | 1 |
| violations vs CPU | diff **4** | diff **284** |

So the CONCLUSION stands on valid evidence this time — the coupler-tree relocation neither fixes nor causes
this window's base fallback — and there is a new reason to keep it off: enabling it takes the violation
difference from 4 to 284. The relocation's demonstrated value remains narrow and unchanged: exact first
Newton steps on coupler chains at 7 and 15 couplers, proven by a unit test that sets the property in-JVM.

#### What the relocation's 284/299 violation diff actually is (2026-09-24, corrected twice)

A row permutation cannot change the solution of a correctly assembled system, so a violation difference
under the relocation is NOT evidence against the relocation — it is evidence that something ELSE still
differs from the CPU, which the relocation exposes. Chasing it that way found it.

Corrections to two things stated earlier in this session, both from sloppy measurement:

1. "Keep the relocation off because it degrades violation parity" — wrong framing, deleted as a
   conclusion. The relocation is a permutation; the diff is a symptom to chase.
2. "States match to 1e-9 kV, so the 284 are pure threshold flicker" — wrong, and drawn from the FIRST few
   `BUSDIFF` lines rather than the worst. The harness already sets the prescribed
   `IncreasedViolationsParameters` (0.5, 0.001, 0.5, 0.001, 0.001), so the documented zero-threshold
   flicker is already suppressed. Nor is it the CPU thread count: at the harness's busbar default of 1
   thread the diff is 299, not 4.

Measured properly, with the relocation ON (`mvn -Dolf.gpu.ziTreeRelocation=true
-Dolf.gpu.ziTreeRelocationMinCouplers=1`), the divergence is CONCENTRATED:

| reproducer | symptom |
| --- | --- |
| `BUS-6014_BBS_2` | worst bus difference **3.982 kV** (most contingencies sit at 1e-9 kV) |
| `BUS-5959_BBS_1` | violations cpu=211 gpu=80 (delta -131) |
| `BUS-5962_BBS_1` | violations cpu=230 gpu=107 (delta -123) |
| any | distributed active power worst **808.1 MW** (1.06 MW with the relocation off) |

Two contingencies carry ~254 of the 299. That is a real physical divergence on a few scenarios, not a
reporting artifact, and it matches the per-scenario gap this file already documents: a contingency that
OPENS a relocated coupler deactivates the `ZERO_V`/`ZERO_PHI` the relocation moved onto the child bus's
rows, and pins the coupler's dummies with `DUMMY_TARGET` identities that those rows no longer host. The
base is fine because nothing is open there; these scenarios are not.

Next step, cheap and named: run `OLF_GPU_SA_CTG_ONLY=BUS-6014_BBS_2` with `OLF_GPU_SA_BUSDIFF=1` and
`OLF_GPU_EQ_DUMP`, relocation on, and check whether that contingency opens a coupler whose
`ziChildBus >= 0`. If it does, the fix is the per-scenario placement already sketched at the top of this
file (both placements unioned in the pattern, the destination chosen per scenario) — not a CPU gate, which
was measured at 0/1000 and rejected.

#### CONFIRMED: the relocation's divergence is exactly the coupler-opening scenarios (2026-09-24)

`OLF_GPU_ZITREE_TRACE` now prints `ZITREE_OPEN ctg=<id> coupler=<id> childSide=<n>` from the batch path's
zi-outage loop whenever a scenario opens a coupler whose rows the tree relocation MOVED (`ziIdx` slot 10,
bits 1-2 nonzero). On window [5000,6000) at N=1000 with the relocation on, the sets are:

| set | count |
| --- | --- |
| contingencies whose violations differ from the CPU (A) | **11** |
| contingencies that open a RELOCATED coupler (B) | 67 |
| **A \ B** (diverge without opening one) | **0** |

`A` is a strict subset of `B`, with no exceptions: the 3.982 kV state difference, the 808 MW distributed
power and all 299 violation diffs come from scenarios that open a relocated coupler, and nothing else
differs. The three named reproducers all appear in B (`BUS-6014_BBS_2` opens TWO relocated couplers,
`BUS-5959_BBS_1` and `BUS-5962_BBS_1` one each). 56 of the 67 do NOT diverge, which fits: when the opened
coupler's subtree becomes an island, the freeze makes every row of it an identity and the placement stops
mattering (the fix committed as 2fee085).

This is the validation target for the per-scenario placement work: after it, **A must be empty while B stays
67**. A metric that only counts violations cannot see the difference — use the bus-level worst over ALL
contingencies (`OLF_GPU_SA_BUSDIFF=1`), not the first lines of it.

It also corrects the earlier rejection of a targeted gate. The gate measured at 0/1000 keyed on ANY coupler
opening, which is every busbar contingency. The precise set is 67 of 1000 (6.7%), and only 11 of those
actually break — so a gate on RELOCATED-coupler openings would cost 6.7% of the batch, not all of it. Still
scaffolding rather than the goal, but an order of magnitude cheaper than what was measured and rejected.

### Per-scenario placement for the relocated coupler rows (2026-09-24)

The relocation moves a child bus's P/Q equations onto its parent coupler's `DUMMY_P`/`DUMMY_Q` rows and the
coupler's `ZERO_V`/`ZERO_PHI` onto the child's own rows. A scenario that OPENS that coupler skips the zi
element (no ZERO equations, no ±dummy coupling) and the host pins the dummy rows with `DUMMY_TARGET`
identities — so the relocated equations get zeroed and the bus is left with no balance. That, and only that,
is what the confirmed set inclusion (A ⊂ B) measured.

Implemented without any new host table, JNI change or `GpuAcData` component, because rows ARE variables: a
relocated child's own P/Q rows are already in `ziIdx` (slots 8/9 phi, 6/7 V) and the relocated dest rows are
slots 4/5. `buildStructure` derives one map, `relocOwnRow[dummyRow] = the child's own row` (both sides
through `rowMap`, since those slots are COLUMNS), and every family's existing alternate-dest channel then
offers the stock placement as its alternate via `stockAltRow`.

The selector is the DUMMY ROW itself, which the host already sets to mode 0 exactly when the coupler opens.
One correctness detail: the existing channel diverts on `mode ∈ {0,4}`, but a relocated child that is
RL-PINNED carries mode 4 on that same dummy row, where its own Q balance legitimately lives — diverting
there would be wrong. So these outputs encode their selector as `-2-row`, which the three pick sites
(`scatterAddDyn`, the residual variant, `ziDynDest`) decode as "divert only while this row is an identity".

Covered: closed branch, open branch, shunt. NOT covered yet: the zi family's own ±dummy couplings into a
relocated child's rows — its alternate channel has Q slots only (4 per element) — which matters for coupler
CHAINS, where a sub-coupler emits into its parent's dummy rows. 1,692 of 2,538 trees are depth 1, so the
measurement below says how much of the gap that leaves.

Measured on window [5000,6000) at N=1000, relocation on (`mvn -D...`, 1 CPU thread — the harness default for
busbar, since a multi-threaded CPU arm lands on a different bus decomposition):

| | before | ungated first attempt | **gated** |
| --- | --- | --- | --- |
| A = contingencies whose violations differ | 11 | 29 | **4** |
| A outside B (should be 0) | 0 | **25** | **0** |
| total violation difference | 299 | 73 | 93 |

The ungated attempt broke the invariant: it applied the fallback to DEAD-END stubs too, whose `childSide` is
also nonzero, so scenarios that open no relocated coupler started diverging (25 of 29). Fixed by marking
tree-relocated couplers with **bit 3** of `ziIdx` slot 10 (`GpuAcFamilyPacker`, set only when
`ziTreeRelocationEnabled()`), which the native map requires. With the property off the bit is never set, so
`relocOwn` is null, `dynReloc` is unchanged and the structure is bit-identical BY CONSTRUCTION rather than
by measurement — the lesson from the ungated run, where the default window moved and there was no
apples-to-apples baseline to compare against.

The two worst cases before this change are FIXED: `BUS-5959_BBS_1` (was cpu=211 gpu=80) and
`BUS-5962_BBS_1` (was cpu=230 gpu=107) no longer appear in A. The four survivors are

| survivor | relocated couplers opened | delta |
| --- | --- | --- |
| `BUS-5958_BBS_1` | 1 | +24 |
| `BUS-6014_BBS_2` | 2 | −2 |
| `BUS-6014_BBS_3` | 2 | −1 |
| `BUS-6029_BBS_2` | 2 | +72 |

THREE of the four open TWO relocated couplers — chains, which is exactly the gap left open: the zi family's
own ±dummy couplings into a relocated child's rows are not covered, because its alternate channel has Q
slots only (4 per element). A sub-coupler in a chain emits into its parent's dummy rows, so opening the
parent strands it. Extending `buildZeroImpDest` to P slots (and `vdl`/`hv`, which have no alternate channel
at all) is the next increment, and `BUS-6029_BBS_2` is the reproducer to drive it.

#### Slice 2 of the per-scenario placement: the zi family's own ±dummy couplings (2026-09-24)

Three of the four survivors opened TWO relocated couplers, i.e. CHAINS: a sub-coupler emits its ±dummyP /
±dummyQ into its parent's DUMMY rows, so opening the parent stranded it exactly as the bus equations were
stranded before. The zi alternate channel carried Q slots only (4 per element), so those couplings had no
stock placement to fall back to.

Extended to 8 slots per element — Q at 0..3 as before, P at 4..7 — with the same `stockAltRow` fallback and
the same `-2-row` selector (divert only while the parent's dummy row is an identity). `buildZeroImpDest` now
takes `relocOwn`, both zi kernels route `d0/d1/d4/d5` through `ziDynDest` (they previously used `d[...]`
directly), and the pattern union adds the stock positions `(ownRow, dummyPCol)` and `(ownRow, dummyQCol)`.

Cumulative on window [5000,6000) at N=1000, relocation on:

| | start | bus equations (6b2499a) | **+ zi couplings** |
| --- | --- | --- | --- |
| A = contingencies whose violations differ | 11 | 4 | **3** |
| A outside B (invariant: 0) | 0 | 0 | **0** |
| total violation difference | 299 | 93 | **16** |

`BUS-5958_BBS_1` (+24) is fixed. The three survivors are `BUS-6014_BBS_2` (−1), `BUS-6014_BBS_3` (−1) and
`BUS-6029_BBS_2` (+18) — two of them differ by a single violation, which at this point is likelier a
threshold edge than a placement error, and should be checked with the bus-level worst before more code.

Remaining candidates for `BUS-6029_BBS_2`, the only substantial one left: `vdl` (load models) and `hv`
(HVDC) have NO alternate-dest channel at all, so their P/Q at a relocated child cannot fall back; and the
DISTR_Q kind-3 terms carry a dummy column into a DISTR_Q row, which may itself be a relocated home row.

#### The last piece for the window: the relocated DISTR_Q row (2026-09-24)

The bus-level check contradicts the violation counts, exactly as the harness warns
("violation parity is a THRESHOLD test and will report zero while a bus is 0.1 kV out"). After both
placement slices, with the relocation on:

| survivor | violation delta | worst bus diff |
| --- | --- | --- |
| `BUS-6014_BBS_2` | −1 | **4.150 kV** |
| `BUS-6029_BBS_2` | +18 | 0.617 kV |
| `BUS-6014_BBS_3` | −1 | 0.041 kV |

A one-violation difference hiding a 4.15 kV state error, so "likely a threshold edge" was the wrong read —
always take the bus-level worst over ALL contingencies, not the violation count and not the first BUSDIFF
lines.

FAST REPRODUCER (about 2 minutes instead of 10): `OLF_GPU_SA_CTG_ONLY=BUS-6014_BBS_2` with
`OLF_GPU_SA_CTG_KINDS=busbar OLF_GPU_SA_BUSBAR_TOP=100000` and the relocation on reproduces the mechanism at
smaller amplitude (41 mV, violations diff 1) — retaining one contingency retains fewer switches, hence a
smaller tree. Validate finally on the full window.

All three survivors open TWO couplers OF THE SAME substation, and the dump shows why that shape is special:

```
zi#0 COUPLER_1 b1=VL-6014_0 b2=VL-6014_1 pRow2=13044 qRow2=13045   (child P and Q both relocated)
zi#1 COUPLER_2 b1=VL-6014_1 b2=VL-6014_2 pRow2=13046 qRow2=-1      (child's Q is a DISTR_Q)
```

A first hypothesis — that `ZERO_V` collides with the child's `DISTR_Q` on its V row — is WRONG, checked:
`GpuAcRowIndexing.distrQRow(es, controller, ziParentEdge, dummyQCol)` already relocates a relocated child's
`DISTR_Q` onto the parent's `DUMMY_Q` column, "leaving its V row free for that coupler's ZERO_V", and it is
used by both `GpuAcActivationExtractor` and `GpuAcOuterLoopPacks`. So `dummyQ` 13047 hosts that `DISTR_Q`
and the layout is consistent.

THE DEFECT is one step further on: when the scenario OPENS that coupler, the host pins `dummyP`/`dummyQ` to
`DUMMY_TARGET` identities, so the relocated `DISTR_Q` at 13047 is ZEROED and nothing routes it back to the
child's V row — which the vanished `ZERO_V` has just freed. The two placement slices give the bus P/Q family
fills their stock fallback; the `dq` contributions never got one.

Next increment: give `buildDistrQDest` the same `relocOwn` fallback the families have — for any contribution
whose row is a relocated `DUMMY_Q`, an alternate at the child's own V row with the `-2-row` selector, plus the
matching pattern positions — and then `fillDistrQBatch` picks per scenario. `relocOwn` is already built and in
scope. `vdl` and `hv` still have no alternate channel at all and remain the other open candidate.

#### NEGATIVE RESULT: routing a relocated DISTR_Q back to the bus's own V row is WRONG (2026-09-24)

Implemented and measured, then reverted. `buildDistrQDest` gained an alternate set of the same six dests
computed at `relocOwn[dummyQRow]` (the child's own V row) with the `-2-row` selector, the pattern gained the
matching positions, and `fillDistrQBatch` picked per scenario. On the fast reproducer
(`OLF_GPU_SA_CTG_ONLY=BUS-6014_BBS_2`, relocation on) the worst bus difference went from **4.086e-02 kV to
6.023 kV** — about 150x worse — so the premise is wrong, not the wiring.

Why the premise looked right and is not: when the coupler opens, the child's `ZERO_V` does vanish and its V
row is free, so "put the DISTR_Q back there" is the obvious move. But a coupler opening also changes the
GROUP: on the CPU, OLF re-evaluates `updateRemoteVoltageControlEquations` over the live controllers, which
can deactivate that `DISTR_Q` entirely (the controller is no longer connected to the pilot), or the child is
islanded, in which case the equation should be zeroed rather than relocated. Blindly relocating it forces an
equation OLF no longer has.

So the next attempt must first establish what OLF's equation set actually IS for that scenario — dump the
CPU side with `EquationSystem#write` for the post-contingency state of `BUS-6014_BBS_2` and compare against
the device's rows — rather than assuming the row simply moves. `GpuEquationDump` writes both sides already;
what is missing is a dump taken with the CONTINGENCY applied, which is worth adding to the harness before
writing more device code.

The tree is left at the best measured state (`dd32d58`): A = 3, total violation difference 16 on the full
window, invariant intact.

### The CPU post-contingency dump, and what it settles (2026-09-24)

`OLF_GPU_EQ_DUMP_CTG=<contingency id>` (+ optional `OLF_GPU_EQ_DUMP_CTG_FILE`) writes OLF's OWN equation
system for the POST-CONTINGENCY state from `GpuBatchedAcSecurityAnalysis.runPostContingencySimulation`,
where `lfContingency.apply()` has already run — so it is the system the CPU actually solves for that
scenario, inactive equations in `[ ]`, with a bus/branch id legend (OLF writes `<symbol><elementNum>`).
This is the reference the device side was missing, and it should be consulted BEFORE writing placement code:
reasoning about where a row could go produced two wrong fixes in a row.

For `BUS-6014_BBS_2` it reads:

```
# disabled buses: [VL-6014_1]
# disabled branches: [BUS-6014_COUPLER_1, BUS-6014_COUPLER_2, LINE-6136-6014#0]
bus_target_v6281 = v6281              ACTIVE    (VL-6014_0)
[ bus_target_v6282 = [ v6282 ] ]      INACTIVE  (VL-6014_1, disabled)
bus_target_v6283 = v6283              ACTIVE    (VL-6014_2)
[ zero_v9042 ] [ zero_v9043 ]         INACTIVE  (both couplers disabled)
dummy_target_p9042/q9042, p9043/q9043 ACTIVE
```

Two conclusions, one of which is a confirmed post-mortem:

1. It EXPLAINS the failed dq attempt. `VL-6014_2` is PV post-contingency (`bus_target_v6283` active), so it
   has NO reactive equation at all — routing its `DISTR_Q` back to its V row added an equation OLF does not
   have, which is why the error grew 150x.
2. The relocation gave `VL-6014_2`'s V row to `zero_v9043`, which is now inactive, so that row must host
   `bus_target_v6283`. The device left it a power row with nothing emitted into it. Fixed in
   `GpuBatchChunkSolver`'s zi-outage loop: when the opened coupler is tree-relocated (bit 3) and the child is
   still voltage-controlled, its V row is set to mode 0 with the control's target. Verified applied by
   `ZITREE_VID` (`child=VL-6014_2 vRow=12566 applied=true`).

HONEST RESULT: the reproducer is UNCHANGED at 4.086e-02 kV. The change matches the CPU reference and closes
a real hole, but it is not the cause of the residual error, so it is kept on the strength of the reference
rather than of a measurement.

Next diagnostic, and the other half of the comparison: dump the DEVICE's per-scenario ROW MODES for this
scenario (`readBatchRowModes` already exists natively) and diff them against this CPU dump row by row for
12563..12566 and 13044..13047. The second `ZITREE_VID` line is the lead — `VL-6014_1` is the bus OLF
DISABLES, and how the device freezes it (or fails to) is now the main suspect.

### Both halves of the comparison, and what they show (2026-09-24)

`OLF_GPU_ROWMODE_DUMP_CTG=<contingency id>` prints the DEVICE's final per-scenario row modes and targets, as
the rows that DIFFER from the base, from the post-solve readback in `GpuBatchChunkSolver`. With
`OLF_GPU_EQ_DUMP_CTG` (OLF's post-contingency equations) that is the full row-by-row comparison, and for
`BUS-6014_BBS_2` it reads:

| row | variable | device mode/target | OLF post-contingency |
| --- | --- | --- | --- |
| 12564 | `BUS_V(VL-6014_1)` | 0 / **1.0** | bus DISABLED — the flat freeze is the device equivalent |
| 12565 | `BUS_PHI(VL-6014_1)` | 0 / 0.0 | disabled |
| **12566** | **`BUS_V(VL-6014_2)`** | **0 / 0.0** | **`bus_target_v6283` ACTIVE**, target 1.142 pu |
| 12567 | `BUS_PHI(VL-6014_2)` | unchanged (1) | P balance active — the stock fallback returns it |
| 13044–13047 | both couplers' dummies | 0 / 0.0 | `dummy_target_*` ACTIVE |

So the device pins a LIVE bus's voltage identity to **zero** instead of to its control target. The
`ZITREE_VID` trace shows the host writing the right value (`child=VL-6014_2 vRow=12566 target=1.142
applied=true`), and the MODE does land (the row is 0, not its base 1) while the TARGET does not.

Next session starts here, with two candidates and a warning:
1. The override may never reach that row — trace `GpuBatchOverrides` for row 12566 end to end (the
   neighbouring zi-outage overrides write 0.0, which is ALSO the base target for a dummy row, so "13044–13047
   are 0.0" is NOT evidence that those overrides landed).
2. A device kernel may rewrite it afterwards — `readBatchRowModes`' own javadoc warns the outer-loop kernels
   rewrite mode AND target, and this dump is read after the solve.

The warning: a mode-0 identity at target 0 on a live bus should drive its voltage to 0 pu and wreck the
scenario, yet the measured worst bus difference is only 4.086e-02 kV. Either the reported voltage does not
come from that row, or the target is rewritten again before the solve that produces the answer. Resolve that
contradiction BEFORE fixing anything — it is the same trap as the two wrong fixes above, where a plausible
story was acted on ahead of the evidence.

### ROOT CAUSE: a coupler opening DISSOLVES a merged voltage control, which the device does not model

Resolved with the two dumps rather than by reasoning, and it is not a placement problem at all.

The override is not lost. `OLF_GPU_TARGET_ROW=12566` shows the host's value reaching the uploaded snapshot:

```
TARGET_ROW row=12566 base=0.000000000 baseMode=1 mode[s=0]=0 ov[s=0]=1.142000000
  row=12566 mode=0<-1 target=0.00000000        <- post-solve readback: back to the BASE value
```

Why it is reset: the base extraction has ONE merged generator voltage control here —
`pilot=VL-6014_0, pilotIsController=true, members={VL-6014_0, VL-6014_2}` — so `VL-6014_2` is a MEMBER,
its V row carries `zero_v9043`, and its reactive lives in the group's DISTR_Q. Side by side:

| | `BUS_V(VL-6014_0)` | `BUS_V(VL-6014_2)` |
| --- | --- | --- |
| device BASE structure | mode 0 (BUS_TARGET_V) | mode 1, hosts `zi#1.ZERO_V` — a MEMBER |
| OLF POST-contingency | `bus_target_v6281` ACTIVE | **`bus_target_v6283` ACTIVE** |

Those buses are one electrical node through the zero-impedance couplers, so OLF MERGES their voltage
controls. Opening the couplers splits the node, the merge dissolves, and each bus gets its own
`BUS_TARGET_V` while the merged group's DISTR_Q structure disappears. The device's structure is frozen at
extraction with the MERGED group, so post-contingency it carries equations OLF no longer has — a different
equation SET, not a permutation of one, which is why no amount of row re-placement closed it.

It also explains the reset: the host write of mode 0 + 1.142 on the member's row is exactly right, but the
gen-VC kernels still believe the merged group and restore the row to the group's own bookkeeping.

Consequences for the plan:
- This is the SAME shape as the pilot-self rotation (slice 2): `pilotIsController=true` with a member on a
  coupled bus. The two lines of work meet here.
- The fix is per-scenario voltage-control re-merging: when a scenario opens the couplers that merged a
  control, the members separated from the pilot need their OWN `BUS_TARGET_V` and must leave the group's
  DISTR_Q. That is a real feature, not a placement tweak, and it wants its own design pass.
- Until then the relocation's residual on this window is understood and bounded: 3 contingencies, all
  opening two couplers of one substation, all of them merged-control splits.

### A real fix, and the residual turns out NOT to be placement at all (2026-09-24)

FIX (`GpuAcOuterLoopPacks.distrQPack`): the DISTR_Q member bookkeeping row was `columns.vCol()[m]`, the
member's own V column, while under the coupler-tree relocation a relocated child's DISTR_Q lives on its
parent coupler's `DUMMY_Q` — the move `buildDistrQContributions` already makes through `distrQRow`'s tree
overload. The row now follows the relocation. Consequence, measured with `OLF_GPU_TARGET_ROW=12566`:

```
before:  row=12566 mode=0<-1 target=0.00000000      <- DISTR_Q target updater wrote the member's V row
after:   row=12566 mode=0<-1 target=1.14200000      <- the BUS_TARGET_V target the scenario needs
```

So the device's post-contingency equation set for `BUS-6014_BBS_2` now matches OLF's: `BUS_TARGET_V` active
on both `VL-6014_0` and `VL-6014_2`, both couplers' dummies pinned, the member's DISTR_Q gone with them.

AND YET the bus difference is unchanged at 4.086e-02 kV — because the residual was never that row. The
BUSDIFF detail shows the error is NOT at `VL-6014_2` but spread network-wide, uniformly signed:

```
BUS-141_BBS  413.070204 vs 413.111063  (-4.086e-02)
BUS-856_BBS  412.302956 vs 412.332075  (-2.912e-02)
BUS-4563_BBS 241.699616 vs 241.726876  (-2.726e-02)
```

and the cause is in the same log: `distributedActivePower: worst 7.386068e+00 MW` for this single
contingency. A 7.4 MW difference in distributed slack shifts every bus slightly, in one direction — exactly
this signature. The full-window worst of 404 MW is the same defect aggregated.

So the remaining work on this window is the DISTRIBUTED SLACK, not row placement. The lead: this scenario
DISABLES a bus (`VL-6014_1`), which OLF removes from the system entirely while the device FREEZES its rows
at a flat state. A frozen bus still has load and generation in the device's active balance, so the mismatch
it distributes differs from OLF's. Check `GpuBatchQualification`'s scheduled-loss computation
(`scheduledActivePowerLoss`) against `lfContingency.getDisabledNetwork()` for a bus-disabling contingency
before touching anything else — and note the base agrees to 3e-9 MW, so this is contingency-specific.

### THE BUG: a relocated bus's TARGET stayed behind when its equation moved (2026-09-25)

The residual on this window was never a missing equation — it was a missing TARGET. The distributed slack is
applied by shifting a bus's P target, and for a relocated bus that target sits on its parent coupler's
`DUMMY_P`, where its P balance lives. The stock-placement fallback moved the EQUATION back to the bus's own
row when the coupler opens and left the target behind on a row that is now a `DUMMY_TARGET` identity, so the
bus's entire scheduled injection vanished and the slack absorbed it.

Found by discriminating instead of guessing. Same contingency, relocation off vs on:

| | slack difference | worst bus difference |
| --- | --- | --- |
| relocation OFF | 0.023 MW | **6.003 kV** |
| relocation ON (before the fix) | **7.386 MW** | 0.041 kV |

Two different defects, which is why each configuration looked "nearly right" in one metric and wrong in the
other. The scheduled loss itself agreed exactly (both 0.0 pu, `OLF_GPU_LOSS_TRACE` vs the `# OLF
activePowerLoss` line), which is what ruled out the pre-distribution and pointed at the solve.

FIX (`GpuBatchChunkSolver`, zi-outage loop): when an opened coupler is tree-relocated, the child's P target
moves from the coupler's `DUMMY_P` to the child's own PHI row, and — for a child that is NOT
voltage-controlled — its Q target moves from `DUMMY_Q` to its V row. Plus a guard that matters as much as the
fix: SKIP all of it when the child is in the island this outage created, because the freeze already holds its
rows at a flat state and these overrides are applied after it. Without the guard the window went from 3
diverging contingencies to 27 (56 of the 67 coupler-openers island their subtree).

| window [5000,6000), N=1000, relocation ON | placement work start | before this fix | after |
| --- | --- | --- | --- |
| contingencies whose violations differ | 11 | 3 | **1** |
| total violation difference | 299 | 16 | **1** |
| worst distributed active power | 808 MW | 10.8 MW | **0.887 MW** |

The isolated reproducer is now EXACT: `BUS-6014_BBS_2` goes from 7.386 MW / 4.086e-02 kV to 1.11e-09 MW /
3.40e-11 kV, violations diff 0. And the relocation is now BETTER than no relocation on this window (violation
difference 1 against 19), which it was not before.

### Hunting the next one: the base fallback is NOT the pilot-self rotation (2026-09-25)

After the target fix the window sits at 897/1000 with ONE fallback left, and it is the BASE load flow. The
gate is now named (`OLF_GPU_REJECT_TRACE=1`):

```
GPU_REJECT slot=0 reason=max-iter-bad-solve outer=6 it=30 ratio=8.139e+11 norm=2.359472e+02
GPU_BASE_ROBUST_RETRY status=-2 -> mode 6      (the robust retry then gives ratio 6.187e+06, norm 7.6e+71)
```

So the base still meets an unfactorable system at outer pass 6 — the pass where the first mode-4 rows appear
— and even the pivoted, matched single-matrix mode cannot solve it.

TESTED AND REJECTED: that this is the pilot-self degeneracy, whose (property-gated, unfinished) rotation was
built for exactly this shape. Re-running with `-Dolf.gpu.pilotSelfRotation=true` on top of today's fixes —
including the `distrQPack` member-row correction the rotation depends on — leaves the base failing at the
SAME pass with the SAME gate (`max-iter-bad-solve` at outer=6, ratio 4.203e+11 against 8.139e+11), while
adding a second fallback and taking the window to 0/1000. So the rotation neither fixes the base nor changes
its failure, and the earlier hypothesis that the base's outer=6 rank deficiency IS the pilot-self duplicated
DISTR_Q is unproven at best. Keep the property off; do not spend more on the rotation until the base failure
is understood on its own terms.

Also measured, so it is not mistaken for a relocation defect: the residual distributed-power spread (412 of
974 contingencies above 1e-9 MW, worst 0.887 MW) is PRE-EXISTING and slightly WORSE without the relocation
(525 of 974, worst 1.057 MW). It is a general GPU-vs-CPU slack difference, not something today's work
introduced.

Next target, in order of value: the base's outer=6 unfactorable system, using the tools that worked today —
`OLF_GPU_EQ_DUMP` for OLF's base equations at that state against `OLF_GPU_VALS_SCAN`'s mode census, and the
KLU rank test on an `OLF_GPU_VALS_DUMP` of that exact fill. That combination localized every defect this
session; guessing at mechanisms did not.

### The distributed-power spread, part 1: a lost account and a 0.066 MW real difference (2026-09-25)

Acting on "this much difference is a difference in equation between cpu and gpu hence a bug", the spread was
taken apart rather than filed as pre-existing. Two separate defects came out of it, and the previous entry's
"general GPU-vs-CPU slack difference" wording is retracted: one of them is a reporting bug, the other is a
real difference in what the two arms distribute.

FIRST, a method note that cost two wrong fixes. `sort -rn` does NOT understand scientific notation, so a
ranking of differences printed `9.9476e-14` above `1.05` and the real offenders never appeared at the top.
Use `sort -g`. Every conclusion drawn from such a ranking before this was worthless — that is how the earlier
claim "the worst converged difference is 9.5e-05 MW" happened.

**Defect 1 (FIXED): a fallen-back run threw away the device's distribution.** The window's base reported
`cpu=14.359916553 gpu=0.000000000 MW`, the largest single disagreement in the comparison. Traced with a new
`OLF_GPU_DS_ACCOUNT=1` on `GpuDriverOuterLoopEngine.DeviceDrivenDistributionLoop`, on the base alone:

```
check FELL BACK run=23a8deea mismatch=-0.001767686 booked=0.000000000 deviceHas=0.14294298360059193
ask numSC=0 mainSC=0 fellBack=true device=0.000000000 cpu=0.000000000
```

The device was HOLDING 14.294 MW at that very check. The loop read `lastDeviceDistributedActivePower()` only
while `deviceDroveLastSolve()`, so a run whose last solve fell back reported OLF's account alone — and OLF's
account is legitimately 0, because the device had already balanced the network (the mismatch it sees is
-0.18 MW, far inside the 1 MW `slackBusPMaxMismatch` deadband, so it correctly books nothing). The two
accounts must ADD: the device's passes moved the targets IN THE STATE, OLF's loop books only what it shifts on
top. Reading the device's amount unconditionally (it is NaN-guarded and reset per solve, so a pure-CPU replay
cannot double count) takes the base to `gpu=14.294298359`.

Three wrong guesses on the way, all from reasoning instead of measuring: that a mid-run handover was losing
the account (the no-arg `getDistributedActivePower` is never called — OLF only uses the per-SC overload at
`AcloadFlowEngine:331`); that the 10-contingency window proved the fix (it has NO fallback, so it exercised
the already-working path); that the two arms were leaking network state into each other (each arm reads its
own `Network`).

**Defect 2 (OPEN): the device distributes 0.066 MW less than OLF.** What is left on the base is
`cpu=14.359916553 gpu=14.294298359`, diff **6.56e-02 MW** — the SAME magnitude and the same direction as the
per-contingency spread (462 of the 465 significant differences share one sign; the device distributes less in
magnitude). The 15 outliers at 0.5-1.5 MW are this same difference AMPLIFIED by the 1 MW deadband: where the
two arms' mismatch straddles it, OLF fires a pass the device does not, and the device then reports the
pre-distribution alone (`recordDistributedActivePower`: `retain[s]` NaN = "no pass fired"), which is why the
worst offenders' GPU values are round numbers like -19.3 and 31.6.

So the base is now the CHEAP REPRODUCER for the whole spread: one load flow per arm instead of 1000
contingencies, printed by `OLF_GPU_SA_BASE_ONLY=1` (`BASE distributedActivePower:`), ~2 min a run.

Leading hypothesis, under test: OLF's DS loop fires only on a CONVERGED inner solve, while the device's
in-kernel pass fires on its own trajectory. A less-converged state underestimates losses, so the device
distributes slightly less, and the remainder hides under the deadband. Prediction: tightening
`OLF_GPU_DBG_CONV_EPS` on BOTH arms shrinks the 0.066 MW.

**Defect 2 ANSWERED, at the DEFAULT criteria: the device's Newton matches OLF's iteration for iteration,
and the 0.066 MW is a symptom of the base DIVERGING at outer pass 7 and being retried.**

A first answer here leaned on a tolerance sweep (tighten `newtonRaphsonConvEpsPerEq` on both arms, watch the
gap collapse to 1.1e-09 MW at 1e-12). The user rejected that, rightly and emphatically: "with the default
criteria and the default precision both cpu and gpu should agree at machine precision... you should be
FORBIDDEN to change the tolerance as you always do it and it is NOT the answer". The sweep only shows both
arms share a fixed point; it says NOTHING about why they take different paths at the settings that ship, and
reaching for it hid the defect for most of a day. The rule for this repo: a tolerance change is a diagnostic
of last resort, never an answer, and never a recommended workaround.

Measured instead, at the DEFAULT 1e-4, with `OLF_GPU_DBG_NR=1` (new: attaches an appender to OLF's own
`NewtonRaphson` + `MaxVoltageChangeStateVectorScaling` loggers) beside the device's `OLF_GPU_CONV_TRACE`:

| iterate | CPU `\|f(x)\|` | device `norm2` |
| --- | --- | --- |
| x0 | 117.84475774904828 | 1.17844757749047915e+02 |
| 0 | 5.463108973064748 | 5.46310897084124392e+00 |
| 1 | 0.22833996913869184 | 2.28339969965081657e-01 |
| 2 | 0.0024533278213928313 | 2.45332782110154910e-03 (converged) |

The two Newtons are the SAME trajectory — including OLF's `Step size: 0.9345794392523364 (3 dv ...)`
MAX_VOLTAGE_CHANGE clamp at iteration 0, which the device reproduces (its `maxVoltageChangeFactorBatch` takes
the largest |dv| and scales the whole step, algebraically OLF's `min(stepSize, maxDv/|dx|)`). And the DS pass
that fires on that trajectory reads `slackMismatchFromAnchor=0.143599166` pu — the CPU's 14.359916553 MW to
9 digits. Parity, at the default criteria, with no knob touched.

The reported 14.294298359 comes from a SECOND base solve. `[DS-CONV]` shows two complete outer sequences: the
first runs outer=0,1 then DIVERGES at outer=7 (norm2 climbing 2.72 -> 12.78 -> 24.58 -> 36.37 -> 48.22 ->
60.15, monotonic, 30 inner iterations), the base falls back (`fallbacks=1`), and the retry re-derives the
distribution from a different starting state — 2 inner iterations instead of 3, reading 0.142942984. The
0.066 MW is the distance between those two legitimate in-deadband distributions.

So this thread ends where the previous one did, on the SAME open item, now with its blast radius known: the
base's failure at outer pass 7 does not merely cost one fallback, it silently changes the reported
distributed power, and every contingency starts from that retried base state — which is the first thing to
re-measure once the divergence is fixed. Note the failure is a DIVERGING Newton (the norm grows smoothly),
not only the "unfactorable system" the earlier entry recorded from `max-iter-bad-solve`.

### The base's outer-6 divergence: pilot-self members have no Q placement and no q (2026-09-25)

Root cause of the base failure, with direct evidence at DEFAULT settings. The pass sequence
(`OLF_GPU_BATCH_TRACE=1`) against OLF's own (`OLF_GPU_DBG_NR=1`, which now also attaches an appender to
`AcloadFlowEngine`, whose "Start outer loop '<name>' iteration <n>" is the CPU counterpart):

```
device: DS(0) -> DS(1) -> HvdcFreeze(2) -> HvdcLimits(3) -> Monitoring(4) -> RL(5) -> RL(6) pinned=16 iters=30 DIVERGES
OLF:    DS(0) -> RL(1) -> RL(2) -> CONVERGED, distributedActivePower=14.35991655253323
```

The diverging pass is REACTIVE LIMITS, and the two arms pin DIFFERENT SETS: OLF's first RL pass switches **19**
buses PV->PQ, the device pins **16**. The three OLF pins the device misses are
`VL-5997_1`, `VL-6003_1`, `VL-6003_2` — and they are exactly the non-first members of two PILOT-SELF groups:

```
gvc pilot=VL-6003_0 controlled=VL-6003_0 pilotIsController=true qRowAlt=12588 members=3:
    VL-6003_0[vCol=12588 qRowAlt=12588]  VL-6003_1[vCol=12590 qRowAlt=-1]  VL-6003_2[vCol=12592 qRowAlt=-1]

row=12588 BUS_V(VL-6003_0) mode=0 terms=[cb#1921.q2@alt]        <- own reactive balance, so it CAN pin
row=12590 BUS_V(VL-6003_1) mode=1 terms=[dq[k=201 ...], ...]    <- DISTR_Q only, no own q anywhere
row=12592 BUS_V(VL-6003_2) mode=1 terms=[dq[k=207 ...], ...]    <- DISTR_Q only
```

Only the PILOT gets a Q-row placement (`qRowAlt`); every other member has `qRowAlt=-1`. So a member's row
carries DISTR_Q and nothing else, the RL check reads its q from `qBuf[row]` — the DISTR_Q residual, ~0 at
convergence, always inside the limits — and it never pins, while OLF computes the controller's own q and sees
`q=192.10 > maxQ=75.36` (and `q=229.12 > maxQ=119.27`, `q=-82.69 < minQ=0.0`). Nor is there a remote-member q
entry to supply the real value: `OLF_GPU_RMQ_TRACE=12590` prints NOTHING, i.e. `fillRemoteMemberQBatch` never
writes that row, although the context does carry the tables (`CTX_FLAGS ... nRlRemote=75 nRmq=193`, checked —
the "the base has no rmq" guess was wrong).

The device's OWN audit says the same thing without the CPU, which is the cheapest reproduction:
`OLF_GPU_RL_AUDIT=1e-6` returns entries UNPINNED and far outside their limits at the final state, e.g.
`entry=949 vrow=12560 mode=0 q=3.511 min=0.0 max=0.955 over=2.556` (that is `VL-5997_0`). OLF's outer loop
only ends after a check finds nothing to switch, so any such line is a fixpoint the device never reached.

WHY IT DIVERGES rather than merely answering differently: when the pilot pins, its row is taken by its own
pinned balance, and no member can host the group's reactive balance because none has a placement. The group's
balance is then enforced NOWHERE — the documented rank-deficiency shape — and Newton walks away smoothly
(norm2 2.72 -> 12.78 -> 24.58 -> ... -> 293 over 30 iterations), which is what a under-determined direction
looks like, not a hard problem.

This REFRAMES the earlier entry "TESTED AND REJECTED: that this is the pilot-self degeneracy". What that test
actually showed is that the pilot-self ROTATION AS IMPLEMENTED does not fix the base — which is consistent
with this root cause and explains why: the rotation moves the pilot's balance onto the first enabled member's
row, but the members have no Q placement at all (`qRowAlt=-1`) and no q for the RL check, so the rotation
addressed a later step than the one that is broken. The pilot-self SHAPE is directly implicated; keep the
rotation property off, and fix the placement + the q first.

Next: give every non-first member of a pilot-self group its own Q-row placement and an rmq entry, so (a) the
RL check sees each controller's real q and pins the same set as OLF, and (b) a pinned pilot has somewhere to
put the group's balance. Then re-measure the base (`OLF_GPU_SA_BASE_ONLY=1` prints the distributed power, and
the 0.066 MW should go with the retry it comes from).

**FIX 1 (committed): the pin set now matches OLF, 16 -> 19.** The cause was not a missing q and not a missing
rmq entry — both of those guesses were WRONG and are corrected here. `OLF_GPU_RL_TRACE_VROW=12590` shows the
device computing `q=1.921012507` pu for `VL-6003_1`, which is OLF's `192.1012507165038` MW exactly, with
correct limits `±0.7536`. (The earlier "`OLF_GPU_RMQ_TRACE` prints nothing, so the row is never filled" was an
artefact: those are device-side `printf`s that did not reach the log, not evidence of a missing fill.)

What actually dropped the member is one line in `reactiveLimitsBatch`:

```cpp
int pvForm = rlRemote[i] == 2 ? 0 : 1;   // 2 = a PILOT that is one of its own controllers
if (limitType[rst] == 0) {
    if (mode[ridx] != pvForm) return;    // member row is mode 1 (DISTR_Q), pvForm 0 -> returns before reading q
```

and every member of a pilot-self group was tagged `rlRemote=2` with its OWN row as the pilot row. The
classification asked `bus.getGeneratorVoltageControl()` — a merged group keeps ONE **MAIN** control and leaves
every other member holding a **DEPENDENT** control that regulates ITSELF, so `isRemoteMemberEntry`'s
`MergeStatus.MAIN` filter dropped the members (returning false) and `isPilotSelfEntry` then answered true for
each of them. Fixed by resolving the MAIN control (`mainGenVc`, mirroring `isRemoteGenVcControllerElement`
right beside it, which already did this) in both predicates and in the pack's own `controlledBus` / target /
nominal-V reads, plus the same resolution in `GpuBatchRlCurves` (whose comment already noted that "a pilot
whose own control is merged is reached through the main control"). The ENTRY SET is unchanged — only the class
and the pilot row — so the curve enumeration cannot drift.

Measured: the members are now `remote=1 pilotVRow=12560/12588`, and the base pins **19**, the same count OLF
switches in its first RL pass. Suite 299/0, checkstyle 214 (baseline).

**STILL OPEN: the base still diverges at outer=6 and still falls back**, with the base's distributed power
unchanged at 14.294298358 against the CPU's 14.359916553. So the pin PARITY was necessary but not sufficient,
and the remaining cause is most likely the corner the pin parity has now opened for the first time on this
network: `VL-6003`'s three members and `VL-5997`'s two now ALL pin, so each group is FULLY pinned, which is
exactly the case the kernel comments call out ("When the LAST live member pins, `updatePilotVeqBatch` flips
the pilot's V identity to a power row (control lost) and restores it on the first unpin"). Next: check the row
modes of both groups after the outer-6 pass — whether the pilot's V identity flipped, and where the group's Q
balance ended up — before reading anything else into the divergence.

### Iteration parity as the standard, and the divergence narrowed to the post-pin assembly (2026-09-25)

Standing requirement from the user, now the working invariant here: **the GPU and the CPU must agree on the
number of outer-loop iterations AND on the inner Newton iteration count of every solve.** Any difference is a
defect to find, not a tolerance to widen. Measured on the base at default settings
(`OLF_GPU_BATCH_TRACE=1` against `OLF_GPU_DBG_NR=1`):

| solve | OLF inner iterations | device inner iterations |
| --- | --- | --- |
| 1 (initial) | 3 | 3 |
| 2 (after DS distributes) | 1 | 1 |
| 3 (after RL pins) | **2, converges** | **30, diverges** |

OLF's whole base is `outerLoopIterations=3, solverIterations=7` (3+1+2+1). The device matches solve 1 and
solve 2 EXACTLY, and after FIX 1 it also pins the SAME 19 buses — verified by name, not just by count
(`comm` on the two lists: empty both ways). So the two arms are in the same state, making the same decisions,
and then the device's third solve runs away. The fault is therefore in the ASSEMBLY AFTER THE PIN, which is
the narrowest this has ever been.

What the assembly looks like at that pass (`OLF_GPU_EQ_DUMP`, which fires on the first mode-4 pass), for
`VL-5997` whose TWO controllers both pin:

```
row=12560 BUS_V(VL-5997_0) mode=0 terms=[]                 <- still a V IDENTITY
row=12562 BUS_V(VL-5997_1) mode=4 terms=[cb#8699.q2, ...]  <- hosts the PILOT's q (12560's "q2@alt" before)
```

OLF's system once every controller is pinned is: `BUS_TARGET_Q` for `_0` on 12560, `BUS_TARGET_Q` for `_1` on
12562, and `BUS_TARGET_V` INACTIVE (control lost, no enabled controller). The device keeps the V identity on
the pilot row, moves the pilot's q onto the member's row, and so the member's OWN pinned balance is enforced
nowhere — one constraint too many and one too few, which is what Newton walks away from.

The device HAS a kernel for this corner, `updatePilotVeqBatch`: when `relocChoose` finds no unpinned carrier it
flips the pilot's V identity to a power row. It never fires for these groups because they are not in the qr
candidate table (`qRowAlt=12588`, a direct row, not the `-2-group` encoding; `nQr=3`).

RETESTED AND STILL NOT THE FIX: `-Dolf.gpu.pilotSelfRotation=true`, which enrols them (`nQr` 3 -> 30,
`qRowAlt=-19`, pilot q on `qRow=12562`). The base diverges at outer=6 with the same 30 iterations, adds a
second fallback, and row 12560 is STILL `mode=0` — the V identity does not flip. This retest matters because
the earlier rejection of the rotation was run with the members MISCLASSIFIED (FIX 1), so it could have been
void; it is not. Enrolment alone does not make the all-pinned corner work for a pilot-self group.

Next: find why `updatePilotVeqBatch` leaves the pilot row at mode 0 for an enrolled pilot-self group — its
`lost` test is `relocChoose(...) == 0`, and for a pilot-self group candidate 0 IS the pilot's own row, so
"fell back to candidate 0" and "correctly chose the pilot" are indistinguishable there. That conflation is the
most likely reason the flip never happens, and it is a property of the pilot-self shape specifically.

**The all-pinned pilot-self assembly, measured row by row (`OLF_GPU_ROW_PROBE`, rotation ON).** A correction
first: the claim that `updatePilotVeqBatch` "never fires" was read off `OLF_GPU_EQ_DUMP`, which prints the
HOST-side extracted `activation.rowMode()`, not the device's live modes — it cannot see a mid-solve flip. The
probe shows the flip DOES happen:

```
pilot  row 12560: outer=0..5 target=1.103000000000e+00 F=0            (V identity, satisfied)
pilot  row 12560: outer=6    target=-2.556000000000e+00 F=1.736762045429e+00
member row 12562: outer=0..6 target=0                   F=~1.8e-08   (DISTR_Q, satisfied)
```

Two defects in that assembly, and together they are the divergence:

1. The flipped pilot target is **-2.556**, which is `maxQ - q` (0.955 - 3.511, the audit's `over=2.556`
   negated), not the pinned limit `maxQ - loadQ = 0.955`. `updatePilotVeqBatch` writes
   `target[pi] = pilotQ[fi]`, the group's captured relocated-Q target — right for a pilot that is NOT one of
   its own controllers, wrong for a pilot-self pilot that has itself PINNED, whose row must carry its own
   Q LIMIT. The row is left with F = 1.74 pu (174 MW) that no Newton step can clear.
2. The member's row keeps `target=0` through the pass that pinned it, so its own pinned Q limit is never
   enforced: that row is the pilot-Q CARRIER, and the carrier/DISTR_Q target update overwrites what
   `reactiveLimitsBatch` wrote there (`target[ridx] = rliMax - rlLoadQ[rli]`).

So both pinned controllers' Q limits are missing from the system and one unsatisfiable row is present — one
constraint too many and two too few. Newton walks away exactly as observed.

WHAT OLF DOES, and it is the simplest possible assembly: once every controller of the group is pinned there is
no relocation and no carrier at all. Each controller's own `BUS_TARGET_Q` sits on its own row with its own
limit as target, and `BUS_TARGET_V` is dropped (control lost). The device already has the per-row form for
this — mode 4 with the member's own-q ALTERNATE placement — so the fix direction is to make the all-pinned
case take that plain form for EVERY member including the pilot, instead of routing it through the carrier
machinery, whose whole purpose (finding a row for the pilot's relocated Q) is moot when no controller is free.

Rotation stays OFF by default: with it on the base still diverges and takes a SECOND fallback.

### VERIFIED: the pinned pass assembles a SINGULAR matrix, and the defect is named (2026-09-25)

The diagnosis above was checked with instruments that do not depend on it. All of the following is MEASURED,
on the base, at default settings, rotation OFF (the shipping config).

`OLF_GPU_VALS_DUMP` wrote the fill at the pinned pass (`outer=6`, `mode4=6`), and `GpuDumpedMatrixRankTest`
factored it with the CPU's own solver:

```
DUMP n=13228 nnz=93161 mode4Rows=[12560 12562 12588 12590 12592 12834]
KLU REFUSED to factor this matrix: klu_factor error KLU_SINGULAR
```

So the matrix the device assembles once the reactive-limit pass pins is **structurally singular**, which by
that test's own premise is OUR assembly's defect and cannot be rescued by any ordering, pivot type or
matching. This also sharpens the earlier record, which had only "unfactorable, ratio 8.1e+11" from
`max-iter-bad-solve`: the failure is rank, not conditioning. And the six mode-4 rows are EXACTLY the
pilot-self group rows identified from the pin-set diff, plus 12834 (entry 997, which `OLF_GPU_RL_AUDIT`
independently flagged as unpinned-but-over-limit).

`OLF_GPU_VALS_SCAN` at that same pass names the deficiency:

```
dupRows=12590,12592  deadCols=1 firstDeadCol=13185  tinyDiag=1 firstBadRow=13185
modes[0..4]=982,12240,0,0,6
```

and the equation dump identifies what 13185 is:

```
zi#25 branch=BUS-6003_COUPLER_2 b1=VL-6003_1 b2=VL-6003_2 qRow1=12590 qRow2=12592 dummyQ=13185 qAlt=[-1,-1]
```

So the two DUPLICATE rows are the two endpoints of a ZERO-IMPEDANCE COUPLER, and the DEAD column is that
coupler's own `DUMMY_Q`. This is the part the earlier description missed: these pilot-self members are the SAME
electrical node split by switches (`VL-6003_0 -COUPLER_1- VL-6003_1 -COUPLER_2- VL-6003_2`), so their two rows
are distinguishable ONLY by the dummy coupling (+dQ in one endpoint's row, -dQ in the other's). At the pinned
pass that coupling is gone from the values, so the rows collapse into duplicates AND the dummy variable is left
unconstrained — one rank deficiency each, which is why KLU refuses. The invariant the repo already documents
for this ("a DUMMY_Q needs an OPPOSING pair to be determined at all") is broken by the pin, not by the
extraction: the HOST tables still list both couplings
(`row=12590 ... zi#24.q2(-dQ), zi#25.q1(+dQ)`, `row=12592 ... zi#25.q2(-dQ)`).

RULED OUT as the mechanism: `applyRowMode`, which returns immediately unless `mode[r] == 0` and so never
touches a mode-4 row. The coupling is therefore lost during the FILL, in the per-scenario destination
selection (`ziDynDest`), not by the identity stamp.

NOT YET VERIFIED, and the next measurement rather than a thing to reason about: WHICH branch of `ziDynDest`
drops it. Its two-way form is `mode[-2-sr] == 0 ? alt[slot] : frozen`, the group form is a `relocChoose`
mode-scan, and `qAlt=[-1,-1]` on this coupler means one of those branches yields "no destination". A first
reading suggested the pinned mode-4 rows take the `frozen` branch and would KEEP the term, which contradicts
the measured dead column — so the selection must be reached another way (the `sel` row may be the pilot's, or
the group branch may apply). Trace the zi q-slot destination per slot per pass for zi#24/zi#25 before
changing anything: a wrong guess here would move a coupling that is currently correct in every other scenario.

### ROOT CAUSE, measured from the dumped matrix: a pinned member's row has no own-q placement

The dumped CSR at the pinned pass settles it without any further inference (parsed straight out of
`OLF_GPU_VALS_DUMP`'s file):

```
row 12588 (PILOT,  mode 4): 5 nonzero — (1768,-29.752) (1769,-3.635) (12588,27.090) (12589,3.635) (13183,+1.0)
row 12590 (member, mode 4): 2 nonzero — (12590,-1.0) (12592,+1.0)   F=-7.536000e-01
row 12592 (member, mode 4): 2 nonzero — (12590,-1.0) (12592,+1.0)   F=-1.192700e+00
rows 12590 and 12592 literally identical: True
column 13185: slot present in rows 12590, 12592, 13183, 13185 — every value 0.0
```

`F = -target` EXACTLY on both member rows (-0.7536 and -1.1927 are those buses' own maxQ), so no q was
evaluated on them at all: every term the host tables list for those rows — `cb#8717.q2`, `zi#24.q2(-dQ)`,
`zi#25.q1(+dQ)` — is absent from the values. What is left is a single +-1 pair on the two V columns, the same
in both rows, hence the duplicate pair; and the coupler's `DUMMY_Q` column is all zeros, hence the dead
column. The PILOT's mode-4 row, by contrast, is assembled correctly, so this is specific to NON-PILOT members.

The mechanism is the design note on `RmqPack`: "that slot is free because **the main qBuf fill suppresses
member rows**". The suppression is deliberate — it frees the member row's qBuf slot for the diagnostic own-q
that the reactive-limit check reads — but it NEVER LIFTS when the member pins and its row must host its own Q
balance. The pilot escapes because it has an own-q placement (`qRowAlt=12588`); the members have `qRowAlt=-1`.

So the very first hypothesis of this investigation was right about the PLACEMENT and wrong about the symptom.
The members' rows genuinely have no own-q placement; what was wrong was calling their q "invisible" (it is
computed correctly, 1.921012507 pu, by the rmq fill into the same row's qBuf slot). FIX 1 (the MAIN-control
classification) made those members pin for the first time, which turned a latent placement gap into a singular
matrix. Both defects are real and FIX 1 stays: without it the device silently solved a different system from
OLF's (16 pins against 19).

THE FIX: give every merged-group member an own-q placement on its OWN row, selected when that row's mode is 4,
exactly as the pilot already has. Then a pinned member's row carries `q_member - limit_member` (its
`BUS_TARGET_Q`), the coupler's +-dQ couplings land on it again (reviving column 13185), the two rows stop being
identical, and the assembly matches what OLF builds: each controller's own BUS_TARGET_Q on its own row, no
carrier, `BUS_TARGET_V` dropped. Touches `ExtractionColumns` (a member's `qRowAlt`) and the cb/zi ALT dest
tables, which today point a member's q at the PILOT's row (`cb#8699.q2@alt` on row 12560) for the relocation
case rather than back at the member's own row for the pinned case — so the alt channel needs to distinguish
those two destinations, not merely toggle.

### FIX 2: the same MAIN-control bug in the PLACEMENT predicates — the singularity is gone (2026-09-25)

The placement gap has the SAME root cause as FIX 1, in two more predicates.
`GpuAcRowIndexing.isRemoteGenVcMember` and `isFreeRemoteGenVcMember` both filtered
`getMergeStatus() == MAIN` on the bus's OWN control, so a merged member — whose own control is DEPENDENT and
regulates itself — answered "not a member". `ExtractionColumns`'s own-row alternate loop is gated on exactly
that predicate, so those members fell through it AND through the pilot-self loop beside it (whose `MAIN`
filter rejects them too) and kept `qRowAlt = -1`. Fixed by resolving the main control, exactly as
`isRemoteGenVcControllerElement` in the same file already did — and whose javadoc had already recorded this
trap in detail. One definition now: `GpuAcOuterLoopPacks.mainGenVc` delegates to `GpuAcRowIndexing.mainGenVc`.

Nothing else had to change: both the build side (`buildCbDynDest`, whose comment already reads "suppressed
frozen dest (remote member): select on the ALT row itself (mode 4)") and the runtime selection
(`ziDynDest`'s `(mode[sr] == 0 || mode[sr] == 4) ? alt : frozen`) were already written for this case and were
simply never reached.

MEASURED on the base's pinned pass, before -> after:

| | before | after |
| --- | --- | --- |
| `dupRows` | 12590,12592 | **-1,-1 (none)** |
| `deadCols` | 1 (col 13185) | **0** |
| `tinyDiag` | 1 (row 13185) | **0** |
| member row 12590 | 2 nonzero, `F = -target` | **6 nonzero incl. both coupler couplings, F = 1.167** |
| member row 12592 | identical to 12590 | **distinct, 3 nonzero incl. (13185,-1.0)** |
| KLU | **REFUSED, KLU_SINGULAR** | **factors it** |

So the rank deficiency this whole thread was chasing is FIXED: each pinned member's row now carries its own
reactive balance with its own limit as target, which is OLF's assembly. Suite 299/0, checkstyle 214.

STILL DIVERGING, for a DIFFERENT and newly visible reason. KLU factors the matrix but the solve is meaningless
(`||dx||inf=3.196391e+15`, residual ratio 1.518), and the null direction is now a set of zero-impedance DUMMY
columns (13199..13224, all mode 1, diag -1, F ~ 0). Column 13207 is touched by rows **12834 (+1.0) and 12836
(+1.0)** — two couplings of the SAME SIGN, where a `DUMMY_Q` needs an OPPOSING pair to be determined. Its
coupler is `zi#36 BUS-6112_COUPLER_1`, whose `qRows=[12836,-1]` the equation dump ALREADY flags as INCOMPLETE,
and whose bus1 (V row 12834) is another pilot-self bus whose q was relocated onto 12836. So the new member
alternate and the pilot-Q relocation now both write into that coupling, one of them with the wrong sign, and
bus2's `-dQ` is missing altogether. That interaction is the next thing to fix, and the existing
INCOMPLETE-coupling section of `GpuEquationDump` plus a sign check on each coupler's pair is the instrument
for it.

**The remaining defect, named: the RL pin writes mode 4 on the entry's OWN row even when that bus's Q balance
is RELOCATED.** Measured placement for the group behind the same-signed pair (post-FIX 2):

```
gvc pilot=VL-6112_0 vCol=12834 qRow=12836 qRowAlt=-1 members=2:
    VL-6112_0[vCol=12834 qRow=12836 qRowAlt=-1 enabled=false]   VL-6112_1[vCol=12836 qRow=-1 qRowAlt=-1 enabled=true]
zi#36 BUS-6112_COUPLER_1 b1=VL-6112_0 b2=VL-6112_1 qRow1=12836 qRow2=-1 dummyQ=13207
```

This pilot is ALREADY PINNED at extraction (`enabled=false`, its `BUS_TARGET_Q` active), so `computeQRow`
relocates its Q balance to the free member's row (12836) and its own row 12834 keeps `BUS_TARGET_V` — which a
sibling controller still needs. That placement is CORRECT and it is why FIX 2's two loops both skip this group
(the pilot has `qRow >= 0`, the member is the free one). The pilot-self pinned pilot cannot host both
`BUS_TARGET_V` and `BUS_TARGET_Q` on one row, and relocating the Q is the device's answer.

What is wrong is the ROW MODE. `reactiveLimitsBatch` pins by writing `mode[ridx] = 4` with
`ridx = s*n + rlVrow[i]`, and `rlVrow` is "the transition WRITE row: the member's own bus row". For a bus whose
Q balance is relocated to a carrier, mode 4 must go on the CARRIER's row (12836) while its own row stays the
mode-0 `BUS_TARGET_V` identity. Today row 12834 is told "host your own Q" while that q is emitted at 12836, so
the coupler's `+dQ` appears at BOTH rows (measured: column 13207 touched by 12834 +1.0 and 12836 +1.0, no
opposing `-dQ`) and the `DUMMY_Q` is left underdetermined — `||dx||inf = 3.2e+15` on an otherwise factorable
matrix.

So the fix is to make the pin's write row follow the RELOCATION, not the bus: `rlVrow` (or a second per-entry
"pin row") must be `qRow[bus]` when that differs from `vCol[bus]`. Note this is the same quantity
`GpuAcOuterLoopPacks` already computes for the DISTR_Q member table, where the comment records the identical
lesson for the coupler-tree case ("The member's DISTR_Q row, NOT simply its V column ... Pointing this
bookkeeping at the V row made the DISTR_Q target updater write there"). Check both `rlVrow` uses (the pin write
and the `q = qBuf[row] + loadQ` read) before changing either: the READ must stay on the row that carries the
bus's own q, which for a relocated bus is the carrier, so the two may already need to differ from `vCol`.

**WHAT NEEDS TO CHANGE, from the tables themselves.** The free member's row, as the host tables list it:

```
row=12836 BUS_V(VL-6112_1) mode=4 terms=[cb#8783.q1, cb#8925.q2, zi#36.q1(+dQ),
                                         dq[k=271 elem=13207 side=0 kind=3 w=-1.00000],
                                         dq[k=274 elem=13207 side=1 kind=3 w=-0.00000]]
```

`zi#36.q1(+dQ)` is bus1's (the PILOT's) coupling and its frozen dest is 12836, because `qRow1 = 12836` — the
pilot's balance rides the carrier. When the pilot's OWN row (12834) goes mode 4, the same term ALSO resolves
there through the alternate, so column 13207 gets +1.0 twice with no opposing -dQ. The kind-3 `dq` term on the
same column (weight -1.0) is correctly gated off by mode 4, so that gate is NOT the problem — it works.

The emission must happen EXACTLY ONCE, at whichever row hosts that bus's Q balance this scenario, and the
two-way channel can already express that. It simply keys on the wrong row:

```cpp
selRow[o] = stockSel <= -2 ? stockSel : (rc.row >= 0 ? rc.row : altRow);   // frozen exists -> select on the CARRIER
```

For a bus whose alternate is its OWN row and whose frozen dest is a DIFFERENT row (a relocation carrier), the
select row must be the OWN row: mode 4 -> alt (own row), else -> frozen (carrier). One emission either way.
The `sel = altRow` branch already does exactly this when the frozen dest is SUPPRESSED (-1); it needs to
extend to the relocated case. What blocks a one-line change is that the build must know `altRow` IS the bus's
own V row, which the cb/zi index tables do not carry today (they carry the frozen q dest and `qRowAlt`, not
the bus's V column for that side) — so this needs either a new per-side field or an explicit sel shipped from
Java, which is where the next change starts.

Two method notes worth keeping. (1) `GpuEquationDump`'s `gvc` line recomputes `ExtractionColumns.of(network,
es)` AT DUMP TIME, so its `qRow`/`qRowAlt` are NOT the values the running structure uses — reading a pinned-pass
dump as if they were sent this investigation after a placement that was never in play. The dumped MATRIX is
the ground truth. (2) `mode4Rows` differs between the first pinned fill and the end of the run
(`[12560 12562 12588 12590 12592 12834]` then `[12562 12588 12590 12592 12836]`), so always state WHICH fill a
mode observation comes from.

**What the two `rlVrow` uses need, and three refuted explanations for the same-signed pair (2026-09-25).**

The two uses genuinely need DIFFERENT rows, so they must not be moved together:
- the q READ (`q = qBuf[rlVrow] + loadQ`) must stay the bus's OWN row, because that is where
  `remoteMemberQPack` writes the diagnostic own-q (`int row = vCol[bus.getNum()]`);
- the pin WRITE (`mode[ridx] = 4`, `target[ridx] = limit`) must be the row that HOSTS that bus's Q balance,
  which for a relocated bus is the carrier.

`relocatedPinCode` already freezes an entry whose pin lives elsewhere, and it is not firing here for a good
reason: at the extraction that matters the pilot is UNPINNED (`qRow = -1`), so nothing is relocated yet and
there is nothing to freeze. That path is correct.

The decisive new fact, from `OLF_GPU_DBG_ROW=12834` — the packer's own inputs, not a recomputation — is that
the base is extracted TWICE with DIFFERENT placements:

```
#1  b1=VL-6112_0(vCol=12834 qRow=12836 qRowAlt=-1)   b2=VL-6112_1(vCol=12836 qRow=-1 qRowAlt=-1)
#2  b1=VL-6112_0(vCol=12834 qRow=-1 qRowAlt=12834)   b2=VL-6112_1(vCol=12836 qRow=-1 qRowAlt=12836)
```

Each is INTERNALLY CONSISTENT: #1 (pilot pinned at extraction) emits the pilot's `+dQ` at the carrier 12836;
#2 (pilot unpinned, both loops of FIX 2 firing) emits it at 12834 when that row goes mode 4, and correctly
emits no `-dQ` because the free member has no Q equation in OLF either. Neither placement alone can put +1.0
on BOTH rows, which is what the dumped matrix shows.

REFUTED, each by reading the mechanism rather than guessing:
1. "The cached structure aliased a different placement." `GpuAcData.sameStructureInputsAs` compares `cbIdx`,
   `ziIdx`, `dqRow` and the rest — all of which embed `qRow`/`qRowAlt` — so two placements cannot content-match.
2. "A pattern slot kept a stale value after the destination moved." The value array is memset on EVERY full
   fill (`fillActiveBatch(false)`).
3. "Row mode 4 triggers F-only fills that reuse a stale J across the pin." `fOnly = it == maxIter ||
   (ctx->mode == 4 && it > 0)` is the cuDSS SOLVE STRATEGY (line 6159: 0 per-scenario, 1 batched, 2
   block-diagonal, 3 UBATCH, 6 robust), where 4 is the chord strategy factorizing at iteration 0 only. It has
   nothing to do with row mode 4.

So the next measurement is to establish WHICH extraction produced the dumped matrix, which the dump does not
currently say: add the structure identity (or the packer's `qRow`/`qRowAlt` for the traced rows) to the
`VALS_DUMP` header, then re-read column 13207 with that known. Until that is pinned down, any change to the
selection would be aimed at a placement that may not be the one in force.

### The instrument, and a retraction: the zi emission is CORRECT and the imbalance is pre-existing (2026-09-26)

`OLF_GPU_VALS_DUMP_COL=<variable column>` now REPLAYS `ziDynDest`'s two-way choice on the host, from the same
tables and the same per-scenario modes the fill kernel reads, and prints it beside the dumped matrix. It exists
because a dumped matrix alone cannot say WHICH placement built it, and the base is extracted more than once
with different placements. Measured at the pinned fill:

```
VALS_DUMP_COL col=13207 zi=36 dummyP=13206 dummyQ=13207 frozenQ=[-1,-1] ziDyn=1
    slot=0(F) frozen=-1 alt=-12836 sel=12834 selMode=4 -> -12836      (alt encodes -2-row, so row 12834)
    slot=1(F) frozen=-1 alt=-12838 sel=12836 selMode=1 -> -1
    slot=2(J) frozen=-1 alt=89495  sel=12834 selMode=4 -> 89495
    slot=3(J) frozen=-1 alt=89516  sel=12836 selMode=1 -> -1
```

RETRACTED: the previous entry's diagnosis ("the two-way select keys on the carrier, so the +dQ is emitted
twice; make the sel the bus's own row"). It is WRONG. Extraction #2 is the one in force (`frozenQ=[-1,-1]`),
the select rows are ALREADY the buses' own rows (12834 and 12836), and the zi channel emits the coupling
exactly ONCE — at 12834, because that row is mode 4 — and correctly emits no `-dQ`, because the free member
has no Q equation in OLF either. Nothing in the zi selection needs changing, and the per-side "own V row" field
that entry called for is not needed.

So the extra `+1.0` at row 12836 comes from ANOTHER channel (by elimination, the DISTR_Q `kind=3` term the host
tables show at that row: `dq[k=271 elem=13207 side=0 kind=3 w=-1.00000]`).

What the null direction looks like, against its healthy analogues — this is the shape to fix:

```
col=13207 touchedBy=3 colSum=+1.0  (12834 mode=4 +1) (12836 mode=1 +1) (13207 mode=1 -1)   <- one too many
col=13209 touchedBy=2 colSum= 0.0  (12840 mode=1 +1) (13209 mode=1 -1)                      <- healthy
col=13199 touchedBy=2 colSum= 0.0  (12734 mode=1 +1) (13199 mode=1 -1)                      <- healthy
```

A healthy coupler dummy column carries `+1` in ONE bus row and `-1` in the dummy variable's own row, summing
to zero. 13207 carries a third entry and sums to +1.

TWO further negatives, both measured:
- **`-Dolf.gpu.pilotSelfRotation=true` does not remove it.** The qr gate in `fillDistrQBatch` that deactivates
  the re-picked free member's DISTR_Q (OLF's "DISTR_Q for every enabled controller except the FIRST", which
  after the pilot pins should strip VL-6112_1's) does not fire even with the group enrolled: the column is
  unchanged, and the base still diverges with a second fallback. Rotation stays OFF.
- **The imbalance is PRE-EXISTING and independent of FIX 2.** Column 13207 is byte-identical in the pre-FIX-2
  dump and the post-FIX-2 dump — same three entries, same `colSum=+1`, same modes. FIX 2 repaired the
  VL-6003/VL-5997 rows (dupRows, dead column, KLU_SINGULAR) and left this coupler exactly as it was. So the
  remaining `||dx||inf = 3.2e+15` is an OLDER defect that the singularity was masking, not a regression.

Next: extend the same replay to the DQ channel (dest, elem, kind, weight, and whether each gate fires) for the
chosen column, which names the third writer directly instead of by elimination.

### The third writer, NAMED, and a one-line clobber fixed (2026-09-26)

The DQ half of the replay (same `OLF_GPU_VALS_DUMP_COL`) names it directly, no elimination — `OLF_GPU_DQ_TRACE`
could not, because it is a DEVICE printf and device printf output never reaches this harness's log (the same
reason the earlier `OLF_GPU_RMQ_TRACE` looked empty and sent this investigation down a wrong path):

```
DQ c=265 kind=3 elem=13207 w=0.818182 fRow=12836 fRowMode=1 jMember=58 qrGroup=-1 carrier=-1 psSkip=0
        gateMode4=0 gateCarrier=0 -> EMITTED
DQ c=268 kind=3 elem=13207 w=0.181818 fRow=12836 fRowMode=1 jMember=59 qrGroup=-1 carrier=-1 psSkip=0
        gateMode4=0 gateCarrier=0 -> EMITTED
```

Two DISTR_Q `kind=3` contributions at row 12836, weights **0.818182 + 0.181818 = 1.000000** — exactly the extra
`+1.0` on column 13207. They are the group's two reactive keys. Per OLF, once the pilot pins, `VL-6112_1` is
the FIRST enabled controller and must LOSE its DISTR_Q; the device keeps it because neither gate fires
(`gateMode4` cannot: that row is mode 1; `gateCarrier` cannot: `qrGroup=-1`).

**FIX 3 (one line): `ExtractionColumns`'s pilot-self alternate loop clobbered the qr enrolment.** Its guard was
`qRowAlt[bn] < 0`, and a group encoding is `-2 - groupId`, which is also negative — so the loop overwrote the
enrolment the loop above had just made, silently, and only for pilots whose q is suppressed at extraction,
which is exactly the unpinned pilot-self case the rotation exists for. `== -1` is the correct test. This is why
`-Dolf.gpu.pilotSelfRotation=true` had NO observable effect on this group in every earlier test: the property
enrolled the group and the next loop un-enrolled it.

Measured with FIX 3 + rotation ON: the DQ contributions now report `qrGroup=24 carrier=12836`, so the
enrolment reaches them; one base attempt converges in **4 inner iterations instead of 30**; and the base's
distributed power moves from **14.294298358 to 14.381569177** against the CPU's 14.359916553 — note 14.3816 is
essentially the 14.381763 both arms reach when fully converged, so the device's base is now solving the right
system far better than before.

STILL NOT DONE, and the next link is exact: the carrier gate is disabled by `psSkip=1`, which asserts "this
group's pilot is NOT pinned this scenario" while `mode[12834] = 4` says it is. `pilotSelfGateBatch` derives it
from `limitType[psPilotEntry[g]] == 0`, the flag IS recomputed from current pin states before every fill, and
`psEntry[g]` is matched by the pilot's own V row — so the RL bookkeeping (`limitType = 0`, unpinned) and the
row mode (4, pinned) genuinely disagree for that entry. Resolve which of the two is wrong before touching the
gate: if the row mode is right, the pin never booked a limitType; if the limitType is right, something set
mode 4 without pinning. Rotation stays OFF by default and FIX 3 is inert without it (a non-pilot-self pilot
fails that loop's `controlledBus == bus` test, so nothing else was being clobbered). Suite 299/0, checkstyle 214.

### Taken from `injection-scope`: a curve-limit parity fix (2026-09-26)

Fetched `origin/injection-scope` (3 commits, branched from the same `main` as this branch) and took all three;
they cherry-pick cleanly, the test file auto-merging with this branch's harness edits.

- `45ce19c` **Move a bus limit only for the generators that actually set it** — a real parity fix in
  `GpuBatchRlCurves.packBusCurves`. `AbstractLfBus.getLimitQ` takes a generator's own limit only in VOLTAGE or
  REMOTE_REACTIVE_POWER control and its fixed targetQ otherwise, so a non-controlling machine is a CONSTANT in
  the bus's Q box; the curve channel packed every participating curve generator and the kernel then moved the
  box by a slope OLF holds still. Comes with its own regression test.
- `f7da13c` its `docs/numbers.md` row, `75a6f81` a harness change (compare one contingency inside the scope
  that produces it).

MEASURED AGAINST OUR CASE: it does NOT touch the outer-6 divergence. With the fix in, rte6515's busbar base is
unchanged — `gpu=14.294298358` against `cpu=14.359916553` (diff 6.56e-02 MW), one fallback, 30 inner iterations
at outer=6, and the two DISTR_Q kind-3 contributions on column 13207 still `EMITTED` with `qrGroup=-1`. Worth
having on its own merits (it is a different defect, on `network.xiidm`'s battery curve), not a lead for this one.

Suite 300/0 with its new test. Checkstyle went 214 -> 215 on the imported test (a double blank line triggering
`noConsecutiveLines`); collapsing it gives **213**, one under the previous baseline, so the baseline for this
branch is now 213.

### Settled: the two bookkeepings AGREE; the device and OLF disagree on the PIN (2026-09-26)

RETRACTED, by measurement: the previous entry's "the RL `limitType` and the row mode genuinely disagree for
that entry". They do not. A new line in the dump replay prints the gate's own inputs per pilot-self group, and
every one of the 27 groups is internally consistent:

```
PS g=24 pilotEntry=997 limitType=0 fallbackRow=12834 rowMode=0 psSkip=1
counts: 21 x (limitType=0, rowMode=0, psSkip=1)   4 x (limitType=2, rowMode=4, psSkip=0)   2 x (limitType=1, rowMode=4, psSkip=0)
```

Zero groups have `rowMode=4` with `psSkip=1`. The claim came from MIXING CONFIGURATIONS: `mode[12834]=4` was
measured with the rotation OFF and `psSkip=1` with it ON, and the placement differs between them (rotation ON
keeps the pilot's row a mode-0 identity and routes its q to the carrier). That is the FOURTH time a
cross-configuration mix-up has produced a wrong conclusion here — always state the config with the number.

The real disagreement is device-against-OLF, on the PIN itself:

```
RL_VROW s=0 entry=997 lt=0 q=1.218000000 minQ=-0.774 maxQ=0.774 mode=0 loadQ=1.218000000
CPU_RL  Switch bus 'VL-6112_0' PV -> PQ, q=91.05402338186495 > maxQ=77.4        (0.9105 pu against 0.774)
```

The device's `q` equals its `loadQ` EXACTLY, so `qBuf[12834] = 0`: the bus's own reactive injection never
reaches the RL check. OLF computes 0.9105 pu, so `qBuf[12834]` should be about -0.3075. With the rotation OFF
the entry pins anyway — 1.218 and 0.9105 are on the same side of 0.774, a right outcome from a wrong number —
and with it ON the entry never pins at all (no `WB_RL` line).

WHY the q is missing, and why this is probably a CONSEQUENCE rather than a second independent defect: the RL
diagnostic refill deliberately uses the FROZEN routing (`RelocTables{}`, "the relocation only concerns the
Newton residual, not this diagnostic Q"), and this pilot's frozen q dest is -1 (suppressed), so only the rmq
twin can supply its q. It HAS an rmq entry (`RMQ_WHY ... skipped=-`), and the terms that entry sums for this bus
are the coupler's `DUMMY_Q` contributions — `dummyQ` being exactly the variable left underdetermined by the
extra `+1.0` on column 13207. An underdetermined dummy gives a meaningless q, which gives a wrong pin decision.

So the root remains the `DUMMY_Q` determinacy, and the thing to remove is still the DISTR_Q `kind=3` pair at row
12836 that OLF drops once the pilot pins (`w=0.818182 + 0.181818 = 1.0`). Next: make that member's DISTR_Q
inactive for the scenario — the qr carrier gate is the intended mechanism and it is now reachable (FIX 3), but
it is correctly held off by `psSkip` while the device believes the pilot is unpinned, which is itself downstream
of the wrong q. Break the loop at the q: give a pilot-self entry a diagnostic q that does not depend on the
dummy it is meant to help determine.

**No duplicated term, and the gap is the DUMMY_Q's VALUE on the UNPINNED base (2026-09-27).**
`OLF_GPU_VALS_DUMP_ROW=<row>` lists every remote-member-q contribution summing into a row's diagnostic slot.
For the pilot:

```
RMQ row=12834 c=144 kind=0 elem=8783 side=0 w=1
RMQ row=12834 c=145 kind=0 elem=8925 side=1 w=1
RMQ row=12834 c=146 kind=3 elem=13207 side=0 w=1 x[elem]=-0.417689
RMQ row=12834 contributions=3
```

REFUTED: the factor-2 reading of "device 0.603 / OLF 0.9105 / loadQ 1.218 are equally spaced by 0.3075" as a
doubled contribution. There are exactly three terms and no duplicate.

So `qBuf[12834] = q(8783) + q(8925) + x[13207]`. The device's early q of 0.603 gives `qBuf = -0.615`, so the
branch pair sums to -0.1973; OLF's 0.9105 gives `qBuf = -0.3075` and therefore implies `dummyQ = -0.1102`,
against the device's **-0.417689**. The entire 0.3075 pu gap is the coupler `DUMMY_Q` variable's VALUE.

AND THAT MATTERS FOR WHERE TO LOOK: this q is read on the CONVERGED UNPINNED base — `mode4Rows` is empty
there, `dupRows=-1,-1`, `deadCols=0`, and KLU factors it. So the wrong dummy is NOT downstream of the pinned
pass's rank deficiency, as the previous entry concluded; the two arms disagree about this coupler's reactive
through-flow in a system that is perfectly well determined. The later `q = 1.218` exactly (`qBuf = 0`) is the
pinned pass's singular state and is the symptom, not the cause.

The chain, now end to end: the device's `DUMMY_Q` for BUS-6112_COUPLER_1 is 0.3075 pu off OLF's on the unpinned
base -> the pilot's diagnostic q reads 0.603 against a 0.774 limit instead of OLF's 0.9105 -> the device does
not pin VL-6112_0 where OLF does -> with the rotation ON the group never reaches the pinned arrangement at all.
Next: compare that one coupler's through-flow against OLF on the unpinned base — the ZERO_V/ZERO_PHI pair and
the two endpoint Q rows for zi#36 — which is a well-determined system, so a discrepancy there is a plain
assembly difference and not a rank question.

### THE FIX IS FOUND AND MEASURED — but it over-reaches, so it is not committed (2026-09-27)

The controlled run first CORRECTS the previous entry. In the SHIPPING config (rotation off) the device's
diagnostic q for the pilot is RIGHT and so is its pin:

```
RL_VROW entry=997 lt=0 q=0.910540234 minQ=0 maxQ=0 mode=0            (limits not yet refreshed)
RL_VROW entry=997 lt=2 q=0.910540234 minQ=-0.774 maxQ=0.774 mode=4   <- pins, at OLF's pass
CPU_RL  VL-6112_0 q=91.05402338186495 > maxQ=77.4                     (0.9105402338 pu)
```

Nine digits against OLF. So the "0.603 against 0.9105, a 0.3075 pu gap in the DUMMY_Q" of the previous entry is
RETRACTED: 0.603 came from the rotation-ON run (a different trajectory) and 1.218 from AFTER the blow-up
(v=1.855). Third comparison across configurations to produce a wrong conclusion; the shipping config is the
only one whose numbers mean anything unless the property is the subject.

WHAT THE PIN ACTUALLY DOES, which is the whole defect: unpinned, the coupler's DUMMY_Q column is BALANCED
(`col 13207 = (12836,+1.0), (13207,-1.0)`, colSum 0 — the 12834 slot is wiped by the mode-0 identity, exactly
like the healthy `col 13209`). Pinning makes 12834 mode 4, so the pilot's `+dQ` now lands there — WITHOUT
removing the member's DISTR_Q `+dQ` at 12836. OLF SWAPS the two (activates the pinned pilot's `BUS_TARGET_Q`,
deactivates the DISTR_Q of the controller that becomes FIRST ENABLED); the device only ADDS. colSum 0 -> +1.

**THE FIX THAT WORKS.** `qrGroupOfPilot` exists precisely to separate "this pilot has a candidate table" from
"route this pilot's Q through the carrier" (its own comment says so). Enrolling pilot-self groups for the TABLE
ONLY — `groupOfPilot[bus] = g` without `qRowAlt[bus] = -2 - g` — makes `fillDistrQBatch`'s carrier gate
reachable, and on rte6515's busbar base:

| | before | after |
| --- | --- | --- |
| the two kind-3 contributions | `gateCarrier=0 -> EMITTED` | `psSkip=0 gateCarrier=1 -> SUPPRESSED` |
| outer pass 6 | 30 iterations, diverging | 1-5 iterations, converging |
| fallbacks | 1 | **0** |
| base distributed power | gpu=14.294298358 (diff 6.56e-02 MW) | **gpu=14.359916553, diff 3.46e-10 MW** |

That is the machine-precision agreement at DEFAULT settings that this whole thread has been chasing, with no
knob touched.

**WHY IT IS NOT COMMITTED:** the table also switches on the other group-keyed kernels —
`updatePilotVeqBatch` (flips a pilot's V identity when the mode scan finds no carrier) and
`updateDistrQTargetsBatch` (hands the pilot's Q target to the carrier row) — and three tests fail:
`deviceMiddleSectionLossRenormalizesDistrQTargetMatchesCpu` and
`deviceMiddleSectionLossWithShuntOnIslandMatchesCpu` (bus vl3_b 401.69025542177314 -> 401.36950114185817) plus
`GpuVsCpuRte6515SaTest`'s own DAP assertion on the default branch scope (worst 1.2115417243381827 MW). Suite
300/3.

ALSO TRIED AND REJECTED, measured: implementing OLF's "all enabled except the FIRST" directly in
`fillDistrQBatch` — suppress a contribution whose row-owner member is the group's current first-enabled,
derived from the `(mRlEntry, limitType)` data the dynamic keys already read. It needs no table, fixes the
divergence (outer 6 back to 2 iterations) but leaves the base at 14.405802445 (diff 4.59e-02 MW) with one
fallback, and it is far worse elsewhere: **suite 300/37**, mostly "the batch must ENGAGE ... census={}". The
predicate is wrong — at extraction the host has already removed the first-enabled member's DISTR_Q, so
recomputing "first enabled" per scenario suppresses rows that legitimately carry one. Reverted.

NEXT, and it is narrow: keep the table-only enrolment and find which of the two other group-keyed kernels
breaks the middle-section tests, then gate THAT one on the group not being pilot-self. The two failing tests
name the machinery directly (DistrQ TARGET renormalization), so `updateDistrQTargetsBatch` is the first
suspect: for a pilot-self group whose Q is NOT re-routed, handing the pilot's Q target to the carrier row
writes a target that belongs nowhere.

### The table-only enrolment, taken apart on the middle-section fixture — and it regresses the batch (2026-09-27)

Bisecting the three failures of the table-only enrolment with a per-group mask (`OLF_GPU_PS_MASK`: bit 1
skips `updateDistrQTargetsBatch`'s carrier hand-off for pilot-self groups, bit 2 skips `updatePilotVeqBatch`)
gave an answer I did not expect: the two middle-section tests fail IDENTICALLY under every mask
(`vl3_b 401.36950114185817` at 0, 1, 2 and 3), while the base stays fixed under every mask (mask 3:
`diff=5.55e-11 MW`, 2 iterations, no fallback). So neither of those kernels is what breaks the fixture, and
the base fix comes purely from the DISTR_Q carrier gate.

The fixture has 16 rows, so the two arms were compared vector by vector at the contingency batch's first fill
(the dump now records `nS`, selects a later context with `OLF_GPU_VALS_DUMP_NTH` — `it` runs on across
contexts, so "first fill of a solve" is `outer==0` after a non-zero outer — and writes `x` and `target`):

```
row  mode  F(with)        F(without)     x(with)     x(without)  target(with)   target(without)
 10  1/1   2.488544e-01   3.155211e-01   1.0125      1.0125      -0.133333333   -0.200000000   <-- TARGET differs
```

Identical Jacobian, identical incoming state, only the TARGET of the surviving member's DISTR_Q row differs —
and `-0.2` is OLF's post-contingency `distr_q5 = -0.5*q(bus5) + 0.5*q(bus3)` (weights renormalized 1/3 -> 1/2
over the two survivors). `OLF_GPU_TARGET_ROW=10` then named the producer: with the table, row 10 receives TWO
overrides, `-0.133333333` (the base value) and `-0.200000000` (the renormalized one), and the base value wins;
without it, only `-0.2`. The extra one is `GpuBatchChunkSolver`'s carrier re-pick on islanding
(`overrides.add(s*n + newCarrier, baseTargets[baseCarrier])`), which moves "the pilot's relocated balance" from
a frozen base carrier to the next live member — correct for a re-routed group, wrong for a table-only one,
which carries no relocated balance at all.

FIX 4 (in the tree, not committed): a per-group `qrRerouted` flag, computed in `ExtractionColumns` where the
decision is made (1 for transformer/shunt groups and re-routed gen-VC groups, 0 for a table-only pilot-self
group), carried on `GpuAcData` beside `qrPtr`/`qrRows`, and honoured by that block. Both middle-section tests
pass with it (2/0).

BUT THE TABLE REGRESSES THE BATCH, measured, and this is the decisive fact of the day:

| scope | committed tree | table-only enrolment (+FIX 4) |
| --- | --- | --- |
| branch scope, on device | 9040 / 9042 | 9034 / 9042 |
| branch scope, DAP | 725 differ, **worst 7.3e-07 MW** | 952-1156 differ, **worst 1.0-1.2 MW** |
| branch scope, violations | **diff 0** | diff 399 |
| busbar window [5000,6000), on device | 900 / 1000 | **0 / 1000** (`RUNTIME:status=-2=976`) |
| busbar base | falls back, 6.56e-02 MW | no fallback, 3.5e-10 MW |

So the enrolment fixes the one base and breaks nearly everything else: the branch scope was essentially
PERFECT at the committed tree (DAP to 7e-07 MW, zero violation difference), and the window goes from 900 to
ZERO scenarios on the device — the outcome the standing directive forbids. The table has too many consumers,
and the ones the mask covers are not the ones doing this (the base is unaffected by them). The runtime
rejections' reasons (`OLF_GPU_REJECT_TRACE`) are the next measurement; until they are known, the enrolment
stays OUT of any commit. What IS established and stays: the base's defect is the DISTR_Q of the controller that
becomes first-enabled when the pilot pins, and the carrier gate that deactivates it is the right mechanism —
it needs a way to reach pilot-self groups WITHOUT the rest of the table's machinery.

**The window's rejections named, the two device consumers masked, and the decision (2026-09-27).**
`OLF_GPU_REJECT_TRACE` on the full window with the table: the batch churns to **outer pass 36** —
`974 x max-iter-bad-solve outer=36`, `62 x pilot-self-mixed-pin outer=35`, one `outer-loop-budget outer=29`
— a batch-wide outer-loop runaway that a 50-scenario range never reaches (49/50 on device at mask 0, 50/50 at
mask 3). Masking the two device consumers (`OLF_GPU_PS_MASK=3`) on the full window restores the batch and
keeps the base fix, but not the rest:

| full window [5000,6000) | committed tree | table + FIX 4, mask 3 |
| --- | --- | --- |
| on device | 900 / 1000 | 972 / 1000 |
| base distributed power | falls back, diff 6.56e-02 MW | diff 2e-09 MW, no fallback |
| outer passes | normal | still churns to 35-36 (`pilot-self-mixed-pin` x 71) |
| violations | diff 19 | **diff 163** |
| DAP worst | 1.057 MW | **4.470 MW** |
| branch scope (9042) | diff 0, worst 7.3e-07 MW | diff 399, worst 1.0 MW |

So the table enrolment half-enables the UNFINISHED pilot-self rotation (the `pilot-self-mixed-pin` gate
exists precisely because the device cannot yet hold a pilot-self group with some controllers pinned and some
not), and the pin/unpin cycle it starts costs more than the one base it fixes. REVERTED: the table-only
enrolment and the `qrRerouted` plumbing (FIX 4) are out of the tree; the diagnostics stay (`VALS_DUMP_ANY`,
`VALS_DUMP_NTH`, `VALS_DUMP_ROW`, the `x`/`target` lines in the dump, the `PS`/`DQ`/`DQROW`/`RMQ` replays, and
the `OLF_GPU_PS_MASK` gate in the two kernels, inert without a pilot-self table).

What stays established, all measured: the base's defect is exactly "a pilot-self pilot pins, and the
controller that becomes first-enabled keeps a DISTR_Q that OLF deactivates"; the DISTR_Q carrier gate is the
right effect; the mixed-pin state that follows needs the rotation FINISHED (the pilot's pinned balance moving
onto the first enabled member's row while the pilot's row keeps BUS_TARGET_V), not the table alone. FIX 4's
separation of "has a table" from "Q re-routed" is correct and will be needed when that work resumes;
`GpuBatchChunkSolver`'s carrier re-pick and the two device consumers must all honour it.

### The rotation path never pinned its pilot because it counted the pilot's q TWICE (2026-09-27)

Under `-Dolf.gpu.pilotSelfRotation=true` the pilot's diagnostic q read 0.603080467 where OLF has 0.910540234,
and the arithmetic on those measured numbers is exact: `1.218 (loadQ) + 2 * (0.910540234 - 1.218) = 0.603080`.
The injection is counted twice. `patchFrozenGroupDests` points a group-relocated output's frozen role-0 dest at
the table's FALLBACK row so that the diagnostic fill (frozen routing) puts the pilot's own injection there — and
for a pilot-self group the fallback row IS the pilot's own row, which ALSO has an rmq twin entry
(`remoteMemberQPack` includes a pilot that is a controller). A non-pilot-self pilot has no rmq entry, so only
this shape doubles. FIX 5: `remoteMemberQPack` skips a pilot-self pilot whose `qRowAlt <= -2`. Measured under
the property: `q=0.910540234`, pins at OLF's pass with `lt=2`. Inert with the property off (suite 300/0).

With the q right, the rotation path reaches the state its gate describes and the dump shows the remaining
defect exactly: at the pinned pass column 13207 is BALANCED (the DQ pair `SUPPRESSED`, the zi coupling routed to
the carrier 12836), but the pin's mode/target landed on the pilot's OWN row —

```
row 12834 (pilot):   mode=4 target=-0.444  F=4.44e-01  nonzero=[(12834,-1),(12836,+1)]   <- no q content: F = -target
row 12836 (carrier): mode=1 target= 0.000  F=-4.18e-01 nonzero=[(12834,-1),(12836,+1),(13207,+1)]
```

— while its q content went to the carrier, which is solved against target 0. The `rlPilotSelfMixedGate`
comment names the layout OLF has here: "keeps the pilot's V identity AND activates the pilot's own Q balance,
hosting it on the reference member's row — a relocation this group has no table for". Under the property the
table now exists, so the remaining change is the pin itself: leave the pilot's row a V identity, put
`limit - loadQ` on the carrier, keep the carrier's DISTR_Q gated (already), and stop rejecting the mixed state.

### The rotation finished: the pilot's pin rides the carrier, "control lost" is pilot-self aware, the base matches OLF's outer sequence (2026-09-27, rotation ON)

All under `-Dolf.gpu.pilotSelfRotation=true`, window [5000,6000) busbar base (the WINDOW changes the base network —
busbar contingencies retain different switches — so `5000:5001` is another base: 14 pins, no pilot-self churn;
every number below is the [5000,6000) base). Method: `OLF_GPU_DBG_RL=1` (CPU per-bus PV<->PQ decisions, TRACE) beside
`OLF_GPU_SW_TRACE=0` (every device entry that switched, with q/min/max/v/vt), plus two new device traces:
`PSG` (per pass, every pilot-self group: pilot pin state, own-row mode/target, `veqOff`, every candidate row with
its mode and the relocChoose pick — needs `OLF_GPU_BATCH_TRACE`) and `OLF_GPU_PSG_STEP=<g>` (the same for one group
after EACH kernel of the RL pass, so a row change is charged to the kernel between two lines).

Four defects, each measured then fixed:

1. **The pin landed on the pilot's own row** (`reactiveLimitsBatch` mode 4, no q content, F = −target, the
   `pilot-self-mixed-pin` reject). Now an ENROLLED pilot (`isEnrolledPilot`: `psEntry[g] == i`) pins WITHOUT
   touching its row; `updateDistrQTargetsBatch`'s carrier hand-off writes `limit − loadQ[pilot]` on the carrier
   when the pilot is pinned (falls through to the normal DISTR_Q target when unpinned); `rlPilotSelfMixedGate`
   skips enrolled pilots. Group VL-6112 (g=24): pilot row 12834 stays `mode=0 target=1.055`, carrier 12836
   `mode=1 target=-0.444`, q = 0.774 = OLF's limit from the pinned pass on.
2. **`updatePilotVeqBatch` toggled its own verdict.** A pilot-self pilot is merged controller #0, so its OWN row
   is candidate 1 of its group: once flipped to a power row (mode 1) `relocChoose` read it as a live member and
   the next pass restored it — groups VL-5997 (g=17) and VL-6003 (g=19) went mode 1/0 every pass to the
   outer-loop budget (`PSG`: `pick=1` iff `mode=1`). And the flipped row's target was `pilotQ` = 0 (the group's
   captured target) where OLF's BUS_TARGET_Q is the pilot's LIMIT. Now for a pilot-self group: lost = pilot pinned
   AND every OTHER candidate row in {0, 4}; flip target = `limit − loadQ` by limitType (re-stamped on drift);
   an unpinned pilot is never "lost". g=19 holds q = 0.747700000 = maxQ from the pinned pass on.
3. **Write-back froze the V target as a Q limit.** `writeBackReactiveLimits` read `deviceTargets[vRow] + loadQ`;
   for an enrolled pinned pilot that row holds the VOLTAGE target: VL-6112_0 frozen at 1.055 + 1.218 = 2.273
   against 0.774 → the CPU-side criterion saw the residual and the base was `MAX_ITERATION_REACHED` with DAP 0.
   `pinnedBalanceRow` repeats the device's pick from `readBatchRowModes`: own row unless it is mode 0 and a
   qr group has it as fallback, then the first other live candidate (the carrier).
4. **Kernel order inside the RL pass.** `updateDistrQTargetsBatch` ran before `updatePilotVeqBatch`, so the
   carrier hand-off picked the carrier while the pilot's row was still flipped (= the pilot's own row) and the
   member that had just switched back kept a DISTR_Q target of 0; the fills then routed the pilot's balance to
   that row against 0, forcing q_pilot = 3.511 instead of 0.955 and the member's q to −1.655 < minQ 0 — re-pin,
   unpin, re-pin until maxSwitch (device outer 7..10, CPU none). The flip/restore now runs first.

Result, base [5000,6000), rotation ON vs OLF (`CPU_RL` / `CPU_NR`):

| pass | CPU | device |
| --- | --- | --- |
| initial NR | 3 it | 3 it |
| DistributedSlack | 1 it | 1 it |
| RL#1 | 19 PV→PQ (same 19 buses), 2 it | 19 pins, **3 it** |
| RL#2 | VL-6106_0 PV→PQ, VL-5997_1 PQ→PV, 1 it | same two switches, 1 it |
| RL#3 | stable | stable |
| result | CONVERGED | CONVERGED, no fallback, no reject |
| distributedActivePower | 14.359916553 MW | 14.359852249 MW (diff 6.4e-05) |

Same outer sequence, same pin set, same decisions. The two remaining differences are one thing: the device iterates
to 1e-8 pu infnorm where OLF stops at ITS criterion (default `DefaultNewtonRaphsonStoppingCriteria`) — one extra
Newton iteration at RL#1 and a DS amount read off a deeper-converged slack mismatch (6.4e-5 MW of 14.36; OLF's
own residual at its stop is up to 1e-4 pu). Machine-precision parity and equal inner counts need the device to
apply OLF's stopping test, not a tighter one — the next item, and NOT a tolerance change (the standing rule).

Kept, env-gated: `PSG`, `OLF_GPU_PSG_STEP`. Edge noted: `updateDistrQTargetsBatch`'s hand-off takes the nearer limit
for realistic-V pins (3/4) — `rlInitTgtQ` is not in that kernel's tables (the flip in `updatePilotVeqBatch` has it).

### The rotation's first fill: the gate ran after the fills it gates (2026-09-28, rotation ON)

The smallest witness of the base's x0 difference (device |f(x0)| 117.908 vs OLF 117.845) is
`GpuLoadFlowTest.remoteVoltageControlWithPilotAsController` (4 buses, 8 rows; failed under the property).
Method: `OLF_GPU_F0_TRACE=<path>` (new: OLF's own f(x0) − target per equation, with type/element/id) beside the
device's first fill (`OLF_GPU_VALS_DUMP_ANY` + `OLF_GPU_VALS_SCAN`), rotation ON and OFF. Seven rows equal to the
digit; row 0 — the free member's row hosting the pilot's DISTR_Q (`distrQRow`) — had F −1.451 against OLF's
−5.715 and no angle columns at all. The DISTR_Q pack was identical in both configs (`DQROW2`, a standalone row
replay that no longer needs zero-impedance branches), so the extra came from another kernel: the pilot's RAW q
(+1.0 · q_pilot on the carrier; the θ columns cancel against the −0.75 weighted terms, F −5.715 + 4.264). Cause:
`pilotSelfGateBatch` (psSkip) was launched just before the DISTR_Q fill, AFTER the closed/open-branch and shunt
fills that route the pilot's raw q through it — so the FIRST fill of every context read an unset gate. Moved to
the head of the fill lambda. Measured: the fixture's first fill equals OLF's f(x0) to the last digit
(6.635650652927396 vs …395) and the test passes on device; rte6515's base [5000,6000):

| pass | CPU | device |
| --- | --- | --- |
| initial NR | 117.84475774904828 → 5.4631090 → 0.22833997 → 2.4533278e-3 (3 it) | 117.844757749047915 → 5.4631090 → 0.22833997 → 2.4533278e-3 (3 it) |
| DistributedSlack | 2.8654457e-5 (1 it) | 2.8654458e-5 (1 it) |
| RL#1 | 0.063687079 → 3.3781509e-5 (2 it) | 0.063687079 → 3.3781509e-5 (**2 it**, was 3) |
| RL#2 | 1.5247732e-3 (1 it) | 1.5247732e-3 (1 it) |
| distributedActivePower | 14.359916553 MW | 14.359916008 MW (diff 5.4e-07, was 6.4e-05) |

Same outer sequence, same inner counts, residuals to 8–9 digits. OLF's `|f(x0)|` of ~31 at each outer-loop
START is its own print (it precedes the target refresh) — the post-step norms are what agree. What remains is
the last-digits drift (8–9 significant digits at every iterate, 5.4e-7 MW on the DS amount): a linear-solve /
summation-order class, next.

Also fixed: the two chain-step harnesses (`GpuZeroImpedanceControllerChainTest`, `GpuZiCouplerTreeRelocationTest`)
called the legacy `GpuAcNewtonSolver.solve` overload that ships NO group table — with an enrolled group the q
alternates carry −2−g and structure creation refused them ("bad cbIdx.qAlt … −2 (allowed [−1, 16))"); they now
pass the packs' tables. Still open under the property: the batch's seeding of a pilot pinned IN THE BASE
(`relocatedPinCode` marks relocation only for `qRowAlt == −1`, so an enrolled pinned pilot gets mode 4 stamped on
its own V row while the table routes its q to the carrier — the window's 976 × `LINEAR_SOLVE_FAILED`), and the
four `GpuBatchedAcSecurityAnalysisTest` pilot-group fixtures.

### The batch under rotation: the activation stamped mode 4 on the CARRIER (2026-09-28, rotation ON)

The [5000,6000) window under the property put 0/1000 on device — every scenario `LINEAR_SOLVE_FAILED` at outer 0,
it 1 with `||J·dx − B||∞ = ||F||∞ = 2.556` (an EMPTY row whose residual is exactly −target: VL-5997's carrier).
The slice knobs cannot reproduce it (`OLF_GPU_SA_MAX_CTG`, a shorter `CTG_RANGE`: the busbar contingency list
decides which switches the base network retains, so a smaller list is ANOTHER base — 5000:5001 has 14 pins,
5000:5050 has nQr=5 and no pilot-self pin); the full batch's first fill had to be dumped (`OLF_GPU_VALS_DUMP_NTH=2`
with `_ANY`, 3 MB): rows 12562 (VL-5997's carrier) and 12836 (VL-6112's) came in at **mode 4** with the pilot's
`limit − loadQ` as target and only the ±1 zero-impedance coupling as content; `relocChoose` skips mode 4, the
pilot's raw q and the couplers' DUMMY_Q fell to the identity pilot rows and vanished — dead columns 13171 and
13207, singular. Two seeding defects had been fixed first and were NOT it (they stay, both keyed on `psEntry`):
`initRlLimitTypeBatch` now takes an enrolled pilot's base pin from OLF's own kind (its V row is never mode 4),
`seedRlModeFromState` no longer stamps its row, and both hand-off kernels (`updateDistrQTargetsBatch`,
`updatePilotVeqBatch`, now lambdas) run once after the seeding so the first fill sees the seeded pins on the rows;
`updatePilotVeqBatch` books a group that lost control IN THE BASE as flipped (`veqOff`) so a later switch-back
restores the V identity. The cause was `GpuAcActivationExtractor`: "a pilot that is one of its own controllers
gets mode 4 whenever its own Q balance is active" — right for the pre-rotation alternate-dest layout, wrong for an
ENROLLED group whose pinned balance is a POWER row (the carrier, or its own row once lost) that the scan routes to.
Now `!enrolled` (the same predicate `ExtractionColumns` enrols with).

| [5000,6000), rotation ON | before | after |
| --- | --- | --- |
| on device | 0 / 1000 (976 × LINEAR_SOLVE_FAILED) | **966 / 1000** (9 × max-iter-bad-solve, 1 × outer-loop-budget, 24 unmappable) |
| base | CONVERGED on device (DAP diff 2.7e-07 MW) | same |
| violations | — (all replayed on the CPU, diff 0) | CPU 71553 vs GPU 71601, diff 48 |
| distributedActivePower | — | 482/974 > 1e-9 MW (the last-digit class, ~1e-7), worst **1.63 MW** |

Rotation OFF on the same window: ~900/1000, violations diff 19, DAP worst 1.06 MW — the rotation now covers more
scenarios than the fallback path and its remaining parity gap is of the same order; the per-contingency work
(the 1.63 MW worst, the 48 violations) continues. Still failing under the property: the two middle-section
fixtures (`deviceMiddleSectionLoss*`, bus-mid 401.69 vs 401.37 kV — a section loss inside a pilot-self group),
and the two chain-STEP harnesses, which go through the legacy non-batch `solveContext` path where the pilot-self
gate (`psSkip`) does not exist — the same chain network through the LoadFlow (batch-of-one) path passes
(`GpuLoadFlowTest.zeroImpedanceControllerChain`, new).

### The chain-step harnesses: the legacy solve path had no pilot-self gate (2026-09-28, rotation ON)

`GpuZiCouplerTreeRelocationTest` / `GpuZeroImpedanceControllerChainTest` (one Newton step against OLF's own
Jacobian) failed under the property by 1.9e-3 while the same chain network through the LoadFlow path stepped
exactly (`GpuLoadFlowTest.zeroImpedanceControllerChain`, new). The harnesses use `GpuAcNewtonSolver.solve`, the
non-batch `SolverContext`, which already passes `withPilotSelf(...)` to its fills but never had `hasPilotSelf`
or `dPsSkip` set — so an enrolled pilot's raw q was routed to the carrier on top of its DISTR_Q terms. Now
`createContext` derives the pilot-self groups from the candidate table alone (the fallback row reappears among
the member candidates) and sets a constant skip of 1 for them: that path has no reactive-limits pass, so such a
pilot is unpinned for the whole solve. Both harnesses pass under the property (2/2).

### The middle-section fixtures: the carrier re-pick stamped a member's DISTR_Q target as a pilot Q (2026-09-28, rotation ON)

`deviceMiddleSectionLoss*` (bus-mid 401.69 vs 401.37 kV under the property). OLF post-contingency
(`OLF_GPU_EQ_DUMP_CTG=bus-mid`): `distr_q3` (the pilot's) inactive, `distr_q5 = −0.5·q5 + 0.5·q3` — the
survivor at weight 1/2. The device's first fill had the same row with the 0.5 weights but the BASE target −0.1333
(1/3), corrected to −0.2 only by the RL-pass hand-off, after which nothing re-solved. `OLF_GPU_TARGET_ROW=10`:
rotation OFF uploads one override (−0.2, the outage seeding's renormalized target); rotation ON uploads TWO
(−0.1333 then −0.2) and the first survives. The extra one is the chunk solver's carrier re-pick (plan-B4: "a
frozen base carrier's pilot Q moves to the next live member's row"), which exists only when a qr group exists:
for a PILOT-SELF group with its pilot UNPINNED the carrier hosts its OWN DISTR_Q, so the base carrier's target is
that member's DISTR_Q target, not a pilot Q. Now skipped for such a group (pilot-self ⇔ the fallback row reappears
among the candidates; unpinned ⇔ OLF's base kind of its class-2 RL entry is not 1..4). Both fixtures pass
(2/2). Note: `GpuBatchOverrides.add` APPENDS duplicates (`duplicateAppends`) and the device kept the first —
worth a hard error some day; today the wrong producer is removed instead.

### OLF re-merges voltage controls per zero-impedance network — now emulated; rotation ON by default (2026-09-28)

Two of the window's worst contingencies, run alone (`OLF_GPU_SA_CTG_ONLY`, both reproduce): BUS-6029_BBS_1 (DAP
1.632 MW, bus diff 5.3 kV at the surviving VL-6029 sections, identical pins and limit updates on both arms, the
pre-distributed loss equal to 2e-14 pu) and BUS-6002_BBS_2 (29 phantom violations, 8.7 kV at VL-6002_2, NOT
rotation-specific). `OLF_GPU_EQ_DUMP_CTG` names it: OLF keeps `bus_target_v` ACTIVE on the surviving section
(VL-6029_1 at 394.82 kV) with no DISTR_Q — `LfZeroImpedanceNetwork.updateVoltageControlMergeStatus` re-runs on the
post-contingency zero-impedance networks: within each, the controls are re-merged with the highest target (then
controlled-bus id) as MAIN, so a section whose controlled bus is lost, or that a coupler outage separates,
re-elects its own main; a single-bus cluster with a local control is a plain PV bus at its own target. The
device applied "control dies → survivors PQ at targetQ" (right only for a survivor that regulated the LOST bus
remotely — `deviceBusLossIslandingMergedPilotSurvivorMatchesCpu` still holds).

`GpuBatchQualification.reHomedControls` now replays that merge (islanded buses + `outagedZiBranches` split the
group's controlled buses into clusters; a cluster without the base main re-elects; one LOCAL control → V identity
at its target via `GenOutage.ROW_V_IDENTITY`, its controllers leave the group's DISTR_Q through the RENORMALIZE
channel; several controls or a remote controller → `PLAN:VC_REMERGE_MULTI`, CPU). Two device details followed:
`updateDistrQTargetsBatch` wrote the member's DISTR_Q target over the re-homed row's VOLTAGE target at the first
RL pass (1.039 → 0.000, MAX_ITERATION) — an identity-mode member row now never receives one; an extra "inactive
member" early return was tried and dropped (it moved `javaOuterLoopModeMatchesCpu` by 1.1e-12 kV).

| | before | after |
| --- | --- | --- |
| BUS-6029_BBS_1 | DAP 1.632 MW, 5.3 kV | DAP 1.5e-08 MW, worst bus 8.9e-11 kV, violations equal |
| BUS-6002_BBS_2 | 29 phantom violations, 8.7 kV | violations equal, worst bus 8.6e-11 kV, DAP 6e-10 MW |
| window [5000,6000) rotation ON | 966/1000, violations diff 48, DAP worst 1.63 MW | **955/1000** (16 × VC_REMERGE_MULTI to the CPU by design, 21 runtime), **violations diff 0**, DAP worst 1.34 MW |
| window rotation OFF | ~900/1000, diff 19 | 903/1000, diff 145 (61 × pilot-self-mixed-pin, 112 × max-iter-bad-solve — OFF's own defects, two contingencies with 0 device violations) |

Rotation ON is better on every count (base parity, coverage, violations) and the suite is green with it, so
`olf.gpu.pilotSelfRotation` now defaults to TRUE (`-Dolf.gpu.pilotSelfRotation=false` keeps the old layout).
Open: the 16 multi-control re-merges (a new remote group after the split — needs per-scenario group tables), the
21 runtime rejects (8 × max-iter-bad-solve, 1 × budget), the DAP tail (1.34 MW worst), the ulp class (~480
contingencies between 1e-9 and 1e-6 MW), and a re-homed PV bus's reactive limits (its RL entry stays a remote
member: OLF checks it as a local PV bus and may pin it; the device does not).

### A re-homed PV bus keeps its reactive limits (2026-09-28)

BUS-6113_BBS_1 (DAP 1.33 MW, 7.1 kV at BUS-6113_BBS_2): the re-homed survivor VL-6113_1 is pinned by OLF at RL#1
(q = −323 < minQ −77.4) while the device never looked — its reactive-limit entry is packed as a REMOTE member of
the base group and the RL kernel drops a remote member whose row is an identity. Now a re-homed control's entry
travels as a (slot, entry, own V target) triple (`GenOutage.reHomesRlEntry`, `setBatchRlLocal`, `dRlLocalOv`/
`dRlLocalVt`): `reactiveLimitsBatch` takes the LOCAL path for it (pin on its own row, switch-back on its own
voltage at its own target) — and pins with mode 4, not 1: a base-packed member's own q reaches its row only
through the ALTERNATE dest, a mode-1 row is empty (the first try failed its linear solve and the CPU replay
produced the "parity"; the census line is what showed it — `RUNTIME:status=-3=1`, 0/1 on device). Measured:
1/1 on device, worst bus 4.0e-11 kV, DAP 6.5e-9 MW, violations equal. Window [5000,6000): 956/1000 on device,
violations diff 0, DAP worst **0.141 MW** (was 1.34; BUS-6003_BBS_3's 0.60 went with it), 20 runtime rejects.
The seven `injection-scope` commits (base-lost pilot restore through the pilot's local entry, no DISTR_Q share
seeded onto a base-pinned member, loop/iteration traces addressed by contingency) are cherry-picked; the
base-lost restore and the pilot-self `veqOff` booking coexist (`pe < 0` selects the former).

### The window's runtime rejects, named (2026-09-28)

`CHUNK_REJECT` (new, under `OLF_GPU_REJECT_TRACE`) maps every rejected slot to its contingency; the [5000,6000)
busbar window batches 960 contingencies in ONE chunk (976 batchable − 16 `VC_REMERGE_MULTI`), and its 4 chunk
rejects are: BUS-536_BBS (outer-loop budget; the CPU does not converge it either), BUS-6079_BBS (`max-iter-bad-solve`
at outer 0 — OLF's OWN Newton oscillates there, 16.978 → 4.720 → 1.191 → 1.060 → 2.528 → 0.980 → 2.909 …, and
the device reproduces that trajectory to 10+ digits: both arms MAX_ITERATION_REACHED, nothing to fix),
BUS-6003_BBS_2 (slot 797, churn to outer 36: the base's pilot-self VL-6003 group) and BUS-6088_BBS (slot 893,
churn to outer 36). The last two, like BUS-612_BBS_2 (the DAP tail's 0.141 MW), are at machine precision when
run ALONE — the single-contingency base retains different switches — so they are batch-only and must be traced
in the window by SLOT (`OLF_GPU_LOOP_TRACE=<slot>`, `OLF_GPU_SW_TRACE=<slot>`, one per run; slots are stable for a
fixed list and chunk size) with `OLF_GPU_SA_BUSDIFF_CTG=<ids>` for the in-scope bus diffs. `OLF_GPU_LOOP_TRACE_CTG`
now says when the named id is not batched at all (a plan reject — BUS-6002_BBS_2 is one of the 16).

### A group lost in the base is booked at seed time, not in the RL phase (2026-09-28)

BUS-6003_BBS_2 in the [5000,6000) batch (slot 797, `max-iter-bad-solve` at RL#2): at RL#1 the pilot-self pilot
VL-6003_0 UNPINS (v 1.134 > vt 0.993, its group lost in the base) and the re-solve fails in 30 iterations. The
`veqOff` booking that lets `updatePilotVeqBatch` restore the V identity on that unpin is done at the post-seed
launch — where the kernel returned early on `!rlActive[s]` (the RL phase is not active at seed time), so the
pilot's row stayed a power row while its raw q was gated off (unpinned ⇒ `psSkip`): an empty row, a bad solve.
The booking now runs regardless of `rlActive`; only the flip/restore/re-stamp branches stay RL-phase gated.
Measured: slot 797 converges (RL passes 36–39, status 0); window 957/1000 on device, violations diff 0, DAP
worst 0.141 MW; fixtures 177/0 both ways; suite 303/0. Chunk rejects left: BUS-536_BBS and BUS-6079_BBS (OLF
does not converge them either — the device reproduces OLF's divergence to 10+ digits) and BUS-6088_BBS (slot 893).
Batch observation: BUS-536's non-convergence drives the DistributedSlack loop for 30 batch passes (`nChanged=1`
for 28 of them), each a full-width re-solve — the budget is per scenario and correct, but the pass count is a
cost the CPU never pays; a scenario whose solve has failed should stop driving the loop.

### The SA extraction must be CANONICAL — the base pin state leaked into the batch structure (2026-09-28)

BUS-6088_BBS (slot 893 of the [5000,6000) busbar window, `max-iter-bad-solve` at RL#2): the CPU and the device
take the same RL#1 decision (pin VL-5997_1, the sibling of the pilot-self pilot VL-5997_0), and the device's re-solve
then fails because the just-pinned member's row 12562 (mode 4) assembles EMPTY (pre-repair nnz=0, F=−target) while
the same pinned row is FULL in the base at outer 6. The per-row scatter replay (`OLF_GPU_VALS_DUMP_ROW=<row>`, the
`CBROW` lines: every closed-branch output whose frozen or alternate dest is the row, with the select row's mode, the
element's disabled flag and its relocation group) showed the two contexts hold DIFFERENT structures: in the base,
row 12562 is the ALTERNATE of the member's two lines (8568/8700, `mode 4` selects it); in the batch, it is the
FROZEN dest of the pilot's line 8699 under relocation group 17 and the member's lines have no route to it at all.
`OLF_GPU_SA_EQ_DUMP=<path>` (the SA extraction's row map, same format as `OLF_GPU_EQ_DUMP`) named the cause: the
batched SA extracts AFTER the base load flow has pinned buses, and `computeQRow` / the DISTR_Q pack / the member
alternate loop all read that state — a pinned pilot-self pilot has BUS_TARGET_Q active, so it fell into the
"regular pilot" branch (its q frozen onto the first ENABLED member's row, 12562), the member became the "free"
member whose own control's controlled bus is itself (never in the pilot set) and lost its alternate, and the DISTR_Q
row on 12562 became the PILOT's equation (weights −1/0) instead of the member's (+0.5/−0.5). The kernels' rotation
dynamics assume the base layout, which is only canonical by luck (nothing is pinned when the base extracts).

Fix: a group with a candidate table (`GpuAcRowIndexing.hasCandidateTable`, now the ONE enrolment predicate
ExtractionColumns uses) is encoded as if every controller were enabled — every controller element's frozen q row
suppressed (own row through the alternate, mode 4), a non-controller pilot's q on the FIRST member (re-picked per
scenario by the mode scan), the DISTR_Q pack's free controller = the first member, `distrQHostMember` = the first
member; `GpuAcData.sameStructureInputsAs` now includes the candidate tables. The per-scenario side stays
state-based: `GpuAcActivationExtractor` puts an ACTIVE Q balance on `GpuAcRowIndexing.stateQRow` (a pinned member's
own V row; the pilot's on the first ENABLED member's row, or its own row once every controller is pinned) — the
canonical row collided with a pinned first member's own Q (12 remote-group fixtures failed, `assigned twice`) and a
pinned pilot-self pilot indexed −1 (the whole precompute fell to the CPU: `GPU_PRECOMPUTE_FAILED` now prints its top
frames unconditionally). Measured: window 958/1000 on device (BUS-6088_BBS now at 1.2e-8 MW; only OLF's own
non-convergences BUS-536_BBS and BUS-6079_BBS remain), violations diff 0, DAP worst 0.141 MW (BUS-612_BBS_2,
unchanged); suites 304/0 with the rotation on and off; base 3/1/2/1 iteration-for-iteration.
Observed on the way: the BASE's distributed active power differs from the CPU's by a run-dependent amount with
identical code (1.1e-8, 4.2e-6, 2.3e-7 MW in three runs; cold=warm within one JVM) — the last-digit drift
(8–10 agreeing digits per iterate) amplified through the DistributedSlack loop. When it lands at 4.2e-6 the whole
window shifts with it (468 contingencies move from ≤1e-6 into the 4.25e-6 class). That drift is the open ulp item.

### The DISTR_Q weight override builder corrupted the network's reactive percents (2026-09-28)

The [5000,6000) window's distributed-active-power tail (0.141 / 0.042 / 0.026 / 0.0076 / 0.0037 MW …) had ONE
mechanism behind most of it. `GpuBatchQualification.distrQWeightOverrides` toggles a contingency's lost generators
OFF, calls `updateReactiveKeys`, re-extracts the whole network and diffs `dqWeight` index by index against the SA
extraction (`OLF_GPU_DQOV_TRACE=<ctg>` lists each override with its members, `DQOV_PACK` the two packs' metadata at
each differing index). It called `updateReactiveKeys` on each touched bus's OWN control — for a member that is the
DEPENDENT (merged) control whose element list is just the member, so its reactive percent became 1.0, and the
restore in `finally` repeated the same call: the corruption stayed in the LfNetwork and ACCUMULATED across the
qualification (differing entries 0, 10, 15, 27, 35, 45, 51, 56 … as the contingency list went on). Every later
contingency then carried bogus overrides for every pilot-self group an earlier one had touched: on BUS-612_BBS_2
(slot 945) the VL-5967 DISTR_Q row became the pilot's raw q alone (weights 1.0 / 0 instead of 0.5 / −0.5; the
row's first-fill residual −0.782 at the base solution, `DQROW2` and `CBROW ... lt/psSkip` cleared the gate and the
keys as suspects), the group's Q split was (1.694, −1.574) against OLF's (0.912, −0.782), and the RL check pinned
both members plus four other groups' pilots (`SW_PASS`, act=1) where OLF pinned none. Fix: `updateMainReactiveKeys`
— the key update on the MAIN control, as OLF does. Measured: no override for an unrelated contingency any more
(`differing=0`); window 958/1000 on device, violations diff 0; DAP tail above 1e-4 MW from 11 contingencies to 4
(BUS-536 = OLF's own non-convergence, BUS-612_BBS_2 0.141 unchanged, BUS-6112_BBS_2 0.0037, BUS-5997_BBS_2
0.0024), above 1e-6 from 19 to 8. BUS-612_BBS_2 has a second cause (its first post-contingency solve takes 3 scaled
steps on the device against OLF's single step) — the next trace.
Lesson: a diagnostic that re-extracts the live LfNetwork must leave it EXACTLY as found; a diff against the SA
extraction turns any leftover into overrides on unrelated scenarios. Diffing counts (`DQOV_PACK differing=`)
that grow with the contingency index are the signature.

### An islanded member's frozen row took a survivor's DISTR_Q target (2026-09-28)

BUS-612_BBS_2 (slot 945), the DAP tail's 0.141 MW after the reactive-key fix: the device's first post-contingency
solve took three MAX_VOLTAGE_CHANGE-clamped iterations (norms 1.168 → 0.700 → 0.233 → 1.7e-4) where OLF took one
(30.48 → 0.0045), so the distributed-slack decision was taken at another residual level and the two arms distributed
0.141 MW apart while every bus agreed to 1.6e-4 kV. `OLF_GPU_TARGET_ROW=1324` (VL-612_1's V row, an islanded
bus) showed TWO overrides for the slot: the frozen flat 1.0 and a later 0.0 that won — a survivor's renormalized
DISTR_Q target from `islandedDistrQEntries`, whose inactive set held only the split's and the re-homing's controllers
(VL-612_2, the re-homed one) and not the ISLANDED packed members themselves, so VL-612_1 counted as a survivor and
its own (frozen) row got the target. The identity row then drove that dead bus's V from 1.0 to 0 in clamped steps,
scaling the whole step with it. Fix: the islanded packed members join the inactive set, and no entry may target a
frozen row (`GenOutage.withoutFrozenTargetRows`). Measured: slot 945 first solve 1 iteration to 4.5005e-3 (OLF
4.500510e-3), DS solve to 1.8959e-6 (OLF 1.8958917e-6), bus diff 1.2e-10 kV; window 958/1000 on device, violations
diff 0, DAP worst 3.7e-3 MW (BUS-6112_BBS_2, then BUS-5997_BBS_2 2.4e-3; the tail above 1e-4 MW is those two plus
OLF's own non-convergence BUS-536), above 1e-6 8 contingencies.
Lesson: two overrides on one (slot,row) is always a bug; `OLF_GPU_TARGET_ROW` names both writers. A frozen row
belongs to a bus OLF removes from the system — nothing else may write it.
Suites after this fix: rotation ON 304/0; rotation OFF 303/1 once — `javaLoadBasedDistributedSlackMatchesCpu`
missed its 1e-12 kV bus-voltage assertion by 1.02e-12 kV (133.19640118822167 vs …065) and passed on two
immediate reruns: the run-to-run last-digit drift, not this change (no island, no frozen row in that fixture).
It is the same open ulp item as the base's run-dependent distributed power; the tolerance stays.

### The two tail contingencies left after the 2026-09-28 fixes (measured, open)

- **BUS-5997_BBS_2** (slot 787, DAP 2.4e-3 MW, worst bus 7.8e-6 kV): it islands VL-5997_1 — the only sibling of the
  base-pinned pilot-self pilot VL-5997_0 — and removes GEN-5997#0. The row layout is right (`OLF_GPU_ROWMODE_DUMP_CTG`:
  the pilot's own row flips to its Q balance at 0.955 − 3.511, the sibling's rows and the coupler dummies are
  identities; OLF's post-contingency dump has `bus_target_q[6280]` active and `bus_target_v6280` inactive, the same
  shape). The SEQUENCE differs: OLF's first solve takes two iterations (0.0898, then 1.25e-4) and its DistributedSlack
  fires at iteration 0 (the lost generation), the device's first solve converges in one (2.7e-3), its slack loop does
  not fire until pass 40 (remainder 1.36 MW) and its first reactive-limit check switches something OLF never does
  (`LOOP_PASS switched=1` at pass 35, no `SW_PASS` line — the switch trace did not name it). The 2.4e-3 MW is the
  slack-loop remainder landing differently. Next: name the pass-35 switch (RL_TRACE by row), then the first solve's
  per-equation residual against OLF's (the F0 port in the post-contingency dump evaluates a stale state — 82.2 vs
  OLF's 30.48 — it needs OLF's own NR-start evaluation).
- **BUS-6112_BBS_2** (slot 924, DAP 3.7e-3 MW, worst bus 4.2e-6 kV): first solve, slack distribution and HVDC pass
  match OLF (3.365e-4 vs 3.373e-4 after the HVDC re-solve); the reactive-limit passes do not. OLF's RL 2 pins seven
  buses (VL-6490_0, 6489_0, 6055_0, 6090_0, 6139_1, 6139_3, 6231_0) and adjusts VL-6226_0's limit, RL 3 pins
  VL-6139_2 and VL-6139_4; the device pins the same buses EXCEPT VL-6055_0 at RL#1 (row 12710, a local PV entry) and
  catches it at RL#2 — OLF sees q = 190.24 MVAr against a 152.4 limit there, a 38 MVAr excess the device's q at
  RL#1 cannot have shown. Next: VL-6055_0's device q at RL#1 (`OLF_GPU_RL_TRACE_VROW=12710`).
  Measured (`OLF_GPU_RL_TRACE_VROW=12710`, prints `RL_VROW`): at the device's first reactive-limit check VL-6055_0
  reads q = 1.4744 pu (base 1.4757; limits ±1.524) where OLF reads 1.9024 at its first check — after the device's
  own RL#1 pins and re-solve it reads 1.8995 and pins. So the two arms' states BEFORE the first reactive-limit
  check differ by ~43 MVAr at that bus while the final buses agree to 4.2e-6 kV: the first-solve residuals are
  close (0.196 / 7.9e-5 against OLF's 0.2124 / 1.456e-4) but not equal, and the slack pass distributes 21.4 MW
  on the device. Next: an intermediate-state comparison (the device's x at the RL#1 check against OLF's state
  before its RL 2 — OLF logs no intermediate state, so a CPU-side hook on the outer-loop check is needed).

### The pilot-self gate did not exist when reactive limits are disabled (2026-09-28)

Isolating the remaining tail with `OLF_GPU_SA_NO_RL=1` (reactive limits off on BOTH arms) exposed a defect of its
own: the GPU arm's BASE load flow reported MAX_ITERATION_REACHED where the CPU converged, so the whole run
produced no post-contingency results (`pre-ctg: CPU=CONVERGED GPU=MAX_ITERATION_REACHED`, 976 status mismatches,
violations 0 vs 75,735 — the vacuous-agreement shape the census exists to catch). The device's first residual was
|f(x0)| = 117.908 against OLF's 117.845, the double-counted pilot-q signature.
Cause: the PILOT-SELF gate table (`psPilotEntry` → `psSkip`) is derived inside `createBatchContext`'s
`if (nrl > 0)` block, and it marked a pilot-self group whose pilot has NO reactive-limit entry with `-1`, the same
value as "not a pilot-self group". Both readings are backwards: a pilot that can never be PINNED controls for the
whole solve, so its raw q must ALWAYS be skipped (it feeds the group's DISTR_Q rows weighted). With limits off
there are no entries at all, so all 28 pilot-self groups emitted their raw q at their carrier rows on top of their
own DISTR_Q terms. The one-step `createContext` path already did the right thing (a constant `psSkip = 1` per such
group) and says why in its comment; the batch path was the outlier.
Fix: the derivation moves OUT of the `nrl > 0` block, and a pilot-self group without an entry is marked
`PS_NO_ENTRY` (-2), which `pilotSelfGateBatch` reads as "always skip"; `-1` keeps meaning "not pilot-self".
Measured (limits off): base CONVERGED on both arms, device |f(x0)| = 117.844754887565969 against OLF's
117.84475488756634 (15 digits), first solve 5.4631120881 / 0.2283399726 / 2.4533278552e-3 against OLF's
5.4631121272 / 0.2283399701 / 2.4533278580e-3, base distributed power within 7.9e-8 MW; window violations diff 0
(75,735 both arms), DAP worst 5.2e-7 MW. Limits ON is INERT — `PILOTSELF_GATE groups=27(+0 never-pinnable) of 30`,
every group has an entry — and the window is unchanged (958/1000, violations diff 0, DAP worst 3.7e-3 MW);
suites 304/0 both ways.
Newly REACHABLE, not a regression: with limits off every batched scenario is now runtime-rejected
(`RUNTIME:status=-2=976`, 0/1000 on device) — the CPU-replayed results are what agree above. That batch path had
never run before, because the base failed first. Next.

### Reactive limits OFF: monitored buses need a limit entry, or the whole run falls back (2026-09-28)

With the gate fixed the limits-off base converged, and the next gate showed itself: 976 of 976 batchable
contingencies were CPU-replayed (`RUNTIME:status=-2=976`, 0/1000 on device) with no native reject line at all.
The refusal is host-side and total — `GpuBatchPhaseOwnership.batchable()` returns false and
`GpuBatchedSecurityAnalysisSolver` leaves every `resultsByIndex` null — because `GpuBatchVoltageMonitoring`
records a SKIP for every standby-automaton SVC bus: it maps a bus to its reactive-limit entry, and with limits
disabled the pack has no entries. The run's own loop list HAS VoltageMonitoring (`GPU_DRIVER_ENGINE
loops=[DistributedSlack, AcHvdcAcEmulationLimits, VoltageMonitoring]`), so the loop is real and unrepresentable.
Fix: `reactiveLimitsPack` emits entries for MONITORING_VOLTAGE buses whatever the reactive-limits flag (5 of them
here). Their limits are dormant — no RL loop runs — but the entry is what the monitoring phase activates through.
Measured (limits off): 958/1000 on device (was 0), violations diff 0 (75,735 both), DAP worst 4.0e-5 MW, 18 runtime
rejects. Limits ON unchanged: 958/1000, violations diff 0, DAP worst 3.7e-3 MW; suites 304/0 both ways.

**What the limits-off arm then says about the production tail**: with reactive limits disabled the two tail
contingencies AGREE — BUS-6112_BBS_2 within 1.6e-7 MW (against 3.7e-3 with limits on), BUS-5997_BBS_2 exactly,
BUS-612_BBS_2 within 3.6e-7 — and the whole window's DAP worst drops to 4.0e-5 MW (BUS-6011_BBS). So the equations,
the slack distribution and the HVDC passes are not the tail: the tail is entirely a REACTIVE-LIMIT decision
difference. BUS-536_BBS (6.4 MW apart, limits off too) is OLF's own non-convergence.

### The production DAP tail, diagnosed to the digit — and why the obvious fix is not the fix (2026-09-28)

The limits-off arm had shown the tail was a reactive-limit DECISION difference. The instrument that closed it was
OLF's own q per check: a TEMPORARY `OLF_RL_Q_TRACE=<busId>` print in the OLF worktree's `ReactiveLimitsOuterLoop`
(added, measured with, then REMOVED and the dependency reinstalled clean; it also printed each incident branch's q
and every neighbour's v/phi). Measured on BUS-6112_BBS_2 (slot 924):
- OLF's FIRST limit check reads q = 1.902356914 pu at VL-6055_0 and pins it; the device reads 1.474409053 and
  pins it one pass later, after its own six other pins.
- The device's six pins in that same pass match OLF's q to 0.0005 MVAr (VL-6490_0 145.0179 vs 145.0184, and so on),
  so the states agree almost everywhere.
- The device's state dump at that check: VL-6055_0's neighbour VL-6036_0 is 1.683e-3 pu off OLF while every other
  neighbour is within 1e-7, and the bus behind it — the pilot-self pilot VL-6112_0 — still holds a BUS_TARGET_V
  identity at v = 1.055 with a 0.86 pu residual on that row. VL-6055_0 has THREE parallel lines to VL-6036_0, so
  1.7e-3 pu there is 0.43 pu of q here.
- OLF has NO voltage control at VL-6112_0 in this scenario: the base load flow PINNED it at a reactive limit and
  this contingency islands its only sibling VL-6112_1, so `updateRemoteVoltageControlEquations` finds no ENABLED
  controller and deactivates the pilot's BUS_TARGET_V. The device keeps it until its reactive-limit phase flips it
  (`updatePilotVeqBatch`, gated on `rlActive`), i.e. after the whole distributed-slack phase.

Two implementations were measured and BOTH REVERTED:
1. Plan-time: require a survivor in `classifyVcSplit` to be `isGeneratorVoltageControlEnabled()`, so the group takes
   the CONTROL_DIES path. The device's q then reads 1.902356914 — OLF's value to every printed digit — it pins in
   the SAME pass, and the window's DAP worst drops to **1.55e-05 MW** (from 3.70e-03; the assertion is 1e-3, so the
   window test PASSED for the first time). But it is PERMANENT, and a base-pinned controller may be switched BACK
   to PV during the scenario: 3 fixtures fail (`deviceRemoteGroupLostInBaseRegainsControlInSamePass`,
   `deviceRemoteGroupUnpinnedByContingencyRecomputesTargetQ`, `deviceRemoteGeneratorFreezeUnderControlledLevel`).
   It also moved BUS-6003_BBS_2's violation count 7 above the CPU's.
2. Per-pass: run the `updatePilotVeqBatch` flip without the `rlActive` gate. BUS-6112_BBS_2 improves 7-fold
   (3.7e-3 → 5.2e-4 MW) and violations stay equal, but 4 fixtures fail: at seed time `lost` is read before the
   per-scenario row modes and overrides are all in place, so it fires on groups that DO control.
Also measured and reverted with them: `relocChoose` returning the pilot's own row as soon as that row is not an
identity, and the pilot-self gate keying on the same fact instead of the limit type alone. Both are no-ops on this
case (a pilot-self group's candidate 1 IS the pilot's row) and together they still failed 3 fixtures, so they are
not free either.
The fix, in two parts, both landed:
1. **Read the fact OLF reads, every pass.** A group's controllers are ENABLED or not, and on the device that is the
   members' per-(scenario, entry) REACTIVE-LIMIT state — current from the seeding onwards — not the row modes the
   kernel used to infer it from (at seed time a member row reads mode 0 both for a frozen island and for an
   identity that means the member controls, which is why the ungated flip fired on groups that DO control).
   `updatePilotVeqBatch` now judges it from `dqMemberQrGroup` + `dqMemberRlEntry` + the islanded mask and applies
   the layout EVERY pass, the first fill included; the restore is the same test, so a member switching back to PV
   brings the identity back. A group whose members have no entries at all (limits off, empty q range) is left
   exactly as the extraction laid it out.
2. **Stamp the flipped row's target from the STATIC limits at seed time.** The curve-evaluated arrays are still
   zero then (`rlCurveLimitsBatch` runs in the reactive-limits phase), so the seed-time flip stamped the pilot's
   own-Q target as 0 instead of (limit − loadQ): row 12834 sat CONVERGED at target 0 through outer 32, the RL
   phase re-stamped −0.444, and the solve had to catch up — still one pass behind OLF. The kernel's re-stamp
   branch keeps the value current once the curve limits exist, so the static value is a correct starting point.
Measured together: window **DAP worst 1.38e-05 MW** (was 3.70e-03; the assertion is 1e-3, and the window test now
PASSES), violations diff **0**, 958/1000 on device; the four contingencies that carried the tail —
BUS-6112_BBS_2, BUS-5997_BBS_2, BUS-612_BBS_2, BUS-6003_BBS_2 — all agree to ~2.5e-07 MW, and only two
contingencies in the window exceed 1e-5 MW (BUS-536_BBS, 8.5 MW, is OLF's own non-convergence and is not
asserted). Suites 304/0 with the rotation on and off. The limits-off window also passes (worst 1.74e-05 MW,
violations diff 0).

### Status of the FULL busbar scope with today's code (2026-09-28, measured)

Everything above was measured on the [5000,6000) window (1000 of the 6876 busbar-section contingencies). The full
scope, run with the same code (`OLF_GPU_SA_CTG_KINDS=busbar OLF_GPU_SA_BUSBAR_TOP=99999`, no range), is a different
regime and is NOT at parity:
- **251 of 6876 on device.** The census is `RUNTIME:status=-3=6357` (LINEAR_SOLVE_FAILED) plus
  `status=-2=59`, `MAPPER:no-batchable-element=209`, `PLAN:VC_REMERGE_MULTI=47`, `PLAN:SIDE_OPEN_UNPACKED=1`,
  `VALIDATOR:ISLANDS_SLACK=1`. The −3 rejects are essentially all the plain sections (6352 `BUS-*`, 5 `GEN-*`), so
  the device barely engages and the CPU replays the rest — which is why the statuses match exactly
  (CONVERGED 6602, FAILED 5, MAX_ITERATION_REACHED 60 on both arms) and the timing is 0.86x (GPU cold 862 s
  against CPU 1-thread 744 s).
- **The BASE itself is 6.56e-02 MW apart** (cpu 14.359916545, gpu 14.294298358) — the 0.066 MW item recorded on
  2026-09-25 and this branch's namesake. Nearly every contingency inherits it: the DAP worst is 1.07 MW with a
  dense cluster around 1.063 MW and ~4180 contingencies above 1e-5 MW. Violations: CPU 503,923 vs GPU 503,905
  (diff 18, within the harness's 0.01% tolerance).
- The base's START is exact: the device's first fill is norm2 = 1.19599687948394333e+02 against OLF's
  |f(x0)| = 119.59968794839484, identical to 15 digits. The arms then DIVERGE: OLF converges in 4 outer loops /
  8 Newton iterations, the device takes 18 iterations across 15 solves and pins 36–37 buses where the window's
  base pins 19. So the extraction and assembly are right at this scope and the OUTER-LOOP sequence is not.
Order of work, from this measurement: (1) the base's outer-loop sequence at full scope (the 0.066 MW; compare the
RL pass-by-pass pin sets, `OLF_GPU_DBG_RL` against `SW_PASS`, as the window work did), then (2) the 6357
linear-solve failures, which is what keeps the device out of 96% of the scope. The window remains the fast
regression gate; it is evidently a benign subset, so a window result must not be reported as a full-scope one.

### The full-scope base's outer-loop sequence: 11 pins OLF never makes, from a first step that is not a Newton step (2026-09-29)

Asked pass by pass (the method the window work used). OLF's base at the full busbar scope: 3 Newton iterations,
DistributedSlack 0, then ReactiveLimits 1/2/3 — 4 outer loops, 8 solver iterations, and **26 switches**
(25 PV→PQ in pass 1 including 3 robust unrealistic-voltage ones, then 1 more and 1 switch-back, then 1). The
device: 18 iterations over 15 solves and **38 switched entries**. Diffing them bus by bus (device `SW_PASS` +
the row map from `OLF_GPU_EQ_DUMP`, against OLF's own switch log):
- the device makes every switch OLF makes (0 missing), plus **11 extra**: VL-2669_1..4, VL-4538_1..4,
  VL-767_1..3 — all in one pass, all `lt=3`, all with q FAR inside their limits (2.4, 2.1, 0.58 MVAr against
  ±100 MVAr). `lt=3` is the robust unrealistic-voltage pin, so the device's state at those controllers was below
  the realistic floor (0.8 × 1.02 pu) where OLF's was not.
So the extra pins are a CONSEQUENCE, and the cause is upstream: the device's first Newton step at this scope is
not a Newton step. `OLF_GPU_LIN_RESID=1` on the base: **||J·dx − F||inf / ||F||inf = 2.05e+11 with ||dx||inf =
8.3e+12** at iteration 0 (7.8e+11 at iteration 1), worst row 14491 every time =
`DUMMY_Q(BUS-6218_COUPLER_4)`, whose only terms are the coupler's ZERO_PHI (+1/−1 on two angle columns) and
NOTHING on its own column. The assembled matrix has **996 rows with an exactly zero diagonal** (pre-repair scan:
`tinyDiag=996 firstBadRow=106 nHaz=2028`), all of them couplers' dummy rows.
**The matrix is not the problem — the factorization is.** `GpuDumpedMatrixRankTest` on the device's OWN dumped
base matrix: KLU (which pivots) factors it and solves to `||J·dx − B||inf = 9.75e-13`, ratio **2.4e-14**, with a
sensible `||dx||inf = 11.5`; its largest dx components are exactly those zero-diagonal coupler rows. The same
matrix through the device's cuDSS UNIFORM BATCH (no pivoting) gives 2.05e+11. The linear-solve guard's own
comment already says this shape ("min pivots of 1e-18 under any unpivoted ordering") — and it is the same cause
as the window-free scope's 6357 `status=-3` rejects, which is why the device engages on 251 of 6876.
Fix direction (not started): the batch needs a zero-free diagonal without pivoting. The pattern is FIXED and
host-built, so a static maximum-transversal row permutation computed once per structure (what KLU does
internally) would give every row a structurally nonzero diagonal; the alternatives are extending the
relocation so every coupler's ZERO_V/ZERO_PHI lands where a real derivative exists (the coupler-tree relocation
does this for chains and still leaves 996 rows here), or a pivoted solve for the base. Acceptance: the base's
residual ratio at iteration 0 near 1e-14 as KLU gets, the same 26 switches as OLF, and the base distributed
power equal (today 6.56e-02 MW apart).

### The unpivoted factorization at full scope is fixable with the mechanism already in the tree (2026-09-29)

`ziTreeRelocation` (the coupler-tree relocation: a coupler's ZERO_V/ZERO_PHI move onto the child bus's own
PHI/V, and the child's power equations onto the coupler's dummy columns) is behind
`-Dolf.gpu.ziTreeRelocation`, **default false**, with a second gate `-Dolf.gpu.ziTreeRelocationMinCouplers`,
**default 4**. At the full busbar scope that leaves every one of the 366 couplers unrelocated (`childSide=0`
for all of them in the `.eq` dump), which is where the 996 zero diagonals come from.
Measured on the full-scope base, three configurations:

| configuration | couplers relocated | zero diagonals (pre-repair) | ratio at it=0 | first iterate |
|---|---|---|---|---|
| default (relocation off) | 0 of 366 | 996 (nHaz 2028) | 2.05e+11 | wrong (step 8.3e+12) |
| relocation on, gate 4 (its default) | 23 of 366 | 928 (nHaz 1926) | 2.49e+05 | wrong |
| relocation on, gate 1 | 329 of 366 | 111 (nHaz 222) | **1.65e-14** | 5.588341866896 vs OLF's 5.588341867362982 |

So the linear-algebra half of the full-scope problem needs no new mechanism: applying the existing relocation to
EVERY coupler tree gives the unpivoted batch a factorable structure (1.65e-14 is the quality KLU gets on the same
matrix, 2.4e-14) and the first iterate then matches OLF to ten digits. The 111 zero diagonals left are the count
the WINDOW also carries, i.e. the shapes the numeric repair already handles.
What remains in that configuration: the base still ends `MAX_ITERATION_REACHED` with the distributed power
6.56e-02 MW apart, so there is a second, outer-loop defect behind the first one — measure its pass sequence and
switch set next (`OLF_GPU_SW_TRACE=0` + `OLF_GPU_BATCH_TRACE=1` with both properties set). Also still to check
before considering the defaults: the window and both suites with the relocation on, since the property is off by
default and its gate of 4 was chosen deliberately ("the numeric row repair is EXACT on short chains ... so a
short tree keeps the proven path").

### What blocks enabling the relocation: the reactive-limit check reads q at a row the relocation repurposed (2026-09-29)

With the relocation on (gate 1) the factorization is right, and the base's SECOND defect steps forward: the first
reactive-limit pass pins 36 buses (OLF pins 25) and the re-solve then burns all 30 iterations without converging,
so the base ends `MAX_ITERATION_REACHED`. Diffing the switch sets in that configuration: 116 switch events over 46
buses against OLF's 27, with **20 device-only buses** — the 8 robust ones from before (VL-2669_*, VL-4538_*,
`lt=3`) plus q-limit pins with impossible reactive powers: VL-6049_0 at **4767 MVAr**, VL-6002_0 at 667.7,
VL-6002_2/3 at 99.5, VL-6113_1 at −78.0, VL-6162_1 at 61.9, VL-6189_0 at 83.3, VL-6315_0 at −51.7.
The `.eq` dump for that configuration names the cause. VL-6113_1's V row (13434) now reads
`terms=[cb#8765.q1@alt, cb#8928.q2@alt, zi#334.ZERO_V]` and the bus is `childSide=2` of coupler 334: the
relocation moved the COUPLER's ZERO_V onto that row, and the bus's own reactive balance onto the coupler's
DUMMY_Q column. VL-6002_0 is `childSide=2` of coupler 316 with an EMPTY V row. But the reactive-limit machinery
still reads each entry's q as `qBuf[rlVrow] + loadQ` at the bus's own V row — which is no longer its reactive
balance — so the check compares a voltage-equality residual against reactive limits and pins on it.
Fix: `rlVrow` (and the q the check reads) must FOLLOW the coupler-tree relocation, the same way
`GpuAcRowIndexing.distrQRow`'s `(ziParentEdge, dummyQCol)` overload already makes a relocated child's DISTR_Q
follow it. That is the blocker for enabling the relocation, which is the blocker for the full scope.
Chain, end to end, as measured: 996 zero diagonals (relocation off) → unpivoted factorization wrong (2.05e+11) →
wrong trajectory → 11 spurious robust pins → base 6.56e-02 MW off and 6357 scenarios rejected; with the
relocation on the factorization is right (1.65e-14, first iterate matching OLF to ten digits) → the RL q is read
at repurposed rows → 20 spurious pins → the pinned solve does not converge. Fix the q row, then re-measure both.

### Why the relocation cannot simply be switched on: the conflict is systemic (2026-09-29, measured)

Tried the narrow fix first — refuse to relocate a bus that is merely CAPABLE of voltage control (today's guard
only refuses one whose BUS_TARGET_V is ACTIVE), since such a bus's reactive-limit entry reads its q, flips its
mode and stamps its voltage target at its OWN V row, which the relocation gives to the coupler's ZERO_V. It
removes the bad pins but it also removes the relocation: **24 of 366 couplers relocated instead of 329**, 926
zero diagonals, ratio back to 2.6e+11. Reverted.
The reason is in the network: of the 329 children the relocation picks, **305 are control-capable** (named in a
`gvc` line; 1345 such buses in the full-scope extraction). A coupler in this model joins busbar sections of a
substation, and those sections are exactly where the generators sit. So there is no subset to carve out.
What a correct placement has to satisfy, both facts measured above:
1. Every row needs a structurally nonzero diagonal. A coupler brings 2 equations (ZERO_V, ZERO_PHI) and 2
   variables (DUMMY_P, DUMMY_Q); pairing the zeros with the dummies gives two zero diagonals (ZERO_V does not
   contain DUMMY_P), so the zeros MUST take an endpoint's v/phi — which is what the relocation does, and why a
   chain of k couplers consumes k of its k+1 buses' v/phi rows and leaves the root's free.
2. A control-capable child still needs somewhere to put a voltage identity when the reactive-limit loop switches
   it back to PV, and its own v row is gone. Its identity CAN live on the component's one free v row (the root's):
   inside a zero-impedance component ZERO_V makes the voltages equal, so constraining any of them is equivalent,
   and OLF has at most one active voltage target per such component (merged voltage controls). Its reactive-limit
   entry must then read q at, and flip, the coupler's DUMMY_Q row (the row the activation extractor ALREADY uses
   for its BUS_TARGET_Q — `stubBr >= 0 ? dummyQCol[stubBr] : qRow[bus]`), while the switch conditions keep
   comparing the bus's own v (the `rlCmpVrow` channel already exists for that split).
So the work is: place a relocated child's voltage identity on the component's free v row, and make the
reactive-limit pack's row follow the relocation (read/flip the dummy row, compare the own v). That spans the
extraction's placement, the activation extractor, the RL pack and the RL kernels — recorded, not started.

### The fix, part 1: a reactive-limit entry reads its q, pin and target at the row that CARRIES them (2026-09-29)

`dRlQrow`, derived in `createBatchContext` from the structure's own zero-impedance packs (no new JNI argument):
per reactive-limit entry, the row its reactive BALANCE and its pin live on — its own V row normally, or, when the
coupler-tree relocation moved its power equations onto the parent coupler's dummy columns, that coupler's
DUMMY_Q row (which is exactly where `GpuAcActivationExtractor` already puts its BUS_TARGET_Q). Voltage reads
stay on the bus's own V row, so `rlVrow` keeps its meaning and every other consumer of it is untouched.
Applied in `reactiveLimitsBatch` (the q read, and all mode/target reads and writes — 16 accesses),
`initRlLimitTypeBatch` (the base pin state and the row target) and `seedRlModeFromState` (the seeded pin).
Measured on the FULL busbar scope with the relocation on (`-Dolf.gpu.ziTreeRelocation=true
-Dolf.gpu.ziTreeRelocationMinCouplers=1`), base only:

| | before today | relocation on, rows not following | relocation on + this fix |
|---|---|---|---|
| zero diagonals (pre-repair) | 996 | 111 | 111 |
| residual ratio at it=0 | 2.05e+11 | 1.65e-14 | 1.40e-14 |
| base status | CONVERGED (wrong state) | MAX_ITERATION_REACHED | **CONVERGED** |
| base distributed power vs OLF | 6.56e-02 MW | 6.56e-02 MW | **0.000000e+00 MW** |

So the full-scope base now matches OLF exactly — the 0.066 MW this branch is named after. Still to do: the
contingency side in that configuration (slot 0 of the batch still ends `status=-2` after 30 iterations), then the
window and both suites with the relocation on, and only then the property defaults.

### The fix, part 2 (per-scenario) and where it stands (2026-09-29)

`dRlQrow` is a STRUCTURE fact, but the relocation is per scenario: a contingency that OPENS the coupler sends the
child's equations back to its own rows and the host pins the dummy row with a DUMMY_TARGET identity. All three
kernels now test for that — mode 0 at the relocated row means "coupler open this scenario, read the bus's own
row", the same test the zi scatter's stock-placement fallback uses (`initRlLimitTypeBatch` tests the BASE mode).
State, all measured with the relocation enabled (`-Dolf.gpu.ziTreeRelocation=true
-Dolf.gpu.ziTreeRelocationMinCouplers=1`):
- **Full-scope BASE: exact.** Distributed power 14.359916545 MW on both arms, diff 0.000000e+00 (was 6.56e-02),
  both CONVERGED, first residual ratio 1.40e-14.
- **Full-scope linear-solve rejects: 6357 → 2.** The whole `status=-3` class is gone.
- **Full-scope worst per-contingency distributed power: 1.07 MW → 2.05e-04 MW**, 4 contingencies above 1e-4.
- **But nothing batches: 0 of 6876 on device**, 6660 rejects now `status=-2` — `max-iter-bad-solve`, ratio ~8e-2,
  i.e. the per-scenario solves no longer converge. The WINDOW shows the same thing faster: with the relocation on
  it drops from 958/1000 to **2/1000** (974 × `max-iter-bad-solve`), while its matrices are clean (tinyDiag 9 in
  the base, 18 in the batch). So this is NOT a zero-diagonal problem: with the relocation on, the scenarios that
  OPEN a relocated coupler fail. That per-scenario return path (the `relocOwn` / `stockAltRow` channel, and
  whatever else must follow it: DISTR_Q dests, hazard donors, the P rows) is the next thing to fix.
- **The DEFAULT path is unaffected** by the two parts landed: window 958/1000, violations diff 0, DAP worst
  1.68e-05 MW, suite 304/0 (the one rotation-off failure is the known flaky 1e-12 kV assertion,
  `javaLoadBasedDistributedSlackMatchesCpu`, same digits as its earlier occurrence, which passed on reruns).
So the property defaults stay off for now: enabling them would trade a correct base for 974 CPU-replayed
scenarios. The order from here is the per-scenario coupler-opening path, then the defaults, then part 3 (a
relocated child that switches BACK to PV, which today has nowhere to put its voltage identity).

### Branch check (2026-09-29): nothing left to import

`test_inteliji-h100` (+71 commits by hash, including the whole coupler-relocation line: "Put a relocated bus's
equations back on its own rows when a scenario opens its coupler", "Move a relocated bus's TARGET with its
equation when its coupler opens", "Make the DISTR_Q member row follow the coupler-tree relocation", "restore the V
identity the relocation displaced", "Relocate only the coupler trees the numeric repair cannot hold") and
`origin/injection-scope` (+10, including "Move a bus limit only for the generators that actually set it") are
BOTH fully contained in `base-outer7-divergence`: every one of their 81 subjects is present here, the hashes
differ only because the work was rebased/cherry-picked, and where a file still differs the version here is the
later one (e.g. `GpuBatchRlCurves` uses `mainGenVc` where injection-scope still has `getGeneratorVoltageControl()
.orElseThrow()`). So the per-scenario relocation work has to be finished here, not fetched.

### Where the contingency side stands with the relocation on (2026-09-29)

The rejects fire at the scenario's FIRST solve (`max-iter-bad-solve outer=0 it=30`, final ratio 8.18e-02, norm
3.34), and that solve's FIRST iteration factors cleanly (ratio 1.24e-14, and 2.59e-12 / 1.21e-15 on the next
slots), so it is not the factorization: the Newton wanders over the 30 iterations, which is what a wrong equation
layout looks like. The failing scenario's dumped matrix has no empty row and 18 zero-diagonal rows, and all 18 are
shapes the BASE also carries (a pilot-self carrier row such as `BUS_V(VL-5997_1)`, which holds the PILOT's balance
and so has no entry of its own, plus unrelocated couplers' `DUMMY_P/DUMMY_Q` rows) — i.e. the repaired shapes that
the window already lives with when the relocation is OFF. So the defect is an equation that is wrong, not missing.
Next measurement, the one that named every earlier defect of this kind: OLF's own post-contingency equations and
residual for that scenario (`OLF_GPU_EQ_DUMP_CTG=<id>`, which now also writes OLF's f(x0) per equation) against
the device's first fill of the same slot (`OLF_GPU_VALS_SCAN_SLOT=<slot>` + `OLF_GPU_VALS_DUMP`), matched row by
row through the `.eq` map — with the relocation ON, so the comparison sees the relocated layout.

### Why the contingencies fail with the relocation on: named rows (2026-09-29)

`VALS_SCAN` now also prints the three largest |F| of the scanned slot (`worstF=[row:value ...]`) — a scenario that
spends its whole Newton budget is a handful of rows whose residual never falls, and naming them names the defect.
On the window's failing slot 0 with the relocation on, the residual is dominated by the SAME three rows at every
iteration: **12590 at 6.5e+02, 12588 at 1.09e+02, 13174 at 3.1e+01**. Read through the SA extraction's own row map
(`OLF_GPU_SA_EQ_DUMP`, since the row indices differ from the base extraction's):
- **12590 = `BUS_V(VL-6003_1)`**, the pilot-self group's carrier row (its own q via the alternate dest plus the
  group's DISTR_Q sharing) — a 650 residual means the reactive sharing is wildly unsatisfied.
- **12588 = `BUS_V(VL-6003_0)`**, the PILOT, `terms=[]` — an EMPTY row. A residual of 109 on an empty row is
  exactly `F = −target`: the row carries a target and nothing assembles into it. That is the empty-row class
  fixed twice already today, now in the relocated layout.
- **13174 = `DUMMY_P(BUS-6002_COUPLER_2)`**, a RELOCATED coupler's dummy row hosting a bus's P equation.
And the relocation is itself pin-state dependent, so the base and the SA get DIFFERENT structures: comparing the
two maps in one run, 44 of 47 couplers are relocated in the base extraction against 41 in the SA's, and the three
that differ are `BUS-5997_COUPLER_1`, `BUS-6003_COUPLER_1`, `BUS-6112_COUPLER_1` — exactly the pilot-self groups
the base load flow PINS (`ownReactiveOnItsVRow` reads `qRow`/the DISTR_Q activation, which the pin changes). So
the relocation must be made canonical the way the extraction was this morning, and the pilot-self candidate table
must follow it where a member's balance has moved — the qr table still names the buses' own V rows.
Next: make `ziTreeRelocation` pin-state independent (the same canonical rule as `hasCandidateTable`), then check
whether the pilot's row still assembles empty in a scenario; the instrument is in place (`worstF` + the two row
maps + `OLF_GPU_SA_EQ_DUMP`).

### The relocation made canonical: the window goes green, and what coverage it costs (2026-09-29)

Two changes make the coupler-tree relocation pin-state independent, which is what the failing scenarios needed:

1. **The candidate table follows the relocation.** `ExtractionColumns` now computes `ziTreeRelocation` BEFORE the
   pilot-self candidate tables and enrols each member at the row its reactive balance actually lives on
   (`parent >= 0 && dummyQCol[parent] >= 0 ? dummyQCol[parent] : vCol[member]`). Before this the `qr` table named
   the members' own `BUS_V` rows, which a relocated member no longer owns — hence row 12588 assembling empty
   (`F = −target`) and 12590 carrying 650 of unsatisfied sharing.
2. **The child test no longer reads the pin state.** `ownReactiveOnItsVRow` was
   `qRow[bn] == vCol[bn] || DISTR_Q active && distrQRow == vCol[bn]`; it is now
   `qRow[bn] == vCol[bn] || isRemoteGenVcControllerElement(bus)`, i.e. a remote controller element's own V row is
   the home of its reactive balance whatever pins it (DISTR_Q while it shares, its own Q once pinned).

Measured on the 1000-section window with `-Dolf.gpu.ziTreeRelocation=true -Dolf.gpu.ziTreeRelocationMinCouplers=1`:
**958/1000 on device, the window test PASSES, violations diff 0, distributed-power worst 2.83e-06 MW**, and the
two extractions now relocate IDENTICALLY (41 childSide=0, 6 childSide=2 in both maps) — the 44-vs-41 split is gone.
That is the same result as the relocation OFF, so the relocation is no longer a regression at window scope.

**But the first canonical child test cost most of the coverage at full scope.** Excluding every bus that has a
voltage control at all (`isGeneratorVoltageControlled()` etc.) relocated only **74 of 366** couplers against 329
with the old pin-dependent rule, and the base's first factorization came back wrong again: `tinyDiag=793`,
`||J.dx−F||inf/||F||inf = 1.3e+10`, the Newton wandering to 8.9e+24 by outer 7 (the base's reported distributed
power still matched, which is what a retry after a divergence looks like — not a clean solve). The reason is
MERGED controls: over a zero-impedance component OLF merges the voltage controls and only ONE of them is main and
visible, so only that one bus gets a `BUS_TARGET_V` row, while every other controlled bus of the component owns
nothing. The canonical owner test is therefore OLF's own `VoltageControl.findMainVisibleControlledBus()`, which is
pin-state independent (its `isHidden` reads element DISABLE, not the reactive-limit pin), and that is what
`hostsVoltageIdentity` now asks. Being measured: full-scope base (`tinyDiag`, `LIN_RESID` ratio, base distributed
power, relocated count) then the window again.

Still open after that: a relocated child that a contingency RE-HOMES as its section's main control has no free V
row for the identity (the coupler is closed, so its V row holds `ZERO_V`). The component's rows are all equal in V,
so the identity can go on the root's V row — the root's own reactive balance then has to move, exactly as OLF moves
a controlled bus's Q. The 16 `VC_REMERGE_MULTI` plan rejects already send that shape to the CPU.

### The two per-scenario row flips the relocation also moves (found by reading, 2026-09-29)

With the relocation ON by default, every per-scenario flip of a bus's own `BUS_V` row has to ask first whether the
relocation gave that row to a coupler's `ZERO_V`. Two such flips live in `GpuBatchQualification`, both on the
re-merge path (`reHomedControls`), and neither asked:

- **A dead control's survivor solving its own reactive balance** (`deadControlBuses`, a `GenOutage` with
  `vRow = vVar.getRow()` and `targetQ`, mode 4 via `deadControlRowMode`). For a relocated child that row holds
  `ZERO_V`, while its reactive balance — and the `BUS_TARGET_Q` the activation extractor writes — are at the parent
  coupler's `DUMMY_Q` row. So the flip must be DIVERTED to that row, exactly as the native reactive-limit pass
  already diverts through `rlQrow`. An outaged coupler puts the equations back on the bus's own rows, so the
  diversion is per scenario (the same test the device makes: mode 0 at the relocated row).
- **A re-homed control's new main becoming a PV bus at its target** (`reHomed.pvTargets()`, a `reHomedPvEntry` at
  `vVar.getRow()`). That flip turns the row into a voltage IDENTITY, and a relocated child has no free row for one:
  taking its V row back would delete the coupler's `ZERO_V` and nothing would then tie the two ends' voltages. Until
  the identity can move (below), such a scenario is a plan reject (`PLAN:VC_REMERGE_RELOC_VROW`) and runs on the CPU.

**The design that removes the reject.** Every bus of a zero-impedance component has the same voltage, so the
component needs ONE identity row, not one per bus that might control. The root keeps its own rows today, and its V
row carries its reactive balance; OLF replaces a controlled bus's reactive balance with the identity, so the
component's identity belongs on the ROOT's V row whichever of its buses is the elected main, with the root's
reactive balance moving the way a controlled bus's does. That also removes `hostsVoltageIdentity` as a coverage
limit: the root would be CHOSEN as the identity owner instead of every capable bus being excluded from relocation.

### The relocation ON by default: the full-scope base is EXACT (2026-09-29)

`hostsVoltageIdentity` asking OLF's own `VoltageControl.findMainVisibleControlledBus()` — of the controls merged
over a zero-impedance component exactly one is main and visible, and its controlled bus carries the component's
single identity — recovers the coverage the "has a control at all" test threw away, while staying pin-state
independent. Measured on rte6515's FULL busbar scope (6876 sections, the objective; the [5000,6000) window is only
the fast regression check), base load flow with `-Dolf.gpu.ziTreeRelocation=true -DziTreeRelocationMinCouplers=1`:

| | no relocation | strict owner test | canonical owner test |
|---|---|---|---|
| zero diagonals (`tinyDiag`) | 996 | 793 | **0** |
| hazard rows (`nHaz`) | — | 1616 | 222 |
| first linear residual ratio | 2.1e+08 | 1.3e+10 | **1.5e-14** |
| base distributed power vs CPU | 6.6e-02 MW | 0 after a divergence + retry | **0.000000e+00 MW, clean** |

So the device's base now factors its own matrix at machine precision at full scope and lands on OLF's distributed
power exactly, with no numeric row repair left to do (222 hazard rows are the unrelocated non-tree couplers).
The properties are flipped to ON / min 1 by default, with the javadoc carrying the measurement.

**The window's precompute then failed loudly, and the message named the defect**: `device row 12562 assigned twice
— column 13026 then ZERO_V elem 9060`. The per-scenario carrier row (`GpuAcRowIndexing.stateQRow`, the first
ENABLED member's own `BUS_V` row, which the in-kernel mode scan picks) did not follow the relocation, so a
relocated member's row got both the pilot's reactive balance and its coupler's `ZERO_V`. The canonical owner test
frees more buses, which is why the extra coverage surfaced it. Fixed by giving `stateQRow` the relocation (a
`ziParentEdge`/`dummyQCol` overload, the same mapping `distrQRow`'s tree overload makes) and by mapping candidate 0
— the pilot's own fallback row — through it too, so the table, the target map and the kernels name one row each.

### Full scope with the relocation on: the base is exact, the BATCH still rejects (2026-09-29)

Full 6876-section run with the defaults (relocation now on), GPU arm reject trace:

| | count |
|---|---|
| contingencies with `status=-3` | 6616 |
| contingencies with `status=-2` | 51 |
| slot-level `linear-solve` rejects | 13232 |
| slot-level `max-iter-bad-solve` | 4 |

and the shape is uniform: **6616 of them reject at `outer=0 it=1`** with `ratio` around 1e+10
(`||J.dx-B||inf` ~ 8.7e+10 against `||F||inf = 3.840`, the same residual on slot after slot). So iteration 0 of
every batched scenario factors fine and iteration 1 does not — the pattern is fixed per context, so something the
device does BETWEEN iterations (the per-scenario row-mode application, the pilot-self gate, the pilot-veq flip, the
reactive-limit seeding) is what breaks the matrix, not the extraction. The same 1000-section window and the
[5000,5050) slice batch cleanly (50/50 on device, violations diff 0, worst 1.7e-07 MW), so this is scope-dependent:
366 couplers and 6876 scenarios against 47 and 1000.

This is NOT a regression from the relocation: full scope with the relocation off was 251/6876 with 6357
`status=-3`, the same failure. What the relocation fixed is the BASE (exact, `tinyDiag=0`, ratio 1.5e-14); the
batch defect was always there underneath and is now the single thing between us and the full scope.
Being measured: the slot-0 scan of the full-scope batch context at it=0 vs it=1 (`OLF_GPU_VALS_SCAN_SLOT=0` +
`OLF_GPU_LIN_RESID`), which prints `tinyDiag` / `dupRows` / `deadCols` / `mode4Rows` per iteration and the worst
row of the linear residual — the pair that named every earlier defect of this class.

### What the full-scope batch was failing on: the mode-4 ALTERNATE row (2026-09-29)

The slot-0 scan of the full-scope batch context named it in one line:
`tinyDiag=7 firstBadRow=13985 deadCols=2 firstDeadCol=14417 mode4Rows=[2888 3364 13985 14403 14415 14417 14473
14475 14497]`, against `tinyDiag=0 deadCols=0` for the base on the same structure, and every failing slot's
`LIN_RESID it=1` agreeing on `worstRow=13053` with `||dx||inf = 1.5e+15`.

Read through the row map, the mode-4 rows are `DUMMY_Q(BUS-1459/5997/6003/6194/6229_COUPLER_n)` — RELOCATED
children's reactive rows, each carrying its bus's `DISTR_Q` terms (`kind=3`, weights 1/3, 1/2, 0.31…). So the
failing shape is: a reactive limit PINS a relocated child. The pin lands on the coupler's `DUMMY_Q` row (the
native pass diverts it there through `rlQrow`, correctly — that is where the child's balance lives), the mode-4
flip gates off the `DISTR_Q` terms that were the only entries in that `DUMMY_Q` column, and every family writes
the bus's OWN q to `qRowAlt[bus]`, which was still the bus's own `BUS_V` row — the row now holding the coupler's
`ZERO_V`. Result: the coupler's dummy column has NO equation (dead column), the `ZERO_V` row is corrupted, the
factorization is garbage from iteration 1 on, and 6616 of 6876 scenarios reject.

The fix is one line in `ExtractionColumns`, beside the `qRow`/`pRow` relocation that was already there: a
relocated child's plain own-row ALTERNATE moves to the coupler's `DUMMY_Q` row too (a candidate-table pilot's
`-2-group` encoding is left alone). It fixes every family at once, since they all read `columns.qRowAlt()`.
Measured right after: the full-scope base has `tinyDiag=0` at EVERY pass (it was 796 once pins landed), linear
residual ratios of 1e-15 to 1e-12 throughout, and the 31-iteration plateau at outer pass 6 is gone — that
plateau WAS this defect in the base. Pattern nnz drops 100086 -> 98892, as it should: the alternate now points at
a row that already carries those columns.

Open right after the fix: the full-scope base reported `MAX_ITERATION_REACHED` in a GPU-arm-only run whose
visible scans stop at outer 6 it=8 (|F| 2.8e-3 and falling), so the later passes run in another context. Being
measured with both arms and `OLF_GPU_DBG_NR` beside the device scans.

### And the reactive-limit check has to READ the relocated row too (2026-09-29)

With the alternate row fixed, the full-scope base stopped being wrong and started being INCOMPLETE: OLF's engine
log for the GPU arm reads `AcLoadFlowResult(outerLoopIterations=0, solverIterations=2…)` and the status comes back
`MAX_ITERATION_REACHED`, with the device's own passes matching OLF's first three solves exactly — 3 Newton
iterations, then 1, then 2 (OLF: `|f(x0)|=119.5997` → 5.5883 → 0.2284 → 0.002453, then DistributedSlack 1
iteration, then ReactiveLimits 2) — and then stopping where OLF does two more ReactiveLimits passes.

The reason is the third consumer of "where does this bus's q live": `GpuAcOuterLoopPacks.remoteMemberQPack`
accumulates each remote controller's OWN reactive injection into the diagnostic q buffer at `vCol[bus]`, while the
native reactive-limit pass reads `qBuf[rlQrow]` — the coupler's `DUMMY_Q` row for a relocated child. So every
relocated member's q read as ZERO and could never trip a limit, which is why the device's ReactiveLimits loop
declared itself done after one pass. OLF's own base makes **28 switches over 3 passes (25 + 2 + 1)**, and its q
values for exactly these buses are what the device must reproduce: `VL-5997_0 q=268.41 > maxQ=95.5`,
`VL-5997_1 q=-82.69 < minQ=0`, `VL-6003_0 q=192.10 > 74.77`, `VL-6003_1 q=192.10 > 75.36`,
`VL-6003_2 q=229.12 > 119.27` (`OLF_GPU_DBG_RL`).

All three places now go through ONE mapping, `GpuAcRowIndexing.relocatedQRow(busNum, ziParentEdge, dummyQCol,
vCol)`, with the list of its consumers in its javadoc: the reactive-limit pin and q read, the candidate table, the
activation target map, the families' mode-4 alternate, and this diagnostic fill. The pattern of the last three
defects is the same one — a consumer left on the bus's own row silently reads or writes ZERO there — so the
javadoc names it rather than letting the next consumer rediscover it.

### The full-scope base now CONVERGES, on OLF's own sequence (2026-09-29)

Two more consumers of "where does this bus's q/p live" were reading the bus's own row, both on the way BACK from
the device (`GpuNewtonRaphson.writeBackDriverMutations`), which is why the device's own state was right and OLF's
re-evaluation of it was not:

- **the slack bus** was resolved by matching `dsSlackPhiRow` against each bus's `BUS_PHI` variable row, and the
  pack ships the RELOCATED row (`pRow`), which no `BUS_PHI` owns. So the match failed, the slack's retained share
  was never written back, and OLF measured the converged state against stale targets.
- **the pinned limit** was read as `deviceTargets[rlVrow]`, the bus's own row — the coupler's `ZERO_V` row for a
  relocated child — so the write-back froze a voltage target on the bus as if it were a reactive limit.

`OLF_GPU_SEQ_TRACE=1` is what named it: `iters=1 norm=1.672228 devNorm=9.454597e-06 status=MAX_ITERATION_REACHED`
— the device converged and OLF re-measured 1.67. After both fixes, the same line reads
**`norm=9.495997519e-06 devNorm=9.454596917e-06 status=CONVERGED`**, against OLF's own final
`|f(x)|=9.495997504e-06`: the two agree to NINE significant digits. And the sequence is OLF's, pass for pass:

| solve | OLF | device |
|---|---|---|
| initial | 3 iterations (119.5997 → 5.5883 → 0.2284 → 0.002453) | 3 |
| DistributedSlack 0 | 1 | 1 |
| ReactiveLimits 1 | 2 | 2 |
| ReactiveLimits 2 | 1 | 1 |
| ReactiveLimits 3 | 1 | 1 |

OLF makes 28 switches over those three ReactiveLimits passes (25 + 2 + 1). Next: the full-scope batch, which could
not even start while the base failed (`precomputed on device: -1`).

### Full scope on device: 6530 of 6876 (2026-09-29)

With the base converging, the full-scope batch runs. Census of the 6876 busbar-section contingencies:

| outcome | count |
|---|---|
| **precomputed on device** | **6530** |
| `MAPPER:no-batchable-element` (by design) | 209 |
| `RUNTIME:status=-2` | 111 |
| `PLAN:VC_REMERGE_MULTI` (by design, CPU) | 47 |
| `RUNTIME:status=-3` | 22 |
| `DEVICE:MAX_ITERATION_REACHED` | 4 |
| `PLAN:VC_REMERGE_RELOC_VROW` (the new guard) | 3 |
| `PLAN:SIDE_OPEN_UNPACKED` / `VALIDATOR:ISLANDS_SLACK` | 1 / 1 |

Slot-level rejects: 44 `linear-solve` (was 13232), 110 `max-iter-bad-solve`, 5 `outer-loop-budget`.
Against the last measurement with the relocation off — 251 of 6876 with 6357 `status=-3` — that is the full
scope working for the first time. What remains is a tail of 346, dominated by the 209 unbatchable elements
(a design limit of the mapper, not a defect) and the 111 `status=-2`.

### The full-scope tail: distributed power (2026-09-29)

Full scope now runs end to end. What the comparison says:

| | value |
|---|---|
| on device | 6529–6530 of 6876 |
| base | CPU CONVERGED, GPU CONVERGED, distributed power 14.359916545 vs 14.359916553 MW |
| statuses | CPU 6602 converged / 5 failed / 60 max-iter; GPU 6601 / 6 / 60 |
| violations | CPU 503923, GPU 503838, diff 85 (0.017%) |
| distributed power | 3694 of 6601 differ by > 1e-9 MW, worst **1.073628e+01 MW** (the assertion wants < 1e-3) |

The worst offenders are big-loss contingencies where the DEVICE distributes MORE than OLF:
`BUS-5962_BBS_1` 984.0996 vs 994.8359 MW (10.74), `BUS-2289_BBS` 22.228 vs 25.042 (2.81), `BUS-5988_BBS_1`
630.954 vs 633.320 (2.37), `BUS-6193_BBS_1` 328.833 vs 331.025 (2.19), then a tail down to ~1 MW. One case is the
mirror image: `BUS-1198_BBS` cpu 1.0459 vs gpu 0.0 — the device's slack pass never fired where OLF's did, right at
the 1 MW `slackBusPMaxMismatch` boundary (both arms take the threshold from the same parameter, so the mismatch
each one measures must differ slightly). `BUS-6194_BBS_2` (1.35 vs 58.26) is in the not-converged-on-both group
and is not asserted.

Note the accounting: the chunk solver composes a scenario's total as `participants + retain`, and when the LAST
pass retained nothing (`retain` NaN) it reports the pre-distribution ALONE — which loses whatever an earlier pass
distributed. Being measured before touching it: the bus diffs of the four worst inside the full list
(`OLF_GPU_SA_BUSDIFF_CTG`), which separate an accounting gap (states agree) from a real difference (they do not).

### The distributed-power tail is REAL, not accounting (2026-09-29)

Bus diffs of the worst four inside the full list settle it:

| contingency | buses over 1e-12 kV | worst | distributed power cpu vs gpu |
|---|---|---|---|
| `BUS-5962_BBS_1` | 5442 of 6885 | **3.122e-02 kV** (rel 6.5e-04) | 984.0996 vs 994.8359 MW |
| `BUS-5988_BBS_1` | 5440 of 6885 | 6.964e-03 kV | 630.9537 vs 633.3204 MW |
| `BUS-2289_BBS` | not converged on both arms | — | 22.228 vs 24.853 |
| `BUS-6194_BBS_2` | GPU produced no bus results (cpu-only=6885) | — | 1.347 vs 58.258 |

So for the big-loss contingencies the two arms end on genuinely DIFFERENT operating points, and the device
distributes about 1% more active power. The base itself is fine in the same run: 1656 of 6886 buses differ by more
than 1e-12 kV with a worst of **3.595e-11 kV** — the known last-digit drift, nothing else.

The device's rule (`GpuBatchChunkSolver`) is OLF's: the pre-distribution is a DISPLACEMENT from the anchor, and
the post-contingency water-fill refills `preLoss + slackMismatch` ONCE from the anchor. So the suspects are the
pre-loss it replays and the slack mismatch it measures. Next: `OLF_GPU_PRELOSS_TRACE=1` at full scope, which
prints `PRELOSS ctg=<id> mismatch=… absorbed=… slackShare=…` per scenario, read for `BUS-5962_BBS_1` against the
CPU's 984.0996 MW total; the per-participant split is `OLF_GPU_DS_FINAL_DUMP` on the device (slot 0 of each chunk
only today) against OLF's own `OLF_DS_TRACE` in the dependency.

### Per-scenario STOCK PLACEMENT: what a dead coupler has to give back (2026-09-29)

With the relocation on by default, two `GpuBatchedAcSecurityAnalysisTest` fixtures fail
(`deviceMiddleSectionLoss…`), and they are the general case worth naming: a busbar-section contingency kills the
MIDDLE section, so both couplers die with it while the far section stays connected through its own line. Everything
the relocation had moved onto those couplers' dummy rows has to come back per scenario. Landed for that:

- **The flags word is not a row.** `checkRange("ziIdx", …, {…, 10}, -1, n)` validated slot 10 — spanning-tree bit,
  child side, coupler-tree bit — as a row index, so the value 13 was rejected on any network with fewer than 14
  rows. Four fixtures threw `bad ziIdx`; now slot 10 is checked as flags in [0, 16).
- **A suppressed member's q needs THREE destinations**, not two: the coupler's `DUMMY_Q` row while a reactive limit
  pins it there (mode 4), NOTHING while the coupler is dead (that row is an identity — writing the raw q into it
  corrupts the identity), and nothing while it shares. Two of the three cannot be expressed with one alternate and
  one selector, so the pair rides the existing reloc POOL marked by a NEGATIVE offset, with the dummy row itself in
  the group slot as the selector (`stockPairOut` / `stockPairPick`, all three picks).
- **The sharing equation moves too.** `buildDistrQDest` now builds a stock twin of every DISTR_Q destination at the
  bus's own row plus a per-contribution selector, and both fills divert to it when the dummy row is an identity.
  The CSR pattern carries the twin's entries, and `updateDistrQTargetsBatch` makes the same diversion so the target
  follows the equation (it used to return early on "the row is an identity").
- **The crash that hid in between**: `patchFrozenGroupDests` reads the group slot as a group id, so a stock pair's
  ROW value indexed `qrPtr` far out of bounds and wrote a garbage destination — surfacing as
  `CUDA error: an illegal memory access` out of `createBatchContext`. It now skips negative-offset entries.

Measured on the fixture: the batch ENGAGES and converges (it was falling back at `DEVICE:MAX_ITERATION_REACHED`),
and bus vl3_b went 395.43 -> **401.35 kV** against OLF's 401.69 — from 6.3 kV out to 0.34. So one consumer still
reads the dead row. Next: dump the SCENARIO's row modes (the batch context, not the base — `nS 1` in a
`OLF_GPU_VALS_DUMP` header means the sequential one) and check which of the member's tables still points at
c12's `DUMMY_Q`: the reactive-limit entry rows, the DISTR_Q member row (`mRow`), or the injection-override rows.

### A dead coupler carries no reactive flow either (2026-09-29)

Reading the device's first fill of the failing scenario against OLF's own post-contingency equations
(`OLF_GPU_EQ_DUMP_CTG=bus-mid`, which lists them with `f(x0) − target` per equation) showed the last shape:
OLF's set is `BUS_TARGET_V vl0_b/vl12_0`, `DISTR_Q vl12_2`, four `DUMMY_TARGET_*` identities for the two dead
couplers, `BUS_TARGET_P` for vl3_b/vl4_b/vl12_0/vl12_2 and `BUS_TARGET_Q` for vl3_b/vl4_b — and the device's
diverted sharing row still carried a coefficient **1 on the dead coupler's `DUMMY_Q`**. A kind-3 DISTR_Q
contribution IS a coupler's dummy reactive flow inside a controller's injection, and the fill dropped kind 0 and
kind 2 contributors when their element was outaged but had no test for kind 3. It has one now, keyed on the
per-scenario mode of that dummy column's ROW (the pack carries the COLUMN, so the row is kept beside it).
Measured: the entry goes to 0 and the row's residual moves from 0.2489 to 0.5815 against OLF's 0.6482.

What is left on that fixture, with every other row matching OLF to the digit: rows for `BUS_TARGET_P vl12_0`
(device −0.4410, OLF 0.7206), the sharing row (0.5815 vs 0.6482) and `BUS_TARGET_P vl12_2` (0.0410 vs −0.2509,
a difference of exactly the dead coupler's dummy-P value at x0). The two P rows differ by the SCHEDULED-LOSS
pre-distribution, which this contingency triggers (it loses a generator), so the next measurement is the device's
pre-distribution shares for this scenario (`OLF_GPU_PRELOSS_TRACE=1`) against OLF's (`OLF_DS_TRACE`), not the
equation wiring.

### What the full-scope differences actually are (2026-09-29)

**The 346 "not batched" are really 138.** Of the 6876 busbar-section contingencies, **209 are not simulated by
anyone**: `MAPPER:no-batchable-element` means the compose funnel resolved to nothing the device models, and OLF
produces no post-contingency result for them either — its own status total is 6667 = 6876 − 209, and none of the
209 appears in the CPU arm's post-contingency trace or in the distributed-power dump. So the population that
matters is 6667 simulated, of which **6529 run on device and 138 do not**:

| class | distinct contingencies |
|---|---|
| `status=-2` (plan rejects + runtime non-convergence) | 111 |
| `status=-3` (linear-solve failure; 44 slot-level events) | 23 |
| `DEVICE:MAX_ITERATION_REACHED` | 4 |

and inside the 111 the census accounts for 52 by design or by guard (47 `PLAN:VC_REMERGE_MULTI`,
3 `PLAN:VC_REMERGE_RELOC_VROW`, 1 `PLAN:SIDE_OPEN_UNPACKED`, 1 `VALIDATOR:ISLANDS_SLACK`), leaving about 59
genuine device non-convergences.

**The 85 violations differ over 29 contingencies, and they split in two.** Joining the `VIOL-DIFF` list with the
per-contingency distributed power:

- **24 contingencies whose states are IDENTICAL** (distributed power apart by 3.7e-05 MW, five of them by 1e-12)
  yet one arm reports a violation the other does not, with margins of 0.2 to 1.5 pu — `LINE-5779-4495 CURRENT TWO
  value=9.527 limit=8.247` on the CPU with nothing on the GPU, and the mirror image on four contingencies. A state
  that close cannot flip a violation that large, so this is a reporting or limit-selection difference, not
  numerics. All 111 one-sided violations are CURRENT; none is a voltage one.
- **5 contingencies whose states genuinely differ** — the distributed-power tail: `BUS-3050_BBS` 1.01 MW,
  `BUS-4337_BBS_1` 0.39, `BUS-6139_BBS_4` 0.80, `BUS-6113_BBS_1` 1.77, `BUS-5962_BBS_1` 10.74.

`BUS-6194_BBS_2` is its own case and the biggest single contributor: 71 of the 98 CPU-only violations, the ONLY
status mismatch in 6876 (GPU FAILED, CPU CONVERGED), it ran ON DEVICE, its distributed power is 1.35 vs 58.26 MW,
and it is a busbar section whose couplers are relocated — the same shape as the two failing
`deviceMiddleSectionLoss…` fixtures. Fixing that shape is expected to fix this contingency too.
Being measured: both arms' flows on the four branches that differ (`OLF_GPU_SA_MON_BRANCH`) plus bus diffs of
their contingencies, which separates "the GPU never reports that branch" from "the GPU uses a different limit".

### A mode for "only the contingencies that matter" (2026-09-29)

`OLF_GPU_SA_CTG_EXCLUDE=<file>` drops the listed contingency ids (one per line, `#` comments) from whatever the
kind knobs built, and `docs/rte6515-impactless-busbar-sections.txt` is that list for rte6515: the **209** busbar
sections OLF itself does not simulate. Measured with it: `209 of 6876 dropped, 6667 kept`.

Two things had to be corrected on the way:

- **They are NOT out of service.** The first guess was the missing `inSolvedNetwork` filter — the busbar list is
  the only one without it — but a run says `0 of 6876 outside the solved network`: every section is connected and
  in the main connected and synchronous component. OLF drops those 209 for a topological reason instead
  (`PropagatedContingency.toLfContingency` finds no branch to open, no bus to lose, no injection to drop, logs
  "has no impact", returns empty). The harness now prints that count instead of claiming a filter would help.
- **The batching share had the wrong denominator.** `6529 / 6876` counts 209 contingencies nobody simulates as
  GPU shortfall. The share now prints both: the list and the SIMULATED population, which is the CPU arm's own
  result count — `6529 / 6876 (6529 of 6667 simulated; 209 simulated by neither arm)`.

The caveat is in the message the mode prints: the contingency list decides which switches the LfNetwork retains,
so an excluded run has a different base network. Its numbers compare with other excluded runs, not with full ones.

### Full scope in exclude mode: 6532 of 6667 (2026-09-29)

With `OLF_GPU_SA_CTG_EXCLUDE=docs/rte6515-impactless-busbar-sections.txt`, and every fix of the day:

| | |
|---|---|
| on device | **6532 of 6667** (98.0%) |
| census | `status=-2` 111, `status=-3` 20, `DEVICE:MAX_ITERATION_REACHED` 4 — and no `MAPPER` class at all |
| statuses | CPU 6602 converged / 5 failed / 60 max-iter; GPU 6601 / 6 / 60 |
| violations | 503923 vs 503853, diff 70 (85 before the day's native fixes) |
| distributed power | worst 1.073628e+01 MW, unchanged (`BUS-5962_BBS_1`) |

`MAPPER:no-batchable-element` disappearing from the census confirms the 209 were exactly that class. The base
network came out the same despite the shorter list (identical status counts and the same distributed-power worst),
so for this network the retained switches did not move.

### The violations difference is now ONE defect: the device's slack loop (2026-09-29)

With the day's native fixes in, the violation differences collapse from 29 contingencies to **5**, and they are
exactly the five whose STATES differ (`BUS-4337_BBS_1`, `BUS-5962_BBS_1`, `BUS-6113_BBS_1`, `BUS-6139_BBS_4`,
`BUS-6194_BBS_2`). The 24 that had identical states and a one-sided violation are gone — that class was the
relocation's dead-coupler shapes, not a limit or reporting difference as the pre-fix log suggested. Of the
remaining 77 CPU-only violations, **71 belong to `BUS-6194_BBS_2` alone** (the GPU produced no result for it), so
the whole 68-70 violation gap is five contingencies, four of them by one or two violations.

Cross-check that says the states are otherwise equal: the four branches whose violations differed before now have
identical flows on both arms — `LINE-5779-4495 I1 cpu=10.748060 gpu=10.748060` (9 digits), `LINE-2533-1468` to
8.9e-13 relative — and those contingencies no longer appear in the diff list at all.

**The root cause of the five, measured on the 16-row fixture** (`deviceMiddleSectionLossRenormalizes…`, a 2-second
reproducer of the same shape as `BUS-6194_BBS_2`), with `OLF_DS_TRACE` on the CPU arm beside
`OLF_GPU_PRELOSS_TRACE` and `OLF_GPU_BATCH_TRACE` on the device:

| | OLF | device |
|---|---|---|
| post-contingency distributing passes | **1** (`DS_RUN mismatch=1.043213465 participants=5`) | **5** (outer 0..4, each `kind=DistributedSlack solved=1`) |
| mismatch after the loop | 0.002238 pu | **−0.477763 pu** (`PRELOSS_AFTER deviceFinal`) |
| per-pass residual | — | 1.294, 0.919, 0.153, 2.55e-2, 4.26e-3 — a constant ~0.17 factor |

So the device nibbles at the mismatch instead of closing it in one water-fill, and retires 0.478 pu away from
OLF. The pre-loss itself agrees to the digit (both 1.043213), and the split the device applies is
participants 0.447091 + slack 0.596122 with `factorSumAll=0.428571`, so the slack's retained share — which OLF
places by moving the slack bus's OWN generators' targetP — is what the device never puts into the iterate.
That single defect explains the distributed-power tail (worst 10.74 MW), `BUS-6194_BBS_2` failing under
`slackDistributionFailureBehavior: FAIL`, and the fixtures' last 0.34 kV.
Note for the fix: `OLF_GPU_DS_DEBUG=1`'s `[DS-GATE]`/`[DS-WF]` device prints did NOT appear — make them work
first, they are the per-pass view of the water-fill.

### "Plan level" is not "by design" (2026-09-29)

A correction to the wording used earlier. **Plan level** means the scenario is refused by
`GpuBatchQualification.planScenarios` (or by the validator it reads) BEFORE any device work, as opposed to a
runtime reject where the device solved and the result was thrown away. None of the four classes is a settled
design; they are unimplemented shapes plus one stopgap, and under the no-CPU-fallback goal all of them are open:

- **`PLAN:VC_REMERGE_MULTI`, 47.** When a contingency splits a zero-impedance component that carried several
  merged voltage controls, the surviving cluster elects a new main control (`reHomedControls`). The device
  handles exactly ONE shape of that: a cluster of a single bus controlling itself locally
  (`cluster.size() == 1 && every controller element is the controlled bus`). A cluster of several buses, or a new
  main controlled remotely, is refused. This is a DELIBERATE scope boundary, drawn when the re-merge emulation
  landed — the entry above reads "several controls or a remote controller → PLAN:VC_REMERGE_MULTI, CPU" — and
  calling it unintended (as this section first did) is wrong. What it needs is real machinery, not a flag: such a
  cluster forms a NEW merged control per scenario, with its own DISTR_Q group and candidate table, and those are
  built per EXTRACTION today, so the shared structure cannot express it. It stays open only in the sense every
  CPU route does: [[no-cpu-fallback-goal]] treats a CPU route as scaffolding.
- **`PLAN:VC_REMERGE_RELOC_VROW`, 3.** The guard added today: the re-homed control's new main is a relocated
  coupler child, whose own `BUS_V` row holds the coupler's `ZERO_V`, so its voltage identity has nowhere to go.
  A deliberate stopgap to avoid a silently wrong answer; the fix is to move a zero-impedance component's identity
  to its root, which is already written up above.
- **`VALIDATOR:ISLANDS_SLACK`, 1.** The outage islands the slack/reference bus.
- **`PLAN:SIDE_OPEN_UNPACKED`, 1 — a MISLABEL, fixed.** That site records a fixed tag for EVERY disqualified
  validator verdict, so the one full-scope entry (an islanded-slack outage) was reported as a side-open shape.
  It now records the validator's own reason.

### The genuine device defects at full scope are 23, and they share a shape (2026-09-29)

Checking every runtime reject against what the CPU does with it (the distributed-power dump marks a contingency
`not-converged` when NEITHER arm converged) settles which are bugs:

| class | count | CPU converges it? |
|---|---|---|
| `status=-2`, device non-convergence | 60 | **58 NO** — OLF reports MAX_ITERATION_REACHED for them too, so declining them is correct |
| `status=-2`, and the CPU converges | **2** | yes — `BUS-1219_BBS` (slot 241), `BUS-6229_BBS_1` (slot 2569) |
| `status=-3`, linear-solve failure | **20** | **yes, all 20** — the system is solvable and the device's factorization fails |
| status mismatch (GPU FAILED, CPU CONVERGED) | **1** | yes — `BUS-6194_BBS_2` |

So 23 contingencies, not 59. And the 20 linear-solve failures share a striking pattern: **every one is a `_BBS_1`
section** — the FIRST section of a multi-section bus (`BUS-1055_BBS_1`, `BUS-11_BBS_1` slot 220, `BUS-1282_BBS_1`,
`BUS-1320_BBS_1`, `BUS-1330_BBS_1`, `BUS-1359_BBS_1`, `BUS-137_BBS_1`, `BUS-1502_BBS_1`, `BUS-1553_BBS_1`,
`BUS-1564_BBS_1`, `BUS-1570_BBS_1`, `BUS-1599_BBS_1`, `BUS-1619_BBS_1`, `BUS-1644_BBS_1`, `BUS-1675_BBS_1`,
`BUS-1843_BBS_1`, `BUS-1869_BBS_1`, `BUS-2123_BBS_1`, `BUS-2151_BBS_1`, `BUS-2182_BBS_1`). A multi-section bus has
couplers, and losing its first section kills them — the same dead-coupler shape as `BUS-6194_BBS_2` and as the two
failing `deviceMiddleSectionLoss…` fixtures. The 58 that neither arm converges are, by contrast, OLF's own
non-convergences and not a device question.
Next: scan one of the 20 inside the full list (`OLF_GPU_VALS_SCAN_SLOT=220` for `BUS-11_BBS_1`) — a singular
matrix names its row, and the guess to test first is a dead coupler's dummy row left without an equation.

### The 20 linear-solve failures: my own rule, and the fix (2026-09-29)

`OLF_GPU_VALS_SCAN_SLOT=220` inside the full list named it in one line: for `BUS-11_BBS_1` the batched scenario has
exactly ONE zero diagonal, `firstBadRow=24`, and row 24 is `BUS_V(VL-11_1)` — a relocated child's own row, holding
`zi#1.ZERO_V` — while the same scan's `mode4Rows` list CONTAINS row 24. So the scenario pins that bus at home
(mode 4 on its own row, because the contingency killed its coupler and the equations came back) and the stock-pair
rule written earlier today said "dummy row is an identity ⇒ write nothing", which left the row empty.

The rule needs TWO modes, not one, and `stockPairPick` now reads both (the dummy row from the group slot, the
bus's own row through the device copy of `relocOwn`):

| dummy row | own row | the bus's raw q goes to |
|---|---|---|
| mode 4 | — | the coupler's `DUMMY_Q` row (pinned, coupler alive) |
| mode 0 | mode 4 | **its OWN row** (coupler dead, pinned at home) |
| mode 0 | else | nowhere — its `DISTR_Q`, diverted home, carries it |
| else | — | nowhere |

Measured right after, full scope in exclude mode: **6552 of 6667 on device** (was 6532), and `RUNTIME:status=-3`
is GONE from the census — all 20 fixed, and they were 20 of the 23 genuine device defects. The census also now
reports `PLAN:ISLANDS_SLACK` where it used to mislabel that entry `SIDE_OPEN_UNPACKED`.

### The +20 verified against the CPU (2026-09-29)

Full comparison in exclude mode after the both-modes stock pair, with the 20 formerly-failing contingencies now
solved on device:

| | |
|---|---|
| on device | **6552 of 6667**, and `0 simulated by neither arm` |
| statuses | CPU 6602 / 5 / 60, GPU 6601 / 6 / 60 — unchanged, so the 20 converge and agree |
| status mismatches | 1, still only `BUS-6194_BBS_2` |
| violations | 503923 vs 503854, diff **69** (70 before the fix) |
| distributed power | worst 1.073628e+01 MW, unchanged |
| runtime rejects | 60 × `status=-2`, and NO `status=-3` |

The twenty extra scenarios converge and their statuses match, but they are NOT difference-free, and the earlier
claim that the "identical state, one-sided violation" class had disappeared was drawn from a single run and is
wrong. The differing set is 9 contingencies here, up from 5, three of them newly batched, and it moves run to run:

| contingency | distributed power apart | the one-sided violation |
|---|---|---|
| `BUS-125_BBS_1` | 9.6e-13 MW | CPU-only `LINE-2533-1468 CURRENT TWO` value 6.654 limit 5.486, margin **1.17** |
| `BUS-1359_BBS_1` | 1.1e-12 MW | 3 CPU-only `APPARENT_POWER`, margins 0.105, 0.105, 0.035 (0.2-0.5% over) |
| `BUS-1869_BBS_1` | 6.2e-13 MW | 2 GPU-only `APPARENT_POWER`, margins 0.196, 0.305 (2-3% over) |
| `BUS-1843_BBS_1` | 7.7e-03 MW | 1 CPU-only |
| the other 5 | 0.39 … 56.9 MW | the state-differing set |

Two shapes are mixed in there: borderline apparent-power violations a hair over the limit, which a last-digit
state difference can flip either way, and at least one CURRENT violation 21% over the limit that one arm does not
report at all while the branch's own flow matches to 1e-13. The second cannot be a state difference. The
discriminator is one run of the GPU arm with `-Dolf.gpu.sa.tableViolations=false`, which replaces the GPU limit
table and its device screen with OLF's own detector on the same states.

What remains of the 23 device defects is 3: `BUS-1219_BBS` and `BUS-6229_BBS_1` (non-converging where the CPU
converges) and `BUS-6194_BBS_2` (the one status mismatch), all of which point at the slack loop's
five-passes-for-one behaviour already written up.

### Statuses agree everywhere but one contingency (2026-09-29)

The standard to hold is: where OLF does not converge, the device need not either — but the two must REPORT the
same thing. Measured across three independent full-scope runs in exclude mode, the harness's per-contingency
status comparison says exactly that:

```
statuses: CPU={CONVERGED=6602, FAILED=5, MAX_ITERATION_REACHED=60}
          GPU={CONVERGED=6601, FAILED=6, MAX_ITERATION_REACHED=60}
STATUS MISMATCHES (1): [BUS-6194_BBS_2]
```

All 60 non-converging contingencies and all 5 that OLF fails carry the same status on both arms. One in 6667
disagrees — `BUS-6194_BBS_2`, CONVERGED on the CPU and FAILED on the GPU — and that is the slack-loop defect.

**That agreement is worthless as evidence, and the check has been replaced.** For a contingency the device
DECLINED, the GPU arm's status comes from the CPU fallback — so comparing it with the CPU arm's status compares
OLF with OLF. Of the 6667, only the 6552 the device solved carry a device verdict at all; the status comparison
covers those and nothing more.

The honest question for the other 115 is the opposite one, and the harness now prints it: of the contingencies the
device declined, how many does OLF converge? `GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_IDS` records which
ids the device actually solved, and the report reads
`DEVICE VERDICT: <n> of <m> simulated solved on device; of the <k> it declined, OLF converged <j>: [ids]`,
followed by a line saying the status comparison covers only the device-solved set. A `j` of zero means every
declined contingency is OLF's own non-convergence; today `j` is 2 (`BUS-1219_BBS`, `BUS-6229_BBS_1`), and with
`BUS-6194_BBS_2` that is the whole list of device failures on a solvable system.

### The nine differing contingencies all differ in STATE (2026-09-29)

Two measurements, one correction. First, the limit path is NOT involved: a full-scope run of the GPU arm with
`-Dolf.gpu.sa.tableViolations=false`, i.e. OLF's own detector on the device's states, gives the IDENTICAL
comparison — same 9 contingencies, same deltas, violations diff 69. So neither the GPU limit table nor its device
screen decides any of it.

Second, and correcting the reading from the distributed-power dump: equal distributed power does NOT mean equal
state. Bus diffs inside the full list for the two contingencies whose distributed power matched to 1e-12 MW:

| contingency | distributed power apart | buses over 1e-12 kV | worst bus |
|---|---|---|---|
| `BUS-125_BBS_1` | 9.6e-13 MW | 5406 of 6885 | **3.078e-02 kV** (4.8e-04 rel) |
| `BUS-1869_BBS_1` | 6.2e-13 MW | 5409 of 6885 | **4.520e-01 kV** (4.9e-03 rel) |

Against the base's own floor in the same runs — 3.6e-11 kV — those are real differences, four to ten orders of
magnitude above it. So all 9 differing contingencies land on a DIFFERENT operating point, and the flipped
violations follow from that rather than from a reporting boundary. The branch monitor now prints both sides
(`I2` beside `I1`), which is what showed the GPU's own side-two current at 6.654294 against a 5.486048 limit with
no violation reported — consistent only with a different state, not a different limit.

One instrument added for the next pass: a one-sided violation now prints `pre=NEW` or
`pre=pre-existing base=<value>`, since powsybl's increased-violations thresholds (0.5 proportional on flows here)
only filter PRE-EXISTING ones — so the two shapes can be told apart without deducing it.

### The three signals must be read together (2026-09-30)

A standing rule, learned the hard way twice in one day: **distributed power, bus voltages and violations together**
decide whether a contingency agrees — never one of them alone. Both of my wrong readings came from taking one:

- reading equal distributed power as an equal state: `BUS-1869_BBS_1` matched to 6.2e-13 MW and its buses were
  4.5e-01 kV apart, 5409 of 6885 of them;
- reading a matching side-ONE current as "the state agrees, so the missing violation must be a limit question":
  side two was never compared, and the states differed.

So the harness now prints the three joined, once per run, for every contingency that differs in ANY of them:

```
TRIPLE CHECK (9 contingencies differ in at least one of the three):
    contingency              dap |d| MW   viol d   bus worst kV
    BUS-125_BBS_1             9.610e-13       -1      3.078e-02
    ...
```

with `-` where a signal was not measured (a bus diff exists only for the ids a run monitors, since monitoring all
of them dies on heap). `reportTriple` builds it from three maps filled where each signal is computed.

### The slack loop: the device retains a share it never places (2026-09-30)

The widened gate record (8 doubles, host-readable, printed by `OLF_GPU_LOOP_TRACE=<slot>`) gives the whole
accounting of each pass. On the middle-section fixture, beside OLF's own `OLF_DS_TRACE`:

OLF, post-contingency: **one** pass — `DS_RUN mismatch=1.043213465 previousMismatch=0.000000000
effective=1.043213465 participants=5`, then `DS_PASS mismatch=0.002238490 distribute=false`. Done, mismatch ≈ 0.

The device, same scenario:

| pass | slack mismatch | retain | rem = mismatch − retain | fired |
|---|---|---|---|---|
| 0 | 0.896259 | 0.050369 | 0.795521 | yes |
| 1 | 1.020064 | 0.356811 | 0.306442 | yes |
| 2 | 0.866892 | 0.407901 | 0.051090 | yes |
| 3 | 0.841359 | 0.416420 | 0.008519 | yes |
| 4 | **0.837101** | **0.835680** | 0.001420 | no — retires |

So the device retires with **0.837101 pu (83.7 MW) of slack mismatch**, having decided the slack "retains"
0.835680 of it, while OLF ends at 0.002238. The slack's target was moved by only 0.172854 (the host's pre-loss
seed), so that retained share is never placed anywhere: not on a participant row, not on the slack's target. Its
`dsRetain` also DOUBLES between the last two passes (0.416420 → 0.835680) as the participants' displacement flips
sign, which is what lets the gate retire.

Where it comes from: `rho = 1 − Σf` over the DEVICE's distributed-slack pack, i.e. the device assumes every unit
of participation it does not carry belongs to the slack bus, whose injection is free. In this scenario
`fsGlobal = 1/3`, so it credits two thirds of the mismatch to the slack, while OLF's pass has **5 participants**
and distributes the whole 1.043213 among them. The kernel cannot fix this itself: `dsSlackShift` is a
host-uploaded seed (`const double*`), so a pass has no way to place the slack's share.

Next, and it is a measurement not a guess: list the device's DS-pack participants for this scenario against
OLF's 5 (`OLF_DS_TRACE` names them per pass with their factors), and see which ones the pack is missing and
whether `rho` should renormalize over the survivors instead of crediting the slack. The renormalization already
exists for ISLANDED participants (`dsIslanded`, "Dropping the stranded ones renormalizes over W' = W − Σ_stranded
w"), so the question is which other exclusion this scenario hits.

### The slack loop, measured on both arms: two fixes and what is left (2026-09-30)

The measurement the diagnosis needed. OLF's participant set for the fixture's post-contingency pass, from a
temporary `DS_PART` trace in the fork (added, measured, reverted, dependency reinstalled):

```
DS_PART n=3 factorSum=7.500000000
DS_PART   element=gs factor=5.000000000     <- the SLACK bus's own generator
DS_PART   element=g0 factor=1.250000000
DS_PART   element=g2 factor=1.250000000
```

So OLF gives the slack bus's generator two thirds of the 1.043213 and the other two a sixth each. The device's
own pack, from `OLF_GPU_DS_BUSMAP` and `OLF_GPU_DS_FINAL_DUMP`, held three rows — `vl12_0` at its own PHI row and
the two RELOCATED children at their couplers' `DUMMY_P` rows — and the one on the dead coupler's row had moved by
**−0.894 pu**: the islanded generator was still receiving slack power.

**Fix 1: mask an islanded participant by its BUS in the pre-loss too.** `islandedDsParticipants` already matches
on the bus as well as the row, precisely because a relocated participant's row is the coupler's `DUMMY_P` while
the frozen rows are the bus's own — and its comment records the 7.386 MW that cost on `BUS-6014_BBS_2`. But the
pre-loss distribution REBUILT the mask from the row test alone and ignored the plan's. With the plan's mask, the
device's split becomes OLF's exactly: `absorbed=0.347738 slackShare=0.695476`, summing to 1.043213.

**Fix 2: let the slack's own share reach the device.** `dsSlackShift` carries the displacement OLF puts in the
slack bus's generator target (the device has no `BUS_TARGET_P` row there — its P is free). It was seeded only in
the slack-only path, and the JNI uploaded it only BESIDE a per-participant anchor shift, which exists only for
operator-strategy pairs. So an ordinary contingency's slack share never reached the device. Both are fixed; the
trace now shows `dsSlackTgt=0.868330` = base 0.172854 + share 0.695476.

Measured effect on the fixture: the final slack mismatch falls from **0.837082 to 0.280556** pu and the bus moves
from 401.35009 to 401.36183 kV against OLF's 401.69026. The loop still fires four times where OLF fires once.

What is left, and the next thing to measure: at pass 0, after a CORRECT pre-loss, the mismatch is only 0.175954
yet `dsRetain` is **−0.173174** — negative, which is what makes the gate's remainder large enough to fire. With
`absorbed = +0.347738` (the pre-loss seeds) and `rho` two thirds, the retain should be about +0.35, not −0.17. So
the retain formula's treatment of the pre-loss seeds is the next suspect: `tot = (slackP − slackTgt) + absorbed`
now double-counts a share that the slack target already carries.

### Fix 2 REVERTED: the slack share and the retain formula are one change (2026-09-30)

Letting the slack's pre-loss share reach the device (seeding `dsSlackShift` for an ordinary contingency and
uploading it independently of the anchor shift) took the suite from 2 failures to **10**: `loadLossBatchMatchesCpu`,
`shuntBranchLoadComboBatchMatchesCpu`, `deviceBusLossIslandingMergedPilotSurvivorMatchesCpu`,
`deviceIslandedRemoteControlGroupMatchesCpu`, `deviceBranchOutageIslandingAGroupMemberBatchesAndMatchesCpu`,
`deviceBasePinnedMemberKeepsItsQWhenASiblingIsIslandedMatchesCpu` and the two middle-section ones. Reverted; the
suite is back to 2.

Why it cannot go in alone: shifting the slack's target changes the mismatch EVERY scenario measures, and the
retain formula then compounds it — the same formula whose first pass reports `dsRetain = −0.173174` where the
pre-loss seeds imply about +0.35, because `tot = (slackP − slackTgt) + absorbed` counts the seeds once in
`absorbed` and again in the shifted `slackTgt`. So the correct change is a single one: seed the slack's share AND
take it out of the retain's `tot`. That is the next step, and the fixture plus these eight fixtures are its test.

What DID land and is verified: masking an islanded participant by its BUS in the pre-loss distribution (fix 1),
which makes the device's split OLF's exactly (`absorbed=0.347738 slackShare=0.695476` against OLF's
`gs 5.0 / g0 1.25 / g2 1.25` over 1.043213) and leaves the suite at its two known failures.
