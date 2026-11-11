# Post-contingency results without writing the state back — plan

**Status** (2026-11-11): P0 starts on branch `state-view` (from `main` e0eb0922). The plan was written 2026-11-09 on branch `device-final-results`;
the issues of §7 were triaged on `main` since, each with its status there, and none of the open ones is a prerequisite
of P0 to P4. The §0 measurements are from network.xiidm, our batched LU (`linearSolver=BATCHED_LU`), GPU arm, phase
timing on (`OLF_GPU_PHASE_TIME=1`, `OLF_SA_PROFILE=1`); every gain must be re-measured A/B, alternating, at least
twice.

## 0. Summary

For every contingency the device answers, the host today pushes the solved state back into OLF's Java objects —
the contingency is applied to the `LfNetwork`, the state is injected into the equation system (variable index update
+ `StateVector.setValuesOnly` + `AcNetworkVector` resync), every `LfBus` gets its V and angle, the result is built by
OLF's evaluables, and the network is restored before the next contingency. All of this exists so that OLF's own
code can read the state. It is about a fifth of the warm run:

| warm, host per-contingency loop | full list (14,266 ctg, 12,732 simulated) | branch list (10,540 ctg) |
|---|---|---|
| `toLfContingency` (propagation, connectivity) | 0.51 s | 0.27 s |
| `lfContingency.apply` (disables, listeners) | **0.54 s** | **0.38 s** |
| state injection (index update + `setValuesOnly`) | **2.16 s** | **1.32 s** |
| `LfBus` V/angle write + realistic check | **0.74 s** | **0.54 s** |
| violation screen + host confirm | 0.57 s | 0.39 s |
| CPU replay of the device rejects | 0.66 s (3 ctg) | — |
| `networkState.restore` | **1.04 s** | **0.79 s** |
| **removable by this plan (bold rows)** | **≈4.5 s of 22.1 s** | **≈3.0 s of 12.6 s** |

gpu-security-analysis (own LU) has no such loop — it builds results from device arrays — and sits at 12.9 s warm end
to end on the branch list against our 13.3–14.7 s; with the bold rows gone (measured with phase timing on, so the
untraced gain may differ somewhat) we would be roughly 1–2.5 s ahead.

The table above calls the 14,266-contingency list "full"; since 2026-11-10 that is "the JSON list", and "the full list"
is every contingency kind with the busbar sections (38,441). The profile of that day
(`docs/network-xiidm-gpu-profile.md` §3, PR #54, `main` 031d7b31, warm) puts what this plan removes, by phase:

| | every branch | the JSON list | the full list |
|---|---|---|---|
| State injection + network update (P1) | 1.90 s | 3.76 s | 15.02 s |
| Apply + restore (P2) | 1.12 s | 1.74 s | 6.15 s |
| The screen's state upload and run (P4) | 0.23 s | 0.33 s | 2.73 s |
| State handback in the batch: host buffer zero-fill, gather, copy, pack (P4) | ~1.1 s | ~1.05 s | ~6.3 s |
| File-backed state store: writes, reads, delete (P4; full list only) | — | — | ~2.0 + 2.45 + 0.8 s |
| **About** | **4.4 s of 12.2** | **6.9 s of 22.1** | **35 s of 130** |

The plan removes the bold rows **without giving up exactness**: the final violation decision and every reported value
stay in OLF's own Java arithmetic — OLF's own formula functions, fed with the doubles the device produced, read from the
state array by base row instead of from OLF's objects (§3). Nothing is decided on the device that OLF would decide
differently (§2), and a bitwise cross-check against today's path runs on every contingency of the gates (§5).

Phases (§4): **P0** oracle and cross-check channel → **P1** confirm from the state array, no injection, no `LfBus`
write → **P2** no apply/restore of the contingency (one fork hook) → **P3** a complete device screen (disabled
elements, opened sides, automation switches, taps, voltage-angle limits) → **P4** screen inside the solve context, read
back only what the host decides with (no state round trip, no state store) → **P5** state monitors and device-answered
operator strategies on the same path.

## 1. What the write-back serves today

Mapped from the code (GPU repo `sa/gpu/`, fork `sa/`, 2026-11-09). For one device-answered contingency C:

1. **Restore of the previous contingency** — `restoreBaseState` override, targeted when the previous one was device
   answered (`NetworkState.restore(touchedBuses, branches, false)`: its disabled buses/branches, lost loads/generators,
   shifted shunts, automation-switched branches, all HVDC/areas/excluded slack buses), else a full `restore()`.
2. **`toLfContingency(C)`** — connectivity analysis on the base network; computes every `ConnectivityResult` field at
   construction (created components, disconnected load/generation P, disconnected element ids). Side effect: may
   relocate the slack (`setExcludedSlackBuses`), undone by the restore.
3. **`lfContingency.apply`** — disables branches/buses/HVDC, opens sides (`setConnectedSide1/2(false)` re-points the
   branch's P/Q/I evaluables to open/ZERO terms), shifts shunts, updates lost loads/generators; fires the equation
   system updater, `AcNetworkVector` and index listeners.
4. **Loss distribution** — replayed without a network write when C has a device state and a replayed value (no
   parameter override, no strategy targeting C); else OLF's own, which moves generator targets.
5. **`runPostContingencySimulation`** (GPU override): lazy voltage restore of C's disabled buses; automation-system
   switches applied with `setDisabled`; the flow read set; **`injectState`** (variables-only index update, row remap,
   `setValuesOnly(values, readSet)` → `AcNetworkVector` refreshes rows, every live bus's V/φ arrays and the flows of
   the read set); **`updateBusesFromState`** (`setV/setAngle` on every live bus + realistic-voltage check from the raw
   array; falls back to `AcSolverUtil.updateNetwork` + `isStateUnrealistic` when the base system holds RHO1/ALPHA1/
   SHUNT_B); FAILED status when unrealistic; `networkResult.update()` (state monitors); `detectViolations`
   (candidates from the device screen, confirmed with OLF's evaluables; side-opened and uncovered branches scanned in
   full; voltage-angle limits on the host); `ConnectivityResult` from the `LfContingency`; `filterIslanded`.
6. **Operator strategies** of C (OLF's loop): conditions read the post-contingency violations (and, for threshold
   conditions, flows); device-answered pairs apply their actions on the host, inject the pair's state and confirm the
   same way; CPU-answered pairs need the post-contingency network (targets, controls, voltages).
7. **Restore** before the next contingency (step 1 of the next one).

What actually needs OLF's **objects** (as opposed to the device's numbers): only CPU-answered strategies and CPU
replays (they run OLF's solver on the network), and two result paths with no static shortcut (zero-impedance branch
flows in a state monitor, §6). Everything else reads a handful of doubles per element through OLF's evaluables.

## 2. The constraint: the final decision stays in OLF's arithmetic

This was decided on 2026-09-10 (`docs/on-device-violation-detection.md`, "the device SCREENS, the host CONFIRMS") and
re-decided when the L12 experiment was deleted (8b02accd, 2026-10-17): the device's values differ from OLF's in the
last bits — CUDA `sin`/`cos`/`hypot` and FMA contraction against OLF's jafama `FastMath` (table based; not
reproducible on the device) — and a value at a threshold is then decided the other way. L12 measured 2.1e-10
relative on currents (nearly cancelling terms), with identical violation SETS on 14,255 contingencies but never
identical values.

The exposure is real wherever a comparison has no tolerance:
- `value > reducedLimit` (strict) — a measure-zero knife-edge, but values must still be OLF's;
- the increased-violations filter (`violationWeakenedOrEquivalent`): with the core defaults the voltage thresholds are
  0, so a pre-existing voltage violation is re-reported when the post value moves by one ulp
  (`docs/increased-violations-thresholds.md`: 6,137 phantom diffs on identical states); and `v2.limit == v1.limit` is
  an exact double equality;
- `LfBus.getV()` is a round trip, `(x · nominalV) / nominalV` — it differs from `x` by one ulp in ~11 % of buses, so
  even the bus voltage must be recomputed the way OLF stores and reads it.

So this plan keeps the split: the device **screens** (over-emits candidates, relaxed band) and the host **decides**
with OLF's exact arithmetic. What changes is where the host gets its inputs: from the state array, by base row, instead
of from OLF's objects after a write-back.

## 3. Target design

### 3.1 The state view: OLF's formulas fed from the device state

A read-only evaluator, per scenario, over the device state array (base row layout, `precomputed.baseVariableRows()`).
Every quantity the detection and the results read has an explicit OLF formula (fork `olf-local-patches` @ db0067e05):

| quantity (what reads it today) | exact call | inputs |
|---|---|---|
| P1 closed (`getP1().eval()` → `AcNetworkVector` array) | `ClosedBranchSide1ActiveFlowEquationTerm.p1(y, sin(ksi), g1, V1, r1, V2, sin(theta1(ksi, φ1, a1, φ2)))` | y, sinKsi, g1 as the **vector** holds them (`AcBranchVector`: cached at build / tap change) |
| Q1, P2, Q2 closed | `ClosedBranchSide1ReactiveFlowEquationTerm.q1`, `ClosedBranchSide2ActiveFlowEquationTerm.p2`, `ClosedBranchSide2ReactiveFlowEquationTerm.q2` (same pattern, `cos` for Q, `theta2` for side 2) | same |
| I1, I2 closed (`getI1()` → scalar term, reads the state vector) | `ClosedBranchSide1CurrentMagnitudeEquationTerm.i1(y, ksi, g1, b1, V1, φ1, r1, a1, V2, φ2)`, `…Side2…i2(…)` | y, ksi, g, b from the **I term's** construction snapshot (the term is refreshed on tap changes only while closed at both sides) |
| S1, S2 | `FastMath.sqrt(p*p + q*q)` | P and Q above |
| side 2 opened (bus 1 live) | `OpenBranchSide2ActiveFlowEquationTerm.p1(y, cos ksi, sin ksi, g1, g2, b2, V1, r1)`, `…ReactiveFlowEquationTerm.q1(…)`, current: **private** static `OpenBranchSide2CurrentMagnitudeEquationTerm.i1(…)`; side-2 values are 0 | y, ksi, g, b from the **open term's** snapshot (open terms are never refreshed on tap changes) |
| side 1 opened (bus 2 live) | `OpenBranchSide1ActiveFlowEquationTerm.p2`, `…q2`, **private** static `OpenBranchSide1CurrentMagnitudeEquationTerm.i2`; side-1 values are 0 | same |
| r1, a1 | `x[row]` when the branch's RHO1/ALPHA1 variable exists with a row, else `piModel.getR1()/getA1()` | variables reachable through the closed I term's `getR1Var()/getA1Var()` (no `VariableSet.getVariable`, which creates on a miss) |
| bus V for detection | `(x[vRow] * nominalV) / nominalV`, compared with `getHigh/LowVoltageLimit()`; reported `vRt * nominalV` | `nominalV` of the bus |
| voltage-angle limit | `d = x[φRow(to)] - x[φRow(from)]` compared in radians; reported `Math.toDegrees(d)` | from/to buses |
| zero-impedance branch, switch | NaN — never in violation | — |
| scaling | I × `PerUnit.ib(side bus nominalV)`, P and S × `PerUnit.SB`, compared against `LfLimit.getReducedValue()` (P with `Math.abs`) | `GpuLimitTable` already holds the scale per group |

Notes that decide bit-identity:
- the closed P/Q path and the scalar path use the same statics and bit-identical `sin`/`cos` (jafama `sinAndCos` vs
  `FastMath.sin/cos`, checked in the bytecode); the vector's own `i1[]`/`i2[]` arrays (`hypot(p, q)/v`) are never
  read and must not be used;
- the open-branch currents are `private static`: P1 adds a two-line fork patch making them `public static`
  (upstream-able), rather than reflection or a verbatim copy that could drift;
- the inputs must be read from the same objects today's consumers read (vector arrays for P/Q, the I term's snapshot,
  the open terms' snapshots) — not recomputed from the `PiModel`, which can differ after a base-case tap change (caveats
  C1/C2 of the mapping);
- the state view never calls `setStateVector`, `setV`, `getVariable` or anything that mutates OLF's objects.

Per scenario the state view also needs what today comes from the mutated `LfNetwork`: the disabled branches and buses
(`lfContingency.getDisabledNetwork()`), the sides it opens, the automation-system switches the device applied
(`precomputed.automationSystemSwitched()`), and — for strategy pairs — the actions' effects (§3.6).

### 3.2 The decision without OLF's objects

A variant of `GpuLimitScanner.detectFromCandidates` reading values from the state view instead of evaluables:
- branch candidates in table order, the first confirming limit of a group wins (OLF's `break`), strict `>`;
- a disabled or side-opened branch per the contingency's sets (not `branch.isDisabled()`); side-opened branches
  scanned in full at their table position with the open formulas;
- bus candidates with the round-tripped V; voltage-angle limits from the angles;
- the `LimitViolation` objects built exactly as today (`createLimitViolation`, `createBusLimitViolation`) and passed
  through the post-contingency `LimitViolationManager`, so the increased-violations filter, the key/ordering rules
  (insertion-ordered map, the type-less key caveat) and the limit reductions are untouched.

The realistic-voltage check already runs on the raw array (`updateBusesFromState`'s fast path) and moves as is, minus
the `setV/setAngle`. Status, distributed active power (device maps) and the `ConnectivityResult` (`LfContingency`
fields) need no state at all.

### 3.3 The contingency life cycle without apply/restore

Fork `AbstractSecurityAnalysis.processContingency` is private. P2 adds one protected hook, called before `apply`:

```java
/** The post-contingency result built without touching the network, or empty to run the contingency as usual. */
protected Optional<PostContingencyResult> resultWithoutNetwork(LfNetwork network, C context,
        PropagatedContingency propagatedContingency, LfContingency lfContingency, ...) { return Optional.empty(); }
```

When it answers: no `apply`, no loss-distribution network write (the device value is replayed, the report node
reproduced as `distributeLoss` does today), no injection, no strategies (the hook declines contingencies with
strategies until P5), and no `restoreBaseState` — except the excluded slack buses `toLfContingency` may have set,
which must be put back (a cheap targeted restore). The lazy voltage restore simply has nothing to undo for these
contingencies; its invariants must still hold when a network-path contingency (CPU replay, strategy, exhausted)
follows a device-final one, and the reverse.

`toLfContingency` stays (0.27–0.51 s): it gives the connectivity result and the disabled sets. Reusing the precompute's
own propagation is a possible follow-up, measured separately.

### 3.4 Eligibility (which contingencies take the new path)

Conservative at first, widened phase by phase. A contingency C is device-final when all hold:
- the batch is ready, C has a converged device state (or is in the exhausted set, which already needs no state);
- no operator strategy targets C (P5 widens);
- no state monitor applies to C, or the monitored results are produced from the state view (P5);
- no `ContingencyLoadFlowParameters` extension on C; no overload-management systems confining connectivity;
- the base system holds only BUS_V, BUS_PHI and DUMMY_P/Q variables, or the state view handles RHO1/ALPHA1/SHUNT_B
  (P1 can include them: r1/a1 rows are covered by §3.1; SHUNT_B only matters to a later CPU solve, see §7 flag a);
- no diagnostic trace that needs the injected state (`OLF_GPU_SA_RESID`; the cross-check of §5 runs both paths).

Everything else keeps today's path, unchanged.

### 3.5 A complete device screen

Today the screen sees base topology and base parameters: no per-scenario disabled mask (outaged branches emit
candidates the host drops), opened sides evaluated closed (the host rescans them), automation switches at their base
status, taps at base values, no voltage-angle limits, uncovered branches scanned by the host. P3 makes the screen see
each scenario as solved:
- the per-scenario disabled mask (contingency + automation switches) — the solve context already holds it;
- opened sides through the side-open dangling rows and tap moves through the per-scenario
  `ClosedBranchParameterOverrides` the solve already uses (`scenario_overrides.cuh`);
- a voltage-angle table (from/to φ rows, low/high in radians);
- branches without a pack today (base-open branches a strategy or automation closes) through the base-open mapper.

The host then never scans in full, except a scenario whose candidates overflow the buffer. The band is re-sized from
measurement: the cross-check reports the largest device-vs-host gap per quantity; the current 1e-9 against L12's
observed 2.1e-10 is a 5× margin, not the "7 orders" its comment claims.

### 3.6 States: screen in the solve context, read back only what decides

Today every converged state leaves the device at harvest (≈21k doubles × 12,726 ≈ 2.1 GB on the full list), lives in
the heap or the file-backed `GpuStateStore`, and is uploaded again, in 512-scenario chunks, for the screen (0.27–0.53 s
per run) — the screen runs lazily, once, at the first contingency.

P4 screens each slot at harvest, in the solve context (states, masks and overrides are there), and hands back per
scenario only what the host decides with: per candidate the inputs of its formula (V1, φ1, V2, φ2, r1, a1 — or V for a
bus, the two φ for an angle limit), plus status, distributed active power and the realistic flag. A full state is read
back only for a scenario that needs it (state monitor, operator strategy, CPU-path diagnostics). Expected: no state
store, no re-upload, a few MB instead of GBs over PCIe, much less host memory.

### 3.7 Operator strategies and state monitors (P5)

- Conditions read the post-contingency violations, which the device-final path produces exactly; threshold conditions
  read flows — computable from the state view.
- A device-answered pair needs its actions' effects in the state view: switch/branch actions are topology (disabled
  sets), tap actions change the branch's parameters (from the `PiModelArray` at the action's tap, read without
  setting it), generator/HVDC actions change no branch formula. Pairs whose actions the state view cannot express
  keep today's path.
- A CPU-answered pair keeps the full network path for its contingency (as today: the whole contingency goes to the
  network path).
- State monitors: branch/bus/3WT results from the state view (§3.1 formulas; flow transfer from the pre-contingency
  result), except zero-impedance branches (`ZeroImpedanceFlows` evaluates whole bus balances — no static shortcut):
  a contingency monitoring one keeps today's path.

## 4. Phases

Each phase lands once its gates pass, with no switch: the final design keeps no `-D` or environment switch that selects
a computation (decided 2026-11-10, tests included), so a phase's A/B lives in test code, and P0's cross-check channel only
prints. Gates in §5.

### P0 — oracle and cross-check (no behaviour change)
- **State view** class (§3.1) with unit tests asserting bitwise equality with OLF's evaluables on fixtures: IEEE 14,
  Eurostag with fixed current limits (I, P, S groups), a phase shifter with an ALPHA1 variable, a transformer with a
  RHO1 variable, branches opened at side 1 and at side 2, a three-winding transformer, a tie line, zero-impedance
  couplers, voltage-angle limits, a base-case tap change before the contingency (caveat C1/C2).
- Fork patch: `OpenBranchSide1/2CurrentMagnitudeEquationTerm.i2/i1` → `public static` (upstream candidate).
- Trace channel `OLF_GPU_STATE_VIEW_CHECK` (`GpuTrace` + `docs/env-vars.md` row): on every device-answered
  contingency, after today's injection, recompute every confirmed quantity through the state view and compare
  bitwise; report the count of mismatches and the first ones (element, quantity, both doubles). The largest
  device-screen vs host gap per quantity is not measured here: the screen hands back limit indices, not values, and #92
  set the band to 1e-7 from L12's measured 2.1e-10.
- Run it on network.xiidm (full list, branch list) and rte6515 (it has S limits): **0 mismatches** before P1 starts.

### P1 — decide from the state view: no injection, no `LfBus` write
- For eligible contingencies (§3.4), `runPostContingencySimulation` skips `injectState` and `updateBusesFromState`
  and decides with the §3.2 scanner; apply and restore stay (the network still carries the disabled flags, so P1 is
  independent of P2).
- The lazy voltage restore: nothing is written, so `staleVoltages`/`deviceStateInjectedFor` bookkeeping changes —
  covered by mixed-list tests (a CPU replay, an exhausted contingency and a strategy contingency between
  device-final ones).
- Removes ≈2.9 s (full list) / ≈1.9 s (branch list) warm; also the 2026-11-09 delta-remap work becomes irrelevant for
  these contingencies (it stays for the network path).

### P2 — no apply, no restore
- Fork hook (§3.3) + GPU override; disabled sets from the `LfContingency`; excluded slack buses put back.
- Removes ≈1.6 s (full) / ≈1.2 s (branch) warm, plus listener churn the timers do not attribute.

### P3 — complete device screen (§3.5)
- Per-scenario disabled mask, opened sides and tap moves through the solve's overrides, automation switches,
  voltage-angle table, base-open branches.
- Fewer spurious candidates (outaged branches), no host full scans, and the screen and the host see the scenario the
  device solved (closes the tap gap of §7 flag g for the screen).

### P4 — screen in the solve context, targeted readback (§3.6)
- Removes the state round trip and the state store for device-final contingencies; the re-upload (0.27–0.53 s) and
  the readback's PCIe/host time (to measure: not attributed today).

### P5 — state monitors and device-answered operator strategies (§3.7)

### Not in this plan (recorded so they are not lost)
- The 3 contingencies per run the device rejects (`REJECTED`, robust retry fails, OLF reports `FAILED`): 0.7–0.9 s
  of CPU replay — answer `FAILED` from the device once the reason is shown to be OLF's (`OLF_GPU_REJECT_TRACE` names
  them).
- The fork's incremental variable index and `AcNetworkVector` resync (loops over `Variable` objects): irrelevant to
  device-final contingencies after P1, still paid on the network path.

## 5. Verification

Correctness is checked at 1 thread, timing at any thread count (memory note "threads only for timing").

Per phase:
1. Unit tests: the state-view fixtures (P0) and, per phase, a test that fails without the phase's key mechanism
   (mutation-checked, as for `GpuStateInjectorTest`).
2. `OLF_GPU_STATE_VIEW_CHECK` on network.xiidm full list and branch list, rte6515 (gen and branch kinds, saved
   1-thread arms): 0 bitwise mismatches, violation lists equal field by field to today's path.
3. `GpuSa*Test` on our batched LU (299/0/2 on 2026-11-10; cuDSS is not gated for now); full suite; checkstyle 0.
4. Against the saved 1-thread CPU arms: network.xiidm full list (statuses, 5,829 violations, DAP worst 1.69e-05 MW
   today), rte6515 gates; nothing may move.
5. Timing: A/B against the previous phase, alternating, two runs each at least, cold and warm, end to end with the
   network read (`NETWORK_READ`), with the GPU checked free and the machine load recorded.

## 6. Risks and open questions

- **Bit-identity of inputs**: the vector's cached parameters, the scalar I terms' snapshots and the open terms'
  snapshots can disagree after tap changes (C1/C2): the state view must read each from the object its consumer reads.
  P0's fixtures include a base-case tap change; the cross-check runs on every contingency.
- **jafama configuration**: host results are bit-identical only within the same `FastMath` configuration
  (`USE_JDK_MATH` flag) — the state view runs in the same JVM as OLF's evaluables, so this holds by construction.
- **Eligibility mistakes**: a consumer of OLF's objects missed by §3.4 would read a base network. Mitigation: a
  conservative predicate, the mixed-list tests, and the cross-check comparing complete results (not only violations).
- **The fork hook** widens the fork's patch set (89 commits ahead of upstream today); it is small and generic
  ("answer a contingency without the network"), and could be proposed upstream with the GPU provider as its user.
- **Zero-impedance flows** for monitored couplers have no static shortcut (`ZeroImpedanceFlows` evaluates whole bus
  balances): those contingencies keep the network path, or the device produces the coupler flows (it solves the dummy
  flow variables).
- **Multi-threaded partitions**: each partition owns its network and run state; the state view is per run state and
  read only.
- **Exhausted contingencies** already skip injection but still apply/restore and run OLF's loss distribution; they
  can join P2 (no state needed).

## 7. Pre-existing issues found while mapping (to triage separately)

- **a.** *Fixed (#88, `controlStateElements`).* The targeted restore leaves the r1/a1/shunt B that the fallback `updateNetwork` wrote in place; a later CPU
  contingency reads them in `initStateVector` and as targets (only when the base system holds those variables).
- **b.** *Not reachable, kept so by a test (#87): the device rejects an unrealistic pair, the CPU replays it.* Device-answered strategy results are never checked for unrealistic voltages: status hard-coded CONVERGED.
- **c.** *Fixed (#87, `anotherStrategyReadsFlows`).* With several strategies per contingency, strategy k ≥ 2's conditions read strategy k−1's state (OLF has the
  same quirk), partially refreshed when k−1 was a device pair with a partial flow read set.
- **d.** *Routed to OLF's path (#91): an exhausted contingency whose strategy has a threshold condition runs on the CPU.* Threshold conditions of an exhausted contingency's strategies read stale flows (nothing injected).
- **e.** *Fixed (#91, the device's generator targets handed back).* An ACTIVE_POWER condition after a device parent sees only the loss-distribution targets, not the device's
  own slack distribution.
- **f.** *No visible effect, left as is (#87).* The HVDC action on the device strategy path reads whatever the state vector holds at that moment.
- **g.** *Host half fixed (36ce83ab, `GpuDeviceTaps`: the moved taps are written back before the result is built, and the retapped branches are scanned in full); the screen still reads base taps, which is P3.* The device's outer-loop tap moves are not applied by the screen nor by the host confirm (both use base
  parameters plus the state's r1/a1); not visible on network.xiidm (no tap loop in the RTE parameters), not verified
  elsewhere — the SA tests compare voltages and violation counts, not flows.
- **h.** *Done (#92): band 1e-7.* The screen band (1e-9) is 5× the device-vs-host gap L12 observed (2.1e-10), not 7 orders as commented.
- **i.** *Done (#92, `containsKey`).* `states().get` is used as an existence test up to four times per contingency — a full file read each with
  the state store.
- **j.** *Done (#92, scanned in place).* Uncovered branches are scanned after the buses, so their violations would come out in a different order
  than OLF's (0 uncovered branches on our cases today).
- **k.** *Done (#92).* `docs/on-device-violation-detection.md` is partly stale (status line, "a screen would not help", "emits all
  of them").

## 8. What not to do

- Decide on the device with device values (L12, deleted in 8b02accd): the last bits differ and zero-tolerance
  comparisons follow them.
- Compare `x` instead of OLF's round-tripped bus voltage, or recompute a branch's parameters from its `PiModel`
  instead of reading the snapshot its consumer reads.
- Loosen a comparison or add a tolerance to make a cross-check pass: a mismatch is a missing input, find it.
