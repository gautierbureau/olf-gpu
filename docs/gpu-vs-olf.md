# GPU vs OLF: what differs, and why

The GPU extension runs OLF's AC load flow and security analysis with the same equations, the same derivatives, the same
outer loops in the same order and the same convergence criteria. A result computed on the device must match plain OLF on
the CPU at machine precision. Nothing in the maths is new.

This page is for a developer who knows OLF. It explains the few notions the GPU code has that OLF does not, and it uses
OLF's names everywhere else. All the differences come from one constraint, so the page starts there.

## 1. One fixed matrix for every contingency

**What OLF does.** For each contingency (and each operator strategy), OLF's security analysis applies the outage to the
`LfNetwork`: elements and buses are disabled, and their equations and terms are deactivated. A PV -> PQ switch of
`ReactiveLimitsOuterLoop` does the same thing to one bus: `BUS_TARGET_V` is deactivated and `BUS_TARGET_Q` activated.
Each time, `EquationSystemIndex` renumbers rows and columns over the *active* equations and variables only. The
Jacobian's structure changes, so `JacobianMatrix` rebuilds it and KLU redoes its symbolic analysis. Then the
contingency is solved, the network is restored, and the next one starts.

**What the device does.** It solves a few hundred contingencies *at the same time*, each in its own lane of one batched
linear solve (cuDSS). All lanes share one sparsity pattern and one symbolic analysis, done once for the whole run. So
the matrix structure can never change, neither from one contingency to the next nor during the outer loops.

The device therefore builds the **union** of everything any contingency may activate, once, from the base network. A
contingency or an outer-loop decision then changes only *values*, per lane:

| OLF changes... | the device changes, in that lane only... |
|---|---|
| a branch, HVDC link or element disabled | its bit in the disabled-element mask: its terms are skipped |
| a branch opened on one side | the branch is evaluated as an open-branch element (OLF's `OpenBranchSide1/2...` terms) |
| a generator or load lost | the bus's targets; its participation in the distributed slack |
| a shunt's sections | that shunt's g and b |
| an equation deactivated / activated (PV -> PQ, a lost last controller...) | which equation the row holds (section 2.2) |
| a bus cut off by the outage | its rows become fixed (identity) rows; it leaves the outer loops |

The base load flow goes through the same device path, as a batch of one.

Everything below follows from this constraint.

## 2. Consequences for the equation system

The device-only code of 2.1, 2.3 and 2.4 lives in one place: the Java package `ac.equations.vector.gpu.fixedmatrix` and
the native directory `native/cuda/fixedmatrix/`. Everything else in the GPU code follows OLF's own layout and names.

### 2.1 Stable rows: one row per variable

OLF's row numbers depend on which equations are active. The device cannot renumber, so each variable keeps one
**permanent column**, its index in OLF's equation system. Each equation sits on the row of "its" variable:

- `BUS_TARGET_P` on the row of that bus's `BUS_PHI`;
- `BUS_TARGET_Q`, or `BUS_TARGET_V` when the bus holds its voltage, on the row of its `BUS_V`;
- `DISTR_Q` on the controller's `BUS_V` row; `BRANCH_TARGET_RHO1` / `DISTR_RHO` on the `BRANCH_RHO1` row; and so on.

Two reasons:
- row numbers never move, so a device state compares with OLF's variable by variable;
- the derivative of each equation with respect to its own variable lands on the diagonal. The batched factorization does
  not pivot (2.4), so it needs that diagonal to be nonzero.

In the code: `fixedmatrix.StableRowIndexing`; the `...Row` / `...Col` fields of `ExtractionColumns`.

### 2.2 Row state: which OLF equation is active on a row

Since a row is permanent, "OLF activates equation X" becomes "the row now holds X", per lane. The code keeps one
small integer per row and per lane, the row *mode* (`GpuRowMode`, section 7):

| code | the row holds | OLF situation |
|---|---|---|
| 0 `FIXED_VARIABLE` | `x = target`: a fixed variable | slack angle (`BUS_TARGET_PHI`), the voltage of a PV bus (`BUS_TARGET_V`), the flows of an open coupler (`DUMMY_TARGET_P/Q`), a fixed tap |
| 1 `EQUATION` | the ordinary equation of that row | `BUS_TARGET_P`, `BUS_TARGET_Q`, `DISTR_Q`, `ZERO_V`, `ZERO_PHI`... |
| 2 `BUS_TARGET_V_WITH_SLOPE` | `BUS_TARGET_V` with the slope terms | a static var compensator with a voltage slope |
| 3 `BRANCH_TARGET_P` | the controlled branch's `BRANCH_TARGET_P`, on a phase controller's a1 row | continuous `PhaseControl`, CONTROLLER mode |
| 4 `BUS_TARGET_Q` | the bus's own `BUS_TARGET_Q`, target = the limit | after `ReactiveLimitsOuterLoop` switched the bus PV -> PQ (its `QLimitType` is set) |
| 6 `CROSS_FIXED_VARIABLE` | a fixed variable written on *another* row | device-only, see 2.4 |

A PV -> PQ switch is therefore `FIXED_VARIABLE` -> `BUS_TARGET_Q` on the device, and the switch back is the reverse.

In the code: the row state is OLF's activation, extracted by `GpuAcActivationExtractor`; it is not in `fixedmatrix`.

### 2.3 Equations placed on another bus's row

Pairing every equation with a variable leaves some equations without a row of their own. They are placed on a row that
OLF leaves empty. These are the same OLF equations; only their row differs.

- **Remote generator voltage control.** Generators c1, c2 hold the voltage of a remote pilot bus p. OLF has
  `BUS_TARGET_V` on p, keeps p's `BUS_TARGET_Q`, deactivates the controllers' `BUS_TARGET_Q`, and puts `DISTR_Q` on
  every controller but the first. On the device, p's `BUS_V` row holds the voltage target, so p's `BUS_TARGET_Q` goes
  on the `BUS_V` row of the **first controller that is still free**, which OLF leaves without an equation. Which one
  is free depends on that lane's PV -> PQ switches. So the pattern carries p's balance on every candidate row, and each
  Jacobian fill picks the first free controller. In the code: the "carrier"; `fixedmatrix.ReactiveRelocation` (which
  row hosts a bus's reactive balance), `ReactiveRelocationTables` / `ReactiveRelocationCandidates` (the candidate rows
  of each group); native `fixedmatrix/reactive_relocation.cuh` (`chooseCarrierRow`), `relocated_scatter.cuh`,
  `relocation_pattern.cu`.
- **Transformer and shunt voltage control.** The controlled bus's `BUS_V` row holds `BUS_TARGET_V`. Its
  `BUS_TARGET_Q` goes on the row of the control variable, `BRANCH_RHO1` or `SHUNT_B`. In the code:
  `ReactiveRelocation.computeQRow` / `computeAlternateQRow`.
- **Zero-impedance branches (couplers).** OLF models a coupler with `DUMMY_P` / `DUMMY_Q` variables and `ZERO_V` /
  `ZERO_PHI` equations on its spanning tree (`LfZeroImpedanceNetwork`). Their rows have zero diagonals as OLF places
  them. The device moves them:
  - **relocation**: for a tree coupler with a child bus, the child's power balances go on the coupler's dummy rows,
    and the coupler's `ZERO_V` / `ZERO_PHI` on the child's rows. Decided once per run, from the base tree
    (`fixedmatrix.CouplerTreeRelocation`; native `relocated_rows.cu`, `zero_impedance_relocation.cu`);
  - **stock placement**: the other couplers keep their rows, with the two equations crosswise, and the zero diagonals
    are handled by the repair (2.4) (native `stock_placement.cuh`);
  - **re-homing**: in a lane whose contingency opens the coupler, the child's balances come back to its own rows
    (`fixedmatrix.CouplerTreeReHoming`, applied per scenario by `batch.CouplerSeed`);
  - **re-merge**: after a busbar-section loss, OLF re-elects which bus of a coupler cluster holds the voltage
    (`LfZeroImpedanceNetwork.updateVoltageControlMergeStatus`, MAIN / DEPENDENT); the device puts the voltage
    target on the elected bus (`VoltageControlReMerge`). OLF behaviour replayed per contingency, not a device-only
    mechanism.

### 2.4 A factorization without pivoting

KLU pivots. The batched cuDSS factorization uses one elimination order, fixed by the analysis, for every lane, and it
does not pivot. A zero diagonal would stop it. Three mechanisms handle that. None of them changes the solution:

- **row repair**: a row with a zero diagonal gets a structurally chosen neighbour row added to it, in the matrix and
  the right-hand side alike. `(I + C) J dx = (I + C) b` has exactly the solution of `J dx = b`;
- **cross identity** (row code 6): when a fixed variable's own row is unusable, `x = target` is written on another row
  whose variable it couples to;
- **robust retry**: a lane whose batched linear solve fails is re-solved alone on the device, on its own matrix, with
  matching and pivoting.

In the code: row repair `fixedmatrix/zero_diagonal_repair.cu` and `zero_diagonal_repair_tables.cu`; cross identity
`fixedmatrix.CrossIdentity`, `fixedmatrix/cross_identity.cuh`, `GpuRowMode.CROSS_FIXED_VARIABLE`; robust retry
`GpuLinearSolveMode.ROBUST`, `solver/linear_solver_settings.cuh` and the second pass of `sa.gpu.GpuBatchPrecomputer`.
Robust retry stays with the solver: it is a mode of the linear solve itself, not a change of the matrix.

## 3. Scenario and chunk

- **Scenario**: one lane of the batch. It is exactly one pass of OLF's security-analysis loop: the post-contingency
  load flow of one contingency, or the load flow of one operator strategy after its contingency. OLF has no noun for it,
  because it runs them one after another.
- **Chunk**: the scenarios solved together in one batched Newton iteration (a few hundred, sized from free device
  memory). Successive chunks reuse the same analysis.

A scenario ends converged, at OLF's iteration limit (`MAX_ITERATION_REACHED`), or failed. A scenario that finishes
stops iterating; when at most half the chunk is still active, the solve narrows to the active lanes (compaction).

## 4. Outer loops

The device runs OLF's outer loops, in OLF's order and with OLF's stability rule: distributed slack, HVDC AC emulation
freeze and limits, secondary voltage control, voltage monitoring, reactive limits, then the phase, transformer,
transformer reactive power and shunt controls. The difference is scheduling. One loop index is shared by the whole
chunk:
- a **pass** is a Newton solve of the active lanes, then the current loop's check on every lane;
- lanes the check changed are solved again in the next pass, and the others wait;
- when the check changes no lane, the chunk moves to the next loop.

Each lane therefore sees exactly OLF's sequence of solves and checks. In the code, the loops are the driver's
**phases** (`DevicePhases`); a loop handled by Java code on the host makes the driver pause at that slot and resume
after it.

## 5. What goes back to OLF

Two kinds of contingency are handed back to OLF's ordinary sequential path, which produces the same result objects:
- shapes the device does not model, **refused** before any solve. For example the loss of the slack bus, or an island
  that takes a member of a remote voltage control;
- scenarios the device could not solve, even after the robust retry.

## 6. Glossary: the device-only terms

Only notions with no OLF counterpart. Everything else uses OLF's name.

| term | meaning | section | code |
|---|---|---|---|
| fixed matrix / union pattern | one sparsity pattern and one symbolic analysis for every contingency of the run | 1 | `structure/shared_structure.cu` |
| stable rows | each variable's permanent row and column; each equation on its variable's row | 2.1 | `fixedmatrix.StableRowIndexing` |
| row state (row mode) | which OLF equation a row holds in a lane | 2.2 | `GpuRowMode`, `GpuAcActivationExtractor` |
| carrier | the free controller whose `BUS_V` row holds a remote pilot's `BUS_TARGET_Q` | 2.3 | `fixedmatrix.ReactiveRelocation*`, `fixedmatrix/reactive_relocation.cuh` |
| relocation, stock placement, re-homing | where coupler equations sit | 2.3 | `fixedmatrix.CouplerTreeRelocation`, `CouplerTreeReHoming`, `fixedmatrix/stock_placement.cuh` |
| row repair, cross identity, robust retry | factoring without pivoting | 2.4 | `fixedmatrix/zero_diagonal_repair*.cu`, `fixedmatrix.CrossIdentity`, `GpuLinearSolveMode.ROBUST` |
| scenario, chunk, compaction | the batch | 3 | `ac.solver.gpu.batch` |
| pass, driver phase | one solve plus one loop check for a whole chunk | 4 | `solver/outer_loop_driver.cu` |
| refusal | a contingency handed back to OLF before any solve | 5 | `batch.plan.GpuBatchQualification` |
| pilot-self | the controlled bus of a remote voltage control that is also one of its own controllers (`GpuVoltageControlRole.PILOT_SELF`) | 2.3 | `PilotSelfPacker` |

The voltage-control re-merge (2.3) is not in this table: it is OLF's `updateVoltageControlMergeStatus`, replayed per
contingency (`batch.plan.VoltageControlReMerge`).

Names in the code spell every notion out (`distributedSlackBusFactor`, `reactiveLimitVoltageRow`); a `d` prefix is
a device buffer, `h` its host copy, `j` the JNI argument that carries it. Only loop indices keep one letter, each
with one meaning everywhere: `s` the scenario (lane) of a chunk, `e` an element of a family, `g` a distributed
slack generator, `c` a DISTR_Q / DISTR_RHO term, `m` a DISTR_Q member, `n` the number of rows. Per-scenario arrays
are indexed `[s * count + index]`.

## 7. Codes shared with the device

Java and the native library exchange some OLF notions as integers. Each code family has one definition per side,
under the same name: a class in `ac.solver.gpu.codes` and a header in `native/cuda/common/codes`
(`GpuCodesMirrorTest` checks they agree). Where OLF has the notion, the family is named `Gpu` + OLF's name and its
constants carry OLF's constant names.

| family | OLF notion | values |
|---|---|---|
| `GpuStateVectorScalingMode` | `StateVectorScalingMode` | `NONE` 0, `LINE_SEARCH` 1, `MAX_VOLTAGE_CHANGE` 2 |
| `GpuDistributedSlackFactorMode` | `ParticipationType.REMAINING_MARGIN` vs every other balance type | `STATIC` 0, `REMAINING_MARGIN` 1 |
| `GpuAcSolverStatus` | `AcSolverStatus` | `CONVERGED` 0, `MAX_ITERATION_REACHED` 1; device only: `DIVERGED` -1 (NaN/Inf mismatch), `REJECTED` -2 (replayed by the CPU), `LINEAR_SOLVE_FAILED` -3 (first linear solve failed, retried in robust mode; closest OLF notion `SOLVER_FAILED`) |
| `GpuOuterLoopKind` | the outer loops, by OLF loop name | `NONE` -1; `DISTRIBUTED_SLACK` 0 … `SECONDARY_VOLTAGE_CONTROL` 16 (in-kernel); `HOST_*` 5-9: a loop the driver pauses at for OLF to run; `COUNT` 17 |
| `GpuAcEmulationStatus` | `LfHvdc.AcEmulationControl.AcEmulationStatus` | `LINEAR_MODE` 0, `SATURATION_MODE_FROM_CS1_TO_CS2` 1, `SATURATION_MODE_FROM_CS2_TO_CS1` 2, `FROZEN` 3 (OLF ordinals) |
| `GpuQLimitType` | `LfBus.QLimitType` (ordinal + 1) | `NONE` 0 (not pinned), `MIN_Q` 1, `MAX_Q` 2, `MIN_REALISTIC_V` 3, `MAX_REALISTIC_V` 4; device only: `FROZEN` 5 (voltage control frozen or all generators lost). Native helpers `isMinLimit` / `isMaxLimit` are OLF's |
| `GpuReactiveLimitsState` | a bus's QLimitType + `pvPqSwitchCount` | `LIMIT_TYPE_MASK` 7 (a `GpuQLimitType`), `PV_PQ_SWITCH_COUNT_SHIFT` 3 |
| `GpuVoltageControlRole` | a bus's place in its `GeneratorVoltageControl` | `LOCAL` 0, `REMOTE` 1 (robust mode rule applies), `PILOT_SELF` 2 (device term, section 6), `REMOTE_ROBUST_MODE_EXEMPT` 3 (own control local) |
| `GpuReactiveLimitsEntryFlag` | bit flags of a reactive-limit entry | `STAND_BY_AUTOMATON` 1 (an SVC under OLF's `StandByAutomaton`), `NOT_CONTROLLER_ELEMENT` 2 |
| `GpuReactiveTermKind` | the terms `createReactiveTerms` sums (DISTR_Q `distrQKind`, remote member `remoteMemberReactiveTermKind`) | `CLOSED_BRANCH_REACTIVE_FLOW` 0, `SHUNT_COMPENSATOR_REACTIVE_FLOW` 1, `OPEN_BRANCH_REACTIVE_FLOW` 2, `DUMMY_Q` 3 |
| `GpuRealisticVoltageFixingLoop` | the last loop whose `canFixUnrealisticState()` is true (robust mode) | `NONE` -1, `REACTIVE_LIMITS` 0, `TRANSFORMER_VOLTAGE_CONTROL` 1 |
| `GpuLinearSolveMode` | device only (OLF factors one KLU matrix per solve) | `UBATCH` 3, `ROBUST` 6 |
| `GpuControlledFlow` | the flow a flow-observed controller acts on: IncrementalTransformerReactivePowerControl, IncrementalPhaseControl CONTROLLER / LIMITER | `REACTIVE_POWER` 0, `ACTIVE_POWER` 1, `CURRENT` 2; the driver selects kinds with `controlledFlowMaskBit` |
| `GpuPhaseControlMode` | `TransformerPhaseControl.Mode` of a continuous phase control (OLF ordinals) | `CONTROLLER` 0, `LIMITER` 1 |
| `GpuPiModelColumn` | a `PiModel` as 8 doubles, in its getters' order (tap tables, `closedBranchParameters`) | `Y` 0, `G1` 1, `G2` 2, `B1` 3, `B2` 4, `KSI` 5, `R1` 6, `A1` 7, `COUNT` 8 |
| `GpuDiscreteControlMemberState` | a discrete transformer / shunt voltage control member: continuous first, then rounded and disabled | `DISABLED` 0, `CONTINUOUS` 1, `ROUNDED` 2 |
| `GpuTransformerVoltageControlMode` | OLF's `TransformerVoltageControlMode`, by ordinal (the two discrete modes the device runs) | `WITH_GENERATOR_VOLTAGE_CONTROL` 0, `AFTER_GENERATOR_VOLTAGE_CONTROL` 1 (OLF's ordinals) |
| `GpuAcVariableType` | OLF's `AcVariableType` ordinals the device reads (the step clamp's varType, one per row) | `BUS_V` 0, `BUS_PHI` 1 (OLF's ordinals) |
| `GpuTwoSides` | powsybl's `TwoSides`, by ordinal (not `getNum()`): every branch side the device reads (DISTR_Q and remote-member reactive terms, controlled flows, limit groups, side-open branches, HVDC curve stations, the coupler child side in zero-impedance slot 10 as side + 1) | `ONE` 0, `TWO` 1 (OLF's ordinals); `NONE` -1 (no side: a coupler with no relocated child, a shunt term, a plain generator's curve entry) |
| `GpuRowMode` | which equation a stable row holds (section 2.2) | `FIXED_VARIABLE` 0, `EQUATION` 1, `BUS_TARGET_V_WITH_SLOPE` 2, `BRANCH_TARGET_P` 3, `BUS_TARGET_Q` 4, `CROSS_FIXED_VARIABLE` 6 |
