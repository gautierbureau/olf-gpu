# One fixed structure for KLU and cuDSS: a convergence plan

*olf-gpu design note, November 2026. Second version: the fixed structure is now built as new code in OLF, and its
first step is done.*

This note extends the design note `fixed-structure-equation-system.tex` (same directory). That note compared OLF's
alternative equations, the GPU superset matrix and upstream draft #1521, and proposed a plan. Three upstream efforts
have appeared since. This note places all of them in one design and replaces the plan.

References to OLF code are to upstream `main` at `394b3149` unless a branch is named. "rte6515" is MATPOWER
`case6515rte`.

## 1. Summary

Six pieces of work solve the same problem from different ends: keep the shape of the AC equation system constant, so
that the sparse factorisation is analysed once and the fill is a plain loop over arrays.

| Work | Author | Where |
|---|---|---|
| Alternative equations | G. Bureau | OLF fork, `alternative-equations-and-sa-mt-copy` and follow-ups |
| GPU superset matrix | this repository | `docs/gpu-vs-olf.md` |
| N-1 performance branch | G. Jamgotchian | `geofjamg/powsybl-open-loadflow`, `claude/clever-edison-qs6x0z` |
| Scatter approach | G. Jamgotchian | upstream draft #1521 |
| "Improve der" | S. Prabakaran | upstream #1564 |
| "Minimize zeros in diagonal matrix" | S. Prabakaran | upstream branch `minimize_zeros_in_diagonal_matrix` |

The upstream efforts were written without the GPU code. Each rediscovers a part of what the GPU does. Each also has
what the GPU side lacks: measurements on the CPU.

**The design** all six fit into has six notions (section 4):

- **Slot.** Every variable owns one row and one column, numbered once. An equation is written in the column of the
  variable it is paired with.
- **Alternative.** The equations that can occupy the same slot. One is active. If none is, the slot holds `x = x`.
  OLF's creator already creates all of them.
- **Pattern.** The union of the alternatives of each slot, built once, sorted. KLU and cuDSS use the same arrays.
- **Destination maps.** For each term output, the index it adds into.
- **State.** A few arrays: the unknowns, the equation each slot holds, the activity of each term, the targets.
- **Delta and lane.** A contingency is a short list of writes to the state arrays. A CPU thread, or one entry of a
  GPU batch, is a lane: its own state arrays over the shared structure.

**The route** is not the convergence of the existing branches. It is new, compact code, written from OLF's current
`main` with everything the six efforts found in view:

- it reads OLF's equation system through its public API. Step 1 changes no class of OLF; step 2 changes one;
- there is one behaviour. The fixed matrix is not an option and not a solver type: every AC computation solves on
  it, load flow, security analysis and sensitivity analysis alike;
- OLF's CPU side comes first. The GPU repository is reworked afterwards, on what has landed;
- the outer loops are due to be rewritten on arrays. The design leaves room for that (section 8).

**Step 1 is done** (section 6, fork PR #150): `FixedEquationSystem`, 890 lines with their documentation. On the
rte6515 and PEGASE fixtures of the `test2` repository:

- the matrix is OLF's Jacobian on every state checked;
- over about 600 branch outages per network, a quarter of them islanding, the pattern never changed, KLU analysed
  the matrix once, and no refactorisation failed;
- KLU is about twice as fast on the fixed matrix as on OLF's own numbering;
- the union pattern costs 5 to 7% more entries and no measurable KLU time.

**Step 2 is done** (section 6.4, fork PR #151): the AC Jacobian itself solves on the fixed matrix. OLF's 1,690 tests
pass on it with unchanged expected values. On rte6515, N-1 on 1,061 lines with default parameters, outer loops on:

- 685 Jacobian matrices were created before, 1 is now, and KLU analyses it once;
- a contingency takes a third less CPU time: 44 to 50 ms against 72 to 76 ms. On the PEGASE fixtures the gain is
  9% and 42%;
- the results are the same: same statuses, and the 71,483 violations in common agree to 2e-10.

## 2. Where the six efforts stand

| Effort | What it does | What it leaves open |
|---|---|---|
| Alternative equations | One column per equation, several bodies, one active. PV/PQ switches and islanding keep the pattern. Partial Jacobian update after a contingency. | Excludes buses with tap or shunt voltage control, zero-impedance branches, converters, fictitious buses. Falls back to a rebuild when the system stops being square. 3,700 lines in the `equations` package. |
| GPU superset | One row per variable, an integer per row saying which equation it holds, a mask per element, destination arrays for the fill, per-lane data for a contingency. Covers every case the line above excludes. | The host re-derives all of it from the `LfNetwork`: about 14,000 lines of extraction, pattern construction and contingency planners that mirror OLF. One mirror bug made 4,183 of 7,494 contingencies wrong. |
| N-1 performance branch | Pairs each equation with a diagonal variable; `klu_refactor` instead of `klu_factor`; a "ghost mode" keeps the columns of disabled buses as identity; direct Jacobian and mismatch fills; incremental restore; array pre-filter for violations. | PV/PQ switches still rebuild the structure. Three separate mechanisms keep the structure. One block of 35 commits, no PR. |
| #1521 | Scatter instead of gather for `eval` and `der` of equation arrays. | The `eval` maps are filtered by activity and rebuilt at each activation change. Draft since 14 October. |
| #1564 | Derivative "cells" built once; an inactive term contributes 0; rows are re-read only when the structure changed. | Activity is one `MutableBoolean` object per term; one handler call per cell; the additional single terms are all still evaluated. |
| `minimize_zeros…` | Orders columns and rows so that P pairs with φ and Q with v. | Pairs by element number only; single equations and control variables are not paired; rows now depend on equation columns. Work in progress. |

Three things are being built several times:

1. **The diagonal pairing, three times.** `StableRowIndexing` on the GPU, `getDiagonalVariableType` in the
   performance branch, the row filler of `minimize_zeros…`.
2. **"Keep the structure", in four layers.** Alternative equations in the equation system, ghost mode in the index,
   `JacobianLuStructure` in a copy below the Jacobian, `SharedStructure` on the GPU.
3. **The derivative fill of `EquationArray`, three times in the same three files.** #1521, #1564 and the
   performance branch.

**Measured by G. Jamgotchian** (rte6515, 7,422 line outages of which 2,162 island a bus, plain Newton-Raphson, one
thread, results identical before and after):

| Fact | Measure |
|---|---|
| Whole security analysis | 317 s to 47 s; lightsim2grid 43 to 49 s on the same machine |
| Unpaired numbering of the vectorised system | 44% more LU fill, KLU about twice as slow, 1,507 failed refactorisations; none once paired |
| Identity for islanded buses instead of removing them | full factorisations 3,380 to 2 |
| Not evaluating the inactive open-branch terms (about 15,000 per equation array) | 76.7 to 58.0 s |
| Default configuration (outer loops on) | 5% gain only: PV/PQ switches still rebuild the structure |
| Four threads | OLF slows down about 2 times per thread, lightsim2grid about 1.2 times |

## 3. What the GPU does today, in OLF's vocabulary

For readers who have not seen this repository. Details: `docs/gpu-vs-olf.md`, sections 1 and 2.

- **Rows are OLF's variable rows.** `fixedmatrix.StableRowIndexing` numbers nothing: a row is `variable.getRow()`.
  Each equation is written on the row of the variable it is paired with: `BUS_TARGET_P` on the bus's `BUS_PHI`,
  `BUS_TARGET_Q` or `BUS_TARGET_V` on its `BUS_V`, `DISTR_Q` on the controller's `BUS_V`, `BRANCH_TARGET_RHO1` and
  `DISTR_RHO` on `BRANCH_RHO1`. The system is square by construction, and the derivative of an equation by its own
  variable is on the diagonal.
- **Row mode.** One small integer per row and per lane says which equation the row holds (`GpuRowMode`, six values).
  Mode 0, "fixed variable", writes `x = target`. It covers the slack angle, the voltage of a PV bus, an open coupler,
  a fixed tap and a disabled bus. A PV to PQ switch is mode 0 to mode 4 on the bus's v row.
- **Fill, then apply.** The fill kernels write the balance equations on every row. A second kernel then rewrites the
  rows whose mode is not "equation": row zeroed, 1 on the diagonal, mismatch `x − target`.
- **Destination arrays.** For each element and each of its outputs, one integer: a Jacobian value index, a mismatch
  row, or "skip". A closed branch has 28 outputs: 4 flows and 24 derivatives. They are the 28 arrays that OLF's
  `AcBranchVector` already holds. There is one destination array per structure, shared by all lanes.
- **Relocation.** With a remote voltage control, the pilot's v row holds the voltage target, so the pilot's reactive
  balance is written on the v row of the first controller that is still free. With a transformer or shunt voltage
  control, it is written on the tap or susceptance row. The pattern holds the balance on every candidate row.
- **A contingency is data.** Per lane: the row modes that differ from the base, a bit mask of disabled elements,
  target overrides, and overrides of the outer loops' inputs. The pattern never changes.
- **No pivoting.** The batched factorisation does not pivot. Three device-only mechanisms keep the diagonal usable:
  a row repair, a "cross identity", and a relocation of zero-impedance couplers.
- **Physics from OLF's Java.** The kernels are generated from OLF's term classes by `java2cuda` and proven against
  goldens. The formulas exist once.

## 4. The design

Each subsection says what is built (step 1) and what is still to build.

### 4.1 Slots

A **slot** is one variable to solve. It owns one row (the variable) and one column (the equation that currently
determines it). Slots are numbered once. The numbering is a function of the variables only.

Rule: *the column of an equation is the row of its slot's variable.*

This is what the GPU does, and it is the fix that `minimize_zeros…` and the performance branch are both after. They
approach it from the other side: they keep the equation columns and move the variable rows to match. The result is
the same matrix while nothing switches. It differs as soon as something does: if rows follow equations, a PV/PQ
switch moves rows; if columns follow variables, nothing moves.

Built: there is one slot for every variable that a term of the system can depend on, whether the term is active or
not, in the order of OLF's variables (`FixedStructure`).

### 4.2 Alternatives: OLF already creates them

The equations that can occupy one slot are its **alternatives**. At most one is active. If none is, the slot holds
`x = x`: this is the performance branch's ghost, and the GPU's "fixed variable".

OLF's creator already creates every alternative up front and toggles them in pairs:

- `BUS_TARGET_V` is created, inactive, on every bus (`AcEquationSystemCreator.java:62-67`);
- `ZERO_V` and `DUMMY_TARGET_Q`, `ZERO_PHI` and `DUMMY_TARGET_P` are created together, the second with "inverted
  logic";
- `BRANCH_TARGET_ALPHA1` and the branch's active power target likewise.

So the slot model adds no equation. It names a relation the creator already maintains: which equations exclude each
other on which variable.

| Slot (variable) | Its alternatives, as OLF equation types |
|---|---|
| `BUS_PHI` of bus i | `BUS_TARGET_P` of i · `BUS_TARGET_PHI` · `BUS_DISTR_SLACK_P` · none (bus disabled) |
| `BUS_V` of bus i | `BUS_TARGET_Q` of i · `BUS_TARGET_V` of i · `DISTR_Q` · the `BUS_TARGET_Q` of a bus that i's generators control remotely · `BRANCH_TARGET_Q1/Q2` · none (bus disabled) |
| `BRANCH_RHO1` of transformer t | `BRANCH_TARGET_RHO1` · the `BUS_TARGET_Q` of the bus t controls · `DISTR_RHO` |
| `SHUNT_B` of shunt s | `SHUNT_TARGET_B` · the `BUS_TARGET_Q` of the bus s controls · `DISTR_SHUNT_B` |
| `BRANCH_ALPHA1` of phase shifter t | `BRANCH_TARGET_ALPHA1` · `BRANCH_TARGET_P1/P2` of the controlled branch |
| `DUMMY_P`, `DUMMY_Q` of a zero-impedance branch | `DUMMY_TARGET_P`, `DUMMY_TARGET_Q` · `ZERO_PHI`, `ZERO_V` |

**Placement, as built.** One table and one rule, with no case named:

1. Each equation type names the type of the variable it is paired with, on its own element (`AcPlacement`, 25
   lines of code). The equations OLF toggles in pairs name the same variable.
2. When two active equations are paired with one variable, the one that fixes the variable (`x = target`) keeps the
   slot.
3. The other takes a free slot: the one it had before, else the slot of a variable it depends on, else any.

A slot is *live* when an active term depends on its variable. A free slot is a live slot that holds no equation.
A slot that is not live holds `x = x`.

Rule 3 gives the GPU's relocations without naming them. A voltage control that holds a bus's voltage frees exactly
one variable: the voltage of a controller, a tap or a susceptance. The displaced reactive balance depends on it when
the two are neighbours, and takes its slot. Any other choice would be correct too: it only puts the equations in
another order. A choice with a nonzero diagonal is the better one for the factorisation, and the batched
factorisation needs it.

The alternative-equations branch makes the other choice: it keeps the pilot's balance in place and writes the
voltage target in the controller's column. It also attaches the alternatives to an equation object and duplicates
bodies. Attaching them to the variable reuses the existing equations and makes the system square by construction.
Its exclusions (tap and shunt control, zero impedance) are the families whose control variables had no slot.

### 4.3 Pattern and scope

The **pattern** of a slot is the union of the entries of the equations placed on it, plus the diagonal. Indices are
sorted. OLF stores the transposed Jacobian by equation columns, which is the Jacobian by rows: KLU and cuDSS take
the same arrays.

Built: the pattern is made from a list of placements (equation, slot) and only grows. Two scopes:

| Scope | In the pattern from the start | The pattern grows when |
|---|---|---|
| `ALL` | every equation OLF created, on the slot it is paired with | a balance takes a slot it never took, and the column lacks its rows |
| `ACTIVE` | the active equations only | also the first time an equation becomes active |

What bounds the cost on the CPU, now measured (section 6):

- **A slot that holds nothing costs no entry.** Its only entry is the diagonal. Islanding, branch outages and slack
  relocation need nothing beyond the base pattern.
- **KLU accepts stored zeros.** Its analysis depends on the pattern only; `klu_factor` pivots inside the blocks;
  `klu_refactor` reuses the pivots, with `klu_factor` as fallback.
- **The union is cheap.** With every equation in scope, the pattern has 5 to 7% more entries than OLF's Jacobian on
  the three fixtures, and KLU takes the same time within the noise.

`ACTIVE` is the adaptive scope of the first version of this note. On this evidence it is not needed. It stays
available at no cost: it is the same mechanism.

### 4.4 Destination maps

For each term of a term array and each of its derivatives, a **destination**: the index of its value in the own
column of its equation. Built at each pattern build, from `EquationTermArray`'s own description of its terms. Under
`ALL` they are computed once.

To build: the same for the terms held as objects (the single equations and the additional terms of array elements).
Today their entries are found by a search at each fill. This is the slow path of the PEGASE fixtures (section 6).

### 4.5 State

| Array | Content | Step 1 |
|---|---|---|
| `x[slot]` | the unknowns | OLF's state vector, by OLF's rows |
| equation of each slot | which alternative the slot holds | built: recomputed when an activation event arrives |
| activity of each term | on or off | read from OLF's objects |
| `target[slot]` | the target of the equation the slot holds | OLF's target vector, by OLF's columns |
| outer-loop vectors | limits, participation factors, groups | not started (section 8) |

Step 1 follows OLF's objects and index. Owning these arrays is step 5. It is what lanes and the array form of the
outer loops need.

### 4.6 One fill algorithm

As built:

```
values[] = 0 ; balance[] = 0
for each term of each term array, if it is active:
    balance[own slot of its equation] += its value
    if its equation is on its own slot:        add each derivative at its destination
    else if its equation is on another slot:   add each derivative there (rare: searched)
for each single equation:
    add its derivatives in the column of its slot; its value is the value of that slot
    if it is a balance, its value is also the balance of its own slot, active or not
for each slot that holds no equation: 1 on the diagonal
```

- **The balance buffer.** `balance[]` holds the P and Q balance of every bus, whatever its slots currently hold.
  The reactive balance of a bus that holds its voltage and the active balance of the slack are entries of it. The
  test checks it against OLF's own `bus.getP()` and `bus.getQ()` on every bus.
- **The GPU's variant.** The device fills the balance on every row, then rewrites the rows that hold something
  else. The result is the same; that order suits a kernel, this one a single thread.
- **To build: the first iteration from the delta.** A contingency starts from the base state, so its first
  Jacobian and balance are the base ones, minus the contributions of the elements it turns off. This generalises
  the reuse of the base flows in the performance branch and the partial update of the alternative-equations branch.

### 4.7 Deltas and the journal

A **delta** is a list of writes to the state arrays.

The cheapest correct way to obtain one is to **record** it. OLF applies the contingency as it does today. The
listener that keeps the fixed system up to date receives every activation; logged with the old value, they are both
the undo and the delta. A recorded delta is exact by construction: OLF computed it. That removes the class of bugs
the GPU's planner mirrors have (4,708 lines re-implementing `LfContingency` and the voltage-control re-merge).

Where recording is too slow, an **analytic delta** is computed without applying: a branch outage that islands
nothing is one element turned off. Analytic deltas are tested against recorded ones.

### 4.8 Lanes

A **lane** is one set of state arrays, with its Jacobian values and numeric factorisation, over a shared immutable
structure.

- On the GPU a lane is one entry of a batch. This exists.
- On the CPU a lane is one thread. Threads share the structure and own a few megabytes each. No `LfNetwork` copy
  per thread. The performance branch attributes OLF's poor scaling on four threads to memory contention between
  large object graphs; lanes remove the cause.

### 4.9 What stays specific to each solver

| | CPU (KLU) | GPU (cuDSS) |
|---|---|---|
| Factorisation | one analysis per structure; `klu_refactor` per iteration; `klu_factor` as fallback | uniform batch, no pivoting; row repair, cross identity, coupler relocation, adaptive refinement |
| Fill | one thread per lane; each output written where its equation is now | one thread per element; `atomicAdd`; fill, then apply the row modes |
| Physics | OLF's term arrays and terms | kernels generated by `java2cuda` from the same term classes |
| Scope | `ALL` or `ACTIVE` | whole scope up front |

## 5. Three points where the efforts disagree

**Inactive terms: keep them or compact them?** #1564 keeps every term in the structure and writes 0 when inactive.
The performance branch keeps compact tables of the active ones and measured a 25% gain from not walking 15,000
inactive open-branch terms. Both are right, about different things. The structure keeps everything. The loop over a
family that is mostly inactive should run over a list of its active elements. Step 1 keeps everything and tests the
activity of each term; the active lists are to build with the destinations of the object terms (4.4).

**Who decides a row: the equation or the variable?** Section 4.1: the variable.

**Remote voltage control: move the target or the balance?** Section 4.2: the target keeps its variable, the balance
moves.

## 6. What steps 1 and 2 established

### 6.1 The code

On the OLF fork, PR #150, on upstream `main`. Four new classes, no existing class changed. Step 2 is what uses
them.

| Class | Lines (code only) | Role |
|---|---|---|
| `equations.fixed.Placement` | 31 (4) | which variable an equation type is paired with; whether it fixes it |
| `ac.equations.AcPlacement` | 51 (25) | the table for the AC equations |
| `equations.fixed.FixedStructure` | 375 (271) | slots, equations and their rows, pattern, destinations |
| `equations.fixed.FixedEquationSystem` | 430 (324) | listens to OLF's events, places the equations, fills, writes the structure as text |

It uses OLF's public API only: `EquationSystemListener` for the events, `EquationTermArray.eval()` and `evalDer()`
for the values, `Equation.getTerms()` for the object terms, and the index for the live variables.

For comparison: the alternative-equations change is 3,700 lines inside OLF's `equations` package, the performance
branch about 3,500, and the GPU's extraction and pattern construction about 9,000.

### 6.2 The tests

`FixedEquationSystemTest`, 53 tests, in OLF's own suite (1,687 tests pass with it, checkstyle and the package
layering test included).

On 17 networks (IEEE 14 to 300, and OLF's test networks for remote voltage control, transformer, shunt and phase
control, reactive power control, HVDC in AC emulation, zero-impedance branches, node-breaker), with scalar and with
vectorised equations:

- on the base state, then after each of 60 random steps (a branch or a bus switched off or on, the voltage control
  of a generator, a transformer or a shunt, a phase control), the matrix equals OLF's `JacobianMatrix` and the
  equation values equal OLF's `EquationVector`, each active equation in the column of its slot and an identity
  column for each slot that holds nothing;
- the balances equal OLF's own evaluation of every bus's P and Q;
- on the base state, the fixed matrix solved with KLU gives OLF's Newton step.

What the random walks showed, with every equation in scope:

| Networks | Pattern |
|---|---|
| IEEE cases, remote generator voltage control, phase control, reactive power control, HVDC, node-breaker | built once, never changed |
| Transformer and shunt voltage control | grew once or twice, when a reactive balance took a tap or shunt slot it had not taken yet |
| Zero-impedance branches with voltage control | the arrays were compiled 8 times in 60 steps: OLF creates and removes equations when a zero-impedance network splits or merges |

### 6.3 On the rte6515 and PEGASE fixtures

Networks: `rte6515_full`, `pegase9241_full`, `pegase13659_full` of the `test2` repository, default parameters with
reactive limits and DC voltage initialisation. The base load flow converges; the measurements are at its solution.
Machine: a shared container with 4 virtual cores. Times are the minimum of 30 to 40 runs and vary by about 20%
between runs.

| | rte6515 | pegase9241 | pegase13659 |
|---|---|---|---|
| Buses, of which hold their voltage | 6,520 / 986 | 9,248 / 1,331 | 13,666 / 4,084 |
| Variables = slots | 13,040 | 18,496 | 27,332 |
| Jacobian entries, OLF | 86,503 | 141,413 | 193,017 |
| Entries, active equations only | 86,503 | 141,413 | 193,017 |
| Entries, every equation | 91,020 (+5.2%) | 150,826 (+6.7%) | 205,358 (+6.4%) |
| Balances on another slot, base state | 3 | 2 | 4 |
| First fill, with the compilation | 135 ms | 274 ms | 369 ms |

KLU through powsybl-math, same values, three matrices:

| | rte6515 | pegase9241 | pegase13659 |
|---|---|---|---|
| Analyse and factor: OLF's numbering | 16.2 ms | 26.9 ms | 30.8 ms |
| Analyse and factor: fixed, every equation | 6.2 ms | 11.1 ms | 13.2 ms |
| Refactor: OLF's numbering | 3.7 ms | 6.8 ms | 8.9 ms |
| Refactor: fixed, active equations only | 1.5 ms | 3.9 ms | 5.0 ms |
| Refactor: fixed, every equation | 1.6 ms | 3.2 ms | 4.9 ms |
| Solve: OLF's numbering | 0.35 ms | 0.63 ms | 0.86 ms |
| Solve: fixed, every equation | 0.16 ms | 0.33 ms | 0.53 ms |

Branch outages, one at a time, with the buses that leave the main component disabled as OLF's contingency does:

| | rte6515 | pegase9241 | pegase13659 |
|---|---|---|---|
| Outages, of which lose buses | 603 / 153 | 618 / 64 | 603 / 179 |
| Pattern builds, compilations | 0, 0 | 0, 0 | 0, 0 |
| KLU analyses | 1 | 1 | 1 |
| Failed refactorisations | 0 | 0 | 0 |
| Largest difference with OLF's Jacobian | 0 | 0 | 0 |
| Fill per outage: fixed | 1.2 ms | 6.5 ms | 9.8 ms |
| Fill per outage: OLF (Jacobian and equation values) | 3.0 ms | 11.3 ms | 16.8 ms |

What this settles:

- **The pairing is worth a factor of two in KLU**, as G. Jamgotchian measured. Here it comes with the slots, without
  renumbering OLF's index.
- **The union costs nothing measurable.** This was the open question of the first version of this note.
- **Outages, islanding included, never change the pattern**, and `klu_refactor` never failed on them.
- **The fill is already 1.7 to 2.4 times faster per outage than OLF's**, before any optimisation.
- **Object terms are the next thing to optimise** (4.4). pegase9241 has 14,658 shunt terms held as objects, against
  242 on rte6515, and its fill is five times slower for less than twice the terms in term arrays.

Not measured: LU fill counts (powsybl-math does not expose them). The full security analysis is in 6.4.

### 6.4 Step 2: the AC Jacobian solves on it

Fork PR #151, stacked on #150. `AcJacobianMatrix` owns a `FixedJacobian` (232 lines, 177 of code) and forwards its
four solves to it. Nothing selects it: it is the only way the AC Jacobian solves. A first version was a new solver
type. It was dropped: two behaviours side by side are what this plan removes.

- **A sparse matrix, created once** on the pattern, through OLF's matrix factory. `fill` then writes into the value
  array of the `SparseMatrix`, and KLU analyses it once.
- **Callers do not change.** Right-hand sides and solutions keep the numbering of the index. A block of right-hand
  sides, as sensitivity analysis uses, is moved to the slots and solved in one call per group. For 500 right-hand
  sides: 85 ms on both matrices on rte6515, 155 ms on the fixed one against 187 ms on pegase13659.
- **`getMatrix()` stays**, in the numbering of the index and built on demand, for what reads the matrix itself:
  Newton-Krylov hands it to Kinsol.
- **Not covered yet:** the DC Jacobian, the two matrices of the fast decoupled solver, the voltage magnitude
  initializer.

**Tests.** OLF's whole suite, 1,690 tests, passes with unchanged expected values: every load flow, security analysis
and sensitivity analysis test now runs on the fixed matrix. Three tests are new. One of them runs a security
analysis of IEEE 118 on its 177 lines and counts the Jacobian matrices created: 139 before, 1 now.

**A security analysis of rte6515.** N-1 on 1,061 of its 7,422 lines, default parameters with the outer loops on, DC
voltage initialisation, one thread, sparse matrices. The machine is shared and its speed varies by 20 to 30% from
one minute to the next, so the times are CPU times of runs that alternate between the two codes.

| | Before | Fixed |
|---|---|---|
| Matrices created (one of them for the DC initialisation) | 685 | 2 |
| KLU analyses of the AC Jacobian | one per matrix | 1 |
| Converged contingencies | 1,061 | 1,061 |
| Violations | 71,490 | 71,483 |
| CPU per contingency, the run of 1,061 | 83.6 ms | 57.1 ms |
| CPU per contingency, three alternating rounds of 256 | 72.4, 75.8, 74.1 ms | 44.4, 49.5, 49.5 ms |

The 71,483 violations in common have the same values, to 2e-10 relative at worst. The 7 others are one violation,
`VL-1512 LOW_VOLTAGE`, which exists before any contingency. Its voltage is 60.974857122836 kV before and after each
of the 7 contingencies, with both codes. A drift in the last bits re-reports it under the default zero thresholds:
the flicker of `docs/increased-violations-thresholds.md`. The code before has that drift on 7 contingencies, the
fixed matrix on none.

The performance branch gained 5% in this configuration, because a bus that stops holding its voltage still rebuilt
the structure (section 2). Here it does not, and the gain is a third.

**The three fixtures**, same configuration, one line in 29: about 255 contingencies per network, all converged with
both codes.

| | rte6515 | pegase9241 | pegase13659 |
|---|---|---|---|
| Matrices created, before / fixed | 157 / 2 | 119 / 2 | 67 / 2 |
| CPU per contingency, before | 72 to 76 ms | 158 ms | 270 ms |
| CPU per contingency, fixed | 44 to 50 ms | 144 ms | 155 ms |
| Violations, before / fixed | 17,221 / 17,218 | 31,979 / 31,979 | 23,284 / 23,284 |

The gain is 35 to 39% on rte6515, 42% on pegase13659 and 9% on pegase9241, whose 14,658 shunt terms held as objects
dominate the fill (6.3). The three violations that differ on rte6515 are the re-reported one above.

**Pivots.** OLF's `JacobianMatrix` analyses again when an equation is switched and chooses new pivots when a term
is: it reuses pivots only from one Newton iteration to the next. The fixed Jacobian first reuses the pivots of the
previous factorisation, whatever changed; KLU's pivot growth check, which powsybl-math already applies, asks for new
pivots when they are no longer good, and a failed update falls back to a full factorisation. Both policies were
measured:

| Plain Newton-Raphson, 256 contingencies | CPU per contingency, three alternating rounds |
|---|---|
| Before | 54, 55, 71 ms |
| Fixed, new pivots after every switch | 49, 42, 49 ms |
| Fixed, pivots reused | 36, 34, 46 ms |

Over the 3,015 refactorisations of the run of the table above, the worst pivot growth is 1.1e-2 and 99.9% are above
0.29, against a rejection threshold of 1e-10. None failed. Reusing the pivots is also what the device does: its
factorisation never chooses new ones, and a lane that fails is solved again alone, with pivoting.

**One case was settled by looking at what the GPU does.** When the whole suite first ran on the fixed matrix, three
tests failed: a zero-impedance branch with a bus at its end that is already off when the equation system is created,
the far side of an open switch that is kept for an action. The two equations that tie its buses together are off,
but their terms are still active, on a variable that is not in the system. On the device, the rows of that bus are
identities, the rows of the branch hold `DUMMY_TARGET_P` and `DUMMY_TARGET_Q`, and the balance read on a lost bus is
"a placeholder, not a measurement": the reactive limits kernel skips it (`reactiveLimitIslanded`), after a measured
bug where it did not. The fixed system places the equations the same way, and gives NaN as the balance of such a
slot.

**What it showed in OLF.** The equations cache the element indices of a single matrix. A second `JacobianMatrix` on
the same equation system therefore corrupts the incremental updates of the first. The fixed matrix keeps its own
destinations, which is why it can live next to the matrix of the index.

## 7. The plan

OLF's CPU side first. Each step is its own pull request on the OLF fork, stacked on the previous one. Step 1 adds
code only. From step 2 on there is one behaviour: the fixed matrix replaces the existing path, with no option.

| Step | Content | Exit |
|---|---|---|
| 1, done | The fixed view of an equation system: slots, placement, pattern, fill, tests | Sections 6.1 to 6.3, fork #150 |
| 2, done | The AC Jacobian solves on it: one sparse matrix on the pattern, filled in place, analysed once, refactored on the previous pivots with a full factorisation as fallback. No solver type and no option; OLF's engine, outer loops, security analysis and sensitivity analysis call it as before | Section 6.4, fork #151 |
| 3 | Equations created at run time keep their slot. OLF removes and re-creates distribution equations when a zero-impedance network splits or merges; a re-created equation is bound to its slot again by type and element | No compilation and no new analysis during a node-breaker security analysis |
| 4 | The DC Jacobian on a fixed matrix: a `DcPlacement`, and the same `FixedJacobian` | One behaviour for AC and DC. OLF's DC load flow, security analysis and sensitivity tests pass on it |
| 5 | Its own state. The slots hold the unknowns, the targets and the activities; `AcNetworkVector` computes on them (fork branch `claude/acnetworkvector-variable-index` decouples it from the index); the journal; the first iteration from the delta; destinations and active lists for the object terms | No index rebuild during a security analysis. Restore in a time proportional to the delta |
| 6 | Outer loops and results on arrays (section 8); lanes | Decisions bit-identical to the reference vectors. Four-thread slowdown per thread close to lightsim2grid's 1.2 |
| 7 | The GPU repository, reworked on what has landed. First a dump of its structure in the text format of `FixedEquationSystem.write`, compared with the CPU's. Then its extraction and pattern construction are deleted, then its planners | olf-gpu's suite passes without its extraction |

**A rule for every step:** when a case is unclear, look first at what the GPU does for it. It already runs these
cases against OLF's results, so its answer is the reference, as in 6.4.

On step 3: the device never re-creates anything. It builds the distribution equations of the base groups once, gives
each lane the subgroups a split produces, and replays OLF's election and spanning tree on the host
(`VoltageControlReMerge`, `ZeroImpedanceTreeReplay`). On the CPU OLF does the election itself; the fixed system only
has to recognise a re-created equation as the one that held the slot.

**What the upstream efforts become.** They improve OLF's own solver, independently: the pairing in the index
(`minimize_zeros…`, the performance branch), `klu_refactor` first (fork #101), the scatter fills (#1521, #1564).
The fixed Jacobian does not depend on them, and for the AC system it replaces what they speed up. What the
performance branch does outside the solver (restore, violation pre-filter, bus voltages in arrays) feeds steps 5
and 6.

## 8. Room for the outer loops on arrays

The outer loops are due to be rewritten in array form: gather, decide, derive, apply, with the decisions written
once and generated for the device (`docs/outer-loop-array-form-proposal.md`; fork PR #105 is the pilot on reactive
limits). The fixed system is built to be what they read and write.

| The loops need | The fixed system gives | State |
|---|---|---|
| The reactive balance of every bus that holds its voltage; the active balance of the slack | `balance[]`, filled at each fill for every bus, whatever its slots hold | built and tested |
| The unknowns by variable | `x` by slot; a slot never moves | step 5 (today OLF's state vector) |
| To switch a bus between holding its voltage and its reactive balance, a tap between fixed and controlling | a write of "which equation this slot holds", among the alternatives of the slot | step 5: today the switch goes through the `LfNetwork` setters and OLF's events |
| To change a target | a write of `target[slot]` | step 5 |
| That a decision never changes the matrix shape | every alternative a loop can activate is in the pattern under `ALL` | built: measured on the fixtures |
| Vectors indexed by something stable (limits, participation, groups) | the slot of a variable | built |
| The same decisions per lane on the device | the per-lane overrides of the vectors are deltas on the same arrays | steps 6 and 7 |

Two consequences for the order of work:

- the fixed system must own its state (step 5) before a loop can write to it directly. Until then the loops stay
  OLF's, and reach it through OLF's events;
- a loop in array form writes to the journal like a contingency does. Its decisions are then part of the delta of a
  lane, and undone with it.

## 9. Risks and open questions

- **Steps 1 and 2 lean on OLF's index and objects.** The live variables come from the index, the activity from the
  objects, the unknowns from OLF's state vector. The Jacobian of step 2 therefore still pays OLF's index update when
  buses are lost. Step 5 removes it.
- **Equations created at run time.** Until step 3, a zero-impedance network that splits or merges under a voltage
  control costs a compilation and a new analysis.
- **Pattern growth.** With transformer and shunt voltage control the pattern grows once or twice, when a reactive
  balance takes a tap or shunt slot for the first time. Each growth is one more analysis.
- **Object terms are slow.** Their entries are searched at each fill. Visible on the PEGASE fixtures, which have
  thousands of shunts.
- **Results change in the last digits.** The fixed matrix puts the equations in another order, so the LU differs in
  rounding: 2e-10 on the violations of rte6515 (6.4). Comparisons of security analyses must use the thresholds of
  `docs/increased-violations-thresholds.md`, or they count re-reported violations.
- **Reused pivots.** Their growth is measured on rte6515 only: 1.1e-2 at worst, no failure (6.4). The guard is
  KLU's growth check, and the fallback a full factorisation. On the PEGASE fixtures the results are the same as
  before, but the growth itself was not recorded.
- **One behaviour means no way back by configuration.** A regression on a kind of network the suite does not cover
  needs a code change. The guards are OLF's 1,690 tests and the fixtures.
- **Systems without a pairing.** The equations of asymmetrical systems and of AC/DC converters have no entry in
  `AcPlacement`: they take free slots by the general rule, and their tests pass. Whether their pattern stays fixed
  under switches is not studied.
- **Still on the matrix of the index:** the DC Jacobian until step 4, the fast decoupled solver, the voltage
  magnitude initializer, and Kinsol in Newton-Krylov.

## 10. Next steps

1. Step 3: equations created at run time keep their slot.
2. Step 4: the DC Jacobian.
3. The security analysis of 6.4 on all 7,422 lines of rte6515, against G. Jamgotchian's `OlfN1Bench` figures.
4. Share this note with S. Prabakaran and G. Jamgotchian.
5. In olf-gpu, when a GPU is at hand: the structure dump (a native getter for the pattern, and the same text format).

## Appendix: what happens to the existing PRs and branches

| Item | Fate |
|---|---|
| Upstream #1521, #1564, `minimize_zeros_in_diagonal_matrix` | independent: they improve OLF's own solver |
| Performance branch | its solver parts are independent; its restore, violation and array parts feed steps 5 and 6; its bench is the yardstick for the security analysis |
| Fork #101 (`klu_refactor` first) | built into the fixed Jacobian for AC; still of use for DC until step 4 |
| Fork #105 (reactive limits on arrays) | pilot of step 6 |
| Fork #108 (monitored branch index) | step 6 |
| Fork #119, #135 (partial restores, restore hooks) | interim for olf-gpu; replaced by the journal of step 5 |
| Fork #120 (`getSortedVariablesToFindOnly`) | interim for olf-gpu; not needed once slots are the rows |
| Fork #127 (`resultWithoutNetwork`) | kept: it is the lane result hook |
| Fork #130, #131 (values-only updates) | interim; replaced by steps 5 and 6 |
| Fork #146 to #149 and the 2.4.1 stack (extension points) | unchanged; several accessors become unnecessary after step 7 |
| Fork branch `claude/acnetworkvector-variable-index` | used by step 5 |
| Fork alternative-equations branches | replaced by the fixed system; their reproducers become cases of its test |
| olf-gpu #50 (outer loops in array form) | step 6 |
| olf-gpu #52 (duplication cleanup) | the packages that do not touch structure code now; the rest is deleted by step 7 |
| olf-gpu `StableRowIndexing`, `ReactiveRelocation*`, `GpuRowMode` | the reference the CPU placement is compared with in step 7; then deleted, except the device numerics |
| olf-gpu `batch/plan` | tested against recorded deltas in step 7, then removed family by family |
