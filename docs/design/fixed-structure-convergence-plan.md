# One fixed structure for KLU and cuDSS: a convergence plan

*olf-gpu design note, November 2026. Third version: the fixed structure is built as new code in OLF, seven steps are
done, and what is left follows one rule: the device does nothing of its own.*

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
  27% and 37%;
- the results are the same: same statuses, and the 71,483 violations in common agree to 2e-10.

**Step 3 is done** (section 6.5, fork PR #152): an equation that OLF removes and creates again keeps its slot. On
153 busbar-section contingencies of rte6515, which split zero-impedance networks, 177 matrices were created before
the stack, 35 with step 2 and 4 with step 3, for the same results.

**Step 4 is done** (section 6.6, fork PR #153): the pattern holds from the start the slots the controls can free,
as the GPU's does. Every security analysis run on the fixtures now creates one AC matrix, analysed once: with
transformer and shunt voltage control on, and on the busbar sections of rte6515.

**Step 5 is done** (section 6.7, fork PR #154): the shunt terms are term arrays in the vectorized system. The fill
of the fixed matrix is 67 to 70% faster on the PEGASE fixtures, which have thousands of shunts, and not slower on
rte6515, which has 121. The results are identical to the last bit.

**Step 6 is done** (section 6.8, fork PR #156): the terms held as objects have destinations too, and a placement
no longer goes through a set for every equation. With it, every term of the structure has its slots and its
destinations in arrays.

**Step 7 is done** (section 6.9, fork PR #157): no slot holds an equation that does not depend on its variable.
Around zero-impedance branches, and for a control that holds the voltage of a bus it is not next to, the equations
exchange their slots, by moves found once in the structure and held by the pattern from the start. On 153
busbar-section contingencies of rte6515 the diagonal had 62 zeros on average and KLU refused 32 of 609 updates of
its factors. It now has none, and KLU refuses one, for a reason of physics. This is the placement the device had to
do on its own.

**What is left follows one rule: the device does nothing of its own** (section 7.1). It reads the pattern, the
placement, the parameters of the terms and the changes of each contingency from the CPU structure. So whatever the
device needs and the CPU does not hold yet is built on the CPU: every term family as arrays (step 8), and a
contingency as a recorded list of changes (step 9). The second is also what lets the CPU restore in proportion to
what changed: applying a contingency through OLF is 0.2 to 0.4% of its cost, restoring the network 6 to 7%
(section 6.10).

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

**Placement, as built.** One table and three rules, with no case named:

1. Each equation type names the type of the variable it is paired with, on its own element (`AcPlacement`). The
   equations OLF toggles in pairs name the same variable.
2. When two active equations are paired with one variable, the one that fixes the variable (`x = target`) keeps the
   slot. It never moves.
3. An equation takes its own slot when it depends on its variable. The others take a slot whose variable they
   depend on, among their **candidates**; an equation in the way moves the same way. This is the search of an
   augmenting path.

The candidates are found once and are in the pattern from the start. They are the slots the controls can free
(step 4, 6.6): the voltage of a controller bus, a tap, a susceptance. And they are the moves that keep zeros off
the diagonal (step 7, 6.9): around a zero-impedance branch, and towards a control that is not next to the bus it
holds.

A slot is *live* when an active term depends on its variable. A free slot is a live slot that holds no equation.
A slot that is not live holds `x = x`.

Rule 3 gives the GPU's relocations without naming them. A voltage control that holds a bus's voltage frees exactly
one variable: the voltage of a controller, a tap or a susceptance. The displaced reactive balance takes its slot
when the two are neighbours; when they are not, it takes the voltage of a neighbour, whose balance moves on. Any
placement is correct: it only puts the equations in another order. One without a zero on the diagonal is what a
factorisation that does not pivot needs, and what lets KLU keep its pivots.

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

Step 1 follows OLF's objects and index. Owning these arrays is step 10. It is what lanes and the array form of the
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

Recording is cheap as well: applying a contingency through OLF's code is 0.2 to 0.4% of a contingency on the CPU.
What OLF's way costs is the restore, 6 to 7%, because it rewrites the whole network (6.10). The delta is what makes
the restore proportional to what changed.

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
| Structure: slots, pattern, candidates, placement, destinations | built once, in OLF | read from the CPU structure, as it is |
| Factorisation | one analysis per structure; `klu_refactor` per iteration; `klu_factor` as fallback | uniform batch, no pivoting; a lane whose solve fails is solved again alone, with matching and pivoting; adaptive refinement |
| Fill | one thread per lane; each output written where its equation is now | one thread per element; `atomicAdd` |
| Physics | OLF's term arrays and terms | kernels generated by `java2cuda` from the same term classes, on the vectors of the term arrays |
| Scope | `ALL` or `ACTIVE` | whole scope up front |

Until step 7 the device column also held a placement of its own: the relocation of the couplers, a row repair and a
cross identity. The structure does that work now (6.9), for KLU as well.

## 5. Three points where the efforts disagree

**Inactive terms: keep them or compact them?** #1564 keeps every term in the structure and writes 0 when inactive.
The performance branch keeps compact tables of the active ones and measured a 25% gain from not walking 15,000
inactive open-branch terms. Both are right, about different things. The structure keeps everything. The loop over a
family that is mostly inactive should run over a list of its active elements. Step 1 keeps everything and tests the
activity of each term; the active lists are to build with the destinations of the object terms (4.4).

**Who decides a row: the equation or the variable?** Section 4.1: the variable.

**Remote voltage control: move the target or the balance?** Section 4.2: the target keeps its variable, the balance
moves.

## 6. What steps 1 to 7 established

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
| CPU per contingency, before | 72.4, 75.8, 74.1 ms | 146.5, 141.6 ms | 186.2, 190.0 ms |
| CPU per contingency, fixed | 44.4, 49.5, 49.5 ms | 105.7, 102.7 ms | 116.3, 121.6 ms |
| Violations, before / fixed | 17,221 / 17,218 | 31,979 / 31,979 | 23,284 / 23,284 |

The gain is 35 to 39% on rte6515, 27% on pegase9241 and 37% on pegase13659. The three violations that differ on
rte6515 are the re-reported one above.

A first version of this table gave 158 and 144 ms for pegase9241, a gain of 9%, and blamed its shunt terms. That was
one round of each code while the machine was slowing down. Every time above is now one of several rounds that
alternate between the two codes. With the default parameters, transformer and shunt voltage control are off: 6.6
measures them.

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

### 6.5 Step 3: an equation that is removed and created again keeps its slot

Fork PR #152, stacked on #151. When a zero-impedance network splits or merges, OLF removes the distribution
equations of the voltage controls it holds and creates them again, with other terms. With step 2 alone, each time,
the arrays were compiled again and the matrix analysed again. A busbar-section or coupler contingency does this in
a node-breaker security analysis.

A single equation now keeps its number in the structure, by type and element. Removed, it holds nothing. Created
again, it takes its number back and its slot with it, and the rows of its new terms are added to those it had. The
pattern stands when its columns already hold these rows, which is the case after a split: the terms of the equations
of a part are among those of the whole. 110 lines in the two classes of step 1.

It is the device's answer to the same case (section 7): the base groups once, subgroups per lane, nothing
re-created.

- On OLF's zero-impedance test network, 60 random steps compile the arrays once, against 8 times, and the pattern
  never changes.
- A security analysis on it, with two contingencies that split the zero-impedance network, creates 1 matrix against
  3 with step 2.
- OLF's suite passes: 1,691 tests.

On rte6515, 153 of its 6,876 busbar sections, default parameters, one run each:

| | Before the stack | Step 2 | Step 3 |
|---|---|---|---|
| Matrices created (one of them for the DC initialisation) | 177 | 35 | 4 |
| Time per contingency, wall | 121 ms | 106 ms | 92 ms |
| Statuses | 149 converged, 2 at the iteration limit | same | same |
| Violations | 13,533 | 13,533 | 13,533 |

The violations have the same values with the three codes, to 1.4e-9. Of the two matrices beyond the first that
step 3 still creates, one comes from an equation the structure has not seen: the angle equation of a bus that
becomes the reference when a contingency takes the reference bus out. Step 4 removes it (6.6).

### 6.6 Step 4: the pattern holds the slots the controls can free

Fork PR #153, stacked on #152. Until this step the pattern learnt where a displaced balance goes as it met each
placement, and grew each time: a new matrix and a new analysis. The device cannot grow its pattern, so it writes
the balance of a pilot on every candidate row beforehand (`ReactiveRelocation`, `ReactiveRelocationTables`), for
the main control and for each dependent one. The CPU now does the same.

- **Candidates.** A `Placement` gives the slots an equation may take besides its own. `AcPlacement` reads them from
  the controls of the network: the balance of a controlled bus on the slot of each controller of its voltage
  control and of the controls merged with it; the equation of a controlled branch on the slot of each controller
  of its reactive power control or of its phase control. 70 lines, the only new model knowledge.
- **Placement.** The candidates are in the pattern from the start. An equation that has to move takes a free
  candidate first, and the equations that have candidates choose before the others.
- **An equation the structure has not seen.** When a contingency takes the reference bus out, OLF creates the angle
  equation of the new one. The arrays are compiled again and the pattern is kept when the new arrays fit in it:
  that equation is only the diagonal of its slot. On the device a slack relocation is data of the lane.

On the 17 test networks, 60 random steps each, the pattern is now built once on every one, the four with
transformer or shunt voltage control included. OLF's suite passes: 1,694 tests.

AC Jacobian matrices created during a security analysis. Each fixture has 300 regulating transformers and 20
regulating shunts.

| Run | Before the stack | Step 3 | Step 4 |
|---|---|---|---|
| rte6515, 153 busbar sections | 176 | 3 | 1 |
| pegase9241, 121 lines, taps take over after the generators have settled | 526 | 9 | 1 |
| pegase13659, same | 495 | 3 | 1 |
| pegase9241, 121 lines, taps as variables from the start | 408 | 1 | 1 |
| pegase13659, same | 611 | 1 | 1 |
| rte6515, 256 lines, shunt control in the equations, incremental taps | 539 | 1 | 1 |

Statuses and violations are those of the code before the stack in every run, with values within 2e-9. rte6515 is
not measured with taps as variables: its base case does not converge in those modes, with either code. OLF's
default transformer mode is the incremental one, where the taps are not variables.

What is left to grow the pattern: an action that merges two groups of controls that were apart when the arrays were
compiled, and an equation created at run time on a variable that has no slot.

### 6.7 Step 5: shunt terms as term arrays

Fork PR #154, stacked on #153. In the vectorized system the shunt terms were the last large family held as
objects: one pair per shunt. The question was how to be efficient with 121 shunts and with 7,329.

The formula is not the cost. Measured on the fixtures, which have 121, 7,329 and 8,762 shunts:

| | Cost |
|---|---|
| A shunt term through its object, evaluated | 60 to 75 ns |
| A shunt term through its object, in the fill of the fixed matrix | about 280 ns |
| Shunts in one loop over arrays, written through destinations | 5 to 7 ns per shunt, for both terms |

Most of the 280 ns go to finding where each derivative lands. A term array removes that: its values come from a
loop over arrays, and its derivatives go through destinations computed once. The cost then follows the number of
shunts with nothing fixed, which is what makes one design fit both ends. It is the device's design for every
family, shunts included.

OLF vectorizes by gathering per equation, so a term array costs something for every bus whether it has a shunt or
not; that is why vectorizing shunts did not pay there. Writing per element does not have that cost.

The change: an `AcShuntVector`, computed by `AcNetworkVector` at each state update with the static formulas of the
two term classes; two evaluators; two hooks in the creator, overridden by the vectorized one as for the
closed-branch terms. The formulas still exist once. The system held as objects is not changed.

| | rte6515 | pegase9241 | pegase13659 |
|---|---|---|---|
| Fill of the fixed matrix, objects / arrays | 1.30 / 1.06 ms | 7.54 / 2.26 ms | 9.75 / 3.21 ms |
| OLF's state update and mismatch vector, objects / arrays | 1.66 / 1.08 ms | 2.67 / 1.97 ms | 4.87 / 2.81 ms |
| OLF's state update and matrix of the index, objects / arrays | 4.37 / 3.00 ms | 13.10 / 4.58 ms | 16.87 / 7.95 ms |
| CPU per contingency, before the stack | 93.4, 78.0 ms | 146.5, 141.6 ms | 178.4, 209.2 ms |
| CPU per contingency, step 4 | 73.9, 48.0 ms | 95.0, 103.6 ms | 122.5, 171.4 ms |
| CPU per contingency, step 5 | 59.1, 46.6 ms | 90.7, 88.9 ms | 112.2, 113.3 ms |

The security analyses of steps 4 and 5 give results that are identical to the last bit on the three fixtures. Over
a whole contingency the arrays gain 3 to 8%, best round against best round: the fill is one part of a contingency.
OLF's suite passes: 1,696 tests, among them a new one that compares each array element with the object term of the
other system.

The terms that are still objects: the `v = target` term of every bus, the terms of the reactive power distribution
equations, HVDC, load models, zero-impedance branches. olf-gpu checks the terms of the equation system against a
list: the two new evaluators have to be added to it when it is ported.

### 6.8 Step 6: destinations for the object terms, and where a contingency spends its time

Fork PR #156, stacked on #154. Two things the fill and the placement did again each time: the terms held as objects
looked up the slot of each variable in a map and searched its entry in the column, and every placed equation was
added again to a set through a boxed key. The object terms are now compiled like the term arrays, and an equation
on its own slot is a bit in a bit set.

- On a system held entirely as objects the fill goes from 14.7 to 8.4 ms on rte6515 and from 36.7 to 22.0 ms on
  pegase9241.
- In the default vectorized system the fill does not change; the placement gains about 5% of a contingency on
  rte6515. The results are identical to the last bit.

**The profile of a contingency**, with Java Flight Recorder on rte6515 after step 5 (531 contingencies, default
parameters). It is what the next steps on the AC side should follow.

| Part | Share of the security analysis |
|---|---|
| Violation detection (`LimitViolationManager`) | 22% |
| KLU refactorisation | 14% |
| Fill of the fixed matrix | 9% |
| Branch currents computed with `hypot` at every state update (`AcNetworkVector.updateClosedBranches`) | 9% |
| Mismatch vector (`EquationVector`) | 8% |
| State restore after a contingency | 6% |
| Reactive limits outer loop | 5% |
| Target vector | 4% |
| A map rebuilt for each load flow by the HVDC freezing outer loop | 4% |
| Placement | 4%, of which 3% in the set step 6 removes |

The linear algebra and the fill are now a quarter of a contingency. The largest single part is the detection of
violations, and three others are computations that a Newton iteration does not need: the currents, the map of the
HVDC loop, the target vector.

One caution: inside the fill, the recorder attributed 4% to the entry search of displaced equations. A counter
shows 25 displaced terms per fill, a few microseconds. Line-level attribution in that loop is not reliable; the
share of the fill as a whole is.

### 6.9 Step 7: no zero on the diagonal

Fork PR #157, stacked on #156.

An equation on a slot whose variable it does not depend on leaves a zero on the diagonal. KLU pivots around it, so
the results were right. Two things made it a step of its own:

- KLU refuses to update its factors in place when a pivot it chose has become zero in another state, and a new
  numerical factorisation follows: 32 of 609 updates on the busbar sections of rte6515;
- the batched factorisation of the device does not pivot. olf-gpu keeps those zeros away with a placement of its
  own, on the device and per lane (`docs/gpu-vs-olf.md`, 2.3 and 2.4).

Two kinds of equations leave such a zero:

- the two equations that tie the buses of a zero-impedance branch together (`ZERO_PHI`, `ZERO_V`), paired with the
  flows of the branch (`DUMMY_P`, `DUMMY_Q`), and the balances of a bus that only such branches connect;
- an equation that a control sends to a variable it does not depend on: the reactive balance of a bus whose voltage
  is held by a shunt, or a generator, of a bus that is not next to it.

**The moves.** Such an equation takes the slot of a variable it depends on, and the equation paired with that slot
takes the slot the first one leaves. On OLF's zero-impedance test network (buses 1, 2 and 3 tied by lines 1 and 3;
bus 1 holds its voltage and the angle reference):

| Slot | OLF's pairing | The structure | Once line 1 is lost |
|---|---|---|---|
| `BUS_PHI:2` | `BUS_TARGET_P:2`, a zero | `ZERO_PHI:1` | `ZERO_PHI:3` |
| `DUMMY_P:1` | `ZERO_PHI:1`, a zero | `BUS_TARGET_P:2` | `DUMMY_TARGET_P:1` |
| `BUS_PHI:3` | `BUS_TARGET_P:3` | `ZERO_PHI:3` | `BUS_TARGET_P:3` |
| `DUMMY_P:3` | `ZERO_PHI:3`, a zero | `BUS_TARGET_P:3` | `BUS_TARGET_P:2` |

The same on the voltage side. With a shunt of bus 1 that holds the voltage of bus 3, the reactive balance of bus 3
goes to the voltage of bus 1, and the reactive balance of bus 1 to the susceptance.

**How.**

- The structure finds the moves once: a breadth-first search from each such equation, through the equations paired
  with the variables it depends on, until one depends on the variable of the slot to fill. Most often the first one
  does, and the chain is an exchange. All the shortest chains are kept, as candidates: the pattern holds them from
  the start.
- An equation only moves to a variable of the kind of its own, which the placement policy gives. For AC: angles,
  phase shifts and active flows on one side; voltages, ratios, susceptances and reactive flows on the other. A
  reactive balance on an angle would be a non-zero entry in structure, and exactly zero on a flat start without
  losses.
- In each state an equation takes its own slot when it depends on its variable. The others are placed by the
  search of an augmenting path through the candidates, on slots whose variables they depend on. For the equations
  of these moves the dependence is asked of the active terms: around zero-impedance branches it changes with the
  state.

**Against the mechanisms of the device.** Each of them is one case of the search:

| On the device | What it does | In the structure |
|---|---|---|
| `CouplerTreeRelocation` | For a coupler of the base spanning tree, the balances of its child bus go on the flows of the coupler, and the equations of the coupler on the voltage and the angle of the child. Decided once, from the base tree | The exchange. On the test network above it gives what the device's rule gives, with the bus that holds the voltage as the root |
| `CouplerTreeReHoming` | In a lane whose contingency opens the coupler, the balances of the child come back | The search runs in each state: the balances come back, and the bus that is left with nothing but a coupler exchanges in its turn |
| Stock placement and row repair | The couplers off the tree keep their zeros. A neighbour row is added to the row, in the matrix and in the right-hand side | No zero is left to repair. A chain of moves does with a permutation what the repair does with a row addition |
| Row repair, for a control that is not next to its bus | The same row addition | The chain through the neighbour |
| `CrossIdentity` | When the own row of a fixed variable is taken, `x = target` is written on another row | Not needed: an equation that fixes its variable never leaves its slot; the others move around it |
| Robust retry | A lane whose solve fails is solved again alone, with matching and pivoting | Stays: it is a mode of the solver |

**Measured.** Security analysis, default parameters, one thread.

| | Step 6 | Step 7 |
|---|---|---|
| rte6515, 153 busbar sections: zeros on the diagonal, average per factorisation | 62 | 0 |
| rte6515, 153 busbar sections: updates refused by KLU | 32 of 609 | 1 of 609 |
| rte6515, 256 lines: updates refused | 0 | 0 |
| pegase9241, 259 lines, taps and susceptances as variables: updates refused | 0 | 0 |
| Pattern, rte6515 with busbar sections | 91,520 values | 91,993 (+0.5%) |
| Pattern, rte6515 | 91,028 | 91,094 (+0.07%) |
| Pattern, pegase9241 with taps and susceptances | 156,256 | 156,354 (+0.06%) |

- The update that is still refused is not a matter of structure. One bus of rte6515 has a branch of negative
  reactance that cancels the others: its own `dP/dφ` is 0.084 against 118.7 off the diagonal. KLU pivots off the
  diagonal there from the base case on, whatever the pairing, and one contingency next to it makes that pivot zero.
- Results: same statuses and same violations, values within 1.6e-10. One more report of a low voltage that is
  already violated before the contingency and sits on the reporting threshold.
- Time: unchanged. 67.8 ms of CPU per contingency before and after on the busbar sections, over three alternating
  rounds. The placement itself goes from 385 to 453 µs on rte6515, about twice per contingency.
- Tests: in every state of the random walks of 6.2, no slot holds an equation that does not depend on its variable
  wherever a placement can avoid it. The test establishes that with a matching of its own, among all the variables
  of each equation. Of about 3,000 states, 24 have no such placement: islands of zero-impedance branches whose
  voltage no bus holds, which OLF cannot solve either.

**Limits.**

- The chains are found in the structure, not in each state. When a contingency takes away every neighbour through
  which a control reaches its bus, the balance falls back on the slot the control gives it, with a zero on the
  diagonal, as before.
- The weak diagonals that the physics gives stay, like the one above.
- What is established is structural: no zero on the diagonal. Whether the batched factorisation then solves every
  lane on this placement is to be measured on a machine with a GPU.

### 6.10 What applying and restoring a contingency cost

olf-gpu's `batch/plan` (4,700 lines) exists because the device needs the changes of each contingency as plain data,
and OLF can only say what a contingency changes by applying it. There are two ways out. The CPU can record the
changes while it applies a contingency. Or the CPU can do as the plan does, and work out the changes without
touching the network. Which one is cheaper was measured.

Shares of the Java time of the contingency loop, with Java Flight Recorder: rte6515, default parameters, one thread,
on step 7. The recorder does not see the time inside KLU.

| | 255 lines | 153 busbar sections |
|---|---|---|
| Working out what the contingency disconnects (`PropagatedContingency.toLfContingency`) | 0.4% | 0.8% |
| Applying it through OLF (`LfContingency.apply`) | 0.2% | 0.4% |
| Redistributing the lost power | 0.8% | 1.6% |
| **Restoring the base state (`NetworkState.restore`)** | **7.3%** | **5.9%** |
| The post-contingency load flow | 61.9% | 78.6% |
| Violation detection | 25.7% | 9.5% |

olf-gpu's own notes say the same of its host side (`docs/tasks.md`, a run of 24,379 contingencies): OLF's apply
0.6 s in total; the plan about 4 s, and 33 s in a run where one of its replays extracted the whole network again
for each generator contingency; the restore 3.6 to 14.9 s.

What the device does is two things, and only one is the efficient one:

- **how it derives a contingency**: the plan re-implements OLF's rules without touching the network. It is not
  cheaper than applying, which is the cheapest part of all, and it is where the mirror bugs are;
- **how it holds a contingency**: a small delta on fixed base arrays, with nothing to restore. That is what the CPU
  lacks. It pays 6 to 7% of each contingency to rewrite every bus, generator and branch, whatever the contingency
  touched. olf-gpu already avoids most of it with the partial restore of fork #119, which the CPU security analysis
  does not use.

So the CPU takes the device's way of holding a contingency, not its way of deriving one. This is step 9: OLF's own
apply produces one recorded delta per contingency, and it has three readers.

- **The CPU's undo.** Only what was written is restored.
- **The lanes of the device.** The delta is the data of a lane. This is what replaces `batch/plan`.
- **CPU threads as lanes**, later. Threads would share the structure and the base state and own value arrays only.
  Today each thread loads its own network and runs its own base case. It needs the solver to hold its own state
  first (step 10).

Two cautions. The outer loops also modify the network during the post-contingency load flow, so the record has to
be complete: elements marked as written at the setters, not a list of listener events. And what the recording
itself costs is not measured yet.

## 7. The plan

OLF's CPU side first. Each step is its own pull request on the OLF fork, stacked on the previous one. Step 1 adds
code only. From step 2 on there is one behaviour: the fixed matrix replaces the existing path, with no option.

| Step | Content | Exit |
|---|---|---|
| 1, done | The fixed view of an equation system: slots, placement, pattern, fill, tests | Sections 6.1 to 6.3, fork #150 |
| 2, done | The AC Jacobian solves on it: one sparse matrix on the pattern, filled in place, analysed once, refactored on the previous pivots with a full factorisation as fallback. No solver type and no option; OLF's engine, outer loops, security analysis and sensitivity analysis call it as before | Section 6.4, fork #151 |
| 3, done | Equations created at run time keep their slot. OLF removes and re-creates distribution equations when a zero-impedance network splits or merges; a re-created equation takes its number and its slot back, by type and element | Section 6.5, fork #152 |
| 4, done | The pattern holds from the start the slots the controls can free: candidates read from the controls of the network, as the GPU's relocation tables. The pattern is kept across a compilation when the new arrays fit in it | Section 6.6, fork #153 |
| 5, done | Shunt terms as term arrays in the vectorized system | Section 6.7, fork #154 |
| 6, done | Destinations for the terms held as objects; placements checked once | Section 6.8, fork #156 |
| 7, done | No zero on the diagonal: the equations around zero-impedance branches, and the balance of a bus held by a control that is not next to it, exchange their slots by moves found once in the structure | Section 6.9, fork #157 |
| 8 | Every term family as arrays, with its parameters in vectors: open branches, HVDC in AC emulation, load models, the terms of zero-impedance branches, the distribution equations, the `x = target` terms. What the device reads has to exist on the CPU | No term held as an object in the default AC system, and no object path in the fill. Results identical to the last bit |
| 9 | A contingency as a recorded delta (6.10). The elements that a contingency and the outer loops write are marked; the restore rewrites those only; the same record is the data of a lane | Restore in a time proportional to what changed: 6 to 7% of a contingency today. On the fixtures, the record of each contingency is what the device's planners derive |
| 10 | Its own state. The slots hold the unknowns, the targets and the activities; `AcNetworkVector` computes on them (fork branch `claude/acnetworkvector-variable-index` decouples it from the index); the first iteration from the delta | No index rebuild during a security analysis |
| 11 | Outer loops and results on arrays (section 8); lanes | Decisions bit-identical to the reference vectors. Four-thread slowdown per thread close to lightsim2grid's 1.2 |
| 12 | The GPU repository reads the CPU structure (7.1): the pattern as it is, the placement, the vectors of the term arrays, the deltas. First a dump of its structure in the text format of `FixedEquationSystem.write`, compared with the CPU's. Then its extraction, its pattern construction, its placement and its planners are deleted | olf-gpu's suite passes with no structure code of its own |
| DC | The DC Jacobian on a fixed matrix: a `DcPlacement`, and the same `FixedJacobian`. After the AC side | One behaviour for AC and DC. OLF's DC load flow, security analysis and sensitivity tests pass on it |

Steps 8 and 9 exist because of the rule of 7.1: they build on the CPU what the device would otherwise build for
itself. Both also pay on the CPU: step 8 removes the object path of the fill, step 9 most of the restore. The other
gains on the AC side follow the profile of 6.8, not a step number.

**A rule for every step:** when a case is unclear, look first at what the GPU does for it. It already runs these
cases against OLF's results, so its answer is the reference, as in 6.4.

On step 3: the device never re-creates anything. It builds the distribution equations of the base groups once, gives
each lane the subgroups a split produces, and replays OLF's election and spanning tree on the host
(`VoltageControlReMerge`, `ZeroImpedanceTreeReplay`). On the CPU OLF does the election itself; the fixed system only
recognises a re-created equation as the one that held the slot.

**What the upstream efforts become.** They improve OLF's own solver, independently: the pairing in the index
(`minimize_zeros…`, the performance branch), `klu_refactor` first (fork #101), the scatter fills (#1521, #1564).
The fixed Jacobian does not depend on them, and for the AC system it replaces what they speed up. What the
performance branch does outside the solver (restore, violation pre-filter, bus voltages in arrays) feeds steps 9
to 11.

### 7.1 The device does nothing of its own

The rule for everything that is left: the device reads the CPU structure and adds nothing to it. It does not add
entries to the pattern, it does not place equations, it does not pack the parameters of the elements, and it does
not work out what a contingency changes. What it keeps is the solver: the kernels, generated from OLF's term
classes, and the batched factorisation.

So each part of olf-gpu that builds structure is replaced by something the CPU holds, and where the CPU does not
hold it yet, it is built on the CPU first.

| Part of olf-gpu | Lines | What replaces it | State |
|---|---|---|---|
| Row indexing and reactive relocation: `StableRowIndexing`, `ReactiveRelocation`, `ReactiveRelocationTables`, `ReactiveRelocationCandidates`, `ExtractionColumns` | about 850 | The slots, the equation of each slot and its candidates | In the structure since step 4 |
| Row modes of the base state: half of `GpuAcActivationExtractor` | about 230 | The equation each slot holds | In the structure since step 1 |
| Native pattern construction: `csr_pattern.cu`, `relocation_pattern.cu`, `phase_control_pattern.cu` | part of 3,200 | The pattern of the structure, taken as it is | In the structure. Since step 7 it also holds the moves the device made for itself |
| Device-only placement: `CouplerTreeRelocation`, `CouplerTreeReHoming`, `CrossIdentity`, the zero-diagonal repair | 620 and native | The placement of step 7 (6.9) | In the structure. To confirm on a GPU: that the batched factorisation solves every lane on it |
| Validation of the equation system: `GpuAcValidation` | 180 | A term the structure compiles is known. One check is left: the device has a kernel for the family | In the structure |
| Element parameters and family packs: `GpuAcFamilyPacker` and the packs | about 700 | The vectors of the term arrays. Closed branches and shunts have them; open branches, HVDC, load models, zero-impedance branches and the distribution equations are objects on the CPU | Step 8 |
| Planners: `batch/plan` | 4,700 | The recorded delta of each contingency (6.10) | Step 9 |
| Distribution and outer-loop packs: `DistrQPacker`, `ReactiveLimitsPacker`, `DistributedSlackPacker` and others | about 1,400 | The inputs of the outer loops, as arrays | Step 11 |

**Why the plan is there, and why it goes.** `batch/plan` holds, for each contingency, what changes: the rows of the
islanded buses, the generator outages and their targets, the lost power, the weights of the reactive distribution
when a controller is lost, the voltage controls that merge again after a busbar loss, the slack when it moves. It
derives all of it by replaying OLF's rules (`LfContingency.processLostGenerators`, the spanning tree of
`LfZeroImpedanceNetwork`, the election of the controlled bus), because the batch must know the changes of every
lane before it solves, and OLF only says what a contingency changes by applying it. Once the CPU records the
changes while OLF applies the contingency, the device reads the record, and there is nothing left to replay. The
cost of applying is not an obstacle: 0.2 to 0.4% of a contingency (6.10).

The order that keeps it safe, on a machine with a GPU:

1. the structure dump: olf-gpu writes its structure in the text format of `FixedEquationSystem.write`, and the two
   texts are compared network by network. It needs a native getter for the pattern;
2. the device takes the pattern, the slots and the placement from the CPU structure. The row indexing, the
   relocation classes, the native pattern construction and the device-only placement are deleted. Each lane is
   then solved on a matrix without a zero on the diagonal, and the robust retry is the measure of what is left;
3. the planners are checked against deltas recorded on the CPU, contingency by contingency, before they are
   removed.

76 of the 99 test classes of olf-gpu need a GPU, so none of this can be validated on a machine without one.

## 8. Room for the outer loops on arrays

The outer loops are due to be rewritten in array form: gather, decide, derive, apply, with the decisions written
once and generated for the device (`docs/outer-loop-array-form-proposal.md`; fork PR #105 is the pilot on reactive
limits). The fixed system is built to be what they read and write.

| The loops need | The fixed system gives | State |
|---|---|---|
| The reactive balance of every bus that holds its voltage; the active balance of the slack | `balance[]`, filled at each fill for every bus, whatever its slots hold | built and tested |
| The unknowns by variable | `x` by slot; a slot never moves | step 10 (today OLF's state vector) |
| To switch a bus between holding its voltage and its reactive balance, a tap between fixed and controlling | a write of "which equation this slot holds", among the alternatives of the slot | step 10: today the switch goes through the `LfNetwork` setters and OLF's events |
| To change a target | a write of `target[slot]` | step 10 |
| That a decision never changes the matrix shape | every alternative a loop can activate is in the pattern under `ALL` | built: measured on the fixtures |
| Vectors indexed by something stable (limits, participation, groups) | the slot of a variable | built |
| The same decisions per lane on the device | the per-lane overrides of the vectors are deltas on the same arrays | steps 11 and 12 |

Two consequences for the order of work:

- the fixed system must own its state (step 10) before a loop can write to it directly. Until then the loops stay
  OLF's, and reach it through OLF's events;
- a loop in array form writes to the journal like a contingency does. Its decisions are then part of the delta of a
  lane, and undone with it.

## 9. Risks and open questions

- **Steps 1 and 2 lean on OLF's index and objects.** The live variables come from the index, the activity from the
  objects, the unknowns from OLF's state vector. The Jacobian of step 2 therefore still pays OLF's index update when
  buses are lost. Step 10 removes it.
- **Equations created at run time.** A re-created equation keeps its slot (6.5). One the structure has not seen
  costs a compilation; the pattern and the analysis are kept when its entries are already there (6.6).
- **Pattern growth.** None on the fixtures since step 4. Left: an action that merges two groups of controls that
  were apart when the arrays were compiled, and an equation created on a variable that has no slot.
- **Zeros on the diagonal.** None on the fixtures since step 7. The moves are found in the structure, not in each
  state: a contingency that takes away every neighbour through which a control reaches its bus leaves one (6.9).
  KLU then pivots; on the device the lane goes to the robust retry. How often is to be counted on a machine with a
  GPU.
- **The device on the CPU placement.** Step 7 establishes that the diagonal has no zero, not that the batched
  factorisation solves every lane on it. Weak diagonals that the physics gives remain, like the bus of rte6515 of
  6.9, and they are in olf-gpu's own placement today as well.
- **Object terms.** They have destinations since step 6. They are still objects: what the device needs of them,
  their parameters in vectors, is step 8.
- **A recorded delta must be complete.** A change of the network that goes through no marked setter would not be
  restored. The guard is a comparison with the full restore, state by state, in the tests of step 9.
- **Measurements on a shared machine.** Its speed varies by 20 to 30% from one minute to the next. One figure of
  this note was wrong for that reason and is corrected in 6.4. Only rounds that alternate between two codes
  compare them.
- **An instrumented build is not comparable to the last digit.** Runs of one build are identical to the last bit.
  A build that registers a shutdown hook to print counters is not the same run: the hook takes one identity hash,
  the order of iteration of some of OLF's hash sets changes with it, and so does the rounding. On the busbar
  sections of rte6515 that re-reported one threshold-edge violation 149 times. Counters go in an instrumented
  build; results are compared between plain builds.
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
- **Still on the matrix of the index:** the DC Jacobian until its own step, the fast decoupled solver, the voltage
  magnitude initializer, and Kinsol in Newton-Krylov.

## 10. Next steps

1. Step 8: the term families that are still objects, as arrays. Open branches first: they are the family a
   contingency activates.
2. Step 9: the recorded delta, starting with the restore of the CPU security analysis, where it is 6 to 7% of a
   contingency.
3. On the AC side, by share of a contingency (6.8): the detection of violations; the currents, the map of the HVDC
   loop and the target vector, which a Newton iteration does not need; the mismatch from the fill.
4. olf-gpu, on a machine with a GPU (7.1): the structure dump, then the pattern and the placement from the CPU
   structure, and the two shunt evaluators of step 5.
5. The security analysis of 6.4 on all 7,422 lines of rte6515, against G. Jamgotchian's `OlfN1Bench` figures.
6. The DC Jacobian, after the AC side.
7. Share this note with S. Prabakaran and G. Jamgotchian.

## Appendix: what happens to the existing PRs and branches

| Item | Fate |
|---|---|
| Upstream #1521, #1564, `minimize_zeros_in_diagonal_matrix` | independent: they improve OLF's own solver |
| Performance branch | its solver parts are independent; its restore, violation and array parts feed steps 9 to 11; its bench is the yardstick for the security analysis |
| Fork #101 (`klu_refactor` first) | built into the fixed Jacobian for AC; still of use for DC until the DC step |
| Fork #105 (reactive limits on arrays) | pilot of step 11 |
| Fork #108 (monitored branch index) | step 11 |
| Fork #119, #135 (partial restores, restore hooks) | the partial restore is what step 9 gives the CPU itself, driven by the record instead of the caller; the hooks stay interim for olf-gpu |
| Fork #120 (`getSortedVariablesToFindOnly`) | interim for olf-gpu; not needed once slots are the rows |
| Fork #127 (`resultWithoutNetwork`) | kept: it is the lane result hook |
| Fork #130, #131 (values-only updates) | interim; replaced by steps 10 and 11 |
| Fork #146 to #149 and the 2.4.1 stack (extension points) | unchanged; several accessors become unnecessary after step 12 |
| Fork branch `claude/acnetworkvector-variable-index` | used by step 10 |
| Fork alternative-equations branches | replaced by the fixed system; their reproducers become cases of its test |
| olf-gpu #50 (outer loops in array form) | step 11 |
| olf-gpu #52 (duplication cleanup) | the packages that do not touch structure code now; the rest is deleted by step 12 |
| olf-gpu `StableRowIndexing`, `ReactiveRelocation*`, `GpuRowMode`, `CouplerTree*`, `CrossIdentity`, the row repair | the reference the CPU placement is compared with in step 12; then deleted. The robust retry stays |
| olf-gpu `batch/plan` | replaced by the recorded deltas of step 9: checked against them contingency by contingency in step 12, then removed |
