# One fixed structure for KLU and cuDSS: a convergence plan

*olf-gpu design note, November 2026.*

This note extends the design note of PR #51 (`docs/design/fixed-structure-equation-system.tex`). That note compared
OLF's alternative equations, the GPU superset matrix and upstream draft #1521, and proposed a plan. Three upstream
efforts have appeared since. This note places all of them in one design and replaces the plan section of #51.

References to OLF code are to `main` at `33fb055a` unless a branch is named. "rte6515" is MATPOWER `case6515rte`.

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

The design that all six fit into has six notions (section 4):

- **Slot.** Every variable owns one row and one column, numbered once. An equation is written in the column of the
  variable it is paired with.
- **Alternative.** The equations that can occupy the same slot. One is active. If none is, the slot holds
  `x = target`. OLF's creator already creates all of them.
- **Pattern.** The union of the alternatives of each slot, built once, sorted. KLU and cuDSS use the same arrays.
- **Destination maps.** For each element and each of its outputs, the index it adds into. Built once, never
  invalidated.
- **State.** A few arrays: the unknowns, the active alternative of each slot, the state of each element, the
  targets. A PV/PQ switch or a contingency changes only these.
- **Delta and lane.** A contingency is a short list of writes to the state arrays. A CPU thread, or one entry of a
  GPU batch, is a lane: its own state arrays over the shared structure.

The plan (section 7) has three stages. Every step is useful to the CPU alone, which is what makes it proposable
upstream.

| Stage | Content | Nature |
|---|---|---|
| A | The same matrix as today, numbered and filled better | Convergence of code that exists; no new concept |
| B | The fixed structure, as an opt-in mode with today's path as fallback | The slot model in OLF; the GPU reads it |
| C | State as data: outer loops, results and contingencies on arrays; lanes | CPU threads without network copies; GPU without planner mirrors |

## 2. Where the six efforts stand

| Effort | What it does | What it leaves open |
|---|---|---|
| Alternative equations | One column per equation, several bodies, one active. PV/PQ switches and islanding keep the pattern. Partial Jacobian update after a contingency. | Excludes buses with tap or shunt voltage control, zero-impedance branches, converters, fictitious buses. Falls back to a rebuild when the system stops being square. 3,700 lines in the `equations` package. |
| GPU superset | One row per variable, an integer per row saying which equation it holds, a mask per element, destination arrays for the fill, per-lane data for a contingency. Covers every case the line above excludes. | The host re-derives all of it from the `LfNetwork`: about 3,100 + 3,400 + 4,700 lines that mirror OLF. One mirror bug made 4,183 of 7,494 contingencies wrong. |
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
   performance branch. #1564 and the performance branch conflict on `EquationArray`, `EquationTermArray` and
   `JacobianMatrix`.

**Measured facts the design relies on.** All from G. Jamgotchian's report (rte6515, 7,422 line outages of which
2,162 island a bus, plain Newton-Raphson, one thread, results identical before and after):

| Fact | Measure |
|---|---|
| Whole security analysis | 317 s to 47 s; lightsim2grid 43 to 49 s on the same machine |
| Unpaired numbering of the vectorised system | 44% more LU fill (211,114 against about 149,000 nonzeros), KLU about twice as slow, 1,507 failed refactorisations; none once paired |
| Identity for islanded buses instead of removing them | full factorisations 3,380 to 2 |
| Direct fills | Jacobian refill 1.74 to 0.60 ms; mismatch 0.50 to 0.09 ms |
| Not evaluating the inactive open-branch terms (about 15,000 per equation array) | 76.7 to 58.0 s |
| Default configuration (outer loops on) | 5% gain only: PV/PQ switches still rebuild the structure |
| Four threads | OLF slows down about 2 times per thread, lightsim2grid about 1.2 times |

**Not measured yet:** the cost in KLU of a union pattern that includes the PV/PQ alternatives. It is the first
measurement of section 8.

## 3. What the GPU does today, in OLF's vocabulary

For readers who have not seen this repository. Details: `docs/gpu-vs-olf.md`, sections 1 and 2.

- **Rows are OLF's variable rows.** `fixedmatrix.StableRowIndexing` numbers nothing: a row is `variable.getRow()`.
  Each equation is written on the row of the variable it is paired with: `BUS_TARGET_P` on the bus's `BUS_PHI`,
  `BUS_TARGET_Q` or `BUS_TARGET_V` on its `BUS_V`, `DISTR_Q` on the controller's `BUS_V`, `BRANCH_TARGET_RHO1` and
  `DISTR_RHO` on `BRANCH_RHO1`. The system is square by construction, and the derivative of an equation by its own
  variable is on the diagonal.
- **Row mode.** One small integer per row and per lane says which equation the row holds (`GpuRowMode`, six values).
  Mode 0, "fixed variable", writes `x = target`. It covers the slack angle (`BUS_TARGET_PHI`), the voltage of a PV
  bus (`BUS_TARGET_V`), an open coupler (`DUMMY_TARGET_P`, `DUMMY_TARGET_Q`), a fixed tap and a disabled bus. A PV to
  PQ switch is mode 0 to mode 4 on the bus's v row.
- **Fill, then apply.** The fill kernels write the balance equations on every row. A second kernel then rewrites the
  rows whose mode is not "equation": row zeroed, 1 on the diagonal, mismatch `x − target`.
- **Destination arrays.** For each element and each of its outputs, one integer: a Jacobian value index, a mismatch
  row, or "skip". A closed branch has 28 outputs: 4 flows and 24 derivatives. They are the 28 arrays that OLF's
  `AcBranchVector` already holds (`p1` to `q2`, `dp1dv1` to `dq2dr1`). There is one destination array per structure,
  shared by all lanes.
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

### 4.1 Slots

A **slot** is one variable to solve. It owns one row (the variable) and one column (the equation that currently
determines it). Slots are numbered once per structure. The numbering is a function of the variable set only.

Rule: *the column of an equation is the row of its slot's variable.*

This is what the GPU does, and it is the fix that `minimize_zeros…` and the performance branch are both after. They
approach it from the other side: they keep the equation columns and move the variable rows to match. The result is
the same matrix while nothing switches. It differs as soon as something does: if rows follow equations, a PV/PQ
switch moves rows; if columns follow variables, nothing moves.

Deriving rows from equations has two more costs, both visible today:

- the performance branch has to permute the state vector when rows are renumbered;
- `getSortedVariablesToFindOnly()` (fork PR #120), which olf-gpu calls on every post-contingency state, relies on
  the variable side of the index being independent of the equation side. With the row filler of `minimize_zeros…`
  it would return rows computed from stale equation columns, without an error.

The order of the slots is an internal choice (section 8, M3). The contract exposes it as arrays, never as a rule.

### 4.2 Alternatives: OLF already creates them

The equations that can occupy one slot are its **alternatives**. At most one is active. If none is, the slot holds
`x = target`, with the current value as target: this is the performance branch's ghost, and the GPU's "fixed
variable".

OLF's creator already creates every alternative up front and toggles them in pairs:

- `BUS_TARGET_V` is created, inactive, on every bus (`AcEquationSystemCreator.java:62-67`);
- `ZERO_V` and `DUMMY_TARGET_Q`, `ZERO_PHI` and `DUMMY_TARGET_P` are created together, the second with "inverted
  logic" (`:465-486`, `:497-519`);
- `BRANCH_TARGET_ALPHA1` and `BRANCH_TARGET_P` likewise (`:532-548`, `:563-584`).

So the slot model adds no equation. It names a relation the creator already maintains: which equations exclude each
other on which variable.

| Slot (variable) | Its alternatives, as OLF equation types |
|---|---|
| `BUS_PHI` of bus i | `BUS_TARGET_P` of i · `BUS_TARGET_PHI` · `BUS_DISTR_SLACK_P` · fixed (bus disabled) |
| `BUS_V` of bus i | `BUS_TARGET_Q` of i · `BUS_TARGET_V` of i · `DISTR_Q` · the `BUS_TARGET_Q` of a bus that i's generators control remotely · `BRANCH_TARGET_Q` · fixed (bus disabled) |
| `BRANCH_RHO1` of transformer t | `BRANCH_TARGET_RHO1` · the `BUS_TARGET_Q` of the bus t controls · `DISTR_RHO` |
| `SHUNT_B` of shunt s | `SHUNT_TARGET_B` · the `BUS_TARGET_Q` of the bus s controls · `DISTR_SHUNT_B` |
| `BRANCH_ALPHA1` of phase shifter t | `BRANCH_TARGET_ALPHA1` · `BRANCH_TARGET_P` of the controlled branch |
| `DUMMY_P`, `DUMMY_Q` of a zero-impedance branch | `DUMMY_TARGET_P`, `DUMMY_TARGET_Q` · `ZERO_PHI`, `ZERO_V` |

Seven of these types are the same alternative, `x = target`, with different targets: `BUS_TARGET_V`,
`BUS_TARGET_PHI`, `BRANCH_TARGET_RHO1`, `BRANCH_TARGET_ALPHA1`, `SHUNT_TARGET_B`, `DUMMY_TARGET_P` and
`DUMMY_TARGET_Q`. This is why six row modes are enough on the GPU.

**Placement rules.** They are the GPU's (`docs/gpu-vs-olf.md` 2.1 and 2.3), stated once for both solvers:

1. An equation that fixes a variable (`x = target`) sits on that variable's slot.
2. A balance sits on its own bus slot when that slot is free: P on φ, Q on v.
3. When a voltage target takes the v slot of a bus whose reactive balance must still hold, the balance moves to
   the slot that the control frees: the v of the first free controller, the tap, or the susceptance. The other
   controllers hold their `DISTR_*` equation on their own freed slot.

Rule 3 is where the alternative-equations branch and the GPU differ. The branch keeps the pilot's balance in place
and writes the voltage target in the controller's column, which costs one entry but leaves a structurally zero
diagonal. The GPU moves the balance, which costs a row of stored zeros per candidate and keeps the diagonals
nonzero when pilot and controller are neighbours. KLU works with both. The batched factorisation needs the second,
and nonzero diagonals also make `klu_refactor` succeed more often. The plan adopts the GPU's rule for both.

A balance that can sit on several slots has its slot as part of the state (the "carrier" of the GPU).

**The alternative-equations branch, seen from here.** It attaches the alternatives to an equation object
(`AlternativeEquation`, `ElementAlternative`) and duplicates bodies: each controller gets a copy of the voltage
target, and the bus's own `BUS_TARGET_V` "stays inactive forever". Attaching them to the variable instead reuses the
existing equations and makes the system square by construction. Its exclusions (tap and shunt control, zero
impedance) are the families whose control variables had no slot. Each becomes one row of the table above.

### 4.3 Pattern and scope

The **pattern** of a slot is the union of the entries of its alternatives, plus the diagonal. It is built once,
with sorted indices. OLF stores the transposed Jacobian by equation columns, which is the Jacobian by rows: KLU and
cuDSS take the same arrays.

Facts that bound the cost on the CPU:

- **The fixed alternative costs no entry.** Its only entry is the diagonal. Islanding, branch outages and slack
  relocation therefore need nothing beyond the base pattern. On the plain Newton-Raphson benchmark the union
  pattern *is* the base pattern, and the performance branch's ghost-mode results are the results of this design.
- **KLU accepts stored zeros.** Its analysis depends on the pattern only; `klu_factor` pivots inside the blocks;
  `klu_refactor` reuses the pivots and is checked (`klu_rcond`), with `klu_factor` as fallback.
- **The cost is where a slot gains entries.** Today the column of a PV bus's voltage target has one entry, and KLU's
  block triangular form sets it aside as a 1×1 block. With the reactive balance in the pattern as stored zeros, it
  no longer can. The same holds for remote-control candidates and `DISTR_*` equations. This is the unmeasured part.

The **scope** of a structure is the set of alternatives it includes. It follows from the run:

| Run | Scope |
|---|---|
| Plain Newton-Raphson, any contingency | base pattern only |
| Reactive limits on | plus the reactive balance on the v slot of buses with finite limits |
| Tap, shunt or phase control on | plus the alternatives of those control slots |
| Remedial actions | plus what `LfTopoConfig` declares, as today |

An activation outside the scope rebuilds the structure with a wider scope: one more analysis, counted and logged.
On the GPU it sends the contingency to the CPU or to a later batch.

If measurement M2 shows the reactive-limits scope to be expensive in KLU, the CPU can use an **adaptive scope**: a
slot gets its wide pattern the first time it switches, at the price of one rebuild, and keeps it. Over a security
analysis the set of buses that ever reach a limit is small and stable. The GPU needs the whole scope up front. The
contract is the same; only the scope parameter differs.

### 4.4 Destination maps

For each element family and each output of an element, a **destination**: the index it adds into. The encoding is
the GPU's: a Jacobian value index, a balance entry, or "skip".

- The maps are composed all the way: term output, to equation, to slot, to value index. #1521 stops at the
  flattened derivative vector, #1564 at a handler call per cell; the performance branch (`529723b0`) records the
  value index. The cells of #1564 (one per equation and variable) are the right unit to build them from.
- Under a fixed structure the destination of an output never changes. The maps are built once per structure and
  never invalidated. The invalidation logic of #1521 disappears.
- An output of a balance that rule 3 can move has a second destination and a selector slot, as on the GPU. On the
  CPU the move is cheaper done after the fill (4.6).

### 4.5 State

Everything that changes during a run is in a few arrays:

| Array | Content | Changed by |
|---|---|---|
| `x[slot]` | the unknowns | the solver |
| `activeAlternative[slot]` | which alternative the slot holds | outer loops, contingencies, actions |
| `elementState[family][element]` | on, off, or open at one side | contingencies, actions |
| `target[slot]` | the target of the active alternative | outer loops, contingencies |
| outer-loop vectors | limits, participation factors, groups | contingencies (stage C) |

Activity is data, not objects. The `MutableBoolean` per term of #1564 becomes a primitive array. The performance
branch measured 5 to 8 s on rte6515 for exactly this access (`6cf6827d`).

### 4.6 One fill algorithm

```
values[] = 0 ; balance[] = 0
for each dense family (closed branches):          # no branch: an element that is off contributes 0
    for each element e:
        out[0..K) = kernel(parameters[e], x)
        for each k: add out[k] at destination[e*K + k]
for each sparse family (branches open at one side, ...):
    for each e in its active list: the same
for each slot s whose active alternative is not its own balance:   # a short list
    rewrite column s and F[s] for that alternative, reading values[] and balance[] where it needs them
for the other slots: F[s] = balance[s] − target[s]
refactor, solve
```

Three properties:

- **The balance buffer is the primary output.** `balance[]` holds the P and Q balance of every bus, whatever its
  slots currently hold. The reactive output of a PV bus and the active mismatch of the slack are entries of it,
  which is what the reactive-limits and distributed-slack loops need. On the CPU it replaces the evaluation of the
  bus's terms by the outer loop. On the GPU it is the `dQbuf` buffer, which a second pass completes today.
- **Alternatives are derived from the natural fill.** A fixed slot zeroes its column and writes 1. A moved balance
  copies a column through a precomputed entry map. A `DISTR_*` equation combines the natural columns of its group.
- **Dense and sparse families are iterated differently.** This settles the first disagreement of section 5.

The first Newton iteration of a contingency needs no evaluation at all. It starts from the base state, so its
Jacobian values and its balance are the base ones, minus the contributions of the elements the contingency turns
off, plus its alternative switches. Both are computed through the maps in a time proportional to the delta. This
generalises two measured optimisations: the reuse of the base flows in the performance branch (`16f3dd24`, 7,421
reuses, all bit-identical) and the partial Jacobian update of the alternative-equations branch.

### 4.7 Deltas and the journal

A **delta** is a list of writes to the state arrays: (array, index, value).

The cheapest correct way to obtain one is to **record** it. With a journal open, OLF applies the contingency as it
does today (`LfContingency.apply`, then the listeners of `AcEquationSystemUpdater`). Every write to the state arrays
is logged with its old value. The log gives:

- **the undo**, in a time proportional to the number of writes. This replaces the full restore, and the incremental
  restore of the performance branch (`42d48c31`), which marks elements in the setters for the same purpose;
- **the delta**, for another thread or for a GPU lane.

A recorded delta is exact by construction: OLF computed it. That removes the class of bugs the GPU's planner
mirrors have (4,708 lines in `batch/plan` re-implementing `LfContingency` and the voltage-control re-merge).

Recording costs one application and one undo per contingency on the host. Where that is too slow, an **analytic
delta** is computed without applying. A branch outage that islands nothing is one element turned off. Analytic
deltas are tested against recorded ones, contingency by contingency, on every test network. The GPU's planners
become analytic delta providers under that test, then move to OLF or are deleted, family by family.

The performance branch's check mode is kept for the journal: after an undo, compare with a full snapshot and fail
at the first difference.

### 4.8 Lanes

A **lane** is one set of state arrays, with its Jacobian values and numeric factorisation, over a shared immutable
structure (pattern, maps, parameters, limits).

- On the GPU a lane is one entry of a batch. This exists.
- On the CPU a lane is one thread. Threads share the structure and own a few megabytes each on a network the size
  of rte6515. No `LfNetwork` copy per thread.

The performance branch found that OLF slows down about 2 times per thread on four threads, against 1.2 for
lightsim2grid, and attributes it to memory contention between large object graphs. Lanes remove the cause.

Lane results come from arrays too: flows from the kernels, violations from a limit table. On the CPU this is the
array pre-filter of the performance branch (`152470c7`) and fork PR #108. On the GPU it is the device violation
screen. `resultWithoutNetwork` (fork PR #127) is the hook both use.

### 4.9 What stays specific to each solver

| | CPU (KLU) | GPU (cuDSS) |
|---|---|---|
| Factorisation | one analysis per structure; `klu_refactor` and `klu_rcond` per iteration; `klu_factor` as fallback; one numeric object per thread | uniform batch, no pivoting; row repair, cross identity, coupler relocation, adaptive refinement, derived from the shared pattern |
| Fill | one thread per lane; compact lists for sparse families; move columns after the fill | one thread per element; `atomicAdd`; second destination chosen in the kernel |
| Physics | `AcNetworkVector` | kernels generated by `java2cuda` from the same term classes |
| Scope | may be adaptive | whole scope up front |

Outer-loop decisions are shared, not specific: written once in OLF's Java as pure functions of arrays, generated
for the device, and checked bit for bit against reference vectors (fork PR #105, `docs/outer-loop-array-form-proposal.md`).

## 5. Three points where the efforts disagree

**Inactive terms: keep them or compact them?** #1564 keeps every term in the structure and writes 0 when inactive.
The performance branch keeps compact tables of the active ones and measured a 25% gain from not walking 15,000
inactive open-branch terms. Both are right, about different things. The structure and the maps keep everything
(#1564). The loop over a family that is mostly inactive runs over a list of its active elements (performance
branch). That list is a cache derived from the state and maintained by the deltas. The closed-branch family is
dense: it keeps the branch-free loop.

The open-branch terms should also stop being separate term objects. A branch open at one side is a third state of
the branch element. Its outputs go to a subset of the closed branch's destinations, so it needs no pattern and no
map of its own.

**Who decides a row: the equation or the variable?** Section 4.1: the variable.

**Remote voltage control: move the target or the balance?** Section 4.2, rule 3: the balance.

## 6. What to take from each effort

| Source | Taken as is | Taken with a change | Replaced later by |
|---|---|---|---|
| GPU | variable-owned rows; placement rules; row mode; destination encoding; contingency as per-lane arrays; generated kernels | the balance buffer becomes the primary output | extraction (B4) and planner mirrors (C4) are deleted |
| Performance branch | benchmark against lightsim2grid; sorted pattern; `klu_refactor` with a check; identity for disabled buses, not counted by the stopping criteria; direct fills; activity array; partial target update; bus voltages in arrays; flat variable table; array pre-filter of violations; interleaved branch inputs; math-native patch; `PiModelArray` fix | pairing rule extended and applied from the variable side; compact active lists applied per family | `JacobianLuStructure` and the state-vector permutation (by B1); the setter tracker (by the journal, C3) |
| #1564 | cells; activity as a value; rows re-read only on a structure change | activity in a primitive array; value index written directly; additional terms as ordinary cells | |
| #1521 | scatter for `eval`; sequential read of term arrays | no filtering by activity under a fixed structure; composed to the value index | its invalidation logic |
| `minimize_zeros…` | the goal; the count of nonzero diagonal terms, kept as a test | pairing by type and from the variable side; single equations and controls included | |
| Alternative equations | the concept; the partial update; the non-square exception as a safety net; the tests and reproducers; forcing the mode on in CI | alternatives attached to the variable | `AlternativeEquation` and the per-family exclusions |

## 7. The plan

Durations are rough. Each step ends on a measurement.

### Stage A: the same matrix, numbered and filled better (2 to 4 weeks, mostly existing code)

| Step | Content | Built from | Exit |
|---|---|---|---|
| A1 | Standalone fixes: `PiModelArray.shiftOneTapPositionToChangeA1`; powsybl-math-native (zero-copy array access, `klu_rcond` check, GCC 14) | performance branch | merged; refactorisation 2.06 to 1.58 ms on rte6515 |
| A2 | Numbering: column of an equation = row of its slot variable; rows a function of the variable set; pattern with sorted indices | `minimize_zeros…`, `287a77a2`, `StableRowIndexing` | every active equation of every test network on a nonzero diagonal, except a listed set of remote placements; 149,000 LU nonzeros instead of 211,114 on rte6515; olf-gpu suite passes |
| A3 | `klu_refactor` first, with the check, when values and zeros changed | fork #101, `287a77a2` | no failed refactorisation on the 5,260 non-islanding outages; KLU time on them from 55 s to about 19 s, A2 included |
| A4 | One scatter for `eval` and `der`: cells, primitive activity, value index written directly, compact list for the additional terms | #1521, #1564, `529723b0`, `3ac7f6ea`, `6cf6827d`, `aff5d5d9` | values equal to the term walk on random activations; Jacobian refill at most 0.6 ms and mismatch at most 0.1 ms on rte6515 |

A2 and A4 touch different files except `JacobianMatrix`. They can be done in parallel by two people, A2 first into
`JacobianMatrix`.

### Stage B: the fixed structure (8 to 12 weeks)

Opt-in through a parameter. The legacy path stays and is the fallback.

| Step | Content | Built from | Exit |
|---|---|---|---|
| B1 | Slots for bus variables. Slot table in `EquationSystemIndex`; equation-to-slot assignment given by the creator; fixed alternative; shared slots on φ and v, including `DISTR_Q` and the moved balance; union pattern by scope; an activation changes values only | ghost mode, alternative equations, GPU placement rules | rte6515 N-1: one structure build in plain Newton-Raphson, and no rebuild on PV/PQ switches with reactive limits on; results equal to the legacy path within the solver tolerance; OLF test suite green with the mode forced on |
| B2 | The fill of 4.6: maps never invalidated, alternatives applied after the natural fill, balance buffer, first iteration from the delta | A4, `16f3dd24`, partial update | no map rebuild during a security analysis; the reactive-limits loop reads the balance buffer |
| B3 | Control slots, one family at a time: tap, shunt, phase, then zero impedance | GPU `ReactiveRelocation*`, `CouplerTree*` | per family: no fallback left on rte6515 with the production parameters |
| B4 | The contract (`FixedStructureView`: slots, pattern, alternatives, maps, state). `JacobianMatrix` fills from it. The GPU builds `SharedStructure` from it, side by side with today's extraction, then alone | #51 phases 3 and 4 | pattern, destinations, row modes and targets equal on every test network; olf-gpu suite green with the extraction deleted |

The GPU side of B4 can start once B1 exists: the side-by-side comparison runs on the networks B1 covers.

### Stage C: state as data (8 to 12 weeks, overlapping B from B2)

| Step | Content | Built from | Exit |
|---|---|---|---|
| C1 | Network state and results in arrays: bus voltages, flat variable table, limit table, violations from arrays | `db71b65e`, `7258e330`, `152470c7`, fork #108 and #127 | same violations; no network round trip per contingency in plain Newton-Raphson |
| C2 | Outer loops in array form: gather, decide, derive, apply; decisions written once | `docs/outer-loop-array-form-proposal.md`, fork #105 | per loop: decisions bit-identical to the reference vectors, on the CPU and on the device |
| C3 | The journal: undo instead of restore; check mode | `42d48c31` (interim), fork #119 and #135 (interim) | check mode passes on the test suite and on rte6515; restore time proportional to the writes |
| C4 | Deltas and lanes: recorded deltas; CPU threads as lanes; GPU lanes packed from deltas; analytic deltas tested against recorded ones | 4.7, 4.8 | four-thread slowdown per thread on rte6515 close to lightsim2grid's 1.2; the GPU reject reasons that exist only because of the mirror are gone |

### What each side gets, stage by stage

| After | CPU | GPU |
|---|---|---|
| A | the measured gains of the performance branch on KLU and on the fills, in reviewable PRs | nothing changes; its row rule is now OLF's |
| B | one analysis per run; no rebuild on PV/PQ switches, islanding or control switches | about 6,500 lines of extraction and pattern construction deleted; the structure is OLF's |
| C | contingencies in a time proportional to their delta; threads without network copies | about 4,700 lines of planner mirrors deleted; exact deltas |

## 8. Measurements to run first

They use the harness of the performance branch (`benchmark/sa-vs-lightsim2grid`: `OlfN1Bench`, `ExportJ`,
`klubench.c`), alternating the variants on one machine.

| | Question | How | Decides |
|---|---|---|---|
| M1 | A common baseline | rte6515 N-1 on `main`, on the performance branch and on the alternative-equations branch; plain and default configurations; 1 and 4 threads | the reference numbers for every later step |
| M2 | What does the union pattern cost in KLU? | export three matrices of the same network: paired base pattern; plus the reactive balance on every PV slot; the GPU superset. Run `klubench.c`: LU nonzeros, analyse, factor and refactor times | the default scope; whether the adaptive scope is needed |
| M3 | Which slot order? | type blocks (all φ, all v, controls) against per-bus interleaving, same matrices: KLU times and fill times | the order used in A2 |
| M4 | What does a recorded delta cost? | time `LfContingency.apply` plus the restore per contingency on rte6515 | how much analytic delta the GPU needs |
| M5 | How much is in scope? | per network: slots by alternative set, moved balances, exclusions; contingencies that need a rebuild | the order of the families in B3 |

Type blocks give affine maps from element number to column and serve the fast-decoupled split. Interleaving keeps
the entries of one bus together. Both upstream efforts end with type blocks; the GPU inherits interleaving from
OLF's `Variable.compareTo`, which compares the element number first.

## 9. Risks and open questions

- **Columns of an equation array are no longer a contiguous range.** `EquationArray` owns `[firstColumn,
  firstColumn + length)` today. With slots, single equations occupy columns inside that range. `EquationVector`,
  `JacobianMatrix.initDer` and the fast-decoupled split assume the range. This is the main code change of A2.
- **Equations created at run time.** `AcEquationSystemUpdater.recreateDistributionEquations` creates equations when
  a zero-impedance network splits or merges. Under a fixed structure these must exist up front. This is why zero
  impedance is last in B3; an earlier attempt on the fork was reverted.
- **Three people on three files.** A2 and A4 need an agreed order of merge before more code is written.
- **Results change in the last digits.** Summation order changes with cells and scatters. The performance branch
  updated two frozen expectations. Comparisons of security analyses must use the thresholds of
  `docs/increased-violations-thresholds.md`.
- **Refactorisation without pivoting.** It relies on the `klu_rcond` threshold and on the fallback. Count the
  fallbacks per run; a family that causes many gets the GPU's numerics-driven placement.
- **The residual found on fictitious buses.** The alternative-equations branch saw a 0.1% difference when freezing
  an islanded star bus. Ghost mode reports identical results on 2,162 islanding outages. To reproduce before B1.
- **Upstream acceptance.** Stage A stands alone. Stage B is opt-in and is tested against the legacy path on the
  whole suite. If it is not accepted, the burden moves to the fork instead of disappearing.
- **Out of scope.** Asymmetrical systems, AC/DC converters, secondary voltage control and solvers other than
  Newton-Raphson stay on the legacy path.

## 10. The next three weeks

1. Share this note with S. Prabakaran and G. Jamgotchian. Agree on the shape of A2 (variable side) and A4 (cells,
   primitive activity, direct index), and on who rebases on whom.
2. Run M1 to M3.
3. Open the two A1 PRs.
4. In olf-gpu: export the superset pattern for M2, and write the pattern comparison tool that B4 needs.
5. In the fork: rebase #101 onto A2; raise the interaction between #120 and `minimize_zeros…`.

## Appendix: what happens to the existing PRs and branches

| Item | Fate |
|---|---|
| Upstream #1521, #1564 | merged into A4 |
| Upstream `minimize_zeros_in_diagonal_matrix` | becomes A2 |
| Performance branch | split over A1 to A4, B1, C1; its tracker is interim until C3 |
| Fork #101 (`klu_refactor` first) | A3, rebased on A2 |
| Fork #105 (reactive limits on arrays) | pilot of C2 |
| Fork #108 (monitored branch index) | C1 |
| Fork #119, #135 (partial restores, restore hooks) | interim for olf-gpu; replaced by C3 |
| Fork #120 (`getSortedVariablesToFindOnly`) | kept until B1, where rows become constants of the structure; breaks if rows are derived from equation columns |
| Fork #127 (`resultWithoutNetwork`) | kept: it is the lane result hook |
| Fork #130, #131 (values-only updates) | interim; replaced by C1 and C4 |
| Fork #146 to #149 and the 2.4.1 stack (extension points) | unchanged; several accessors become unnecessary after B4 |
| Fork alternative-equations branches | source of tests, reproducers and the partial update for B1 and B2; `AlternativeEquation` is not carried over |
| olf-gpu #51 | this note replaces its plan section |
| olf-gpu #50 (outer loops in array form) | C2 |
| olf-gpu #52 (duplication cleanup) | the packages that do not touch structure code now; the rest is deleted by B4 |
| olf-gpu `StableRowIndexing`, `ReactiveRelocation*`, `GpuRowMode` | specification of A2, B1, B3; deleted at B4, except the device numerics |
| olf-gpu `batch/plan` | tested against recorded deltas in C4, then removed family by family |
