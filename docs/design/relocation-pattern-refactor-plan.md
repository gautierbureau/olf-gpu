# Plan: shrink `relocation_pattern.cu` without changing behaviour

Status: proposal. Nothing here is implemented, and the line counts for the result are estimates.

## The problem

`native/cuda/fixedmatrix/relocation_pattern.cu` is the largest file of the fixed-matrix code (539 lines).
It builds two things for every element output that can be written on more than one row
(docs/gpu-vs-olf.md, section 2.3):

- the extra entries of the union pattern (`collect<Family>RelocationPattern`);
- the per-output alternate destinations and selectors (`build<Family>RelocationDestination`).

Most of the file is the same code written several times.

### Duplication 1: once per element family

There is one `collect` and one `build` function per family:

| Family | `collect` | `build` | Index stride | Alternate row comes from |
|---|---|---|---|---|
| closed branch | lines 81–118 | lines 124–205 | 12 | `ix[10 + rowEnd]` |
| shunt | 216–253 | 255–336 | 5 | `ix[4]` |
| open branch | 394–431 | 433–514 | 6 | separate array, `alternateQRows[e * 2 + rowEnd]` |
| HVDC AC emulation | 343–356 | 358–384 | 4 | none (stock placement only) |

Measured on `main`: after replacing the family names, the closed-branch and shunt `build` functions
(82 lines each) are identical apart from whitespace. The open-branch one differs by 14 lines, all from the
alternate row living in a separate array. The HVDC pair is the stock-placement subset of the same logic.

### Duplication 2: `collect` and `build` walk the same decision tree

For each output, both functions decide which case applies, in the same order:

1. stock pair (a coupler-tree child whose balance sits on the coupler's dummy row);
2. group relocation (`alternateRow <= -2`: every candidate row of a carrier group);
3. dependent pilot (the candidate pool of another group, plus the home row);
4. plain alternate row.

`collect` then adds `(row, col)` pairs to the pattern; `build` looks the same pairs up with `pos.at(row, col)`.
Nothing ties the two together except that they were written to match. A case added to one and not the other
is a lookup failure at structure build, or a destination that is never used.

### Why it grew this way

Each relocation case was added on top of "an output has one frozen row", one family at a time. The cases are
correct; the structure is not shared.

## Goal and non-goals

Goal: the same pattern and the same destination arrays, bit for bit, from about 200 lines instead of 539.

Non-goals:

- No change to the kernels, to `relocated_scatter.cuh`, to `stock_placement.cuh` or to the Java packers.
- No change to which rows an output may land on. Remodelling rows as slots with alternatives is the subject of
  `docs/design/fixed-structure-equation-system.tex` (PR #51); under that design most of this file disappears.
  This plan is the cheap step that is worth doing regardless.

## Steps

Each step is one commit and leaves the test suite green.

### Step 0 — Pin the current output

The refactor must be provably behaviour-preserving, and the existing tests only check solutions, not arrays.

1. Add a structure fingerprint: a hash over `rowPtr`, `colIdx` and, per family, the alternate destination,
   selector row, relocation group, candidate offset, candidate destinations and dependent-pilot arrays.
   Expose it through a new trace channel in `native/cuda/common/trace.cuh` (with its row in
   `docs/env-vars.md`), printing one line per structure built. A channel only prints.
2. Record the fingerprints on `main` for the networks that exercise each case:
   - transformer and shunt voltage control (plain alternate row): `GpuTransformerVoltageControlTest`,
     `GpuSaShuntControlTest`;
   - remote voltage control groups and dependent pilots: `GpuSaRemoteVoltageControlTest`,
     `GpuPilotSelfGroupTest`;
   - coupler-tree children and the stock pair: `GpuZiCouplerTreeRelocationTest`, `GpuSaBusbarAndIslandTest`;
   - HVDC on a coupler-tree child: the busbar scope of `network.xiidm`;
   - the rte6515 gate scopes, which cover all of them at scale.
3. Keep the recorded values in the PR description; they are the acceptance criterion for every later step.

### Step 1 — One description per family

Introduce a small traits type per family, used only inside the `.cu` file:

```cpp
struct ClosedBranchRelocation {
    using Scatter = ClosedBranchScatter;
    static constexpr int STRIDE = 12;
    static constexpr int OUTPUT_COUNT = CLOSED_BRANCH_OUTPUT_COUNT;
    static const Scatter& scatter(int k) { return CLOSED_BRANCH_SCATTER[k]; }
    static RC resolve(const Scatter& s, const int* ix) { return resolveClosedBranch(s, ix); }
    static int alternateRow(const Scatter& s, const int* ix, const int* /*separate*/, int /*e*/) {
        return s.rowEq == 1 ? ix[10 + s.rowEnd] : -1;
    }
};
```

The shunt, open-branch and HVDC traits differ only in these members. The open-branch one reads the separate
array; the HVDC one returns -1.

### Step 2 — One template for `build`, one for `collect`

Replace the four `build` functions by `template <typename F> void buildRelocationDestination(...)` and the four
`collect` functions likewise. Keep the eight public function names in `relocation_pattern.cuh` as one-line
wrappers, so `shared_structure.cu` does not change in this step.

HVDC needs care: its current functions skip the group and dependent-pilot cases and return early when
`relocatedChildOwnRow` is null. Check with the fingerprint that the general template gives the same arrays
when the alternate row is always -1; if it does not, keep HVDC on its own short function and say so.

Expected removal: about 250–300 lines.

### Step 3 — One enumeration of placements

Extract the decision tree into a single function that, for one output, reports its case and the rows it may
land on:

```cpp
enum class RelocationCase { NONE, STOCK_PAIR, GROUP, DEPENDENT_PILOT, ALTERNATE };

struct Placements {
    RelocationCase kind;
    // rows, in the order build() must emit them
    ...
};

template <typename F>
Placements placementsOf(const typename F::Scatter& s, const int* ix, ...);
```

`collect` becomes "for each row of `placementsOf`, add `(row, col)`"; `build` becomes "for each row, emit
`pos.at(row, col)` or `-2 - row`". The pattern then contains every destination by construction.

Constraint: `relocationCandidateDestinations` is appended to in traversal order and the offsets stored per
output index into it, so the enumeration must return rows in exactly the order `build` pushes them today.
The fingerprint catches any reordering.

Expected removal: about 60–80 more lines, and the two functions can no longer disagree.

### Step 4 — Tidy the call sites

1. Remove the wrappers and call the templates from `shared_structure.cu` (lines 107–128 and 227–332 today).
2. Replace the long, repeated parameter lists (group start, candidate rows, dependent group of row) by one
   `ReactiveRelocationTablesView` struct passed by const reference.
3. Fold `hasRelocatable` onto the traits (it hard-codes strides 12 and 5 and offsets 10, 11 and 4).

### Step 5 — Look for the same shape elsewhere

Out of scope for the PR, to be listed in `docs/tasks.md` afterwards:

- `structure/csr_pattern.cu` adds relocated-child "own row" duplicates per family (around lines 247–258);
- `fixedmatrix/zero_impedance_relocation.cu` and `structure/distribution_dests.cu` handle the same
  "own row or dummy row" choice for couplers and `DISTR_Q`;
- the Java side (`CouplerTreeRelocation`, `ReactiveRelocation`, `ReactiveRelocationTables`) has not been
  checked for duplication.

## Verification

- Fingerprints equal to the ones recorded in step 0, on every network listed there, after every step.
- `tools/run_gpu_tests.sh` green.
- The rte6515 gate scopes unchanged: same number of contingencies on the device, same results.
- `tools/run_bench.sh` cold and warm within noise: the change is host-side structure build only, so the cold
  number is the one that could move.

All of this needs the GPU machine. The refactor cannot be compiled or run in a CUDA-free environment, so it
should be written and verified there, or written elsewhere and not merged until the fingerprints match.

## Risks

- **Silent reordering of candidate destinations.** The per-output offsets depend on push order. Mitigated by
  the fingerprint; this is the reason step 0 comes first.
- **HVDC is not a pure subset.** If the general template changes its arrays, it stays separate (step 2).
- **Templates in a `.cu` file.** They are host code, instantiated four times in one translation unit; no
  device code is involved and the header keeps plain declarations.
- **Effort spent on code the slot design removes.** Steps 0 to 3 are small (estimated two to three days with
  the GPU machine at hand) and step 0's fingerprint is reusable as the equality gate when the GPU structure is
  later built from an OLF-side description.

## Expected result

| | Today | After (estimate) |
|---|---|---|
| `relocation_pattern.cu` | 539 lines | about 200 |
| Decision tree copies | 8 (4 families × collect/build) | 1 |
| Pattern ⊇ destinations | by convention | by construction |
