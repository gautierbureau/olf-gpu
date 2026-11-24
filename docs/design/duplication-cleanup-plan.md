# Plan: remove the duplicated code, without changing behaviour

Status: proposal. Nothing here is implemented. The measurements are from `main`; the sizes after the work are
estimates.

## What was measured

`tools/clone_scan.py` drops comments, blank lines and includes, collapses whitespace, and counts a line as
repeated when it lies in a window of six consecutive code lines that occurs more than once.

| Code | Code lines | In a repeated block |
|---|---|---|
| Native, `native/cuda` (generated files excluded) | 15 951 | 1 341 (8.4%) |
| Java, `src/main/java` | 14 870 | 365 (2.5%) |

With identifiers and numbers replaced too (`--shape --window 8`: same shape, different names) the native figure
is 20.9%. That is an upper bound; it also counts boilerplate that merely looks alike.

Native, by directory (exact repeats):

| Directory | Repeated / code lines | Share |
|---|---|---|
| `fixedmatrix` | 195 / 915 | 21.3% |
| `equations` | 215 / 1 039 | 20.7% |
| `lu` | 212 / 1 879 | 11.3% |
| `diagnostics` | 52 / 557 | 9.3% |
| `jni` | 219 / 2 672 | 8.2% (31% by shape) |
| `structure` | 64 / 869 | 7.4% |
| `outerloop` | 271 / 4 056 | 6.7% |
| `solver` | 113 / 3 278 | 3.4% |

The scan finds textual repeats only. Code that does the same thing written differently is not counted, and
only the hot spots below were read.

## What this plan is and is not

Removing every exact repeat would take out roughly 1 000 to 1 500 native lines, 5 to 8% of the native code.
Duplication is a real but minor part of the size. The larger part is code that mirrors OLF (structure building,
contingency planning, outer loops), and that only shrinks through the fixed-structure work
(`docs/design/fixed-structure-equation-system.tex`, PR #51).

The reason to do this plan anyway is correctness: a fix applied to one copy and not the other is a bug, and
two of the packages below (1 and 4) remove a place where two pieces of code must agree by convention.

Every package is behaviour-preserving. None changes which rows an output lands on, a kernel's arithmetic, or a
result.

## Common ground rules

- One package per pull request, each a short series of commits that leaves the suite green.
- **Package 0 first.** The existing tests check solutions, not arrays; a refactor of structure-building code
  needs a stronger check.
- Each package states its own acceptance test. All of them need the GPU machine: none of this can be compiled
  or run in a CUDA-free environment.
- After each package, re-run `tools/clone_scan.py` and record the new figure in the pull request.

## Package 0 — Pin the current output

1. Add a structure fingerprint: a hash over `rowPtr`, `colIdx` and, per element family, the frozen
   destination, alternate destination, selector row, relocation group, candidate offset, candidate
   destinations and dependent-pilot arrays. Expose it through a new trace channel in
   `native/cuda/common/trace.cuh` (with its row in `docs/env-vars.md`), one line per structure built. A channel
   only prints.
2. Record the fingerprints on `main` for the networks that exercise each relocation case:
   - plain alternate row (transformer and shunt voltage control): `GpuTransformerVoltageControlTest`,
     `GpuSaShuntControlTest`;
   - carrier groups and dependent pilots: `GpuSaRemoteVoltageControlTest`, `GpuPilotSelfGroupTest`;
   - coupler-tree children and the stock pair: `GpuZiCouplerTreeRelocationTest`, `GpuSaBusbarAndIslandTest`;
   - HVDC on a coupler-tree child: the busbar scope of `network.xiidm`;
   - the rte6515 gate scopes, which cover all of them at scale.
3. Record the cold and warm numbers of `tools/run_bench.sh` as the performance reference.

The fingerprint is reusable later as the equality gate when the GPU structure is built from an OLF-side
description.

## Package 1 — `fixedmatrix/relocation_pattern.cu` (largest single hot spot)

**Finding.** 183 of the file's 332 code lines repeat. It builds, for every element output that can be written
on more than one row (docs/gpu-vs-olf.md, section 2.3), the extra pattern entries
(`collect<Family>RelocationPattern`) and the per-output destinations (`build<Family>RelocationDestination`).

- *Once per family.* One `collect` and one `build` each for closed branch, shunt, open branch and HVDC. After
  replacing the family names, the closed-branch and shunt `build` functions (82 lines each) are identical
  apart from whitespace; the open-branch one differs by 14 lines, all from its alternate row living in a
  separate array. The HVDC pair is the stock-placement subset.
- *Twice per family.* `collect` and `build` walk the same decision tree in the same order (stock pair, group
  relocation, dependent pilot, plain alternate row). `collect` adds `(row, col)` pairs; `build` looks the same
  pairs up. Nothing ties them together except that they were written to match.

| Family | `collect` | `build` | Index stride | Alternate row comes from |
|---|---|---|---|---|
| closed branch | lines 81–118 | 124–205 | 12 | `ix[10 + rowEnd]` |
| shunt | 216–253 | 255–336 | 5 | `ix[4]` |
| open branch | 394–431 | 433–514 | 6 | separate array, `alternateQRows[e * 2 + rowEnd]` |
| HVDC AC emulation | 343–356 | 358–384 | 4 | none (stock placement only) |

**Steps.**

1. One traits type per family, local to the `.cu` file: scatter table, stride, output count, resolve
   function, alternate-row accessor.
2. One template for `build` and one for `collect`. Keep the eight public names in `relocation_pattern.cuh` as
   one-line wrappers so `shared_structure.cu` does not change yet.
3. One function that, for one output, returns its case and the rows it may land on. `collect` becomes "add
   `(row, col)` for each row"; `build` becomes "emit `pos.at(row, col)` or `-2 - row` for each row". The
   pattern then contains every destination by construction.
4. Call the templates directly from `shared_structure.cu` (lines 107–128 and 227–332 today), replace the
   repeated parameter lists (group start, candidate rows, dependent group of row) by one view struct, and fold
   `hasRelocatable` onto the traits (it hard-codes strides 12 and 5 and offsets 10, 11 and 4).

**Risks.** `relocationCandidateDestinations` is appended to in traversal order and per-output offsets index
into it, so the enumeration must return rows in exactly today's push order. HVDC skips the group and
dependent-pilot cases and returns early on a null `relocatedChildOwnRow`; if the general template changes its
arrays, it stays on its own short function.

**Acceptance.** Fingerprints equal to package 0 on every listed network. Expected: 539 → about 200 lines.

## Package 2 — The two LU variants (`lu/scalar_lu.cu`, `lu/supernodal_lu.cu`)

**Finding.** The two files share their preamble and layout conversion verbatim:

- `toInterleaved` and `fromInterleaved`: identical kernels in both files (scalar lines 37–84, supernodal
  38–84);
- `zeroFill` and the `uploadVector` template: identical;
- the head of `factorize…Lanes` (scalar 387–397, supernodal 593–603) and of the split solve launch;
- inside each file, the forward and backward level kernels share their opening (scalar 192–201 against
  212–221; supernodal 319–328 against 430–439).

The factorisation kernels themselves, the `create…Lu` functions and the bodies of `factorize…Lanes` genuinely
differ; they are not candidates.

**Steps.**

1. Move `toInterleaved`, `fromInterleaved`, `zeroFill` and `uploadVector` to a shared `lu/lu_lanes.cuh` /
   `.cu`, used by both variants.
2. Extract the common head of `factorize…Lanes` (lane-map validation and layout selection) into one helper.
3. Leave forward and backward level kernels as they are unless the shared opening extracts into an inline
   helper with no effect on the generated code; these are hot kernels.

**Risks.** Kernels moved to another translation unit must keep the same launch configuration. Step 3 touches
the hottest code of the batched LU, hence the "only if free" rule.

**Acceptance.** `GpuBatchedLuTest` green; `OLF_GPU_LU_COMPARE` reports no new difference against cuDSS on the
SA suite; warm and cold bench within noise. Expected: about 120 to 150 lines removed.

## Package 3 — Fill kernels (`equations/`)

**Finding.** Most fills exist twice: a full version (values and Jacobian) and a mismatch-only one, kept
separate on purpose because the mismatch-only path is cheaper.

- `bus_injection_fill.cu`: `fillShuntBatch` against `fillShuntBatchF` (16 repeated lines) and `fillHvdcBatch`
  against `fillHvdcBatchF` (11); 61 of the file's 122 code lines repeat.
- `branch_fill.cu`: `fillOpenBranchBatch` against `fillOpenBranchBatchF` (13 lines). For the closed branch
  the full kernel reads its inputs through `closedBranchOutputs` while the mismatch-only one unpacks the same
  parameters, tap override and state inline.
- The mismatch-only kernels address their outputs by literal positions (`{0, 7, 14, 21}` for the closed
  branch), guarded by `static_assert`s on the scatter roles.
- `distribution_fill.cu` repeats itself (20 shared windows) across the `DISTR_Q`, `DISTR_RHO` and
  `DISTR_SHUNT_B` kernels.

**Steps.**

1. One "element inputs" helper per family (parameters with the per-scenario override applied, plus the state
   values the element reads), used by both the full and the mismatch-only kernel. The two kernels stay
   separate; only their shared head and tail go.
2. Derive the mismatch positions from the generated scatter table with a `constexpr` function instead of
   literals, keeping the `static_assert`s as a cross-check until the literals are gone.
3. One shared scatter tail for the mismatch-only kernels (plain, relocated, and the reactive buffer copy).
4. In `distribution_fill.cu`, extract the common per-term loop of the three linear-distribution kernels.

**Risks.** These are the hottest kernels after the LU. Helpers must be `__device__` inline and the generated
code checked for regressions.

**Acceptance.** `GpuFullNewtonTest` and `GpuLoadFlowTest` green (the Newton reference catches a missed or
doubled term); SA suite and rte6515 gate scopes unchanged; bench within noise, warm number first since the
fills run every iteration. Expected: about 100 to 150 lines removed.

Note: a fully generated fill kernel (one template rendered per family by `java2cuda`) would replace this
package. It depends on OLF owning the scatter maps, so it is not planned here.

## Package 4 — Reactive limits and its audit (`outerloop/reactive_limits.cu`, `diagnostics/driver_traces.cu`)

**Finding.** The audit in `driver_traces.cu` deliberately runs the real reactive-limit check on the final
state rather than a second implementation. But it does so by repeating the launch sequence, each launch with
its twenty-odd arguments: `reactive_limits.cu` 516–528, 534–555 and 557–563 reappear at `driver_traces.cu`
359–371, 389–410 and 412–418. A new kernel argument has to be added in both places.

**Steps.**

1. Extract `launchReactiveLimitsCheck(ctx, ...)`: the capability-curve launch, the check launch and the
   follow-up, taking the few values that differ between the loop and the audit.
2. Call it from both.

**Acceptance.** SA suite green; `OLF_GPU_RL_AUDIT` output unchanged on the rte6515 scopes. Expected: about 40
lines removed, and one place to change when the kernel signature changes.

A wider point, not planned here: kernel launches across `outerloop/` and `solver/` pass twenty to forty
context fields each. View structs built once per context would shorten every launch site; it is a larger,
mechanical change that deserves its own decision.

## Package 5 — JNI glue (`jni/`)

**Finding.** Little exact repetition (8%) but the highest by shape (31%): every entry point unpacks its
arrays, validates, calls, and copies back in the same way. The clearest case is `batched_lu_jni.cu`, whose
test entry points repeat the same set-up (unpack the plan arrays, validate the lane map, build the plan,
create the LU, allocate and upload three device buffers): lines 129–154 reappear at 224, 291–321 and 364–376.
`reactive_limits_jni.cu` and `structure_jni.cu` hold over thirty array accesses each.

**Steps.**

1. In `batched_lu_jni.cu`, a fixture struct that owns the unpacked plan, the LU and the device buffers, built
   by one function and used by every entry point.
2. Extend `jni_common.cuh` with the array helpers the other files re-spell (pinned or copied int and double
   arrays with their length, optional arrays, scoped release).
3. Stop there. Generating this layer from a schema of the shared description would remove it rather than
   tidy it, and belongs with the fixed-structure work.

**Acceptance.** Whole suite green: JNI errors show as crashes or wrong arrays immediately. Expected: about 150
to 200 lines removed.

## Package 6 — Java (`src/main/java`), small

**Finding.** 365 repeated lines in all. The clusters:

- `GpuTransformerVoltageControlOuterLoop` and `GpuShuntVoltageControlOuterLoop` share blocks (605–620 against
  146–161, 691–696 against 180–185), and so do their incremental variants (77–96 against 69–86);
- `CouplerSeed` repeats itself (155–166 against 272–283, 107–128 against 259–267);
- `ReactiveCapabilityCurves` repeats itself (121–134 against 142–152).

**Steps.** Extract the shared blocks into a common base or helper for the two voltage-control loops and their
incremental variants; one private method each in `CouplerSeed` and `ReactiveCapabilityCurves`.

**Acceptance.** `mvn validate` clean; suite green. Expected: about 80 lines removed.

## Order and effort

| Order | Package | Depends on | Estimate | Why here |
|---|---|---|---|---|
| 1 | 0 Fingerprint and bench reference | — | 1 day | Gate for 1 and 3 |
| 2 | 1 Relocation pattern | 0 | 2–3 days | Largest hot spot; removes a "must agree" pair |
| 3 | 4 Reactive limits launch | — | half a day | Small; removes a "must agree" pair |
| 4 | 2 LU shared pieces | — | 1 day | Mechanical |
| 5 | 3 Fill kernels | 0 | 2 days | Hot code, needs the bench reference |
| 6 | 5 JNI helpers | — | 1–2 days | Mechanical; superseded later by generation |
| 7 | 6 Java | — | half a day | Smallest |

The estimates assume the GPU machine is at hand. Packages 1 and 4 are the ones to do even if the rest is
dropped.

## Expected result

| | Today | After (estimate) |
|---|---|---|
| Native lines in a repeated block | 1 341 (8.4%) | about 400 (under 3%) |
| Java lines in a repeated block | 365 (2.5%) | about 280 |
| `relocation_pattern.cu` | 539 lines | about 200 |
| Places where two copies must be kept in step by hand | relocation collect/build ×4, reactive-limit launch ×2, full/mismatch fill heads ×4 | none |

## Not covered

- `structure/csr_pattern.cu` (relocated-child duplicates per family, around lines 247–258),
  `fixedmatrix/zero_impedance_relocation.cu` and `structure/distribution_dests.cu` handle the same "own row or
  dummy row" choice in three more places. They share an idea with package 1 rather than text, so the scan
  does not flag them; they are candidates once package 1's placement enumeration exists.
- The Java relocation classes (`CouplerTreeRelocation`, `ReactiveRelocation`, `ReactiveRelocationTables`)
  show no textual repetition and were not reviewed for logical duplication.
- Code that mirrors OLF. It is the bulk of the repository and is addressed by the fixed-structure design, not
  by this plan.
