# On-device limit-violation detection — scope

**Status**: step 1 (the flat table + a host scanner over it) LANDED 2026-09-10; the kernel itself is
next. Written 2026-09-09, re-measured 2026-09-10 on rte6515_full, 2000 branch contingencies, RTE
production parameters, DEVICE arm, 1 thread.

## Step 1 landed: the table, scanned on the host

The table this document called day 1 is built (`GpuLimitTable`) and scanned by `GpuLimitScanner`.
A/B at 2000 contingencies, violation sets **identical (0 differing lines, 133134 violations each)**:

| arm | wall | detectViolations |
|---|---|---|
| flat table (`olf.gpu.sa.tableViolations` on, default) | **36.1 s** | **6.9 s** |
| OLF's own walk (property off) | 38.8 s | 10.8 s |

So flattening alone took 36% off detection without touching the device — the walk through
per-call suppliers and `List<LfLimitsGroup>` was more than a third of it. What remains (6.9 s) is
evaluating the six per-branch quantities and comparing them, which is exactly what the kernel takes,
and ~0.1 s of violation building, which stays on the host either way.

Validated two ways: `GpuLimitTableTest` in the suite, and `OLF_GPU_VIOL_CROSSCHECK=1`, which runs
BOTH detectors on every contingency of a census and compares field for field — 287 contingencies,
19482 violations, zero disagreements.

## Step 3: the kernel

`native/cuda/sa/olf_gpu_violations.cu` — a SEPARATE translation unit that needs no cuDSS and
nothing from the solver's context, only the generated flow formulas, so it compiles in seconds and
cannot perturb the solver's codegen (adding a parameter to a solve kernel has flipped a 1e-12 test
before).

- `uploadViolationTable` uploads the flattened table ONCE per run — a contingency cannot change a
  threshold — and returns a handle;
- `screenViolationsBatch` uploads a chunk of states (the same arrays the host is about to inject)
  and runs one thread per (scenario, branch) and one per (scenario, bus), emitting candidates into a
  per-scenario buffer with `atomicAdd`;
- a scenario whose candidates exceed the buffer is not wrong — its count comes back above the
  capacity and the host screens it itself.

Records come back unordered (threads finish when they finish) and are SORTED on the host, which is
why a bus record is encoded ABOVE every limit index: one ascending sort restores table order for the
"first exceeded of the group wins" rule and leaves the bus block at the end.

It takes nothing from the batch solver — no context, no chunk lifetime, no share of the memory
ceiling that already OOMs at 2781 scenarios.

## Why

`LimitViolationManager.detectViolations` is the largest remaining block of the GPU security
analysis wall time that is not a device solve:

| phase (1 thread, wall ~34 s) | time |
|---|---|
| batched section (device solves + orchestrators) | 13.7 s |
| **detectViolations** | **9.8 s** |
| state injection + network update | 1.7 s |
| base load flow + setup | 3.0 s |
| extraction, OLF apply/restore, 2 CPU fallbacks | ~6 s |

It is ~29% of the wall at 1 thread and ~5 s at the 2–4 threads we recommend. The work is
9042 branches x 2 sides x 3 limit types (plus ~6500 buses) per contingency — about 54,000
comparisons each, 108 million over the run, at ~90 ns apiece. That is comparison arithmetic over
data the device already holds.

**A screen would not help**: on this case 2000 of 2000 contingencies have violations (133,154 in
total, ~67 each), so "flag candidates on the device, let the host confirm" skips nothing. The
detection itself has to move.

**Caveat on the payoff**: the CPU security analysis pays this same cost, so removing it makes the
product faster in absolute terms — it does not close a GPU-specific inefficiency. Expected effect:
1-thread wall ~34 s -> ~24 s, ratio against the CPU provider 2.98x -> ~4.1x.

## What makes it tractable

Two properties, both verified in OLF's source rather than assumed:

1. **The thresholds are static per run.** `AbstractLfBranch` builds each side's
   `List<LfLimitsGroup>` lazily and caches it, with the limit reductions already applied — and
   OLF's own comment in `createLimits` states the reason it may do so: *"the only supported
   ContingencyContext for LimitReduction is ALL"*. So a threshold table can be built once and
   uploaded; it does not vary per contingency.

2. **Only the VALUE varies.** `createLimitViolation` fills subject
   (`branch.getMainOriginalId()`), operational limits group id, type, limit name, acceptable
   duration, `limit * scale`, reduction and side entirely from the branch and the `LfLimit` — every
   field except `value` is a property of the table entry. The bus path is the same: subject is
   `bus.getVoltageLevelId()`, the limit and scale come from the bus.

Together these mean **the device never constructs a violation**. It emits
`(scenario, tableEntryIndex, value)` and the host rebuilds the exact `LimitViolation` from the
entry it already holds. No violation metadata is re-implemented, so none of it can drift.

## Design

**Static table** (host, built once after extraction):
- Branch entries, ordered so one contiguous slice belongs to each
  `(branch, side, limitType, operationalLimitsGroup)`, and within a slice the temporary limits stay
  in `getSortedLimits()` order. The kernel walks a slice and takes the FIRST exceeded, then stops —
  OLF's `break`, which its comment explains as *"Only detect the most serious one ... limit
  violations are ordered by severity"*.
- Per entry: the comparison threshold (`LfLimit.getReducedValue()`, per-unit) and the quantity
  selector (i / |p| / s, side 1 or 2).
- Bus entries: high and low voltage limits per bus, `NaN` meaning absent.
- A parallel host-side array holds the `LfBranch` / `LfLimit` / group id / side / type for
  rebuilding, indexed identically.

**Device kernel** (one thread per (scenario, branch), the shape the census suggests: 9042 branches
x ~500 scenarios per chunk saturates on its own):
- Evaluate i1, i2, p1, p2, s1, s2 at the converged state. These are the closed-branch formulas the
  fill kernels already carry, so they come from the existing generated family — no new physics.
- Skip a branch disabled for this scenario (the per-scenario disabled mask is already on device)
  and skip a side whose bus is absent (`getBus1()`/`getBus2()` null in the host check).
- Compare against the branch's table slices, emit compact records.
- Buses: compare `v` against the bus entry.

**Host**: rebuild each record into a `LimitViolation` from its table entry and pass it through the
existing `addLimitViolation`, which keeps the reference/increased-violations filtering against the
pre-contingency manager unchanged. ~67 records per contingency instead of 54,000 comparisons.

### Step 2 landed: the screen/confirm split, proven in Java first

`GpuViolationScreen` computes each branch's flows from the SAME packed element data the solver
uploads (`cbIn`/`cbIdx`, `obIn`/`obIdx`, a controlled tap read out of the state vector) and emits
candidates against a relaxed threshold; `GpuLimitScanner.detectFromCandidates` re-tests each with
OLF's exact `>` on OLF's own value.

Doing this in Java BEFORE writing any CUDA answers the question the kernel actually turns on — does
a flow computed independently of OLF's evaluables select the same violations? On rte6515/300 with
`OLF_GPU_VIOL_CROSSCHECK=1`: **287 contingencies, 19482 violations, zero field-level disagreements,
and 0 branches uncovered** (every branch with limits maps to the closed or open pack). The kernel now
has an oracle, and the whole host half — descriptor, candidate encoding, confirm, fallback — is
already exercised on real data.

Table size on rte6515, which is the work per contingency the kernel takes over:
9042 branches (8825 with limits), **34714 groups (I 17509, S 17205, P 0), 138817 limit entries**,
6520 buses.

### Revised 2026-09-10: the device SCREENS, the host CONFIRMS

The original design had the device emit the value it computed. That value would differ from the
host's in the last bits (FMA contraction, a different `sin`/`cos`), and violations of an element that
already violates at BASE are decided by a comparison with **zero tolerance** — so a few ulps there
move real reported violations (see the knife-edge below). Two changes remove the exposure entirely:

- the kernel compares against a **relaxed threshold** (`value > threshold * (1 - band)`), so it can
  only ever emit MORE candidates than OLF would, never fewer;
- it does **not** stop at the first exceeded limit of a group — it emits all of them, so the host is
  free to reject the first;
- the host, holding the exact values already (the network vector is updated for the result anyway),
  re-tests each candidate with OLF's exact `>` and takes the first that confirms.

The record then needs no value at all — an int per candidate. Correctness stops depending on device
precision and depends only on the band being wider than the device-vs-host discrepancy, which is
measurable (and a debug mode should report the observed maximum against the band).

**Safety valve**: a bounded per-scenario record capacity with an overflow flag; a scenario that
overflows falls back to the host `detectViolations`. Also the natural kill switch for the whole
feature.

**Stays on the host**: voltage-angle limits (`network.getVoltageAngleLimits()` — a handful, needing
only bus angles) and the reference filtering.

## Effort

| step | estimate |
|---|---|
| table builder + a test asserting it reproduces `LimitViolationManager`'s decisions element-by-element on the base case | 1 day |
| kernel + JNI + record buffers + host rebuild + fallback | 1.5 days |
| parity validation (suite, then rte6515 violation-SET diff against the CPU arm) | 1 day |

~3.5 days. The kernel is the small part; the table and the validation are the work.

## Risks and open questions

- **`computeApparentPower1/2`** is an `LfBranch` method, not an equation term — confirm it is
  `hypot(p, q)` in the vectorized implementation before reproducing it, and reproduce OLF's own
  formula rather than a re-derivation. Its `NaN` case (fictitious branches) must be skipped exactly
  as the host does.
- **Three-winding transformers** appear as `LfLegBranch` in `network.getBranches()`;
  `branch.getOriginalSide()` decides the reported side. Metadata only, but the table must carry it.
- **Zero-impedance branches** and open sides: confirm which ones the host actually reaches.
- **Per-unit scales**: current uses `PerUnit.ib(bus.getNominalV())` of the SIDE's bus, power uses
  `PerUnit.SB`, bus voltage uses the bus nominal V. The table must store the scale per entry.
- **Validation must diff violation SETS, not counts** — and must expect a knife-edge that is NOT
  about limits. Violation counts on rte6515 vary between runs (133154 in eleven runs, 133870 and
  135152 in two others). Diffing the sets names the cause exactly: every one of the 716 extra lines
  in the 133870 run is the same element, `VL-1512 LOW_VOLTAGE`, and the base violation set is
  byte-identical.

  `VL-1512` already violates at BASE (60.974857123 kV against a 61.000002 kV limit), so every
  post-contingency violation of it goes through `LimitViolationManager.violationWeakenedOrEquivalent`,
  which for LOW_VOLTAGE reports it only when
  `postValue < baseValue - min(lowVoltageAbsoluteThreshold, baseValue * lowVoltageProportionalThreshold)`.
  Both of those thresholds default to **0.0** (flow violations get 10%, voltage gets nothing), so
  the test is `postValue < baseValue` — strictly, by any amount, down to one ulp. In the hundreds of
  contingencies whose outage is far from that bus the post-contingency voltage equals the base value
  to the last bits, and the GPU base load flow's ~1e-9 run-to-run wobble (atomic ordering in the fill
  kernels) decides them all at once.

  So the sensitivity is NOT to the limit — the margin there is 0.025 kV, enormous by comparison. It
  is to the BASE VALUE, through a zero-tolerance comparison. The CPU never shows it because it is
  deterministic, not because it is less sensitive; and a CPU-vs-GPU comparison of this case will
  disagree on such an element for many contingencies without either side being wrong. Any element
  that violates at base is a candidate. Either give the comparison a small
  `increased-low-voltage-violations-absolute-threshold`, or exclude base-violating elements from the
  parity gate.

## What NOT to do

- Do not build a device pre-screen that defers to the host: measured, it skips nothing here.
- Do not re-implement the violation metadata on the device; emit table indices.
