# On-device limit-violation detection — scope

**Status**: scoped, not started. Written 2026-09-09 from measurements on rte6515_full, 2000
branch contingencies, RTE production parameters.

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
- **Validation must diff violation SETS, not counts.** The GPU base load flow is nondeterministic
  at ~1e-9 (atomic ordering in the fill kernels), and because the pre-contingency violations filter
  the post-contingency ones, a single element sitting that close to its limit shifts every
  contingency at once — observed once as 135152 violations where other runs reported 133154, a
  difference of exactly 1998, one per batched scenario. A count comparison will therefore show
  alarming swings that are one knife-edge element.

## What NOT to do

- Do not build a device pre-screen that defers to the host: measured, it skips nothing here.
- Do not re-implement the violation metadata on the device; emit table indices.
