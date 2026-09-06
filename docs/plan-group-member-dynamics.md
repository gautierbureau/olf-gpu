# Plan: on-device group member dynamics (DISTR_RHO shared-TVC groups & DISTR_Q remote-group pins)

Status 2026-09-06 — both features are behind **hard whole-run gates** after the audits of
`e82290f` (shared TVC) and `9be4a41` (remote-group pin dynamics). This document is the
implementation plan to lift those gates. Read it together with
`docs/gpu-vs-cpu-capabilities.md` (the DISTR_RHO contingency row and the islanding row
carry the audit summaries).

Ground rules that apply to every step below:

- **1e-12 or a documented ulp story** (1e-11 max, user-approved machine precision).
- **Never key a pack/guard predicate on control-ENABLED state** — key on equations
  EXISTING (`es.getEquation(...).isPresent()`), never on `Equation.getColumn() >= 0`
  (inactive equations have column −1) and never on `is*VoltageControlled()`.
  This bug class produced both silent-wrong findings (3.7e-3 kV / guard bypass).
- **Context-scoped native setters** for new per-scenario channels
  (`setBatchRowCriteria` / `setBatchRlCurves` / `setBatchDqWeightOverrides` are the
  models) — no `solveBatchContext` signature growth through the orchestrators.
- Validate every step with an A/B (feature off = measured drift, feature on = 1e-12)
  and with the CPU report (`OLF_GPU_DBG_REPORT=1`) proving the CPU actually exercises
  the semantics (the "RL-pinned remote controller" test that only ever pinned in the
  BASE is the cautionary tale).

---

## Part A — Shared transformer control groups (DISTR_RHO), then member outage

Gate to lift: `GpuBatchTransformerControl.transformerControlSupported` rejects any
group with `getMergedControllerElements().size() > 1` (whole-run CPU).

### CPU semantics (verified 2026-09-06)

- `AcEquationSystemCreator.createR1DistributionEquations`: every member i gets a
  DISTR_RHO equation `0 = (1/c − 1)·r1_i + Σ_{j≠i} (1/c)·r1_j`; the coefficients are
  **live DoubleSuppliers** over `controllerBranches.stream().filter(!isDisabled).count()`
  — renormalization on a member outage is automatic and purely **analytic**.
- `updateRemoteVoltageControlEquations(vc, es, DISTR_RHO, BRANCH_TARGET_RHO1)` sets the
  activation: enabled members — the FIRST gets DISTR_RHO **inactive** (its rho is the
  free degree carrying the controlled bus's V equation), the others DISTR_RHO active,
  all rho identities inactive; a DISABLED member gets DISTR_RHO inactive and its
  **BRANCH_TARGET_RHO1 identity re-activated** (rho pinned at current value); cross
  terms referencing a disabled member's rho deactivate with the element.
- `vEq.setActive(!enabledControllerElements.isEmpty())`.

### Device gaps (why the gate exists)

1. `GpuAcOuterLoopPacks.buildDistrRhoContributions` packs only rows whose DISTR_RHO is
   ACTIVE at extraction (`GpuAcRowIndexing.row()` is column-based ⇒ activation-based).
   At an AFTER_GENERATOR base the control is off ⇒ `ndr == 0` ⇒ the coupling physics is
   absent from the CSR pattern entirely.
2. `GpuBatchTransformerControl.buildTransformerControllers` toggles each control
   **alone** for its ON-diff; for a group, per-member alone-deltas cannot compose into
   the all-enabled activation (free role + DISTR_RHO row set depend on the enabled SET).
3. The same per-branch shape is used by `GpuBatchSimpleTvc` (WITH_GENERATOR) and the
   AFTER_GENERATOR machine in `GpuBatchTransformerControl` (clamp/round stages index
   `tcs.get(c).rhoRow()` per branch).

### Implementation steps (in order)

**A1. Pack DISTR_RHO on EXISTENCE (structure superset).**
`buildDistrRhoContributions`: iterate groups via
`bus.getTransformerVoltageControl().filter(MAIN)` (activation-independent, like the
`e82290f` guard fix) and pack a contribution for every member pair whose DISTR_RHO
equation **exists** (`es.getEquation(num, DISTR_RHO).isPresent()`), with the
all-enabled coefficient (`1/c − 1` self, `1/c` cross). The union CSR pattern then hosts
the rows whether or not the base had them active; an identity row-mode wipes them when
off (same principle as the PV bus's Q scatter). Risk: none for existing runs — rows
inactive at base stay identity until an orchestrator flips them. Bench the nnz growth
(`docs/numbers.md` row; expected negligible — groups are rare and small).

**A2. Free-member row handling in the activation extractor.**
When VC is ON, the free member's rho column carries the controlled bus's V equation via
the qRow remap (`GpuAcRowIndexing.computeQRow` already selects "first member with
DISTR_RHO inactive" — activation-based, correct per state). Verify the extractor
produces, for the toggled-ALL-ON state: controlled bus row = mode 0 identity (V target),
free rho row = mode 1 (the relocated pilot Q), other member rho rows = mode 1
(DISTR_RHO, target 0). Add `GpuAcActivationExtractor` acceptance if any of these throws
today (run `GpuTransformerSharedControlTest`'s network through
`extractActivation` with VC on — a unit test, no device needed).

**A3. Group-level TransformerController.**
Restructure `GpuBatchTransformerControl.TransformerController` into
`record TransformerController(Member[] members, int vCol, double targetV, double halfDeadband,
int[] deltaRows, int[] deltaModesOn, double[] deltaTargetsOn)` with
`record Member(int rhoRow, double minR1, double maxR1, double[] taps, int elemIndex)`.
`buildTransformerControllers`: one controller per GROUP; ON-diff by toggling
`setVoltageControlEnabled(true)` on **all members together** (restore after). A
single-member group degenerates to today's shape — keep behavior identical for it
(regression guard: all existing TVC tests at 1e-12).
Consumers to convert (mechanical, member loops):
  - `GpuBatchTransformerControl` AFTER_GENERATOR machine (deadband check on the
    controlled bus V — group-level; clamp + round PER MEMBER; the "fixedR1" seeding).
  - `GpuBatchSimpleTvc.applyVcOn` / `roundAndResolve` (round each member's rho).
  - `GpuBatchChunkSolver` call sites (types only).
The `disabledMask[slot][elemIndex]` skip becomes per-MEMBER: skip an outaged member's
rounding, not the whole group (see A5 for the activation part).

**A4. Shared-group SA validation (no outage yet).**
Un-gate `transformerControlSupported` for shared groups. Reuse
`deviceSharedTransformerControlFallsBackWholeRun` inverted back to
`...MatchesCpu` at 1e-12 with tap-moving scenarios
(`createWithTransformerSharedRemoteControl`, targetV low enough that BOTH members
move taps — verify via the CPU report that DISTR_RHO actually drove both, and that the
group survives the parallel-line outage). AFTER_GENERATOR first; then a
WITH_GENERATOR (SimpleTvc) twin.

**A5. Member outage (the original goal).**
- New context setter `setBatchDrCoefOverrides(handle, int[] slotContrib, double[] coef)`
  — clone of `setBatchDqWeightOverrides` (kernel: `fillDistrRhoBatch` gains the
  `{s, contrib}` scan; two launch sites: main fill + adjoint).
- Qualification (`GpuBatchQualification.planScenarios`): a contingency outaging member d
  of group G (found via `distrRhoControllerBranches` — after A1 make that set
  existence-keyed too) becomes batchable when the group KEEPS ≥ 1 enabled member AND
  **d is not the FREE member** (free role = first enabled — outaging it shifts the
  pilot-Q row to another member's rho column, which is per-scenario scatter re-routing;
  reject to CPU, step A6 territory). Overrides are ANALYTIC:
  survivors' self = `1/c' − 1`, cross-survivor = `1/c'`, any contribution touching d's
  rho (row or col) = 0. Per-scenario row-mode: d's rho row stays/becomes identity
  (mode 0) — matches the CPU's re-activated BRANCH_TARGET_RHO1; target = base rho
  (the BRANCH_TARGET_RHO1 target already equals it — verify, else override).
- The chunk solver assembles per-chunk pairs exactly like `dqOvIdxList`.
- Test: 3-member group (extend the factory with a third T2wT), outage the SECOND member
  (non-free), tap-moving scenario, 1e-12; outage the FIRST member → assert CPU routing
  (per-contingency, batch still engages on the others); outage down to one member.

**A6 (stretch). Free-member outage.**
Needs the pilot-Q row to move to another member's rho column per scenario — the
alt-dest/selRow pattern (`computeQRowAlt` covers only the two-candidate bus case today;
a group needs member-indexed candidates). Design sketch: extend qRowAlt to a per-group
candidate LIST + a per-scenario select index channel; the cb/sh fill kernels already
consult `(altDest, selRow, mode)` — generalize the select to an index. Do NOT start
this before A1–A5 are landed and benched.

### Part A landing checklist
- [x] A1 pack (existence-keyed, all-enabled config, free = first) — 2026-09-06; nnz bench row still owed
- [x] A2 extractor acceptance test (host-only: sharedGroupPackAtDisabledBaseAndOnActivation)
- [x] A3 group controller (per-member off-deltas via subset toggling; free-member clamp -> STATUS_REJECTED
      per-scenario CPU route; SimpleTvc converted) + all TVC tests green at 1e-12
- [x] A4 deviceSharedTransformerControlMatchesCpu (AFTER_GENERATOR, engagement verified at targetV 33.0);
      gates now flavor-specific: AFTER_GENERATOR + WITH_GENERATOR lifted, INCREMENTAL /
      useInitialTapPosition / JAVA hybrid stay gated. WITH_GENERATOR shared-group SA test still owed
- [ ] A5 drOv channel + member-outage tests (non-free batched / free CPU-routed)
- [ ] docs: capability rows + gap list; memory update; both benches (cold first)

---

## Part B — DISTR_Q remote-group pin dynamics (RL ↔ remote generator VC)

Gate to lift: `GpuSaRunScope` adds `ReactiveLimits[remoteGroupPinDynamics]` (both
modes) whenever the RL loop is active and any MAIN non-local generator VC group member
has finite bus Q limits. NOTE this is harder than Part A: the dynamics are
**in-kernel** (RL fires between Newton iterations), not orchestrator-level.

### CPU semantics (verified 2026-09-06)

- `ReactiveLimitsOuterLoop.checkPvBus` iterates ALL controller-capable buses, remote
  members included: over-limit ⇒ `controllerBus` → PQ
  (`setGeneratorVoltageControlEnabled(false)`) ⇒ updater ⇒
  `updateRemoteVoltageControlEquations(…, DISTR_Q, BUS_TARGET_Q)`:
  the pinned member's DISTR_Q deactivates, its BUS_TARGET_Q activates (target = limit −
  loadQ), the FREE role re-picks "first enabled", `vEq` stays active while any member
  remains enabled. qPct weights do NOT renormalize on an RL pin (only on element
  DISABLE — `updateReactiveKeys` filters `isDisabled`, not enabled).
- Switch-back (`checkPqBusWithQLimitType`): pinned member, v-condition against the
  **pilot's** target ⇒ re-enable ⇒ DISTR_Q reactivates, BUS_TARGET_Q deactivates,
  free role re-picks.

### Device gaps

1. RL pack membership: an UNPINNED remote member has no own `BUS_TARGET_V` ⇒ not in the
   pack ⇒ the device can never pin it. A BASE-pinned member IS in the pack
   (`basePqCapable`), but its switch-back writes mode 0 + vtarget at its own v row —
   the CPU instead reactivates DISTR_Q.
2. The q-scatter routing is a structure-time snapshot
   (`computeQRow`: unpinned member ⇒ `qRow = −1`, its own Q scatter SUPPRESSED and its
   flows feed the DISTR_Q rows; pinned member ⇒ own row). A pin/unpin must re-route the
   member's own q1/q2/shunt scatter AND remove/add the DISTR_Q emissions at its row —
   per scenario, inside the solve loop.
3. Free-role re-pick on pin of the free member: the pilot's Q row moves to another
   member's v column (same class as A6).
4. The JAVA hybrid re-extracts activation only — it cannot express any of this either.

### Implementation steps

**B1. RL pack: admit remote members.**
`GpuAcOuterLoopPacks` RL section: pack every member of a MAIN non-local group with
finite limits, marked with a new per-entry kind `rlRemote[i] = 1` and its pilot's
vtarget (the switch conditions test the PILOT's voltage — `getBusV(controllerBus)`?
**Verify against the CPU first**: `checkPqBusWithQLimitType` uses
`getBusV(controllerCapableBus) < getBusTargetV(controllerCapableBus)` — bus-local or
pilot? Read `getBusTargetV` for a remote controller before coding; this determines
which v row/vt the kernel compares).

**B2. Structure: alternate scatter destinations for member rows.**
Extend the dyn-relocation machinery: for each remote member bus, the cb/sh fills need
TWO destinations for its own q outputs — suppressed/`DISTR_Q`-side (unpinned) vs own
row (pinned) — selected per scenario by the member's pin state. The existing
`(altDest, selRow, mode)` select in `fillClosedBranchBatch`/`fillShuntBatch` is the
template (`qAlt1/qAlt2` slots in cbIdx, `[4]` in shIdx are already threaded); the
DISTR_Q emissions at the member's row need the inverse gate (emit only while
unpinned — a per-contribution `dqSel` row check against mode/limitType, like `pcSel`).
The pin-state signal available to fills is the row MODE only — define: member row
mode 1 + `limitType != 0` = pinned … but limitType is an RL-array, not indexable by
row in the fills. Cleaner: introduce row mode 4 = "pinned remote member" (RL kernel
writes it instead of 1; applyRowModeBatch treats it as 1) so fills can gate on
`mode == 4` without new arrays. THIS IS THE CORE DESIGN DECISION — prototype it first
on a 2-member group before generalizing.

**B3. RL kernel: remote transitions.**
`reactiveLimitsBatch` gains the remote branch (keyed on `rlRemote[i]`):
- PV→PQ: q computed from the member's own flows — but qBuf at its row holds the
  DISTR_Q value while unpinned, NOT the bus q! The member's own reactive injection
  needs its own accumulation (a second qBuf pass over member buses, or reuse the
  suppressed-scatter alt destination to accumulate into a scratch row). Decide with
  B2's prototype.
- transition writes: mode 4 + target = limit − loadQ; switch-back: mode 1 + target 0
  (DISTR_Q) — and the free-role/vEq bookkeeping when ALL members pin (vEq off = pilot
  V row → mode 1) — enumerate the states with a table BEFORE coding; add
  `OLF_GPU_BATCH_TRACE` lines for every transition.

**B4. Free-member pin (needs A6's generalized select).** Until then: per-scenario
reject (route the scenario to CPU when the FREE member would pin — detectable only
in-flight ⇒ conservative pre-gate: free member has finite limits ⇒ scenario-level
reject stays; group still batchable when only NON-free members are pinnable).

**B5. Validation.**
- Extend `createWithGeneratorRemoteControl` with limits chosen so a CONTINGENCY (not
  the base) pins g2 — verify via CPU report that the pin happens POST-contingency.
- Unpin scenario: base-pinned g2 + a contingency that relieves it (verify switch-back
  in the CPU report).
- All-members-pinned scenario (vEq deactivation).
- The JAVA hybrid on the same networks (it needs B2's structure + its own activation
  re-extraction is then sufficient? NO — the hybrid uses the same fills, so B2/B3 give
  it the mechanism; verify).
- Both benches after (kernel-side changes! same-day A/B, cold first).

### Part B landing checklist
- [ ] B0 read `getBusTargetV`/`getBusV` remote semantics; write the state table
- [ ] B1 RL pack extension
- [ ] B2 mode-4 + alt-dest prototype on a 2-member group (sequential test first)
- [ ] B3 kernel transitions + trace
- [ ] B4 free-member pre-gate kept
- [ ] B5 tests (post-contingency pin / unpin / all-pinned / hybrid) + benches
- [ ] gate lifted; docs + memory updated

---

## Suggested order

A1→A5 first (orchestrator-level, no kernel state machine, immediate user value:
shared groups are in real datasets). Then B0–B2 as a spike to validate the mode-4
design before committing to B3. A6 and B4 (free-role dynamics) last, sharing the
generalized per-scenario select machinery.
