# Task list (recovered 2026-07-21)

The task tracker (an MCP server) disconnected mid-session, so this file is the durable
record. It was reconstructed from the 2026-07-21 session transcript: every entry below is
backed by a commit, a measurement, or an explicit decision recorded in that session.

**Fidelity boundary.** Only tasks referenced during that session are here. Numbers below
#24 from earlier sessions are NOT recoverable from it (except #18, which came up
explicitly) — if the tracker ever comes back, reconcile rather than assume this file is
complete. Keep the numbering: it is cited in commit messages.

## Open

### #61 — GPU SA host overhead: the GPU is no longer the bottleneck (2026-09-21)

H100, rte6515, 14,260 contingencies (all kinds but busbar sections), RTE params: GPU arm ~47–51 s on one
host thread vs CPU 118.6 s (8 threads, warm) / 68.5 s (16), 0 differing violations. Of the GPU arm, only
~8 s is device solve: the rest is single-threaded host code — full `NetworkState.restore` per contingency
(14.9 s), the per-contingency host loss redistribution the device already did (4.6 s), full branch
recompute on state injection (6.1 s), connectivity copies in qualification (2.55 s), … Chunk size is
closed (flat above ~2000). Leads L1–L9, their constraints, gains, risks and the order to do them in:
**`docs/gpu-sa-host-overhead-assessment.md`** — read it before touching any of them. It also records L10, a
RESULT defect — `getDistributedActivePower()` differed on 5,095 contingencies (up to 124.5 MW) and the base
reported 0: the device's own DS passes were never reported — FIXED 2026-09-22 (worst now 2.2e-6 MW); and
L11, an upstream OLF unit bug in injection threshold conditions — FIXED on olf-local-patches (e97bb4e57).

### #60 — Generator and load contingencies on rte6515: first fold (2026-09-16)

Both kinds have been batchable and unit-tested for a while; what had never been run is a REAL set.
`GpuVsCpuRte6515SaTest` can now pick its kinds — `OLF_GPU_SA_CTG_KINDS=branch,generator,load`, with
`OLF_GPU_SA_GEN_MIN_KV` (default 400) and `OLF_GPU_SA_LOAD_TOP` (default 50) — so a first pass is a
small deliberate slice, not every injection. The all-branch 9042 set stays the default and the reference.

**The transmission level in this case is 380 kV, not 400.** Generators by nominal voltage:
3 kV … 63 kV=550, 90 kV=232, 150 kV=18, 225 kV=63, **380 kV=106**. A 400 kV filter selects nothing.

**How it folds** (106 generators at 380 kV + the 50 biggest loads, 156 contingencies):

| | value |
| --- | --- |
| batched | 83 / 156 at first fold, **101 / 156** after the two fixes below |
| violations | CPU 14073, GPU 14070 at first fold, **14073 / 14073, 0 differing** now |
| warm vs CPU-8t | x0.69 — still slower, because a third of the set is CPU-routed |

Loads alone: 41 / 50 batched and **0 differing violations**. The two kinds fold differently and the
generator side is where the work is.

**Three things to correct before going full throttle.**

1. **`PLAN:GEN_OUTAGE_SHAPE` rejected 64 of the 106 generator contingencies** — by far the biggest item.
   The reactive-capability CURVE channel's `p_now` linkage models the DISTRIBUTED-SLACK delta only
   (`pNow = basePpu + share * (target[pRow] - baseRowTargetP)`), so an injection override at a curve bus's
   P row would charge every generator on that bus a share of the lost generator's own output. Losing a
   generator IS such an override, which is why it hit most of them.

   **Half of it is now batched, on an argument rather than a model change.** When the outage takes the
   bus OUT of voltage control (`GenOutage.toPq()` — losing the only generator at a bus does exactly
   that), its reactive-limit entry is inert for the rest of the solve: the RL pass's unpinned branch
   wants a V row and finds a power row, its pinned branch wants a limit type and finds none. The curve
   channel still writes a dynamic limit for that entry and nobody reads it, and a skew that cannot reach
   an answer is not a reason to leave the batch. **64 rejections -> 46, batched 42 -> 60 of 106**, with
   the branch reference set untouched at 9040/9042 and the same two lines.

   What stays on the CPU is the case the argument does not cover: a bus with SEVERAL generators where the
   survivors still control voltage. Those need the model change — `pNow` corrected by the outage step so
   only the DS part is shared, the dead generator's own curve entry dropped, and the bus's aggregate
   `rlMinQ`/`rlMaxQ` reduced by its limits. That needs per-GENERATOR outage identity in the curve channel,
   which the plan currently aggregates per bus.
2. **`PLAN:INJECTION_OUTAGE_SHAPE` rejects 9 of the 50 loads** — the shape computation returns null for a
   slack bus, more than one distributed-slack bus, or a remote generator. Tagged `PLAN:SHUNT_SHAPE` until
   now, a copy-paste from the block above it; the census is how we pick what to fix, so a tag that lies
   about the reason sends the next person at the wrong gate. Renamed.
3. **Three differing violations, all on generator contingencies** (GEN-5997, GEN-5997#0, GEN-6130, each
   −1). **Two of the three are now fixed, and the cause was a real modelling gap**, not a knife edge.

   A probe settled it in one run: routing out every generator outage at a bus that KEEPS voltage control
   took the difference from 3 to 0. The device's `rlMinQ`/`rlMaxQ` are RUN-CONSTANT — the bus aggregate as
   extracted — while OLF's `bus.getMinQ()`/`getMaxQ()` sum the generators still there. Lose one generator
   at a two-generator bus and OLF's reactive limits shrink while the device's do not, so the two arms
   disagree about when that bus pins, and every equation still converges on both.

   `GenOutage` now carries the reactive-limit entry and the capability the lost generators take with them,
   and it reaches the kernel as a per-(scenario, entry) delta on the same dynamic-limit arrays the CURVE
   channel writes. Those arrays existed only when curves did; the selection is now `useDynRlLimits`, true
   for either channel, and they are allocated lazily when only the new one is live. **3 differing -> 1**,
   the branch reference set untouched at 9040/9042 with the same two lines, suite 261/0.

   **GEN-6130 is fixed too, and all three hypotheses above were wrong.** The instrument that found it:
   `OLF_GPU_SA_CTG_ONLY=<id>` runs one contingency, describes its bus, and monitors every voltage level.
   It said the difference was LOCAL — 0.995 kV at that node against 2.2e-4 kV everywhere else — and
   `OLF_GPU_GENOUT_TRACE` then said `busPv=false`, `busGens=1`.

   So the bus was ALREADY PQ: the base loadflow's reactive-limits loop had pinned it. The RL pack keeps
   such a bus as a `basePqCapable` entry with a seeded limit type, exactly so a RELIEVING contingency can
   switch it back to PV — and with its only generator gone there is nothing to switch back to, while OLF
   has no controller there at all. `GenOutage` now reports when a bus keeps NO generator and the entry is
   masked for that scenario through the same channel a contingency-stranded controller uses.

   **The whole set now agrees**: 106 generators at 380 kV plus the 50 biggest loads, 14073 violations on
   both arms, 0 differing. On GEN-6130 alone, every one of the 29832 buses agrees to 3.6e-13 RELATIVE
   (worst absolute 5.7e-11 kV, which moves run to run — round-off, not an offset). Note that an absolute
   1e-12 kV bar is below one ulp of a 400 kV bus (~5.7e-14), so the instrument reports both.

### #59 — Both routes build their phase tables with one function (2026-09-16)

The base load flow and the batched security analysis called the SAME nine builders with the SAME
arguments from two places, 150 lines against 60. `GpuBatchPhases.build` is now the only one, taking a
`Request` of which loops the run configures (the analysis fills it from its `BatchConfig`, the base from
the configured outer loops) and returning every phase's controllers.

Building is shared; INSTALLING is not and cannot be — the base uploads straight onto its batch-of-one
context, the analysis builds before its chunk context exists and uploads per chunk. HVDC stays inline on
the base path for the same reason: it applies its freeze into that solve's own target overrides.

Two accidental divergences fell out of merging them. The base constructed a fresh
`GpuBatchElementMapper` per builder, three times over, where the analysis built one and shared it. And a
parameter: the base fed the flow builder the incremental transformer REACTIVE POWER loop's
`getMaxTapShift()` while I merged it onto a `Request` field I had named after the transformer VOLTAGE
loop, because the `BatchConfig` field it comes from on the analysis side is named
`incTvcMaxTapShift` and holds the ratio-tap-control value. The test caught it at 1.3e-4 kV on BUS_2 —
`deviceIncrementalTransformerReactivePowerControlMatchesCpu` — and the field is named
`rpcMaxTapShift` now.

Behaviour-neutral otherwise: suite 261/0, rte6515 unchanged at 9040/9042 with the same two lines.

**Second half, same day:** the tables now TRAVEL as that record too. `solveQualifiedInChunks` took ten
positional phase arguments, several of the same type and several nullable, which is precisely how a new
phase gets mis-wired — a swapped pair compiles. They are one `GpuBatchPhases.Built` parameter now, the
equation-system context folded in with them, and the chunk solver's body reads them off the record.

### #58 — PLAN: what "every builder is total" would take (scoped 2026-09-16)

Totality would let the four host-pause slots go. Since #57 it is an OPTIMISATION, not a correctness
requirement: a builder that cannot represent a controller now hands the whole loop back loudly instead
of dropping it silently. So this is about how often a loop leaves the device, not about whether the
answer is right.

The nine skip sites fall into four classes, and only one of them is real work.

**1. Already impossible — turn the skip into an assertion.** `tapRows.length == 0` on a tap controller.
Since the tap table moved off reflection onto `PiModelArray.getModels()` (#52), an empty table means the
branch carries a continuous `SimplePiModel` and has no taps, which a voltage or phase controller never
does. Five sites. Cost: an afternoon, and it converts five "can't happen" branches into proof.

**2. Not skips at all — stop counting them.** `branch.isDisabled()`, `!isConnectedAtBothSides()`,
`shunt.getControllers().isEmpty()`. OLF's own loops skip these too, so representing them is not owed.
They share a `continue` with the real refusals today; splitting the two is mechanical.

**3. The real work: a variable that is not in the equation system.** `rhoRow` / `a1Row` / `vRow` /
`shuntBRow` / `rlIdx` null. One root cause: the row index is taken at extraction time, and several loops
run with their control DISABLED at the base, so the variable does not exist yet. The project already has
the fix pattern and uses it twice — `GpuBatchContinuousPhase` toggles a CONTROLLER-mode control on and
diffs the activation, `GpuBatchTransformerControl` does the ON/OFF toggle diff — and the standing lesson
says it outright: never key a pack or a guard on control-ENABLED state, key it on the equations
EXISTING. Generalising the toggle-and-diff to every builder is the piece that would move the needle.

**4. Genuinely unmodelled, and honest about it.** A control element that is zero-impedance in AC (the
element mapper packs those through the zero-impedance family, not as a branch), and a controller or SVC
shunt whose conductance varies with section (`elemIndex < 0 && gVaries`) — the device cannot carry the g
change for a shunt it does not pack. These want extraction work or a whole-run gate, not a per-controller
skip.

**Do not chase this blind.** Zero skips on rte6515 and across the whole 261-test suite, so there is no
measured case to fix. `GpuBatchSkips.summary()` is on every run and both routes log when it is non-empty:
let a production network name a class-3 or class-4 case, then fix that one.

### #57 — One ownership rule for both routes, and the partial-representation hole (2026-09-16)

Asked why the four host-pause slots are needed when every loop has a kernel. Answering it found the
hole they were only half covering.

Every phase builder walks OLF's controllers and SKIPS the ones it cannot express — a control whose
variable is not in the equation system, a branch the extraction does not pack. Nine such sites, all of
which only LOGGED. That gives three cases, and only two were handled:

| builder result | what happened | correct? |
| --- | --- | --- |
| complete | kernel drives the loop | yes |
| EMPTY | kernel declines, the base pauses and the SA refuses | yes |
| PARTIAL | kernel takes the loop, the pause slot is CLEARED, the skipped controllers are dropped | NO |

The partial case is the dangerous one precisely because nothing flags it: the device drives a loop on a
subset of what OLF drives and every equation still converges. So the slots were the all-or-nothing net,
covering only the all-or-nothing case.

`GpuBatchSkips` now counts the skips per thread (the SA builds on OLF's partition threads) and
`GpuBatchPhaseOwnership` turns the count into the one decision both routes take: a loop is device-owned
only when its representation is COMPLETE. Incomplete hands it back — the base load flow clears the
kernel tables and restores the host-pause bit so OLF's own loop runs at its own slot, and the batched
SA, which has no pause slot for anything but secondary voltage control, routes the run to the CPU.

**Measured before enforcing**: zero skips on rte6515 and across the whole 261-test suite. So the rule
is behaviour-neutral on everything we validate against, which is also why the hole had never shown up —
and why it would have been found the hard way, on a customer network, without this.

The slots stay. Deleting them needs every builder to be provably total, which is a different piece of
work from porting AreaInterchangeControl and AutomationSystem: those two never used the pause
mechanism at all.

### #56 — The base load flow takes the same route as the SA (2026-09-16)

`SecondaryVoltageControl` was the one in-kernel phase the base load flow never handed to the driver.
The kernel landed on the chunk path, where it is the default; `GpuNewtonRaphson` builds every other
phase (monitoring, incremental TVC, incremental shunt, the flow-observed family, continuous phase,
discrete TVC with the generator freeze, simple shunt) and never built this one. So
`buildHostLoopMask` set its bit and nothing cleared it: every network with a zone PAUSED the driver
there, by construction rather than by refusal — while the same network ran it on device under the
security analysis. Two routes through the same loop, which is the divergence class this project keeps
paying for.

Now built and uploaded the same way, from the same builder, with the same inputs. Everything it needs
was already on the base path: the reactive-limit rows, the state vector, the plausible-target-V bounds
off `getNetworkParameters()`, an element mapper.

**What the chunk path did not need and the base path does: a write-back.** The batched SA restores the
network per scenario, so the moved zone targets never have to reach it; a load flow has to leave them
on the network exactly as OLF's own loop does. The first attempt read the moved target out of the
CONTROLLED bus's row, which looks right and is wrong — that row carries a Q target the moment the bus
is pinned or is a shared-distribution member, and writing 0.165 pu in as a target voltage puts the
control outside the plausible band, so OLF drops it and the run stops batching. Two JAVA-mode tests
caught it. The targets now come from the phase's own state (`dSvcTargetV`, read back per
(scenario, controller)), which is the thing the kernel actually settled on.

This also closes the most reachable route into the restart gap: a paused loop whose mutation dirties
the structure is what cancels the pause and forces a restart, and moving generator VC targets is
exactly that kind of mutation.

### #55 — Freeze a REMOTE generator voltage controller in-kernel (landed 2026-09-16)

The last whole-run CPU gate in the AFTER_GENERATOR phase. `freezeCandidates` returns null the moment
one candidate is a REMOTE controller, because the kernel's freeze writes a V identity row
(`mode 0/2 -> 1, target = qBuf - loadQ`) and a remote controller's own row is a DISTR_Q SHARE, not a V
identity. OLF itself does not care: `disableGeneratorVoltageControlsUnderMaxControlledNominalVoltage`
walks `getMergedControllerElements()` and freezes every enabled one at `getQ().eval()`, local or
remote alike. So this is a device representation gap, not a semantic one.

**The primitive already exists.** A frozen remote member is the reactive-limit remote PIN with a
different target: row mode 4 ("the row hosts its own Q balance"), `target = qBuf - loadQ`, saved mode
and target for the restore — which is exactly what `rlCheckBatch`'s remote branch writes, with a Q
LIMIT as the target instead of the current injection. `qBuf` at a member's own row is already the
member's own reactive injection (`fillRemoteMemberQBatch`), and OLF's freeze target is
`getQ().eval() - loadTargetQ`, the same expression the local freeze uses.

**What comes for free, and what does not.** Checked against the kernels:

- carrier relocation and the all-members-gone pilot flip key on `mode` (`relocChoose` counts members
  at mode 1, `updatePilotVeqBatch` flips the pilot's V identity when none is left), so a frozen member
  at mode 4 is handled like a pinned one — nothing to do;
- the DISTR_Q key renormalization keys on `limitType[(scenario, entry)] != 0`, NOT on the mode. A
  freeze that only wrote mode 4 would leave the survivors on their old shares while the frozen member
  stopped participating. It needs its own limit type.

**Shape of the change.** Add limit type 5, "frozen by the generator voltage control manager", beside
1 MIN_Q / 2 MAX_Q / 3 MIN_REALISTIC_V / 4 MAX_REALISTIC_V — the renormalization's `!= 0` test then
drops the member with no change, while the type stays distinguishable from a Q-limit pin.
`freezeCandidates` keeps remote candidates and records each one's RL entry index (−1 when local);
`FreezeTables` carries it; the kernel freeze branches on it; `rlCheckBatch` skips an entry at type 5 in
BOTH branches, since OLF's reactive-limits loop never sees a bus whose voltage control is off; the
COMPLETE restore puts mode, target and limit type back. The packed RL state already has three bits for
the type, so no JNI width change.

**Landed, and the test earned its keep.** Built as scoped, plus one thing the plan missed: freezing a
remote controller can take the LAST live member of its group, and then the group controls nothing —
OLF's freeze disables the voltage control, so the CONTROLLED bus stops carrying a V identity. That
flip is `updatePilotVeqBatch`, which the reactive-limits phase runs after its own pins; the freeze
happens two loops later (OLF order 8 against 6), so the pilot held a target for a control that no
longer existed until the next sweep came back round to RL. Measured 0.6 kV on BUS_2. The same kernel
now runs after the discrete transformer phase too, and it restores the identity at COMPLETE on the
same rule.

The test builds a remote controller whose CONTROLLED bus is under the transformer-controlled level
WITHOUT being the bus the transformers control — putting it there takes their control away and the
whole thing passes vacuously, which is what the first attempt did (`controllers=0`). Two settings: the
automatic limit freezes only the remote controller, the not-controlled side keeps a PV bus, the #53
bracket decides locally and the scenario BATCHES at 1e-12 with the freeze in-kernel; forcing the limit
to 150 kV freezes the local generator too, empties the min mask, and the bracket sends the scenario to
the CPU, where the two arms still have to agree.

rte6515 unchanged at 9040/9042 with the same two lines.

### #54 — A second SA run in one JVM re-pays the cuDSS ANALYSIS (found 2026-09-16)

Measured, not suspected. `OLF_GPU_TRACE=1` now tags every ANALYSIS with the context's batch size, and
the production bench (three GPU runs: a 20-scenario qualification pass, then cold and warm at 250)
prints six:

```
ubatch nScenarios=1   LU_NNZ=260076      <- base load flow, run 1
ubatch nScenarios=20  LU_NNZ=260076
ubatch nScenarios=1   LU_NNZ=260076      <- base load flow, run 2
CONTEXT_EVICT cached chunk=20 -> requested chunk=250
ubatch nScenarios=250 LU_NNZ=260076
ubatch nScenarios=1   LU_NNZ=260076      <- base load flow, run 3
CONTEXT_EVICT cached chunk=250 -> requested chunk=250
ubatch nScenarios=250 LU_NNZ=260076
```

Within ONE analysis the contingencies are fine: the chunk context is acquired before both the chunk
loop and the strategy-pair group, `ctx->analyzed` latches, and every chunk of every group shares that
one ANALYSIS. The base load flow pays its own, for a batch of one.

The third eviction is the finding: `chunk=250 -> chunk=250`, the same batch size and the same LU_NNZ, so the
PATTERN is identical and the symbolic analysis would have been reusable. `acquireBatchContext` misses
because it keys on `GpuAcData.sameAs`, which compares element PARAMETER VALUES (`cbIn` and friends) —
and the previous run left the network at different tap positions. The base load flow already has the
right pattern for this: `sameStructureInputsAs` + `refreshStructureValues`, reuse the device structure
and push the new values.

What makes it more than a one-line change: `createBatchContext` takes the reactive-limit and
distributed-slack packs as CONSTRUCTOR arguments (`rlMinQ`, `rlVtarget`, `dsFactor`, …) and
`GpuBatchOuterLoopTables.install` does not re-upload them, so reusing a context across runs needs
those packs to become setters like the channels installed beside them.

**Size it against the measured cost.** The trace now times the ANALYSIS: ~64 ms at the chunk's batch
size, ~80 ms for the base load flow's batch of one, and the cost is INDEPENDENT of the batch size
(63.4 ms at 20, 63.6 ms at 250) because UBATCH analyzes a single block and carries the batch size as
`CUDSS_CONFIG_UBATCH_SIZE`. So ~144 ms per run, about 1.8% of a 7.9 s warm bench and 0.2% of the
82 s rte6515 run. Worth doing, not urgent.

**And one analysis CANNOT serve two batch sizes — tested, not assumed.** A COST that does not move with
the batch size invites analyzing once and re-setting `CUDSS_CONFIG_UBATCH_SIZE`, which is what the
compaction path's tier change would avoid. Tried: cuDSS ACCEPTS the resize without a new analysis, the
production bench ran 11% faster warm (7073 ms vs 7917 ms) with EXACT violation parity, and then
pegase9241 died with `CUDA error: an illegal memory access was encountered` — the data object's
buffers are sized for the batch size it was analyzed at. The analysis does not cost more for a bigger
batch, but its STATE belongs to that batch size. The compaction path's destroy-and-re-analyze is right; the note there now says
why, so nobody re-derives the same false economy. A bench that gets faster while corrupting memory is
the reason to run the whole suite on a perf idea, not just the bench.

### #30 — Multi-stage conditional actions
Pending. The only item left **in** the operator-strategy machinery; everything else in #24
is done. Single-stage conditions (True / violation / threshold) all qualify today via
speculative precompute. Stage-on-stage needs the stage-1 state evaluated before stage 2 is
composed — i.e. a second warm continuation off the pair's own result, not one extra
scenario. That does not fit the current "a (ctg, strategy) pair is one extra BatchContingency"
model, which is why it is the remaining architectural piece.

### #34 — Slack-relocating contingencies (`ISLANDS_SLACK`)
Pending, and **measure before building**. #33 left two node-breaker shapes on CPU. The big
one: an outage that isolates the slack. CPU (`PropagatedContingency`) relocates the slack
into the largest connected component and re-anchors there; the batched pack anchors its
targets and distributed-slack participation on the BASE slack row, so a relocated slack is
not expressible as a per-scenario mask. FIRST instrument the `ISLANDS_SLACK` rejection
counter on the production 500-ctg bench to learn what share of a real node-breaker run this
costs — build relocation support only if the share justifies it. Second, smaller shape: a
switch outage leaving BOTH sides energized (the zero-impedance spanning tree would have to
be re-picked per scenario).

### #39 — Tap action on a transformer under ACTIVE voltage control
Pending; likely cheaper than when it was first carved out. `RatioTapChangerTapPositionAction`
is batched, but a transformer under active voltage control falls back to CPU, because a
controlled transformer's rho is a solver VARIABLE whose row moves (`computeQRow` relocates
the controlled bus's Q balance onto the `BRANCH_RHO1` column), so a static full-row override
cannot express the tap change. Two things changed since: transformer voltage control is now
batched (#16), and the 2026-07-20 phase-control work established exactly how control rows
pair (`BRANCH_TARGET_RHO1` / `BRANCH_TARGET_ALPHA1` identity rows vs the power row they swap
against, and how a row-mode identity flip expresses "control off, value pinned"). Plausibly
just: pin rho via its identity row at the action's tap value.
**Check CPU semantics first** — does `LfRatioTapChangerAction` on a regulating transformer
disable the control or keep regulating from the new position? Mirror it; do not invent.

### #27 — Batched-SA perf at the RTE 1e-4 criterion — RESOLVED by operating point
Kept open only for optional work. The original framing ("GPU loses at 1e-4, recover it") is
moot operationally: the per-arm eps split (`7ac0627`) measured, in ONE controlled run,
**GPU@1e-7 warm 23.52 ms/ctg vs CPU@1e-4 33.45 ms/ctg → ×1.42**, violations 305=305,
statuses identical. GPU@1e-7 is a production-or-BETTER (tighter) answer delivered faster
than the CPU delivers its production answer.
Decisive supporting finding: the CPU is nearly **eps-insensitive** (1e-4 → 33.45 ≈ 1e-7 →
33.32), so it cannot recover by loosening; the historical "CPU@1e-4 ≈ 25" figures were
cross-run machine-state noise, not an eps effect — which is why an earlier paper estimate of
~×1.1 was wrong and the same-run measurement mattered.
Remaining OPTIONAL: make GPU@1e-4 itself fast by killing the outer-loop churn (levers:
decision hysteresis — only re-check RL on scenarios whose q moved > tol; narrow fills/norms
when few scenarios are active). Worth it only if a consumer insists on eps=1e-4 on the GPU.
Keep the fixed-iter bench and the SA tests at baseline for any such work.

### #40 — Batched transposed solve (adjoint/VJP) — ONE primitive, THREE consumers
Not started. The single highest-leverage missing capability on the batched device path,
because three separate roadmap items all need the *same* thing: **a transposed solve against
the per-scenario cuDSS factorization we have already computed**, without shipping Jacobians
back to the host.

**Consumer 1 — GNN training with an implicit load-flow layer** (the driver). A converged AC
LF is an implicit function `F(x, p) = 0`; by the implicit function theorem
`dx*/dp = −J⁻¹ ∂F/∂p`. Training a GNN whose forward pass contains an LF layer means
backpropagating through it, which in JAX is a `custom_vjp` whose backward is the adjoint

```
Jᵀ λ = (∂y/∂x)ᵀ ȳ          ← ONE transposed solve, reusing the factorization
p̄    = (∂y/∂p)ᵀ ȳ − (∂F/∂p)ᵀ λ
```

Reverse mode is what training needs (many parameters → one scalar loss), so this is one
transposed solve total, versus one solve per variable in the forward-mode sensitivity
analysis that exists today. Design already worked out in
`~/Projects/energnn/project-tertiary-voltage-control/docs/`:
`differentiable_power_flow.md` (the `custom_vjp` boundary, the phased plan, and §4's list of
what OLF must expose), `adjoint_block_and_vjp_math.md`, `energnn_vjp_training_integration.md`.
The reverse-mode derivation — specifically *why* each transpose is forced rather than a
layout convention — is in `~/Projects/powsybl/powerflow/pyloadflow/docs/reverse_mode_vjp_explained.md`.

**Consumer 2 — SecondaryVoltageControl, batched.** ~~Its sensitivity RHS is built on existing
`BUS_TARGET_V` rows and solved with `j.solveTransposed`~~ — NOT a consumer (corrected
2026-09-15): OLF's `solveTransposed` is the Newton direction, and the batched port already
solves it forward on the driver's factors (see the loop table below).

**Consumer 3 — IncrementalPhaseControl, batched.** Same primitive: a transposed solve with
`toRadians(1)` injected at the `BRANCH_TARGET_ALPHA1` row, one column per controller.

**PRIOR ART — do not start from scratch. Three layers already exist:**

1. **A working CPU adjoint in OLF.** `~/Projects/powsybl/powsybl-open-loadflow-5`, branch
   **`adjoint-vjp`** (~12 commits, head `a9de15a3d`). `AcSensitivityAnalysis.runAdjoint(...)`
   is exposed as a public SPI entry on `OpenSensitivityAnalysisProvider`. It contracts
   `θ̄ = Sᵀ·ȳ` **without materialising S**, using a single
   `context.getJacobianMatrix().solveTransposed(factorsStates)` and an **O(functions+variables)
   minimal factor set** (validated against the full cross product). The public surface is
   deliberately small: `runAdjoint`, `AdjointBlock`/`AdjointVariable`, `functionCotangentKey`.
   Returns the UNSCALED sensitivity (physical dual of `get_sensitivity_matrix`).
   10 tests in `AcSensitivityAnalysisAdjointTest`, including **finite-difference gates**
   (`runAdjointMatchesForwardSensitivityAndFiniteDifferenceOnIeee14`), distributed slack, the
   SVC pilot lever, line admittance, linearity in the cotangents, and silent-failure guards
   (rejects a cotangent keying no block function; rejects empty declarations) — the commit
   log shows those guards were added after real silent-zero/silent-drop bugs.
   **This is the semantics the GPU path must reproduce, not reinvent.**
2. **The lifetime mechanism, already solved on CPU.** `runAdjoint` reuses the AC load flow
   retained in `NetworkCache` (`networkCacheEnabled`) — the cached LF *is* how the
   factorization survives between the forward and the adjoint call. The GPU equivalent is the
   open question in item 2 below.
3. **The Python prototype that preceded this OLF GPU work** —
   `~/Projects/powsybl/powerflow/pyloadflow`. `docs/reverse_mode_vjp_explained.md` derives why
   each transpose is *forced* vs a layout convention; `docs/jacobians_and_implicit_diff.md`;
   and `docs/batched_solves_cpu_gpu.tex` works the batching design explicitly, including
   §"Scenario B — the forward *with* a gradient (VJP)" and §"Four ways to lay out B systems".

**What the batched GPU path already has:** UBATCH cuDSS with a per-scenario LU over a fixed
superset CSR pattern — exactly the "uniform batch" layout that the tex note identifies as
correct when members share a sparsity pattern (as N-1 scenarios do).

**What is missing — i.e. the GPU/batched delta over the working CPU adjoint:**
1. **Batched transposed solve against the existing per-scenario factorization** — cuDSS
   solve-transpose on the factors already computed, no refactorization, no host round-trip.
   CPU does this with one `solveTransposed` on a retained `JacobianMatrix`; the device
   equivalent over a UBATCH of factors is the core new piece.
2. **Factorization LIFETIME across the fwd/bwd boundary.** The requirement that differs in
   kind from security analysis, and the easiest to underestimate. SA frees its factors as soon
   as the forward converges; a VJP must keep the per-sample factors alive *between* `f_fwd`
   and `f_bwd`, and JAX may separate, reorder under `jit`, or repeat those calls. CPU solved
   this via `NetworkCache`; the device needs a context with `create → adjoint_solve → free`,
   an opaque handle in the `custom_vjp` residual, matrices staying in device memory, and
   thread-safe re-entrancy for parallel training. NB our batched solver already had a
   JVM-global-state incident (#38) — do not repeat it; make context state per-call.
3. **`(∂y/∂x)ᵀ` RHS extraction and `(∂F/∂p)ᵀ` products** on device for a declared
   function/variable set (CPU's `buildAdjointFactors` + the scatter of `der()` into equation
   rows is the reference implementation).
4. **Batching for different networks — cuDSS's NON-UNIFORM batch is the primary answer, and
   pyloadflow already validated it.** Training runs on **different networks**, so the uniform
   UBATCH the entire SA path rests on (one shared sparsity pattern, B sets of values) **does
   not apply to a training batch**. Of the four layouts in
   `pyloadflow/docs/batched_solves_cpu_gpu.tex` §"Four ways to lay out B systems":
   - **(3) non-uniform batch — the fit.** cuDSS accepts a *list* of differently-patterned
     matrices as one batched call: each member keeps its own pattern, the library overlaps
     their work; no shared symbolic analysis, but one launch instead of B. Available since
     **cuDSS ≥ 0.4.0** (we pin 0.8.0). pyloadflow states the batched *"non-uniform / uniform
     cuDSS solve confirmed for the VJP"* (`docs/gpu_security_analysis.md`), and its SA design
     already **buckets** work into pattern-preserving → uniform vs topology-changing →
     non-uniform, with a classifier in `jacobian/pattern.py`. Port that bucketing idea rather
     than reinventing: different networks simply always land in the non-uniform bucket.
   - **(4) block-diagonal / fused** — glue the B grids onto one diagonal: one analysis + one
     factorization, no fill between independent blocks, and the transpose of a block-diagonal
     is block-diagonal with each block transposed, so the orientation fix composes unchanged.
     Viable alternative, but its symbolic analysis is per batch *composition*, so it only
     amortizes if membership is stable across steps; reshuffled minibatches could make
     re-analysis dominate.
   Decide by measurement (non-uniform's per-member analysis vs block-diagonal's per-composition
   analysis), not by argument. **Do not treat block-diagonal as the only option** — the
   non-uniform path is the one with existing evidence behind it.
   Note consumers 2 and 3 (SVC, incremental phase control) run INSIDE the SA batch where the
   pattern IS shared, so they stay on the uniform path and never hit this — another reason to
   do them first.
5. **Non-smoothness.** `x*(p)` is piecewise-smooth: PV↔PQ switching on Q-limits, tap limits,
   distributed-slack participation changes. The correct derivative is that of the locally
   active KKT system, so the active set must be exposed and freezable while linearizing.
   `differentiable_power_flow.md` §6 calls this the correctness frontier — prototype on a
   network with known binding Q-limits before trusting it broadly. (CPU's adjoint already
   handles distributed slack and the SVC pilot lever, so check what it does there first.)

**THE ORIENTATION PROBLEM — the crux, and why the CPU adjoint does not port.**
CPU and GPU store the Jacobian in **opposite conventions**, deliberately:

- **GPU** (`olf_gpu_nr_jni.cu` header): *"J is true CSR with row = equation column, col =
  variable row, so cuDSS solves `J dx = -F` directly — no transposition games (contrast
  powsybl-math's CSC-as-CSR convention, which exists to serve solveTransposed)."*
- **CPU** (powsybl-math): CSC-as-CSR, i.e. the stored matrix **is `M = Jᵀ`**.

The consequence is the opposite of what one first assumes, and the CPU adjoint says so in
code: `// λ = J⁻ᵀ x̄: the stored matrix is M = Jᵀ, so plain solve() IS the adjoint solve. One
solve.` → `context.getJacobianMatrix().solve(lambda)`. On CPU the **backward is a plain
`solve()`**; `solveTransposed()` is what the **forward** sensitivity uses (`J (dx/dp) = −∂F/∂p`).

**On CPU BOTH directions are free, and the convention is not what makes them so.** KLU
implements `klu_solve` *and* `klu_tsolve` against the **same numeric factorization** — a
transpose solve is just the triangular solves taken in the other order, at the same cost and
with no refactorization. So one factorization serves both directions. The storage convention
only decides which method *name* maps to which mathematical direction; it costs nothing
either way:

| | stored | one factorization gives | |
|---|---|---|---|
| CPU (powsybl-math + KLU) | `Jᵀ` | `solve()` → `J⁻ᵀ` (**backward**) | `solveTransposed()` → `J⁻¹` (**forward**) |
| GPU (ours, true CSR + cuDSS) | `J` | `solve()` → `J⁻¹` (**forward**) | **transpose solve: not available** |

**So the problem is not our convention — it is a missing cuDSS capability.** cuDSS (pinned
0.8.0) offers only the non-transposed solve against its factors, so a GPU factorization yields
**one** direction instead of two. Flipping our storage convention would not help: it would
merely move the unavailable direction from the adjoint to the Newton, which is strictly worse
(the Newton does many solves per sample, the adjoint one at the end).

**The precise ask of cuDSS is parity with KLU: a transpose solve against an existing
factorization.** The cuDSS team has said this is in progress. The moment it ships, our
true-CSR convention costs nothing, the workaround below deletes, and the adjoint becomes one
extra solve rather than an extra factorization.

**Resolution while cuDSS lacks it (works today):** keep the forward Newton in its current
orientation and, at convergence, build `Jᵀ` explicitly, factorize it **once**, then the adjoint
is a plain non-transposed cuDSS solve. This buys the second direction with a second
factorization — precisely the cost KLU does not pay.
- The value transpose is a **gather with a precomputed permutation**: for a given sparsity
  pattern the `CSR(J) → CSR(Jᵀ)` index map is symbolic, computed once per distinct pattern,
  then each sample's values are permuted by one kernel. cuDSS's symbolic analysis for `Jᵀ` is
  likewise once per pattern.
- Cost: **+1 factorization per sample per training step**, against the ~4–8 refactorizations
  the forward Newton already performs — roughly 12–25% overhead. That is the price of no
  solve-transpose, and it is acceptable.

**Build the workaround to be DELETED.** When cuDSS ships the transpose solve, the explicit
`Jᵀ` build *and* its factorization both disappear and the GPU reaches KLU parity: one
factorization, both directions. So put the adjoint behind a small seam (one function: "solve
with `Jᵀ` given the forward's factors") whose first implementation is transpose-and-
refactorize, so the swap is local rather than a rewrite. Re-check on every cuDSS bump —
`native/cudss-swap.sh` already does the A/B, and cuDSS version choice is already tracked as a
decisive perf lever.
(If cuDSS instead exposes the raw `L`/`U` factors, the same seam takes the other classical
route: `J = LU` ⇒ `Jᵀ = UᵀLᵀ`, so an adjoint solve is two transposed triangular solves against
the existing factors — again one factorization, which is exactly how KLU's `klu_tsolve`
works.)

**Active-set / control handling in the CPU adjoint (checked 2026-07-21).** The reassuring
finding for the GPU port: **both distributed slack and the SVC pilot are handled entirely on
the RHS (`∂F/∂p`) side — neither touches `J`, its factorization, or the transpose machinery.**

- **Distributed slack** enters only as `slackParticipationByBus`
  (`AcSensitivityAnalysis:447-456`), folded into the RHS by
  `initFactorsRhs → fillRhsSensitivityVariable → factorGroup.fillRhs(rhs, participationByBus)`
  (`AbstractSensitivityAnalysis:772-794`). With `isDistributedSlack()` it is
  `getParticipatingElements(...)` mapped to `-factor` per bus; otherwise the single entry
  `{slack bus: -1.0}`. So a parameter perturbation carries its slack redistribution inside the
  `∂F/∂p` column. Note this takes the participation **at the converged state** and treats it as
  fixed — i.e. it IS the "freeze the locally-active branch" linearization the differentiability
  design calls for, already in practice.
- **SVC pilot closed loop** is `fillSvcPilotFactorsRhs` (`AcSensitivityAnalysis:250-296`) →
  `SvcPilotPointClosedLoopSensitivity.buildCoordination(context)` →
  `coordination.weightsForPilot(pilotBus)`, which fills that variable group's RHS column with
  per-pilot weights over the controller buses. The coordination system is a **separate small
  matrix with its own `LUDecomposition`** (`SvcPilotPointClosedLoopSensitivity.Coordination`,
  field `luB`), NOT a modification of the LF Jacobian — and it is built + factorized **once**
  because it is pilot-independent, then reused for every queried pilot.
- The adjoint reuses the forward's RHS builder verbatim: `analyseAdjoint` calls the same
  `initFactorsRhs` + `fillSvcPilotFactorsRhs`, commented *"∂F/∂p columns (one per variable
  group), including the SVC pilot closed-loop — identical to forward."*

**DECIDED — the SVC coordination matrix stays on CPU in the end-goal design, for the adjoint.**
It is small, and per converged state it costs one LU plus one back-substitution per pilot. It
is pilot-independent, so it is built once per state and reused for every pilot. Porting it to
device would be substantial work (a second solver path for a small system, exactly the regime
where a GPU is latency-bound and loses) for no measurable gain. For the adjoint the data
crossing the boundary is small — per-pilot weights used to fill an RHS column — and the host
boundary is already being crossed anyway, since the cotangents come from the host. So: keep it
on CPU, deliberately, and do not revisit.

**ALSO SETTLED — the SVC OUTER LOOP may stay host-side in batched SA too. Investigated
2026-07-21; my earlier "the synchronization could dominate" worry was overstated.**
The batched architecture is ALREADY hybrid, and transformer voltage control (#16) is the
proven production precedent:
- **On device, inside the native solve loop:** distributed slack + reactive limits — the HOT
  loops, iterating many times per solve, where host round-trips would be fatal.
- **On host, orchestrated around whole batched solves:** transformer voltage control.
  `orchestrateTransformerControl` reads `chunkResults[slot]` (the converged state per
  scenario), loops `for (int step = 0; step < 8; step++)`, and each step rebuilds per-scenario
  `x0` / row-modes / targets and calls `solveBatchContext` again, copying `r2[slot]` back for
  the affected slots.

The decisive fact: **`solveBatchContext` already returns the full per-scenario state to the
host after every batched solve.** A host-orchestrated outer loop therefore costs **no extra
transfer** — that data crosses regardless. And the device fills honour an `active[s]` mask
(15 sites in the JNI), so an orchestration step only does work for the scenarios still moving.
So the state changing per iteration is not an objection: re-solving on changed state is exactly
what the orchestrator does, and TVC already does it in production under RTE parameters.

What this means for SVC: **follow the TVC pattern** — host-side orchestration around batched
solves, with the coordination matrix built on host per converged state (see above). The real
cost driver is **iteration count, not data volume**: each orchestration step is one batched
re-solve. TVC bounds itself at 8 steps; SVC must be similarly bounded, so check its
convergence behaviour before wiring it. The residual host cost is B small coordination
builds+LUs per outer iteration — at B ≈ 500 these are tiny matrices and parallelizable across
host threads, so measure it but do not pre-optimize. Only if that measurement surprises does
batching the B small systems (block-diagonal / non-uniform cuDSS) become worth it.

**Consequences for the GPU work:** (a) the transpose/factorization design above is orthogonal
to both controls — they do not complicate `Jᵀ`; (b) what must be ported is the **RHS
construction** (participation-weighted for slack, coordination-weighted for the SVC pilot),
not new solver machinery; (c) the SVC coordination's build-once/reuse-for-all-pilots structure
is the pattern to keep on device; (d) the "freeze the active set at the converged point"
question is, for distributed slack at least, already answered by existing practice — mirror
it rather than inventing a convention.

**Acceptance gate, inherited from the CPU work:** three-way agreement — finite difference
(independent ground truth) vs forward sensitivity vs the adjoint. **The orientation mismatch
makes this gate non-optional**: an inverted convention produces a plausible-looking but wrong
gradient, and FD is what catches it. The two analytic paths
should agree to solver tolerance (same kernels, opposite contraction order); FD agrees to
truncation error and catches a systematic error common to both. `AcSensitivityAnalysisAdjointTest`
already encodes this pattern; mirror it rather than inventing a new gate.

**Sequencing note:** consumers 2 and 3 are self-contained inside our batch and would prove
the primitive on ground we control, before the JAX/pypowsybl context plumbing of consumer 1.

**FIRST STEPS (ordered).** Principle: prove the risky primitive on ground we control, before
any JAX or batching complexity.

**Prerequisite — how the CPU adjoint reaches our build. DONE 2026-07-21** (rebased onto main,
`git merge --squash adjoint-vjp` as `ec2063108` "Bring in the CPU adjoint (VJP) from adjoint-vjp",
pushed to `gb`). Plan and reasoning kept below because it documents *why* squash rather than
cherry-pick. The GPU extension builds against OLF `2.4.0-SNAPSHOT` from
`powsybl-open-loadflow-3` (`olf-local-patches`); `runAdjoint` lives on `adjoint-vjp` in
`powsybl-open-loadflow-5`. Facts checked:
- **All the OLF checkouts are WORKTREES OF ONE REPOSITORY** (`git worktree list`), so every
  object is already shared: `adjoint-vjp`'s tip `a9de15a3d` is directly reachable from OLF-3.
  **No remote, no fetch needed** — cherry-pick/merge works across the worktrees as-is.
- **`olf-local-patches` is 2 ahead / 2 BEHIND main** (merge base `6ef8e6fc7` "Bump to
  v2.4.0-SNAPSHOT"), while `adjoint-vjp` is based on current main. **So rebase
  `olf-local-patches` onto main FIRST**, then take the adjoint work. (Main's `fe0bd3dfb`
  = #1451, which adds outer-loop names to status text, touches ~60 files — that, not our work,
  is what makes a raw two-tip diff look large.)
- **`olf-local-patches` does NOT contain shunt or SVC sensitivity** — its only 2 commits are the
  shunt *reconnection action* work (#32: `TerminalsConnectionAction` on a base-open LINEAR
  shunt + the `getShuntIdsToShift()` accessor). **It does not need to:** both
  `SvcPilotPointClosedLoopSensitivity` (+ its test) and `SHUNT_COMPENSATOR_SUSCEPTANCE`
  sensitivity are **already on main**, so the rebase brings them in.
- **What to take from `adjoint-vjp`:** 21 commits vs main, but not all are wanted —
  `e7ed4632b` is already upstream as main's `fe0bd3dfb` (#1451), and four prototype commits
  are later deleted wholesale by `168999764` ("Remove the superseded
  LoadFlowAdjoint/ClassicVoltageControlVjp prototype", −608 lines). Cherry-picking the range
  verbatim would add-then-delete a prototype. **Prefer `git merge --squash adjoint-vjp` after
  the rebase** — it takes the tip state in one commit with no prototype churn. Collision risk
  is low: the adjoint touches only 9 files, almost all under `sensi/`, disjoint from the
  shunt-action work.
- After the rebase: rebuild + reinstall OLF into the isolated m2 and re-run the GPU suite, since
  the extension builds against it (this is also the moment to re-check the pinned
  `powsybl-open-loadflow.version`).

The CPU adjoint is the ORACLE for every step below, and an exact reference is far stronger than
finite differences alone — which is why it is worth doing this properly rather than comparing
against recorded numbers.

**Steps 1a and 1b are DONE (2026-07-22).** `GpuTransposedSolveTest` proves the transpose machinery
on synthetic non-symmetric matrices (~1e-16, with a guard test that transposing changes the answer),
and `GpuAdjointOnRealJacobianTest` now passes on a real ieee14 Jacobian:
`vs solve abs=5.3e-15 rel=7.4e-16 | vs solveTransposed abs=6.06e+00 | residual=5.6e-15`.
**The orientation claim above is CONFIRMED by measurement, not inverted:** powsybl's CSC arrays read
as CSR are already one transpose, that is the matrix KLU factorizes, so OLF's plain `solve()` IS the
adjoint direction and our single GPU transpose lands on it.

> **Pivoting is a hard prerequisite for any GPU solve on a RAW Jacobian.** 1b failed at 7.164e-03 for
> a week on a misdiagnosis ("stale extraction"). The extraction was always correct — a dense solve
> built from those exact arrays matches `solve()` to 6.2e-15. The real cause: the Jacobian has
> **5 exactly-zero diagonal entries and 0 of 28 diagonally dominant rows**, while `applyReorderAlg`
> forces AMD with no pivoting configured; cuDSS substitutes a static perturbation for the zero pivot
> and returns a wrong `x` **with a matching residual**, which is why it read as a correctness bug.
> Measured: AMD 7.16e-03, nested dissection 7.18e-02, natural order 6.2e-15, **AMD + MC64-style
> matching 3.6e-15** (the fix, `applyAdjointMatching`). Iterative refinement does NOT help (IR=0 and
> IR=3 bit-identical). The batched NR never hit this only because it builds a superset pattern with a
> guaranteed nonzero diagonal — **steps 4–5 must keep that guarantee or set matching explicitly.**
> Natural ordering is not an option at scale (fill explodes on grid patterns).

1. **Single-scenario GPU adjoint, no batching.** Run one network to convergence on the GPU,
   build `Jᵀ` via the permutation, factorize, adjoint-solve, contract to `θ̄`. Compare against
   `runAdjoint` on the same network with the same cotangents, plus an FD check. This isolates
   the highest bug risk — orientation/permutation — from everything else. **Deliberately pick
   a case with NO binding Q-limits**, so any mismatch is the transpose and not the active-set
   question. No batching, no JAX, no context lifetime.
2. **Measure the cost assumption early.** The "+1 factorization ≈ 12–25% overhead" above is an
   ESTIMATE, not a measurement. Time transpose-build + factorize against the forward's existing
   factorizations on a real case. If it is much worse, the sequencing changes — it may be
   better to wait for cuDSS's transpose solve than to build around the workaround. Cheap to
   check, and exactly the kind of assumption that has bitten this project before.
3. **Put in the seam** (one function: "solve with `Jᵀ` given the forward's factors", with
   transpose-and-refactorize as its only implementation) once step 1 proves correctness, so the
   cuDSS upgrade is a local swap rather than a rewrite.
4. **Extend to the uniform batch** — per-scenario `Jᵀ`, batched factorize, batched adjoint
   solve; still the shared-pattern UBATCH, still no JAX.
5. **Prove it through a real consumer: batched SVC or incremental phase control.** Needs no
   JAX, no non-uniform batching, no context-lifetime plumbing, and both are independently on
   the roadmap. If the primitive works here, it is real.

Only then: non-uniform batching (port pyloadflow's pattern bucketing) and the JAX/pypowsybl
context for training.

Verify cheaply alongside step 1: (a) that cuDSS 0.8.0 genuinely lacks a transpose-solve entry
point — confirm the API surface rather than assume; it also establishes the re-check procedure
for each bump; (b) what the CPU adjoint does for distributed slack and the SVC pilot, since it
already handles both (see "Active-set / control handling in the CPU adjoint" below).

### #18 — GPU-vs-CPU LF validation project
Blocked on RTE Nexus access.
**Standing constraint: never read or load `.arc` files or any real network data.**

## Completed 2026-07-21

| # | Task | Evidence |
|---|---|---|
| 38 | Batched solver outer-loop config per-call, not JVM-global static | `aaaaa14` |
| 37 | Pegase full-suite failure (`checked 0`) — root-caused to the config leak | `5177543` |
| 36 | Emit the CSR diagonal for the `BRANCH_ALPHA1` identity row (phase control INCREMENTAL now solves on GPU) | `e8cf120` |
| 35 | Re-time benches on an idle box (the 2026-07-20 row was contention) | `d24e38b` |
| 33 | Batch node-breaker contingencies under propagation | `f7b7226` |
| 24 | Batched operator strategies (scope complete; remainder re-scoped into #30 + upstream-blocked actions) | `856767c` |

## Not tracked, but outstanding

- **Split-bench re-run after the #38 refactor.** Interrupted before it ran. The refactor is
  behaviour-neutral in principle (same values reach the JNI call) and the full suite passes,
  but this bench is what would expose a mis-wired `dsRejectResidual` — a wrong value flips
  scenarios to CPU fallback and tanks the GPU time. Baseline to beat: GPU warm ~23.5 ms/ctg,
  ×1.42, from `7ac0627`. Run with `OLF_GPU_SA_GPU_CONV_EPS=1e-7` on an idle box.

### #41 — Outer-loop ordering & composition in the DEVICE batch path (investigated 2026-09-11)

**Problem.** The GPU DEVICE batch path runs host orchestrators in a sequence that differs from
OLF's `DefaultAcOuterLoopConfig` ordering, AND does not implement OLF's nested stabilization
model (each loop to stability; repeat whole sweep on change). Root cause is architectural: the
in-kernel driver phases (DS, HVDC, Monitoring, RL) complete before host orchestrators run, so
loops that OLF positions BEFORE RL (SecondaryVc order 4, VoltageMonitoring order 5) necessarily
run AFTER RL on the GPU.

**Current GPU host-orchestrator order** (`GpuBatchChunkSolver` lines 1137-1291):
```
[In-kernel: DS → HVDC freeze → HVDC limits → (HOST pause) → Monitoring → RL]
Host orchestrators (sequential, after main solve):
1. IncrementalTvc        ← OLF order 8
2. VoltageMonitoring     ← OLF order 5 (if not in-kernel)
3. SecondaryVc           ← OLF order 4
4. ContinuousPhase       ← OLF order 7
5. SimpleShuntVc         ← OLF order 10
6. SimpleTvc             ← OLF order 8
7. IncrementalFlowControl ← OLF orders 7+9
8. IncrementalShuntVc    ← OLF order 10
9. TransformerControl    ← OLF order 8
```

vs OLF CPU: DS(0) → HVDC(1,2) → SVC(4) → Monitoring(5) → RL(6) → Phase(7) → TVC(8) → TRPC(9) → Shunt(10).

**Three distinct issues:**

1. **Pre-RL loops running post-RL.** SecondaryVc and VoltageMonitoring are pre-RL on CPU but
   post-RL on GPU (because RL is in-kernel). VoltageMonitoring can run in-kernel (already
   implemented). SecondaryVc needs a transposed sensitivity solve and a host-side dense zone
   system — it cannot be a pure kernel. The `LOOP_HOST` pause/resume mechanism
   (`olf_gpu_nr_jni.cu:69-75, 4852-4853, 5396-5407`) was built for exactly this: a host-owned
   loop sitting in its correct OLF position between HVDC_LIMITS and MONITORING, with the driver
   pausing and resuming around it. Currently used only in base LF, not in batched SA.

2. **Post-RL orchestrators out of OLF's relative order.** IncrementalTvc runs before SecondaryVc
   and ContinuousPhase, but on CPU TVC (8) is after Phase (7). All three run after RL on both
   paths, so this is just a sequencing bug — reorder them:
   ```
   ContinuousPhase (7) → IncrementalTvc / SimpleTvc / TransformerControl (8) →
   IncrementalFlowControl/TRPC (9) → IncrementalShuntVc / SimpleShuntVc (10)
   ```

3. **No nested stabilization.** OLF's engine has a double loop: inner (each loop to stability),
   outer (repeat sequence on change). GPU orchestrators run once. This means a TVC tap change
   that destabilizes DS is caught only at the next orchestrator's re-solve (where DS runs
   in-kernel), not by restarting the sequence from DS. The JAVA hybrid mode
   (`GpuHybridOuterLoopBridge`) implements the correct nested model.

**Data flow findings** (who needs what — determines porting approach):

| Orchestrator | Sensitivity solve? | RL state R/W? | Discrete tables? |
|---|---|---|---|
| SecondaryVc | YES | YES | No |
| VoltageMonitoring | No | YES | No |
| IncrementalTvc | YES | No | Tap positions + pi params |
| ContinuousPhase | No | No | Tap positions + pi params |
| IncrementalFlowControl | YES | No | Tap positions + pi params |
| IncrementalShuntVc | YES | No | Section B values |
| SimpleShuntVc | No | No | Section B values |
| TransformerControl | No | No | Tap positions + pi params |
| SimpleTvc | No | No | Tap positions + pi params |

**In-kernel feasibility per loop** (ascending difficulty):

- **VoltageMonitoring**: Already in-kernel (LOOP_MONITORING). Done.
- **ContinuousPhase / TransformerControl / SimpleTvc**: No sensitivity. Need discrete
  tap tables on device (per-branch: list of (r1, x, g, b, a1, ksi, y) per tap position).
  Decision logic (deadband, closest-tap search, ratchet) is simple enough for CUDA.
  Medium difficulty — blocked on uploading the discrete tables.
- **IncrementalTvc / IncrementalShuntVc / IncrementalFlowControl**: Need
  `solveBatchSensitivities` which requires a separate solve step — a transposed solve
  against the factored J. Cannot be folded into the Newton iteration. Must remain as
  host orchestrators or use task #40's batched transposed solve to run in-driver.
- **SecondaryVc**: Needs sensitivity + host-side dense zone system (small matrix LU).
  Must remain host-orchestrated. The `LOOP_HOST` mechanism is the path: pause the driver,
  run SVC on host, resume. Not yet wired for batched SA.
- **SimpleShuntVc**: Rounds continuous B to section — simple, but needs section B tables
  on device. Low-medium difficulty.

**Actionable next steps (ordered):**

A. **Reorder host orchestrators** in `GpuBatchChunkSolver` to match OLF's relative post-RL
   ordering. Low risk, immediate fidelity improvement. Validate with the production SA bench.

B. **Wire `LOOP_HOST` for SecondaryVc in the batched SA.** The mechanism exists in the native
   code; the Java side needs to detect the pause, run the orchestrator, and resume. This puts
   SVC in its correct pre-RL position (between HVDC and Monitoring in the driver sequence).

C. **Upload discrete tap/section tables to device.** Prerequisite for any in-kernel phase or
   transformer control. Once available: ContinuousPhase, SimpleTvc, TransformerControl,
   SimpleShuntVc can become driver phases.

D. **Nested outer-loop sweep.** The extended driver's own nesting (`do { for each loop }
   while(changed)`) provides OLF's repeat-until-stable semantics — no separate wrapper needed.

E. **Base LF on GPU.** Currently the base LF uses `GpuNewtonRaphson.run()` with only DS+RL
   in-kernel. All other outer loops run as OLF Java loops on CPU. Solution: enable the
   `LOOP_HOST_*` slots, enter a `solveBatchContext → readBatchDriverPaused → run OLF loop
   → setBatchDriverResume` cycle. `GpuDriverOuterLoopEngine` suppresses ALL loops from the
   Java engine.

### Sensitivity-dependent loops — kernel port assessment (2026-09-12)

`solveBatchSensitivities` is a FORWARD solve (`J dx = e_i`), NOT transposed. This means
IncrementalTvc and IncrementalShuntVc are NOT blocked on task #40.

| Loop | Solve | Observable | Blocker | Verdict |
|------|-------|-----------|---------|---------|
| IncrementalTvc | Forward ✓ | variable column | Upload tap tables | **LANDED 2026-09-14/15** (`LOOP_TVC_KERNEL`; shared groups too since 2026-09-15 — `tvcGroupDecisionBatch`, OLF's adjustWithSeveralControllers per (scenario, group)) |
| IncrementalShuntVc | Forward ✓ | variable column | Upload section tables | **LANDED 2026-09-15** (`LOOP_SHUNT_KERNEL`, OLF adjustB round-robin per controlled-bus group in-kernel — several controller shunts and sub-controllers per group) |
| IncrementalFlowControl | Forward ✓ | J-row · dx on device | — | **LANDED 2026-09-15** (`LOOP_PHASE_KERNEL` order 7 / `LOOP_TRPC_KERNEL` order 9: the controlled branch's P/Q gradient from `evalClosedBranch`, I from OLF's reI/imI decomposition incl. the I = 0 one-sided rule, dotted with the unit-RHS solution at the branch's six variable columns) |
| SecondaryVc | Forward ✓ (**corrected 2026-09-15**: OLF's `j.solveTransposed` IS the Newton direction — NewtonRaphson.java:62 takes its step with the same call; `GpuBatchSecondaryVc` already solves it with `solveBatchSensitivities`) | branch Q + shunt Q gradients · dx, pilot dx (all on device since the flow-observed port) | none — the observables were kept on OLF's EquationTerms by CHOICE (device-target-writers risk), not by necessity | **Not blocked; deferred** (host-paused slot; SVC off by default, 0 occurrences on rte6515) |

To port IncrementalTvc to kernel: upload per-branch tap model tables (y,g1,g2,b1,b2,ksi,r1,a1
per tap position) + implement deadband/closest-tap/ratchet as a CUDA kernel + the forward
sensitivity solve is already available via the existing factorization.

To port IncrementalShuntVc: upload per-sub-controller section B/G tables + one-section-at-a-time
kernel with magnitude-descending ordering.

IncrementalFlowControl's observable (`calculateSensi`) is `Σ (∂Flow/∂x_j) · (dx_j/dControl)` —
a dot product of the term's Jacobian row entries with the sensitivity solution. BOTH pieces are
already on the device: `∂Flow/∂x_j` = the J entries the fill kernels scatter, `dx_j/dControl` =
the sensitivity solution from `solveBatchSensitivities`. So the observable CAN be computed
on device as a gather-and-dot from J at the flow term's variable columns. Needs: (a) a map from
each flow term to its J column positions (the scatter table already has this), (b) a kernel that
reads J[term_cols] · dx[term_cols] → flow sensitivity. Medium effort, NOT blocked.

### SecondaryVoltageControl — the in-kernel phase (landed 2026-09-15, DEFAULT)

The earlier "blocked on task #40" verdict was WRONG. OLF's `SecondaryVoltageControlOuterLoop` builds
its sensitivities with `j.solveTransposed(rhs)` — the very call `NewtonRaphson.run` uses for the
Newton step (OLF stores the Jacobian variable-major, so "transposed" is the forward system in our
layout). Task #40 is needed by the GNN adjoint (reverse mode) only.

`LOOP_SVC_KERNEL` (OLF order 4) is the whole loop on the device. Nothing is hand-derived: a
controller bus's reactive injection is its CLOSED-BRANCH side flows, whose gradient comes from the
codegen'd `evalClosedBranch` exactly as `flcGatherSensiBatch` takes it, plus its shunts' q = -b*v^2 —
and that one OLF exposes as three static lines (`ShuntCompensatorReactiveFlowEquationTerm`:
dq/dv = -2bv, dq/db = -v^2). The pilot's calculated V is the BUS_V variable itself, so its
sensitivity is dx at the pilot row. The pieces, in `check()`'s order:

- `svcTriggerBatch` — `tryToReEnableHelpfulControllerBuses` first (a zone with SOME but not all
  controllers at a reactive limit gives the limited ones back their control, restoring the PV form
  from `dRlSavedMode`), then OLF's zone filter (a zone with NO enabled controller is not looked at
  at all), then the pilot-deadband / K-spread trigger with K from the pass's own `qBuf`.
- one unit-RHS solve per controlled bus on the driver's own factors, then `svcGatherBatch` for
  `calculateSensiK` and `calculateSensiVpp`.
- `svcAssembleBatch` + **cuBLAS batched dense LU** (`cublasDgetrfBatched` / `cublasDgetrsBatched`) +
  `svcCommitBatch` — A (per-zone block), K0, `b = A.Jk^T`, the zone's LAST controlled bus's row
  replaced by the pilot sensitivities and its rhs by the pilot dv. NO hand-written linear algebra: the
  systems are assembled column-major into device memory and handed to the library, one matrix per
  scenario. A controller at a reactive limit is padded out with an identity row and a zero rhs, which
  leaves the enabled block exactly OLF's system and gives the disabled one dv = 0. A singular system
  (cuBLAS info != 0) or an implausible target rejects the scenario to the CPU, where OLF's own
  MatrixException / FAILED verdict would land it.
- `svcApplyBatch` — the moved target on an unpinned controlled bus's V identity, and the switch-back
  comparison target for a pinned one (`dSvcRlVt`, which `rlTargetOf` consults).

Two layers remain: the in-kernel phase, and the driver PAUSE at the same slot where the kernel
refuses the tables (an observable on an unpacked element, or a zone system whose per-scenario
workspace exceeds `SVC_MAT_BUDGET_BYTES`).

**Sizing, for real networks.** OLF solves ONE system over every enabled controller of every zone —
A is block diagonal by zone but Jk is full, since a controller in one zone moves a controlled bus in
another through the network — so N is the TOTAL controller count, not a per-zone one. Production
cases are quoted at 30 to 60 zones of 4 to 20 controllers, i.e. N = 120 to 1200. Two consequences
the original 16-controller cap hid: the batched LU holds an N*N matrix per scenario, so the workspace
is scenarios * N^2 * 8 bytes (58 MB at N=120 with 500-scenario chunks, 5.8 GB at N=1200 — hence the
budget, and chunk sizing will have to follow N); and Jk needs one unit-RHS solve per controlled bus,
so N solves per scenario per check, which is inherent (the matrix is genuinely N functions by N
parameters, so neither forward nor adjoint avoids it) and is what will dominate at the top of that
range. NOT YET VALIDATED above N = 4: rte6515 carries two zones of two.

**The bug that cost the most, and the wrong turn it caused.** The phase first landed 1.04e-10 kV off
on the SecondaryVc unit tests. `OLF_GPU_SVC_INERT=1` (slot in the sweep, body empty) reproduced the
shift exactly, and I read that as "an extra sweep slot perturbs the trajectory" and wrote it up as
such. It was not. The inert probe isolates everything `nSvc > 0` enables, and that includes the new
`dSvcRlVt` buffer — which `initRlLimitTypeBatch` consults through `rlTargetOf` and which was
initialised AFTER that seeding. The reactive-limit switch-back was comparing against uninitialised
memory: ieee14's z1 went 4 pins -> 2 (un-pinning two controllers) where it should go 4 -> 6. Filling
it with NaN at upload AND before the seeding fixes it, and all three SecondaryVc tests pass at 1e-11.
LESSON: an inert-slot probe proves "something enabled by this table", NOT "the slot itself"; enumerate
what else that flag switches on before naming a cause.

**Every OLF outer loop is now a device phase.** rte6515, 9042 branch contingencies, both zones
firing (`OLF_GPU_SVC=1 OLF_GPU_SVC_PILOT_SHIFT=1.0`), all 4 differing violations identical. Cold and
warm are two identical SA runs in the same JVM, so COLD carries the CUDA context, the cuDSS analysis
and JIT; WARM is the fair comparator when asking what a LOOP costs:

| chunk path for SVC | cold | warm |
|---|---|---|
| host orchestrator (old) | 208 s | 172 s |
| driver pause | 150 s | 142 s |
| in-kernel | 119 s | 91 s |
| in-kernel + base load flow on the driver | 83 s | 85 s |
| (same case with SVC switched off) | 80 s | 83 s |

So the port is worth 172 s -> 91 s of steady-state time, and SecondaryVoltageControl now costs about
3% over the SVC-off baseline (85 vs 83 s) where the host orchestrator cost more than double (172 vs
83 s). **Do NOT read the 119 -> 83 cold pair as the driver flag's doing**: that run's cold carried a
28.7 s first-run penalty (119 cold vs 91 warm) which the other run did not (83 cold vs 85 warm), and
on warm terms the flag is worth about 6%. Calibration for any single-run comparison here: the CPU arm
does identical work in every one of these runs and ranged 96 s to 109 s, so differences under ~10%
are not attributable.

### #42 — Base load flow on the FULL device driver + in-kernel incremental TVC (landed 2026-09-14)

**Outcome.** With `OLF_GPU_DRIVER_OUTER_LOOPS=1` the SA's base load flow is a batch-of-one on the SAME
driver as the contingency chunks: DistributedSlack, HVDC freeze/limits, MonitoringVoltage,
ReactiveLimits (and incremental TVC in-kernel when every controlled bus has a single controller) —
`GpuDriverOuterLoopEngine` keeps NOTHING on the Java engine (`GPU_DRIVER_ENGINE ... keptOnHost=[]`).
rte6515 / RTE params: base CONVERGED under OLF's own criterion (inf-norm 9.4e-6), 9040/9042 on device.

**What actually blocked it** (the "write-back" was only half of it): the base context received
`createBatchContext` + row criteria and nothing else, while the chunk solver also installs the RL
curve channel, the SVC quadratic limits, the remote-member RL kinds (`setBatchRlRemote`) and the
per-generator DS participants. Without the remote kinds `initRlLimitTypeBatch` read every remote
member's DISTR_Q share row (base mode 1) as a base PQ pin — limit type seeded, never re-checked —
and three rte6515 remote controllers came out pinned where OLF's loop never touched them
(BUS_TARGET_Q residuals of 1.46 pu). `GpuBatchOuterLoopTables.install` is now the ONE installer for
both paths; `GpuBatchHvdcAcEmulation.installDriverLoops` / `GpuBatchVoltageMonitoring.installDriverLoop`
likewise. A batch of one drives exactly what a batch of five hundred drives.

**Write-back** (`GpuNewtonRaphson.writeBackDriverMutations`, order matters): HVDC statuses through
`AcEmulationControl.switchTo*`; monitoring activations (VC on, SVC → VOLTAGE, automaton target);
DistributedSlack per GENERATOR from the water-fill's own `genCum` (`readBatchDsGenCum`), the slack
bus's generators splitting the retained share (`readBatchDsRetain`) by factor; ReactiveLimits pins
via `freezeGenerationTargetQAndDisableGeneratorVoltageControl` at the DEVICE's pinned target (the live
SVC limit — the packed base-V aggregate was 1.3e-4 pu off on VL-6226_0); taps (`readBatchTvcTaps`).
The extraction uses the run's own loops and DS params (`DistributedSlackOuterLoop.
getActivePowerDistribution()` — OLF getter added), never `GpuDsParams.DEFAULT`.

**Measured 2026-09-14 (rte6515, 9042 branch contingencies, `GpuVsCpuRte6515SaTest`):** driver ON
80.8 s cold (×1.20 vs CPU 8 threads 97 s), driver OFF 82.1 s — the base on the driver costs nothing.
Both arms' base violations agree to 1e-11 (one value differs in the 10th decimal). The 2 differing
post-contingency lines are the test's own documented residual ("2 out of 583k", identical with the
driver off): both sit exactly at the SA's 0.1 % increased-violation proportional threshold
(post/base = 1.00100 on LINE-2533-1468 under LINE-1550-228 and LINE-2219-439 under LINE-5466-2443).

**Three base-on-driver corners found and fixed the same day** (all invisible from the chunk path,
which extracts AFTER a converged base): (1) `rlV2Base` is NaN on a never-solved network and
`0 × NaN` poisoned every RL entry's limit — the quadratic SVC delta is now applied to SVC entries
only (`svcLim`); (2) a SINGLETON remote group has members but no DISTR_Q row, and
`setBatchRlRemote` keyed `hasDqQrGroups` on the DISTR_Q rows, so the pilot's control-loss flip
never armed — keyed on the member list now; (3) a device REJECTION (status −2) on the base is a CPU fallback like on the chunks — and since
2026-09-15 the forced-PV case no longer rejects at all: OLF's "strongest stays PV" election runs
in-kernel (`rlForcedPvElectBatch`, keys via `setBatchRlElection`), so the forced-PV network solves
on the device in the chunks AND as the base; `GpuDriverOuterLoopEngine`
now wraps every device-owned loop in `DeviceDrivenLoop`: STABLE while the solver ran on the device,
OLF's real loop (initialized late) the moment a solve fell back to the CPU — dropping the loops left
the fallback base without DS/RL (1.3e-4 kV on the zero-impedance member network).

**2026-09-15 — host-paused loops on the base, second pass.** A host loop that changes the equation
LAYOUT, the row modes or the element values (a tap/section move, a control it enables or disables:
every TVC/shunt/phase loop does) voids the paused sequence — the resume channel carries targets
only. The driver now RESTARTS: the driver's own decisions are written back to the network first,
the pause is cancelled, the extraction re-runs, the RL state is carried by bus, OLF's loop contexts
persist across the restart (`hostLoopPairs`), and the restarted sweep resumes AT the loop that went
UNSTABLE (`setBatchDriverStartLoop`) before restarting from the first loop — OLF's `runOuterLoop`
order. Two other fixes on the way: the driver owns loops only when `GpuDriverOuterLoopEngine` created
the solver (`AcloadFlowEngine.createSolver` hook; a plain `LoadFlow.run` keeps OLF's engine and the
legacy pure-Newton mode, or DS/RL ran twice), and `DeviceDrivenLoop` defers to OLF's loop whenever
the engine's current solver is not the GPU driver (CPU-routed contingencies had lost DS/RL). The
per-generator DS accumulator is zeroed per solve (a restarted sweep that never reaches DS handed
garbage to the write-back → NaN targetP).

**Same day, fourth pass — 39 → 0: `GpuBatchedAcSecurityAnalysisTest` 136/136 under the flag.** The
"DS re-anchoring class" was exactly that, and it was the base integration, not the loop: OLF's
`ActivePowerDistribution.run` resets every participant to its INITIAL target P before each
distribution (`resetToInitialState`, adding back what that undid as `previousMismatch`) and
re-derives everything — factors, sign rule, limits — from there. The device kernel has done that
since 2026-09-08 (`absorbedNow`), but the base path fed it the batched-SA extraction, whose contract
anchors at the row targets the solve starts from (right for the SA: converged base, per-scenario
channels relative to it), and every restart re-anchored at the written-back targets. Now
`GpuDsParams.anchorOnInitialTargetP` (base path only): the pack anchors the row targets, the
per-generator limits, the sign rule and the P-proportional factors at `initialTargetP`
(`anchorOffset` = Σ(tp − tp₀) over the participants — `absorbedNow` then IS OLF's
`previousMismatch`); the write-back sets `tp = tp₀ + genCum` and, for the slack bus's own
generators, `tp₀ + retain` with the retain captured at the last check that FIRED
(`dDsRetainApplied`, NaN when none did — a STABLE check's ρ·mismatch is not a distribution);
slack-only bases write back `readBatchSlackAbsorbed`. Two more: a restart starts from the network's
last state through `PreviousValueVoltageInitializer` exactly as `runOuterLoop` does (the old state
vector was not even the right length after a control was disabled); and the driver's decisions so
far (P, RL pins, HVDC statuses, activations) are handed to the network BEFORE each host-loop check
— SecondaryVoltageControl skips a zone's RL-pinned controllers, and a check that still saw them
enabled went UNSTABLE where OLF's said STABLE (one extra adjust round, 2.5e-8 kV). The SA contract
is untouched: with static factors, anchoring at the converged base is linearly equivalent
(caps are absolute, the sign never flips); PROPORTIONAL_TO_GENERATION_P (|tp₀| vs |tp|) and
REMAINING_MARGIN margins are the only balance types where it is not — latent for the SA, noted.

**Same day, third pass — 44 → 39, no ENGAGE and nothing above 1e-3 kV left.** Five more base-only
defects: (1) a converter bus without an active BUS_TARGET_P row (the slack) received the HVDC
freeze/saturation delta on its PHI identity row (base MAX_ITER, 3 ENGAGE) — packed as ~row now, the
angle still reads there, no target is ever written; (2) after a host loop's STABLE resume the RL
check was silently skipped (the resumed call's status vector said "not converged" → rlActive 0):
the pause saves and the resume restores the per-scenario statuses (SecondaryVoltageControl
7.3e-2 → 2.5e-8); (3) an SVC's limits are Bmin/Bmax·V² and a never-solved network has NaN V, so
the packed limits were NaN and the slope SVC never pinned (0.33 kV): the pack evaluates at a flat
1 pu and restores the NaN; (4) the slack's retained DS share was written back at every check,
STABLE ones included (ρ·sub-tolerance mismatch, 3.4 kW), now only when a distribution happened;
(5) the split of that share used LfGenerator.getParticipationFactor(), which OLF fills only inside
its own distribution step (0 here) — the run's GpuDsParams factors, as the pack uses (l121 severe
0.71 → 4.2e-4). Remaining 39: 17 at ≤1e-9 kV, 7 at ≤1e-6 kV, 15 between 2e-6 and 4e-4 kV
(AFTER_GENERATOR/shared TVC, TRPC, node-breaker, l121) — the DS re-anchoring class: OLF resets
every participant to its INITIAL target and re-derives the whole distribution from the anchor at
each check, the restarted driver distributes from the written-back base; and the driver re-enters
the sweep once more than OLF after a STABLE host check (checks only, no solve).

Result (second pass): 61 → 44 failing under the flag. By size: 13 at ≤1e-9 kV and 7 at ≤1e-6 kV — the base's
converged state differs from OLF's at solver-tolerance level (a different linear solver), and
AFTER_GENERATOR transformer control FREEZES state-dependent quantities (generator Q under the
controlled level, the tap rounded from the solved ratio), so the frozen targets differ by ~1e-5 pu;
inherent to a base on another solver, not a sequence bug (sequences now match call for call).
Real remaining: 3 HVDC ENGAGE (freeze/saturation/DS carry), 10 medium (shunt VC 9.6e-4, shared
TVC 4e-4, ...), 11 large (SecondaryVoltageControl 7.3e-2 ×4, slope-SVC pinned 0.33, l121 severe
0.71, TRPC 1.7e-3, shared incremental TVC 1.2e-3).

**Earlier inventory (2026-09-14):** `GpuBatchedAcSecurityAnalysisTest` with
`OLF_GPU_DRIVER_OUTER_LOOPS=1` = 61/136 failing (default path: all green). By shape: 16 "batch must
ENGAGE" (the base's write-back leaves the network in a state the plan or the JAVA guards reject —
continuous PhaseControl after the host-paused loop, shunt/TVC/SVC host loops), 12 BUS_2 voltage
mismatches (transformer/shunt voltage-control networks — the host-paused TVC/shunt slots on the
base), 3 strategy-pair precompute counts, 3 HVDC (freeze/saturation, DS carry), REMAINING_MARGIN DS.
Every remote gen-VC test (24/26) passes; the rte6515 production case passes. Next: take the failing
shapes one by one with `OLF_GPU_BATCH_TRACE=1` (HOST_LOOP lines show the pause/resume sequence),
starting with the host-paused loops (order 7–10) on the base.

### #45 — Flow-observed incremental loops in-kernel: RPC + incremental phase (landed 2026-09-15)

`LOOP_PHASE_KERNEL` (OLF order 7, phase CONTROLLER + LIMITER) and `LOOP_TRPC_KERNEL` (order 9, RPC)
share one controller table and one step (`flcStep(kindMask)`). Per (scenario, controller):
`flcCheckBatch` evaluates the CONTROLLED branch's flow at the current state — P/Q through the
codegen'd `evalClosedBranch` (the very functions the fill scatters, with the branch's tap override
applied), I through OLF's `ClosedBranchSide{1,2}CurrentMagnitudeEquationTerm` decomposition
(reI/imI and their eight partials) — and OLF's deadband test (LIMITER one-sided). One unit-RHS
solve per needed controller at its rho/a1 identity row on the driver's own factors;
`flcGatherSensiBatch` dots the flow gradient with dx at the branch's six variable columns — this IS
`EquationTerm.calculateSensi`, including its dr1-less current sensitivity and its `hypot(dre·dx,
dim·dx)` one-sided rule at I = 0. `flcDecisionBatch`: RPC → closest r1 within the run's
maxTapShift (no gate); CONTROLLER → `|sensi·toRadians(1)| > SENSI_EPS`, closest a1 unbounded;
LIMITER → `FirstTapPositionAboveFinder` (walk tap by tap while the remaining shift reaches past the
next a1) — ratchet as everywhere. `flcApplyBatch` pins the identity row; the r1 override rebuild is
shared with the TVC phase (`tvcRebuildOverrides` now merges both tables' moved taps). Base path:
`getMaxTapShift()` (OLF getter) for RPC, `readBatchFlowTaps` → `PiModel.setTapPosition`. The host
orchestrator (`GpuBatchIncrementalFlowControl.orchestrate`) stays for a controlled branch that is
not a closed branch. With this every sensitivity-dependent incremental loop runs in-kernel; the
chunk path's host orchestrators left: SecondaryVoltageControl (transposed solve, task #40); the
non-incremental TVC / shunt VC flows followed the same day (#47).

### #53 — fixTransformerVoltageControls is a per-SCENARIO check (2026-09-16)

The last of the four gaps in the ported phases: the freeze approximation. The generator freeze itself
was already OLF's (`GeneratorVoltageControlManager` frozen at `q.eval()`, restored at COMPLETE); what
was approximate was the connectivity check that runs beside it.

`LfNetwork.fixTransformerVoltageControls` disables a just-enabled transformer voltage control when the
not-controlled side's connected component holds no enabled generator voltage control. It computes that
component on the POST-CONTINGENCY graph, cutting the contingency's branches AND the ENABLED controller
branches. The device built ONE table, at base topology, cutting EVERY controller branch, and per
scenario asked only which of those buses was still PV. Two errors in opposite directions: a contingency
that split the component was invisible, and cutting non-enabled controller branches made the component
smaller than OLF's.

The dynamic half stays in the kernel — which buses are PV depends on the freeze and on the reactive
limits, and only the kernel knows that. The topological half moves to the host, per scenario. The
obstacle is that OLF's cut set is not host-knowable: the contingency is, but the ENABLED controllers
are decided inside the solve. So the host brackets it with two masks over a fixed universe (the
{@code BUS_V} row of every generator-voltage-controlled bus, one bit each):

| mask | cut set | relation to OLF's component |
| --- | --- | --- |
| min | contingency + EVERY controller | subset |
| max | contingency only | superset |

A PV bus in min means OLF sees one too, so the control stays enabled. No PV bus in max means OLF sees
none either, so it is disabled. Between the two the answer genuinely depends on which controllers the
scenario enabled, and the scenario goes to the CPU rather than to a guess. The freeze-driven cases —
where the check fires because everything under the level is frozen, which is what the mode is for —
land outside the bracket and still batch; it is the connectivity-driven ones that are ambiguous.

Cost: two connectivity sessions per scenario, each O(PV buses + members) — the component NUMBER of
every PV bus is read once and members look up their own component's mask, instead of walking a
component per member. Masks upload per chunk, not per run.

`GpuBatchTvcFixMaskTest` pins the builder directly rather than through a security analysis, where the
same scenario can be routed to the CPU for unrelated reasons and hide the difference: BUS_2 reaches the
only generator over LINE_12 alone, a 33 kV loop through a non-regulating transformer keeps the pocket
energized when that line opens, and the base mask and the scenario's differ exactly as designed.

### #52 — Remote LIMITER phase control, and the tap table without reflection (2026-09-16)

Two of the four gaps left in the ported phases. Both turned out to be self-imposed.

**Remote LIMITER phase control.** A CONTINUOUS phase control in LIMITER mode whose regulation
terminal sits on another branch routed the whole run to the CPU, in DEVICE and JAVA alike, on the
reading that "OLF's own loop is local-only". It is local BY CONSTRUCTION, not by restriction:
`PhaseControlOuterLoop.changeTapPositions` caps `controllerBranch.getI1()`/`getI2()` against the
target and `isSensitivityCurrentPerA1Positive` differentiates those same terms, so the phase
control's controlled branch never enters the decision at all. LIMITER mode is also never enabled as
an equation control — `initialize` enables CONTROLLER mode only — so there is no row-mode or
`cbPcSide` question either. The device already built its I term from the controller branch, so the
gate was refusing a computation it performs identically. Lifted in all three places; both modes now
batch local and remote. New tests build PS1 in CURRENT_LIMITER mode regulating the parallel line L2,
DEVICE and JAVA, both at 1e-12 with the batch engaged.

**Transformer control without a tap table.** `tapModels` read `PiModelArray.models` by REFLECTION and
returned an empty table on any failure, which meant a JVM configuration could silently turn into
unrounded continuous ratios and into whole-run CPU fallbacks (`generatorFreezeSupported`) that no
gate would explain. `PiModelArray.getModels()` already existed, package-private; it is public on the
patched OLF now (`olf-local-patches`), and the reflection is gone. An empty table means one thing
after this: the branch carries a continuous `SimplePiModel` and has no taps, which a voltage or phase
controller never does.

### #51 — Gaps inside the ported phases: robust mode, forced-PV, distributed slack (2026-09-16)

Three of the gaps left inside the phases that are already on the device. All three were
verified against the current code first — two of the three turned out not to be what the
earlier summary said they were.

**(a) Reactive limits: `voltageRemoteControlRobustMode`.** OLF's default is TRUE, so these
criteria were active in every default run and the RL kernel did not implement any of them —
a silent divergence, not a fallback. `ReactiveLimitsOuterLoop.checkControllerBus` applies two
things to a bus whose generator voltage control is REMOTE (`controlledBus != bus`):

  1. a bus that just switched PV -> PQ on a reactive limit, and whose OWN voltage is outside
     `[minRealisticVoltage * 1.02, maxRealisticVoltage / 1.02]`, is reseeded at 1 pu;
  2. a bus whose Q is still INSIDE its limits but whose own voltage is already outside that
     band stops maintaining the remote voltage: it is frozen at its INITIAL generation target
     Q (the sum of its generators' `targetQ`) with a `MIN_REALISTIC_V` / `MAX_REALISTIC_V`
     limit type, and is likewise reseeded at 1 pu.

Ported into the RL phase. The freeze is the existing remote pin (row mode 4) with a different
target, so only three things are new: a per-entry `rlInitTargetQ` table, limit types 3 and 4
alongside 1 (MIN_Q) and 2 (MAX_Q), and the 1 pu reseed. Two details matter. The switch-back
reads 3 like a min limit and 4 like a max limit (`LfBus.QLimitType.isMinLimit()`), but the
"Q limit moved while pinned" re-freeze is guarded to MIN_Q/MAX_Q in OLF and so skips 3/4. And
the reseed is a WRITE to the state vector the check kernel is reading, so it runs as its own
pass (`rlRobustVResetBatch`) against the pre-pass state rather than racing inside the check.

The packed per-(controller, scenario) RL state widened from two limit-type bits to three
(`(limitType & 7) | (switchCount << 3)`), on both sides of the JNI.

The other half of that gap was the realistic-voltage check. `AcloadFlowEngine.isStateUnrealistic`
runs after every solver run once the last `canFixUnrealisticState` loop has run, and an unrealistic
state ends the outer-loop cycle with `UNREALISTIC_STATE`: the run FAILS and reports no violations.
The device only ever checked the FINAL restored state, so a scenario that passed through an
unrealistic state and recovered came back CONVERGED where OLF reports FAILED. Which solve the check
belongs to depends on the configured loop order and on robust mode, so rather than reproduce that
ordering, the device flags a scenario that is unrealistic inside the sweep and routes it to the CPU
replay, which decides the status itself.

The ordering could NOT be skipped, and measuring said so. Checking after every solve cost four
rte6515 contingencies (9040 -> 9036 batched), and naming them showed every hit landing at the FIRST
solve, before any outer loop had run — exactly where OLF defers. Under robust mode
`AcloadFlowEngine` keeps `checkUnrealisticStates` false up to and including the last loop that
declares `canFixUnrealisticState()` (ReactiveLimits, or the discrete TransformerVoltageControl),
turns it on for the rest of the sweep, and resets it at the top of the next one — the whole point
being to let those loops repair the state instead of failing the run on it. The device now takes
that loop's identity as a hint from the Java side (−1 none, 0 ReactiveLimits, 1 the discrete
TransformerVoltageControl), finds its slot in its own loop order and checks only past it. rte6515 is
back to 9040 of 9042 with no sweep rejections at all, same two differing lines.
`OLF_GPU_UNREALISTIC_TRACE` names the slot, the pass and the loop that rejected, which is what made
the four visible. The temporary switch that turned the intermediate check off is gone with them: the
deferred check is what OLF does, so a switch selecting the other behaviour would only preserve a known
wrong answer — the same reason `olf.gpu.sa.hvdcLoopsInKernel` and `olf.gpu.sa.batchStrandedDs` were
deleted rather than defaulted.

**(b) Forced-PV: the per-scenario rejection detector.** The stated gap was that the in-kernel
election had landed but a silent per-scenario rejection to the CPU remained as a safety net.
The first reading — that the detector was dead code because the election keys were always
installed — was wrong, and the contract check that replaced it failed five tests. The keys were
installed by the driver's table install, and the RL phase runs on ANY context carrying RL
entries, including a full load flow where OLF (not the device) owns the outer loops. Those
contexts had no keys, so the detector was live there and every such scenario went to the CPU.

Fixed at the root: the keys are now installed where the context is CREATED (all four creators,
including the one-shot `solveBatch` entry, which has no network behind it and so gets neutral
keys that reduce the rule to OLF's last tie-break, lowest id stays PV). The detector is deleted
and `solveBatchContext` refuses a context that reaches it with RL entries and no keys.

**(c) Distributed slack.** Re-verified against the current code. `dsRejectResidual` is ON by
default (OLF's `slackDistributionFailureBehavior` default is FAIL, not LEAVE_ON_SLACK_BUS), and
it flags a saturated distribution on the ENTRY band (`fsGlobal * slackBusPMaxMismatch`) where OLF
decided the distribution failed on `ActivePowerDistribution.P_RESIDUE_EPS` (1e-5 pu, 1 kW). A
scenario that saturated with, say, 0.3 MW still unplaced therefore came back CONVERGED from the
device where the CPU reports FAILED. That was the one live DIVERGENCE among the slack gaps, and
it now flags on P_RESIDUE_EPS. The flag only routes the scenario to the CPU fallback, so the
tighter test can cost batching share and can never make the device answer where OLF refuses to;
on rte6515 it costs nothing, the batched share stays at 9040 of 9042.

The rest are fallbacks rather than divergences, with what each one needs:

- **`slackDistributionFailureBehavior` on-device.** FAIL (the default) could be answered
  in-kernel — a FAILED post-contingency result reports no violations — instead of replaying the
  scenario on the CPU. What stops it being a pure win is `distributedActivePower`: OLF reverts
  its reporting on a FAILED distribution, and the batched result carries that field.
  DISTRIBUTE_ON_REFERENCE_GENERATOR needs the reference generator's row; THROW must reach the
  host for its message.
- **Strategy-pair legs take the aggregate water-fill.** A re-anchored scenario (`dsAnchorShift`
  non-zero) skips the per-GENERATOR water-fill, because the anchor shift is a BUS-level
  displacement and nothing in the pack says how to split it across a bus's generators. Closing
  it means capturing the parent's per-generator cumulative deltas (`genCum`, already per
  (scenario, generator) on the device) and feeding them back as a per-generator anchor shift.
- **`loadPowerFactorConstant` + load-based balance, whole-run CPU.** `ensurePowerFactorConstant`
  moves each load's targetQ with its targetP, per round. The DS pack is per BUS with a
  per-generator sub-list and has no per-LOAD granularity, and the device never writes a Q target
  during distribution. Needs a per-load pack (Q row, participation factor, initial P and Q, the
  `ensurePowerFactorConstantByLoad` flag) plus a Q-target write in the water-fill.
- **`useLoadModel` + load-based balance, whole-run CPU.** Rescaling a load's targetP rescales its
  voltage-dependent coefficients, which are static on the device.
- **Custom `ContingencyActivePowerLossDistribution` SPI, whole-run CPU.** Only the "Default" impl
  is replayed host-side; an arbitrary SPI cannot be.

### #50 — PLAN: fold the JAVA hybrid into the driver's host-slot mechanism (reviewed 2026-09-16)

Not started. A review, not a design commitment — the measurement in stage 0 decides whether it is worth
doing at all.

**Why now.** The hybrid (`GpuHybridBatchOuterLoopEngine`, 1484 lines, plus `GpuHybridOuterLoopBridge`
270 and `HybridScenarioState` 85) exists because OLF's own Java loops make decisions no port can get
subtly wrong. It pays for that with a per-scenario, per-round save/restore of the compact network
state, which measured **54% snapshots / 44% Java loops / 2% device**. It is also an all-or-nothing
MODE: choosing JAVA replays EVERY loop per scenario in Java, even the ones the device now owns.

Since 2026-09-15/16 that trade is obsolete. Every loop in OLF's order is a device phase except two,
and the driver already knows how to stop at a named loop's OLF position, hand control to Java, and
resume with its sweep state intact (`LOOP_HOST_*`, `readBatchDriverPaused` / `setBatchDriverResume`,
flat per-scenario `setBatchTargetDeltas`, the RL v-target and un-pin channels). SecondaryVoltageControl
used exactly that before its kernel phase landed, at chunk scale, and it held parity.

OLF's `DefaultAcOuterLoopConfig` order, with what owns each slot today:

| order | loop | owner |
|---|---|---|
| 0 | DistributedSlack | kernel |
| 1 | FreezingHvdcACEmulation | kernel |
| 2 | AcHvdcAcEmulationLimits | kernel |
| **3** | **AreaInterchangeControl** | **nobody — whole-run CPU** |
| 4 | SecondaryVoltageControl | kernel (pause as fallback) |
| 5 | MonitoringVoltage | kernel |
| 6 | ReactiveLimits | kernel |
| 7 | PhaseControl / Incremental | kernel |
| 8 | TransformerVoltageControl, all 3 flavors | kernel |
| 9 | TransformerReactivePowerControl | kernel |
| 10 | ShuntVoltageControl, both modes | kernel |
| **11** | **AutomationSystem** | **nobody — whole-run CPU** |

So the hybrid's remaining reason to exist is two loops, and the proposal is to serve them with a host
SLOT at their own position instead of a whole parallel engine.

**What it would buy.** The per-scenario snapshot/restore disappears for every loop the device owns,
which is where most of that 54% goes. The duplicated outer-loop sequencer goes too: the bridge
re-implements OLF's nested `runOuterLoop` so it can drive one scenario at a time, while the device
sweep already sequences in OLF's order with OLF's restart-on-UNSTABLE semantics — one sequencer
instead of two. And the two coverage gates (`JAVA_SUPPORTED_LOOPS` versus the DEVICE filter) collapse
into one question: does the kernel own this loop, or does it have a host slot.

**Stages.**

0. **Measure first.** Re-run the hybrid's cost split now that every loop is in-kernel, and confirm the
   claim above. If a real case never enables AreaInterchangeControl or AutomationSystem, the honest
   answer is to delete the hybrid outright and route those two loops to the CPU, which is cheaper than
   any of the work below.
1. **Generalise the slot mechanism.** `hostLoopMask` is five bits keyed to `HOST_LOOP_SVC..SHUNT`.
   Make it a registry of (OLF order, loop name, kind) so a new host-owned loop is a table entry rather
   than a new constant in three files. Add `LOOP_HOST_AIC` at 3 and `LOOP_HOST_AUTOMATION` at 11.
2. **AreaInterchangeControl first.** It adjusts generator targetP to meet area interchange targets, so
   like DistributedSlack and SecondaryVoltageControl its mutations should be TARGET-only, which the
   existing `setBatchTargetDeltas` channel already carries. That makes it a near-copy of the SVC pause,
   including the budget rule (a host-UNSTABLE verdict must consume `prodIters`, or the pause cycle can
   spin — see #49). CONFIRM the target-only claim against OLF before building; its interaction with
   DistributedSlack at order 0 is the part to read carefully.
3. **AutomationSystem is the hard one and should be scoped separately.** It switches topology per
   round, so it changes the equation LAYOUT, not just targets. The base path handles that by
   re-extracting and restarting the sweep, which does not generalise to a 500-scenario chunk where one
   scenario changed layout. Two candidate answers: reject the scenarios it actually fires on to the
   CPU (cheap and honest if it fires rarely), or express the switching through the per-scenario
   element-disable and zero-impedance on/off channels that already exist. Decide with a measurement of
   how often it fires, not a priori.
4. **Retire the hybrid.** Once both loops have slots, delete `GpuHybridBatchOuterLoopEngine`, the
   bridge, `HybridScenarioState`, the `OuterLoopMode` enum and `javaUnsupportedOuterLoops`. That is
   ~1,800 lines and one of the two coverage gates.

**Risks to weigh before starting.**

- A paused Java loop still reads the `LfNetwork`, not just the state vector, so running it for one
  scenario still needs that scenario's network state. The slot NARROWS this to the loops and scenarios
  that need it; it does not remove it. If the unported loop fires on most scenarios, the snapshot cost
  comes straight back and the hybrid's structure was not the problem.
- The per-scenario Java work at a pause is serial. With 500-scenario chunks that is the thing to
  measure in stage 0, not to assume.
- Two host slots means two places the sweep can stop per pass. The SVC experience says an added slot is
  cheap, but it also says a new context buffer read by an existing kernel must be initialised at UPLOAD
  time, not only in the per-solve reset (#49's `dSvcRlVt` bug).

### #49 — Host orchestrators deleted; the device driver is the default (2026-09-16)

Every decision loop that ran AFTER the device sweep is gone: `GpuBatchSecondaryVc.orchestrate`,
`GpuBatchVoltageMonitoring.orchestrate`, `GpuBatchContinuousPhase.orchestrate`,
`GpuBatchIncrementalTvc/ShuntVc/FlowControl.orchestrate`, `GpuBatchSimpleTvc` and
`GpuBatchSimpleShuntVc.roundAndResolve`, and `GpuBatchTransformerControl.
orchestrateTransformerControl` — with the helpers that only served them (the clamp/round machine, the
per-step delta builder, the tap-row emitter, the per-loop ratchet movers, and the DS capture/seed
carry triple). About 1,500 lines.

They were never ports of OLF's loops, they were COMPENSATIONS for running outside the sweep: an
orchestrator re-solves after everything, so it needed the DS carry triple to undo its own re-upload of
base targets, and two of them in a chain discard each other's target moves (measured 2.4e-1 kV on
SecondaryVc + AFTER_GENERATOR TVC). Their correctness was conditional on being the only decision loop
in the run.

A loop the kernel REFUSES now routes the whole group to the CPU (`refusedLoops`, leaving
`resultsByIndex` null, which is already how the caller sends a contingency to the stock path).
Un-accelerated is the only honest alternative to a different algorithm. The one remaining fallback is
SecondaryVoltageControl's driver PAUSE at its own OLF slot — that is IN OLF's order, so it is not an
orchestrator.

**The device driver is also the DEFAULT for the base load flow** (`driverOuterLoopsEnabled`, formerly
`OLF_GPU_DRIVER_OUTER_LOOPS=1`). A batch of one is still a batch, and running the base through the
same driver as the chunks is what keeps the two on one trajectory. rte6515 parity is identical either
way (9040/9042, the same 2 knife-edge lines) and the driver is never slower.
`OLF_GPU_DRIVER_OUTER_LOOPS=0` or `-Dolf.gpu.driverOuterLoops=false` puts OLF's Java engine back in
charge of the base.

### #48 — Obsolete gates lifted, and the SecondaryVoltageControl ORDER gate (2026-09-15)

Three gates written for the host-orchestrator world were re-examined once every control loop became a
driver phase.

**Lifted.** (a) INCREMENTAL and WITH_GENERATOR transformer voltage control no longer require
`transformerControlSupported`: neither loop has a GeneratorVoltageControlManager, so the
generator-under-the-controlled-level condition never applied to them — it was a blanket copied from
AFTER_GENERATOR (`deviceIncremental/SimpleTvcGeneratorUnderLevelMatchesCpu`, OLF's four-bus network
with 90 kV generators under a 230 kV controlled bus). (b) The WITH_GENERATOR shunt VC co-presence set
shrank from seven loop names to nothing: it existed because the host orchestrator CHAIN could not
model OLF interleaving two simultaneously-unstable loops, and the driver sweeps in OLF's order with
OLF's restart-on-UNSTABLE semantics (`deviceSimpleShuntWithTransformerControlMatchesCpu`, a shunt at
the 132 kV bus and a transformer at the 33 kV bus, AFTER_GENERATOR and INCREMENTAL).

**Added, for correctness.** SecondaryVoltageControl co-present with ANY control loop is now a
whole-run CPU gate. SVC is OLF order 4, before phase (7), transformer (8), RPC (9) and shunt (10), and
it is the last HOST orchestrator on the chunk path: it runs AFTER the entire device sweep. So an
in-kernel control loop decides on a state SVC has not corrected yet, and SVC then decides on a state
whose taps have already moved. Measured on a zone whose pilot bus feeds an AFTER_GENERATOR
transformer: **1.7e-3 kV** with the control loops in-kernel and **2.4e-1 kV** with the
pre-2026-09-15 host-orchestrator chain — that combination was SILENTLY WRONG before this gate, and
routing the host chain back was measured worse, not better (the chain's re-solves do not compose).
`deviceSecondaryVcWithAfterGeneratorTvcRoutesToCpu` pins it; flip it to `assertGpuMatchesCpu` when
the gate lifts. This also answers the "restart with a machine mid-flight" question: the shape is
unreachable while the gate stands, because SVC is the only loop that can force a restart.

**LIFTED the same day — SecondaryVoltageControl runs as a DRIVER PAUSE on the chunk path.** The kernel
already had the `LOOP_HOST_SVC` slot, `hostLoopMask`, `readBatchDriverPaused`/`setBatchDriverResume` and
flat per-scenario `setBatchTargetDeltas`; only the base path used them. The chunk solver now arms the
mask when the run has SVC controllers and, after the main solve, loops: decide on the paused states →
push the moved V targets as flat deltas → resume. The decision is the SAME code as the host
orchestrator (`decideRound`, extracted so the two cannot drift), and the distributed-slack carry triple
is not needed at all because the driver owns the slack. `OLF_GPU_SVC_HOST_ORCHESTRATOR=1` restores the
old after-the-sweep orchestrator for A/B.

Getting there took two real defects, both found by BISECTING the machinery away from the decision
rather than by reading the code — the technique is worth keeping:

1. Arm the pause but resume every pause as STABLE, leaving the host orchestrator to decide. Still
   wrong (361 differing violations vs 4) ⇒ the machinery, not the decision.
2. Keep the slot in the sweep but make it an in-kernel no-op, so no call split happens
   (`OLF_GPU_HOST_LOOP_NOOP=1`). Correct again (1 differing on the slice, like the baseline) ⇒ the
   SPLIT of `solveBatchContext`, not the extra slot.

That pinned it to state a resumed call re-uploads over the sweep's own work:

- **The disabled-element mask.** A resume re-uploaded it from the host, undoing the element bits the
  HVDC AC-emulation freeze and limits phases had set during the paused sweep. The host mirror is the
  pre-solve one. Guarded with `!resume`: contingency outages do not change during a sweep, so the
  device copy is authoritative. This is why only rte6515 broke — ieee14 has no HVDC.
- **The outer-iteration budget.** A host-loop UNSTABLE verdict did not consume it, while every device
  loop's change does. Each resume then began a fresh `outer` budget, so a loop that kept moving
  targets could never retire a scenario: a 2000-contingency slice did not finish in 28 minutes. The
  resume path now increments `prodIters` for the changed scenarios and retires one that exhausts
  `RL_MAX_OUTER` to the CPU, exactly as the device loops do.

Result on rte6515 with both zones firing, 9042 contingencies:

| chunk path for SVC | differing violations | batched | GPU cold | vs CPU 8 threads |
|---|---|---|---|---|
| host orchestrator | 4 | 9040/9042 | 204 s | x0.47 |
| driver pause (default) | 4 | 9037/9042 | 150 s | x0.66 |
| no SVC, same case | 2 | 9040/9042 | 81 s | x1.20 |

Identical parity, 27% faster, and 3 scenarios conservatively retired to the CPU by the budget rule.
The remaining 4 differing lines are a pre-existing SVC scale gap shared by both paths (2 of them are
the baseline knife-edge lines) — that is the next SVC question, not this one. SVC still makes the case
about 2x slower than without it on either path, so it needs a cost pass of its own.

Still on the host inside the pause: the observables (OLF's own `EquationTerm.calculateSensi`) and the
small zone LU. Moving those into the kernel is a later, optional step — see the stage list below; it
buys wall time, not correctness.

### #47 — Discrete TransformerVoltageControl and WITH_GENERATOR shunt VC in-kernel (landed 2026-09-15)

`LOOP_TVD_KERNEL` (OLF order 8) runs OLF's non-incremental transformer loops per scenario, one thread
per scenario (OLF keeps ONE step for all controls): AFTER_GENERATOR (`tvdMode 0`) — INITIAL enables
every group whose pilot is out of its deadband (TransformerRatioManager bounds computed over the
alive members), CONTROL clamps each continuous ratio outside its (group-averaged when
useInitialTapPosition) range to the extreme, rounds it (`tvdRoundToTap`: from the CURRENT tap, strictly
closer only — `ClosestTapPositionFinder`) and fixes it, else remaps (`updateContinuousRatio`) + rounds
everything and goes COMPLETE, UNSTABLE in any case (the COMPLETE polish solve); WITH_GENERATOR
(`tvdMode 1`, SimpleTransformerVoltageControl) — the control is ON at entry (`tvdEntryBatch` applies the
ON deltas for a chunk, or reads them off the base activation OLF's initialize produced, `entryOn`),
the first check rounds and fixes every live member. Rows are laid out by ROLE (`tvdLayoutGroup`): the
pilot's V identity and the other non-member deltas as captured (the scenario's OFF target is SAVED on
the way on — a per-scenario override on the pilot's Q row comes back), the first alive continuous member
hosts the relocated pilot Q (the fills' mode scan — `relocChoose` / the DISTR_RHO carrier skip — picks
the same row), the others DISTR_RHO (0), fixed/outaged members identity pins; a group with no continuous
member left is OFF. Fixed taps ride the shared r1 override rebuild. The plan-A6 carrier-clamp rejection
is OFF by default: the in-kernel sequence IS OLF's, and `deviceSharedCarrierClampBatchesAndMatches`
now batches the clamp scenario (and drives the clamping base) at 1e-12 — `OLF_GPU_TVD_REJECT_CARRIER_CLAMP`
restores the CPU route for diagnosis. Base path: `readBatchDiscreteTvcState` → `setR1(roundFrom);
roundR1ToClosestTap(); setVoltageControlEnabled(false)` per fixed member (OLF's own calls).

`LOOP_SVD_KERNEL` (OLF order 10) is the shunt twin (`svdEntryBatch` / `svdCheckBatch`): ON at entry,
the first check runs `LfShuntImpl.dispatchB` verbatim per controller shunt (residue / remaining to the
closest section from the CURRENT position over [minPosition, size), strictly closer only), pins the
SHUNT_B identity at the section total and restores the pilot; OLF's dispatchB leaves the shunt's g
alone, so no (dG, dB) override is emitted (the host machine's dg was an OLF deviation, harmless on
constant-G sections). Base path: `readBatchDiscreteShuntState/Positions` → `setB(contB); dispatchB()`.

A solve that FALLS BACK to the CPU Newton-Raphson (a rejected shape) now turns the driver off for the
REST of that engine run (`driverFallbackForRun`): OLF's real loops took over there and hold the loop
state, so the device must not restart its machines from INITIAL against contexts sitting at
CONTROL/COMPLETE (measured 2.2e-3 kV on the carrier-clamp base before the rejection was dropped).
The host machines (`GpuBatchTransformerControl.orchestrate…`, `GpuBatchSimpleTvc`,
`GpuBatchSimpleShuntVc.roundAndResolve`) stay for a control the upload refuses (a member without a tap
table).

**GeneratorVoltageControlManager + fixTransformerVoltageControls (same day, second commit).** The
last AFTER_GENERATOR gate is gone: generator voltage controls under the transformer-controlled
level are handled in-kernel. Java (`buildFreezeTables`) replicates the manager's static part —
`minNominalVoltageLimit` (the `generatorVoltageControlMinNominalVoltage` override, else the max
nominal voltage over the transformer-controlled buses whose controllers all join two different
levels, `Double.MIN_VALUE` when none), the candidate controller buses (at or under the limit, no
`hasStepUpTransformers` exemption; a REMOTE candidate keeps the CPU route, its row is a DISTR_Q
share) with their controlled-bus V row and load Q, and per member the V rows of the generator-
controlled buses in the not-controlled side's component with every transformer-controller edge
removed (`LfNetwork.getConnectivity()` temporary changes). The kernel's INITIAL step, once needRun
is known: every candidate whose row is still a V identity (mode 0/2 — an RL-pinned one is NOT frozen,
like OLF's `isGeneratorVoltageControlEnabled` test) saves its mode/target and becomes a Q row at
`qBuf − loadQ` (OLF freezes the generation target at `q.eval()`, the NET injection, so the row
target is that minus the load); then each just-enabled member whose PV list holds no bus still in
mode 0/2 is disabled again (frozen AND pinned buses count as off, exactly `fixTransformerVoltage
Controls`); then the groups lay out. The COMPLETE transition restores the frozen rows
(`enableGeneratorVoltageControlsUnderMaxControlledNominalVoltage`). RL ignores a frozen row (mode 1,
limit type 0) as OLF ignores a bus without QLimitType. Base write-back:
`freezeGenerationTargetQAndDisableGeneratorVoltageControl(target + loadQ)` while frozen,
`setGeneratorVoltageControlEnabledAndRecomputeTargetQ(true)` once unfrozen (only a bus whose network
state shows OUR freeze: control off, target frozen, no QLimitType — the RL write-back keys on the
QLimitType and never touches it). `getMaxControlledNominalVoltageOverride()` is the second OLF
getter of the day. Approximations, documented: the component lists are the BASE topology with ALL
controller edges removed (OLF removes the ENABLED ones plus the contingency's branches per solve),
so a contingency splitting a not-controlled-side component, or an in-band group whose branch is
the only path to the PV buses, can differ. Tests: `deviceTransformerControlGeneratorFreezeMatchesCpu`
(2T2wt at 150 kV: G1 frozen and the controls disabled by the fix; 90 kV / auto: no freeze) and
`deviceSeveralTvcGeneratorFreezeMatchesCpu` (OLF's four-bus network: auto freezes the 90 kV
generators; 230 kV freezes all and the fix disables the three controls), 1e-12, count asserted,
driver off and on.

### #46 — Continuous PhaseControl in the phase phase (landed 2026-09-15)

`cphCheckBatch` shares `LOOP_PHASE_KERNEL` with the incremental kinds (`nCph > 0` selects it).
Per (scenario, controller) at the phase slot's FIRST check (`dCphIter[s] == 0`): a CONTROLLER
switches off — its a1 row leaves mode 3 (free a1, P block re-emitted) for the identity pinned at
the closest tap's a1 (`x` seeded there) — and the scenario is UNSTABLE whenever a live controller
exists (OLF: `status = UNSTABLE` for every controller at iteration 0, even one whose rounding
changes nothing — that still costs the CPU a Newton pass, so it costs us one too). From the second
check on, a LIMITER whose controlled I exceeds its limit moves one tap against sign(dI/da1) —
`flcFlowAndGrad(kind 2)` at the branch's current state, local control only (remote LIMITERs stay
CPU-routed at extraction). Taps ride the shared r1 override rebuild (`tvcRebuildOverrides` merges
TVC + phase + flow tables + the caller's list). Base path: `readBatchContinuousPhaseTaps` →
`PiModel.setTapPosition` + `setPhaseControlEnabled(false)`; the `DeviceDrivenLoop` proxy now
forwards `initialize()` to the real loop for every driven loop except FreezingHvdcACEmulation
(PhaseControl's initialize enables CONTROLLER control, which the extraction must see). The host
orchestrator (`GpuBatchContinuousPhase.orchestrate`) stays only for a chunk whose tables did not
upload. 10/10 phase tests at 1e-12, driver off and on.

### #44 — Shared-group incremental TVC in the TVC phase (landed 2026-09-15)

`tvcGroupDecisionBatch` replaces the per-controller decision kernel: one thread per (scenario,
controlled-bus group) runs OLF's `checkAndAdjustControlledBus` — `adjustWithOneController` (full
diffV, maxTapShift) for a singleton, `adjustWithSeveralControllers` (one tap per controller per pass
in merged order, remainingDiffV through each sensitivity, MIN_SENSI_FILTER per controller, until a
pass moves nothing) otherwise, both through `tvcMoveTap` (closest r1 within the shift bound and the
ratchet's range, strictly closer than the current one). The JNI upload no longer refuses shared
buses, so the host TVC orchestrator is off for every incremental group; `deviceIncrementalSharedTvc
MatchesCpu` runs its 3-member group in-kernel (`nCtrl=3 nGroups=1 kernel=true`) at 1e-12, driver
off and on.

### #43 — In-kernel IncrementalShuntVoltageControl (landed 2026-09-15)

`LOOP_SHUNT_KERNEL` at OLF order 10, same shape as the TVC phase: `shcDeadbandCheckBatch` per
(scenario, controlled-bus group) → one unit-RHS solve per needed controller at its SHUNT_B identity
row on the driver's own factors (`shcGatherSensiBatch` reads dV at the controlled bus) →
`shcAdjustBatch`, one thread per (scenario, group), OLF's `adjustB` verbatim: round-robin over the
|b|-sorted controller shunts, one section per sub-controller per pass, the per-round maxSectionShift
cap, `updateSectionB` (closest section within one step and the ratchet's allowed range, strict <),
`updateAllowedDirection` (MAX_DIRECTION_CHANGE 3), remainingDiffV through the sensitivity →
`shcApplyBatch` pins the section total at the identity row (target + state seed) →
`shcRebuildOverrides` appends the static (dG, dB) per moved controller to the caller's shunt
overrides (`applyShuntOverride` SUMS matching entries, so the loop's delta composes additively) —
redone on a driver resume. Unlike the TVC phase it takes shared groups (the round-robin runs in the
kernel), so the host orchestrator is now off whenever the tables upload. Base path: the loop's
`getMaxSectionShift()` (OLF getter), `readBatchShuntPositions` → `LfShunt.Controller.updateSectionB`.
Tests: the 7 `*Shunt*` SA tests pass driver off and on with a section actually moving
(`SHUNT_KERNEL ... subsOffBase=1`); no shared-group incremental shunt test exists yet — worth adding.

## Method notes worth keeping

- **A GPU test that only compares against CPU cannot detect that the GPU never ran.**
  CPU-identical results are the fallback CONTRACT. Assert
  `GpuNewtonRaphson.CPU_FALLBACK_COUNT` (see `GpuTestIsolationExtension`), not logs (the test
  logback config attaches no appender) and not `olf.gpu.cpuFallback=false` alone (it only
  covers throws reaching that one catch). `GpuPhaseControlTest` passed for a whole session
  while the GPU never executed.
- **Before marking a capability ⏩ ("a GPU gap"), check the CPU side does anything at all** —
  `LfActionUtils.createLfAction`'s `default` throws. Four action rows were once ⏩ for actions
  OLF does not implement; batching them would make the GPU SUCCEED where CPU throws.
- **Never trust bench timings from a loaded box.** Sample load over several seconds first; the
  2026-07-20 row read ×0.74 purely from contention + thermal throttle.

## Coupler-tree relocation: the remaining step is PER-SCENARIO, on the device (2026-09-23)

`-Dolf.gpu.ziTreeRelocation` (off by default, `…MinCouplers` threshold) lets the BASE solve
rte6515's busbar windows that the numeric row repair cannot hold — window [5000,6000) converges on
device with the trees relocated. What it does NOT yet do is carry the SCENARIOS: a contingency that
opens a relocated coupler deactivates that coupler's `ZERO_PHI`/`ZERO_V` (which the relocation moved
onto the child's `PHI`/`V` rows) and pins its dummies with `DUMMY_TARGET` identities (which those
rows no longer host), so the scenario's system is mis-placed.

Rejected (measured, 2026-09-23): routing coupler-opening scenarios to the CPU. Busbar-section
contingencies ARE coupler openings — the gate left **0 of 1000** scenarios on the device for that
window, i.e. it deletes the workload it was meant to protect. The CPU is not the answer here.

The answer is the mechanism the code already has twice: **both placements in the CSR pattern, the
destination chosen per scenario.**

| scenario has the coupler | child bus P/Q row | coupler `ZERO_PHI`/`ZERO_V` | coupler dummy rows |
| --- | --- | --- | --- |
| closed (the base) | `DUMMY_P/Q(coupler)` | child's `PHI`/`V` | — |
| **opened** | the child's own `PHI`/`V` | deactivated | `DUMMY_TARGET` identity |

One selector bit per (relocated coupler, scenario) — "is this coupler open here" — which the batch
already knows. Precedents to copy, not to reinvent: dynamic Q-relocation (§11 — `qRowAlt`,
`cbAltRow`, `hasRelocatable`: two candidate rows unioned into the pattern, the fill picking by row
mode) and continuous phase control (`cbPcSide`, `collectCbPhasePattern`, `buildCbPhaseDest`: extra
positions unioned, emission gated by a per-scenario mode). Both placements are permutations of the
same equations, so mirroring OLF exactly is preserved.

Open, noticed while measuring the rejected gate: in that experimental config (relocation at
threshold 1, every scenario CPU-routed) the window failed the distributed-active-power assertion
with a worst diff of 14.3599 MW — exactly the base's own distributed total, so one arm reported the
base value where the other reported ~0. CPU-routed scenarios go through
`super.runPostContingencySimulation` (`GpuBatchedAcSecurityAnalysis:1305`), and the default suite is
green, so this is not a shipped defect — but it wants explaining before the fallback path is trusted
for anything but the handful of scenarios it carries today.

## The busbar windows: what actually blocks them (corrected, 2026-09-23)

**A window's BASE network depends on its contingency set.** OLF retains the switches a contingency
needs, so more busbar-section contingencies ⇒ more retained switches ⇒ more zero-impedance couplers
in the base LfNetwork. Measured on window [5000,6000) with `OLF_GPU_SA_MAX_CTG`:

| slice | on device |
| --- | --- |
| 200 | 193 / 200 |
| 600 | 578 / 600 |
| 1000 | **0 / 1000** — the BASE LF itself falls back |

So "window [5000,6000) fails" is not a property of those contingencies: it is the switch count their
union retains. Every windowed measurement has to name its slice, and a slice that passes says nothing
about the full window. This also corrects an earlier attribution here: the coupler-tree relocation was
credited with fixing this window's base. It does not — the N=1000 base fails IDENTICALLY with
`-Dolf.gpu.ziTreeRelocation` on and off (same two fallbacks, same order). The relocation's real result
stands on its own (the coupler chain's first Newton step, exact at 7/15 couplers where the numeric
repair is wrong), but it is not this window's blocker.

The two real blockers, both in the base LF, both before any contingency runs:

1. **`row mode 4 (pinned remote gen-VC member) uses stale static qPct on-device`** — an explicit
   `UnsupportedOperationException`, i.e. a deliberate CPU route, not a numeric failure. A reactive-limit
   pin makes OLF renormalize the group's reactive keys (`updateReactiveKeys`) and fire NO
   equation-system event, so the cached extraction kept its pre-pin `qPct`.
   FIXED (this commit): `GpuAcActivationExtractor.pinnedRemoteMembers` exposes the pin SET, and
   `GpuNewtonRaphson` re-extracts when it changes. The fresh pack's `dqWeight` differs, which
   `GpuAcData.sameStructureAs` already compares, so the structure is REBUILT with OLF's renormalized
   keys instead of being value-refreshed in place. The throw now only covers modes above 4.
   Still open: the BATCHED path pins on-device (the RL kernels set mode 4 themselves) and cannot
   renormalize keys there — the same staleness may exist per scenario. Not yet investigated.
2. **A NaN** on the first base attempt once the reactive-limits loop engages on a switch-heavy base.
   Next up.

### The remaining blocker, characterized (2026-09-23)

`OLF_GPU_LIN_RESID=1` on the N=1000 window separates the two candidate explanations, and it is NOT a
diverging Newton. Three base solves run in sequence (all `slot=0`, n=13228):

| solve | first-iteration `‖J·dx−F‖/‖F‖` | `‖dx‖inf` | `‖F‖inf` at entry |
| --- | --- | --- | --- |
| A | 8.6e-09 → 1.2e-13 over 3 iters | 11 → 1e-2 | 40.6 (flat start) |
| B | 1.2e-11 | 3.2e-03 | 3.3e-03 |
| **C** | **6.7e+08** | **7.7e+13** | **1.73 (a sound state)** |

Solve C enters from a reasonable state and its FIRST linear solve is wrong by eight orders of
magnitude; every later iteration stays at 1e7–1e9 with steps of 7.7e13, and the Newton then walks to
NaN. So the matrix of that outer-loop pass does not factor unpivoted — the divergence and the NaN are
CONSEQUENCES, and `MAX_VOLTAGE_CHANGE` clamping (which the RTE params set, and which both the
sequential and batched paths do pass — `lsMode=2`, `maxVoltageChangeFactorBatch`) cannot save a step
computed from a broken factorization.

The batch's own divergence in the same log is also a consequence, not a second defect: with the base
fallen back, every scenario starts from a flat state (`‖F‖inf = 5.555`, identical to 7 digits across
all slots) instead of the converged base.

Leads for the next session:
- The worst rows repeat — 12582, 12914, 12933, 13005, 13181, all high indices of 13228, i.e. the
  non-bus variable region (DUMMY_P/DUMMY_Q, BRANCH_RHO1, SHUNT_B). Resolve them to equations first:
  a zero/tiny diagonal there is the unpivoted-LU failure this smells like, and the same family as the
  coupler-tree work even though the tree relocation does not change this outcome.
- What changes between the sound solve B and the broken solve C is one OUTER-LOOP pass. Identify
  which loop acts there (PV→PQ, a reactive-limit pin, a tap or shunt move) — that names the row whose
  mode/placement flips into the unfactorable shape.
- The single-solve linear-residual guard runs only at `it == 1`
  (`olf_gpu_nr_jni.cu:2678`), so a factorization that degrades later is unguarded on that path; the
  batched path's `lastSolveRatio` check is what actually caught this one, reported as a NaN.

#### Hypotheses eliminated by measurement (same window, one run each)

- **Fill-reducing ordering.** `OLF_GPU_REORDER_ALG=nd` → still NaN. Worth testing because the robust
  config hard-codes AMD when the variable is unset, and this very file documents that "AMD IS NOT SAFE
  WHEN THE SYSTEM HAS ZERO-IMPEDANCE BRANCHES" (AMD is symmetric and assumes a non-zero diagonal;
  ZERO_V/ZERO_PHI sit on the DUMMY rows and carry v1−v2 / phi1−phi2, never their own dummy) — on a base
  whose whole difficulty is retained switches. The reorder choice is NOT the cause here, but the robust
  path bypassing `applyReorderAlg(config, hasZeroImpedance)` still looks wrong on its own terms.
- **Robust mode engaging mid-run.** `-Dolf.gpu.batchMode=6` from the first context → still NaN. Tested
  because cuDSS permits ONE analysis per data object and matching's permutation is value-dependent, so
  a robust context built after a reject analyses mid-run values; the comment credits the full-scope
  base's convergence to analysing the run's FIRST values. Not the cause either.

So the robust single-matrix path (matching + pivoting, AMD or ND, from the start or after a reject)
returns NaN on this structure, while the uniform batch returns a finite-but-wrong solve (ratio 1e7).
Both orderings and both modes failing IDENTICALLY points away from the factorization and toward the
matrix itself.

**Next probe (needs a native rebuild, small).** The device checks `F` for finiteness in both paths
(`olf_gpu_nr_jni.cu:2578`, `:5192`) but NOTHING checks `dVals` — the Jacobian — before handing it to
cuDSS. An env-gated scan of `dVals` for non-finite entries, reporting the first offending (row, col)
and the element that fills it, decides between "the factorization cannot handle this matrix" and "a
fill wrote NaN into the matrix". Do that before any more ordering/mode sweeps: a NaN entering from a
fill would explain every result above at once.

### ROOT CAUSE LOCALIZED: the device's own RL pin assembles a rank-deficient system (2026-09-23)

The base solve that fails is the FIRST fill carrying **mode-4 rows** — an RL-pinned remote gen-VC member.
The per-fill census makes it unambiguous (`OLF_GPU_VALS_SCAN=1`):

| fill | mode 0 | mode 1 | **mode 4** | dupRows | deadCols | tinyDiag |
| --- | --- | --- | --- | --- | --- | --- |
| healthy (outer=0) | 998 | 12230 | **0** | none | 0 | 0 |
| **broken (outer=6)** | 982 | 12243 | **3** | none | 0 | 0 |

The matrix is finite, well-scaled (maxAbs 4.07e+04), every diagonal present and above 1e-12, no duplicate
row, no dead column — and unsolvable. **KLU, the CPU's own solver, cannot solve it either**
(`GpuDumpedMatrixRankTest` on an `OLF_GPU_VALS_DUMP`: `||dx||inf = 3.97e+16`, residual ratio **7.59**
where a real solve is ~1e-12). So cuDSS is NOT at fault and no solver configuration can help: the system
we assemble is rank-deficient. Since OLF converges on this network on the CPU, OLF's matrix at the
equivalent pass is not this matrix — **the defect is ours**.

KLU's blown-up solution is a null-space direction, and its largest components name where the rank is
lost: columns 13171, 13207, 13201, 13185, 13177, 13179, 13175, 13199, 13170, 13181 — all with
`diag = ±1` and `F = 0`, i.e. DUMMY_P / DUMMY_Q columns — clustered around mode-4 row 12560 (the three
pinned rows are 12560, 12588, 12834). That is the failure class `GpuAcActivationExtractor:129-146`
already describes: the existence-keyed DISTR_Q kind-3 terms staying live ALONGSIDE the zero-impedance
family's direct ±DUMMY_Q into the same row, the two cancelling and "leaving the coupler's DUMMY_Q in NO
equation. 45 such columns at 2000 busbar sections, which is what collapsed batching above ~1000" — and
N=1000 is exactly where this window breaks. The `pilotSelf` fix recorded there evidently does not cover
this instance.

Two further facts that matter for the fix:
- The pin happens **on the device**, inside the RL kernels, so no host-side check sees it. The
  pre-change Java produces a byte-identical broken fill (`modes=982,12243,0,0,3`), which is how we know
  the `row mode 4` de-fallback did not cause this — it only stopped hiding it.
- Ruled out by measurement, so do not re-sweep them: orderings AMD / ND / BTF-COLAMD, pivot types
  auto / global column / global row with threshold 1.0, matching auto and off, NaN entering from a fill,
  zero and tiny diagonals, duplicate rows, dead columns.

Next: compare the device's assembled system at that pass against OLF's pinned system for the same
group — specifically whether a pinned member's own Q row and its DISTR_Q row both carry, or both drop,
the coupler's DUMMY_Q term. Fix the assembly; the factorization is innocent.

Tools left in place for it: `OLF_GPU_VALS_SCAN=1` (per-fill census: finiteness, maxAbs, diagonals,
duplicate rows, dead columns, mode histogram), `OLF_GPU_VALS_DUMP=<path>` (one-shot CSR+rhs+modes at the
first mode-4 fill), `GpuDumpedMatrixRankTest` (factor it with KLU offline, print the null direction), and
`OLF_GPU_PIVOT_TYPE` / `OLF_GPU_PIVOT_THRESHOLD` / a now-honoured `OLF_GPU_MATCHING_ALG` in robust mode.

#### The wrong equation, named

The null direction's dummy columns show the defect structurally:

```
col=13171  touchedBy=3  (row=12560 mode=4 v=+1) (row=12562 mode=1 v=+1) (row=13171 v=-1)
col=13207  touchedBy=3  (row=12834 mode=4 v=+1) (row=12836 mode=1 v=+1) (row=13207 v=-1)
```

A zero-impedance branch's `DUMMY_Q` is determined by an OPPOSING pair: `+1` in one endpoint's Q row and
`-1` in the other's (`fillZeroImpBatch`: `addAt(d2, dummyQ)` / `addAt(d3, -dummyQ)`). Here both real
entries are `+1` (the `-1` on the diagonal is the numeric repair's donor, `nHaz=242`), so the
opposite-side coupling is MISSING. That endpoint is a remote gen-VC member whose own Q row is canonically
SUPPRESSED (`qRow = -1`, `ExtractionColumns:164-172`) because its reactive balance moved into the group's
DISTR_Q row — so the `-dummyQ` dest resolved to `-1` (skip) and the term was dropped. The dummy is then
undeterminable, and across the pinned cluster the matrix loses rank. The identical pattern at both pinned
members (12560/12562/13171 and 12834/12836/13207) is a systematic assembly bug, not a numerical accident.

OLF does not lose this: the suppressed member's reactive balance still exists inside the group's DISTR_Q
equation. Our layout drops the coupling instead of ROUTING it to wherever that member's balance now lives.

Fix direction: the zi family already carries a per-side alternate Q row for exactly this
(`ziQAlt`, `ziDynDest`), so the question to answer first is whether `ziQAlt` is `-1` for such a bus (the
alternate was never computed for a suppressed remote member) or whether it is set but not SELECTED when
the other side is at mode 4. Fix that, then re-measure the N=1000 window: the base should factor, and
976 of 976 scenarios are already planned onto the device behind it.

### THE DEFECT, read off our own equation dump (2026-09-23)

`GpuEquationDump` (new, the counterpart of OLF's `EquationSystem#write`) prints one line per device row
plus a per-coupler section with every row slot, `-1`s included. Run it with `OLF_GPU_EQ_DUMP=<path>`; it
also writes OLF's own dump of the same state to `<path>.olf`. On the failing window it names the defect:

```
zi#18 branch=BUS-5997_COUPLER_1 b1=VL-5997_0 b2=VL-5997_1
      pRow1=12561 qRow1=12562 pRow2=12563 qRow2=-1 dummyP=13170 dummyQ=13171 qAlt=[-1,-1]
zi#36 branch=BUS-6112_COUPLER_1 ... qRow2=-1 dummyQ=13207 qAlt=[-1,-1]
```

Those are EXACTLY the two columns KLU's null direction pointed at. The `-dQ` coupling has no row
(`qRow2 = -1`) and no alternate (`qAlt = [-1,-1]` — so it was never computed, not merely unselected),
because that endpoint is a remote gen-VC member whose own Q row is canonically suppressed: its reactive
balance lives in the group's DISTR_Q row.

Why the base is healthy until the pin, which is what makes this conclusive:

- BEFORE the pin the dummy's missing second coupling is supplied by the DISTR_Q **kind-3** term
  (`dq[elem=13171 kind=3 w=qPct-1]`), so the column IS determined and the base converges.
- AT the pin `updateReactiveKeys` zeroes that member's own key, so every DISTR_Q term on its row becomes
  the exact negation of the member's own terms (`w = 0 - 1 = -1`): the dump shows `dq[k=106 elem=8699
  w=-1]` against `cb#8699.q2` and `dq[k=107 elem=13171 kind=3 w=-1]` against `zi#18.q1(+dQ)`. The
  compensation disappears, `DUMMY_Q` keeps a single `+1`, the column is free and the matrix loses rank.

So the row layout is over-subscribed for such a group: one row (12562, `BUS_V(VL-5997_1)`) carries the
relocated pilot reactive balance AND has to become the member's OWN Q balance when the member is pinned.
OLF has no such problem because its rows are EQUATIONS — `BUS_TARGET_Q` and `DISTR_Q` each get their own.

Fix options, in the order worth trying:
1. Give the zi family the missing alternate: compute `ziQAlt` for a suppressed remote gen-VC member (its
   reactive balance row), so `ziDynDest` can route `-dQ` there when the member is pinned. Smallest change,
   uses machinery that already exists.
2. Keep the DISTR_Q kind-3 compensation alive for the dummy when the member is pinned (i.e. gate only the
   terms that duplicate the member's own q, not the coupler's).
Either way the invariant to assert afterwards is the one the dump now checks: a coupler's dummy coupling
is complete, or the missing side has an alternate that is live in every row mode.

### THE FIX: 0 -> 900 of 1000 scenarios on the device (2026-09-23)

The RL bookkeeping keys every pin on the entry's own V row, which is only valid when that bus's reactive
balance actually lives there. On a PILOT that is one of its own controllers, pinned while a SIBLING
controller still holds the voltage, it does not: the pilot's V row must keep `BUS_TARGET_V`, so
`computeQRow` relocates its reactive balance to the free member's V row. Two entries then key off one row
— the pilot's (whose pin marker is elsewhere) and the receiving member's (which reads a marker that is not
its own) — and the device pinned one row twice while unpinning the row the balance lived on. Exactly 2
buses on rte6515 have this shape (`VL-5997_0`, `VL-6112_0`); the other 3 relocated pilots have candidate
tables (`qRowAlt = -2-group`) and their re-pick machinery is fine.

Fix: the host marks such entries with `GpuAcOuterLoopPacks.RL_BASE_RELOCATED` (a base limit-type code, 6),
and the device turns that into `RL_LIMIT_FROZEN` so the reactive-limits pass, the switch-back AND
`seedRlModeFromState` all leave the row alone. The last one mattered: `RL_LIMIT_FROZEN` is non-zero, so
without an explicit skip that seeder stamped mode 4 back onto the wrong row and undid the fix.

| window [5000,6000), N=1000 | before | after |
| --- | --- | --- |
| scenarios on device | **0 / 1000** | **900 / 1000** |
| statuses vs CPU | — | identical (974 CONVERGED, 2 MAX_ITERATION both arms) |
| violations | 71538 vs 71533 (diff 5) | 71538 vs 71534 (diff 4) |
| fallbacks | 2 | 1 |

Suite: 288 tests, 0 failures. Checkstyle unchanged at 214.

Still open on this window, in order:
1. The base LF still falls back once (NaN). Understood now: the ALL-PINNED corner legitimately flips the
   pilot's V identity to mode 4 (`updatePilotVeqBatch`), but in this TABLE-LESS group neither the pilot nor
   the free member has an alternate dest (`qRowAlt = -1`), so the flipped row assembles EMPTY. Giving them
   `qRowAlt = vCol` in `ExtractionColumns` was TRIED and REVERTED: it changes the alternate-dest selection
   and the unioned pattern for every scenario, and took the window back to 0 / 1000. The alternate has to
   be introduced without disturbing the batch — most likely by making the all-pinned flip route the q terms
   itself, rather than by widening the static canonical encoding.
2. `distributedActivePower`: 525 of 974 contingencies differ, worst 1.057 MW. Unchanged by this fix and
   identical in the all-CPU-routed run, so it belongs to the base-fallback path, not to the batch.
3. Violation diff 4 of 71538.

### Why the base still falls back: a DEGENERATE DISTR_Q, not a missing row (2026-09-24)

Printing the host's pinned rows next to the device's settles what the remaining NaN is. At the failing
pass the two disagree outright:

```
host  (from OLF's state) hostMode4Rows=[12562 12588 12590 12592 12836]
device (ctx->dMode)      mode4Rows=[12560 12588 12834]        <- at outer=6, the first pinned fill
```

The device is not mis-pinning: it runs its own RL decisions against the FIRST extraction's layout, and in
THAT layout its choice is routable and right. `OLF_GPU_EQ_DUMP_WHEN=any` shows it:

```
VL-5997_0 (pilot)  qRow=-1 qRowAlt=12560 enabled=true   row=12560 mode=0 terms=[cb#8699.q2@alt]
VL-5997_1 (member) qRow=-1 qRowAlt=-1    enabled=true   row=12562 mode=1 terms=[dq ... w=0.500000 ...]
```

Both q scatters are suppressed into the group's DISTR_Q row with weight 0.5 while both controllers are
enabled — correct. The device then pins the pilot at 12560, where its `@alt` dest routes its q terms
properly, and the key renormalization gives the SURVIVOR a share of 1.0. At that point
`DISTR_Q_member = 1·Σq − q_member = q_pilot`, which is the SAME equation as the pilot's own Q balance now
hosted at 12560. Two identical equations, which is precisely the two `+1` entries measured on the
`DUMMY_Q` column and the lost rank. Nothing is missing from the assembly; one equation is DUPLICATED.

OLF cannot hit this: when a controller pins, the survivor becomes the FREE controller, its `DISTR_Q`
ceases to exist and `BUS_TARGET_V` takes that row. That ROLE ROTATION is what the candidate table
(`qRowAlt = -2-group`, `relocChoose`, `updatePilotVeqBatch`) implements — and this group has none, because
the qr-group loop deliberately excludes "a pilot that is ALSO one of its own controllers" (leaving it in
the table double-counted its q). Exactly the 2 rte6515 buses in that shape are the ones that fail.

So step 1 is NOT a small patch. Two ways to close it, both real work:
1. Extend the candidate table to pilot-self groups: give the group a table whose member list drives the
   re-pick while keeping the pilot's own q out of it (the double-count the exclusion was protecting
   against). Then the survivor's row gets the V identity and its DISTR_Q is dropped, as on the CPU.
2. A targeted rotation: when the device pins the pilot of a pilot-self group, set the surviving free
   member's row to mode 0 with the group's V target and gate off its DISTR_Q — the same effect, driven by
   a small host-provided pairing (pilot row, survivor row, V target) instead of a general table.

Tried and rejected on the way here, so do not repeat them:
- Static alternates for both buses (`qRowAlt = vCol`): took the window from 900 back to 0 / 1000. The
  selector reads the FROZEN row's mode, and mode 4 there is the pilot's own relocated pin, so the pilot's
  q got diverted off the row that must host it.
- Excluding relocated entries from the all-pinned flip test (`pilotRowAllPinnedBatch`): kept, it is
  correct on its own terms (the flip cannot route q in a table-less group), but it did NOT clear the NaN —
  the flip was never the trigger.

#### Parity check: option 1 IS the CPU behavior (2026-09-24)

`AcEquationSystemCreator.updateRemoteVoltageControlEquations` is the spec, and it is a ROTATION:

```java
vEq.setActive(!enabledControllerElements.isEmpty());       // BUS_TARGET_V iff >= 1 controller enabled
for (disabled)   { distr OFF; ctrl /*BUS_TARGET_Q*/ ON; }  // every PINNED controller gets its own Q balance
for (i, enabled) { distr.setActive(i != 0); ctrl OFF; }    // the FREE one is the FIRST ENABLED - no DISTR_Q
```

`i != 0` over the LIVE ENABLED list is the free-role rotation, and `relocChoose` ("pick the first member row
whose per-scenario mode is neither identity nor pinned") is its device-side evaluation. The candidate table
is therefore not a convenience for a corner case — it is the model of OLF's rule, so **option 1 is the
parity behavior**.

The divergence at the failing pass, in those terms:

| | active reactive equations |
| --- | --- |
| OLF | `BUS_TARGET_V(pilot)`, `BUS_TARGET_Q(pilot)` |
| device | `BUS_TARGET_Q(pilot)` at 12560, PLUS `DISTR_Q(member)` at 12562 with its key renormalized to 1.0, which is algebraically `q_pilot = target` |

i.e. `BUS_TARGET_V` is MISSING and `BUS_TARGET_Q` is DUPLICATED. The rank loss is the symptom of that, not
a numerical accident — which is also why no solver setting could ever have fixed it.

Option 2 (a targeted 2-row rotation) is parity-correct ONLY for two-member groups. Pilot-self groups on
rte6515, counted from the dump: 974 with 1 member, 19 with 2, 10 with 3, and one each with 4, 5 and 6. The
13 groups with >= 3 members need the ORDERED candidate list, because after one pin several survivors remain
and only the first goes free while the others keep their DISTR_Q. So option 2 is a stopgap the data already
outgrows; do option 1.

The hook already exists: `qrGroupOfPilot` is deliberately a SEPARATE channel from `qRowAlt = -2-group`
(`GpuAcOuterLoopPacks:241`, "Read from qrGroupOfPilot, NOT decoded from qRowAlt") so that table membership
and Q-routing are independent. That is exactly what a pilot-self group needs — a table that drives the
re-pick WITHOUT re-routing its own q, which is the double-count the exclusion in the qr-group loop was
protecting against.

### Option 1, slice 1: the pilot-self group table (2026-09-24)

`GpuAcOuterLoopPacks.pilotSelfGroups` packs every MAIN remote generator voltage control whose merged
controllers CONTAIN the controlled bus, with >= 2 controllers: member rows AND member RL ENTRIES in merged
order, the pilot's entry and row, and the group's V target. On rte6515: **27 groups — 17 of size 2, 7 of
size 3, one each of 4, 5 and 6** (`OLF_GPU_RL_TRACE` prints `GPU_PILOTSELF`).

Two design facts decided the shape, both read off the dumps rather than assumed:

- **`BUS_TARGET_V` cannot change rows.** A mode-0 identity constrains its OWN row's variable, so on a
  member's row it would pin that member's v to the pilot's target. It stays on the controlled bus's row.
  What rotates is a PINNED pilot's reactive balance, onto the first ENABLED member's row —
  `computeQRow` already puts it there, but FROZEN at extraction while the device's RL loop moves pins.
- **The choice cannot be read off row modes.** The row to pick is mode 4 BECAUSE it carries the relocated
  balance, which is circular. `limitType` per RL entry is the ground truth OLF's own rule uses, so the
  table is keyed by ENTRY.

Slice 1 is additive — nothing consumes the pack yet, so the 900/1000 window and every existing group are
untouched (suite 290/0, checkstyle 214). `GpuPilotSelfGroupTest` pins the semantics on a purpose-built
3-controller group (one generator controlling its own bus, two more controlling it remotely) instead of
inside rte6515's 13k-variable system, and asserts the merged ORDER, since OLF's free role is index 0 of the
live enabled list.

Slice 2 (next): a device kernel over the pack that reproduces
`updateRemoteVoltageControlEquations` from the pin states — for each group, per scenario: the first enabled
member is free (its `DISTR_Q` gated off), every pinned controller's row hosts its own Q balance, and a
pinned pilot's balance rides the free member's row. Then re-measure the window; the base's remaining NaN is
exactly this rotation being missing.

Trap worth remembering, cost an hour: `ExtractionColumns.of` returns `vCol = -1` for a bus whose variables
the LAZY equation index has not sorted yet. The real pipeline extracts after the solver materialises the
index; a test must call `AcSolverUtil.initStateVector` first or it silently packs nothing.

### METHOD CORRECTION: MAVEN_OPTS does not reach the surefire fork (2026-09-24)

Measured today, unambiguously, on the same run shape: with
`MAVEN_OPTS="-Dolf.gpu.pilotSelfRotation=true"` the property arrives as FALSE in the test JVM (0 enrolment
traces); with `mvn -Dolf.gpu.pilotSelfRotation=true` it arrives as TRUE (1917 traces). Surefire forks its
own JVM and propagates `-D` given on the MAVEN COMMAND LINE, not `MAVEN_OPTS`, which only configures the
Maven JVM. **Always pass a GPU property as `mvn -D<prop>=<value>`.**

This invalidates earlier entries here that switched a property through `MAVEN_OPTS`. Specifically:

- **VOID**: "window [5000,6000)'s base fails identically with `-Dolf.gpu.ziTreeRelocation` on and off" — the
  "on" run had it OFF, so both arms were the same configuration. The conclusion drawn from it, that the
  coupler-tree relocation is NOT this window's blocker, is therefore UNPROVEN and must be re-measured.
- **VOID**: the coupler-tree threshold experiments run through `MAVEN_OPTS` (the 0/1000 and base-failure
  numbers attributed to "relocation at threshold 1").
- **STILL VALID** (no property involved, or set in-JVM): the coupler-chain first-step results
  (`GpuZiCouplerTreeRelocationTest` sets the property with `System.setProperty`); the RL_BASE_RELOCATED fix
  and its 0 -> 900 of 1000 measurement; the KLU rank analysis; every equation-dump finding; the
  DISTR_Q-duplication diagnosis, which was read off dumps rather than a property switch.

### Option 1, slice 2: the pilot-self gate, property-gated and NOT yet working (2026-09-24)

Landed behind `-Dolf.gpu.pilotSelfRotation` (default false; suite 290/0 with it off, so the shipped path is
untouched). What works: pilot-self groups now enrol in the candidate table, and the device DERIVES both
facts the gate needs with no new host table — a group is pilot-self exactly when its fallback row (candidate
0, the pilot's own row) reappears among its member rows, and the pilot's RL entry is the one sitting on that
row. `PILOTSELF_GATE groups=27 of 30` on rte6515, the other 3 being the pre-existing table-ful groups: the
derivation is right.

`pilotSelfGateBatch` recomputes, per (scenario, group) before every fill, whether the pilot is unpinned, and
that gate now suppresses BOTH halves — the pilot's raw-q emission at the carrier row (`RelocTables.psSkip`
in all three pick sites) and the carrier's own `DISTR_Q` suppression in `fillDistrQBatch`. The second half
was missing at first and cost a run: for a table-ful group the carrier's `DISTR_Q` is always inactive
because the pilot always has a balance to place, but a pilot-self group's pilot has none while it controls,
so suppressing it dropped an equation without gaining one.

With both halves gated the window is still **0 / 1000 with 2 fallbacks** (vs 900 / 1000 with the property
off), so something in the enrolment is still wrong. Do NOT enable it. Next suspects, in order:
1. All 27 groups enrol, including 17 of size 2 where the pilot and one member are the whole group; check a
   size-2 group by hand against OLF's rule before trusting the general path.
2. `qRowAlt[pilot] = -2-g` REPLACES the own-row alternate the second canonicalization loop gave these
   pilots. The all-pinned corner then relies on candidate 0 falling back to that row — verify it does.
3. The pattern union (`collectCbAltPattern` and friends) now unions every candidate row for these groups;
   confirm the extra positions exist for the rows the gate can select, or a pick lands on a slot that was
   never patterned.

#### The re-measurement the MAVEN_OPTS error forced (2026-09-24)

Re-ran window [5000,6000) at N=1000 with `mvn -Dolf.gpu.ziTreeRelocation=true
-Dolf.gpu.ziTreeRelocationMinCouplers=1`, confirmed live by 2565 `ZITREE` traces:

| | relocation OFF | relocation ON (threshold 1) |
| --- | --- | --- |
| scenarios on device | 900 / 1000 | 900 / 1000 |
| base fallbacks | 1 | 1 |
| violations vs CPU | diff **4** | diff **284** |

So the CONCLUSION stands on valid evidence this time — the coupler-tree relocation neither fixes nor causes
this window's base fallback — and there is a new reason to keep it off: enabling it takes the violation
difference from 4 to 284. The relocation's demonstrated value remains narrow and unchanged: exact first
Newton steps on coupler chains at 7 and 15 couplers, proven by a unit test that sets the property in-JVM.

#### What the relocation's 284/299 violation diff actually is (2026-09-24, corrected twice)

A row permutation cannot change the solution of a correctly assembled system, so a violation difference
under the relocation is NOT evidence against the relocation — it is evidence that something ELSE still
differs from the CPU, which the relocation exposes. Chasing it that way found it.

Corrections to two things stated earlier in this session, both from sloppy measurement:

1. "Keep the relocation off because it degrades violation parity" — wrong framing, deleted as a
   conclusion. The relocation is a permutation; the diff is a symptom to chase.
2. "States match to 1e-9 kV, so the 284 are pure threshold flicker" — wrong, and drawn from the FIRST few
   `BUSDIFF` lines rather than the worst. The harness already sets the prescribed
   `IncreasedViolationsParameters` (0.5, 0.001, 0.5, 0.001, 0.001), so the documented zero-threshold
   flicker is already suppressed. Nor is it the CPU thread count: at the harness's busbar default of 1
   thread the diff is 299, not 4.

Measured properly, with the relocation ON (`mvn -Dolf.gpu.ziTreeRelocation=true
-Dolf.gpu.ziTreeRelocationMinCouplers=1`), the divergence is CONCENTRATED:

| reproducer | symptom |
| --- | --- |
| `BUS-6014_BBS_2` | worst bus difference **3.982 kV** (most contingencies sit at 1e-9 kV) |
| `BUS-5959_BBS_1` | violations cpu=211 gpu=80 (delta -131) |
| `BUS-5962_BBS_1` | violations cpu=230 gpu=107 (delta -123) |
| any | distributed active power worst **808.1 MW** (1.06 MW with the relocation off) |

Two contingencies carry ~254 of the 299. That is a real physical divergence on a few scenarios, not a
reporting artifact, and it matches the per-scenario gap this file already documents: a contingency that
OPENS a relocated coupler deactivates the `ZERO_V`/`ZERO_PHI` the relocation moved onto the child bus's
rows, and pins the coupler's dummies with `DUMMY_TARGET` identities that those rows no longer host. The
base is fine because nothing is open there; these scenarios are not.

Next step, cheap and named: run `OLF_GPU_SA_CTG_ONLY=BUS-6014_BBS_2` with `OLF_GPU_SA_BUSDIFF=1` and
`OLF_GPU_EQ_DUMP`, relocation on, and check whether that contingency opens a coupler whose
`ziChildBus >= 0`. If it does, the fix is the per-scenario placement already sketched at the top of this
file (both placements unioned in the pattern, the destination chosen per scenario) — not a CPU gate, which
was measured at 0/1000 and rejected.

#### CONFIRMED: the relocation's divergence is exactly the coupler-opening scenarios (2026-09-24)

`OLF_GPU_ZITREE_TRACE` now prints `ZITREE_OPEN ctg=<id> coupler=<id> childSide=<n>` from the batch path's
zi-outage loop whenever a scenario opens a coupler whose rows the tree relocation MOVED (`ziIdx` slot 10,
bits 1-2 nonzero). On window [5000,6000) at N=1000 with the relocation on, the sets are:

| set | count |
| --- | --- |
| contingencies whose violations differ from the CPU (A) | **11** |
| contingencies that open a RELOCATED coupler (B) | 67 |
| **A \ B** (diverge without opening one) | **0** |

`A` is a strict subset of `B`, with no exceptions: the 3.982 kV state difference, the 808 MW distributed
power and all 299 violation diffs come from scenarios that open a relocated coupler, and nothing else
differs. The three named reproducers all appear in B (`BUS-6014_BBS_2` opens TWO relocated couplers,
`BUS-5959_BBS_1` and `BUS-5962_BBS_1` one each). 56 of the 67 do NOT diverge, which fits: when the opened
coupler's subtree becomes an island, the freeze makes every row of it an identity and the placement stops
mattering (the fix committed as 2fee085).

This is the validation target for the per-scenario placement work: after it, **A must be empty while B stays
67**. A metric that only counts violations cannot see the difference — use the bus-level worst over ALL
contingencies (`OLF_GPU_SA_BUSDIFF=1`), not the first lines of it.

It also corrects the earlier rejection of a targeted gate. The gate measured at 0/1000 keyed on ANY coupler
opening, which is every busbar contingency. The precise set is 67 of 1000 (6.7%), and only 11 of those
actually break — so a gate on RELOCATED-coupler openings would cost 6.7% of the batch, not all of it. Still
scaffolding rather than the goal, but an order of magnitude cheaper than what was measured and rejected.

### Per-scenario placement for the relocated coupler rows (2026-09-24)

The relocation moves a child bus's P/Q equations onto its parent coupler's `DUMMY_P`/`DUMMY_Q` rows and the
coupler's `ZERO_V`/`ZERO_PHI` onto the child's own rows. A scenario that OPENS that coupler skips the zi
element (no ZERO equations, no ±dummy coupling) and the host pins the dummy rows with `DUMMY_TARGET`
identities — so the relocated equations get zeroed and the bus is left with no balance. That, and only that,
is what the confirmed set inclusion (A ⊂ B) measured.

Implemented without any new host table, JNI change or `GpuAcData` component, because rows ARE variables: a
relocated child's own P/Q rows are already in `ziIdx` (slots 8/9 phi, 6/7 V) and the relocated dest rows are
slots 4/5. `buildStructure` derives one map, `relocOwnRow[dummyRow] = the child's own row` (both sides
through `rowMap`, since those slots are COLUMNS), and every family's existing alternate-dest channel then
offers the stock placement as its alternate via `stockAltRow`.

The selector is the DUMMY ROW itself, which the host already sets to mode 0 exactly when the coupler opens.
One correctness detail: the existing channel diverts on `mode ∈ {0,4}`, but a relocated child that is
RL-PINNED carries mode 4 on that same dummy row, where its own Q balance legitimately lives — diverting
there would be wrong. So these outputs encode their selector as `-2-row`, which the three pick sites
(`scatterAddDyn`, the residual variant, `ziDynDest`) decode as "divert only while this row is an identity".

Covered: closed branch, open branch, shunt. NOT covered yet: the zi family's own ±dummy couplings into a
relocated child's rows — its alternate channel has Q slots only (4 per element) — which matters for coupler
CHAINS, where a sub-coupler emits into its parent's dummy rows. 1,692 of 2,538 trees are depth 1, so the
measurement below says how much of the gap that leaves.

Measured on window [5000,6000) at N=1000, relocation on (`mvn -D...`, 1 CPU thread — the harness default for
busbar, since a multi-threaded CPU arm lands on a different bus decomposition):

| | before | ungated first attempt | **gated** |
| --- | --- | --- | --- |
| A = contingencies whose violations differ | 11 | 29 | **4** |
| A outside B (should be 0) | 0 | **25** | **0** |
| total violation difference | 299 | 73 | 93 |

The ungated attempt broke the invariant: it applied the fallback to DEAD-END stubs too, whose `childSide` is
also nonzero, so scenarios that open no relocated coupler started diverging (25 of 29). Fixed by marking
tree-relocated couplers with **bit 3** of `ziIdx` slot 10 (`GpuAcFamilyPacker`, set only when
`ziTreeRelocationEnabled()`), which the native map requires. With the property off the bit is never set, so
`relocOwn` is null, `dynReloc` is unchanged and the structure is bit-identical BY CONSTRUCTION rather than
by measurement — the lesson from the ungated run, where the default window moved and there was no
apples-to-apples baseline to compare against.

The two worst cases before this change are FIXED: `BUS-5959_BBS_1` (was cpu=211 gpu=80) and
`BUS-5962_BBS_1` (was cpu=230 gpu=107) no longer appear in A. The four survivors are

| survivor | relocated couplers opened | delta |
| --- | --- | --- |
| `BUS-5958_BBS_1` | 1 | +24 |
| `BUS-6014_BBS_2` | 2 | −2 |
| `BUS-6014_BBS_3` | 2 | −1 |
| `BUS-6029_BBS_2` | 2 | +72 |

THREE of the four open TWO relocated couplers — chains, which is exactly the gap left open: the zi family's
own ±dummy couplings into a relocated child's rows are not covered, because its alternate channel has Q
slots only (4 per element). A sub-coupler in a chain emits into its parent's dummy rows, so opening the
parent strands it. Extending `buildZeroImpDest` to P slots (and `vdl`/`hv`, which have no alternate channel
at all) is the next increment, and `BUS-6029_BBS_2` is the reproducer to drive it.

#### Slice 2 of the per-scenario placement: the zi family's own ±dummy couplings (2026-09-24)

Three of the four survivors opened TWO relocated couplers, i.e. CHAINS: a sub-coupler emits its ±dummyP /
±dummyQ into its parent's DUMMY rows, so opening the parent stranded it exactly as the bus equations were
stranded before. The zi alternate channel carried Q slots only (4 per element), so those couplings had no
stock placement to fall back to.

Extended to 8 slots per element — Q at 0..3 as before, P at 4..7 — with the same `stockAltRow` fallback and
the same `-2-row` selector (divert only while the parent's dummy row is an identity). `buildZeroImpDest` now
takes `relocOwn`, both zi kernels route `d0/d1/d4/d5` through `ziDynDest` (they previously used `d[...]`
directly), and the pattern union adds the stock positions `(ownRow, dummyPCol)` and `(ownRow, dummyQCol)`.

Cumulative on window [5000,6000) at N=1000, relocation on:

| | start | bus equations (6b2499a) | **+ zi couplings** |
| --- | --- | --- | --- |
| A = contingencies whose violations differ | 11 | 4 | **3** |
| A outside B (invariant: 0) | 0 | 0 | **0** |
| total violation difference | 299 | 93 | **16** |

`BUS-5958_BBS_1` (+24) is fixed. The three survivors are `BUS-6014_BBS_2` (−1), `BUS-6014_BBS_3` (−1) and
`BUS-6029_BBS_2` (+18) — two of them differ by a single violation, which at this point is likelier a
threshold edge than a placement error, and should be checked with the bus-level worst before more code.

Remaining candidates for `BUS-6029_BBS_2`, the only substantial one left: `vdl` (load models) and `hv`
(HVDC) have NO alternate-dest channel at all, so their P/Q at a relocated child cannot fall back; and the
DISTR_Q kind-3 terms carry a dummy column into a DISTR_Q row, which may itself be a relocated home row.

#### The last piece for the window: the relocated DISTR_Q row (2026-09-24)

The bus-level check contradicts the violation counts, exactly as the harness warns
("violation parity is a THRESHOLD test and will report zero while a bus is 0.1 kV out"). After both
placement slices, with the relocation on:

| survivor | violation delta | worst bus diff |
| --- | --- | --- |
| `BUS-6014_BBS_2` | −1 | **4.150 kV** |
| `BUS-6029_BBS_2` | +18 | 0.617 kV |
| `BUS-6014_BBS_3` | −1 | 0.041 kV |

A one-violation difference hiding a 4.15 kV state error, so "likely a threshold edge" was the wrong read —
always take the bus-level worst over ALL contingencies, not the violation count and not the first BUSDIFF
lines.

FAST REPRODUCER (about 2 minutes instead of 10): `OLF_GPU_SA_CTG_ONLY=BUS-6014_BBS_2` with
`OLF_GPU_SA_CTG_KINDS=busbar OLF_GPU_SA_BUSBAR_TOP=100000` and the relocation on reproduces the mechanism at
smaller amplitude (41 mV, violations diff 1) — retaining one contingency retains fewer switches, hence a
smaller tree. Validate finally on the full window.

All three survivors open TWO couplers OF THE SAME substation, and the dump shows why that shape is special:

```
zi#0 COUPLER_1 b1=VL-6014_0 b2=VL-6014_1 pRow2=13044 qRow2=13045   (child P and Q both relocated)
zi#1 COUPLER_2 b1=VL-6014_1 b2=VL-6014_2 pRow2=13046 qRow2=-1      (child's Q is a DISTR_Q)
```

A first hypothesis — that `ZERO_V` collides with the child's `DISTR_Q` on its V row — is WRONG, checked:
`GpuAcRowIndexing.distrQRow(es, controller, ziParentEdge, dummyQCol)` already relocates a relocated child's
`DISTR_Q` onto the parent's `DUMMY_Q` column, "leaving its V row free for that coupler's ZERO_V", and it is
used by both `GpuAcActivationExtractor` and `GpuAcOuterLoopPacks`. So `dummyQ` 13047 hosts that `DISTR_Q`
and the layout is consistent.

THE DEFECT is one step further on: when the scenario OPENS that coupler, the host pins `dummyP`/`dummyQ` to
`DUMMY_TARGET` identities, so the relocated `DISTR_Q` at 13047 is ZEROED and nothing routes it back to the
child's V row — which the vanished `ZERO_V` has just freed. The two placement slices give the bus P/Q family
fills their stock fallback; the `dq` contributions never got one.

Next increment: give `buildDistrQDest` the same `relocOwn` fallback the families have — for any contribution
whose row is a relocated `DUMMY_Q`, an alternate at the child's own V row with the `-2-row` selector, plus the
matching pattern positions — and then `fillDistrQBatch` picks per scenario. `relocOwn` is already built and in
scope. `vdl` and `hv` still have no alternate channel at all and remain the other open candidate.

#### NEGATIVE RESULT: routing a relocated DISTR_Q back to the bus's own V row is WRONG (2026-09-24)

Implemented and measured, then reverted. `buildDistrQDest` gained an alternate set of the same six dests
computed at `relocOwn[dummyQRow]` (the child's own V row) with the `-2-row` selector, the pattern gained the
matching positions, and `fillDistrQBatch` picked per scenario. On the fast reproducer
(`OLF_GPU_SA_CTG_ONLY=BUS-6014_BBS_2`, relocation on) the worst bus difference went from **4.086e-02 kV to
6.023 kV** — about 150x worse — so the premise is wrong, not the wiring.

Why the premise looked right and is not: when the coupler opens, the child's `ZERO_V` does vanish and its V
row is free, so "put the DISTR_Q back there" is the obvious move. But a coupler opening also changes the
GROUP: on the CPU, OLF re-evaluates `updateRemoteVoltageControlEquations` over the live controllers, which
can deactivate that `DISTR_Q` entirely (the controller is no longer connected to the pilot), or the child is
islanded, in which case the equation should be zeroed rather than relocated. Blindly relocating it forces an
equation OLF no longer has.

So the next attempt must first establish what OLF's equation set actually IS for that scenario — dump the
CPU side with `EquationSystem#write` for the post-contingency state of `BUS-6014_BBS_2` and compare against
the device's rows — rather than assuming the row simply moves. `GpuEquationDump` writes both sides already;
what is missing is a dump taken with the CONTINGENCY applied, which is worth adding to the harness before
writing more device code.

The tree is left at the best measured state (`dd32d58`): A = 3, total violation difference 16 on the full
window, invariant intact.

### The CPU post-contingency dump, and what it settles (2026-09-24)

`OLF_GPU_EQ_DUMP_CTG=<contingency id>` (+ optional `OLF_GPU_EQ_DUMP_CTG_FILE`) writes OLF's OWN equation
system for the POST-CONTINGENCY state from `GpuBatchedAcSecurityAnalysis.runPostContingencySimulation`,
where `lfContingency.apply()` has already run — so it is the system the CPU actually solves for that
scenario, inactive equations in `[ ]`, with a bus/branch id legend (OLF writes `<symbol><elementNum>`).
This is the reference the device side was missing, and it should be consulted BEFORE writing placement code:
reasoning about where a row could go produced two wrong fixes in a row.

For `BUS-6014_BBS_2` it reads:

```
# disabled buses: [VL-6014_1]
# disabled branches: [BUS-6014_COUPLER_1, BUS-6014_COUPLER_2, LINE-6136-6014#0]
bus_target_v6281 = v6281              ACTIVE    (VL-6014_0)
[ bus_target_v6282 = [ v6282 ] ]      INACTIVE  (VL-6014_1, disabled)
bus_target_v6283 = v6283              ACTIVE    (VL-6014_2)
[ zero_v9042 ] [ zero_v9043 ]         INACTIVE  (both couplers disabled)
dummy_target_p9042/q9042, p9043/q9043 ACTIVE
```

Two conclusions, one of which is a confirmed post-mortem:

1. It EXPLAINS the failed dq attempt. `VL-6014_2` is PV post-contingency (`bus_target_v6283` active), so it
   has NO reactive equation at all — routing its `DISTR_Q` back to its V row added an equation OLF does not
   have, which is why the error grew 150x.
2. The relocation gave `VL-6014_2`'s V row to `zero_v9043`, which is now inactive, so that row must host
   `bus_target_v6283`. The device left it a power row with nothing emitted into it. Fixed in
   `GpuBatchChunkSolver`'s zi-outage loop: when the opened coupler is tree-relocated (bit 3) and the child is
   still voltage-controlled, its V row is set to mode 0 with the control's target. Verified applied by
   `ZITREE_VID` (`child=VL-6014_2 vRow=12566 applied=true`).

HONEST RESULT: the reproducer is UNCHANGED at 4.086e-02 kV. The change matches the CPU reference and closes
a real hole, but it is not the cause of the residual error, so it is kept on the strength of the reference
rather than of a measurement.

Next diagnostic, and the other half of the comparison: dump the DEVICE's per-scenario ROW MODES for this
scenario (`readBatchRowModes` already exists natively) and diff them against this CPU dump row by row for
12563..12566 and 13044..13047. The second `ZITREE_VID` line is the lead — `VL-6014_1` is the bus OLF
DISABLES, and how the device freezes it (or fails to) is now the main suspect.

### Both halves of the comparison, and what they show (2026-09-24)

`OLF_GPU_ROWMODE_DUMP_CTG=<contingency id>` prints the DEVICE's final per-scenario row modes and targets, as
the rows that DIFFER from the base, from the post-solve readback in `GpuBatchChunkSolver`. With
`OLF_GPU_EQ_DUMP_CTG` (OLF's post-contingency equations) that is the full row-by-row comparison, and for
`BUS-6014_BBS_2` it reads:

| row | variable | device mode/target | OLF post-contingency |
| --- | --- | --- | --- |
| 12564 | `BUS_V(VL-6014_1)` | 0 / **1.0** | bus DISABLED — the flat freeze is the device equivalent |
| 12565 | `BUS_PHI(VL-6014_1)` | 0 / 0.0 | disabled |
| **12566** | **`BUS_V(VL-6014_2)`** | **0 / 0.0** | **`bus_target_v6283` ACTIVE**, target 1.142 pu |
| 12567 | `BUS_PHI(VL-6014_2)` | unchanged (1) | P balance active — the stock fallback returns it |
| 13044–13047 | both couplers' dummies | 0 / 0.0 | `dummy_target_*` ACTIVE |

So the device pins a LIVE bus's voltage identity to **zero** instead of to its control target. The
`ZITREE_VID` trace shows the host writing the right value (`child=VL-6014_2 vRow=12566 target=1.142
applied=true`), and the MODE does land (the row is 0, not its base 1) while the TARGET does not.

Next session starts here, with two candidates and a warning:
1. The override may never reach that row — trace `GpuBatchOverrides` for row 12566 end to end (the
   neighbouring zi-outage overrides write 0.0, which is ALSO the base target for a dummy row, so "13044–13047
   are 0.0" is NOT evidence that those overrides landed).
2. A device kernel may rewrite it afterwards — `readBatchRowModes`' own javadoc warns the outer-loop kernels
   rewrite mode AND target, and this dump is read after the solve.

The warning: a mode-0 identity at target 0 on a live bus should drive its voltage to 0 pu and wreck the
scenario, yet the measured worst bus difference is only 4.086e-02 kV. Either the reported voltage does not
come from that row, or the target is rewritten again before the solve that produces the answer. Resolve that
contradiction BEFORE fixing anything — it is the same trap as the two wrong fixes above, where a plausible
story was acted on ahead of the evidence.

### ROOT CAUSE: a coupler opening DISSOLVES a merged voltage control, which the device does not model

Resolved with the two dumps rather than by reasoning, and it is not a placement problem at all.

The override is not lost. `OLF_GPU_TARGET_ROW=12566` shows the host's value reaching the uploaded snapshot:

```
TARGET_ROW row=12566 base=0.000000000 baseMode=1 mode[s=0]=0 ov[s=0]=1.142000000
  row=12566 mode=0<-1 target=0.00000000        <- post-solve readback: back to the BASE value
```

Why it is reset: the base extraction has ONE merged generator voltage control here —
`pilot=VL-6014_0, pilotIsController=true, members={VL-6014_0, VL-6014_2}` — so `VL-6014_2` is a MEMBER,
its V row carries `zero_v9043`, and its reactive lives in the group's DISTR_Q. Side by side:

| | `BUS_V(VL-6014_0)` | `BUS_V(VL-6014_2)` |
| --- | --- | --- |
| device BASE structure | mode 0 (BUS_TARGET_V) | mode 1, hosts `zi#1.ZERO_V` — a MEMBER |
| OLF POST-contingency | `bus_target_v6281` ACTIVE | **`bus_target_v6283` ACTIVE** |

Those buses are one electrical node through the zero-impedance couplers, so OLF MERGES their voltage
controls. Opening the couplers splits the node, the merge dissolves, and each bus gets its own
`BUS_TARGET_V` while the merged group's DISTR_Q structure disappears. The device's structure is frozen at
extraction with the MERGED group, so post-contingency it carries equations OLF no longer has — a different
equation SET, not a permutation of one, which is why no amount of row re-placement closed it.

It also explains the reset: the host write of mode 0 + 1.142 on the member's row is exactly right, but the
gen-VC kernels still believe the merged group and restore the row to the group's own bookkeeping.

Consequences for the plan:
- This is the SAME shape as the pilot-self rotation (slice 2): `pilotIsController=true` with a member on a
  coupled bus. The two lines of work meet here.
- The fix is per-scenario voltage-control re-merging: when a scenario opens the couplers that merged a
  control, the members separated from the pilot need their OWN `BUS_TARGET_V` and must leave the group's
  DISTR_Q. That is a real feature, not a placement tweak, and it wants its own design pass.
- Until then the relocation's residual on this window is understood and bounded: 3 contingencies, all
  opening two couplers of one substation, all of them merged-control splits.

### A real fix, and the residual turns out NOT to be placement at all (2026-09-24)

FIX (`GpuAcOuterLoopPacks.distrQPack`): the DISTR_Q member bookkeeping row was `columns.vCol()[m]`, the
member's own V column, while under the coupler-tree relocation a relocated child's DISTR_Q lives on its
parent coupler's `DUMMY_Q` — the move `buildDistrQContributions` already makes through `distrQRow`'s tree
overload. The row now follows the relocation. Consequence, measured with `OLF_GPU_TARGET_ROW=12566`:

```
before:  row=12566 mode=0<-1 target=0.00000000      <- DISTR_Q target updater wrote the member's V row
after:   row=12566 mode=0<-1 target=1.14200000      <- the BUS_TARGET_V target the scenario needs
```

So the device's post-contingency equation set for `BUS-6014_BBS_2` now matches OLF's: `BUS_TARGET_V` active
on both `VL-6014_0` and `VL-6014_2`, both couplers' dummies pinned, the member's DISTR_Q gone with them.

AND YET the bus difference is unchanged at 4.086e-02 kV — because the residual was never that row. The
BUSDIFF detail shows the error is NOT at `VL-6014_2` but spread network-wide, uniformly signed:

```
BUS-141_BBS  413.070204 vs 413.111063  (-4.086e-02)
BUS-856_BBS  412.302956 vs 412.332075  (-2.912e-02)
BUS-4563_BBS 241.699616 vs 241.726876  (-2.726e-02)
```

and the cause is in the same log: `distributedActivePower: worst 7.386068e+00 MW` for this single
contingency. A 7.4 MW difference in distributed slack shifts every bus slightly, in one direction — exactly
this signature. The full-window worst of 404 MW is the same defect aggregated.

So the remaining work on this window is the DISTRIBUTED SLACK, not row placement. The lead: this scenario
DISABLES a bus (`VL-6014_1`), which OLF removes from the system entirely while the device FREEZES its rows
at a flat state. A frozen bus still has load and generation in the device's active balance, so the mismatch
it distributes differs from OLF's. Check `GpuBatchQualification`'s scheduled-loss computation
(`scheduledActivePowerLoss`) against `lfContingency.getDisabledNetwork()` for a bus-disabling contingency
before touching anything else — and note the base agrees to 3e-9 MW, so this is contingency-specific.

### THE BUG: a relocated bus's TARGET stayed behind when its equation moved (2026-09-25)

The residual on this window was never a missing equation — it was a missing TARGET. The distributed slack is
applied by shifting a bus's P target, and for a relocated bus that target sits on its parent coupler's
`DUMMY_P`, where its P balance lives. The stock-placement fallback moved the EQUATION back to the bus's own
row when the coupler opens and left the target behind on a row that is now a `DUMMY_TARGET` identity, so the
bus's entire scheduled injection vanished and the slack absorbed it.

Found by discriminating instead of guessing. Same contingency, relocation off vs on:

| | slack difference | worst bus difference |
| --- | --- | --- |
| relocation OFF | 0.023 MW | **6.003 kV** |
| relocation ON (before the fix) | **7.386 MW** | 0.041 kV |

Two different defects, which is why each configuration looked "nearly right" in one metric and wrong in the
other. The scheduled loss itself agreed exactly (both 0.0 pu, `OLF_GPU_LOSS_TRACE` vs the `# OLF
activePowerLoss` line), which is what ruled out the pre-distribution and pointed at the solve.

FIX (`GpuBatchChunkSolver`, zi-outage loop): when an opened coupler is tree-relocated, the child's P target
moves from the coupler's `DUMMY_P` to the child's own PHI row, and — for a child that is NOT
voltage-controlled — its Q target moves from `DUMMY_Q` to its V row. Plus a guard that matters as much as the
fix: SKIP all of it when the child is in the island this outage created, because the freeze already holds its
rows at a flat state and these overrides are applied after it. Without the guard the window went from 3
diverging contingencies to 27 (56 of the 67 coupler-openers island their subtree).

| window [5000,6000), N=1000, relocation ON | placement work start | before this fix | after |
| --- | --- | --- | --- |
| contingencies whose violations differ | 11 | 3 | **1** |
| total violation difference | 299 | 16 | **1** |
| worst distributed active power | 808 MW | 10.8 MW | **0.887 MW** |

The isolated reproducer is now EXACT: `BUS-6014_BBS_2` goes from 7.386 MW / 4.086e-02 kV to 1.11e-09 MW /
3.40e-11 kV, violations diff 0. And the relocation is now BETTER than no relocation on this window (violation
difference 1 against 19), which it was not before.

### Hunting the next one: the base fallback is NOT the pilot-self rotation (2026-09-25)

After the target fix the window sits at 897/1000 with ONE fallback left, and it is the BASE load flow. The
gate is now named (`OLF_GPU_REJECT_TRACE=1`):

```
GPU_REJECT slot=0 reason=max-iter-bad-solve outer=6 it=30 ratio=8.139e+11 norm=2.359472e+02
GPU_BASE_ROBUST_RETRY status=-2 -> mode 6      (the robust retry then gives ratio 6.187e+06, norm 7.6e+71)
```

So the base still meets an unfactorable system at outer pass 6 — the pass where the first mode-4 rows appear
— and even the pivoted, matched single-matrix mode cannot solve it.

TESTED AND REJECTED: that this is the pilot-self degeneracy, whose (property-gated, unfinished) rotation was
built for exactly this shape. Re-running with `-Dolf.gpu.pilotSelfRotation=true` on top of today's fixes —
including the `distrQPack` member-row correction the rotation depends on — leaves the base failing at the
SAME pass with the SAME gate (`max-iter-bad-solve` at outer=6, ratio 4.203e+11 against 8.139e+11), while
adding a second fallback and taking the window to 0/1000. So the rotation neither fixes the base nor changes
its failure, and the earlier hypothesis that the base's outer=6 rank deficiency IS the pilot-self duplicated
DISTR_Q is unproven at best. Keep the property off; do not spend more on the rotation until the base failure
is understood on its own terms.

Also measured, so it is not mistaken for a relocation defect: the residual distributed-power spread (412 of
974 contingencies above 1e-9 MW, worst 0.887 MW) is PRE-EXISTING and slightly WORSE without the relocation
(525 of 974, worst 1.057 MW). It is a general GPU-vs-CPU slack difference, not something today's work
introduced.

Next target, in order of value: the base's outer=6 unfactorable system, using the tools that worked today —
`OLF_GPU_EQ_DUMP` for OLF's base equations at that state against `OLF_GPU_VALS_SCAN`'s mode census, and the
KLU rank test on an `OLF_GPU_VALS_DUMP` of that exact fill. That combination localized every defect this
session; guessing at mechanisms did not.

### The distributed-power spread, part 1: a lost account and a 0.066 MW real difference (2026-09-25)

Acting on "this much difference is a difference in equation between cpu and gpu hence a bug", the spread was
taken apart rather than filed as pre-existing. Two separate defects came out of it, and the previous entry's
"general GPU-vs-CPU slack difference" wording is retracted: one of them is a reporting bug, the other is a
real difference in what the two arms distribute.

FIRST, a method note that cost two wrong fixes. `sort -rn` does NOT understand scientific notation, so a
ranking of differences printed `9.9476e-14` above `1.05` and the real offenders never appeared at the top.
Use `sort -g`. Every conclusion drawn from such a ranking before this was worthless — that is how the earlier
claim "the worst converged difference is 9.5e-05 MW" happened.

**Defect 1 (FIXED): a fallen-back run threw away the device's distribution.** The window's base reported
`cpu=14.359916553 gpu=0.000000000 MW`, the largest single disagreement in the comparison. Traced with a new
`OLF_GPU_DS_ACCOUNT=1` on `GpuDriverOuterLoopEngine.DeviceDrivenDistributionLoop`, on the base alone:

```
check FELL BACK run=23a8deea mismatch=-0.001767686 booked=0.000000000 deviceHas=0.14294298360059193
ask numSC=0 mainSC=0 fellBack=true device=0.000000000 cpu=0.000000000
```

The device was HOLDING 14.294 MW at that very check. The loop read `lastDeviceDistributedActivePower()` only
while `deviceDroveLastSolve()`, so a run whose last solve fell back reported OLF's account alone — and OLF's
account is legitimately 0, because the device had already balanced the network (the mismatch it sees is
-0.18 MW, far inside the 1 MW `slackBusPMaxMismatch` deadband, so it correctly books nothing). The two
accounts must ADD: the device's passes moved the targets IN THE STATE, OLF's loop books only what it shifts on
top. Reading the device's amount unconditionally (it is NaN-guarded and reset per solve, so a pure-CPU replay
cannot double count) takes the base to `gpu=14.294298359`.

Three wrong guesses on the way, all from reasoning instead of measuring: that a mid-run handover was losing
the account (the no-arg `getDistributedActivePower` is never called — OLF only uses the per-SC overload at
`AcloadFlowEngine:331`); that the 10-contingency window proved the fix (it has NO fallback, so it exercised
the already-working path); that the two arms were leaking network state into each other (each arm reads its
own `Network`).

**Defect 2 (OPEN): the device distributes 0.066 MW less than OLF.** What is left on the base is
`cpu=14.359916553 gpu=14.294298359`, diff **6.56e-02 MW** — the SAME magnitude and the same direction as the
per-contingency spread (462 of the 465 significant differences share one sign; the device distributes less in
magnitude). The 15 outliers at 0.5-1.5 MW are this same difference AMPLIFIED by the 1 MW deadband: where the
two arms' mismatch straddles it, OLF fires a pass the device does not, and the device then reports the
pre-distribution alone (`recordDistributedActivePower`: `retain[s]` NaN = "no pass fired"), which is why the
worst offenders' GPU values are round numbers like -19.3 and 31.6.

So the base is now the CHEAP REPRODUCER for the whole spread: one load flow per arm instead of 1000
contingencies, printed by `OLF_GPU_SA_BASE_ONLY=1` (`BASE distributedActivePower:`), ~2 min a run.

Leading hypothesis, under test: OLF's DS loop fires only on a CONVERGED inner solve, while the device's
in-kernel pass fires on its own trajectory. A less-converged state underestimates losses, so the device
distributes slightly less, and the remainder hides under the deadband. Prediction: tightening
`OLF_GPU_DBG_CONV_EPS` on BOTH arms shrinks the 0.066 MW.

**Defect 2 ANSWERED, at the DEFAULT criteria: the device's Newton matches OLF's iteration for iteration,
and the 0.066 MW is a symptom of the base DIVERGING at outer pass 7 and being retried.**

A first answer here leaned on a tolerance sweep (tighten `newtonRaphsonConvEpsPerEq` on both arms, watch the
gap collapse to 1.1e-09 MW at 1e-12). The user rejected that, rightly and emphatically: "with the default
criteria and the default precision both cpu and gpu should agree at machine precision... you should be
FORBIDDEN to change the tolerance as you always do it and it is NOT the answer". The sweep only shows both
arms share a fixed point; it says NOTHING about why they take different paths at the settings that ship, and
reaching for it hid the defect for most of a day. The rule for this repo: a tolerance change is a diagnostic
of last resort, never an answer, and never a recommended workaround.

Measured instead, at the DEFAULT 1e-4, with `OLF_GPU_DBG_NR=1` (new: attaches an appender to OLF's own
`NewtonRaphson` + `MaxVoltageChangeStateVectorScaling` loggers) beside the device's `OLF_GPU_CONV_TRACE`:

| iterate | CPU `\|f(x)\|` | device `norm2` |
| --- | --- | --- |
| x0 | 117.84475774904828 | 1.17844757749047915e+02 |
| 0 | 5.463108973064748 | 5.46310897084124392e+00 |
| 1 | 0.22833996913869184 | 2.28339969965081657e-01 |
| 2 | 0.0024533278213928313 | 2.45332782110154910e-03 (converged) |

The two Newtons are the SAME trajectory — including OLF's `Step size: 0.9345794392523364 (3 dv ...)`
MAX_VOLTAGE_CHANGE clamp at iteration 0, which the device reproduces (its `maxVoltageChangeFactorBatch` takes
the largest |dv| and scales the whole step, algebraically OLF's `min(stepSize, maxDv/|dx|)`). And the DS pass
that fires on that trajectory reads `slackMismatchFromAnchor=0.143599166` pu — the CPU's 14.359916553 MW to
9 digits. Parity, at the default criteria, with no knob touched.

The reported 14.294298359 comes from a SECOND base solve. `[DS-CONV]` shows two complete outer sequences: the
first runs outer=0,1 then DIVERGES at outer=7 (norm2 climbing 2.72 -> 12.78 -> 24.58 -> 36.37 -> 48.22 ->
60.15, monotonic, 30 inner iterations), the base falls back (`fallbacks=1`), and the retry re-derives the
distribution from a different starting state — 2 inner iterations instead of 3, reading 0.142942984. The
0.066 MW is the distance between those two legitimate in-deadband distributions.

So this thread ends where the previous one did, on the SAME open item, now with its blast radius known: the
base's failure at outer pass 7 does not merely cost one fallback, it silently changes the reported
distributed power, and every contingency starts from that retried base state — which is the first thing to
re-measure once the divergence is fixed. Note the failure is a DIVERGING Newton (the norm grows smoothly),
not only the "unfactorable system" the earlier entry recorded from `max-iter-bad-solve`.

### The base's outer-6 divergence: pilot-self members have no Q placement and no q (2026-09-25)

Root cause of the base failure, with direct evidence at DEFAULT settings. The pass sequence
(`OLF_GPU_BATCH_TRACE=1`) against OLF's own (`OLF_GPU_DBG_NR=1`, which now also attaches an appender to
`AcloadFlowEngine`, whose "Start outer loop '<name>' iteration <n>" is the CPU counterpart):

```
device: DS(0) -> DS(1) -> HvdcFreeze(2) -> HvdcLimits(3) -> Monitoring(4) -> RL(5) -> RL(6) pinned=16 iters=30 DIVERGES
OLF:    DS(0) -> RL(1) -> RL(2) -> CONVERGED, distributedActivePower=14.35991655253323
```

The diverging pass is REACTIVE LIMITS, and the two arms pin DIFFERENT SETS: OLF's first RL pass switches **19**
buses PV->PQ, the device pins **16**. The three OLF pins the device misses are
`VL-5997_1`, `VL-6003_1`, `VL-6003_2` — and they are exactly the non-first members of two PILOT-SELF groups:

```
gvc pilot=VL-6003_0 controlled=VL-6003_0 pilotIsController=true qRowAlt=12588 members=3:
    VL-6003_0[vCol=12588 qRowAlt=12588]  VL-6003_1[vCol=12590 qRowAlt=-1]  VL-6003_2[vCol=12592 qRowAlt=-1]

row=12588 BUS_V(VL-6003_0) mode=0 terms=[cb#1921.q2@alt]        <- own reactive balance, so it CAN pin
row=12590 BUS_V(VL-6003_1) mode=1 terms=[dq[k=201 ...], ...]    <- DISTR_Q only, no own q anywhere
row=12592 BUS_V(VL-6003_2) mode=1 terms=[dq[k=207 ...], ...]    <- DISTR_Q only
```

Only the PILOT gets a Q-row placement (`qRowAlt`); every other member has `qRowAlt=-1`. So a member's row
carries DISTR_Q and nothing else, the RL check reads its q from `qBuf[row]` — the DISTR_Q residual, ~0 at
convergence, always inside the limits — and it never pins, while OLF computes the controller's own q and sees
`q=192.10 > maxQ=75.36` (and `q=229.12 > maxQ=119.27`, `q=-82.69 < minQ=0.0`). Nor is there a remote-member q
entry to supply the real value: `OLF_GPU_RMQ_TRACE=12590` prints NOTHING, i.e. `fillRemoteMemberQBatch` never
writes that row, although the context does carry the tables (`CTX_FLAGS ... nRlRemote=75 nRmq=193`, checked —
the "the base has no rmq" guess was wrong).

The device's OWN audit says the same thing without the CPU, which is the cheapest reproduction:
`OLF_GPU_RL_AUDIT=1e-6` returns entries UNPINNED and far outside their limits at the final state, e.g.
`entry=949 vrow=12560 mode=0 q=3.511 min=0.0 max=0.955 over=2.556` (that is `VL-5997_0`). OLF's outer loop
only ends after a check finds nothing to switch, so any such line is a fixpoint the device never reached.

WHY IT DIVERGES rather than merely answering differently: when the pilot pins, its row is taken by its own
pinned balance, and no member can host the group's reactive balance because none has a placement. The group's
balance is then enforced NOWHERE — the documented rank-deficiency shape — and Newton walks away smoothly
(norm2 2.72 -> 12.78 -> 24.58 -> ... -> 293 over 30 iterations), which is what a under-determined direction
looks like, not a hard problem.

This REFRAMES the earlier entry "TESTED AND REJECTED: that this is the pilot-self degeneracy". What that test
actually showed is that the pilot-self ROTATION AS IMPLEMENTED does not fix the base — which is consistent
with this root cause and explains why: the rotation moves the pilot's balance onto the first enabled member's
row, but the members have no Q placement at all (`qRowAlt=-1`) and no q for the RL check, so the rotation
addressed a later step than the one that is broken. The pilot-self SHAPE is directly implicated; keep the
rotation property off, and fix the placement + the q first.

Next: give every non-first member of a pilot-self group its own Q-row placement and an rmq entry, so (a) the
RL check sees each controller's real q and pins the same set as OLF, and (b) a pinned pilot has somewhere to
put the group's balance. Then re-measure the base (`OLF_GPU_SA_BASE_ONLY=1` prints the distributed power, and
the 0.066 MW should go with the retry it comes from).

**FIX 1 (committed): the pin set now matches OLF, 16 -> 19.** The cause was not a missing q and not a missing
rmq entry — both of those guesses were WRONG and are corrected here. `OLF_GPU_RL_TRACE_VROW=12590` shows the
device computing `q=1.921012507` pu for `VL-6003_1`, which is OLF's `192.1012507165038` MW exactly, with
correct limits `±0.7536`. (The earlier "`OLF_GPU_RMQ_TRACE` prints nothing, so the row is never filled" was an
artefact: those are device-side `printf`s that did not reach the log, not evidence of a missing fill.)

What actually dropped the member is one line in `reactiveLimitsBatch`:

```cpp
int pvForm = rlRemote[i] == 2 ? 0 : 1;   // 2 = a PILOT that is one of its own controllers
if (limitType[rst] == 0) {
    if (mode[ridx] != pvForm) return;    // member row is mode 1 (DISTR_Q), pvForm 0 -> returns before reading q
```

and every member of a pilot-self group was tagged `rlRemote=2` with its OWN row as the pilot row. The
classification asked `bus.getGeneratorVoltageControl()` — a merged group keeps ONE **MAIN** control and leaves
every other member holding a **DEPENDENT** control that regulates ITSELF, so `isRemoteMemberEntry`'s
`MergeStatus.MAIN` filter dropped the members (returning false) and `isPilotSelfEntry` then answered true for
each of them. Fixed by resolving the MAIN control (`mainGenVc`, mirroring `isRemoteGenVcControllerElement`
right beside it, which already did this) in both predicates and in the pack's own `controlledBus` / target /
nominal-V reads, plus the same resolution in `GpuBatchRlCurves` (whose comment already noted that "a pilot
whose own control is merged is reached through the main control"). The ENTRY SET is unchanged — only the class
and the pilot row — so the curve enumeration cannot drift.

Measured: the members are now `remote=1 pilotVRow=12560/12588`, and the base pins **19**, the same count OLF
switches in its first RL pass. Suite 299/0, checkstyle 214 (baseline).

**STILL OPEN: the base still diverges at outer=6 and still falls back**, with the base's distributed power
unchanged at 14.294298358 against the CPU's 14.359916553. So the pin PARITY was necessary but not sufficient,
and the remaining cause is most likely the corner the pin parity has now opened for the first time on this
network: `VL-6003`'s three members and `VL-5997`'s two now ALL pin, so each group is FULLY pinned, which is
exactly the case the kernel comments call out ("When the LAST live member pins, `updatePilotVeqBatch` flips
the pilot's V identity to a power row (control lost) and restores it on the first unpin"). Next: check the row
modes of both groups after the outer-6 pass — whether the pilot's V identity flipped, and where the group's Q
balance ended up — before reading anything else into the divergence.

### Iteration parity as the standard, and the divergence narrowed to the post-pin assembly (2026-09-25)

Standing requirement from the user, now the working invariant here: **the GPU and the CPU must agree on the
number of outer-loop iterations AND on the inner Newton iteration count of every solve.** Any difference is a
defect to find, not a tolerance to widen. Measured on the base at default settings
(`OLF_GPU_BATCH_TRACE=1` against `OLF_GPU_DBG_NR=1`):

| solve | OLF inner iterations | device inner iterations |
| --- | --- | --- |
| 1 (initial) | 3 | 3 |
| 2 (after DS distributes) | 1 | 1 |
| 3 (after RL pins) | **2, converges** | **30, diverges** |

OLF's whole base is `outerLoopIterations=3, solverIterations=7` (3+1+2+1). The device matches solve 1 and
solve 2 EXACTLY, and after FIX 1 it also pins the SAME 19 buses — verified by name, not just by count
(`comm` on the two lists: empty both ways). So the two arms are in the same state, making the same decisions,
and then the device's third solve runs away. The fault is therefore in the ASSEMBLY AFTER THE PIN, which is
the narrowest this has ever been.

What the assembly looks like at that pass (`OLF_GPU_EQ_DUMP`, which fires on the first mode-4 pass), for
`VL-5997` whose TWO controllers both pin:

```
row=12560 BUS_V(VL-5997_0) mode=0 terms=[]                 <- still a V IDENTITY
row=12562 BUS_V(VL-5997_1) mode=4 terms=[cb#8699.q2, ...]  <- hosts the PILOT's q (12560's "q2@alt" before)
```

OLF's system once every controller is pinned is: `BUS_TARGET_Q` for `_0` on 12560, `BUS_TARGET_Q` for `_1` on
12562, and `BUS_TARGET_V` INACTIVE (control lost, no enabled controller). The device keeps the V identity on
the pilot row, moves the pilot's q onto the member's row, and so the member's OWN pinned balance is enforced
nowhere — one constraint too many and one too few, which is what Newton walks away from.

The device HAS a kernel for this corner, `updatePilotVeqBatch`: when `relocChoose` finds no unpinned carrier it
flips the pilot's V identity to a power row. It never fires for these groups because they are not in the qr
candidate table (`qRowAlt=12588`, a direct row, not the `-2-group` encoding; `nQr=3`).

RETESTED AND STILL NOT THE FIX: `-Dolf.gpu.pilotSelfRotation=true`, which enrols them (`nQr` 3 -> 30,
`qRowAlt=-19`, pilot q on `qRow=12562`). The base diverges at outer=6 with the same 30 iterations, adds a
second fallback, and row 12560 is STILL `mode=0` — the V identity does not flip. This retest matters because
the earlier rejection of the rotation was run with the members MISCLASSIFIED (FIX 1), so it could have been
void; it is not. Enrolment alone does not make the all-pinned corner work for a pilot-self group.

Next: find why `updatePilotVeqBatch` leaves the pilot row at mode 0 for an enrolled pilot-self group — its
`lost` test is `relocChoose(...) == 0`, and for a pilot-self group candidate 0 IS the pilot's own row, so
"fell back to candidate 0" and "correctly chose the pilot" are indistinguishable there. That conflation is the
most likely reason the flip never happens, and it is a property of the pilot-self shape specifically.

**The all-pinned pilot-self assembly, measured row by row (`OLF_GPU_ROW_PROBE`, rotation ON).** A correction
first: the claim that `updatePilotVeqBatch` "never fires" was read off `OLF_GPU_EQ_DUMP`, which prints the
HOST-side extracted `activation.rowMode()`, not the device's live modes — it cannot see a mid-solve flip. The
probe shows the flip DOES happen:

```
pilot  row 12560: outer=0..5 target=1.103000000000e+00 F=0            (V identity, satisfied)
pilot  row 12560: outer=6    target=-2.556000000000e+00 F=1.736762045429e+00
member row 12562: outer=0..6 target=0                   F=~1.8e-08   (DISTR_Q, satisfied)
```

Two defects in that assembly, and together they are the divergence:

1. The flipped pilot target is **-2.556**, which is `maxQ - q` (0.955 - 3.511, the audit's `over=2.556`
   negated), not the pinned limit `maxQ - loadQ = 0.955`. `updatePilotVeqBatch` writes
   `target[pi] = pilotQ[fi]`, the group's captured relocated-Q target — right for a pilot that is NOT one of
   its own controllers, wrong for a pilot-self pilot that has itself PINNED, whose row must carry its own
   Q LIMIT. The row is left with F = 1.74 pu (174 MW) that no Newton step can clear.
2. The member's row keeps `target=0` through the pass that pinned it, so its own pinned Q limit is never
   enforced: that row is the pilot-Q CARRIER, and the carrier/DISTR_Q target update overwrites what
   `reactiveLimitsBatch` wrote there (`target[ridx] = rliMax - rlLoadQ[rli]`).

So both pinned controllers' Q limits are missing from the system and one unsatisfiable row is present — one
constraint too many and two too few. Newton walks away exactly as observed.

WHAT OLF DOES, and it is the simplest possible assembly: once every controller of the group is pinned there is
no relocation and no carrier at all. Each controller's own `BUS_TARGET_Q` sits on its own row with its own
limit as target, and `BUS_TARGET_V` is dropped (control lost). The device already has the per-row form for
this — mode 4 with the member's own-q ALTERNATE placement — so the fix direction is to make the all-pinned
case take that plain form for EVERY member including the pilot, instead of routing it through the carrier
machinery, whose whole purpose (finding a row for the pilot's relocated Q) is moot when no controller is free.

Rotation stays OFF by default: with it on the base still diverges and takes a SECOND fallback.

### VERIFIED: the pinned pass assembles a SINGULAR matrix, and the defect is named (2026-09-25)

The diagnosis above was checked with instruments that do not depend on it. All of the following is MEASURED,
on the base, at default settings, rotation OFF (the shipping config).

`OLF_GPU_VALS_DUMP` wrote the fill at the pinned pass (`outer=6`, `mode4=6`), and `GpuDumpedMatrixRankTest`
factored it with the CPU's own solver:

```
DUMP n=13228 nnz=93161 mode4Rows=[12560 12562 12588 12590 12592 12834]
KLU REFUSED to factor this matrix: klu_factor error KLU_SINGULAR
```

So the matrix the device assembles once the reactive-limit pass pins is **structurally singular**, which by
that test's own premise is OUR assembly's defect and cannot be rescued by any ordering, pivot type or
matching. This also sharpens the earlier record, which had only "unfactorable, ratio 8.1e+11" from
`max-iter-bad-solve`: the failure is rank, not conditioning. And the six mode-4 rows are EXACTLY the
pilot-self group rows identified from the pin-set diff, plus 12834 (entry 997, which `OLF_GPU_RL_AUDIT`
independently flagged as unpinned-but-over-limit).

`OLF_GPU_VALS_SCAN` at that same pass names the deficiency:

```
dupRows=12590,12592  deadCols=1 firstDeadCol=13185  tinyDiag=1 firstBadRow=13185
modes[0..4]=982,12240,0,0,6
```

and the equation dump identifies what 13185 is:

```
zi#25 branch=BUS-6003_COUPLER_2 b1=VL-6003_1 b2=VL-6003_2 qRow1=12590 qRow2=12592 dummyQ=13185 qAlt=[-1,-1]
```

So the two DUPLICATE rows are the two endpoints of a ZERO-IMPEDANCE COUPLER, and the DEAD column is that
coupler's own `DUMMY_Q`. This is the part the earlier description missed: these pilot-self members are the SAME
electrical node split by switches (`VL-6003_0 -COUPLER_1- VL-6003_1 -COUPLER_2- VL-6003_2`), so their two rows
are distinguishable ONLY by the dummy coupling (+dQ in one endpoint's row, -dQ in the other's). At the pinned
pass that coupling is gone from the values, so the rows collapse into duplicates AND the dummy variable is left
unconstrained — one rank deficiency each, which is why KLU refuses. The invariant the repo already documents
for this ("a DUMMY_Q needs an OPPOSING pair to be determined at all") is broken by the pin, not by the
extraction: the HOST tables still list both couplings
(`row=12590 ... zi#24.q2(-dQ), zi#25.q1(+dQ)`, `row=12592 ... zi#25.q2(-dQ)`).

RULED OUT as the mechanism: `applyRowMode`, which returns immediately unless `mode[r] == 0` and so never
touches a mode-4 row. The coupling is therefore lost during the FILL, in the per-scenario destination
selection (`ziDynDest`), not by the identity stamp.

NOT YET VERIFIED, and the next measurement rather than a thing to reason about: WHICH branch of `ziDynDest`
drops it. Its two-way form is `mode[-2-sr] == 0 ? alt[slot] : frozen`, the group form is a `relocChoose`
mode-scan, and `qAlt=[-1,-1]` on this coupler means one of those branches yields "no destination". A first
reading suggested the pinned mode-4 rows take the `frozen` branch and would KEEP the term, which contradicts
the measured dead column — so the selection must be reached another way (the `sel` row may be the pilot's, or
the group branch may apply). Trace the zi q-slot destination per slot per pass for zi#24/zi#25 before
changing anything: a wrong guess here would move a coupling that is currently correct in every other scenario.

### ROOT CAUSE, measured from the dumped matrix: a pinned member's row has no own-q placement

The dumped CSR at the pinned pass settles it without any further inference (parsed straight out of
`OLF_GPU_VALS_DUMP`'s file):

```
row 12588 (PILOT,  mode 4): 5 nonzero — (1768,-29.752) (1769,-3.635) (12588,27.090) (12589,3.635) (13183,+1.0)
row 12590 (member, mode 4): 2 nonzero — (12590,-1.0) (12592,+1.0)   F=-7.536000e-01
row 12592 (member, mode 4): 2 nonzero — (12590,-1.0) (12592,+1.0)   F=-1.192700e+00
rows 12590 and 12592 literally identical: True
column 13185: slot present in rows 12590, 12592, 13183, 13185 — every value 0.0
```

`F = -target` EXACTLY on both member rows (-0.7536 and -1.1927 are those buses' own maxQ), so no q was
evaluated on them at all: every term the host tables list for those rows — `cb#8717.q2`, `zi#24.q2(-dQ)`,
`zi#25.q1(+dQ)` — is absent from the values. What is left is a single +-1 pair on the two V columns, the same
in both rows, hence the duplicate pair; and the coupler's `DUMMY_Q` column is all zeros, hence the dead
column. The PILOT's mode-4 row, by contrast, is assembled correctly, so this is specific to NON-PILOT members.

The mechanism is the design note on `RmqPack`: "that slot is free because **the main qBuf fill suppresses
member rows**". The suppression is deliberate — it frees the member row's qBuf slot for the diagnostic own-q
that the reactive-limit check reads — but it NEVER LIFTS when the member pins and its row must host its own Q
balance. The pilot escapes because it has an own-q placement (`qRowAlt=12588`); the members have `qRowAlt=-1`.

So the very first hypothesis of this investigation was right about the PLACEMENT and wrong about the symptom.
The members' rows genuinely have no own-q placement; what was wrong was calling their q "invisible" (it is
computed correctly, 1.921012507 pu, by the rmq fill into the same row's qBuf slot). FIX 1 (the MAIN-control
classification) made those members pin for the first time, which turned a latent placement gap into a singular
matrix. Both defects are real and FIX 1 stays: without it the device silently solved a different system from
OLF's (16 pins against 19).

THE FIX: give every merged-group member an own-q placement on its OWN row, selected when that row's mode is 4,
exactly as the pilot already has. Then a pinned member's row carries `q_member - limit_member` (its
`BUS_TARGET_Q`), the coupler's +-dQ couplings land on it again (reviving column 13185), the two rows stop being
identical, and the assembly matches what OLF builds: each controller's own BUS_TARGET_Q on its own row, no
carrier, `BUS_TARGET_V` dropped. Touches `ExtractionColumns` (a member's `qRowAlt`) and the cb/zi ALT dest
tables, which today point a member's q at the PILOT's row (`cb#8699.q2@alt` on row 12560) for the relocation
case rather than back at the member's own row for the pinned case — so the alt channel needs to distinguish
those two destinations, not merely toggle.

### FIX 2: the same MAIN-control bug in the PLACEMENT predicates — the singularity is gone (2026-09-25)

The placement gap has the SAME root cause as FIX 1, in two more predicates.
`GpuAcRowIndexing.isRemoteGenVcMember` and `isFreeRemoteGenVcMember` both filtered
`getMergeStatus() == MAIN` on the bus's OWN control, so a merged member — whose own control is DEPENDENT and
regulates itself — answered "not a member". `ExtractionColumns`'s own-row alternate loop is gated on exactly
that predicate, so those members fell through it AND through the pilot-self loop beside it (whose `MAIN`
filter rejects them too) and kept `qRowAlt = -1`. Fixed by resolving the main control, exactly as
`isRemoteGenVcControllerElement` in the same file already did — and whose javadoc had already recorded this
trap in detail. One definition now: `GpuAcOuterLoopPacks.mainGenVc` delegates to `GpuAcRowIndexing.mainGenVc`.

Nothing else had to change: both the build side (`buildCbDynDest`, whose comment already reads "suppressed
frozen dest (remote member): select on the ALT row itself (mode 4)") and the runtime selection
(`ziDynDest`'s `(mode[sr] == 0 || mode[sr] == 4) ? alt : frozen`) were already written for this case and were
simply never reached.

MEASURED on the base's pinned pass, before -> after:

| | before | after |
| --- | --- | --- |
| `dupRows` | 12590,12592 | **-1,-1 (none)** |
| `deadCols` | 1 (col 13185) | **0** |
| `tinyDiag` | 1 (row 13185) | **0** |
| member row 12590 | 2 nonzero, `F = -target` | **6 nonzero incl. both coupler couplings, F = 1.167** |
| member row 12592 | identical to 12590 | **distinct, 3 nonzero incl. (13185,-1.0)** |
| KLU | **REFUSED, KLU_SINGULAR** | **factors it** |

So the rank deficiency this whole thread was chasing is FIXED: each pinned member's row now carries its own
reactive balance with its own limit as target, which is OLF's assembly. Suite 299/0, checkstyle 214.

STILL DIVERGING, for a DIFFERENT and newly visible reason. KLU factors the matrix but the solve is meaningless
(`||dx||inf=3.196391e+15`, residual ratio 1.518), and the null direction is now a set of zero-impedance DUMMY
columns (13199..13224, all mode 1, diag -1, F ~ 0). Column 13207 is touched by rows **12834 (+1.0) and 12836
(+1.0)** — two couplings of the SAME SIGN, where a `DUMMY_Q` needs an OPPOSING pair to be determined. Its
coupler is `zi#36 BUS-6112_COUPLER_1`, whose `qRows=[12836,-1]` the equation dump ALREADY flags as INCOMPLETE,
and whose bus1 (V row 12834) is another pilot-self bus whose q was relocated onto 12836. So the new member
alternate and the pilot-Q relocation now both write into that coupling, one of them with the wrong sign, and
bus2's `-dQ` is missing altogether. That interaction is the next thing to fix, and the existing
INCOMPLETE-coupling section of `GpuEquationDump` plus a sign check on each coupler's pair is the instrument
for it.

**The remaining defect, named: the RL pin writes mode 4 on the entry's OWN row even when that bus's Q balance
is RELOCATED.** Measured placement for the group behind the same-signed pair (post-FIX 2):

```
gvc pilot=VL-6112_0 vCol=12834 qRow=12836 qRowAlt=-1 members=2:
    VL-6112_0[vCol=12834 qRow=12836 qRowAlt=-1 enabled=false]   VL-6112_1[vCol=12836 qRow=-1 qRowAlt=-1 enabled=true]
zi#36 BUS-6112_COUPLER_1 b1=VL-6112_0 b2=VL-6112_1 qRow1=12836 qRow2=-1 dummyQ=13207
```

This pilot is ALREADY PINNED at extraction (`enabled=false`, its `BUS_TARGET_Q` active), so `computeQRow`
relocates its Q balance to the free member's row (12836) and its own row 12834 keeps `BUS_TARGET_V` — which a
sibling controller still needs. That placement is CORRECT and it is why FIX 2's two loops both skip this group
(the pilot has `qRow >= 0`, the member is the free one). The pilot-self pinned pilot cannot host both
`BUS_TARGET_V` and `BUS_TARGET_Q` on one row, and relocating the Q is the device's answer.

What is wrong is the ROW MODE. `reactiveLimitsBatch` pins by writing `mode[ridx] = 4` with
`ridx = s*n + rlVrow[i]`, and `rlVrow` is "the transition WRITE row: the member's own bus row". For a bus whose
Q balance is relocated to a carrier, mode 4 must go on the CARRIER's row (12836) while its own row stays the
mode-0 `BUS_TARGET_V` identity. Today row 12834 is told "host your own Q" while that q is emitted at 12836, so
the coupler's `+dQ` appears at BOTH rows (measured: column 13207 touched by 12834 +1.0 and 12836 +1.0, no
opposing `-dQ`) and the `DUMMY_Q` is left underdetermined — `||dx||inf = 3.2e+15` on an otherwise factorable
matrix.

So the fix is to make the pin's write row follow the RELOCATION, not the bus: `rlVrow` (or a second per-entry
"pin row") must be `qRow[bus]` when that differs from `vCol[bus]`. Note this is the same quantity
`GpuAcOuterLoopPacks` already computes for the DISTR_Q member table, where the comment records the identical
lesson for the coupler-tree case ("The member's DISTR_Q row, NOT simply its V column ... Pointing this
bookkeeping at the V row made the DISTR_Q target updater write there"). Check both `rlVrow` uses (the pin write
and the `q = qBuf[row] + loadQ` read) before changing either: the READ must stay on the row that carries the
bus's own q, which for a relocated bus is the carrier, so the two may already need to differ from `vCol`.

**WHAT NEEDS TO CHANGE, from the tables themselves.** The free member's row, as the host tables list it:

```
row=12836 BUS_V(VL-6112_1) mode=4 terms=[cb#8783.q1, cb#8925.q2, zi#36.q1(+dQ),
                                         dq[k=271 elem=13207 side=0 kind=3 w=-1.00000],
                                         dq[k=274 elem=13207 side=1 kind=3 w=-0.00000]]
```

`zi#36.q1(+dQ)` is bus1's (the PILOT's) coupling and its frozen dest is 12836, because `qRow1 = 12836` — the
pilot's balance rides the carrier. When the pilot's OWN row (12834) goes mode 4, the same term ALSO resolves
there through the alternate, so column 13207 gets +1.0 twice with no opposing -dQ. The kind-3 `dq` term on the
same column (weight -1.0) is correctly gated off by mode 4, so that gate is NOT the problem — it works.

The emission must happen EXACTLY ONCE, at whichever row hosts that bus's Q balance this scenario, and the
two-way channel can already express that. It simply keys on the wrong row:

```cpp
selRow[o] = stockSel <= -2 ? stockSel : (rc.row >= 0 ? rc.row : altRow);   // frozen exists -> select on the CARRIER
```

For a bus whose alternate is its OWN row and whose frozen dest is a DIFFERENT row (a relocation carrier), the
select row must be the OWN row: mode 4 -> alt (own row), else -> frozen (carrier). One emission either way.
The `sel = altRow` branch already does exactly this when the frozen dest is SUPPRESSED (-1); it needs to
extend to the relocated case. What blocks a one-line change is that the build must know `altRow` IS the bus's
own V row, which the cb/zi index tables do not carry today (they carry the frozen q dest and `qRowAlt`, not
the bus's V column for that side) — so this needs either a new per-side field or an explicit sel shipped from
Java, which is where the next change starts.

Two method notes worth keeping. (1) `GpuEquationDump`'s `gvc` line recomputes `ExtractionColumns.of(network,
es)` AT DUMP TIME, so its `qRow`/`qRowAlt` are NOT the values the running structure uses — reading a pinned-pass
dump as if they were sent this investigation after a placement that was never in play. The dumped MATRIX is
the ground truth. (2) `mode4Rows` differs between the first pinned fill and the end of the run
(`[12560 12562 12588 12590 12592 12834]` then `[12562 12588 12590 12592 12836]`), so always state WHICH fill a
mode observation comes from.

**What the two `rlVrow` uses need, and three refuted explanations for the same-signed pair (2026-09-25).**

The two uses genuinely need DIFFERENT rows, so they must not be moved together:
- the q READ (`q = qBuf[rlVrow] + loadQ`) must stay the bus's OWN row, because that is where
  `remoteMemberQPack` writes the diagnostic own-q (`int row = vCol[bus.getNum()]`);
- the pin WRITE (`mode[ridx] = 4`, `target[ridx] = limit`) must be the row that HOSTS that bus's Q balance,
  which for a relocated bus is the carrier.

`relocatedPinCode` already freezes an entry whose pin lives elsewhere, and it is not firing here for a good
reason: at the extraction that matters the pilot is UNPINNED (`qRow = -1`), so nothing is relocated yet and
there is nothing to freeze. That path is correct.

The decisive new fact, from `OLF_GPU_DBG_ROW=12834` — the packer's own inputs, not a recomputation — is that
the base is extracted TWICE with DIFFERENT placements:

```
#1  b1=VL-6112_0(vCol=12834 qRow=12836 qRowAlt=-1)   b2=VL-6112_1(vCol=12836 qRow=-1 qRowAlt=-1)
#2  b1=VL-6112_0(vCol=12834 qRow=-1 qRowAlt=12834)   b2=VL-6112_1(vCol=12836 qRow=-1 qRowAlt=12836)
```

Each is INTERNALLY CONSISTENT: #1 (pilot pinned at extraction) emits the pilot's `+dQ` at the carrier 12836;
#2 (pilot unpinned, both loops of FIX 2 firing) emits it at 12834 when that row goes mode 4, and correctly
emits no `-dQ` because the free member has no Q equation in OLF either. Neither placement alone can put +1.0
on BOTH rows, which is what the dumped matrix shows.

REFUTED, each by reading the mechanism rather than guessing:
1. "The cached structure aliased a different placement." `GpuAcData.sameStructureInputsAs` compares `cbIdx`,
   `ziIdx`, `dqRow` and the rest — all of which embed `qRow`/`qRowAlt` — so two placements cannot content-match.
2. "A pattern slot kept a stale value after the destination moved." The value array is memset on EVERY full
   fill (`fillActiveBatch(false)`).
3. "Row mode 4 triggers F-only fills that reuse a stale J across the pin." `fOnly = it == maxIter ||
   (ctx->mode == 4 && it > 0)` is the cuDSS SOLVE STRATEGY (line 6159: 0 per-scenario, 1 batched, 2
   block-diagonal, 3 UBATCH, 6 robust), where 4 is the chord strategy factorizing at iteration 0 only. It has
   nothing to do with row mode 4.

So the next measurement is to establish WHICH extraction produced the dumped matrix, which the dump does not
currently say: add the structure identity (or the packer's `qRow`/`qRowAlt` for the traced rows) to the
`VALS_DUMP` header, then re-read column 13207 with that known. Until that is pinned down, any change to the
selection would be aimed at a placement that may not be the one in force.
