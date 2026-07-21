# Task list (recovered 2026-07-21)

The task tracker (an MCP server) disconnected mid-session, so this file is the durable
record. It was reconstructed from the 2026-07-21 session transcript: every entry below is
backed by a commit, a measurement, or an explicit decision recorded in that session.

**Fidelity boundary.** Only tasks referenced during that session are here. Numbers below
#24 from earlier sessions are NOT recoverable from it (except #18, which came up
explicitly) — if the tracker ever comes back, reconcile rather than assume this file is
complete. Keep the numbering: it is cited in commit messages.

## Open

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

**Consumer 2 — SecondaryVoltageControl, batched.** Its sensitivity RHS is built on existing
`BUS_TARGET_V` rows and solved with `j.solveTransposed` (validated on plain LF 2026-07-21,
`87f61f1`; batched port still ⏩).

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

**Prerequisite — how the CPU adjoint reaches our build. SETTLED 2026-07-21: rebase, then
cherry-pick/squash.** The GPU extension builds against OLF `2.4.0-SNAPSHOT` from
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
