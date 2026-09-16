# Task list (recovered 2026-07-21)

The task tracker (an MCP server) disconnected mid-session, so this file is the durable
record. It was reconstructed from the 2026-07-21 session transcript: every entry below is
backed by a commit, a measurement, or an explicit decision recorded in that session.

**Fidelity boundary.** Only tasks referenced during that session are here. Numbers below
#24 from earlier sessions are NOT recoverable from it (except #18, which came up
explicitly) — if the tracker ever comes back, reconcile rather than assume this file is
complete. Keep the numbering: it is cited in commit messages.

## Open

### #54 — A second SA run in one JVM re-pays the cuDSS ANALYSIS (found 2026-09-16)

Measured, not suspected. `OLF_GPU_TRACE=1` now tags every ANALYSIS with the context's batch width, and
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
one ANALYSIS. The base load flow pays its own, at width 1.

The third eviction is the finding: `chunk=250 -> chunk=250`, the same width and the same LU_NNZ, so the
PATTERN is identical and the symbolic analysis would have been reusable. `acquireBatchContext` misses
because it keys on `GpuAcData.sameAs`, which compares element PARAMETER VALUES (`cbIn` and friends) —
and the previous run left the network at different tap positions. The base load flow already has the
right pattern for this: `sameStructureInputsAs` + `refreshStructureValues`, reuse the device structure
and push the new values.

What makes it more than a one-line change: `createBatchContext` takes the reactive-limit and
distributed-slack packs as CONSTRUCTOR arguments (`rlMinQ`, `rlVtarget`, `dsFactor`, …) and
`GpuBatchOuterLoopTables.install` does not re-upload them, so reusing a context across runs needs
those packs to become setters like the channels installed beside them. Size it against the measured
ANALYSIS cost first — the cold/warm delta on the bench is ~1.4 s over 500 contingencies and the
analysis is only part of that.

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
