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
| SecondaryVc | Needs transposed ❌ | OLF Q-terms (CPU) | Task #40 + Java eval | **Blocked** |

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

### SecondaryVoltageControl — full in-kernel scoping

A full SVC kernel port is **blocked on task #40** (batched transposed solve). The five
stages: (1) trigger check — easy, needs device-side Q per bus; (2) re-enable — easy,
reads/writes RL state on device; (3) sensitivity — BLOCKED, needs `J^T λ = e_i` which
cuDSS 0.8.0 cannot do; (4) zone system — small dense LU, better on host; (5) apply —
already device-native.

Phased approach: Phase 1 = `LOOP_HOST_SVC` pause/resume (now); Phase 2 = stages 1-2 as
kernels + device transposed solve (after task #40); Phase 3 = optional batch small systems.

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
table). Scope unchanged: no generator voltage control under the transformer-controlled level
(`transformerControlSupported` — the GeneratorVoltageControlManager freeze is still not ported).

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
