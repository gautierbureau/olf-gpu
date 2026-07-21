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
4. **Non-uniform batching — DECIDED, and it is required.** Training runs on **different
   networks**, so the uniform UBATCH the entire SA path rests on (one shared sparsity pattern,
   B sets of values) **does not apply to a training batch**. Options from the pyloadflow tex
   note: (3) non-uniform list of shapes, or (4) **block-diagonal / fused** — glue the B grids
   into one giant block-diagonal matrix, one analysis + one factorization; the blocks are
   independent so there is no fill between them, and (usefully) the transpose of a
   block-diagonal is block-diagonal with each block transposed, so the orientation fix below
   composes unchanged. Caveat for (4): the symbolic analysis is per batch *composition*, so it
   only amortizes if membership is stable across steps — with reshuffled minibatches every
   step, re-analysis could dominate. Measure before committing.
   Note consumers 2 and 3 (SVC, incremental phase control) run INSIDE the SA batch where the
   pattern IS shared, so they never hit this — another reason to do them first.
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

**Acceptance gate, inherited from the CPU work:** three-way agreement — finite difference
(independent ground truth) vs forward sensitivity vs the adjoint. **The orientation mismatch
makes this gate non-optional**: an inverted convention produces a plausible-looking but wrong
gradient, and FD is what catches it. The two analytic paths
should agree to solver tolerance (same kernels, opposite contraction order); FD agrees to
truncation error and catches a systematic error common to both. `AcSensitivityAnalysisAdjointTest`
already encodes this pattern; mirror it rather than inventing a new gate.

**Sequencing note:** consumers 2 and 3 are self-contained inside our batch and would prove
the primitive on ground we control, before the JAX/pypowsybl context plumbing of consumer 1.

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
