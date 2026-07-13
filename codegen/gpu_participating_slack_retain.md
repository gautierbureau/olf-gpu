# GPU batched SA — participating-slack "retain" and its bounded error

Design note on the one place the device batched security-analysis solver cannot
reproduce OLF's distributed-slack outer loop *exactly*: when the **slack/reference
bus's own generators participate** in the distribution. Companion docs:
`codegen/gpu_assembly.md` (CPU/GPU split), `codegen/gpu_roadmap.md` (rung journal),
`codegen/gpu_rte_full_support.md` (RTE parameter coverage). Status: 2026-07-02,
branch `claude/level-a-closed-branch-formulas`, characterized in commit `ec7b5751`.

Code: `native/cuda/jni/olf_gpu_nr_jni.cu` (`computeSlackRetain`,
`distributeSlackWaterfillBatch`), `GpuAcDataExtractor` (the `dsFactor` normalization).
Characterization test: `GpuBatchedParticipatingSlackConvergenceTest`.

---

## TL;DR

- For a **non-participating** slack the device runs the *same* distributed-slack loop
  as OLF and matches the CPU to **machine precision** (`GpuSecurityAnalysisComparisonTest
  #cpuVsGpuSecurityAnalysisBranchN1StatesMatch` — ~1e-14 at the **loose 1-MW** production
  DS tolerance).
- For a **participating** slack the device uses a **"retain" trick** that carries a
  **loss linearization**: it converges to a fixed point a fixed distance from OLF's,
  measured at **~4.4e-7** (tolerance-independent). It is **NOT** a convergence artifact
  and does **NOT** shrink with a tighter tolerance.
- The error is **small and bounded** (~1e-7, the same order as the PEGASE tight-tol
  residual) — well inside any production tolerance and below limit-check thresholds, so
  SA **verdicts are unaffected**.
- The "clean" structural fix (give the reference bus a real P-equation) is **not worth
  it** (see [§4](#4-why-not-just-add-a-p-equation-at-the-reference-bus)). The issue is
  documented, characterized, and guarded, not fixed.

---

## 1. Why the slack is special in the fixed CSR pattern

The batched solver holds **one fixed CSR sparsity pattern** shared by every scenario, so
cuDSS analyzes it once and reuses the factorization. In that pattern the reference bus has:

- `BUS_TARGET_PHI` active (θ = 0, the angle reference), and
- **`BUS_TARGET_P` inactive** — its active power is *free*. That freedom is exactly what
  makes it the reference: it absorbs whatever active imbalance the rest of the network
  cannot.

So there is **no active-power equation row at the reference bus**. Every other
participating bus has a `BUS_TARGET_P` row that the distributed-slack loop retargets
(`target += factor · mismatch`), but the slack has none to retarget.

## 2. What OLF does with a participating slack

OLF's `DistributedSlackOuterLoop` / `ActivePowerDistribution`, `PROPORTIONAL_TO_GENERATION_P_MAX`
(factor ∝ `maxP/droop`): it redistributes the slack-bus mismatch over **all** participating
generators **including the slack bus's own generators**, by modifying their generator target
P and re-solving. The slack's generators absorb their share `f_slack · T` of the total
imbalance `T`; the slack bus's residual mismatch converges below `slackBusPMaxMismatch`.

## 3. What the GPU does — the "retain" trick, and its floor

The device cannot retarget the slack's generators (no P row). Instead it **keeps the slack's
participation share undistributed at the slack** and distributes only the rest:

```
retain   = f_slack · (slackMismatch + absorbed)              # computeSlackRetain, re-evaluated each pass
rem      = slackMismatch − retain                            # distributeSlackWaterfillBatch
if |rem| ≤ slackDistTol: stop                                # the DS gate, matched to OLF's
else: water-fill rem over the NON-slack participating buses ∝ their factors
```

where `f_slack = 1 − Σ(non-slack normalized factors)` (the slack's own normalized share,
folded into the `dsFactor` denominator by the extractor so the non-slack shares shrink to
`f_i/Σ_all`), and `absorbed = Σ_non-slack (target − baseTarget)` is what has been distributed
so far. `computeSlackRetain` **re-evaluates `retain` every outer pass** from the *current*
state — the earlier "frozen at pass 0" bug (`freezeSlackRetain`) is gone.

**The residual — a loss linearization.** At convergence `rem → 0`, so
`slackMismatch = f_slack · (slackMismatch + absorbed)`. But `slackMismatch + absorbed` equals
the total imbalance seen at the *current* operating point, `M0 − Δloss` (redistributing
generation off the slack changes the AC losses), whereas the share OLF's slack generators end
up absorbing corresponds to `M0 + Δloss`. The retain formula therefore lands
`~2·f_slack·Δloss` off — a **second-order AC-loss term**. It is a genuine fixed-point offset,
not a stopping-criterion slack.

**Measured** (`GpuBatchedParticipatingSlackConvergenceTest`, a 3-bus net whose slack gen
participates, GPU vs CPU `LoadFlow.run` at the *same* DS tolerance, tight inner Newton both):

| `slackBusPMaxMismatch` | worst \|gpu − cpu\| | note |
|---|---|---|
| 1 MW | 5.6e-17 | mismatch below the gate → **no distribution**, retain not exercised |
| 1e-2 MW | 7.3e-17 | idem |
| 1e-4 MW | **4.44e-7** | distribution triggers → floor |
| 1e-6 MW | 4.44e-7 | **tolerance-independent** |
| 1e-8 MW | 4.44e-7 | ″ |
| 1e-10 MW | 4.44e-7 | ″ |

Tightening the tolerance does **not** drive it to zero — it plateaus. That is the signature
of a linearization, not a convergence artifact.

**Interaction with the DS tolerance and reactive limits.** On a network with a *large*
redistribution and the loose 1-MW gate, the DS loop stops before the retain iteration reaches
its floor, so the *observed* error can be larger than the floor — e.g. IEEE14 losing L1-2-1
with a participating slack shows ~1.4e-3 in the SA state comparison. That is the floor
**amplified** by the loose gate (and the RL↔DS coupling), not a second bug; IEEE14 cannot be
run at a tight DS tol because the RL+DS loops oscillate below ~1e-4 MW. On PEGASE-9241 (tight
DS tol) the participating-slack residual sits at ~1.5e-7 — right at the floor.

## 4. Why not just "add a P-equation at the reference bus"?

Because a bus's P is free **precisely because** its angle is the θ = 0 reference. You cannot
add a `BUS_TARGET_P` balance there without also giving it a free angle — i.e. **moving the
θ = 0 reference to another (non-participating) bus** and letting the old slack participate
with a normal P row.

- It is **not a new sparsity pattern** — it is a *row swap* (which bus carries the identity-θ
  row vs the assembled P-balance row); the nnz structure and the cached cuDSS analysis are
  essentially unchanged.
- But it **shifts every absolute angle** by the new reference's offset and **diverges the
  device's reference from OLF's chosen slack**, and it needs a genuinely non-participating bus
  to exist as the reference.
- The prize is a **~4e-7** floor already 4+ orders inside production tolerance.

Not worth it. If sub-1e-6 participating-slack accuracy ever becomes a hard requirement, the
right fix is a **correction to the retain's loss term** (add the `Δloss` the formula omits),
**not** the reference move — but it is loss-state-dependent (needs the converged `Δloss`,
which is circular → a fixed-point iteration) and delicate to land without regressing the
currently-exact cases. Until then: characterized and bounded.

## 5. Guardrails in the suite

- `GpuBatchedParticipatingSlackConvergenceTest` — asserts the floor stays **small and
  bounded** (`< 1e-6`) and prints the tolerance sweep (this document's table).
- `GpuSecurityAnalysisComparisonTest#cpuVsGpuSecurityAnalysisBranchN1StatesMatch` — the
  apples-to-apples GPU-SA-vs-CPU-SA state comparison; runs a **non-participating** slack on
  purpose (retain = 0) so it holds at machine precision (~1e-14) and isolates this variable.
- `GpuBatchedDistributedSlackSaturationTest` — participating-slack case at `agreeTol = 1e-8`
  (the retain floor bounds it).
