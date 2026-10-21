# GPU track — full OLF support under `rte-loadflow-default-parameters.json`

Goal: run a plain `LoadFlow.run` on **real RTE networks with the RTE default
parameters** entirely on the GPU solver (`GPU_NEWTON_RAPHSON`), converging to the
same state OLF's CPU solver reaches, with **no extractor rejection and no silent
loss of an outer-loop effect**. This document is the live tracker for closing that
gap. Companion docs: `docs/archive/gpu_roadmap.md` (the rung journal + Priority 2/3/4),
`codegen/README.md` (equation codegen), `docs/archive/gpu_participating_slack_retain.md`
(the one bounded ~4e-7 device-vs-OLF gap: participating-slack distribution). Status
date: 2026-06-16, branch `claude/level-a-closed-branch-formulas`.

---

## 0. How to read this against the codebase

The GPU is **only the inner AC Newton-Raphson** (`GpuNewtonRaphson extends
AbstractAcSolver`). OLF's `AcLoadFlowEngine` runs **every outer loop on the CPU** and
re-invokes `GpuNewtonRaphson.run()` once per outer iteration
(`GpuNewtonRaphson.java:144`). Between invocations, event-driven invalidation
(`Invalidation`, an `EquationSystemIndexListener` + `LfNetworkListener`) sets
`structureDirty` (full re-extract + validate, rebuild cuDSS context if the pattern
really changed) or `activationDirty` (refresh `rowMode`/`targetMap` only). Targets are
re-read from the `TargetVector` on every solve, so target-only changes need nothing.

**An outer loop "works on GPU" iff the equation system it produces stays inside
`GpuAcDataExtractor`'s allowlist.** The hard boundary, verbatim from the extractor:

| Check | Location | Throws on |
|---|---|---|
| variable type | `GpuAcDataExtractor.java:191` | anything other than `BUS_V` / `BUS_PHI` |
| equation type | `:177`, `:238` | anything other than `BUS_TARGET_{P,Q,V,PHI}` |
| equation-array type | `:200` | anything other than `BUS_TARGET_P/Q` arrays |
| term (array + scalar) | `:207,:214,:225,:233` | terms outside the closed/open/shunt/hvdc allowlist |
| runtime disconnection | `:243` | a branch with `isDisconnectionAllowedSide{1,2}` |

The Newton **step** itself is undamped (`solveContext(... maxIter, GPU_TOLERANCE)`):
no `StateVectorScaling` is applied on the device.

---

## 1. The target parameter set

`rte-loadflow-default-parameters.json` (verbatim, the source of truth for this doc):

```jsonc
voltageInitMode            : DC_VALUES
useReactiveLimits          : true
twtSplitShuntAdmittance    : true
componentMode              : MAIN_SYNCHRONOUS
hvdcAcEmulation            : true
// open-load-flow-parameters:
slackDistributionFailureBehavior        : FAIL
maxOuterLoopIterations                  : 30
transformerVoltageControlMode           : AFTER_GENERATOR_VOLTAGE_CONTROL
maxNewtonRaphsonIterations              : 30
stateVectorScalingMode                  : MAX_VOLTAGE_CHANGE
maxVoltageChangeStateVectorScalingMaxDv : 0.4
maxVoltageChangeStateVectorScalingMaxDphi : 1
transformerVoltageControlUseInitialTapPosition : true
generatorVoltageControlMinNominalVoltage : 120
fictitiousGeneratorVoltageControlCheckMode : FORCED
minRealisticVoltage : 0.8 ; maxRealisticVoltage : 1.2
minNominalVoltageRealisticVoltageCheck : 180
disableInconsistentVoltageControls : true
extrapolateReactiveLimits : true
```

Plus **two features implicit in real RTE networks** that these params do **not**
disable (the benchmarks forcibly disabled them — see §3): **remote/secondary voltage
control** and **zero-impedance (very-low-Z) branches**.

---

## 2. Coverage matrix (parameter → GPU status)

Legend: ✅ works today · 🟡 works but unverified for this combo · ❌ extractor throws
or effect silently lost · ➖ CPU-side, GPU-irrelevant.

| Parameter / feature | Equation-system effect | GPU status | Closes via |
|---|---|---|---|
| `voltageInitMode=DC_VALUES` | seeds `x0` via a DC solve before the AC solve | ➖ | verify DC step never routes to GPU |
| `useReactiveLimits=true` | PV→PQ switching outer loop | ✅ | — (activation `rowMode`) |
| `extrapolateReactiveLimits=true` | Q-limit value detail in PV→PQ | ✅ | — |
| `twtSplitShuntAdmittance=true` | g1/b1/g2/b2 split in the π-model values | 🟡 | verify extracted branch values |
| `componentMode=MAIN_SYNCHRONOUS` | component selection | ➖ | — |
| `hvdcAcEmulation=true` | HVDC AC-emulation terms | ✅ | verify interplay w/ outer loops |
| `slackDistributionFailureBehavior=FAIL` + distributed slack | P-target adjust outer loop | ✅ | — (targets re-read each solve) |
| `maxNewtonRaphsonIterations`, `maxOuterLoopIterations` | iteration caps | ✅ / ➖ | — |
| `generatorVoltageControlMinNominalVoltage=120` | which gens hold V → which buses are PV | ✅ | — (more PV rows, `rowMode`) |
| `fictitiousGeneratorVoltageControlCheckMode=FORCED` | fictitious-gen V control setup | 🟡 | verify resulting es stays in scope |
| `disableInconsistentVoltageControls=true` | drops inconsistent controls CPU-side | ➖ | — |
| `min/maxRealisticVoltage`, `minNominalVoltage…Check` | post-solve realistic-V check | 🟡 | **W5** — confirm check + `UNREALISTIC_STATE` |
| **`stateVectorScalingMode=MAX_VOLTAGE_CHANGE`** (+maxDv/maxDphi) | **damps the Newton step** | ❌ | **W1** device-side scaling |
| **`transformerVoltageControlMode=AFTER_GENERATOR_VOLTAGE_CONTROL`** + `useInitialTapPosition` | **`BRANCH_RHO1` var + `DISTR_RHO`/V-controlled-by-twt eq** | ❌ | **W2** transformer voltage control |
| **(implicit) remote / secondary voltage control** | **`DISTR_Q` equations, shared-control vars** | ❌ | **W3** remote voltage control |
| **(implicit) zero-impedance branches** | **`ZERO_V`/`ZERO_PHI` equations** | ❌ | **W4** zero-impedance branches |

**Bottom line:** four hard gaps (W1–W4) plus a verification sweep (W5/W6). W1 is a
correctness gap *even on networks that don't trip the extractor*, because the params
explicitly request step damping the GPU isn't doing — on stressed cases the CPU
(damped) and GPU (undamped) runs can diverge.

---

## 3. Why the benchmarks don't already prove this

`GpuBenchmarkTest` / `GpuBatchedN1BenchmarkTest` force the network INTO current scope:
`REPLACE_BY_MIN_IMPEDANCE_LINE` (kills zero-impedance → avoids W4) and **local-only
voltage control** with `maxSlackBusCount=1` (avoids `DISTR_Q` → avoids W3), and they
do not enable transformer voltage control (avoids W2) or non-trivial scaling (avoids
W1). RTE production parameters do the opposite of all four. So "validated on PEGASE"
means *validated in the reduced scope*; this doc tracks lifting each restriction.

---

## 4. Foundation (shared by W2–W4)

W2–W4 all break the same two assumptions, so build the foundation once:

- **F1 — variables beyond `BUS_V`/`BUS_PHI`.** The fixed-pattern CSR, diagonal pairing
  (P↔`BUS_PHI`, Q↔`BUS_V`), `rowMode`, `targetMap` and the four fill kernels all assume
  exactly two variables per bus and no others. Transformer ρ (`BRANCH_RHO1`), shunt b
  (`SHUNT_B`) and the distribution variables add **non-bus rows/cols** with their own
  diagonal-pairing and Jacobian columns. Generalize the row-indexing + diagonal-pairing
  contract and the CSR builder to admit a typed set of extra variables.
- **F2 — equations beyond `BUS_TARGET_{P,Q,V,PHI}`.** `DISTR_*`, `ZERO_*` and the
  transformer/shunt target equations need fill code and scatter rows. These are new
  fused-kernel families OR a generic sparse-term path; decide per equation (most are
  small/linear — a generic linear-term filler may beat a codegen family).
- **F3 — invalidation level for these.** Adding/removing a control mid-run is a
  STRUCTURE change (re-extract + cuDSS re-analyze). Confirm the `Invalidation`
  classification routes each new event correctly (compile-time exhaustiveness already
  forces a decision on new OLF events).

Until F1/F2 land, W2–W4 stay ❌. W1 and W5/W6 are independent of the foundation.

---

## 5. Workstreams (ordered)

### W1 — State-vector scaling `MAX_VOLTAGE_CHANGE` on device  *(no foundation needed)*
- **What OLF does:** after each Newton step, `MaxVoltageChangeStateVectorScaling` caps
  the step so no |Δv| > `maxDv` (0.4) and no |Δφ| > `maxDphi` (1 rad), scaling the whole
  `dx` by the most-violating ratio. Cheaper than `LINE_SEARCH` (no extra residual eval).
- **GPU gap:** `solveContext` applies `x += dx` undamped. Add the clamp in the device
  iteration: after the cuDSS solve, a reduction finds `max(|dv|/maxDv, |dphi|/maxDphi)`
  over the step, and a scaled `x += dx/max(1,ratio)`. Pass `maxDv`/`maxDphi` through the
  JNI solve params; map OLF's `stateVectorScalingMode` → an enum arg.
- **Why first:** self-contained, no equation-system change, and it's a *correctness*
  fix for the explicit param — without it GPU≠CPU on stressed networks.
- **Measured evidence (2026-06, RTE A1000).** Two CPU experiments scoped this:
  - `GpuVsCpuFixedIterTest` confirms the device Newton matches OLF's undamped CPU Newton
    **bit-for-bit per iteration** (ratio 1.000) — so the slow early convergence is the
    formulation, not the GPU.
  - `W1DampingCpuTest` measures whether `MAX_VOLTAGE_CHANGE` (RTE: maxDv=0.4, maxDphi=1)
    actually cuts iterations vs `NONE`, on PEGASE-9241: **base case 5→3 iters** (damping
    avoids the cold-start overshoot), but on **47 DC-warm-started N-1 contingencies it is a
    WASH** (mean 3.60 vs 3.62; fewer iters in only 1 of 46; same 46/47 converged).
  - **Conclusion:** W1 is **robustness + production parity** (RTE enables it, and it helps
    hard/cold starts), **NOT a throughput win for batched N-1** — the warm-started
    contingencies already converge in ~3-4 undamped iterations. So prioritize W1 for
    *correctness/scope* (matching the RTE solver), not to speed up the N-1 payoff.
- **Validation:** a stressed network where CPU needs scaling to converge; assert GPU
  iteration trajectory matches CPU within tol and converges. Add `LINE_SEARCH` later if
  any RTE profile uses it (RTE default is `MAX_VOLTAGE_CHANGE`).
- [ ] JNI param plumbing (mode, maxDv, maxDphi) · [ ] device clamp + reduction ·
  [ ] mode enum mapping · [ ] stressed-case test · [ ] EquationsTest/regression green

### W2 — Transformer voltage control  *(needs F1+F2)*  — explicitly enabled by the params
- **What OLF does:** `transformerVoltageControlMode=AFTER_GENERATOR_VOLTAGE_CONTROL`
  makes a controlling transformer's ratio ρ a variable (`BRANCH_RHO1`), with a target
  equation holding the controlled bus V and, for groups, a `DISTR_RHO` sharing equation.
  `useInitialTapPosition=true` seeds ρ from the tap.
- **GPU gap:** `BRANCH_RHO1` variable (F1) + the V-by-transformer / `DISTR_RHO`
  equations (F2). The closed-branch fill kernels already compute `da1`/`dr1` derivatives
  (used in the scalar formulas) — the Jacobian columns for ρ are derivable from the same
  residual source; this is where the codegen pays off.
- **Validation:** a network with a voltage-controlling transformer; GPU state == CPU.
- [ ] F1 ρ-variable rows · [ ] V-by-twt + DISTR_RHO fill · [ ] ρ Jacobian columns ·
  [ ] discrete tap rounding outer loop interplay · [ ] test

### W3 — Remote / secondary voltage control (`DISTR_Q`)  *(needs F1+F2)*  — implicit in RTE nets
- **What OLF does:** multiple generators controlling one (possibly remote) bus share the
  reactive burden via `DISTR_Q` equations; secondary voltage control adds pilot-bus
  coupling.
- **GPU gap:** `DISTR_Q` equations (F2) + any shared-control coupling vars (F1). Today
  the benchmark sidesteps this with local-only control + `maxSlackBusCount=1`.
- **Validation:** multi-generator remote-control network; GPU state == CPU.
- [ ] F2 DISTR_Q fill + scatter · [ ] coupling columns · [ ] PV→PQ interplay w/ shared Q ·
  [ ] test

### W4 — Zero-impedance branches (`ZERO_V`/`ZERO_PHI`)  *(needs F2)*  — implicit in RTE nets
- **What OLF does:** very-low-impedance branches become exact `v1=v2`/`φ1=φ2` constraint
  equations instead of power-flow terms (avoids ill-conditioning).
- **GPU gap:** `ZERO_V`/`ZERO_PHI` equation fill (F2). They're linear identity-style
  rows; a generic linear-row filler may be simpler than a codegen family. Removes the
  `REPLACE_BY_MIN_IMPEDANCE_LINE` requirement.
- **Validation:** a network with genuine zero-impedance branches (not min-impedance
  replaced); GPU state == CPU, and condition number stays sane.
- [ ] F2 ZERO_* fill + scatter rows · [ ] CSR pattern incl. constraint rows ·
  [ ] drop the min-impedance forcing in a dedicated test · [ ] test

### W5 — Realistic-voltage check & status semantics  *(no foundation needed)*
- **What OLF does:** `min/maxRealisticVoltage` (0.8/1.2 above `minNominalVoltageCheck`
  180 kV) can mark a result `UNREALISTIC_STATE`. CPU applies it via the stopping
  criteria; the GPU already judges final status CPU-side (`getStoppingCriteria().test`).
- **GPU gap:** confirm the realistic-voltage check is actually exercised on the GPU
  result and that `UNREALISTIC_STATE` / failure behavior matches CPU (the GPU iterates to
  a fixed 1e-8 infnorm regardless; the *verdict* must use the configured criteria).
- [ ] trace the criteria object the GPU uses · [ ] forced-unrealistic test GPU==CPU verdict

### W6 — Integration verification of the "already ✅/🟡" rows
The cheap-but-necessary sweep that proves the *combination* works end-to-end:
- [ ] `DC_VALUES` init seeds `x0` and the AC solve runs on GPU (DC step stays CPU)
- [ ] `twtSplitShuntAdmittance` branch values extracted correctly (compare a split-shunt
      transformer's GPU fill vs CPU term)
- [ ] HVDC AC emulation **with** distributed slack + reactive limits active together
- [ ] distributed slack with `FAIL` behavior surfaces failure identically
- [ ] full RTE param object on a small in-scope network: GPU `LoadFlow.run` == CPU

---

## 6. Definition of done

A test `GpuRteParametersTest` loads `rte-loadflow-default-parameters.json`, builds a
representative RTE-like network exercising transformer voltage control, remote voltage
control, zero-impedance branches, HVDC and step damping, runs `LoadFlow.run` with
`GPU_NEWTON_RAPHSON` and with the default CPU solver, and asserts bus states agree to
the configured stopping-criteria slack (~1e-5) — with **no extractor exception** and the
same convergence verdict. Until then, `GpuAcDataExtractor` keeps throwing loudly on the
uncovered features (correct: fail rather than mis-solve).

## 7. Sequencing recommendation

1. **W1** (independent, correctness, explicitly in the params).
2. **F1+F2 foundation**, then **W4** (simplest equation family) to shake out the
   foundation, then **W2** (codegen leverage), then **W3** (most coupled).
3. **W5/W6** verification throughout; run the §6 gate at the end.

This is essentially all of `gpu_roadmap.md` Priority 2, reframed and prioritized by what
the RTE parameter set actually demands.
