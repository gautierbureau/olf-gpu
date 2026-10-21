# Zero-default voltage thresholds make SA violation reporting flicker — upstream question

Status: draft for an upstream discussion (powsybl-core + powsybl-open-loadflow), written
2026-07-17/18 from the GPU-vs-CPU production benchmark investigation (see
`docs/archive/gpu_benchmark.md`, entries "The 6,137 diff SOLVED" and the drill that followed).

## The observed problem (measured)

Comparing two AC security analyses of the SAME 500 contingencies on case9241pegase whose solved
states were verified IDENTICAL (monitored bus voltages equal to 5 decimals, pre and post), the
two runs reported **6,137 one-sided violations** (2,899 only-in-A + 3,238 only-in-B) across 263
contingencies. None of it was physics: every one-sided entry was a PRE-EXISTING violation whose
post-contingency value drifted by ~1 ulp .. 1e-6 pu against the run's OWN pre-contingency
baseline — below any physical meaning, and in a different direction on each side.

This affects ANY pair of runs whose floating-point trajectories differ — two OLF versions, two
linear solvers (KLU vs anything), CPU vs GPU, even the same build on two machines — and it also
means single production runs re-report borderline pre-existing violations noisily.

## Mechanism (exact code, current versions)

powsybl-core `SecurityAnalysisParameters.IncreasedViolationsParameters` defaults:

| threshold | default |
|---|---|
| flowProportionalThreshold | **0.1** (10 %) — sane |
| lowVoltageAbsoluteThreshold | **0.0 kV** |
| lowVoltageProportionalThreshold | **0.0** |
| highVoltageAbsoluteThreshold | **0.0 kV** |
| highVoltageProportionalThreshold | **0.0** |

powsybl-open-loadflow `LimitViolationManager.violationWeakenedOrEquivalent` compares a
post-contingency violation to the pre-contingency one with, for HIGH_VOLTAGE:

```java
double value = Math.min(parameters.getHighVoltageAbsoluteThreshold(),
                        violation1.getValue() * parameters.getHighVoltageProportionalThreshold());
return violation2.getValue() <= violation1.getValue() + value;
```

Two consequences:

1. **Zero-width voltage suppression by default**: with the core defaults, a pre-existing voltage
   violation is re-reported whenever the post value exceeds the pre value by ONE ULP.
2. **The `min(abs, value*prop)` trap**: because the two legs are combined with `min`, setting
   ONLY the absolute threshold (the intuitive move) leaves the effective threshold at
   `min(abs, value*0) = 0` — a silent no-op. Users must set BOTH legs, which nothing documents
   or enforces.

## Questions / proposed upstream fixes

1. powsybl-core: should the voltage thresholds default to small non-zero values (e.g. 0.1–0.5 kV
   absolute + 0.1 % proportional), mirroring the flow threshold's non-zero default? A zero-width
   deadband on a floating-point comparison is noise by construction.
2. powsybl-open-loadflow: should the `min()` treat an UNSET (zero) leg as "no bound" — i.e.
   effectively `min` over the CONFIGURED legs only (or `max`, per taste)? Today `abs=0.5,
   prop=0` is indistinguishable from "no threshold at all", which reads like a bug.
3. At minimum: document that both legs must be set together.

## Recommended settings meanwhile (used by our benchmarks)

For any SA comparison (and arguably production runs) set explicitly, e.g.:

```java
sap.setIncreasedViolationsParameters(new SecurityAnalysisParameters.IncreasedViolationsParameters(
        0.5,    // lowVoltageAbsoluteThreshold [kV]
        0.001,  // lowVoltageProportionalThreshold
        0.5,    // highVoltageAbsoluteThreshold [kV]
        0.001,  // highVoltageProportionalThreshold
        0.001)); // flowProportionalThreshold (0.1 % — tighter than the 10 % default, deliberate)
```

With these, the 6,137 phantom diffs collapsed to 22 genuine threshold-edge cases in 6
contingencies (all within the solvers' shared convergence tolerance of their limits).
