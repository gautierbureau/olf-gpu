# powsybl-open-loadflow-gpu

GPU extension for [powsybl-open-loadflow](https://github.com/powsybl/powsybl-open-loadflow) (OLF):

- **`OpenLoadFlowGpu`** (load flow) — a `LoadFlowProvider` extending OLF's: OLF's load flow with the
  AC Newton-Raphson on the GPU. Select it by provider name; OLF's own parameters apply unchanged, and
  the GPU solver's parameters (`GpuLoadFlowParameters`: ordering, matching, pivoting, iterative
  refinement) are its specific parameters, settable in a parameters JSON file, by name/value or in the
  `open-loadflow-gpu-default-parameters` configuration module.
- **`GPU_NEWTON_RAPHSON`** — the cuDSS-backed full-Newton-Raphson AC solver behind it, plugged in
  through OLF's `AcSolver` SPI (`OpenLoadFlowParameters.setAcSolverType(GpuNewtonRaphsonFactory.NAME)`
  also selects it under OLF's own provider). One JNI call
  runs the whole NR loop on the device (fixed-pattern CSR Jacobian, fused fill kernels,
  cuDSS solves); reruns reuse the cached device context.
- **`OpenLoadFlowGpu`** (security analysis) — a `SecurityAnalysisProvider` running single-branch N-1
  contingencies as one GPU batch (`GpuBatchedAcSecurityAnalysis`), degrading to the
  standard AC path when no GPU is present. Best measured: pegase9241, 500 contingencies,
  K=4 — ~4.8 ms/scenario warm on an RTX A1000 (cuDSS 0.8.0, uniform batch).

**Stock OLF is never modified.** The fused CUDA kernels in `native/cuda/` are PARSED and
transpiled from OLF's own hand-written equation Java by `codegen/java2cuda` (no symbolic
math — OLF hand-writes both the values and the Jacobian); the proof suite pins them to a
frozen golden manifest and to the committed kernels, and `codegen/check_scatter.py`
validates the Jacobian scatter layout against OLF's real equation-system wiring. See
`codegen/README.md`.

## Build

Java (needs the cuDSS-enabled powsybl-core/math-native, see pom properties):

    mvn install

Native (opt-in; without it `isAvailable()` is false and everything falls back to CPU):

    # needs CUDA toolkit + cuDSS; see native/build-gpu.sh header for the env vars
    native/build-gpu.sh

Regenerate the kernels after an OLF formula change (needs an OLF checkout at OLF_SRC):

    codegen/regenerate.sh

## Current known deltas vs fully-stock dependencies

- `powsybl-core` / `powsybl-math-native`: this extension builds against the cuDSS-enabled
  forks (`CuDssMatrixFactory` + the `libmathcudss` JNI) — see the pom version properties.
- `powsybl-open-loadflow`: built from the `olf-local-patches` branch — upstream main plus
  ONE 4-line commit adding `PropagatedContingency.getShuntIdsToShift()` (candidate
  upstream PR; the equation formulas remain byte-identical to upstream).
- `CommonTestConfig`/`ServiceParameterResolver` are vendored under `src/test` (OLF's
  test-jar does not export them; the copies must stay signature-compatible with the
  `AbstractOpenSecurityAnalysisTest` shipped in OLF's test-jar, which links against them).
