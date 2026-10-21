# CLAUDE.md — powsybl-open-loadflow-gpu

Context for sessions on this repo: a STANDALONE Maven extension for stock
powsybl-open-loadflow — the GPU load flow (`OpenLoadFlowGpu` LoadFlowProvider, over the AcSolver SPI
`GPU_NEWTON_RAPHSON`), the GPU-batched N-1 security analysis (`OpenLoadFlowGpu` provider), and the
**java2cuda equation codegen**. Full design: **`codegen/README.md`** +
**`codegen/java2cuda/README.md`**; deltas vs stock dependencies: **`README.md`**.

## Project split (history)

This code lived on the `olf_gpu` branch of the OLF fork
(`~/Projects/powsybl/powsybl-open-loadflow-3`, kept as a branch for history). That
checkout is now THE working OLF worktree for this project: branch
**`olf-local-patches`** = upstream main + the minimal OLF fixes the GPU extension
needs (currently ONE 4-line `PropagatedContingency.getShuntIdsToShift()` commit,
candidate upstream PR). Rebuild the dependency after changing it:
`cd ../powsybl-open-loadflow-3 && mvn -s ~/Projects/powsybl/tmp/settings.xml
-DskipTests -Dcheckstyle.skip -Dmaven.javadoc.skip -Dlicense.skip install`.
It is also what `OLF_SRC` points at for the codegen parse. Never build the
dependency from `olf_gpu` — its jar would duplicate this project's classes. GPU classes live in
`com.powsybl.openloadflow.*` SPLIT packages (classpath access to OLF internals);
`CommonTestConfig`/`ServiceParameterResolver` are vendored under `src/test` (OLF's
test-jar does not export them — keep them signature-compatible).

## Architecture (current — java2cuda)

**The hand-written OLF equation Java is the single source of truth.** The term
classes in `src/main/java/.../ac/equations/` are BYTE-IDENTICAL to upstream main
(never edit a formula there except as an upstream-able change); the fused CUDA
kernels + scatter tables in `native/cuda/equations/generated/` are PARSED and TRANSPILED from them by
`codegen/java2cuda/` (stdlib parser reusing `extract_golden.py`'s flattening +
one Jinja template — **no SymPy**: OLF hand-writes values AND Jacobian, nothing
is derived). Per-family specs `codegen/java2cuda/spec_<family>.py` are METADATA
ONLY (files, trig params, hoisted helpers, CUDA identity, scatter stamp); the one
declared formula exception is HVDC's `extern_methods` (powsybl-iidm's
getHvdcLineLosses closed form, another repo).

Families: closed_branch, open_branch, shunt, hvdc (the GPU fill kernels, with
scatter) + closed_branch_i (current magnitude, read by the current-limiter loops:
per-method functions and calculateSensi's private components, no assembler
routine; outerloop/flow_observation.cuh mirrors calculateSensi line for line).

Parser mechanisms (all general, no special cases): local-decl + recursive
helper/sibling inlining, per-file helper scope (defining file wins — HVDC's two
isController, open-i's two gres/bres), alpha-equivalent cross-file dedup
(open-branch shunt), overload selection by visibility (public i1 over private
i1(...,theta)), helper hoisting to kernel locals with dependencies
(shunt before reI/imI), Java `?:` ternaries pass through as C, abs→fabs.

## Regenerate / verify

```
codegen/regenerate.sh      # prove -> check_scatter -> render into native/cuda/equations/generated/
mvn -o validate            # checkstyle — must be 0 violations
```
Proof (`codegen/java2cuda/prove.py`, stdlib): A parse == golden manifest
(codegen/golden/, frozen from the ORIGINAL Java — a formula change fails loudly;
update the golden deliberately with the re-render), B finite-difference audit of
OLF's hand Jacobian (skipped for HVDC's intentional approximation), C rendered ==
committed kernels (drift guard), D rendered == golden.
`codegen/check_scatter.py` validates the stamps against the real term-class
wiring. CI: `.github/workflows/codegen-check.yml` (render-up-to-date +
formulas-proven). An upstream formula change ⇒ clean merge (files are upstream's
own) then `codegen/regenerate.sh` + commit kernels (+ golden if math changed).

## Conventions & gotchas

- **Never hand-edit** `native/cuda/equations/generated/*` / `codegen/scatter/*.scatter.json` — they are generated
  from the Java (CI flags any diff).
- The parser handles OLF's expression-only Java subset (locals + single return,
  helper calls, boolean predicates, ternaries, Math/FastMath); anything else
  fails the parse loudly.
- Kernel bodies are the VERBATIM flattened Java text (last-ulp reassociation vs
  the old SymPy kernels only; proven ≤2e-14 over 4M compiled points).
- History: the first pipeline (Python residual specs + SymPy autodiff emitting
  generated Java + CUDA) was retired at commits ac3eef3ea (Java reverted to
  pristine upstream) + the step-2 commit; the goldens and extract_golden.py
  survive from it. GPU NR returns the BEST visited iterate when maxIter is
  exhausted (4c2b48cb9 — a plateau's last sample is a lottery; see
  GpuLineSearchTest).

## GPU runtime (native package + full Newton-Raphson)

- Root `CMakeLists.txt` builds **libolfgpu** — the FULL GPU Newton-Raphson, the ONE
  native library — into `target/classes/natives/<arch>/` (bundled in the jar, loaded
  from the classpath via scijava `NativeLoader`, like powsybl-math-native). Static
  cudart; cuDSS REQUIRED at build time and linked dynamically (never bundled: the bundled .so has NO
  RPATH, and the system loader resolves `libcudss`/`libcublas` through `LD_LIBRARY_PATH` or the
  ldconfig cache — `native/local-env.sh` exports it; without them `GpuAcNewtonSolver.isAvailable()`
  is false). `native/build-gpu.sh` builds it (cuDSS default `../cudss`); after `mvn clean` just
  re-run it. cuDSS fill-reducing ordering: AMD by default on >= 0.8.0 (2026-07-16 sweep: −15% LU
  fill vs the library default, which is ND there; COLAMD family NOT_SUPPORTED on UBATCH) — the
  `reorderingAlgorithm` entry of `GpuLoadFlowParameters` (config module
  `open-loadflow-gpu-default-parameters`) overrides it. (History: rung 1
  was a separate fill-only dense closed-branch assembler — old `libolfgpu` +
  `ClosedBranchGpuAssembler` — beside the full-NR `libolfgpunr`; the assembler was
  removed and the full-NR lib renamed `libolfgpu` once rungs 2/3 subsumed it.)
- `GpuAcNewtonSolver.solve(...)` = ONE JNI call for the whole NR loop: host builds the
  fixed-pattern CSR (rows = equation columns, cols = variable rows — TRUE J, cuDSS solves
  `J dx = -F` directly, no transpose games) from the generated scatter tables; the four
  fused kernels fill J values + mismatch **in cuDSS's device arrays** each iteration;
  per iteration only the n-double mismatch crosses PCIe. Scatter ROLE is folded into the
  host-built dest arrays (`>=0` J slot, `-2-row` F row, `-1` skip) so kernels never index
  the constexpr tables. Slack/PV `BUS_TARGET_V/PHI` rows are identity entries.
- **AcSolver SPI**: `GpuNewtonRaphsonFactory` (`@AutoService`, name `GPU_NEWTON_RAPHSON`)
  → `LoadFlow.find("OpenLoadFlowGpu")` (`GpuLoadFlowProvider`) selects it on a copy of the run's parameters;
  `OpenLoadFlowParameters.create(p).setAcSolverType(GpuNewtonRaphsonFactory.NAME)` does too under OLF's provider.
  The GPU solver's own settings are `GpuLoadFlowParameters`, the provider's specific parameters (JSON, name/value,
  config module `open-loadflow-gpu-default-parameters`).
  `GpuNewtonRaphson` re-extracts per run (outer loops work: distributed slack, PV→PQ)
  and CACHES the native solver context (per-solve device buffers + cuDSS analysis, via
  `createContext(structHandle)`/`solveContext`/`destroyContext`, freed by a `Cleaner`).
  The STRUCTURE (row permutation, superset CSR pattern host+device, per-family device
  data + role-folded dest arrays, DISTR_Q/RHO + zero-impedance packs, per-element
  static trig) is a SEPARATE native object — `SharedStructure`, built by
  `createStructure`/freed by `destroyStructure`, held by `shared_ptr` in BOTH context
  kinds — cached per extraction Java-side (`GpuAcStructures`, weak-keyed by
  `GpuAcData` + Cleaner; destroy only drops the Java ref, contexts keep it alive). So
  the base LF and the batched SA sharing one `GpuAcDataCache` extraction share ONE
  device-resident structure: `createBatchContext(structHandle, nS, mode, rl/ds packs)`
  and `recommendBatchSize(structHandle, maxCap)` no longer rebuild the pattern, and
  the chunk-size LU probe result is cached ON the structure (one probe per
  extraction, not per SA run). `refreshContextValues` mutates the SHARED structure in
  place (families + `ziIn` + re-derived static trig — one extraction = one value set)
  and the refresher then calls `GpuAcStructures.rebind(old, new)`, which drops EVERY
  key aliased to the refreshed structure and re-keys it to the new extraction, so
  content matching never aliases stale values. The hand-off is ASSERTED in
  `GpuSaSolverTest#gpuBatchedEngagesAndMatchesCpuOnDevice` via
  `GpuAcStructures.hitCount()`. Sharing is instance-first with a CONTENT fallback:
  another extraction of the same state (another distributed-slack setting, the
  `GpuAcDataCache` key, or a re-extraction after the cache dropped one) is a different
  `GpuAcData` instance; the RL/DS packs are batch-CONTEXT inputs, so on an instance
  miss the cache content-matches the structure inputs (`GpuAcData.sameStructureInputsAs`)
  and aliases the existing structure (refcounting `Structure` holder — Cleaner fires
  when the last aliased key is collected). Measured 2026-10-21: ~430 content hits over
  the suite, one per rte6515 gate scope — not dead code. Rows use the STABLE
  indexing (P row = the bus's `BUS_PHI` variable row,
  Q row = `BUS_V` — OLF's renumbered equation columns are NOT in the GPU structure), so
  PV→PQ switching is value-only: per solve, `extractRowMode` (power-eq vs slack/PV
  identity mask, applied by the `applyRowMode` kernel after the fills: row zeroed,
  diag 1, F = x−target) + `extractTargets` (stable order). ONE context build per
  LoadFlow run, reruns solve in ~8–12 ms (PEGASE 9k); 13k ≈ 2.2× faster than KLU;
  `GpuAcDataExtractor` VALIDATES the equation system and throws on anything uncovered
  (non-BUS_V/PHI variables, unknown equation types/terms, disconnection-allowed
  branches). The ENGINE's es is VECTORIZED (equation arrays for closed-branch terms;
  `getEquation(bus,type)` resolves through arrays, inactive elements give column -1) —
  validation allowlists the four `ac_{p,q}_array_closed_{1,2}` evaluators + per-element
  scalar terms. GPU iterates to 1e-8 pu infnorm; final status judged by the CONFIGURED
  stopping criteria CPU-side (so LoadFlow states differ from the CPU run only within
  the criteria's slack, ~1e-5). The batched SA applies OLF's MAX_VOLTAGE_CHANGE step clamp per scenario and OLF's exact ‖F‖₂ stopping test when `convEpsPerEq` is passed; on networks with zero-impedance branches the batched solves are refined ADAPTIVELY (our own residual check per lane, `OLF_GPU_IR_TOL`/`OLF_GPU_IR_MAX`; cuDSS's internal step off, `OLF_GPU_IR_BASE=0`; `OLF_GPU_IR_STEPS=<n>` forces a fixed count).
- Tests: `GpuLoadFlowTest` (rung 3: plain LoadFlow.run GPU vs CPU per family network),
  `GpuFullNewtonTest` (rung 2: one per family network; reference = OLF's own
  `JacobianMatrix` Newton on the SAME equation system — any missed/duplicated term makes
  states diverge). Both gated on `isAvailable()` → CUDA-free builds skip. Runner:
  `native/cuda/run_gpu_tests.sh` (build + both suites).

## Repo / git

- **Open work lives in `docs/tasks.md`** (the MCP task tracker disconnected 2026-07-21;
  that file is now the durable record — open items, what is resolved and why, plus the
  method notes worth not relearning). Reconcile rather than overwrite if the tracker returns.
- THIS repo pushes to `origin` (`gautierbureau/olf-gpu`), the user's own fork — explicit refspecs
  (`git push origin <branch>:refs/heads/<branch>`). Only on an explicit ask; the default is to
  commit locally. (This line used to read "no remote yet"; it was stale by 59 commits.)
- The OLF fork checkout (`~/Projects/powsybl/powsybl-open-loadflow-3`, `~/olf-gpu/powsybl-open-loadflow-3`
  on the H100 box): **`olf-local-patches` is pushed to the `gb` remote** (`gautierbureau/powsybl-open-loadflow`,
  the user's own fork; 2026-10-07) — `git push gb olf-local-patches:refs/heads/olf-local-patches`, only on an
  explicit ask. **Never `origin`** (upstream powsybl). `test6` was the earlier target. This repo's code calls
  fork-only API (setValuesOnly, getSortedVariablesToFindOnly, ...), so push the fork with it.
- powsybl-math-native (`~/Projects/powsybl/powsybl-math-native-1`, branch
  `cudss_solver_1`) holds only the cuDSS JNI bindings (libmathcudss); powsybl-core
  (`~/Projects/powsybl/powsybl-core-1`, branch `cudss_factory`) holds
  CuDssMatrixFactory. Everything GPU/codegen lives HERE.
- Builds use the isolated repo: `mvn -s ~/Projects/powsybl/tmp/settings.xml ...`
  (never the shared ~/.m2; do not use `-o`).


## Benchmarks (two tracks — keep both)

- **Fixed-iter throughput vs gpusim2grid**: `native/cuda/run_bench.sh`
  (`GpuBatchedN1FixedIterTest`, tol=0, K=4, 500 matched ctg) + the
  `/gpu-bench-compare` skill. Report COLD FIRST. `OLF_GPU_TOL=1e-6` switches the same
  bench to the production path (per-iteration convergence + retirement) — see the
  2026-07-17 tol>0 entry in `codegen/gpu_benchmark.md` (straggler dominates, 8.4x).
- **Production SA vs CPU OLF**: `GpuVsCpuProductionSaBenchmarkTest` (sa package) —
  real AcSecurityAnalysis both sides, RTE params from `rte-loadflow-default-parameters.json`
  (repo root), same 500 ctg, violations included. SPARSE factory forced (the harness
  dense default makes the 9241 base FAIL after ~200 s). CPU arm multithreaded
  (`OLF_GPU_SA_CPU_THREADS`, default 4; `OLF_GPU_SA_CPU_SINGLE=1` adds the 1-thread
  reference); GPU arm host-single-threaded (threadCount>1 SIGSEGVs — open bug below).
  2026-07-17 numbers: CPU-4t 13.2 s vs GPU warm 22.2 s (GPU 0.59x — loses today; the
  straggler early-exit + device-norm levers are what flip it).
- OPEN PRODUCT BUGS (found 2026-07-17): (1) GPU provider + threadCount>1 → concurrent
  precomputeBatch → SIGSEGV in cuLaunchKernel (needs provider-side serialization);
  (2) on a FAILED base LF the GPU provider reported pre-contingency CONVERGED and
  still produced post-contingency results; (3) violation-count drift vs CPU (~2.8%,
  13,412 vs 13,791) to investigate.

- Violation-reporting flicker: powsybl's ZERO-default voltage thresholds + OLF's
  `min(abs, value*prop)` semantics re-report pre-existing violations on ulp-level drift —
  see `docs/increased-violations-thresholds.md` (upstream question + the thresholds every
  SA comparison here must set). Measured: 6,137 phantom diffs on identical states.
