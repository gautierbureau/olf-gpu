# Refactoring plan — splitting the big files into cohesive units

Status: proposal. Nothing here is done yet. The goal is to break the few oversized files into
smaller, sub-function-oriented classes/packages **without changing behaviour** — every step compiles
and keeps the full GPU suite (110 tests) green before the next.

## 0. The hotspots

| File | LOC | Kind |
|---|---|---|
| `native/cuda/jni/olf_gpu_nr_jni.cu` | 3764 | native (one translation unit) |
| `ac/solver/GpuBatchedSecurityAnalysisSolver.java` | 1596 | batched N-1 solver |
| `ac/equations/vector/gpu/GpuAcDataExtractor.java` | 943 | equation-system → device data |
| `sa/GpuBatchedAcSecurityAnalysis.java` | 939 | SA-engine integration |
| `ac/solver/GpuNewtonRaphson.java` | 496 | base-LF AC solver |
| `ac/equations/vector/gpu/GpuAcNewtonSolver.java` | 404 | JNI binding + native loader |
| `test/…/GpuBatchedAcSecurityAnalysisTest.java` | 929 | (test) |

## 1. Guiding principles

1. **Behaviour-preserving, test-gated, incremental.** Each extraction is a mechanical move + a
   compile + a full-suite run. No numeric changes ride along. The code is heavily commented and
   well-tested; that safety net is the reason a refactor is feasible, and it must stay green.
2. **Group by sub-function, don't over-fragment.** One cohesive responsibility per class, not one
   class per method. Several proposed classes below are ~100–250 LOC — that's the target grain.
3. **Mirror OLF's package structure with a `.gpu` leaf.** The extraction layer already does this
   (`ac.equations.vector.gpu`). Extend the same convention to the solver and SA so *all* GPU code
   lives in `.gpu` leaves and none sits loose in OLF's own packages.
4. **The native split is a separate, later workstream.** It is C++/CUDA, not Java packages; it needs
   CMake changes and carries numeric risk. Treat it independently and optionally (§6).

## 2. The one real obstacle: homeless shared state

The deep read of all four files found the same thing every time. The blocker to splitting is **not**
access (see §3) and **not** poor cohesion — the clusters are already visible. It is that each big
file threads a bundle of shared state through inline code with no object to hold it:

- **Extractor**: `phiCol[]`/`vCol[]`/`qRow[]` + the per-family element-index maps
  (`cbElemOf`/`obElemOf`/`shElemOf`), computed once at the top of the 356-line `extract()` and read
  by every family packer; DISTR_Q additionally depends on the element maps.
- **Batched solver**: a **21–24 parameter bundle** threaded `solveBatchN1` → `solveQualifiedInChunks`
  → `orchestrateTransformerControl`, plus the `*ByIndex` scratch arrays.
- **SA integration**: **four cache fields** (`precomputedStates`, `precomputedIslandedBuses`,
  `baseVariableRows`, `batchEquationSystem`) joining precompute to injection, and a second bundle of
  `current*` capture fields joining `runSimulations` to the precompute driver.
- **Native**: the `SharedStructure` `shared_ptr` both solver contexts hold.

**So the refactor is as much "introduce the missing value objects" as "move code into packages."**
Introducing each value object is the *enabling* step that must precede the corresponding split. This
is the spine of the migration order in §7.

## 3. Constraint check: placement is free

Verified against OLF core: **no GPU class is forced into an OLF package by package-private access.**
Everything the GPU code touches is `protected` (reached through inheritance) or `public`:

- `GpuBatchedAcSecurityAnalysis` uses `AbstractSecurityAnalysis`'s `protected` `monitorIndex` /
  `zeroImpedanceMonitoredIndex` / `reportNode` / `logPostContingency*` (accessible from a subclass in
  any package) and the `public` `PostContingencyNetworkResult` / `LimitViolationManager` /
  `StateMonitorIndexes`.
- `GpuNewtonRaphson` uses `AbstractAcSolver`'s `protected` fields and the `public` `NewtonRaphson`
  ctor + `AcSolverUtil.updateNetwork`.
- The extraction layer already sits in `ac.equations.vector.gpu` and reaches only the public
  equation-system API (proof: it compiles in a *sub*-package of `ac.equations.vector`, so it cannot
  be using that package's package-private members).

Conclusion: we can move any GPU class into a dedicated `.gpu` sub-package. The plan is a pure
cohesion decision, unconstrained by visibility.

## 4. Target package layout

```
ac.equations.vector.gpu                     (extraction + device-data layer)
    GpuAcData, GpuAcActivation, GpuAcStructures, GpuAcDataCache
    .extract
        GpuAcDataExtractor        (thin orchestrator)
        GpuAcValidation           (the validate() phase)
        GpuAcRowIndexing          (col/row/branchVarCols/computeQRow — the stable-row policy)
        GpuAcFamilyPacker         (closed/open/shunt/HVDC/zi packing over an ExtractionColumns ctx)
        GpuAcOuterLoopPacks       (DS + RL + DISTR_Q + DISTR_RHO packs)
        GpuAcActivationExtractor  (extractActivation/extractVarType)
        ExtractionColumns         (value object: phiCol/vCol/qRow/… + *ElemOf maps)
    .jni
        GpuAcNewtonSolver         (native method decls)
        GpuNativeLoader           (load()/preloadNvidiaChain — extracted from GpuAcNewtonSolver)

ac.solver.gpu                               (NEW — GPU AC solvers, out of OLF's ac.solver)
    GpuNewtonRaphson, GpuNewtonRaphsonFactory
    GpuNativeContext              (the RAII Cleaner handle lifecycle, from GpuNewtonRaphson)
    GpuSolverInvalidation         (the listener-based cache invalidation, from GpuNewtonRaphson)
    .batch
        GpuBatchedSecurityAnalysisSolver   (thin orchestrator: extract→qualify→override→solve→assemble)
        GpuBatchQualification              (the inlined qualification loop, carved out)
        GpuBatchInjectionOutages           (computeInjectionOutages/computeBusGenOutage/GenOutage)
        GpuBatchChunkSolver                (solveQualifiedInChunks)
        GpuBatchTransformerControl         (transformerControlSupported/buildTransformerControllers/
                                            orchestrateTransformerControl + TransformerController)
        GpuBatchContextCache               (CachedBatchContext + acquire/clear + resolveChunkSize/Mode)
        BatchSolveContext                  (value object: the 21–24-param bundle)
        dto: BatchContingency, BatchScenarioResult, BatchConfig, LoadLoss, ShuntShift, TapR1Override

sa.gpu                                      (NEW — GPU SA, out of OLF's sa)
    GpuSecurityAnalysisProvider
    GpuBatchedAcSecurityAnalysis   (thin: 5 overrides delegate to the below)
    GpuSaRunScope                  (runSimulations decision logic → BatchConfig + tolerances + guard)
    GpuBatchPrecomputer            (precomputeBatch driver)
    GpuContingencyMapper           (generator/load/shunt/closed-branch shape mapping)
    GpuStrategyComposer            (strategyApplies/composeStrategyScenario/tapOverride)
    GpuStateInjector               (injectState/filterIslanded + the two result-builders — OLF-only)
    CapturedRun, PrecomputedBatch  (the two value objects that replace the shared fields)
```

Every leaf is a `.gpu` (or deeper) package; nothing new lands in OLF's own `ac.solver` / `sa`.

## 5. Per-file decomposition (Java)

### 5a. `GpuAcDataExtractor` (943 → ~thin orchestrator + 6 units)

The file is one 356-line inline `extract()` plus helpers. Cohesion is already clear; the entangling
state is `phiCol/vCol/qRow` + the `*ElemOf` maps.

- **`GpuAcValidation`** ← `validate()` (575–669) + `isDummyCouplingTerm` (671–676) + the two
  `SUPPORTED_*` sets (74–85). Self-contained; only reads `es.getIndex()`. **Cleanest cut.**
- **`GpuAcRowIndexing`** ← `col` (678), `row` (682), `branchVarCols` (686), `computeQRow` (699–766).
  This is the stable-row-indexing policy (diagonal-paired variable rows), the shared kernel every
  packer and `extractActivation` leans on. Extract first — everything else depends on it.
- **`ExtractionColumns`** (new value object) ← the per-call arrays computed at 135–151
  (`phiCol/vCol/branchRhoCol/branchAlphaCol/dummyPCol/dummyQCol/qRow`) plus the `*ElemOf` maps the
  family loop fills. This is the object that lets the packers become separate methods.
- **`GpuAcFamilyPacker`** ← the family loop (168–319): closed/open/zero-impedance/shunt/HVDC packing,
  reading `ExtractionColumns`. These stay together (they share the loop and build the element maps);
  base-open append blocks (229–291) move here too since they mutate the same packs.
- **`GpuAcOuterLoopPacks`** ← DISTR_Q driver+builder (321–337, 768–847), DISTR_RHO
  (339–349, 849–883) + `distrRhoControllerBranches` (885–910), RL block (351–384), DS block
  (386–467). `GpuAcData.sameStructureInputsAs` already treats RL/DS as separable "batch-context"
  inputs — the code endorses this boundary.
- **`GpuAcActivationExtractor`** ← `extractActivation` (490–553) + `extractVarType` (562–573). Already
  separable — produces `GpuAcActivation`, not `GpuAcData`.
- Flatteners (`flatD/flatI/toIntArray/toDoubleArray`) → a tiny package-private util.

`GpuAcDataExtractor.extract()` becomes: validate → build `ExtractionColumns` → family pack → outer-loop
packs → assemble `GpuAcData`.

### 5b. `GpuBatchedSecurityAnalysisSolver` (1596 → thin orchestrator + 6 units)

Three methods hold ~50% of the file (`solveQualifiedInChunks` 312, terminal `solveBatchN1` 228,
`orchestrateTransformerControl` 133). Cleanly extractable, in rough order of ease:

- **DTO records** (`LoadLoss`, `ShuntShift`, `TapR1Override`, `BatchContingency`,
  `BatchScenarioResult`, `BatchConfig`) → own files. (`GenOutage`, `TransformerController`,
  `FrozenIsland` stay with their logic clusters.) Also move the shared constants
  `STATUS_*` / `STRATEGY_KEY_SEP` / `NO_FROZEN_*` to a small shared holder.
- **`GpuBatchInjectionOutages`** ← `computeInjectionOutages` (944–975) + `computeBusGenOutage`
  (977–1067) + `GenOutage`. The most self-contained non-trivial cluster (~125 LOC, pure functions
  that null-signal CPU fallback).
- **`GpuBatchContextCache`** ← `CachedBatchContext` (270–282), `acquireBatchContext` (292–310),
  `clearContextCache` (313–318), the shutdown hook (286–289), `resolveChunkSize` (1577–1592),
  `resolveBatchMode` (255–257). **Owns the file's only mutable global state — extract as one unit**
  so the `synchronized` accessors + shutdown hook stay together.
- **`GpuBatchTransformerControl`** ← `transformerControlSupported` (118–133),
  `buildTransformerControllers` (138–191), `orchestrateTransformerControl` (1115–1247),
  `TransformerController` (97–112), `tapModels`/`tapR1Values` (195–228). ~245 LOC, reasonably cohesive.
- **`GpuBatchQualification`** ← the qualification loop currently **inlined** in the terminal overload
  (692–847). First carve it into a method, then a class. It produces the batched-index list + the
  frozen-island arrays (`islandedFrozen`/`addFrozenVarRow` move with it).
- **`BatchSolveContext`** (new value object) ← the 21–24-param bundle (`n`, `nElements`, `tol`,
  `reactiveLimitTol`, `slackDistTol`, `convEpsPerEq`, `varType`, `config`, the `*ByIndex` arrays, DS
  override arrays). This is the enabling step for the next one.
- **`GpuBatchChunkSolver`** ← `solveQualifiedInChunks` (1249–1560), taking a `BatchSolveContext`. It
  owns the two `solveBatchContext` JNI call sites and the per-scenario scratch buffers.

Terminal `solveBatchN1` becomes: extract data → `GpuBatchQualification` → `GpuBatchInjectionOutages`
→ `GpuBatchChunkSolver` (which calls `GpuBatchTransformerControl`) → assemble results. The 9 thin
overload delegators stay as the public API funnel.

### 5c. `GpuBatchedAcSecurityAnalysis` (939 → thin subclass + 5 units)

The natural fault line is sharp: **"build & solve the batch" (solver-coupled) vs "inject state &
build results" (OLF-only)**, joined by exactly four cache fields.

- **`PrecomputedBatch`** (new value object) ← `precomputedStates`, `precomputedIslandedBuses`,
  `baseVariableRows`, `batchEquationSystem`. These are the *only* coupling between precompute and
  injection — bundling them is the clean cut.
- **`CapturedRun`** (new value object) ← the `current*` fields (network / context / contingencies /
  strategies / actions / lfParams). Replaces the field-side-channel between `runSimulations`,
  `createLoadFlowContext`, and the precompute driver.
- **`GpuStateInjector`** ← `injectState` (584–592) + `filterIslanded` (600–618) + the two
  near-duplicate result-builders (bodies of `runActionSimulation` 550–579 and
  `runPostContingencySimulation` 904–937). **OLF-only, zero solver dependency — the cleanest
  cluster.** Deduplicate the two result-builders while here.
- **`GpuSaRunScope`** ← the `runSimulations` decision body (169–312) + `gpuSupports`/sysprop helpers.
  Produces a `BatchConfig` + tolerances + the batchability verdict.
- **`GpuContingencyMapper`** ← `generatorOutages`, `loadOutages`, `shuntShifts`, `singleShuntOutage`,
  `closedBranchOutages` (nearly pure functions).
- **`GpuStrategyComposer`** ← `strategyApplies` (622–638), `composeStrategyScenario` (643–809),
  `tapOverride` (815–829). The 167-line action-type switch is the single biggest blob in the file;
  isolating it is high-value.
- **`GpuBatchPrecomputer`** ← `precomputeBatch`/`precomputeBatchLocked` + `GPU_DEVICE_LOCK`, calling
  the mapper + composer, returning a `PrecomputedBatch`.

`GpuBatchedAcSecurityAnalysis` shrinks to the 5 overrides delegating to `GpuSaRunScope` /
`GpuBatchPrecomputer` / `GpuStateInjector`, holding one `CapturedRun` + one `PrecomputedBatch`.

### 5d. The two medium files (opportunistic, low-risk)

- **`GpuNewtonRaphson` (496)**: extract the two self-contained nested classes — `NativeContext` (RAII
  Cleaner handle lifecycle) → `GpuNativeContext`, and `Invalidation` (the `EquationSystemIndexListener`
  + `LfNetworkListener` cache invalidation) → `GpuSolverInvalidation`. Leaves a focused `AcSolver`.
- **`GpuAcNewtonSolver` (404)**: extract the native-library loader (`load`/`preloadNvidiaChain`/
  `subdir`/`firstExistingDir`/`preloadOne`, ~90 LOC) → `GpuNativeLoader`. The remainder is pure JNI
  declarations grouped by subsystem (structure / single-context / batch-context / adjoint) + Java
  convenience wrappers — keep as one binding class.

## 6. Native side (`olf_gpu_nr_jni.cu`, 3764) — separate workstream

This is the biggest file but a different problem: one CUDA translation unit compiled by the **root
`CMakeLists.txt`** (there is no `native/CMakeLists.txt`). Everything except the JNI entry points is in
one anonymous namespace, so a naive file split hits cross-TU symbol resolution immediately.

**Two tiers, in order of value/risk:**

**Tier 1 — readability, no TU split (low risk, do first):**
- Break up the three giant functions in place: `solveBatchContext` (2520–3190, ~670 LOC) into
  sub-functions (override upload; the 6-way `mode` switch; the sequential RL→DS outer-loop driver;
  readback) — each already a labelled block; `createBatchContext` (188, per-mode blocks are
  independent); `buildStructure` (163, sequential build stages).
- ~~**Delete dead code**: the old per-(bus,scenario) single-clamp slack kernel, defined but never
  launched.~~ **DONE** — removed; the surviving full-water-filling kernel was renamed to
  `distributeSlackBatch`.

**Tier 2 — actual TU split (higher effort, needs CMake, do only if Tier 1 isn't enough):**
- Cleanest extractions, sharing only macros + RAII wrappers:
  - `gpu_infra.cuh` ← includes, `CUDA_CHECK`/`CUDSS_CHECK`, `DeviceBuf`, `CudssCtx`, version shims.
  - `gpu_config.cuh` ← IR/reorder/matching knobs + `traceLuNnz`.
  - `gpu_adjoint.cu` ← `solveTransposedExplicit` (3692–3764). **The single cleanest extraction** —
    fully self-contained, no `SharedStructure` dependency.
- The hard split — single-path vs batch-path into separate `.cu` — requires first moving
  `SharedStructure` + the shared device helpers (`scatterAdd`, `addAt`, `grid`,
  `initClosedBranchTrig`) to a common header, **and** either declaring them `static inline __device__`
  or enabling `CUDA_SEPARABLE_COMPILATION`. Each new `.cu` must be added to the
  `add_library(olfgpu SHARED …)` source list. Higher risk; defer unless the batch path keeps growing.

Every native change is gated by a native rebuild + the full GPU suite (numeric), not just a compile.

## 7. Migration order

Each numbered step is one PR-sized change: compiles, full GPU suite green, no numeric diff. Ordered
so the enabling value objects land before the splits that need them, easiest/lowest-risk first.

1. **Package moves, no code change.** Move `GpuNewtonRaphson`/`Factory` → `ac.solver.gpu`,
   `GpuBatchedSecurityAnalysisSolver` → `ac.solver.gpu.batch`, `GpuBatchedAcSecurityAnalysis` /
   `GpuSecurityAnalysisProvider` → `sa.gpu`. Update the `@AutoService` provider + any test imports.
   Pure relocation — smallest possible diff, establishes the layout.
2. **Medium-file nested-class extractions** (§5d) — `GpuNativeLoader`, `GpuNativeContext`,
   `GpuSolverInvalidation`. Low risk, immediate LOC win.
3. **Extractor**: `GpuAcValidation`, then `GpuAcRowIndexing`, then `GpuAcActivationExtractor` (all
   already separable), then introduce `ExtractionColumns` and split `GpuAcFamilyPacker` +
   `GpuAcOuterLoopPacks`.
4. **Batched solver**: DTOs first, then `GpuBatchInjectionOutages` + `GpuBatchContextCache` +
   `GpuBatchTransformerControl` (all cleanly separable), then carve `GpuBatchQualification` out of the
   terminal, then introduce `BatchSolveContext` and split `GpuBatchChunkSolver`.
5. **SA integration**: introduce `PrecomputedBatch` + `CapturedRun`, extract `GpuStateInjector`
   (OLF-only, safest), then `GpuContingencyMapper` + `GpuStrategyComposer`, then `GpuSaRunScope` +
   `GpuBatchPrecomputer`.
6. **Tests**: relocate the 45 misfiled tests to packages matching what they exercise (batch → 
   `ac.solver.gpu.batch`, SA → `sa.gpu`, extraction → `ac.equations.vector.gpu`); split the 929-line
   `GpuBatchedAcSecurityAnalysisTest` along the new class boundaries. Add `package-info.java` to each
   new package (none exist today).
7. **Native Tier 1** (in-function breakup + dead-code delete). **Native Tier 2 only if warranted.**

Steps 1–2 are safe warm-ups. 3–5 are the substance and each sub-bullet is independently shippable, so
the work can pause at any green state. 6 is cleanup. 7 is optional.

## 8. What NOT to split, and risks

- **Don't split the extractor's family-packing loop** into per-family classes that each re-walk the
  network — they legitimately share one pass and build the element maps together. Keep them one
  `GpuAcFamilyPacker` over an `ExtractionColumns` context.
- **Don't force the native single/batch TU split** for its own sake. Tier 1 (in-function breakup +
  dead-code removal) gets most of the readability with none of the cross-TU risk.
- **Don't over-fragment the DTOs.** Small related records can share a file; one-record-per-file is
  noise.
- **The provider `@AutoService`** regenerates `META-INF/services` on build — verify the SA provider
  still registers after the package move (step 1).
- **Risk acknowledgement**: this is a large, behaviour-preserving churn over working, well-commented,
  well-tested code. The payoff is navigability and testability, not correctness or speed. It is worth
  doing *incrementally* and *only* while the full suite gates every step; it is not worth doing as one
  big-bang change. Any step that cannot stay green should be reverted, not forced.
