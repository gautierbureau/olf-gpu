# Architecture — powsybl-open-loadflow-gpu

How the GPU extension is structured, why it is structured that way, and — the question this
document exists to answer precisely — **how much of OLF's CPU path the security analysis reuses**
(a lot: it subclasses the CPU engine and replaces only the numeric solve).

Nothing here modifies stock OLF. The extension plugs into two published SPIs and the CUDA kernels
are transpiled from OLF's own equation Java (see `codegen/README.md`). Everything degrades to the
standard CPU behaviour when no GPU is present.

---

## 1. The two entry points

The extension exposes exactly two things a user selects; each is a stock OLF extension point.

| Feature | SPI | Selector | Class |
|---|---|---|---|
| Full-NR AC solver on the GPU | `AcSolver` (OLF) | `OpenLoadFlowParameters.setAcSolverType("GPU_NEWTON_RAPHSON")` | `GpuNewtonRaphsonFactory` → `GpuNewtonRaphson` |
| Batched N-1 security analysis | `SecurityAnalysisProvider` (powsybl-security) | provider name `"OpenLoadFlowGpu"` | `GpuSecurityAnalysisProvider` → `GpuBatchedAcSecurityAnalysis` |

Both are opt-in and both fall back: `GpuAcNewtonSolver.isAvailable()` gates every device path, and
`GpuSecurityAnalysisProvider` extends `OpenSecurityAnalysisProvider`, so a CUDA-free build behaves
exactly like stock OLF.

---

## 2. Layering

Four layers, bottom to top:

```
  ┌─ sa/            GpuSecurityAnalysisProvider, GpuBatchedAcSecurityAnalysis
  │                 (integration into OLF's AC security-analysis engine)
  ├─ ac/solver/     GpuNewtonRaphson (base LF), GpuBatchedSecurityAnalysisSolver (batch),
  │                 GpuBatchContingencyValidator, GpuBatchElementMapper, factories
  ├─ ac/equations/…/gpu/   the extraction layer: OLF EquationSystem → flat device data
  │                 GpuAcDataExtractor, GpuAcData, GpuAcActivation, GpuAcStructures,
  │                 GpuAcDataCache, GpuAcNewtonSolver (JNI)
  └─ native/cuda/   libolfgpu: fused fill kernels (*.cuh) + JNI (olf_gpu_nr_jni.cu) + cuDSS
```

### 2a. The extraction layer (`ac/equations/vector/gpu/`)

This is the layer that turns an OLF `EquationSystem<AcVariableType, AcEquationType>` into the flat
arrays the device consumes. It is shared by both solvers.

- **`GpuAcDataExtractor`** — reads the equation system and produces `GpuAcData`, "mirroring exactly
  the wiring `AcEquationSystemCreator` produced." It **validates first and throws** on anything the
  GPU model does not cover (unsupported variable/equation/term types, vectorized equation arrays),
  so an unsupported network fails loudly rather than converging to a wrong state. This same
  throw-on-unsupported extract is reused as the *support probe* everywhere (`gpuSupports(...)`).

- **`GpuAcData`** — the flat, per-family packing the kernels consume: closed branches, open
  branches, shunts, HVDC AC-emulation, plus the DISTR_Q (remote generator VC), DISTR_RHO
  (transformer ratio distribution), and zero-impedance (bus-coupler) families. **The key design
  choice lives here: STABLE row indexing.** Every equation row is identified by its
  diagonal-paired *variable* row (a bus's P row is its `BUS_PHI` variable row, its Q row its
  `BUS_V` variable row), never by OLF's equation column — because OLF renumbers columns on every
  activation change (PV→PQ switching). The consequence: the extracted pattern is a **fixed
  superset**, identical across outer-loop activation switching, so a cached device context survives
  switching instead of being rebuilt.

- **`GpuAcActivation`** — what *activation* decides, in that stable indexing, recomputed only when
  an activation event fires: `rowMode[r]` (1 = power equation active, 0 = slack/PV target identity,
  applied as a device-side mask) and `targetMap[r]` (which OLF column's target lands in stable row
  `r`). This is the trick that lets one fixed pattern represent many PV/PQ configurations: the
  *pattern* never changes; a per-solve **row-mode mask** turns individual rows into identities.

- **`GpuAcStructures`** — per-extraction cache of the native *shared structure* (device-resident
  pattern, families, destination arrays, static trig) that both solver contexts derive from one
  `GpuAcData`; refcounted, `Cleaner`-released.

- **`GpuAcDataCache`** — an invalidation-guarded cache so the base LF and the batch extract *once*
  per state, not once each. In a cold SA pass the base LF's last extraction (post outer loops) is
  exactly the state the batch needs, so the batch's extract is a cache hit. Invalidation is
  exhaustively conservative (every index/network event clears the entry).

- **`GpuAcNewtonSolver`** — the JNI binding to `libolfgpu`, and the NVIDIA-library preloader
  (`libcublasLt → libcublas → libcudss` from `CUDSS_ROOT`/`olf.cudss.path`). `isAvailable()` is the
  global gate. This is also where `solveTransposedExplicit` (the adjoint/VJP transpose solve) lives.

### 2b. The native layer (`native/cuda/`)

`libolfgpu` keeps the fixed-pattern CSR Jacobian, the mismatch vector, and the state resident on the
device. Four **fused fill kernels** (`closed_branch.cuh`, `open_branch.cuh`, `shunt.cuh`,
`hvdc.cuh`, plus the `_i` current-variant headers) fill `J` and `F` each iteration; cuDSS analyzes
once and refactorizes/solves per iteration; a kernel applies the state update. The kernels are
transpiled from OLF's hand-written equation Java by `codegen/java2cuda` — no symbolic math.

**Jacobian orientation (matters for the adjoint work, #40):** the GPU builds *true CSR* (row =
equation, col = variable), so a plain cuDSS solve is the *forward* Newton direction. powsybl-math
stores CSC-as-CSR, i.e. its stored matrix is `M = Jᵀ`, so *its* plain `solve()` is the *adjoint*
direction and `solveTransposed()` is forward. See `docs/tasks.md` #40 for the consequences.

---

## 3. How outer loops work — CPU baseline, then GPU

### 3a. The CPU outer-loop framework (what we integrate with)

A stock AC load flow is `AcloadFlowEngine.run()`: an initial Newton-Raphson solve, then a
`do…while` loop over an *ordered* list of `AcOuterLoop`s (`DistributedSlack`, `SecondaryVoltage`,
`VoltageMonitoring`, `ReactiveLimits`, `PhaseControl`, `TransformerVoltageControl`,
`ShuntVoltageControl`, `AutomationSystem`, …). Each loop's `check()` returns `STABLE`, `UNSTABLE`,
or `FAILED`; an `UNSTABLE` result re-runs the AC solver from the previous state, and the whole
sequence repeats until every loop reports stable or the outer-iteration cap is hit.

The outer loop is a **fixed-point iteration wrapped around the Newton-Raphson inner solve.** That
framing is what the GPU has to reproduce — and it does so differently in the two modes.

### 3b. Mode 1 — base load flow: GPU inner solve, **CPU outer loops unchanged**

`GpuNewtonRaphson` is an `AcSolver`. It replaces **only the inner Newton-Raphson** with one JNI
call; the entire `AcloadFlowEngine` and every outer loop above it run on the CPU exactly as stock.
When an outer loop returns `UNSTABLE`, OLF calls the solver again — and that call goes to the
device. Activation events (PV→PQ, slack change) only refresh the row-mode mask; structural events
(topology, branch parameters) force a re-extract + device-context rebuild. The final
`CONVERGED / MAX_ITERATION_REACHED` verdict is decided by the *configured* stopping criteria,
evaluated on the CPU from the written-back state — identical to the CPU NR. On any extraction/device
failure it falls back to the CPU `NewtonRaphson` (counted in `CPU_FALLBACK_COUNT`).

**So mode 1 reuses the entire CPU outer-loop machinery.** The device is just a faster inner solve.

### 3c. Mode 2 — batched N-1: outer loops **reimplemented on/around the device**

The batch cannot use the CPU outer loops, because those iterate one network at a time and the whole
point is to solve *hundreds of post-contingency networks at once* over one shared factorization
pattern. `GpuBatchedSecurityAnalysisSolver` therefore reproduces the outer-loop fixed points itself,
in two sub-modes:

- **Device-resident: distributed slack + reactive limits.** These run *inside* the batched Newton
  loop in native code (per scenario, with per-scenario convergence retirement). The row-mode mask
  provides the PV↔PQ reactive-limits switching; a per-scenario distributed-slack reweight handles a
  generator loss at a participant bus. Tolerances are taken from the run's parameters
  (`batchReactiveLimitTol`, `batchSlackDistTol`) so the device's switch *decisions* match the CPU's.

- **Host-orchestrated: transformer voltage control** (`orchestrateTransformerControl`). This is the
  clearest illustration of the pattern: the host runs the outer-loop *logic* (which controls are
  out of deadband? clamp out-of-range continuous ratios, then round to the nearest tap) while each
  step's *re-solve* is a batched device call (`solveBatchContext`). It is the CPU
  `TransformerVoltageControlOuterLoop` fixed point, hoisted out of the device and driving the batch.

Anything mode 2 does not model on the device is **detected and routed to the CPU per-contingency
path** (see §5). The scope line: device DS + RL + AFTER_GENERATOR transformer control; everything
else falls back.

---

## 4. Supporting solver-layer classes

- **`GpuBatchContingencyValidator`** — per-scenario qualification: does this outage keep the main
  synchronous component solvable over the fixed superset pattern? Rejects what the mask cannot
  represent (`ISLANDS_SLACK`, `FRAGMENTS_NETWORK`, `NOT_IN_GPU_MODEL`). In full-loadflow mode
  fragmenting outages are *accepted* and their islanded buses frozen to identity at 1∠0.

- **`GpuBatchElementMapper`** — maps `LfBranch`/`LfShunt` to the element index used by the
  per-scenario disabled mask, re-enumerating in the *same order* the extractor packs them (closed
  branches, then open, then base-open closables; zero-impedance branches are a separate family and
  deliberately map to −1 so a bus-coupler outage routes to the CPU).

- **`GpuNewtonRaphsonFactory`** — the trivial `AcSolver` SPI factory (name `GPU_NEWTON_RAPHSON`).

- **`BatchConfig`** (record inside the solver) — the run's outer-loop parameterization
  (`lsMode/maxDv/maxDphi`, `transformerControl`, `maxOuterIterations`, `maxPqPvSwitch`,
  `dsRejectResidual`), threaded **per call** from the provider down to the device. It replaced
  JVM-global static config whose leak between a provider run and a direct-solver run caused a
  full-suite failure — see its javadoc.

---

## 5. How security analysis is implemented — **and what it reuses from the CPU path**

This is the heart of the question. **The GPU SA reuses almost the entire CPU security-analysis
engine.** `GpuBatchedAcSecurityAnalysis extends AcSecurityAnalysis` and overrides just a few seams;
everything else — the contingency loop, connectivity handling, `NetworkState` save/restore, limit
violation detection, result building, and operator-strategy *condition evaluation* — is stock OLF,
inherited unchanged.

What the stock CPU AC SA does per contingency: apply the `LfContingency` to the shared `LfNetwork`,
run a **complete `AcloadFlowEngine.run()`** (full NR + full outer-loop sequence), detect violations,
build the result, then `NetworkState.restore()`. It is a full re-solve per contingency — there is
**no** Woodbury/sensitivity shortcut in AC SA (that exists only in the *DC* path,
`WoodburyDcSecurityAnalysis`).

The GPU replaces **only that per-contingency full re-solve**, at three intercept seams:

1. **`runSimulations`** — captures the network + contingency list, decides whether the batch is
   viable (the outer-loop guard, §5b), and optionally swaps the base LF's solver to
   `GpuNewtonRaphson` so the one-time CUDA init is absorbed into the base phase (pre-warm).

2. **`afterPreContingencySimulation`** → **`precomputeBatch()`** — from the converged base state,
   solves *all* qualified single-branch (and combos: branch+gen+load+shunt+tap) contingencies in
   one `GpuBatchedSecurityAnalysisSolver.solveBatchN1(...)` call, right after the base LF. Results
   are cached by contingency id. Crucially it **reuses the engine's own equation system and target
   vector** — creating a second system would re-point every bus/branch evaluable off the engine's
   system and break fall-through CPU solves.

3. **`runPostContingencySimulation`** (and **`runActionSimulation`** for strategies) — for a
   contingency the batch solved, **inject** the precomputed device state into the equation system
   (`injectState` + `AcSolverUtil.updateNetwork`) instead of running Newton-Raphson, then hand off
   to the **inherited** violation detection and result building. For a contingency the batch did not
   solve, call `super.…` — the stock CPU path, unchanged.

So the reuse boundary is sharp:

| Concern | GPU SA |
|---|---|
| Contingency iteration, `NetworkState` save/restore | **reused** (inherited from `AcSecurityAnalysis`) |
| `LfContingency.apply`, connectivity, `PropagatedContingency` | **reused** |
| Limit-violation detection, `PostContingencyNetworkResult`, result building | **reused** |
| Operator-strategy *condition* evaluation (`checkCondition`) | **reused** (engine evaluates it before our override runs) |
| The per-contingency **numeric solve** | **replaced** — batched device solve, state injected |
| DS / RL / transformer-control **outer loops** during that solve | **reimplemented** on/around the device (§3c) |

The honest one-liner: *the GPU SA does not reuse the CPU per-contingency **solver**, but it reuses
essentially everything else in the CPU security-analysis engine.* The batched solver and its
device-side outer loops are the genuinely new code; the SA integration is a thin subclass.

### 5a. State injection (why not just `setV`/`setAngle`)

Injection remaps the device state (base row layout) variable-by-variable through a snapshot of the
base layout, because an islanding `LfContingency.apply()` disables buses, deactivates their
variables, and *renumbers rows* — a raw index-wise copy would shift every variable after the
disabled bus. Islanded buses (frozen to 1∠0 on the device) are excluded from results by
**result-space filtering** (`filterIslanded`), never by `LfBus.setDisabled` on the live network
(that would fire equation-system listeners and corrupt the next scenario's injection).

### 5b. The safety contract: detect-and-fallback

Enabling the GPU must **never change results** — only accelerate them. Every place the device model
is narrower than OLF, the code detects it and routes to `super` (the stock CPU path):

- **Outer-loop guard** (`runSimulations`): if the resolved outer-loop list contains anything beyond
  {distributed slack, reactive limits, AFTER_GENERATOR transformer control, structural-no-op HVDC
  loops, VoltageMonitoring-without-SVC}, the *whole run* stays on the CPU — a silently-dropped
  per-scenario loop was measured as thousands of violation diffs. Non-default balance types,
  `useActiveLimits=false`, and non-`LEAVE_ON_SLACK_BUS` slack-failure behaviour also disqualify.
- **Per-contingency qualification** (`GpuBatchContingencyValidator` + the scoping in `solveBatchN1`
  / `precomputeBatch`): outages the fixed pattern cannot represent (≥3 branches, partial-side open,
  bus loss, DISTR_RHO-controller or zero-impedance-branch outage, ZIP-load removal, remote/slack
  generator loss, non-linear shunt) are rejected to the CPU per contingency.
- **Operator strategies** (`composeStrategyScenario`): only single-stage TrueCondition-reachable
  strategies whose actions map to batchable deltas become extra batch scenarios; multi-stage or
  out-of-scope actions fall back. Condition semantics stay 100% stock — the engine evaluates the
  condition on CPU-detected violations *before* our override, so a precomputed pair simply goes
  unused if its condition does not pass.
- **Solver-level fallback** (`GpuNewtonRaphson`): any extraction/device failure reverts to the CPU
  `NewtonRaphson`.

### 5c. Concurrency

Mono-threaded device by design (its parallelism *is* the batch). `threadCount > 1` still parallelises
the CPU fall-through solves across partition workers; each partition's batch precompute serialises on
a global device lock, and under multithreading the base/fall-through GPU solver swap is skipped
(concurrent device solves race in native code).

---

## 6. Design choices, summarised

- **Fixed superset pattern + stable row indexing + row-mode masks.** One device context, built once,
  survives PV↔PQ switching and every non-structural change. This is what makes both the cached base
  solver and the *batched* factorization possible — every scenario shares one CSR pattern, differing
  only in per-scenario value masks, so cuDSS symbolic analysis is amortised over the whole set.

- **Reuse the CPU engine wherever it is not the bottleneck.** SA subclasses `AcSecurityAnalysis`
  rather than reimplementing it; the base LF keeps all CPU outer loops. Only the numeric solve —
  where the device actually wins — is replaced.

- **Detect-and-fallback over feature-completeness.** The device model is deliberately narrow; the
  contract is that anything outside it produces *identical* CPU results, never wrong GPU ones. Scope
  grows by moving items from "fall back" to "batched", each gated by a test.

- **Transpile, don't fork.** Kernels are generated from OLF's own equation Java, so a formula change
  in OLF regenerates the kernels rather than diverging a hand-written copy. Stock OLF is never
  modified.

---

## 7. Pointers

- Scope / capability matrix: `docs/gpu-vs-cpu-capabilities.md`
- Roadmap and the adjoint/VJP work: `docs/tasks.md`
- Performance numbers: `docs/numbers.md`
- Kernel generation: `codegen/README.md`
- cuDSS version switching / build traps: `native/cudss-swap.sh`, `native/build-gpu.sh`
