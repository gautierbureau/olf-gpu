# Hybrid outer-loop batch SA solver — production design

**Status:** design for promotion to a real SA solver on `main` (2026-08-19)
**Companion to:** [`hybrid-outer-loop-prototype.md`](hybrid-outer-loop-prototype.md) (the original idea + throughput analysis)

## 1. Goal

Promote the Java-outer-loop hybrid batch solver from a validation prototype
(`GpuHybridBatchValidationTest` + `GpuHybridOuterLoopBridge`) into a **first-class, selectable
outer-loop mode** of the production GPU batched security analysis, coexisting with the existing
on-device (CUDA-kernel) outer loops.

The hybrid reuses OLF's **validated Java** reactive-limits / distributed-slack / voltage-monitoring
outer loops for the *decisions*, and batches only the inner Newton solves on the device. Its value
is **exact CPU-fidelity by construction** — there is no second outer-loop implementation to keep in
sync with OLF core.

**End state:** JAVA supports **every OLF outer loop** — at minimum full parity with the on-device
path, and in fact a superset (real HVDC emulation, SVC, load-based distributed slack), because it
runs OLF's own loop code. Coverage lands in phases (§5.7); a temporary gate (§5.6) routes not-yet-
validated runs to CPU in the meantime, and is deleted once coverage is complete.

## 2. What we now know (validated)

Both correctness and cost are established (pegase9241, 500 single-branch N-1 contingencies):

- **Correctness parity with CPU:** 499/500 converge and match CPU. The one miss,
  `TWT-106-2546`, is a **genuine non-convergence** — the production CPU LoadFlow (DC init,
  `MAX_VOLTAGE_CHANGE`) also fails it (`MAX_ITERATION_REACHED`, residual −156 MW). All paths (CPU,
  on-device batch, hybrid) correctly fail/route it.
- **Throughput after optimization:** hybrid productive time **26.2 ms/ctg** vs on-device
  **16.9 ms/ctg** = **×1.55**. Journey: ×5.82 (naïve) → ×2.81 (cross-round compaction) → ×2.19
  (compact snapshot) → ×1.55 (straggler routing + snapshot save-skip). The remaining ×1.55 is the
  Java outer-loop work itself (state load + `updateNetwork` + OLF `check()` + activation re-extract
  per scenario per round) — the inherent fidelity cost.

### Hard-won correctness invariants (MUST be preserved in the production port)

1. **Equation system built AFTER the base LF's DC init.** OLF's DC voltage initializer attaches DC
   equations to the network that overwrite the AC equations of an already-created AC equation
   system. Building the device/bridge `EquationSystem` *before* the base run corrupts it and sends
   the inner solve of ill-conditioned outages (e.g. `LINE-26-4231`) onto a spurious high-loss branch
   (slack 6.44 pu) that no outer loop recovers. Build it after the base run.
2. **`initialTargetP` anchored at the original generation.** OLF's distributed slack accounts
   cumulatively vs `LfGenerator.getInitialTargetP()`. Anchor it once (per scenario per round) at the
   pre-base-distribution generation so `resetToInitialState` is consistent.
3. **Device state-vector scaling mirrors the CPU solver.** Pass `lsMode`/`maxDv`/`maxDphi` derived
   from `StateVectorScalingMode` (rte production = `MAX_VOLTAGE_CHANGE`). Undamped Newton (`lsMode=0`)
   makes hard outages blow up (‖F‖→1e9) where the clamped CPU stays bounded.
4. **Per-scenario outer-loop state isolation.** All scenarios share one mutable `LfNetwork`; each
   scenario's PV/PQ switches, distributed `targetP`, HVDC state must be saved/restored between
   scenarios (see §5.3).
5. **Faithful `runOuterLoop` nesting.** Drive the outer loops sequentially/nested exactly like
   `AcloadFlowEngine.runOuterLoop` (stabilize one loop before the next, re-solve only on UNSTABLE) —
   not a flat "check all loops each round". `GpuHybridOuterLoopBridge` already encodes this state
   machine.

## 3. Current architecture (integration seams)

```
GpuSecurityAnalysisProvider
  └─ GpuBatchedAcSecurityAnalysis  (extends AcSecurityAnalysis; holds AcLoadFlowContext, BatchConfig)
       ├─ GpuSaRunScope.evaluate(...)  → Scope{ BatchConfig, tolerances, ... }   ← MODE DECIDED HERE
       └─ GpuBatchedSecurityAnalysisSolver.solveBatchN1(..., BatchConfig)
            └─ GpuBatchChunkSolver     → per-chunk GpuAcNewtonSolver.solveBatchContext(...)
                                          with DS/RL packs + maxOuterIterations  ← ON-DEVICE OUTER LOOPS
```

`BatchConfig` (`record`) already carries `lsMode/maxDv/maxDphi/transformerControl/
maxOuterIterations/maxPqPvSwitch/dsRejectResidual`. Note: `GpuBatchedAcSecurityAnalysis` lives in
`com.powsybl.openloadflow.sa.gpu`, a *different* package from `com.powsybl.openloadflow.ac`, so it
**cannot** reach the package-private `AcOuterLoopContext(LfNetwork)` ctor either — production needs a
real core factory for it, not reflection (§5.4). OLF core is patched on this fork as a matter of
course (branch `olf-local-patches`), so this is a normal, small change rather than a blocker.

## 4. Target architecture

Add an outer-loop **mode** to `BatchConfig` and branch inside the solver:

```
BatchConfig{ ..., OuterLoopMode outerLoopMode }        // DEVICE | JAVA
GpuSaRunScope.evaluate → reads OpenLoadFlowParameters (new gpuOuterLoopMode) → sets the mode
GpuBatchedSecurityAnalysisSolver.solveBatchN1
   ├─ DEVICE → GpuBatchChunkSolver          (unchanged)
   └─ JAVA   → GpuHybridBatchOuterLoopEngine (NEW, promoted from the test round-loop)
```

Everything upstream of the outer-loop driving is **shared** by both modes: base LF, equation-system
build, islanding/transformer qualification, contingency→mask mapping, chunking, the inner
`solveBatchContext` call, and result assembly. Only the *outer-loop orchestration* differs.

## 5. Component design

### 5.1 `OuterLoopMode` + config plumbing
- New enum `OuterLoopMode { DEVICE, JAVA }` in `...ac.solver.gpu.batch`.
- New field on `BatchConfig`; `DEVICE_DEFAULTS` keeps `DEVICE`.
- **Selection via a system property**, matching the module's existing GPU toggles (there is *no*
  GPU parameters extension; the module reads `OpenLoadFlowParameters` for OLF-concept settings and
  uses `-Dolf.gpu.sa.*` system properties for GPU-specific operational knobs like
  `gpuBaseLoadFlow`/`gpuSequentialLoadFlow`). Add:

  ```java
  // GpuBatchedAcSecurityAnalysis
  public static final String GPU_OUTER_LOOP_MODE_PROPERTY = "olf.gpu.sa.outerLoopMode"; // DEVICE|JAVA, default DEVICE
  ```

  `GpuSaRunScope.evaluate` reads it and stamps `batchConfig.outerLoopMode`, AND — when `JAVA` is
  requested — widens the coverage guard (§5.5) to the JAVA-supported outer-loop set.
- Rationale: DEVICE-vs-JAVA is a GPU-specific *implementation* choice with no standard-OLF analogue,
  so it does not belong on `OpenLoadFlowParameters`. **Future upgrade path** if it must become a
  per-run, serializable, user-facing mode: a dedicated `GpuSecurityAnalysisParameters extends
  AbstractExtension<SecurityAnalysisParameters>` (JSON serializer + `@AutoService` + defaults) —
  not `OpenLoadFlowParameters`. Defer until `JAVA` graduates from ops/oracle toggle to a shipped
  user mode.

### 5.5 Fallback — JAVA falls back to CPU (the stock `AcSecurityAnalysis`), never to DEVICE

This reuses the **existing** GPU-batch fallback machinery unchanged; no new fallback path is built.

- **Whole-run** (mode-selection time, in `GpuSaRunScope`): if the run's configured outer loops /
  scenario types are outside JAVA's validated coverage (§7), set `scope.batchDisabled()`. The
  provider then behaves *exactly like the stock CPU `AcSecurityAnalysis`* via `super.runSimulations`
  — this is the already-shipped "safety contract" the DEVICE batch uses when its own outer-loop
  guard trips. Logged with the specific unsupported loop.
- **Per-scenario** (a single contingency the JAVA driver cannot converge, e.g. `TWT-106-2546`, or
  one routed early by the straggler rule): mark it `needsCpuFallback`, exactly as the prototype's
  `routedByBudget` does. It then finishes on CPU through the existing per-contingency path
  (`GpuStateInjector.injectState` of the base/last state + `super.runPostContingencySimulation`),
  identical to how the DEVICE batch already handles a non-precomputed scenario.

**Why CPU and not DEVICE** — the decision, with rationale:
1. **Fidelity consistency.** JAVA exists precisely to avoid trusting a second (CUDA) outer-loop
   implementation. Falling back to DEVICE would silently reintroduce exactly that risk on the runs
   where JAVA bailed. CPU is the ground truth — the only fallback that never weakens the guarantee.
2. **Reuse of a tested path.** `batchDisabled → super.runSimulations` and `GpuStateInjector →
   super.runPostContingencySimulation` already exist and are validated as the DEVICE fallback.
   JAVA borrows them verbatim; DEVICE-inside-JAVA would be new, untested plumbing.
3. **DEVICE is already its own explicit mode.** A user who wants GPU throughput without the fidelity
   guarantee selects `DEVICE` directly; JAVA→DEVICE would blur two deliberately distinct choices.

**Rejected alternative (documented):** a `JAVA → DEVICE → CPU` cascade would keep runs on the GPU
when DEVICE covers an outer loop JAVA does not yet (e.g. transformer VC). It is rejected as the
default because it undermines reasons (1) and (3). It could be offered later as an explicit,
opt-in "speed-priority" fallback flag, but only after DEVICE's coverage of that loop is itself
validated against CPU on the network in question.

### 5.2 `GpuHybridBatchOuterLoopEngine` (NEW, promoted from the test)
Owns the round loop that currently lives in `GpuHybridBatchValidationTest`. Responsibilities:
- Build the per-scenario `GpuHybridOuterLoopBridge` list from OLF's real outer loops
  (`acParameters.getOuterLoops()`), each with a proper `AcOuterLoopContext` created via core APIs
  (no reflection). One context set **per scenario** so switch/redistribution bookkeeping never
  contaminates across scenarios.
- Run the round loop with the four production optimizations:
  - **Cross-round compaction** — recreate the device batch context (cheap; empty DS/RL packs) sized
    to the active-scenario count each round; solve/scatter only the active subset.
  - **Compact per-scenario snapshot** — §5.3.
  - **Snapshot save-skip** — only snapshot a scenario that continues to the next round.
  - **Straggler early-routing** — route a scenario to CPU after *k* consecutive device rejects
    (`status = -2`) or on exceeding `maxOuterLoopIterations`.
- Return per-scenario `BatchScenarioResult` (converged state or `needsCpuFallback`) — the **same
  contract** the on-device path returns, so downstream assembly is unchanged.

### 5.3 `HybridScenarioState` (compact per-scenario snapshot)
Replaces the prototype's `NetworkState` (which allocates `BusState` for **every** bus +
`BranchState` for **every** branch, ~25 000 objects per save). Captures only what the active outer
loops mutate:
- OLF `BusState` over **generator voltage-control controller buses only**
  (`network.getControllerElements(VoltageControl.Type.GENERATOR)`, ~1600 vs ~9000) — the sole buses
  reactive limits can switch PV↔PQ (captures VC-enabled, Q-freeze, `qLimitType`, control types).
- `HvdcState` for the HVDC AC-emulation loop.
- Generator `targetP` (managed as a flat `double[]` — `BusState` does not capture it).
- **Branch disable handled incrementally** — re-enable the previous scenario's outage, disable the
  current (no per-branch snapshot; no transformer VC ⇒ taps never change; loads static under
  generation-based DS).

Correct because bus V/φ come from the device state each round (overwritten by `updateNetwork`).
Measured 3.9× cheaper than the full `NetworkState`.

### 5.4 OLF-core touch point — outer-loop-context factory (drops the reflection)
`AcOuterLoopContext`'s constructor is **package-private** to `com.powsybl.openloadflow.ac`, and the
GPU module lives in `com.powsybl.openloadflow.sa.gpu` — a *different* package — so it cannot reach
the ctor without reflection (the prototype's `getDeclaredConstructor(...).setAccessible(true)`).
This is a maintenance liability in shipped code (breaks silently on a core ctor change; may be
denied under JPMS / a SecurityManager; duplicates the init sequence).

Patching OLF core is the established mode of work on this fork (branch `olf-local-patches` already
carries small exposures like *Expose Newton-Raphson convEpsPerEq* and *Add
PropagatedContingency.getShuntIdsToShift() accessor*), so add a public factory that returns the
fully-initialized per-loop contexts — the *exact* sequence `AcloadFlowEngine.run()` runs internally:

```java
// com.powsybl.openloadflow.ac
public static List<Pair<AcOuterLoop, AcOuterLoopContext>> createOuterLoopsAndContexts(AcLoadFlowContext context) {
    return context.getParameters().getOuterLoops().stream()
        .filter(ol -> ol.isNeeded(context))
        .map(ol -> {
            AcOuterLoopContext ctx = new AcOuterLoopContext(context.getNetwork());
            ctx.setLoadFlowContext(context);
            ol.initialize(ctx);
            return Pair.of(ol, ctx);
        })
        .toList();
}
```

`GpuHybridBatchOuterLoopEngine` calls this **once per scenario** to build that scenario's own
outer-loop `(loop, context)` list — no reflection, and provably in lockstep with core because it is
core. It also adopts the `isNeeded(context)` filter (the prototype omitted it). Placement: a public
static on `AcloadFlowEngine`, or a tiny `AcOuterLoops` helper in the same package.

- Optional second touch point: expose the RL controller-bus set + a lightweight bus state
  save/restore so the compact snapshot (§5.3) lives in core rather than the GPU module.

### 5.6 Coverage gate in `GpuSaRunScope` (TEMPORARY — shrinks to empty as coverage lands, §5.7)

The endgame is **full outer-loop coverage** (≥ the on-device path — §1). The gate is the safety
mechanism *during* the phased rollout: any outer loop / config not yet validated with the JAVA
driver routes the whole run to CPU (§5.5), never a silent mis-solve. It replaces the DEVICE
`unsupportedOuterLoops` computation when `outerLoopMode == JAVA`:

```java
/** Outer loops the JAVA driver is VALIDATED to run today (endgame: all of them, §5.7). JAVA runs
 *  OLF's own loop code, so for these there is NO device-port parameterization limit — the only
 *  constraints are (a) not-yet-validated loops and (b) config that mutates network state the compact
 *  per-scenario snapshot (controller buses + HVDC + generator targetP) does not yet capture. */
private static final Set<String> JAVA_SUPPORTED_LOOPS = Set.of(
        DistributedSlackOuterLoop.NAME, ReactiveLimitsOuterLoop.NAME, "VoltageMonitoring",
        "AcHvdcAcEmulationLimits", "FreezingHvdcACEmulation");

private static List<String> javaUnsupportedOuterLoops(LfNetwork net, AcLoadFlowParameters acParameters,
                                                      LoadFlowParameters lfParams) {
    boolean noHvdc = net.getHvdcs().isEmpty();
    boolean noSvc = net.getBuses().stream().flatMap(b -> b.getGenerators().stream())
            .noneMatch(g -> g instanceof LfStaticVarCompensator);
    List<String> unsupported = new ArrayList<>();
    for (AcOuterLoop ol : acParameters.getOuterLoops()) {
        String name = ol.getName();
        boolean ok = JAVA_SUPPORTED_LOOPS.contains(name)
                // conservative structural no-ops: allow the HVDC/SVC-touching loops only when they cannot
                // fire yet (widen once JAVA's HVDC/SVC handling is validated — the snapshot already captures
                // HvdcState, and SVCs are generator-voltage-control buses already in the controller set):
                && !("VoltageMonitoring".equals(name) && !noSvc)
                && !(("AcHvdcAcEmulationLimits".equals(name) || "FreezingHvdcACEmulation".equals(name)) && !noHvdc);
        if (!ok) {
            unsupported.add(name);
        }
    }
    // Distributed slack: the snapshot isolates generator targetP but NOT loads, so only GENERATION-based
    // balance is safe today. JAVA needs NEITHER the device's PROPORTIONAL_TO_GENERATION_P_MAX-only restriction
    // NOR useActiveLimits=true — it runs OLF's ActivePowerDistribution with whatever limits config.
    if (acParameters.getOuterLoops().stream().anyMatch(ol -> DistributedSlackOuterLoop.NAME.equals(ol.getName()))) {
        LoadFlowParameters.BalanceType bt = lfParams.getBalanceType();
        boolean generationBased = bt == LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P
                || bt == LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX
                || bt == LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN
                || bt == LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR;
        if (!generationBased) {
            unsupported.add("DistributedSlack[balanceType=" + bt + " mutates loads; snapshot omits them]");
        }
    }
    return unsupported;
}
```

Wired into `evaluate`:

```java
OuterLoopMode mode = readOuterLoopMode();   // olf.gpu.sa.outerLoopMode, §5.1
List<String> unsupportedOuterLoops = mode == OuterLoopMode.JAVA
        ? javaUnsupportedOuterLoops(lfNetwork, acParameters, lfGuardParams)
        : deviceUnsupportedOuterLoops(...);  // the existing DEVICE filter, unchanged
boolean batchDisabled = !unsupportedOuterLoops.isEmpty();
```

**Key differences from the DEVICE gate** (all because JAVA runs OLF's real loop code):
- No `PROPORTIONAL_TO_GENERATION_P_MAX`-only restriction — any generation-based balance works.
- No `useActiveLimits == true` requirement.
- The only *new* JAVA restriction vs DEVICE is `TransformerVoltageControl`: DEVICE batches it (via
  `GpuBatchTransformerControl`), JAVA does not yet (§5.7) — the one place DEVICE currently out-covers
  JAVA, and the reason the JAVA→DEVICE cascade was tempting (§5.5, rejected).

### 5.7 Path to full coverage (delete the gate row by row)

JAVA reaches full parity-and-beyond mainly by (1) widening the compact snapshot's captured fields
and (2) feeding the matching per-round overrides into the inner solve — **not new kernels**:
`solveBatchContext` already accepts shunt overrides (`shOvSlotElem/shOvGb`), tap/r1 overrides
(`r1OvSlotElem/r1OvVal`), and RL-state in/out (`rlStateIn/rlStateOut`) — the same override channels
the on-device path uses.

| Outer loop / config | State it mutates | To support in JAVA | Device override |
|---|---|---|---|
| DistributedSlack (generation-based) | gen `targetP` | done | — |
| ReactiveLimits | controller-bus PV/PQ + Q-freeze | done | — |
| HVDC AC-emulation (real HVDC) | `HvdcState` | validate end-to-end (snapshot already captures it) | HVDC handled in inner solve |
| VoltageMonitoring + SVC | SVC = controller bus | validate (SVC already in controller set) | — |
| DistributedSlack (load-based) | load `targetP/Q` | add load state to the compact snapshot | — |
| **TransformerVoltageControl** | tap `r1/a1` | add tap state to snapshot **+** pass changed taps each round | `r1Ov*` (exists) |
| ShuntVoltageControl | shunt `B/G` | add shunt state to snapshot **+** pass changed shunts each round | `shOv*` (exists) |
| SecondaryVoltageControl, AreaInterchange, … | pilot/area targets, controller Q | analyze mutated state, extend snapshot, validate | assess |

As each row is validated, its entry leaves `JAVA_SUPPORTED_LOOPS` restrictions / the guards, and the
gate shrinks. When every OLF outer loop is covered, `javaUnsupportedOuterLoops` returns empty for
all runs and the gate is removed — the target end state where JAVA supports every outer loop the
on-device path does (and, via the snapshot extensions, several it does not: real HVDC emulation,
SVC, load-based DS).

## 6. Coexistence & selection guidance
Two GPU-batch modes plus the always-available CPU safety net, in speed order:

| Mode | Outer loops | ms/ctg | Fidelity | Falls back to |
|---|---|---|---|---|
| `DEVICE` (default) | CUDA-kernel port | 16.9 | port must be kept faithful | CPU (`batchDisabled`) |
| `JAVA` | OLF's own Java loops | 26.2 (×1.55) | exact by construction | **CPU** (§5.5), never DEVICE |
| CPU (`AcSecurityAnalysis`) | OLF's own Java loops, per-contingency | slow | ground truth | — |

- **`DEVICE`**: pick for peak throughput; accepts the maintenance burden of keeping the CUDA outer
  loops in sync with OLF core.
- **`JAVA`**: pick when (a) fidelity to a specific OLF outer-loop behavior matters more than peak
  throughput, (b) a network exercises a loop where the CUDA port has a known/suspected gap, or
  (c) as an in-production **oracle** to validate the device port on real networks. It still gives a
  large speedup over per-contingency CPU while guaranteeing CPU-identical decisions.
- Both GPU modes degrade to **CPU** — never to each other — on an unsupported outer loop or a
  non-convergent scenario (§5.5).

## 7. Coverage roadmap (the bulk of the remaining work)
Full outer-loop coverage is the goal (§1); it lands in phases behind the temporary gate (§5.6),
each phase = extend the compact snapshot + feed the existing inner-solve overrides + validate
(§5.7), then relax the gate.
- **Phase A (done / validate):** DS (generation-based) + RL — validated on pegase. Real HVDC
  emulation and SVC/VoltageMonitoring: the snapshot mechanism already covers them; needs an
  end-to-end validation network before relaxing their structural guards.
- **Phase B — device parity:** `TransformerVoltageControl` (tap state in snapshot + `r1Ov*`
  overrides). This is the one loop DEVICE batches and JAVA does not yet; closing it reaches parity.
- **Phase C — beyond device:** load-based DS (load state in snapshot), `ShuntVoltageControl`
  (shunt state + `shOv*`), then secondary voltage control / area interchange.
- Each phase also needs the "keep one bus PV" / voltage-dependent Q-limit re-pin paths exercised —
  they are OLF's own code here (so correct by construction), but were the two latent gaps in the
  device port that motivated the hybrid, so assert them explicitly.
- **N-2 / operator strategies / injection & shunt-shift contingencies.** Prototype covers
  single-branch N-1; the device path already handles more (islanding freeze, injection outages).
  The Java driver must reach the same scenario coverage or defer those scenarios via the gate.
- **Concurrency / context reuse.** The device batch context is single-threaded; confirm the round
  loop's per-round context recreation (compaction, §5.2) interacts correctly with
  `GpuBatchContextCache`.

## 8. Phased implementation plan
1. **Extract & unit-test the driver.** Move the round loop from the test into
   `GpuHybridBatchOuterLoopEngine` + `HybridScenarioState` in `main`, with the four optimizations.
   Keep `GpuHybridBatchValidationTest` as the byte-level regression (it already asserts the
   `LINE-26-4231` trajectory and 499/500).
2. **Remove reflection.** Add the core outer-loop-context construction API; wire the engine to it.
3. **Config plumbing.** `OuterLoopMode` on `BatchConfig`, the parameter, `GpuSaRunScope` selection,
   and the outer-loop-coverage fallback gate.
4. **Solver branch.** `GpuBatchedSecurityAnalysisSolver.solveBatchN1` dispatches DEVICE/JAVA; JAVA
   returns the same `BatchScenarioResult` contract.
5. **End-to-end SA test.** Run the full `GpuSecurityAnalysisProvider` in `JAVA` mode on pegase and
   assert violation-set identity with CPU (reuse the `GpuVsCpuProductionSaBenchmarkTest` harness at
   shared 1e-4).
6. **Perf gate.** Add the head-to-head (DEVICE vs JAVA, warm + cold) to the bench suite; alert if
   the gap regresses past ~×2.
7. **Docs + default.** Ship with default `DEVICE`; document `JAVA` and its coverage envelope.

## 9. Validation plan
- **Correctness:** `JAVA`-mode SA violation set == CPU on pegase at shared 1e-4 (the strict check);
  the fixed-iter K=4 bench for raw device throughput; the byte-level `LINE-26-4231` trajectory
  regression; a genuine-non-convergence case (`TWT-106-2546`) asserted as routed on all paths.
- **Both benches every change** (fixed-iter K=4 + split-bench), cold-first reporting — per the
  standing bench discipline.
- **Coverage matrix:** one network per outer-loop feature in §7, DEVICE vs JAVA vs CPU.

## 10. Risks & open questions
- Closing the outer-loop coverage matrix is the bulk of the real work; the round-loop mechanics are
  done and measured.
- The OLF-core factory (§5.4) is a small local patch on `olf-local-patches` (same style as the
  existing exposures on that branch) — no upstream-acceptance question blocks it. Only open choice:
  place it on `AcloadFlowEngine` vs a new `AcOuterLoops` helper.
- Whether `JAVA` ships as a user-facing mode or first as an internal validation oracle; the latter
  de-risks GA while still delivering the fidelity guarantee for the device port.
