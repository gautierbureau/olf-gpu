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

---

## 11. Dynamic Q-relocation (the shared-structure fix)

### 11.1 The defect this fixes
A voltage-controlled bus's **reactive-balance (Q) row is relocated** off its own V row onto the
controller's control variable when control is active:
- transformer VC → the controller branch's `RHO1` row,
- shunt VC → the controller shunt's `SHUNT_B` row,
- generator remote/pilot VC → a `DISTR_Q` row.

The relocation is what keeps the Jacobian non-singular while V is pinned by the control target. Its
destination is **frozen at extraction** into the family index packs: for a closed branch it is
`cbIdx[1]`/`cbIdx[3]` (the two endpoints' Q rows), which `resolveCb` reads as
`ix[rowEnd*2 + rowEq]` and `buildDest` bakes into fixed CSR slots inside `createStructure`. `shIdx`
does the same for shunts.

The batch shares **one** structure across all scenarios and all outer-loop rounds
(`GpuBatchedSecurityAnalysisSolver` calls `createStructure` once at ~L449 and reuses it across
`MAX_ROUNDS`; only `bridges[s].applyOuterLoopStep` re-extracts the per-scenario *activation* /
row-mode, never the structure). So when an outer loop **toggles a control on↔off per scenario or per
round** — transformer VC moving between its continuous and rounded phases, shunt VC changing section,
a bus losing control at a reactive limit — the frozen Q destination becomes wrong for that scenario.
The solve still converges, but on a subtly wrong equation system.

Measured (fork, "Find batched-transformer Jacobian defect"): a FRESH structure agrees with CPU to
`4.68e-17` at the relocated row; the STALE shared structure lands `1.98e-2` off there, surfacing as
the ~5e-3 kV TVC SA residual. The **same** frozen-destination limit is why on-device shunt VC via the
transformer mirror is infeasible (a one-terminal shunt has no second terminal to hold its control
column, so `SHUNT_B` freezes at warm-start). Dynamic Q-relocation is the single fix for the whole
class.

### 11.2 Core idea
Stop freezing the destination. For every controllable bus there are exactly **two** candidate Q rows:
its own V row (control OFF) and the controller's control-variable row (control ON). Make the batch:

1. **Pattern the union** of both candidate rows into the shared CSR structure, so a valid slot exists
   for whichever row is active.
2. **Pick the destination per scenario at fill time** — scatter each controlled-bus q term (and its
   Jacobian entries) to whichever candidate row is *free* this scenario; the other candidate is an
   identity row that round, and its stray zero-contributions are overwritten by the existing row-mode
   mask.

The per-scenario selector already exists: it is the **row-mode** (which of the two rows is mode-1 /
free vs. mode-0 / identity). The activation the hybrid re-extracts every round already carries it. So
no new decision logic is introduced — only a runtime choice of which pre-patterned slot to write.

### 11.3 The change is anchored in `collectPattern` / `resolve*` / `buildDest`

The whole relocation lives in one native triad: `collectPattern`, the `resolve*` functions
(`resolveCb` / `resolveOb` / `resolveSh`), and `buildDest`. All three call the *same* `resolve`, and
its scope is exactly **one** row per Q output:

```c
// resolveCb — pack layout: P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col
rc.row = ix[s.rowEnd * 2 + s.rowEq];   // Q-role output (rowEq==1) -> ix[1] or ix[3] = FROZEN Q row
```

`ix[rowEnd*2+1]` is whatever `computeQRow` froze. **It relocates to the control-var row (rho / SHUNT_B
/ a free controller's row) only when `vActive && qActive`** — i.e. only when the control is *active in
the base extraction*. `collectPattern` emits a pattern entry only at that frozen row; `buildDest`
bakes `pos.at(rc.row, rc.col)` into one static slot the fill scatters to. That single-row scope is the
entire bug.

**The alternate candidate depends on the base's control state — this decides whether extraction
changes.** There are two regimes, and which one the batch is in must be *verified first* (the base LF
leaves transformer/shunt controls **rounded off** in the SA case per the tight-`convEpsPerEq`
finding, so regime B is the one that bites):

- **Regime A — base control ON (frozen = control-var row).** The bus's own V column is already in the
  pack at `ix[4+rowEnd*2+1]` (`ix[5]`/`ix[7]` for a closed branch; `ix[2]` for a shunt). The alternate
  is derivable with **no extraction change** — union the frozen row with `ix[4+rowEnd*2+1]`.
- **Regime B — base control OFF (frozen = bus V row).** The controller's control-var row is **not
  referenced by the feeding branches' packs** — only the controller element's own pack carries it
  (`ix[9]` = `branchRhoCol`, `shIdx[3]` = `shuntBCol`). So `ix[4+rowEnd*2+1]` == the frozen bus V row
  and the union dedups to nothing. Reaching the control-var row as the alternate needs `computeQRow`
  to compute it **unconditionally** (it already computes `rhoRow` — just gated behind
  `vActive && qActive`) and thread it into the feeding packs as a second Q-row field. **This is a
  small, contained extraction change** (one extra int per relocatable endpoint), not the wider rewrite
  §11 first implied.

**Precondition for either regime — the control-var COLUMN must exist in the base structure.** OLF
creates `BRANCH_RHO1` / `SHUNT_B` variables from control *capability*, and outer loops toggle equation
*activation*, not variable creation — so the column normally persists even when the base control is
rounded off, and the control-var row self-pairs via the controller's `ix[9]`/`shIdx[3]` claim in
`buildRowMap`. **Must-verify:** confirm no path drops the variable when the base is control-off; if one
does, dynamic relocation is impossible there without pre-allocating the column (growing `n`), and that
case must fall back to CPU.

So the fix is a widening of the native triad's scope (plus, in regime B, one extraction int), in this
order:

**A. `collectPattern` (the union — the crux).** For each Q-role output, emit at **both** candidate
rows (frozen row + alternate). `std::unique` dedups when they coincide. This is the gating step:
without the alternate slot in the CSR pattern, `buildDest`'s `pos.at()` for that row would miss and the
fill could not target it.

**B. `buildDest` (two dests per output).** Emit a dest slot for **both** candidate rows per Q output
(two ints where there is one today), both now guaranteed present by (A).

**C. Fill kernels (per-scenario selection).** `fillClosedBranchBatch` / `fillShuntBatch`, their `…F`
residual-only variants, and the single `fillClosedBranch` / `fillShunt` for parity: read the
per-scenario row-mode of the two candidate rows, scatter q and its Jacobian to the **free** one's
dest, skip the identity one (its stray zeros are overwritten by the row-mode mask). The selector is
the row-mode the activation already carries — no new upload.

**D. `buildRowMap` bijection** — see the dedicated analysis in §11.3.1.

### 11.3.1 Row-map bijection — risk analysis

`buildRowMap` builds a static bijection `rowMap: row → col` (one distinct pivot column per equation
row), validated by two throws: *conflicting diagonal pairing* (a row paired to two different columns)
and *not a bijection* (a column claimed by two rows). `buildStructure` rejects the system if either
fires or if any row is left unpaired. The union must keep this a valid bijection in **both** control
layouts of the one shared structure.

> **EMPIRICALLY CONFIRMED (2026-08-20, `DynQRelocationProbeTest`, no native build).** Extracting the
> packs in both the control-off and control-on layouts of the same network and running a faithful Java
> port of `buildRowMap` on each: `rowMap` is a **valid bijection in both** and **byte-identical across
> the two layouts** — for a single local transformer VC *and* for the multi-controller shared-remote
> `DISTR_RHO` case (where control 1 is the free relocation target and control 2 carries `DISTR_RHO` in
> the ON layout). The control-var columns (`branchRhoCol` = 4, 7) persist in the OFF layout, pinned by
> `BRANCH_TARGET_RHO1`. So the precondition holds and **`buildRowMap` needs no change** — the shared
> static pivot assignment already serves both layouts; the union only has to supply the off-diagonal
> entries.

**Finding — for the simple 1-controller case the bijection is already layout-invariant, so
`buildRowMap` needs *no* change.** The two pairs it produces for a controlled bus are the *same set*
in both states — only the *meaning* of each diagonal swaps:

| | control ON (Q on rho row) | control OFF (Q on bus V row) |
|---|---|---|
| `(busV, busV)` | v−target **identity** (claimed by `claimMovedV`) | dq/dv **Q strong-diagonal** (claimed by `pairRc(ix[1],ix[1])`) |
| `(rho, rho)` | dq/dr1 **Q strong-diagonal** (claimed by `pairRc(ix[1],ix[1])`/`ix[9]`) | r1−target **identity** (claimed by `ix[9]`) |

Both `(busV,busV)` and `(rho,rho)` are claimed **unconditionally** — the rho self-pair comes from the
controller's `ix[9]` (`shIdx[3]`) claim, which fires whenever the control-var column exists,
independent of where Q currently sits; the busV self-pair comes from either `pairRc(ix[1],ix[1])`
(OFF) or `claimMovedV` (ON). `pairRc` is idempotent and both claims resolve to the same column, so the
union never trips the conflict throw. **The bijection is therefore invariant to the toggle** — the
same `rowMap` serves both layouts, and the fill's per-scenario row-mode decides which diagonal is the
1.0 identity vs. the real Q derivative. This drops the piece I originally flagged as highest-risk down
to *no logic change* for local transformer VC, shunt VC, and single-feeder buses.

**Where the real bijection risk lives — multi-controller distribution and the compounding toggles:**

1. **`DISTR_RHO` / `DISTR_SHUNT_B` / `DISTR_Q` (several controllers on one pilot bus).** `computeQRow`
   relocates Q onto the *free* controller — "the one whose `DISTR_*` is inactive." The non-free
   controllers carry `DISTR_*` rows that self-pair on their own control column. When the outer loop
   changes **which** controller is free (or activates/deactivates a `DISTR_*` row), the *identity of
   the free row* moves between controllers — a different pivot column per layout. The union must
   pattern **every** controller's candidate row and `buildRowMap` must pair them without double-claim
   in each reachable layout. This is the genuine case where the current single-layout bijection can
   fail, and it needs per-layout validation.
2. **Remote pilot bus with a structurally zero Q diagonal.** `computeQRow`'s comment notes the
   remote-controller case leaves "the pilot-Q diagonal zero → cuDSS pivoting." Unioning a second
   candidate row for such a bus must not break the pivot's ability to find a stable diagonal in either
   layout — verify cuDSS still factorizes both.
3. **Reactive-limit toggle compounding the control toggle.** A generator controller switched to PQ by
   reactive limits *reactivates* `BUS_TARGET_Q` (the `qActive` branch in `computeQRow`) and the bus
   recovers its own busV-backed Q row. This is a **third** state on the same bus, orthogonal to the
   VC on/off toggle — the union must cover the busV row here too (it already does, since busV is a
   candidate), but the row-mode selector must encode all three reachable states, not just two.

**Bijection validation (the specific gate for piece D):** dump `rowMap` for the continuous
(control-on) and rounded (control-off) activations of the *same* extracted structure and assert (a) no
`buildRowMap` throw in either, (b) identical pair-sets on the simple case, (c) consistent, conflict-
free pair-sets on a multi-controller `DISTR_RHO`/`DISTR_Q` case, (d) cuDSS factorizes both layouts.
Only after (a)–(d) hold do the fill-kernel selection changes (C) become meaningful.

**E. Consumers**
- **JAVA hybrid (`GpuHybridBatchOuterLoopEngine`)** — no change: its per-round row-mode already
  toggles; with dynamic relocation the Q simply follows, and the residual vanishes at *default*
  tolerance (no need for the tight-`convEpsPerEq` crutch).
- **DEVICE orchestrator (`GpuBatchTransformerControl.orchestrateTransformerControl`)** — can drop the
  `deltaModesOn` layout-diff overrides; the structure now expresses the toggle. An on-device **shunt**
  orchestrator becomes writable the same way.

### 11.4 What it unlocks
- Batch/hybrid **transformer VC with rounding** agrees with CPU at default tolerance (kills the
  ~5e-3 kV residual).
- Batch/hybrid **shunt VC** — the fix the shunt architectural limit needs; on-device shunt VC becomes
  feasible.
- Generator remote/pilot **`DISTR_Q` relocation** rides the same mechanism (same `qRow` path).

### 11.5 Decisions to make before starting
1. **Selector source:** derive the active Q row from the row-mode *in-kernel* (no new upload channel;
   recommended) vs. an explicit per-scenario Q-relocation dest array uploaded per round (mirrors the
   `r1Ov` / `shOv` override channels — simpler kernels, extra upload + extraction bookkeeping).
2. **Single-path parity:** the single solve already re-extracts on toggle, so it does not *need* this.
   Apply anyway for one unified code path, or gate the union+dynamic-fill batch-only.
3. **Interim stopgap:** land the fork's **reject-to-CPU** option first (detect a per-scenario control
   toggle against the frozen layout, route those scenarios to the sequential CPU path, keep batching
   the rest). Correct immediately, keeps most of the batch speedup, and de-risks GA while the full
   change is built.

### 11.6 Effort & risk
- **Effort:** mostly native — the `collectPattern` union, `buildDest` dual-dest, and the fill-kernel
  selection across two families (closed-branch, shunt) × (batch + single) × (full-J + F-only) — plus,
  in regime B (base control-off, the SA case), one **small extraction change**: `computeQRow` emits
  the control-var row unconditionally and the packer threads it as a second Q-row field. Then
  `native/build-gpu.sh` and full numerical re-validation.
- **Bijection risk is lower than first thought (§11.3.1):** for local transformer/shunt VC the
  `buildRowMap` pairs are layout-invariant → **no logic change**. The real risk concentrates in
  **multi-controller `DISTR_RHO`/`DISTR_SHUNT_B`/`DISTR_Q`** (the free-controller identity moves per
  layout) and the **remote zero-Q-diagonal pivot** case — both gated by the §11.3.1 (a)–(d) bijection
  validation before any fill change.
- **Must-verify precondition:** the control-var column persists in the base structure when the base
  control is rounded off (capability-based variable creation). If any path drops it, that case cannot
  use dynamic relocation without growing `n` → CPU fallback.
- **Regression bar:** must reduce **exactly** to today's behavior when nothing toggles (plain PF,
  static transformer, fixed shunt).

### 11.7 Validation plan
- **Bijection gate first (§11.3.1 a–d):** dump `rowMap` for the continuous vs. rounded activations of
  the same structure; assert no `buildRowMap` throw in either, identical pairs on the simple case,
  conflict-free pairs on a multi-controller `DISTR_RHO`/`DISTR_Q` case, and cuDSS factorizes both.
  This must pass before the fill-selection change is meaningful.
- New **default-`convEpsPerEq`** tests: batch TVC-with-rounding and batch shunt-VC SA, each asserting
  per-bus `getV()` agreement with CPU at inner-Newton precision (≈1e-6 kV or tighter), *without* the
  tight-convEps crutch currently masking the residual.
- No regression across `GpuFullNewtonTest`, `GpuBatchedN1Test`, `GpuBatchedReactiveLimitsTest`,
  `GpuTransformerVoltageControlTest`, `GpuShuntVoltageControlTest`, `GpuBatchedAcSecurityAnalysisTest`.
- Both benches (fixed-iter K=4 + split-bench), cold-first reporting — per the standing bench
  discipline; confirm the union-pattern nnz increase does not regress warm throughput past noise.

### 11.8 Implementation status (2026-08-20)

**Closed-branch increment landed** (commit `5ef436c`, branch `hybrid-java-outer-loop`), gated behind
`OLF_GPU_DYN_QRELOC` (default off):
- `cbIdx` widened 10→12 (alternate Q rows at `[10]`,`[11]`), `computeQRowAlt`, the `collectPattern`
  union (`collectCbAltPattern`), the dual-dest + frozen-row selector (`buildCbDynDest`), and the fill
  selection (`scatterAddDyn`/`scatterAddFDyn`) across all three closed-branch fills.
- `buildRowMap` **untouched** — the bijection is layout-invariant, re-confirmed at stride 12 by
  `DynQRelocationProbeTest`.
- **No regression:** flag-off is byte-identical (29/29 batch SA + 10/10 core); flag-on all 39 pass and
  the relocation fires (`OLF_GPU_DYN_QRELOC_DEBUG` reports the relocatable-output count).

**Finding that reframes the value.** The ~6e-3 kV two-terminal transformer-VC SA residual is **not**
the wrong-row bug. For a two-terminal control the *final* rounded structure is control-off — the same
state the base extraction froze — so relocation fires only in the intermediate continuous phase and
the final result is unchanged (flag-on and flag-off both `5.994e-3` at default convEps). That residual
is convergence **depth**, which the tight-convEps setting in `transformerVoltageControlBatchMatchesCpu`
masks. Dynamic Q-relocation's real payoff is the **one-terminal shunt-VC** case (the batch
architectural limit where `SHUNT_B` freezes at warm-start), which needs the **shunt-family extension**
(`shIdx` 4→5, `fillShunt`/`fillShuntBatch` selection, a shunt-VC batch test) to demonstrate end-to-end.

**Shunt family landed + value demonstrated** (commits `ad25d28`, `9e6ee55`). `shIdx` widened 4→5 with
the alternate Q row at `[4]`; `collectShAltPattern` / `buildShDynDest` and the `scatterAddDyn` wiring
mirror the closed-branch path across `fillShunt`/`fillShuntBatch`/`fillShuntBatchF`. Flag-off stays
byte-identical (40/40); flag-on 39/39.

`GpuShuntVoltageControlTest#dynQRelocUnlocksSharedStructureShuntControl` proves the one-terminal fix at
the single-solve level (no SA/JAVA machinery): extract the structure control-OFF, solve it with the
control-ON activation (what the batch does per round on the fixed structure).

| | `‖F‖∞` | `max\|x_gpu − x_olf\|` |
|---|---|---|
| flag OFF | `3.45e-14` (converges) | **`2.995e+00`** — converged-but-WRONG: `SHUNT_B` frozen at warm-start |
| flag ON | `3.45e-14` | **`2.398e-14`** — machine-precision match to OLF |

This is the case dynamic Q-relocation exists for: a two-terminal transformer keeps its control column
constrained by the other terminal (so its *final* result is unaffected — §11.8 first paragraph); a
one-terminal shunt does not, so a frozen structure freezes `SHUNT_B` and converges on a wrong system.
Dynamic relocation routes Q to the free control-variable row and recovers the correct solve.

**Next (increment 3):** exercise this in a real batch outer loop — re-enable shunt VC in the JAVA
hybrid (`GpuSaRunScope.JAVA_SUPPORTED_LOOPS`, reverted earlier only because of this Q-relocation
blocker) or add a device shunt orchestrator — and a batch shunt-VC SA test asserting CPU parity.
