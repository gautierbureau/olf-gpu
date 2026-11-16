# Outer loops on the device: the current workflow

How an OLF outer loop runs in the GPU engine today, step by step, with the code that does it.
`ReactiveLimitsOuterLoop` is the running example because it is the loop with the most history.
The companion document, [`outer-loop-array-form-proposal.md`](outer-loop-array-form-proposal.md),
proposes how to stop maintaining the logic twice.

File references are to this repository unless they start with `OLF:`, which means
`powsybl-open-loadflow` (`src/main/java/com/powsybl/openloadflow/...`). Line counts were taken on
2026-11-10 at `031d7b3` (this repo) and `144a54c` (OLF).

## 1. What OLF does (the reference)

Three pieces of OLF matter, and they live in three different places.

**The sequencer** — `OLF: ac/AcloadFlowEngine.java`. Loops are nested: each loop is driven to
STABLE, re-solving after every UNSTABLE check, then the next loop runs, and the whole list restarts
until a sweep re-solves nothing.

```java
// AcloadFlowEngine.runOuterLoop
do {
    outerLoopResult = outerLoop.check(outerLoopContext, olReportNode);
    if (outerLoopResult.status() == OuterLoopStatus.UNSTABLE) {
        runningContext.lastUnstableOuterLoop = outerLoop;
        runningContext.lastSolverResult = runAcSolverAndCheckRealisticState(solver, new PreviousValueVoltageInitializer(), ...);
        runningContext.outerLoopTotalIterations++;
    }
} while (outerLoopResult.status() == OuterLoopStatus.UNSTABLE
        && runningContext.lastSolverResult.getStatus() == AcSolverStatus.CONVERGED
        && runningContext.outerLoopTotalIterations < context.getParameters().getMaxOuterLoopIterations());
```

**The decision** — one class per loop under `OLF: ac/outerloop/`. A check walks the `LfNetwork`
object graph, compares a few numbers and calls setters on the objects it wants to change:

```java
// ReactiveLimitsOuterLoop.checkControllerBus
double minQ = controllerBus.getMinQ();
double maxQ = controllerBus.getMaxQ();
double q = controllerBus.getQ().eval() + controllerBus.getLoadTargetQ();
if (q < minQ - maxReactivePowerMismatch) {
    buses.add(new ControllerBusToPqBus(controllerBus, q, minQ, LfBus.QLimitType.MIN_Q));
    remainsPV = false;
} else if (q > maxQ + maxReactivePowerMismatch) {
    buses.add(new ControllerBusToPqBus(controllerBus, q, maxQ, LfBus.QLimitType.MAX_Q));
    remainsPV = false;
}
...
// ReactiveLimitsOuterLoop.switchPvPq
controllerBus.freezeGenerationTargetQAndDisableGeneratorVoltageControl(pvToPqBus.qLimit);
controllerBus.setQLimitType(pvToPqBus.limitType);
contextData.incrementPvPqSwitchCount(controllerBus.getId());
```

**The effect on the equation system** — not in the loop at all. The setter fires a network
listener, and `OLF: ac/equations/AcEquationSystemUpdater.java` re-derives which equations are
active for the whole voltage control from the objects' flags:

```java
// AcEquationSystemUpdater
public void onGeneratorVoltageControlChange(LfBus controllerBus, boolean newVoltageControllerEnabled) {
    updateVoltageControls(controllerBus.getGeneratorVoltageControl().orElseThrow().getControlledBus());
}

// AcEquationSystemCreator.updateRemoteVoltageControlEquations (called from there)
// activate voltage control at controlled bus only if at least one controller element is enabled
vEq.setActive(!enabledControllerElements.isEmpty());
for (T controllerElement : disabledControllerElements) {
    equationSystem.getEquation(controllerElement.getNum(), distrEqType).ifPresent(eq -> eq.setActive(false));
    equationSystem.getEquation(controllerElement.getNum(), ctrlEqType).orElseThrow().setActive(true);
}
// activate distribution equation and deactivate control equation at all enabled controller buses except one (first)
for (int i = 0; i < enabledControllerElements.size(); i++) {
    boolean active = i != 0;
    ...
}
```

Two properties of this design drive everything below:

- the loop reads its inputs through getters that hide a lot (`getMinQ()` evaluates capability
  curves at the current target P; `getQ().eval()` sums equation terms);
- the activation is **re-derived from scratch** from flags, it is not edited incrementally. A
  contingency is applied to the `LfNetwork` before the loop runs, so the loop code never knows
  about contingencies.

## 2. The device pipeline in one picture

```
            Java (host)                                   native (libolfgpu)
 ┌──────────────────────────────┐
 │ 1 PACK    LfNetwork + es ──► │ tables (int[]/double[] per entry)
 │   ReactiveLimitsPacker, ...  │
 ├──────────────────────────────┤        ┌──────────────────────────────────────────────┐
 │ 2 INSTALL *Native.setBatch*  ├───JNI──► device buffers on BatchSolverContext          │
 ├──────────────────────────────┤        ├──────────────────────────────────────────────┤
 │ 3 SOLVE   solveBatchContext  ├───JNI──► outer_loop_driver.cu                          │
 │   (ONE call for the chunk)   │        │   loop order ── OLF's, from installed tables  │
 │                              │        │   newton_iteration.cu                         │
 │                              │        │     fill → factor → solve → step → test       │
 │                              │        │     converged scenarios ─► outer_loop_sequence│
 │                              │        │        checkOuterLoop (outer_loop_check.cu)   │
 │                              │        │          observe  qBuf / flows / sensitivities│
 │                              │        │          decide + effect  outerloop/*.cu      │
 │                              │        │          unstable[s] ─► host C++ (nS bytes)   │
 │                              │        │        UNSTABLE ─► restart that scenario      │
 ├──────────────────────────────┤        ├──────────────────────────────────────────────┤
 │ 4 READ BACK states, final    ◄───JNI──┤ states of the ended scenarios, loop state     │
 │   loop state                 │        └──────────────────────────────────────────────┘
 │ 5 WRITE BACK to LfNetwork    │
 │   Gpu*OuterLoop.applyToNetwork
 └──────────────────────────────┘
```

Nothing crosses to Java between step 3's entry and its return. The only per-check traffic is
host C++ ↔ device inside the native call (section 6).

## 3. Step by step, on ReactiveLimits

### 3.1 Pack: object graph → tables

`ac/equations/vector/gpu/ReactiveLimitsPacker.java` walks the network once per extraction and
flattens everything the kernel will need into one entry per controller bus. Each list is something
OLF reads through a getter at check time:

```java
static ReactiveLimitEntries reactiveLimitsPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                 ExtractionColumns columns) {
    List<Integer> reactiveLimitVoltageRow = new ArrayList<>();       // the bus's BUS_V row
    List<Double> reactiveLimitMinQ = new ArrayList<>();              // bus.getMinQ()
    List<Double> reactiveLimitMaxQ = new ArrayList<>();              // bus.getMaxQ()
    List<Double> reactiveLimitLoadQ = new ArrayList<>();             // bus.getLoadTargetQ()
    List<Double> reactiveLimitTargetV = new ArrayList<>();           // the voltage control's target
    List<Byte> reactiveLimitVoltageControlRole = new ArrayList<>();  // LOCAL / REMOTE / PILOT_SELF
    List<Integer> reactiveLimitComparedVoltageRow = new ArrayList<>(); // the CONTROLLED bus's V row
    // forced-PV election keys (OLF switchPvPq: controlled bus nominal V, live target P, id)
    List<Double> reactiveLimitNominalV = new ArrayList<>();
    ...
```

The result is the record `ReactiveLimitEntries` (18 arrays). Things a getter hides become extra
tables with their own packers: the capability curves (`ReactiveCapabilityCurves.java`, because
`getMinQ()` depends on the target P the distributed slack moves), the SVC susceptance limits
(`addSvcQuadratic`, because an SVC's limit is `B·v²` at the live voltage), the remote members'
reactive terms (`RemoteMemberQPacker.java`), the `DISTR_Q` groups (`DistrQPacker.java`) and the
pilot-self groups (`PilotSelfPacker.java`).

### 3.2 Install: tables → device buffers

One JNI entry point per channel, declared in `ac/outerloop/gpu/ReactiveLimitsNative.java` (16
`native` methods for this loop) and implemented in `native/cuda/jni/reactive_limits_jni.cu`. They
are all the same shape — check lengths, upload, set a flag:

```cpp
// native/cuda/jni/reactive_limits_jni.cu
OLF_JNI(ac_outerloop_gpu, ReactiveLimitsNative, setBatchReactiveLimitStrongestPvKeys)(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jNomV, jintArray jPRow, jdoubleArray jStaticP, jintArray jIdRank) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    ...
    ctx->dReactiveLimitNominalV.upload(nv, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(double));
    ctx->dReactiveLimitActivePowerRow.upload(pr, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(int));
    ...
    ctx->hasReactiveLimitStrongestPvKeys = 1;
}
```

Several channels are **per scenario**, because a contingency changes what OLF's loop would see
and the device has no per-scenario `LfNetwork` to apply it to:
`setBatchReactiveLimitIslanded` (a controller the contingency strands in another component),
`setBatchReactiveLimitGeneratorsLost`, `setBatchReactiveLimitLocalOverrides` and
`setBatchReactiveLimitComparedVoltageRowOverrides` (voltage controls OLF re-merges after a
zero-impedance split). Each one is OLF's "apply the contingency, then run the same loop"
re-expressed as a table.

`DevicePhases.java` builds the controller tables once for both routes (base load flow and
security analysis), so a loop cannot be wired into one and not the other.

### 3.3 Loop order

`native/cuda/solver/outer_loop_driver.cu` builds the list in OLF's `DefaultAcOuterLoopConfig`
order from whichever tables were installed:

```cpp
if (distributionInLoopOrder && !ctx->areaInterchangeControlOuterLoop) {
    loopOrder[nLoops++] = GpuOuterLoopKind::DISTRIBUTED_SLACK;
}
if (ctx->nHvdcAcEmulation > 0) {
    loopOrder[nLoops++] = GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION;  // OLF: FreezingHvdcACEmulation
    ...
}
...
if (ctx->reactiveLimitsOuterLoop && ctx->nReactiveLimitEntries > 0) {
    loopOrder[nLoops++] = GpuOuterLoopKind::REACTIVE_LIMITS;  // OLF order 6: ReactiveLimits
}
```

The 14 kinds are `ac/solver/gpu/codes/GpuOuterLoopKind.java`, mirrored by
`native/cuda/common/codes/gpu_outer_loop_kind.cuh`.

### 3.4 Sequencer: OLF's `runOuterLoops`, per scenario

`native/cuda/solver/outer_loop_sequence.cu` is the port of section 1's sequencer. The difference
is that every scenario stands at **its own** loop, with its own outer-iteration budget, and that
the checks happen inside the batched Newton loop rather than between whole-chunk solves:

```cpp
// native/cuda/solver/newton_iteration.cu — inside the batched Newton iteration
// ---- the outer loops of the scenarios whose solve just converged, in place ----
if (std::find(solveConverged.begin(), solveConverged.end(), 1) != solveConverged.end()) {
    checkSolvedScenarios(ctx, drv, sequence, solveConverged, restarted);
    for (int s = 0; s < nS; s++) {
        if (restarted[s]) {
            it[s] = 0;              // this scenario's Newton restarts; its chunk mates go on with theirs
        }
    }
}
```

```cpp
// outer_loop_sequence.cu — checkSolvedScenarios, after checkOuterLoop filled outerLoopUnstable[]
if (!outerLoopUnstable[s]) {
    continue;                                   // STABLE: on to the next loop, against the same state
}
seq.checkPending[s] = 0;                        // UNSTABLE: re-solve, then this loop checks again
if (++seq.productiveIterations[s] >= seq.maxOuterLoopIterations) {
    res.status[s] = GpuAcSolverStatus::REJECTED; // budget exhausted: the CPU replays it
    live[s] = 0;
    continue;
}
active[s] = 1;
restarted[s] = 1;
seq.lastUnstableLoopIdx[s] = i;                 // OLF's lastUnstableOuterLoop
```

A STABLE check never re-solves: the device Newton forces at least one iteration per solve, so a
no-op solve would move a converged state off the CPU's (measured 4e-4 kV).

### 3.5 Check: observe

`native/cuda/solver/outer_loop_check.cu` (`checkOuterLoop`) first uploads which scenarios are
being checked, then builds the bus balances the loops read. OLF's `controllerBus.getQ().eval()`
becomes an accumulator, `dQbuf`, filled by the same generated branch kernels that fill the
Jacobian, with the Jacobian output switched off:

```cpp
CUDA_CHECK(cudaMemcpy(ctx->dOuterLoopCheckActive.p, outerLoopCheckActive.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
// closed branches: already accumulated by the scenario's last Newton fill (dQbufClosed)
copyToScenarioMajor(n, ctx->layout, ctx->dQbufClosed.as<double>(), ctx->dQbuf.as<double>(), ...);
if (ctx->structure->openBranch.elementCount) fillOpenBranchBatch<<<...>>>(..., /*vals*/ nullptr, ctx->dQbuf.as<double>(), ...);
if (ctx->structure->shunt.elementCount)      fillShuntBatch<<<...>>>(..., nullptr, ctx->dQbuf.as<double>(), ...);
...
```

Loops that need a sensitivity (incremental transformer / shunt / flow control, secondary voltage
control) do a unit-RHS solve on the factors the driver already holds (`outerloop/sensitivity.cu`)
and evaluate flows with the generated branch functions (`outerloop/flow_observation.cuh`).

Then the dispatch, one branch per loop kind:

```cpp
int loopKind = loopOrder[loopIdx];
if (loopKind == GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL) {
    incrementalTransformerVoltageControlStep(ctx, drv);
} else if (...) {
    ...
} else if (inDistributedSlackPhase) {
    distributedSlackCheck(ctx, drv);
} else {
    reactiveLimitsCheck(ctx, drv, loopKind);
}
CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
CUDA_CHECK(cudaMemcpy(outerLoopUnstable.data(), ctx->dOuterLoopUnstable.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
```

### 3.6 Check: decide and apply, in one kernel

`native/cuda/outerloop/reactive_limits.cu`, `reactiveLimitsBatch`: one thread per
(scenario, entry). The **decision** is OLF's comparison. The **effect** is where the two engines
part: OLF calls a setter and a listener re-derives the activation; the device has a fixed matrix
with permanent rows, so the kernel writes the row's mode and target directly
(`GpuRowMode`: `FIXED_VARIABLE` = "x = target", `EQUATION` = "the row's balance"):

```cpp
// currently PV — check the reactive limits
double q = qBuf[qIdx] + reactiveLimitLoadQ[li];              // OLF: controllerBus.getQ().eval() + getLoadTargetQ()
if (q < liMin - tol) {                                       // OLF: q < minQ - maxReactivePowerMismatch
    savedTarget[scenarioEntry] = target[targetQIdx];         // remembered for the switch back
    savedMode[scenarioEntry] = static_cast<char>(mode[targetQIdx]);
    mode[targetQIdx] = localOv ? GpuRowMode::BUS_TARGET_Q : GpuRowMode::EQUATION;   // BUS_TARGET_V off, BUS_TARGET_Q on
    target[targetQIdx] = liMin - reactiveLimitLoadQ[li];     // OLF: freezeGenerationTargetQ...(minQ)
    limitType[scenarioEntry] = GpuQLimitType::MIN_Q;         // OLF: setQLimitType(MIN_Q)
    switchCount[scenarioEntry]++;                            // OLF: incrementPvPqSwitchCount
    switched[s] = 1;                                         // OLF: status = UNSTABLE
    passSwitched[scenarioEntry] = 1;
}
```

The switch back, with the cap and the limit refresh of `checkPqBusWithQLimitType`:

```cpp
bool wantsBack = !slopeRow
        && ((atMaxLimit && v > targetV)
         || (atMinLimit && v < targetV));
bool back = wantsBack && switchCount[scenarioEntry] < maxSwitch;
if (back) {
    mode[targetQIdx] = fixedVariableMode(crossRow);          // PQ → PV: resume voltage control
    target[targetQIdx] = targetV;
    limitType[scenarioEntry] = GpuQLimitType::NONE;
    switched[s] = 1;
} else if (!wantsBack && (perScenarioReactiveLimits || svcLim) && ...) {
    double lim = limitType[scenarioEntry] == GpuQLimitType::MIN_Q ? liMin : liMax;
    if (fabs(lim - (target[targetQIdx] + reactiveLimitLoadQ[li])) > tol) {
        target[targetQIdx] = lim - reactiveLimitLoadQ[li];   // re-freeze at the moved limit
        switched[s] = 1;
    }
}
```

`reactiveLimitsBatch` takes 52 parameters. Most of them are not decision inputs: they say *where*
this scenario's q is read and *which row* the switch writes (relocated rows, re-merged controls,
pilot-self groups, per-scenario limits).

### 3.7 Check: the passes around the kernel

`reactiveLimitsCheck` (same file) launches up to nine kernels per check, in an order that matters:

| order | kernel | OLF counterpart |
|---|---|---|
| 1 | `reactiveCapabilityCurveLimitsBatch`, `launchForcedTargetQLimits` | `bus.getMinQ()/getMaxQ()` at the current target P |
| 2 | `reactiveLimitsBatch` | `checkControllerBus`, `checkPqBus`, the switches |
| 3 | `pilotSelfMixedPvPqGate` | none: a layout the device has no row for → CPU |
| 4 | `reactiveLimitsRobustVoltageResetBatch` | `controllerBus.setV(1)` in robust mode |
| 5 | `reactiveLimitsForcedPvBatch` | `switchPvPq`: "keep the strongest bus PV" |
| 6 | `launchReMergedVoltageTargetRowUpdate` | `updateRemoteVoltageControlEquations` (first enabled controller) |
| 7 | `pilotRowWithoutEnabledControllerBatch` | `vEq.setActive(!enabledControllerElements.isEmpty())` |
| 8 | `launchPilotTargetVEquationUpdate` | same, for relocation groups |
| 9 | `launchDistrQTargetsInner` | `DISTR_Q` activation and reactive keys |

Rows 6 to 9 are the device's version of the ~60-line `updateRemoteVoltageControlEquations`. They
live in `remote_voltage_control.cu` (419 lines) and `pilot_voltage.cu` (462 lines). OLF re-derives
the group's layout from flags; these kernels **edit** it incrementally (flip a pilot row, hand a
target to a new carrier, restore a saved mode), which is why their relative order is load-bearing
and why most reactive-limit defects recorded in `docs/tasks.md` are in them rather than in the
comparisons of section 3.6.

The forced-PV election shows how an order-dependent, whole-network rule is ported: one thread per
scenario, a sequential scan over the entries, then an *undo* of the elected bus's switch:

```cpp
// reactiveLimitsForcedPvBatch — one thread per scenario
if (!outerLoopCheckActive[s] || remainingPvBusCount[s] > 0) return;
for (int i = 0; i < nReactiveLimitEntries; i++) {
    if (passSwitched[(size_t) s * nReactiveLimitEntries + i] != 1) continue;   // not switched PV -> PQ in this check
    double v = nomV[i];
    double p = pRow[i] >= 0 ? target[(size_t) s * n + pRow[i]] : staticP[i];
    bool better = best < 0 || v > bestV || (v == bestV && (p > bestP || (p == bestP && idRank[i] < bestId)));
    ...
}
mode[idx] = savedMode[scenarioEntry];            // give the elected bus its PV row back
target[idx] = savedTarget[scenarioEntry];
```

### 3.8 What a kernel cannot express goes to the CPU, whole

A device loop never shares a scenario with OLF. Three exits, all ending in OLF's own sequential
path for that contingency:

- **refused before the solve** (qualification): a shape the tables cannot hold;
- **rejected by a check**: `loopRejected[s]`, e.g. `pilotSelfMixedPvPqGate`, a singular secondary
  voltage control system, an unrealistic intermediate state;
- **budget**: a scenario that exhausts its `maxOuterLoopIterations`.

### 3.9 Read back and write back

After the native call returns, the final loop state is read per (scenario, entry)
(`ReactiveLimitsNative.readBatchReactiveLimitsFinalState`, used by `GpuNewtonRaphson` and
`GpuBatchChunkSolver`) and replayed onto the `LfNetwork` with the same setters OLF's loop calls,
so that OLF's own equation system and result code see the state OLF's loop would have left:

```java
// ac/outerloop/gpu/GpuReactiveLimitsOuterLoop.applyToNetwork
if (limitType != GpuQLimitType.NONE && bus.isGeneratorVoltageControlEnabled()) {
    int limitRow = frozenTargetQRow(data, i, vRow, devModes);     // where the device put the frozen Q
    double qLimit = deviceTargets[limitRow] + data.reactiveLimits().loadQ()[i];
    bus.freezeGenerationTargetQAndDisableGeneratorVoltageControl(qLimit);
    bus.setQLimitType(GpuQLimitType.toOlf(limitType));
} else if (limitType == GpuQLimitType.NONE && !bus.isGeneratorVoltageControlEnabled() && ...) {
    bus.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(true);
    bus.setQLimitType(null);
}
```

`frozenTargetQRow` has to repeat, in Java, the row choice the kernel made (relocated row, carrier
row of a pilot-self group): a third copy of the same placement rule.

## 4. Every loop, where it lives

| OLF order | OLF loop | device kernel file (`native/cuda/outerloop/`) | Java side (`ac/outerloop/gpu/`) | observes | writes |
|---|---|---|---|---|---|
| 0 | DistributedSlack | `distributed_slack.cu` | `GpuDistributedSlackOuterLoop` | slack balance | P targets |
| 1 | FreezingHvdcACEmulation | `hvdc_ac_emulation.cu` | `GpuFreezingHvdcACEmulationOuterloop` | frozen flag | HVDC status |
| 2 | AcHvdcAcEmulationLimits | `hvdc_ac_emulation.cu` | `GpuHvdcAcEmulationLimitsOuterLoop` | angles | HVDC status |
| 3 | AreaInterchangeControl | `area_interchange_control.cu` | `GpuAreaInterchangeControlOuterLoop` | boundary flows | P targets |
| 4 | SecondaryVoltageControl | `secondary_voltage_control.cu` | `GpuSecondaryVoltageControlOuterLoop` | q, pilot v, sensitivities | V targets |
| 5 | MonitoringVoltage | `voltage_monitoring.cu` | `GpuMonitoringVoltageOuterLoop` | v | row mode, target |
| 6 | ReactiveLimits | `reactive_limits.cu` + 5 files | `GpuReactiveLimitsOuterLoop` | q, v | row mode, target |
| 7 | PhaseControl / Incremental | `phase_control.cu`, `incremental_flow_control.cu` | `GpuPhaseControlOuterLoop`, `GpuIncrementalPhaseControlOuterLoop` | flows, sensitivities | tap overrides |
| 8 | TransformerVoltageControl (3 flavours) | `transformer_voltage_control.cu`, `incremental_transformer_voltage_control.cu` | `GpuTransformerVoltageControlOuterLoop`, ... | v, sensitivities | tap overrides, row mode |
| 9 | TransformerReactivePowerControl | `incremental_flow_control.cu` | `GpuIncrementalTransformerReactivePowerControlOuterLoop` | q flow, sensitivities | tap overrides |
| 10 | ShuntVoltageControl (2 modes) | `shunt_voltage_control.cu`, `incremental_shunt_voltage_control.cu` | `GpuShuntVoltageControlOuterLoop`, ... | v, sensitivities | shunt overrides |
| 11 | AutomationSystem | `automation_system.cu` | `GpuAutomationSystemOuterLoop` | currents | element disable |

## 5. What crosses between host and device

| when | direction | what | size |
|---|---|---|---|
| per extraction | host → device | structure, loop tables | once |
| per chunk | host → device | per-scenario overrides, targets, modes | per scenario |
| per check | host C++ → device | `outerLoopCheckActive` | nS bytes |
| per check | device → host C++ | `outerLoopUnstable` (+ a reject flag for some loops) | nS bytes |
| per Newton iteration | device → host C++ | norms for the convergence test | 4·nS doubles |
| per call | device → Java | states of the ended scenarios, final loop state | n per scenario |

No state vector, no q, no row mode goes to the host during the outer loops.

## 6. Why the device side is three times OLF's size

Lines of code, comments and blanks excluded:

| | OLF | device | of which |
|---|---|---|---|
| all outer loops | 3,286 (`ac/outerloop` + `lf/outerloop`) | 10,621 | CUDA 5,060 · JNI 1,705 · Java 3,856 |
| ReactiveLimits alone | 331 | 2,965 | CUDA 1,487 · JNI 472 · Java 1,006 |

The sequencer is extra on both sides (`AcloadFlowEngine` 278; `solver/outer_loop_*` 689).

The comparison is not like for like, and the gap has four separate causes. Only the last two are
architecture we chose.

1. **OLF's loops are small because the model does the work.** The 331 lines of
   `ReactiveLimitsOuterLoop` call `getMinQ()`, `getQ().eval()` and a setter. Behind those sit the
   generators' capability curves, the equation terms and `AcEquationSystemCreator` (1,311 lines,
   of which `updateRemoteVoltageControlEquations` is what a PV/PQ switch triggers). None of that is
   counted in OLF's column; its device equivalent — curves, remote voltage control, pilot
   voltage: 883 lines of CUDA — is counted in ours.
2. **One network per contingency versus one matrix for all.** OLF applies the contingency to the
   `LfNetwork` and runs unchanged loop code. The device keeps one fixed structure, so each thing a
   contingency can change for a loop is a per-scenario table plus kernel branches (islanded,
   generators lost, re-merged controls, relocated rows). This is the price of batching and it is
   inherent to the fixed-matrix design ([`gpu-vs-olf.md`](gpu-vs-olf.md), section 1).
3. **Every loop is written four times over.** Packer (Java) → JNI install (C++) → kernel (CUDA) →
   write-back (Java), with the entry layout repeated by hand in each: `ReactiveLimitEntries` has
   18 arrays, `ReactiveLimitsNative` 16 entry points, `reactiveLimitsBatch` 52 parameters. The JNI
   layer alone is 1,705 lines of near-identical upload code. This part is mechanical and could be
   generated from one description of the tables.
4. **The effect is edited, not derived.** Section 3.7: OLF recomputes a voltage control's
   activation from its flags in one function; the device applies transitions (save, flip, hand
   over, restore) across several kernels whose order matters. Transition code is longer than
   derivation code and is where the defects were. Inside `reactiveLimitsBatch` the same decision
   is also written twice, once for local and once for remote entries, because the two differ in
   *where* they read and write, not in *what* they decide.

About a quarter of the device lines are comments (1,230 of 6,658 in `outerloop/`), mostly
measurements that justify a branch; they are excluded from the table above.

So the honest reading is: causes 1 and 2 would remain under any design; causes 3 and 4 are what
the companion proposal targets.
