# Improving the outer loops' device design, within olf-gpu

**Status:** proposal, nothing implemented (2026-11-10).
**Scope:** this repository only. OLF stays as it is, the device keeps its hand-written port of
each loop, and nothing here depends on
[`outer-loop-array-form-proposal.md`](outer-loop-array-form-proposal.md) (which would change OLF).
**Read first:** [`outer-loops-on-device.md`](outer-loops-on-device.md) for the current workflow.

Every item below is a refactoring: the same results, less code to keep in step. Each can land
alone, and none changes what a kernel computes.

## 1. What is wrong, in numbers

Counted on 2026-11-10 at `031d7b3`.

| symptom | measure |
|---|---|
| kernel signatures | 54 `__global__` kernels in `native/cuda/outerloop/`; median 15 parameters; 16 kernels with 20 or more; `distributeSlackBatch` 54, `reactiveLimitsBatch` 52, `updateDistrQTargetsBatch` 40 |
| JNI installs | 61 entry points across the 11 per-loop JNI files (1,975 lines); 84 array-length checks and throws written by hand |
| the context | `solver/batch_context.cuh` is 622 lines; 41 of them are reactive-limit members of one flat struct |
| loop identity | the loop kind is switched on in four files (`outer_loop_driver.cu`, `outer_loop_check.cu`, `loop_kinds.cuh`, `driver_traces.cu`) |
| placement rule | "which row holds this entry's reactive balance this scenario" is written in the check kernel, in `exportReactiveLimitTargetQ`, and again in Java (`GpuReactiveLimitsOuterLoop.frozenTargetQRow`) |
| decision rule | OLF's PV→PQ and PQ→PV comparisons are inline twice in `reactiveLimitsBatch` (local entries, remote entries) |
| effect | a voltage control's layout is maintained by transition kernels that save, flip and restore, guarded by flags, in an order that matters |

How it got this way is not mysterious. Each channel was added to fix one measured defect, and the
cheapest safe way to add one was a new array, a new setter and a new kernel parameter, with a null
pointer meaning "channel absent" so that runs without it stayed byte-identical. Each step was
reasonable; the structure is what accumulated.

## 2. The improvements

### A. Pass a view, not fifty pointers

The pattern already exists: `ReactiveRelocationTables` (`fixedmatrix/reactive_relocation.cuh`) and
`SlotEntries` (`equations/slot_entries.cuh`) are small structs of device pointers passed to
kernels by value. The outer-loop kernels predate them.

Today, the launch site resolves every optional channel by hand:

```cpp
// outerloop/reactive_limits.cu — reactiveLimitsCheck, 40 lines of this
reactiveLimitsBatch<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(
        ctx->nSecondaryVoltageControlControllerBuses > 0 ? ctx->dSecondaryVoltageControlReactiveLimitTargetV.as<double>() : nullptr,
        ctx->nReactiveLimitEntries, nS, n, REACTIVE_LIMIT_TOL, MAX_PQ_PV_SWITCH,
        ctx->dReactiveLimitVoltageRow.as<int>(),
        ctx->dReactiveLimitReactivePowerRow.p ? ctx->dReactiveLimitReactivePowerRow.as<int>() : nullptr,
        usePerScenarioReactiveLimits ? ctx->dReactiveLimitMinQPerScenario.as<double>() : ctx->dReactiveLimitMinQ.as<double>(),
        ...
```

Proposed: three views, built once where the buffers are owned.

```cpp
// outerloop/reactive_limits.cuh
struct ReactiveLimitEntriesView {          // static, per extraction
    int count = 0;
    const int* voltageRow = nullptr;
    const int* reactivePowerRow = nullptr; // null: every entry on its own row
    const double* loadQ = nullptr;
    const char* voltageControlRole = nullptr;
    const int* comparedVoltageRow = nullptr;
    ...
};
struct ReactiveLimitScenarioView {         // per scenario: limits, overrides, islanded...
    const double* minQ = nullptr;          // already resolved: per scenario or shared
    const double* maxQ = nullptr;
    int limitStride = 0;                   // 0: shared by every scenario
    const char* islanded = nullptr;
    SlotEntries targetVOverrides;
    ...
};
struct ReactiveLimitState {                // what the loop mutates
    char* limitType = nullptr;
    char* switchCount = nullptr;
    char* savedMode = nullptr;
    double* savedTarget = nullptr;
    ...
};

__global__ void reactiveLimitsBatch(ReactiveLimitEntriesView entries, ReactiveLimitScenarioView scenario,
                                    ReactiveLimitState state, SolveView solve /* x, qBuf, mode, target, checked */,
                                    ReactiveLimitSettings settings /* tol, maxSwitch, robust band */);
```

The launch becomes one line, and the kernels that read the same tables
(`reactiveLimitsForcedPvBatch`, `pilotSelfMixedPvPqGate`, `exportReactiveLimitTargetQ`,
`seedReactiveLimitRowModeFromState`) take the same views instead of each re-listing them.

- **Risk:** none on results; kernel bodies are unchanged apart from `x` → `solve.x`.
- **Check:** the GPU suite, plus the two benches (a struct by value must not cost anything
  measurable).
- **Do first.** It makes every later item smaller.

### B. One packed table per loop, one generic install

The equation families already travel this way: a `parameters` block and an `indices` block with a
fixed stride per element, and the stride is the only thing Java and CUDA have to agree on. The
loop tables do not: `ReactiveLimitEntries` is 18 parallel arrays, each with its own setter
argument, device buffer and length check.

Today, per table, per field:

```cpp
// jni/reactive_limits_jni.cu — repeated for every array of every setter
if (env->GetArrayLength(jNomV) != ctx->nReactiveLimitEntries || env->GetArrayLength(jPRow) != ctx->nReactiveLimitEntries
        || ...) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchReactiveLimitStrongestPvKeys: array length != nReactiveLimitEntries");
    return;
}
jdouble* nv = env->GetDoubleArrayElements(jNomV, nullptr);
...
ctx->dReactiveLimitNominalV.upload(nv, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(double));
```

Proposed: an entry is `intWidth` ints and `doubleWidth` doubles, with named offsets mirrored the
way `ac/solver/gpu/codes/*` already is.

```java
// ac/solver/gpu/codes/GpuReactiveLimitEntryLayout.java  (mirrored by gpu_reactive_limit_entry_layout.cuh)
public static final int INT_VOLTAGE_ROW = 0;
public static final int INT_COMPARED_VOLTAGE_ROW = 1;
public static final int INT_FLAGS = 2;            // role, NOT_CONTROLLER_ELEMENT, has-slope...
public static final int INT_WIDTH = 6;
public static final int DOUBLE_MIN_Q = 0;
public static final int DOUBLE_MAX_Q = 1;
public static final int DOUBLE_WIDTH = 12;
```

```java
// one native method for every loop's static table
static native void installLoopTable(long handle, int loopKind, int count, int[] ints, double[] doubles);
```

```cpp
// jni: one implementation, one length check
installLoopTable(ctx, kind, count, ints /* count * intWidth(kind) */, doubles /* count * doubleWidth(kind) */);
```

Per-scenario channels already have the generic form (`SlotTable`, `jni/slot_table_install.cuh`);
the remaining hand-written per-scenario setters move onto it.

What "channel absent" becomes needs care. Today it is a null pointer and a branch. In a packed
entry it is either a flag bit in `INT_FLAGS` or a neutral value, and a neutral value must be shown
inert: the kernel already records one that was not (`0 × NaN` from an unsolved base poisoned every
entry's limits). Prefer flag bits.

- **Risk:** low on results, but real: field order mistakes are silent. The check is that the
  per-loop decision records (`OLF_GPU_RL_REASONS=1`) are identical before and after on rte6515 and
  network.xiidm.
- **Expected:** the JNI layer for a loop shrinks to the entry points that are not uploads
  (read-backs, settings). I have not measured the result; my estimate for ReactiveLimits is the
  525-line file dropping to roughly a quarter.
- A reading aid comes free: the layout file is the list of everything the loop reads.

### C. Give each loop its own state object

`BatchSolverContext` holds every loop's buffers side by side, so nothing says which buffers
belong to which loop, which are static, which are per-scenario and which must be reset per solve.
That is the cause of a recorded class of defect: `dSvcRlVt` was read by an existing kernel and
initialised after the seeding that consulted it (`docs/tasks.md`, SecondaryVoltageControl), and
the 2026-10-06 flake was a device read-before-write (#78).

Proposed: one struct per loop, owning its buffers and exposing the views of item A, with the
lifecycle explicit.

```cpp
struct ReactiveLimitsLoop {
    // owned buffers (static table, per-scenario channels, state)
    ...
    void install(const LoopTable& table);            // per extraction
    void resetForSolve(BatchSolverContext& ctx);     // every buffer a kernel may read gets a defined value here
    ReactiveLimitEntriesView entries() const;
    ReactiveLimitScenarioView scenario(const DriverPass& drv) const;
    ReactiveLimitState state();
    bool enabled() const;                            // today: ctx->reactiveLimitsOuterLoop && ctx->nReactiveLimitEntries > 0
    void check(BatchSolverContext& ctx, DriverPass& drv);
};
```

`resetForSolve` is the point: a buffer added to a loop is initialised in one obvious place, not
in whichever of upload / per-solve reset the author remembered.

- **Risk:** none on results; this is moving members.
- **Check:** the GPU suite; the #78 test in a loop.

### D. One registry of loops

The loop kind is decoded in four places. Adding a loop means editing the order in
`outer_loop_driver.cu`, the `if / else if` chain in `outer_loop_check.cu`, the name table and the
trace table, and they can disagree.

```cpp
// today — outer_loop_check.cu
if (loopKind == GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL) {
    incrementalTransformerVoltageControlStep(ctx, drv);
} else if (loopKind == GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL) {
    ...
} else {                                               // reactive-limits outer loop
    reactiveLimitsCheck(ctx, drv, loopKind);
}
```

Proposed: one table, in OLF's order, that the driver, the dispatch, the names and the timing all
read.

```cpp
struct DeviceLoop {
    int kind;
    const char* olfName;                               // "ReactiveLimits"
    bool (*enabled)(const BatchSolverContext&);
    void (*check)(BatchSolverContext&, DriverPass&);
};
// OLF's DefaultAcOuterLoopConfig order
static const DeviceLoop DEVICE_LOOPS[] = {
    {GpuOuterLoopKind::DISTRIBUTED_SLACK, "DistributedSlack", distributedSlackEnabled, distributedSlackCheck},
    {GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION, "FreezingHvdcACEmulation", hvdcEnabled, hvdcFreezeCheck},
    ...
};
```

The two order rules that are not a plain list stay as they are, as predicates: DistributedSlack
is dropped when AreaInterchangeControl is configured, and the incremental and discrete flavours
of a control exclude each other.

- **Risk:** none on results.
- **Check:** `OLF_GPU_LOOP_TRACE` output identical on a traced contingency.

### E. Write OLF's rule once, and test it without a GPU

Inside `reactiveLimitsBatch` the PV→PQ test and the switch-back test are each written twice, once
in the remote-member branch and once in the local branch. The branches differ in *where* they
read and write; the rule is the same. Several recorded defects are one branch drifting from the
other or from OLF (the cap bounding the switch and not the limit refresh; the slope exemption).

Proposed: the rule as small `__host__ __device__` functions in a header with no CUDA dependency
beyond the qualifier macro, hand-written (this document does not assume generation from OLF).

```cpp
// outerloop/reactive_limits_rules.cuh
OLF_HD int pvBusDecision(double q, double minQ, double maxQ, double tol,
                         bool robustRemote, double v, double vLow, double vHigh);
OLF_HD int pqBusDecision(int limitType, double controlledV, double targetV, bool canSwitchToPv,
                         double minQ, double maxQ, double frozenQ, double tol);
```

Both branches of the kernel call them. Because the header compiles as plain C++, a small host
test executable can exercise it in CI on a machine with no GPU: boundary cases transcribed from
OLF's `ReactiveLimitsOuterLoop` (exactly at `minQ - tol`, `v == targetV`, NaN target, each limit
type, the cap). There is no native test target today; this would be the first, and the same
harness serves the other loops' rules (deadbands, closest tap, ratchet).

- **Risk:** low; the kernel's comparisons are replaced by calls that must be the same
  comparisons. The decision records before and after are the check.
- **Value beyond size:** a rule defect is found by a unit test in seconds instead of by diffing
  24,000 trajectories.

### F. One placement function, and Java stops re-deriving it

"Where does this entry's reactive balance live in this scenario" depends on the coupler-tree
relocation, the pilot-self carrier and whether the group lost control. It is computed in the
check kernel, again in `exportReactiveLimitTargetQ` (which takes the placement tables a second time), and a
third time in Java:

```java
// GpuReactiveLimitsOuterLoop.frozenTargetQRow — repeats chooseCarrierRow and the relocation test
int relocated = CouplerTreeReHoming.relocatedTargetQRow(data.zeroImpedanceBranch().indices(), vRow);
if (relocated != vRow && devModes != null && relocated < devModes.length && devModes[relocated] != GpuRowMode.FIXED_VARIABLE) {
    vRow = relocated;
}
...
int carrier = ReactiveRelocationTables.chooseCarrierRow(data.reactiveRelocation(), g, devModes);
```

Its own comment records what a mismatch costs: a voltage target read as a reactive limit, 2.273
frozen on a bus whose limit was 0.774.

Proposed:

- one `__device__` function `reactiveBalanceRow(entries, placement, mode, s, i)` used by every
  kernel that needs the row;
- the base load flow's write-back uses the value the device already exports
  (`readBatchReactiveLimitTargetQ`, which the batched SA uses today) instead of `deviceTargets`
  plus `frozenTargetQRow`, and `frozenTargetQRow` is deleted.

- **Risk:** low to medium. The Java path and the export kernel must currently agree for this to
  be a no-op; if they do not, that is a defect this item finds.
- **Check:** assert, for one release, that the two agree on every entry of the base load flow
  across the suite, then delete the Java copy.

### G. Derive a voltage control's layout from its flags

This is the largest item and the only uncertain one.

After the reactive-limit kernel, up to four more kernels bring the group layout in line:
`launchReMergedVoltageTargetRowUpdate`, `pilotRowWithoutEnabledControllerBatch`,
`updatePilotTargetVEquationBatch`, `updateDistrQTargetsBatch` (`remote_voltage_control.cu` 419
lines, `pilot_voltage.cu` 462). They work by transition: remember the mode and target, flip the
row, restore later, guarded by "only rows this kernel flipped are flipped back — with one
exception". Their order is load-bearing (the comments record the measured failure of the reverse
order), and most reactive-limit defects in `docs/tasks.md` are in this code, not in the
comparisons.

OLF does not work by transition. `updateRemoteVoltageControlEquations` recomputes the layout from
the members' flags every time: the controlled bus holds its voltage equation if any member is
enabled; an enabled member holds a sharing equation unless it is the first enabled one; a member
that is not enabled holds its own reactive balance.

Proposed: one kernel per (scenario, group) that computes the same thing from the per-entry limit
types and the static group tables, and writes every row of the group — mode and target — from
that result alone. No saved mode, no saved target, no flip flags, no ordering between passes.

What makes this uncertain is that the device's layout is not OLF's activation one for one. A row
is permanent, so the result must also say which row carries a relocated balance (the carrier),
what a cross identity holds, and what a row holds when its coupler is open in this scenario. Some
of the current transition logic may encode a constraint a from-scratch derivation cannot meet.

So it should be built in **shadow** first, which fits the trace-channel rule (a channel only
prints, it never changes a result): under a new trace channel, compute the derived layout next to
the transition kernels and print every (scenario, row) where the two differ. Zero differences on
the busbar and injection scopes of rte6515 and network.xiidm is the precondition for replacing
anything; a non-zero list is either a defect in today's code or a constraint to write down.

- **Risk:** high if replaced directly, low in shadow.
- **Expected if it holds:** the two files shrink substantially and the ordering comments
  disappear. If it does not hold, the shadow run has documented exactly why.

### H. One status read-back per check

Each check ends with a stream synchronise and a copy of the `unstable` flags. Several loops add
their own synchronise-and-copy for a second flag: the pilot-self gate inside the reactive-limit
check, the slack distribution failure after the slack check, and the copies counted inside the
incremental loops (4 to 8 sync or copy sites each). #83 measured that a re-solve pass costs about
400 ms whether it solves 450 scenarios or 2,400, and left the fixed part to be broken down.

Proposed: one per-scenario status byte per check with bits `UNSTABLE`, `REJECT`, and a reason
code, written by every loop's kernels and read once.

- **Risk:** none on results.
- **Expected:** unknown until measured. This item should start with the breakdown #83 asked for
  (`OLF_GPU_PHASE_TIME`), not with the change.

## 3. Order

| step | item | results risk | needs a GPU to validate | depends on |
|---|---|---|---|---|
| 1 | A views | none | yes | — |
| 2 | D registry | none | yes | — |
| 3 | E rules header + host tests | low | tests no, kernel swap yes | — |
| 4 | C per-loop state | none | yes | A |
| 5 | F one placement function | low–medium | yes | A |
| 6 | B packed tables, generic install | low | yes | A, C |
| 7 | H one status read-back | none | yes (it is a measurement) | D |
| 8 | G derived layout, shadow then replace | high | yes | A, C, F |

Do ReactiveLimits first through steps 1 to 6, then judge by what is left of its 2,965 lines
before applying the pattern to the other loops. DistributedSlack is the natural second: its main
kernel has the largest signature.

**Validation, every step:** the GPU suite; per-loop decision records identical on rte6515 and
network.xiidm; both benches cold first, since several of these touch hot kernels' signatures.

## 4. What this does not fix

- The device still ports each OLF loop by hand. A change in OLF's rule still has to be noticed
  and copied; item E makes the copy one function with a test, not zero functions.
- The per-scenario channels remain: they are the price of one matrix for every contingency.
- The Java packers still walk the `LfNetwork`; item B simplifies what they emit, not what they
  have to know.
