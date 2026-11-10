# Proposal: array-form outer loops, one source for CPU and GPU

**Status:** stages 1 and 2 of the ReactiveLimits pilot are built (sections 5.1 and 5.2); the rest is a proposal (2026-11-10).
**Read first:** [`outer-loops-on-device.md`](outer-loops-on-device.md) — the current workflow and
the size accounting this proposal responds to.

## 1. The problem

Every OLF outer loop exists twice: OLF's Java (3,286 lines of code) and our device port (10,621).
The equations had the same problem and java2cuda solved it: the term classes are upstream's own
Java and the kernels are generated from them. The outer loops cannot use that path today, for a
reason that is about OLF's shape and not about the parser:

```java
// OLF today: input, decision and effect are one traversal of an object graph
context.getNetwork().<LfBus>getControllerElements(VoltageControl.Type.GENERATOR).forEach(bus -> {
    if (bus.isGeneratorVoltageControlEnabled()) {
        checkControllerBus(bus, pvToPqBuses, remainingPvBusCount);   // getMinQ(), getQ().eval(), setV(1), list.add
    } else {
        checkPqBus(bus, pqToPvBuses, busesWithUpdatedQLimits, !bus.hasGeneratorsWithSlope());
    }
});
```

There is nothing in that code a transpiler can lift: the inputs are virtual getters, the outputs
are setter calls, and the consequence for the equation system happens in a listener elsewhere.

The current device port also pays for two things it did not have to (causes 3 and 4 of the size
accounting): the table layout of each loop is written by hand four times (packer, JNI, kernel,
write-back), and the effect on the equation system is *edited* by transition kernels where OLF
*derives* it from flags.

## 2. The idea

Give every loop the same four-stage shape, with the loop's data as plain arrays, and make that
shape the contract between CPU and GPU:

```
              ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
 state x  ──► │  GATHER    ├──►│  DECIDE    ├──►│  DERIVE    ├──►│  APPLY     ├──► equation system
 network      │ inputs[]   │   │ decision[] │   │ activation │   │            │
              └────────────┘   └────────────┘   └────────────┘   └────────────┘
 CPU:          getters          SAME SOURCE      SAME SOURCE      LfBus setters, reports
 device:       qBuf, x          (generated)      (generated)      row mode / target / overrides
```

- **GATHER** is engine-specific. On the CPU it reads the `LfNetwork`; on the device it reads `x`,
  the reactive buffer and the per-scenario override tables.
- **DECIDE** is a pure function over primitives: inputs in, a decision code out. It is written
  once, in OLF, in the Java subset java2cuda parses, and generated as a `__device__` function.
- **DERIVE** is the activation as a pure function of flags (which equation each member of a
  control holds). OLF already computes it this way, but against objects; in array form it is also
  one source.
- **APPLY** is engine-specific. On the CPU it calls today's setters and writes today's reports,
  so nothing downstream of the loop changes. On the device it maps the activation to row modes
  and targets through the placement tables.

The arrays between the stages — one struct-of-arrays per loop, call it the loop's **vector** —
are the interoperability contract. OLF already has the pattern for equations
(`OLF: ac/equations/vector/AcNetworkVector`, `AcBusVector`, `AcBranchVector`); this extends it to
controls.

What this buys, against the four causes in the size accounting:

| cause | today | with this proposal |
|---|---|---|
| 1. the model does OLF's work | inherent | inherent (the device GATHER stays) |
| 2. one matrix for all contingencies | per-scenario tables and branches | per-scenario **overrides of vector fields**, one generic mechanism; DECIDE no longer sees them |
| 3. layout written four times | by hand | packer, JNI install and kernel signature generated from the vector's description |
| 4. effect edited, not derived | transition kernels, order-dependent | DERIVE recomputes a control's layout from flags, as OLF does |

It does not make the device port free. Honest expectation for ReactiveLimits: the decision is
about 40 lines of the 637 in `reactive_limits.cu`. Generating DECIDE alone removes rule drift,
not volume. The volume is in causes 3 and 4, which is why the pilot below goes all the way
through DERIVE before it is judged.

## 3. ReactiveLimits in array form

Worked against `OLF: ac/outerloop/ReactiveLimitsOuterLoop.java` at `144a54c`. The sketches are
meant to be exact about semantics; names are placeholders.

### 3.1 The vector

One entry per bus that OLF's loop iterates
(`getControllerElements(VoltageControl.Type.GENERATOR)`).

```java
/** Struct of arrays over the controller buses of generator voltage controls. */
public final class ReactiveLimitsVector {
    // --- static (per extraction) ---
    public final int[] busNum;
    public final boolean[] remoteController;      // isGeneratorRemoteController(bus)
    public final boolean[] canSwitchToPv;         // !bus.hasGeneratorsWithSlope()
    public final double[] controlledNominalV;     // forced-PV key 1
    public final int[] idRank;                    // forced-PV key 3 (rank of bus id)
    public final double[] initialTargetQ;         // sum of generators' targetQ (robust mode)

    // --- gathered before each check ---
    public final boolean[] disabled;
    public final double[] q;                      // bus.getQ().eval() + bus.getLoadTargetQ()
    public final double[] minQ, maxQ;             // bus.getMinQ(), bus.getMaxQ()   (curves at the current P)
    public final double[] v;                      // bus.getV()
    public final double[] controlledV;            // voltage control's controlled bus V
    public final double[] targetV;                // voltage control's target
    public final double[] targetP;                // forced-PV key 2

    // --- loop state (persists across checks) ---
    public final boolean[] voltageControlEnabled;
    public final byte[] qLimitType;               // NONE, MIN_Q, MAX_Q, MIN_REALISTIC_V, MAX_REALISTIC_V
    public final int[] pvPqSwitchCount;           // today a HashMap<String, MutableInt> in ContextData
    public final double[] frozenTargetQ;          // bus.getGenerationTargetQ() while PQ

    // --- outputs of a check ---
    public final byte[] decision;
    public final boolean[] resetV;                // robust mode: controllerBus.setV(1)
}
```

This is, field for field, what `ReactiveLimitsPacker` already extracts into `ReactiveLimitEntries`
(18 arrays). The difference is that OLF would own it and OLF's own loop would read it.

### 3.2 DECIDE: pure functions, in the transpilable subset

Each function below is an expression over primitives — locals, a single return, ternaries,
boolean helpers — which is the subset `codegen/java2cuda` already parses for the equation terms
(to be confirmed for `int` returns and `boolean` parameters).

```java
public final class ReactiveLimitsDecisions {

    public static final int NONE = 0;
    public static final int TO_PQ_MIN_Q = 1;
    public static final int TO_PQ_MAX_Q = 2;
    public static final int TO_PQ_MIN_REALISTIC_V = 3;
    public static final int TO_PQ_MAX_REALISTIC_V = 4;
    public static final int TO_PV = 5;
    public static final int REFREEZE_AT_LIMIT = 6;

    /** checkControllerBus: a bus with voltage control enabled. robustRemote = robustMode && remote controller. */
    public static int pvBusDecision(double q, double minQ, double maxQ, double tol,
                                    boolean robustRemote, double v, double vLow, double vHigh) {
        return q < minQ - tol ? TO_PQ_MIN_Q
             : q > maxQ + tol ? TO_PQ_MAX_Q
             : robustRemote && v < vLow ? TO_PQ_MIN_REALISTIC_V
             : robustRemote && v > vHigh ? TO_PQ_MAX_REALISTIC_V
             : NONE;
    }

    /** "Set V to a safe 1 p.u. for next computation": every robust remote controller that leaves PV out of the band. */
    public static boolean resetsV(int decision, boolean robustRemote, double v, double vLow, double vHigh) {
        return decision != NONE && robustRemote && (v < vLow || v > vHigh);
    }

    /** The target Q a PV -> PQ switch freezes. */
    public static double frozenQ(int decision, double minQ, double maxQ, double initialTargetQ) {
        return decision == TO_PQ_MIN_Q ? minQ
             : decision == TO_PQ_MAX_Q ? maxQ
             : initialTargetQ;
    }

    /** checkPqBusWithQLimitType: a bus switched to PQ, whose qLimitType is set. */
    public static int pqBusDecision(int qLimitType, double controlledV, double targetV, boolean canSwitchToPv,
                                    double minQ, double maxQ, double frozenQ, double tol) {
        boolean atMin = qLimitType == MIN_Q || qLimitType == MIN_REALISTIC_V;
        boolean wantsBack = atMin ? controlledV < targetV : controlledV > targetV;
        double limit = atMin ? minQ : maxQ;
        boolean limitMoved = (qLimitType == MIN_Q || qLimitType == MAX_Q) && Math.abs(limit - frozenQ) > tol;
        return wantsBack && canSwitchToPv ? TO_PV
             : limitMoved ? REFREEZE_AT_LIMIT
             : NONE;
    }
}
```

These four functions are the lines of `reactiveLimitsBatch` that are OLF's rule. Today the kernel
has them inline, twice (local and remote branch); the comparisons there were re-derived from
reading OLF, and several entries in `docs/tasks.md` are a comparison that drifted (the strict
`v < targetV` with no tolerance, the slope exemption, "the cap bounds the switch, not the
refresh").

The whole-network part is order-dependent, so it is a plain loop over the vector — sequential per
network on the CPU, one thread per scenario on the device, which is how the forced-PV election is
already ported:

```java
/** One check. Returns true when anything changed (OuterLoopStatus.UNSTABLE). */
public static boolean check(ReactiveLimitsVector r, double tol, boolean robust, double vLow, double vHigh, int maxPqPvSwitch) {
    int remainingPv = 0;
    int toPq = 0;
    for (int i = 0; i < r.busNum.length; i++) {
        r.decision[i] = NONE;
        r.resetV[i] = false;
        if (r.disabled[i]) {
            continue;
        }
        if (r.voltageControlEnabled[i]) {
            boolean robustRemote = robust && r.remoteController[i];
            int d = pvBusDecision(r.q[i], r.minQ[i], r.maxQ[i], tol, robustRemote, r.v[i], vLow, vHigh);
            r.decision[i] = d;
            r.resetV[i] = resetsV(d, robustRemote, r.v[i], vLow, vHigh);
            if (d == NONE) {
                remainingPv++;
            } else {
                toPq++;
            }
        } else if (r.qLimitType[i] != QLimit.NONE) {
            r.decision[i] = pqBusDecision(r.qLimitType[i], r.controlledV[i], r.targetV[i], r.canSwitchToPv[i],
                                          r.minQ[i], r.maxQ[i], r.frozenTargetQ[i], tol);
        }
    }
    if (toPq > 0 && remainingPv == 0) {
        r.decision[strongestCandidate(r)] = NONE;         // switchPvPq: "keep one bus PV, the strongest one"
    }
    boolean changed = false;
    for (int i = 0; i < r.busNum.length; i++) {
        if (r.decision[i] == TO_PV && r.pvPqSwitchCount[i] >= maxPqPvSwitch) {
            r.decision[i] = NONE;                         // switchPqPv: blocked by the cap
        }
        changed |= r.decision[i] != NONE;
    }
    return changed;
}
```

Two places where this is not a transcription and must be shown equivalent before it lands:

- OLF calls `setV(1)` and re-freezes a moved limit *during* the scan, and switches *after* it.
  Here everything is decided on the pre-check inputs and applied afterwards. The device already
  does it this way (`reactiveLimitsRobustVoltageResetBatch` runs as its own pass "reading the same
  pre-pass x"), and the trajectories match OLF bin for bin (`docs/tasks.md` #79), but the argument
  belongs in the OLF change: a controller's `setV(1)` can only reach another entry's decision
  through `controlledV`, i.e. when a remote controller is itself another control's controlled bus.
- A bus the election keeps PV still has its V reset in OLF when it was out of the band. `resetV`
  is computed before the election, so that is preserved; it needs a test of its own.

### 3.3 CPU side: the loop becomes gather → check → apply

```java
@Override
public OuterLoopResult check(AcOuterLoopContext context, ReportNode reportNode) {
    ReactiveLimitsVector r = ((ContextData) context.getData()).vector;
    gather(context.getNetwork(), r);                                  // today's getters, into arrays
    boolean changed = ReactiveLimitsDecisions.check(r, maxReactivePowerMismatch, robustMode, vLow, vHigh, maxPqPvSwitch);
    apply(context.getNetwork(), r, reportNode);                       // today's setters and reports
    return new OuterLoopResult(this, changed ? OuterLoopStatus.UNSTABLE : OuterLoopStatus.STABLE);
}

private void apply(LfNetwork network, ReactiveLimitsVector r, ReportNode reportNode) {
    for (int i = 0; i < r.busNum.length; i++) {
        LfBus bus = network.getBus(r.busNum[i]);
        if (r.resetV[i]) {
            bus.setV(1);
        }
        switch (r.decision[i]) {
            case TO_PQ_MIN_Q, TO_PQ_MAX_Q, TO_PQ_MIN_REALISTIC_V, TO_PQ_MAX_REALISTIC_V -> {
                bus.freezeGenerationTargetQAndDisableGeneratorVoltageControl(
                        ReactiveLimitsDecisions.frozenQ(r.decision[i], r.minQ[i], r.maxQ[i], r.initialTargetQ[i]));
                bus.setQLimitType(toQLimitType(r.decision[i]));
                r.pvPqSwitchCount[i]++;
            }
            case TO_PV -> {
                bus.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(true);
                bus.setQLimitType(null);
            }
            case REFREEZE_AT_LIMIT -> bus.freezeGenerationTargetQAndDisableGeneratorVoltageControl(
                    r.qLimitType[i] == QLimit.MIN_Q ? r.minQ[i] : r.maxQ[i]);
            default -> { }
        }
    }
    // reports: same nodes as today, built from the decision codes
}
```

The setters, the listener and the equation system are untouched, so OLF's behaviour is unchanged
by construction and its existing tests are the regression suite. `gather` is O(controllers) per
check, small against the Newton solve that follows an UNSTABLE check; if it shows up in a profile
the vector can be listener-maintained like `AcNetworkVector`.

The remote reactive power controllers (`getReactivePowerControllerElements`, the second list in
`check`) reuse `pvBusDecision` on a second, smaller vector.

### 3.4 Device side: what changes

- **DECIDE:** a new java2cuda family, `reactive_limits_decisions`, renders the four functions
  into `native/cuda/equations/generated/` with the same prove/golden/drift-guard discipline as the
  equation families. `reactiveLimitsBatch` calls them in place of its inline comparisons, once,
  for local and remote entries alike.
- **GATHER:** stays hand-written and device-specific — which row holds this scenario's q, which
  limits apply. It fills the same fields the Java vector has, per (scenario, entry). The
  per-scenario channels (islanded, generators lost, re-merged, compared-row override) become
  overrides of named vector fields (`disabled[s,i]`, `controlledV` row, `targetV[s,i]`).
- **Layout plumbing:** the vector's field list drives a generator for the packer record, the JNI
  upload functions and the kernel parameter list, so adding a field is one line instead of four
  edits.
- **Write-back:** the device hands back `decision`-equivalent final state in the vector's own
  layout; `GpuReactiveLimitsOuterLoop.applyToNetwork` becomes OLF's `apply` on that vector
  instead of a second implementation with its own row-choice logic.

### 3.5 DERIVE: the activation as a function of flags

This is the part with the most to gain and the most to prove. OLF's rule for a generator voltage
control, in `AcEquationSystemCreator.updateRemoteVoltageControlEquations`, reads as a function:

```
inputs  per member i of the control:  disabled[i], enabled[i]        (enabled = voltage control on)
outputs per control:   vEqActive      = any member enabled
        per member:    targetQActive  = !enabled[i]
                       distrQActive   = enabled[i] && i is not the first enabled member
```

In array form:

```java
/** Activation of one voltage control's equations, from its members' flags. */
public static void deriveGroup(int start, int end, boolean[] disabled, boolean[] enabled,
                               boolean[] targetQActive, boolean[] distrQActive, boolean[] vEqActive, int group) {
    int firstEnabled = -1;
    for (int i = start; i < end; i++) {
        if (disabled[i]) {
            continue;                                 // OLF discards disabled controller elements
        }
        targetQActive[i] = !enabled[i];
        distrQActive[i] = enabled[i] && firstEnabled >= 0;
        if (enabled[i] && firstEnabled < 0) {
            firstEnabled = i;
        }
    }
    vEqActive[group] = firstEnabled >= 0;
}
```

On the CPU, `AcEquationSystemUpdater` would call it and then `setActive` from the three arrays.
On the device, one kernel per (scenario, group) would run the generated `deriveGroup` after
DECIDE and map its output to row modes through the existing placement tables (carrier row,
relocated row), replacing the transition kernels of section 3.7 of the workflow document
(`pilotRowWithoutEnabledControllerBatch`, the pilot-V flip/restore, the `DISTR_Q` target
hand-off, `savedMode`/`savedTarget`).

**This is a hypothesis.** The mapping from OLF's activation to the device's rows is where the
fixed-matrix constraints live (relocation, cross identities, pilot-self carriers, the hidden and
merged controls of `updateRemoteVoltageControlEquations`'s other branch), and some of the current
transition logic may encode a constraint a from-scratch derivation cannot meet. The pilot's last
stage exists to find out on real networks, not to assume.

## 4. Interoperating CPU and GPU through the vector

With a loop in this form, three execution modes share one contract, per loop:

| mode | GATHER | DECIDE + DERIVE | APPLY | use |
|---|---|---|---|---|
| CPU | `LfNetwork` | Java | setters | OLF, unchanged behaviour |
| device | kernels | generated kernels | kernels | production batch |
| **host-decide** | kernels, then download the vector's input fields | Java, in parallel over scenarios | upload `decision[]`, device APPLY | a loop not generated yet; an oracle for one that is |

Host-decide is what the retired hybrid engine should have been. That engine rebuilt each
scenario's whole `LfNetwork` state before every check (its last profile: 54% snapshots, 44% Java
loops, 2% device). Here the host never touches an `LfNetwork` during the solve: it receives
`nScenarios × nEntries` rows of the fields in section 3.1, runs `ReactiveLimitsDecisions.check`
on each scenario's slice (independent, so a parallel stream), and returns one byte per entry.
pegase9241 has about 1,600 controller buses; with five gathered doubles each that is about 64 kB
per checked scenario per check, and no object graph. Only the checked scenarios' slices cross.

It also gives a decision-level oracle that today's tests cannot: the same inputs through the Java
function and the generated function must give the same bytes. Because a `__device__` function
with no CUDA intrinsics is plain C++, that comparison compiles and runs on the host, in CI,
without a GPU.

## 5. The pilot: ReactiveLimits, end to end

ReactiveLimits is the right first loop: it is 73% of the Newton iterations on the production
case (`docs/tasks.md` #79), it has the most recorded drift, and its effect layer is the hardest
one, so a design that survives it will survive the others.

Each stage has its own exit test and is useful if the next one is never done.

| stage | what | where | exit test |
|---|---|---|---|
| 0 | Freeze a reference: per-check decision records on both arms (`CPU_RL_CHECK` on the fork, `OLF_GPU_RL_REASONS=1` on the device) for pegase9241, rte6515, network.xiidm | — | counts reproduce #84 (170,898 unstable checks on network.xiidm) |
| 1 | `ReactiveLimitsVector` + `ReactiveLimitsDecisions` + gather/apply in OLF's loop | OLF fork, `olf-local-patches` | OLF's test suite green; stage-0 CPU records byte-identical |
| 2 | java2cuda family for the decision functions; golden manifest; host-compiled equivalence test against Java on recorded and random inputs | this repo, `codegen/` | prove passes; equivalence exact; runs in `codegen-check.yml` |
| 3 | `reactiveLimitsBatch` calls the generated functions; local and remote branches share them | `native/cuda/outerloop/` | GPU suite green; stage-0 device records identical; rte6515 and network.xiidm parity unchanged |
| 4 | Host-decide mode for this loop behind a trace channel | this repo | same results as stage 3; measured µs per scenario-check and total overhead vs in-kernel |
| 5 | `deriveGroup` in OLF's updater and as a device kernel replacing the transition kernels | both | busbar and injection scopes unchanged; `remote_voltage_control.cu` + `pilot_voltage.cu` shrink |
| 6 | Generate packer / JNI / kernel signature from the vector description | this repo | no hand-written upload code left for this loop |

### 5.1 Stage 1 as built (2026-11-10)

On the OLF fork, branch `claude/hopeful-edison-8lqr3p`, cut from `olf-local-patches` and one commit ahead of it
(`1f53c13`); `olf-local-patches` itself is untouched. Three classes in `OLF: ac/outerloop/`:
`ReactiveLimitsVector` (the arrays), `ReactiveLimitsDecisions` (the rules, no network), and
`ReactiveLimitsOuterLoop.check` rewritten as gather, decide, apply with the same setters, reports and traces.

**Checked.** OLF's 1,531 tests pass before and after. With `OLF_RL_TRACE=1` and `OLF_RL_TRACE_BUS=""` (every bus)
and the loggers silenced, the whole suite prints 6,945 decision records (`RL_PVCHECK`, `RL_DECIDE`, `RL_REFREEZE`,
`RL_PASS`, `RL_SWITCH`): the same records before and after, in the same order for 147 of the 148 contingency
labels; for the `base` label the order differs only where multi-threaded tests print. The suite reaches every
branch: the strongest bus kept PV (22 times), a capped switch back, a re-freeze at a moved limit (9), the remote
reactive power controllers, a realistic-voltage switch. The decisions also have their own test without a
network (`ReactiveLimitsDecisionsTest`).

**Where it differs from the sketch in section 3**, and why:

- **A bus is read, decided and given its in-scan effects before the next bus is read.** Section 3.2 proposed
  deciding every bus on the pre-check inputs. That is not always what OLF does: a bus's field holds one generator
  voltage control, and a bus that is the controlled bus of one control and a remote controller of another keeps
  the second, so its `setV(1)` can be read as the controlled voltage of a later bus. Stage 1 keeps OLF's order
  exactly; `ReactiveLimitsDecisions.decideEntries` (all at once) is there for an engine that decides on a whole
  vector, with that one stated difference.
- **Two more decision codes**, `TO_PV_BLOCKED` and `KEPT_PV`: the loop reports both, so "nothing happens" was
  not enough.
- **The election keys are gathered only when the election runs** (every controlling bus would switch), and the
  id rank is among its candidates only. Ranking every bus id at every check would have cost a sort per check.
- **The entries are rebuilt at every check**, since the controller elements change with contingencies and
  merged controls; the switch counts are kept per bus number.

**Not checked:** nothing here ran against olf-gpu (no GPU in the container). olf-gpu uses only the loop's public
API, which is unchanged. The fork's two pre-existing checkstyle failures (`AbstractSecurityAnalysis`,
`NewtonRaphson`) are not from this change.

### 5.2 Stage 2 as built (2026-11-10)

On this repository's branch `claude/reactive-limits-decisions-codegen` (cut from `main`, not merged), with one
more commit on the OLF fork's pilot branch (`a72acc1`).

- **OLF side:** `ReactiveLimitsDecisionsVectorsTest` keeps
  `src/test/resources/reactiveLimitsDecisionsVectors.txt`: 2,644 calls of the rules with the Java's result, one
  per line, doubles as the hexadecimal digits of their bits. They stand on each comparison of the rules, one bit
  under it, one bit over it, far from it and undefined, then at 300 seeded random points. The test fails when the
  file is not what the Java gives now.
- **Here:** a second java2cuda shape, the rule family (`codegen/java2cuda/rules.py`). It keeps the Java
  statements one for one with their types — the existing parser handles `double` expressions only and inlines
  everything into one — and emits `native/cuda/outerloop/generated/reactive_limits_decisions.cuh`: seven
  functions (`pvBusDecision`, `resetsV`, `frozenQ`, `isMinLimit`, `pqBusDecision`, `isSwitchToPq`,
  `limitTypeOf`) and the fourteen decision and limit codes.
- **Proof, no GPU:** parse == golden manifest; render == committed header; and the committed header, compiled
  for the host with `g++`, gives the Java's result on every reference vector, bit for bit. Four mutations of the
  header (`<` to `<=`, `>` to `>=`, a wrong frozen target) are each caught. `codegen/regenerate.sh` passes with
  the six equation families unchanged.

**What it does not show.** That a kernel calls these functions with the right arguments: none does yet. The
functions that walk the vector (`keepStrongestPv`, `blockSwitchesToPv`) have loops and are not transpiled; the
parser refuses them. Section 2's expectation stands: this removes rule drift, not volume.

**A dependency to mind.** The rule family is generated from a class that is on the fork's pilot branch and not on
`olf-local-patches`. `codegen-check.yml` on that branch therefore checks the pilot branch out (`OLF_REF`). The
two OLF commits have to reach `olf-local-patches` before the codegen branch merges, and `OLF_REF` goes back then.

**Decision points.**

- After stage 1: is the refactored loop something upstream OLF would take? If not, it lives on
  `olf-local-patches` and the upstream-merge cost of that patch is the price.
- After stage 4: the measured cost of deciding on the host is the answer to "what do we pay to
  keep outer loops in Java". Rough budget from existing numbers: network.xiidm runs about 207,000
  re-solves in a 110 s GPU run, so host-decide must cost about 30 µs per scenario-check to stay
  within 10% (an estimate, to be replaced by the measurement). If it meets it, loops that are
  rare or awkward to port can stay in Java permanently.
- After stage 5: if a derived layout cannot replace the transition kernels, stop there. Stages
  1–4 still leave one decision source and the oracle; the effect layer stays hand-written and
  this document should say why.

**What to measure at the end**, against today's ReactiveLimits numbers (331 lines in OLF; 2,965
on the device: CUDA 1,487, JNI 472, Java 1,006): hand-written device lines remaining for the
loop, and reactive-limit defects found by the decision oracle versus by trajectory diffing.

## 6. After the pilot

Order by how much of each loop is DECIDE versus effect:

1. **MonitoringVoltage, the discrete transformer and shunt voltage controls, phase control** —
   deadbands, clamps, closest tap: almost pure DECIDE over small vectors, effect is one override.
2. **The incremental loops** (transformer, shunt, flow) — DECIDE plus a sensitivity as a gathered
   input; the ratchet state (`IncrementalContextData`) becomes vector state.
3. **DistributedSlack / AreaInterchangeControl** — the participation water-fill is a loop over
   participants per network, the same shape as the forced-PV election.
4. **SecondaryVoltageControl** — the dense zone system stays a library call on both sides; only
   the trigger and the commit are DECIDE.
5. **AutomationSystem** — changes topology; last, and possibly never (host-decide or CPU).

## 7. Risks

- **Upstream.** The value depends on OLF's own loop reading the vector. A fork-only refactor
  trades two implementations for a patch to rebase.
- **Parser scope.** The decision functions fit the expression subset; `check` and `deriveGroup`
  need `for`, `if`, array writes and compound assignment. That is a real extension of java2cuda,
  with its own proof obligations (the finite-difference audit does not apply to integer
  decisions; exhaustive and recorded-input equivalence replaces it).
- **Report fidelity.** OLF's reports name buses and carry values in a specific order; `apply`
  must reproduce them from decision codes. Any reordering is a visible diff for OLF's users.
- **Semantics of deferred effects.** Section 3.2's two bullets. Small, but they are the kind of
  difference that shows up as one contingency in 24,000.
- **DERIVE on the device.** Section 3.5: unproven, and the place where the fixed matrix may say
  no.
