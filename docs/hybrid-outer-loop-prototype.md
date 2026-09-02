# Hybrid outer-loop batch solver — design + prototype

**Branch:** `hybrid-java-outer-loop`
**Status:** prototype / design (2026-08-14)

## Motivation

The batched SA solver (`solveBatchContext`) **reimplements** OLF's reactive-limits and
distributed-slack outer loops as CUDA kernels. The single-solve GPU path
(`GpuNewtonRaphson`) does not — it reuses OLF's Java `AcloadFlowEngine` outer loops and only
replaces the inner-Newton linear solve, and it matches the CPU to ~1e-12 at any tolerance.

The reimplementation is a correctness liability:

- A port audit (2026-08-14) found the RL/DS orchestration and the local switch rules
  faithful, but **two latent gaps**: the "keep one bus PV" safeguard (`switchPvPq`, when
  every PV bus would switch to PQ) and the voltage-dependent Q-limit re-pin
  (`checkPqBusWithQLimitType`). Neither triggers on pegase, but both are real.
- At the production 1e-4 criterion the batch and CPU diverge by ~2e-4 on a handful of
  contingencies, root-caused to a PV↔PQ limit-cycle on two boundary generators
  (`VL-7616_0`, `VL-9110_0`) where the batch retires a scenario one outer step before OLF's
  engine settles. It is *not* a rule bug — it is the loose-criterion non-uniqueness — but it
  is a second implementation reaching a different valid fixpoint than the validated one.

## The idea

Keep OLF's **validated Java** outer-loop *decisions*; batch only the *inner Newton solves* on
the device. The switch decisions are cheap (a couple of comparisons per controller); the
Jacobian factorization + solve — what cuDSS batches — is the bottleneck.

### Why it is throughput-neutral (measured)

`OLF_GPU_DBG_TIME=1`, full 500-scenario pegase batch, production 1e-4 full-loadflow:

```
BATCHTIME nS=500  wall≈11477ms
  inner (fill + cuDSS + step)      = 89.7%
  outer (Q-buffer + RL/DS kernels) = 10.2%   ← most of this is the Q-buffer fill, which stays
  other (host readbacks)           =  0.07%
  outerPasses = 70
```

The RL/DS **decision** kernels are ~1-3% of wall time. The hybrid removes them and adds a
per-pass pull of controller Q+V (`nrl·nS·2·8B` ≈ 5.8 MB/pass × 70 ≈ 400 MB ≈ ~35 ms ≈ 0.3%).
Net: within noise.

### Why it should reproduce the CPU (strong prior)

Single-solve already proves `OLF Java outer loops + GPU inner solve == CPU to 1e-12`. The
hybrid is that same combination with the inner solve **batched**. Making the switch decisions
with OLF's exact code on OLF-equivalent states should dissolve the `VL-7616` divergence.

## Architecture

The bridge needs **no new switch-translation code** — it chains two existing components:

1. `ReactiveLimitsOuterLoop.check(context)` / `DistributedSlackOuterLoop` — mutate the
   equation-system **activation** (deactivate `BUS_TARGET_V`, activate `BUS_TARGET_Q` on a
   PV→PQ switch, etc.). Already validated in production.
2. `GpuAcActivationExtractor.extractActivation(network, es, n)` → `GpuAcActivation(rowMode,
   targetMap)` — reads that activation into the device row-mode + target mapping. Already the
   batch's own activation encoder.

### Orchestration (Java, per chunk)

```
seed rowMode[s]/target[s] from the base extraction for every scenario s
loop outer round r = 0,1,2,… until every scenario is STABLE or maxOuterLoopIterations:
    states = solveBatchInner(rowMode, target, x0PerScenario=states)   // native, RL/DS packs EMPTY:
                                                                       // pure batched inner Newton to convergence
    for each active scenario s:                                        // cheap Java, per scenario
        applyState(baseNetwork, states[s])       // set LfBus V/φ from the flat device state
        applyContingency(baseNetwork, s)         // disable branch(es) for scenario s
        status_s = reactiveLimitsOuterLoop.check(ctx_s)   // + distributedSlack; MUTATES es activation
        act = GpuAcActivationExtractor.extractActivation(baseNetwork, es, n)
        rowMode[s] = act.rowMode(); target[s] = extractTargets(es, act, …)
        stable[s] = (status_s == STABLE)
        undoContingency(baseNetwork, s); restore es activation baseline
    if all stable: break
return states
```

- The inner solve is `solveBatchContext` with **empty RL/DS packs** (`fullLoadflow=false` →
  one batched Newton to convergence for the active scenarios), reusing `x0PerScenario` for the
  warm restart across rounds. No device outer-loop kernels run.
- One `LfNetwork` instance is mutated per scenario per round (as OLF's CPU SA already does
  within a thread). The state-set + check + re-extract is the only per-scenario host work.

## Prototype status

- `GpuHybridOuterLoopBridge` (this branch) implements the **core bridge** — the one novel
  step: set a scenario's device state onto the `LfNetwork`, run OLF's
  `ReactiveLimitsOuterLoop.check`, and re-extract `rowMode` from the mutated activation. This
  is the piece the whole design hinges on; everything else is orchestration around existing
  batch machinery.
- **Remaining for a full solver:** per-scenario contingency apply/undo + `AcOuterLoopContext`
  lifecycle management, the round loop wired to `solveBatchContext` with empty packs +
  warm-restart, `DistributedSlackOuterLoop` in the same round, straggler/compaction handling,
  and the retirement bookkeeping the native driver already does.

## Prototype run findings (2026-08-14) — IMPORTANT, corrects the hypothesis

`GpuHybridBatchValidationTest` drives the two divergent contingencies with the device doing
inner Newton + **device DS**, and OLF's Java `ReactiveLimitsOuterLoop` deciding RL. It compiles,
runs on pegase, and converges cleanly (2 rounds with per-scenario `NetworkState` snapshots).

**Result: VL-7616_0 ends PV — same as the device kernel batch, still NOT CPU's PQ.** Moving RL
alone to Java does NOT fix the divergence. Why: the RL switch DECISION depends on the STATE, and
the state is still produced with the **device distributed-slack** (waterfill). The ~4e-7 DS
difference vs OLF's `ActivePowerDistribution`, amplified at VL-7616's maxQ boundary, keeps its Q
just under maxQ — so OLF's check, fed that state, correctly holds it PV. **The DS is the seed, not
a bystander.**

Two consequences for a full solver:
1. **DS must move to Java too** (device = pure inner Newton, empty RL AND DS packs) so the state
   feeding the decisions matches CPU. Reuse `DistributedSlackOuterLoop` the same way, and rebuild
   the target vector after it redistributes generator P (a fresh `AcTargetVector` re-reads it).
   Transformer voltage control (AFTER_GENERATOR in the RTE params) would need the same treatment
   or the state won't fully match CPU.
2. **The activation-extraction bridge needs to reliably capture OLF's switches:** the prototype
   showed a case (v=1.1323 with a PV flag) where the `setGeneratorVoltageControlEnabled` switch
   did not fully propagate into the equation activation that `GpuAcActivation.fromEquationSystem`
   re-reads — so the device rowMode lagged the Java decision. Verify the equation-system listener
   / activation is in sync after each `check` before extracting.

Net: the hybrid architecture is sound and runs, but "fall in line with CPU" requires the FULL
outer loop (DS+RL, and transformer VC) in Java — not RL alone. The prototype's value was pinning
this down empirically.

## Validation plan

1. **Correctness:** run `GpuVsCpuProductionSaBenchmarkTest#productionSaGpuVsCpu` at the shared
   production 1e-4 with the hybrid solver; expect `onlyGPU=0` (vs the device path's
   `onlyGPU=5`) — i.e. the `VL-7616` divergence gone, because decisions come from OLF's code.
2. **Throughput:** compare wall time to the device-kernel path; expect within a few percent
   (the ~0.3% transfer cost vs the ~1-3% kernel saving).
3. **Regression:** the full 110-test GPU suite stays green.
