/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.openloadflow.ac.AcOuterLoopContext;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivationExtractor;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.ac.solver.AcSolverResult;
import com.powsybl.openloadflow.ac.solver.AcSolverStatus;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.lf.outerloop.OuterLoopResult;
import com.powsybl.openloadflow.lf.outerloop.OuterLoopStatus;
import com.powsybl.openloadflow.network.LfNetwork;

/**
 * PROTOTYPE — the core bridge of the hybrid batch solver (see {@code docs/hybrid-outer-loop-prototype.md}).
 *
 * <p>The batched SA solver reimplements OLF's reactive-limits / distributed-slack outer loops as CUDA
 * kernels. This bridge instead REUSES OLF's already-validated Java outer loops for the *decisions*, so
 * only the inner Newton solves need batching on the device. It chains two existing components and needs
 * no new switch-translation logic:
 *
 * <ol>
 *   <li>{@link AcOuterLoop#check} (e.g. {@code ReactiveLimitsOuterLoop}) reads the current state and
 *       MUTATES the equation-system activation on a PV↔PQ switch — the production-validated decision;</li>
 *   <li>{@link GpuAcActivationExtractor#extractActivation} reads that activation back into the device
 *       {@code rowMode} / {@code targetMap} the batched solver consumes.</li>
 * </ol>
 *
 * <p>One round of the hybrid outer loop, per scenario: load the device-solved state onto the network,
 * run OLF's {@code check}, and re-extract the row mode. The orchestration around this — per-scenario
 * contingency apply/undo, the round loop wired to {@code solveBatchContext} with empty RL/DS packs +
 * warm restart, retirement/straggler handling — is described in the design doc and is the remaining work.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuHybridOuterLoopBridge {

    private final LfNetwork network;
    private final EquationSystem<AcVariableType, AcEquationType> equationSystem;
    private final int n;
    private final java.util.List<AcOuterLoop> outerLoops;
    private final java.util.List<AcOuterLoopContext> outerLoopContexts;

    private boolean lastStable = true;
    private double lastSlackMismatch;
    private String lastSlackDetail = "";
    private final OuterLoopStatus[] lastStatuses;
    private final int[] unstableCount;                                // per-loop count of UNSTABLE results (oscillation finder)

    // --- OLF-faithful NESTED outer-loop state machine (persists across device rounds) ---
    // Mirrors AcloadFlowEngine.run(): the loops are nested (innermost first), each fully stabilized before
    // the next, and the whole sequence is re-cycled until a full pass triggers no inner re-solve. A device
    // round == one inner Newton re-solve, so this state advances one loop-check at a time (see applyOuterLoopStep).
    private int loopIdx = 0;                                          // the loop currently being stabilized
    private boolean initialized = false;                             // outer loops initialize()d for this scenario yet?
    private boolean cycleHadResolve = false;                         // did the current do/while pass trigger a re-solve
    private int outerLoopTotalIterations = 0;                        // global re-solve counter (OLF's outerLoopTotalIterations)
    private final int[] loopIteration;                               // per-loop-type iteration (OLF's outerLoopIterationByType)

    /**
     * @param outerLoops        the FULL ordered OLF outer-loop set for this scenario (innermost first —
     *                          distributed slack, then reactive limits, then transformer voltage control, …),
     *                          each already {@code initialize}d on its context
     * @param outerLoopContexts the matching per-loop contexts (loadflow context set, network attached)
     */
    public GpuHybridOuterLoopBridge(LfNetwork network,
                                    EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                    int n, java.util.List<AcOuterLoop> outerLoops,
                                    java.util.List<AcOuterLoopContext> outerLoopContexts) {
        this.network = network;
        this.equationSystem = equationSystem;
        this.n = n;
        this.outerLoops = outerLoops;
        this.outerLoopContexts = outerLoopContexts;
        this.lastStatuses = new OuterLoopStatus[outerLoops.size()];
        this.unstableCount = new int[outerLoops.size()];
        this.loopIteration = new int[outerLoops.size()];
    }

    /** Per-loop UNSTABLE counts across all passes — the outer loop that never stops for a scenario stands out. */
    public int[] unstableCount() {
        return unstableCount;
    }

    /** Names of the outer loops, index-aligned with {@link #unstableCount()}. */
    public String loopName(int i) {
        return outerLoops.get(i).getName();
    }

    public int loopCount() {
        return outerLoops.size();
    }

    /** Max |per-SC slack active-power mismatch (pu)| at the last step — huge/steady ⇒ islanding or DS not applied. */
    public double lastSlackMismatch() {
        return lastSlackMismatch;
    }

    /** The actual slack-bus set + per-bus target/mismatch fed to DS at the last step (compare vs the CPU LF). */
    public String lastSlackDetail() {
        return lastSlackDetail;
    }

    /**
     * One hybrid outer-loop step for a single scenario, faithful to OLF's NESTED {@code runOuterLoop}.
     *
     * <p>OLF's {@code AcloadFlowEngine.run()} does NOT check every outer loop once per Newton solve. It runs
     * the loops <em>nested and sequential</em>: {@code runOuterLoop} fully stabilizes ONE loop (check → on
     * UNSTABLE re-solve the inner Newton → re-check the SAME loop → … until STABLE) before moving to the next,
     * and the whole ordered sequence is re-cycled (the outer {@code do/while}) until a full pass triggers no
     * re-solve. A naive "check all loops every round" flat driver instead lets distributed-slack and
     * reactive-limits fire simultaneously against one stale state — which, for a slack-adjacent outage, sends
     * the distributed-slack loop into a never-settling crawl.
     *
     * <p>Here a device round equals exactly one inner Newton re-solve (the batch solves every active scenario
     * at once), so this method advances the nested state machine by the MINIMUM work that needs a re-solve:
     * it keeps checking loops in order against the current state — advancing on STABLE (a STABLE check never
     * mutates, so several loops can clear in one call with no re-solve) — until either a check returns
     * UNSTABLE (it mutated activation/targets, so a device re-solve is required: stay on this loop, return
     * with {@link #isStable()} false) or a full pass completes with no UNSTABLE (every loop simultaneously
     * stable → converged, {@link #isStable()} true). This reproduces OLF's "re-run the solver only on
     * UNSTABLE" and its DS-fully-then-RL-fully ordering exactly.
     *
     * @param scenarioState the flat converged state ({@code n} doubles) the batched inner solve produced
     * @return the device activation (row mode + target map) AFTER OLF's decisions; pair with
     *         {@link #isStable()} to know whether this scenario still needs another inner re-solve
     */
    public GpuAcActivation applyOuterLoopStep(double[] scenarioState) {
        // 1. load the batched inner-solve state so LfBus V/φ/Q reflect this scenario
        equationSystem.getStateVector().set(scenarioState);
        AcSolverUtil.updateNetwork(network, equationSystem);

        // 1b. FIRST step for this scenario: initialize() every outer loop exactly as OLF's runOuterLoop does
        //     before its first check — some loops set up required context data (DistributedSlack/ReactiveLimits)
        //     and, crucially, ShuntVoltageControl ENABLES its control ONLY in initialize() (its check() only
        //     rounds/disables at iteration 0), so without this the shunt is never voltage-controlled and freezes
        //     at its base section. If initialize CHANGED the activation (a control was enabled/disabled), return
        //     the fresh activation so the device re-solves under it BEFORE any check runs (matching OLF: solve
        //     after initialize, then check) — a one-terminal shunt needs that continuous solve before check()
        //     rounds. When it did NOT change the activation (DS/RL set only context data), fall straight through
        //     to the check loop to avoid a spurious extra re-solve round.
        if (!initialized) {
            initialized = true;
            int[] before = GpuAcActivationExtractor.extractActivation(network, equationSystem, n).rowMode();
            for (int i = 0; i < outerLoops.size(); i++) {
                outerLoops.get(i).initialize(outerLoopContexts.get(i));
            }
            GpuAcActivation after = GpuAcActivationExtractor.extractActivation(network, equationSystem, n);
            if (!java.util.Arrays.equals(before, after.rowMode())) {
                lastStable = false;
                return after;
            }
        }

        // per-synchronous-component slack active-power mismatch, exactly as NewtonRaphson builds it for the
        // solver result (the distributed-slack loop reads it from the context's last solver result).
        java.util.Map<Integer, Double> slackMismatch = new java.util.TreeMap<>();
        for (com.powsybl.openloadflow.network.LfSynchronousNetwork sc : network.getSynchronousNetworks()) {
            slackMismatch.put(sc.getNumSC(),
                    sc.getSlackBuses().stream().mapToDouble(com.powsybl.openloadflow.network.LfBus::getMismatchP).sum());
        }
        lastSlackMismatch = slackMismatch.values().stream().mapToDouble(Math::abs).max().orElse(0.0);
        // one-shot diagnostic: the actual slack bus set + per-bus mismatch this scenario feeds to DS (compare vs CPU LF)
        StringBuilder sd = new StringBuilder();
        sd.append("SCs=").append(network.getSynchronousNetworks().size());
        for (com.powsybl.openloadflow.network.LfSynchronousNetwork sc : network.getSynchronousNetworks()) {
            for (com.powsybl.openloadflow.network.LfBus sb : sc.getSlackBuses()) {
                sd.append(String.format(" [sc%d %s tP=%.3f mismP=%.3f disabled=%b]",
                        sc.getNumSC(), sb.getId(), sb.getTargetP(), sb.getMismatchP(), sb.isDisabled()));
            }
        }
        lastSlackDetail = sd.toString();
        AcSolverResult solverResult = new AcSolverResult(AcSolverStatus.CONVERGED, 1, slackMismatch);

        // 2. advance OLF's NESTED runOuterLoop state machine against this single device state. A STABLE check
        //    never mutates, so we sweep forward through the loop list without a re-solve; the first UNSTABLE
        //    check mutates and needs a device round, so we return there. (See the method javadoc.)
        while (true) {
            AcOuterLoop loop = outerLoops.get(loopIdx);
            AcOuterLoopContext ctx = outerLoopContexts.get(loopIdx);
            ctx.setIteration(loopIteration[loopIdx]);
            ctx.setOuterLoopTotalIterations(outerLoopTotalIterations);
            ctx.setLastSolverResult(solverResult);
            OuterLoopResult result = loop.check(ctx, ReportNode.NO_OP);
            lastStatuses[loopIdx] = result.status();

            if (result.status() != OuterLoopStatus.STABLE) {
                // UNSTABLE: this loop mutated the activation / targets / taps -> a device re-solve is required
                // before it (or any later loop) can be checked again. Stay on this loop (OLF's runOuterLoop
                // re-checks the SAME loop after the re-solve until it stabilizes).
                unstableCount[loopIdx]++;
                loopIteration[loopIdx]++;
                outerLoopTotalIterations++;
                cycleHadResolve = true;
                lastStable = false;
                return GpuAcActivationExtractor.extractActivation(network, equationSystem, n);
            }

            // STABLE: this loop is settled for now; move to the next one, checked against the SAME state.
            loopIdx++;
            if (loopIdx >= outerLoops.size()) {
                loopIdx = 0;
                if (cycleHadResolve) {
                    // OLF's outer do/while: a pass that re-solved must be followed by another full pass, because
                    // a later loop's re-solve can have destabilized an earlier one. Confirm on the (unchanged)
                    // state; if everything is still stable the next wrap exits with cycleHadResolve == false.
                    cycleHadResolve = false;
                    continue;
                }
                // full pass, no UNSTABLE anywhere -> every outer loop is simultaneously stable -> converged.
                lastStable = true;
                return GpuAcActivationExtractor.extractActivation(network, equationSystem, n);
            }
        }
    }

    /** @return true when the last {@link #applyOuterLoopStep} left every outer loop STABLE (no re-solve needed). */
    public boolean isStable() {
        return lastStable;
    }
}
