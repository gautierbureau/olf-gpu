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
    private final OuterLoopStatus[] lastStatuses;
    private final int[] unstableCount;                                // per-loop count of UNSTABLE results (oscillation finder)

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

    /**
     * One hybrid outer-loop step for a single scenario: load its device-solved state onto the network,
     * let every OLF outer loop decide in order (each mutating the activation / targets / taps), and
     * re-extract the device row mode + target mapping from the mutated activation. The device runs a
     * PURE inner Newton, so the state fed to these decisions is OLF's own — not a device-outer-loop
     * approximation — which is what makes the decisions match the CPU.
     *
     * @param scenarioState the flat converged state ({@code n} doubles) the batched inner solve produced
     * @return the device activation (row mode + target map) AFTER OLF's decisions; pair with
     *         {@link #isStable()} to know whether this scenario still needs another inner re-solve
     */
    public GpuAcActivation applyOuterLoopStep(double[] scenarioState) {
        // 1. load the batched inner-solve state so LfBus V/φ/Q reflect this scenario
        equationSystem.getStateVector().set(scenarioState);
        AcSolverUtil.updateNetwork(network, equationSystem);

        // per-synchronous-component slack active-power mismatch, exactly as NewtonRaphson builds it for the
        // solver result (the distributed-slack loop reads it from the context's last solver result).
        java.util.Map<Integer, Double> slackMismatch = new java.util.TreeMap<>();
        for (com.powsybl.openloadflow.network.LfSynchronousNetwork sc : network.getSynchronousNetworks()) {
            slackMismatch.put(sc.getNumSC(),
                    sc.getSlackBuses().stream().mapToDouble(com.powsybl.openloadflow.network.LfBus::getMismatchP).sum());
        }
        lastSlackMismatch = slackMismatch.values().stream().mapToDouble(Math::abs).max().orElse(0.0);
        AcSolverResult solverResult = new AcSolverResult(AcSolverStatus.CONVERGED, 1, slackMismatch);

        // 2. every OLF outer loop, in order — the validated decisions, mutating activation / targets / taps
        lastStable = true;
        for (int i = 0; i < outerLoops.size(); i++) {
            outerLoopContexts.get(i).setLastSolverResult(solverResult);
            OuterLoopResult result = outerLoops.get(i).check(outerLoopContexts.get(i), ReportNode.NO_OP);
            lastStatuses[i] = result.status();
            if (result.status() != OuterLoopStatus.STABLE) {
                lastStable = false;
                unstableCount[i]++;
            }
        }

        // 3. re-encode the (possibly changed) activation into the device row mode the batch consumes
        return GpuAcActivationExtractor.extractActivation(network, equationSystem, n);
    }

    /** @return true when the last {@link #applyOuterLoopStep} left every outer loop STABLE (no re-solve needed). */
    public boolean isStable() {
        return lastStable;
    }
}
