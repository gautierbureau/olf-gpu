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
    private final AcOuterLoop outerLoop;
    private final AcOuterLoopContext outerLoopContext;

    private OuterLoopStatus lastStatus = OuterLoopStatus.STABLE;

    /**
     * @param outerLoop        an OLF outer loop (reactive limits or distributed slack), already
     *                         {@code initialize}d on {@code outerLoopContext}
     * @param outerLoopContext its context, with the loadflow context set and the network attached
     */
    public GpuHybridOuterLoopBridge(LfNetwork network,
                                    EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                    int n, AcOuterLoop outerLoop, AcOuterLoopContext outerLoopContext) {
        this.network = network;
        this.equationSystem = equationSystem;
        this.n = n;
        this.outerLoop = outerLoop;
        this.outerLoopContext = outerLoopContext;
    }

    /**
     * One hybrid outer-loop step for a single scenario: load its device-solved state onto the network,
     * let OLF's validated outer loop decide (mutating the activation), and re-extract the device row
     * mode + target mapping from the mutated activation.
     *
     * @param scenarioState the flat converged state ({@code n} doubles) the batched inner solve produced
     * @return the device activation (row mode + target map) AFTER OLF's switch decisions; pair with
     *         {@link #isStable()} to know whether this scenario still needs another inner re-solve
     */
    public GpuAcActivation applyOuterLoopStep(double[] scenarioState) {
        // 1. load the batched inner-solve state so LfBus V/φ/Q reflect this scenario
        equationSystem.getStateVector().set(scenarioState);
        AcSolverUtil.updateNetwork(network, equationSystem);

        // 2. OLF's validated decision — mutates the equation-system activation on any PV↔PQ switch
        OuterLoopResult result = outerLoop.check(outerLoopContext, ReportNode.NO_OP);
        lastStatus = result.status();

        // 3. re-encode the (possibly switched) activation into the device row mode the batch consumes
        return GpuAcActivationExtractor.extractActivation(network, equationSystem, n);
    }

    /** @return true when the last {@link #applyOuterLoopStep} left the scenario STABLE (no re-solve needed). */
    public boolean isStable() {
        return lastStatus == OuterLoopStatus.STABLE;
    }
}
