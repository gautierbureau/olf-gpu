/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.SlackBusSelectionMode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.CONVERGED;
import static com.powsybl.loadflow.LoadFlowResult.ComponentResult.Status.MAX_ITERATION_REACHED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * OLF's {@code NewtonRaphson.run} counts a step BEFORE testing it and reports MAX_ITERATION_REACHED
 * whenever {@code iterations >= maxIterations}: a solve whose stopping test passes after the LAST
 * allowed step is not CONVERGED there. The device tested convergence before it tested the budget, so
 * with the budget set to the exact number of steps the case needs it answered CONVERGED where OLF
 * answers MAX_ITERATION_REACHED, and the base load flow mapped the status from OLF's criterion on the
 * returned state alone, never from the device's own verdict.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuIterationBudgetTest {

    static {
        GpuTestPaths.init();
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    private static LoadFlowParameters params(boolean gpu, int maxIterations) {
        LoadFlowParameters p = new LoadFlowParameters()
                .setUseReactiveLimits(false)
                .setDistributedSlack(false);
        OpenLoadFlowParameters ext = OpenLoadFlowParameters.create(p)
                .setSlackBusSelectionMode(SlackBusSelectionMode.FIRST)
                .setMaxNewtonRaphsonIterations(maxIterations);
        if (gpu) {
            ext.setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        }
        return p;
    }

    private static LoadFlowResult.ComponentResult.Status run(boolean gpu, int maxIterations) {
        Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        LoadFlowResult result = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()))
                .run(network, params(gpu, maxIterations));
        return result.getComponentResults().get(0).getStatus();
    }

    @Test
    void convergingOnTheLastAllowedStepIsMaxIterationReachedLikeOlf() {
        // the smallest budget OLF converges under: the case needs (budget - 1) steps
        int budget = -1;
        for (int m = 1; m <= 30 && budget < 0; m++) {
            if (run(false, m) == CONVERGED) {
                budget = m;
            }
        }
        assertTrue(budget >= 2, "the fixture must need at least one step, smallest converging budget " + budget);
        System.out.printf("iteration budget [Eurostag]: the CPU converges under %d, not under %d%n", budget, budget - 1);
        assertEquals(MAX_ITERATION_REACHED, run(false, budget - 1),
                "OLF: the test passes after the last allowed step and the status is MAX_ITERATION_REACHED");
        assertEquals(MAX_ITERATION_REACHED, run(true, budget - 1),
                "device: the same rule at the same budget");
        assertEquals(CONVERGED, run(true, budget),
                "device: one more step allowed and the solve is CONVERGED, like OLF (budget " + budget + ")");
    }
}
