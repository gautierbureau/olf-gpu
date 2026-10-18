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
import com.powsybl.openloadflow.OpenLoadFlowParameters.SlackDistributionFailureBehavior;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.SlackBusSelectionMode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A slack-only distribution — the slack generator is the only participant — on the device. OLF's water-fill
 * places what the slack generator's rooms allow and the failure behaviour decides on the remaining mismatch.
 * Eurostag with the slack at VLGEN: the load flow asks GEN to go down by 1.44 MW, so a minimum P of 606 MW
 * leaves it 1 MW of room.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSlackOnlyDistributionTest {

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    private static Network network(double minP) {
        Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        network.getGenerator("GEN").setMinP(minP);
        return network;
    }

    private static LoadFlowParameters params(boolean gpu, SlackDistributionFailureBehavior behavior) {
        LoadFlowParameters p = new LoadFlowParameters().setUseReactiveLimits(false).setDistributedSlack(true);
        OpenLoadFlowParameters ext = OpenLoadFlowParameters.create(p)
                .setSlackBusSelectionMode(SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("VLGEN"))
                .setSlackDistributionFailureBehavior(behavior);
        if (gpu) {
            ext.setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        }
        return p;
    }

    /** Runs both arms and returns {cpu, gpu}; asserts the statuses and the distributed active power are OLF's. */
    private static LoadFlowResult.ComponentResult[] compare(double minP, SlackDistributionFailureBehavior behavior,
                                                           long expectedFallbacks) {
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));
        Network cpu = network(minP);
        Network gpu = network(minP);
        LoadFlowResult.ComponentResult cpuResult = runner.run(cpu, params(false, behavior)).getComponentResults().get(0);
        long fallbacksBefore = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        LoadFlowResult.ComponentResult gpuResult = runner.run(gpu, params(true, behavior)).getComponentResults().get(0);
        assertEquals(expectedFallbacks, GpuNewtonRaphson.CPU_FALLBACK_COUNT.get() - fallbacksBefore, "solves the device refused");
        assertEquals(cpuResult.getStatus(), gpuResult.getStatus());
        assertEquals(cpuResult.getIterationCount(), gpuResult.getIterationCount());
        assertEquals(cpuResult.getDistributedActivePower(), gpuResult.getDistributedActivePower(), 1e-9);
        if (cpuResult.getStatus() == LoadFlowResult.ComponentResult.Status.CONVERGED) {
            assertEquals(cpu.getBusBreakerView().getBus("NLOAD").getV(), gpu.getBusBreakerView().getBus("NLOAD").getV(), 1e-9);
        }
        return new LoadFlowResult.ComponentResult[] {cpuResult, gpuResult};
    }

    /** Within the slack generator's rooms the device decides FAIL itself: no refusal. */
    @Test
    void failWithinTheRoomsRunsOnTheDevice() {
        var results = compare(-9999, SlackDistributionFailureBehavior.FAIL, 0);
        assertEquals(LoadFlowResult.ComponentResult.Status.CONVERGED, results[0].getStatus());
    }

    /** Beyond them OLF fails the load flow; the device flags the remaining mismatch and OLF's loop decides. */
    @Test
    void failBeyondTheRoomsFailsLikeOlf() {
        var results = compare(606, SlackDistributionFailureBehavior.FAIL, 1);
        assertEquals(LoadFlowResult.ComponentResult.Status.FAILED, results[0].getStatus());
    }

    /** LEAVE_ON_SLACK_BUS beyond the rooms: OLF reports what the rooms let it place (1 MW), not the mismatch. */
    @Test
    void leaveBeyondTheRoomsReportsWhatOlfPlaced() {
        var results = compare(606, SlackDistributionFailureBehavior.LEAVE_ON_SLACK_BUS, 0);
        assertEquals(-1.0, results[1].getDistributedActivePower(), 1e-9);
        assertTrue(results[1].getStatus() == LoadFlowResult.ComponentResult.Status.CONVERGED);
    }
}
