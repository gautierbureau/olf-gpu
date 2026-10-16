/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.extensions.ActivePowerControlAdder;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
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
 * The GPU load flow as a SAFE DEFAULT: a load flow the device driver refuses goes, for the rest of the run,
 * to OLF's own CPU Newton-Raphson and its own outer loops, and produces exactly the CPU solution — never
 * wrong, only un-accelerated. The refused shape here is a slack distribution whose failure behaviour the
 * device cannot apply ({@code FAIL}) with nobody participating: the only generator does not participate, and
 * the mismatch is inside OLF's slack band, so OLF converges without distributing.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuCpuFallbackTest {

    static {
        GpuTestPaths.init();
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    private static Network network() {
        Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        network.getGenerator("GEN").newExtension(ActivePowerControlAdder.class).withParticipate(false).add();
        return network;
    }

    private static LoadFlowParameters params(boolean gpu) {
        LoadFlowParameters p = new LoadFlowParameters().setUseReactiveLimits(false).setDistributedSlack(true);
        OpenLoadFlowParameters ext = OpenLoadFlowParameters.create(p)
                .setSlackBusSelectionMode(SlackBusSelectionMode.NAME)
                .setSlackBusesIds(List.of("VLGEN"))
                .setSlackDistributionFailureBehavior(OpenLoadFlowParameters.SlackDistributionFailureBehavior.FAIL)
                .setSlackBusPMaxMismatch(5);
        if (gpu) {
            ext.setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        }
        return p;
    }

    @Test
    void refusedLoadFlowRunsOnTheCpuNewtonRaphson() {
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));

        Network cpu = network();
        LoadFlowResult cpuResult = runner.run(cpu, params(false));
        assertTrue(cpuResult.isFullyConverged());

        long fallbacksBefore = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        Network gpu = network();
        LoadFlowResult gpuResult = runner.run(gpu, params(true));
        assertTrue(gpuResult.isFullyConverged());
        assertTrue(GpuNewtonRaphson.CPU_FALLBACK_COUNT.get() > fallbacksBefore, "the device must refuse this load flow");

        assertEquals(cpuResult.getComponentResults().get(0).getDistributedActivePower(),
                gpuResult.getComponentResults().get(0).getDistributedActivePower(), 0.0);
        for (String id : new String[]{"NGEN", "NHV1", "NHV2", "NLOAD"}) {
            assertEquals(cpu.getBusBreakerView().getBus(id).getV(), gpu.getBusBreakerView().getBus(id).getV(), 0.0,
                    "the CPU Newton-Raphson's solution (V @ " + id + ")");
            assertEquals(cpu.getBusBreakerView().getBus(id).getAngle(), gpu.getBusBreakerView().getBus(id).getAngle(), 0.0,
                    "the CPU Newton-Raphson's solution (angle @ " + id + ")");
        }
    }
}
