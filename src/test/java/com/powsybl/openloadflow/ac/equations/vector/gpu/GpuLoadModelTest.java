/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Bus;
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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Voltage-dependent load models (ZIP + exponential) on the GPU inner Newton: the vdl family adds
 * {@code c_n·target·V^n} to the bus P/Q rows with the derivative on the own-V column — exactly OLF's
 * {@code AbstractLoadModelEquationTerm} (the constant part stays in the target vector). Mirrors OLF's
 * own {@code LoadModelTest} networks; the GPU must actually SOLVE (fallback counter asserted) and
 * match the CPU states at machine precision.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuLoadModelTest {

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    private static Network zipNetwork() {
        Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        network.getVoltageLevel("VLLOAD").newLoad()
                .setId("ZIPLOAD").setBus("NLOAD").setP0(50).setQ0(30)
                .newZipModel()
                    .setC0p(0.5).setC0q(0.55)
                    .setC1p(0.3).setC1q(0.35)
                    .setC2p(0.2).setC2q(0.1)
                    .add()
                .add();
        return network;
    }

    private static Network expNetwork() {
        Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        network.getVoltageLevel("VLLOAD").newLoad()
                .setId("EXPLOAD").setBus("NLOAD").setP0(50).setQ0(30)
                .newExponentialModel()
                    .setNp(0.8).setNq(0.9)
                    .add()
                .add();
        return network;
    }

    private void compare(Network cpuNetwork, Network gpuNetwork, String label) {
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));

        LoadFlowParameters cpuParameters = new LoadFlowParameters();
        OpenLoadFlowParameters.create(cpuParameters).setUseLoadModel(true)
                .setNewtonRaphsonConvEpsPerEq(1e-10);
        LoadFlowResult cpuResult = runner.run(cpuNetwork, cpuParameters);
        assertTrue(cpuResult.isFullyConverged(), "CPU reference load flow must converge");

        LoadFlowParameters gpuParameters = new LoadFlowParameters();
        long fallbacksBefore = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        OpenLoadFlowParameters.create(gpuParameters).setUseLoadModel(true)
                .setNewtonRaphsonConvEpsPerEq(1e-10)
                .setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        LoadFlowResult gpuResult = runner.run(gpuNetwork, gpuParameters);
        assertTrue(gpuResult.isFullyConverged(), "GPU load flow must converge");
        assertEquals(0, com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get() - fallbacksBefore,
                "the GPU must actually SOLVE the load-model system (a silent fallback proves nothing)");

        double worstV = 0;
        double worstA = 0;
        int compared = 0;
        for (Bus cpuBus : cpuNetwork.getBusView().getBuses()) {
            Bus gpuBus = gpuNetwork.getBusView().getBus(cpuBus.getId());
            if (gpuBus == null || Double.isNaN(cpuBus.getV()) || Double.isNaN(gpuBus.getV())) {
                continue;
            }
            worstV = Math.max(worstV, Math.abs(cpuBus.getV() - gpuBus.getV()));
            worstA = Math.max(worstA, Math.abs(cpuBus.getAngle() - gpuBus.getAngle()));
            compared++;
        }
        System.out.printf("GPU vs CPU [%s load model]: %d buses, max dV = %.2e kV, max dAngle = %.2e deg%n",
                label, compared, worstV, worstA);
        assertTrue(compared > 0, "must compare at least one bus");
        assertTrue(worstV < 1e-8, "bus voltages must match (max dV = " + worstV + " kV)");
        assertTrue(worstA < 1e-8, "bus angles must match (max dAngle = " + worstA + " deg)");
    }

    @Test
    void zipLoadModel() {
        compare(zipNetwork(), zipNetwork(), "ZIP");
    }

    @Test
    void exponentialLoadModel() {
        compare(expNetwork(), expNetwork(), "exponential");
    }
}
