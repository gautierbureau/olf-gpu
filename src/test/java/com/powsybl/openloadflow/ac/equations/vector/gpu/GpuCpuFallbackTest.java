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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The GPU solver as a SAFE DEFAULT: a network whose equation system the GPU extractor
 * rejects (here a ZIP load model — an unsupported variable-length equation term) must
 * still converge through {@link com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson}'s
 * CPU fallback ({@code olf.gpu.cpuFallback}, default on), producing the same solution
 * as a plain CPU run — never wrong, only un-accelerated. This is what makes routing the
 * security analysis's base load flow to the GPU solver unconditionally safe.
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

    private static Network zipLoadNetwork() {
        // Eurostag + a ZIP-model load: the ZIP equation term (variable-length sum) is outside
        // the GPU extractor's scope, so extraction rejects the system and the solver must
        // fall back to the CPU Newton-Raphson.
        Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        network.getVoltageLevel("VLLOAD").newLoad()
                .setId("ZIPLOAD")
                .setBus("NLOAD")
                .setP0(50)
                .setQ0(30)
                .newZipModel()
                    .setC0p(0.5)
                    .setC0q(0.55)
                    .setC1p(0.3)
                    .setC1q(0.35)
                    .setC2p(0.2)
                    .setC2q(0.1)
                    .add()
                .add();
        return network;
    }

    private static LoadFlowParameters params(boolean gpu) {
        LoadFlowParameters p = new LoadFlowParameters().setUseReactiveLimits(false).setDistributedSlack(false);
        OpenLoadFlowParameters ext = OpenLoadFlowParameters.create(p)
                .setSlackBusSelectionMode(SlackBusSelectionMode.FIRST)
                .setUseLoadModel(true);   // makes the ZIP term real — the GPU extractor rejects it
        if (gpu) {
            ext.setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        }
        return p;
    }

    @Test
    void gpuSolverFallsBackToCpuOnUnsupportedNetwork() {
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));

        Network cpu = zipLoadNetwork();
        LoadFlowResult cpuResult = runner.run(cpu, params(false));
        assertTrue(cpuResult.isFullyConverged());

        Network gpu = zipLoadNetwork();
        LoadFlowResult gpuResult = runner.run(gpu, params(true));
        assertTrue(gpuResult.isFullyConverged(), "GPU solver must fall back to CPU on an unsupported network");

        // 1e-12: the fallback runs the SAME CPU Newton-Raphson; only numerical-zero noise
        // (e.g. a 1e-18 slack angle) may differ from the direct CPU run.
        for (String id : new String[]{"NGEN", "NHV1", "NHV2", "NLOAD"}) {
            assertEquals(cpu.getBusBreakerView().getBus(id).getV(), gpu.getBusBreakerView().getBus(id).getV(), 1e-12,
                    "fallback must produce the CPU solution (V @ " + id + ")");
            assertEquals(cpu.getBusBreakerView().getBus(id).getAngle(), gpu.getBusBreakerView().getBus(id).getAngle(), 1e-12,
                    "fallback must produce the CPU solution (angle @ " + id + ")");
        }
    }
}
