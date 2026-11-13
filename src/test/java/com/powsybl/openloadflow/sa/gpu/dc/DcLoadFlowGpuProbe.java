/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu.dc;

import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Generator;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.gpu.GpuLoadFlowProvider;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The full DC load flow on rte6515 (RTE parameters, dc=true): the stock provider against the GPU
 * provider — wall (cold, second run warm), every branch flow (1e-8 MW), every angle, every
 * generator dispatch bit for bit. A single DC solve is where the device CANNOT win (cuDSS analysis
 * alone outweighs KLU's whole run): this probe states that number honestly; the pay-off is the
 * repeated solve (security analysis, time series) on the same assembled context.
 */
class DcLoadFlowGpuProbe {

    private static final Path CASE_FILE = Path.of("rte6515_full.xiidm.gz");
    private static final Path PARAMS_FILE = Path.of("rte-loadflow-default-parameters.json");

    @Test
    void stockVersusGpu() {
        assumeTrue(Files.exists(CASE_FILE), "rte6515_full.xiidm.gz not found");
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        LoadFlowParameters parameters = JsonLoadFlowParameters.read(PARAMS_FILE).setDc(true);
        Network cpu = Network.read(CASE_FILE);
        Network gpu = Network.read(CASE_FILE);
        LoadFlow.Runner stock = new LoadFlow.Runner(new OpenLoadFlowProvider(new SparseMatrixFactory()));
        LoadFlow.Runner device = new LoadFlow.Runner(new GpuLoadFlowProvider(new SparseMatrixFactory()));
        for (int pass = 0; pass < 2; pass++) {
            long t0 = System.nanoTime();
            LoadFlowResult cpuResult = stock.run(cpu, parameters);
            long t1 = System.nanoTime();
            LoadFlowResult gpuResult = device.run(gpu, parameters);
            long t2 = System.nanoTime();
            assertTrue(GpuLoadFlowProvider.LAST_GPU_COMPONENT_COUNT.get() >= 1, "the device solved a component");
            double maxP = 0;
            int branches = 0;
            for (Branch<?> b : cpu.getBranches()) {
                Branch<?> o = gpu.getBranch(b.getId());
                for (int side = 1; side <= 2; side++) {
                    double a = side == 1 ? b.getTerminal1().getP() : b.getTerminal2().getP();
                    double c = side == 1 ? o.getTerminal1().getP() : o.getTerminal2().getP();
                    if (!Double.isNaN(a) || !Double.isNaN(c)) {
                        maxP = Math.max(maxP, Math.abs(a - c));
                        branches++;
                    }
                }
            }
            double maxAngle = 0;
            for (Bus b : cpu.getBusView().getBuses()) {
                Bus o = gpu.getBusView().getBus(b.getId());
                if (o != null && !Double.isNaN(b.getAngle()) && !Double.isNaN(o.getAngle())) {
                    maxAngle = Math.max(maxAngle, Math.abs(b.getAngle() - o.getAngle()));
                }
            }
            int genDiff = 0;
            for (Generator g : cpu.getGenerators()) {
                double a = g.getTerminal().getP();
                double c = gpu.getGenerator(g.getId()).getTerminal().getP();
                if (Double.doubleToLongBits(a) != Double.doubleToLongBits(c) && !(Double.isNaN(a) && Double.isNaN(c))) {
                    genDiff++;
                }
            }
            System.out.printf("DC-LF rte6515 %s | CPU %d ms (%s, %d components) | GPU %d ms (%s, %d on device)"
                    + " | max |dP| %.3e MW over %d terminals | max |dAngle| %.3e deg | generators differing %d%n",
                    pass == 0 ? "cold" : "warm", (t1 - t0) / 1_000_000, cpuResult.getComponentResults().getFirst().getStatus(),
                    cpuResult.getComponentResults().size(), (t2 - t1) / 1_000_000, gpuResult.getComponentResults().getFirst().getStatus(),
                    GpuLoadFlowProvider.LAST_GPU_COMPONENT_COUNT.get(), maxP, branches, maxAngle, genDiff);
            assertEquals(cpuResult.isFullyConverged(), gpuResult.isFullyConverged());
            assertTrue(maxP < 1e-8, "flows differ by " + maxP + " MW");
            assertEquals(0, genDiff, "generator dispatch differs");
        }
    }
}
