/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * DIAGNOSTIC PROBE (kept as the reproducer of an OPEN parity investigation, non-asserting):
 * CPU vs GPU-sequential Newton trajectories on the outaged 4-bus AC-emulation hvdc network
 * (createHvdcLinkedByTwoLinesAndSwitch(VSC), realistic impedances). Observed 2026-09-04: the FIRST
 * Newton step from flat differs materially — GPU b4 = 399.250000 vs CPU b4 = 399.177860 (the root)
 * — although both arms converge to the BIT-IDENTICAL root by iteration 3, and the hv formulas,
 * inputs (p0/k/lossFactors/r) and Jacobian destinations were all verified identical against OLF.
 * The un-contracted first-step difference makes default-criterion stops land up to ~4e-4 kV apart
 * on scenarios whose CPU stop is shallow (a live-hvdc branch outage). Next step: elementwise J/F
 * dump at the flat start on this 10-variable system.
 */
class GpuHvdcTrajectoryProbe {

    static {
        com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths.init();
    }

    private static Network net() {
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcLinkedByTwoLinesAndSwitch(
                com.powsybl.iidm.network.HvdcConverterStation.HvdcType.VSC);
        network.getHvdcLine("hvdc23").setMaxP(500);
        network.getLineStream().forEach(l -> l.setR(1).setX(10));
        network.getHvdcLine("hvdc23").getExtension(com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControl.class)
                .setDroop(2);
        network.getLine("l14").getTerminal1().disconnect();     // stand-in for the outage scenario
        network.getLine("l14").getTerminal2().disconnect();
        return network;
    }

    @Test
    void trajectories() {
        assumeTrue(com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        for (int maxIter = 1; maxIter <= 4; maxIter++) {
            Network cpu = net();
            LoadFlowParameters p = new LoadFlowParameters().setDistributedSlack(false).setUseReactiveLimits(false)
                    .setHvdcAcEmulation(true);
            OpenLoadFlowParameters.create(p).setMaxNewtonRaphsonIterations(maxIter).setAlwaysUpdateNetwork(true);
            new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory())).run(cpu, p);
            System.err.printf("TRAJ CPU it<=%d b4=%.12f b3=%.12f angle3=%.12f%n", maxIter,
                    cpu.getBusBreakerView().getBus("b4").getV(), cpu.getBusBreakerView().getBus("b3").getV(),
                    cpu.getBusBreakerView().getBus("b3").getAngle());

            Network gpu = net();
            LoadFlowParameters p2 = new LoadFlowParameters().setDistributedSlack(false).setUseReactiveLimits(false)
                    .setHvdcAcEmulation(true);
            OpenLoadFlowParameters.create(p2).setMaxNewtonRaphsonIterations(maxIter).setAlwaysUpdateNetwork(true)
                    .setAcSolverType(com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory.NAME);
            new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory())).run(gpu, p2);
            System.err.printf("TRAJ GPU it<=%d b4=%.12f b3=%.12f angle3=%.12f%n", maxIter,
                    gpu.getBusBreakerView().getBus("b4").getV(), gpu.getBusBreakerView().getBus("b3").getV(),
                    gpu.getBusBreakerView().getBus("b3").getAngle());
        }
    }
}
