/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowResult;
import com.powsybl.openloadflow.ac.solver.AcSolverStatus;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A network that once needed the GPU load flow's DEVICE-BROKEN safety net, and no longer does: a
 * remote gen-VC group, a zero-impedance branch at member b2 feeding a LOADED far bus b8, and a Q-heavy
 * area that PINS g2 during the run. The first device solve used to return an exploded state (phi ~1e22
 * at b8) while the same system solved fine on the CPU, and the safety net reran it on the CPU — never
 * wrong, only un-accelerated, with the device-side cause recorded as open.
 *
 * <p>The cause was the pin: OLF renormalizes a group's reactive keys ({@code updateReactiveKeys}) and
 * fires NO equation-system event, so the cached extraction kept its PRE-pin static {@code qPct} and the
 * device distributed reactive power with stale weights. {@code GpuNewtonRaphson} now watches the pinned
 * set ({@code GpuAcActivationExtractor.pinnedRemoteMembers}) and re-extracts when it changes, so this
 * run stays on the device from end to end. The test therefore asserts the OPPOSITE of what it once did —
 * no fallback, counted by GpuNewtonRaphson.CPU_FALLBACK_COUNT — and keeps the CPU comparison that is the real
 * contract.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSeqBaseFallbackTest {

    private static Network zeroImpedanceMemberNetwork() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        var s = network.getSubstation("s");
        var vl5 = s.newVoltageLevel().setId("vl5").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl5.getBusBreakerView().newBus().setId("b5").add();
        vl5.newLoad().setId("l5").setBus("b5").setConnectableBus("b5").setP0(150).setQ0(300).add();
        for (String id : new String[] {"l45", "l45b"}) {
            network.newLine().setId(id)
                    .setVoltageLevel1("vl4").setBus1("b4").setConnectableBus1("b4")
                    .setVoltageLevel2("vl5").setBus2("b5").setConnectableBus2("b5")
                    .setR(1.5).setX(40.0).setG1(0).setB1(0).setG2(0).setB2(0).add();
        }
        network.getGenerator("g2").newMinMaxReactiveLimits().setMinQ(-1000).setMaxQ(186).add();
        var vl8 = s.newVoltageLevel().setId("vl8").setNominalV(400)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl8.getBusBreakerView().newBus().setId("b8").add();
        vl8.newLoad().setId("l8").setBus("b8").setConnectableBus("b8").setP0(5).setQ0(10).add();
        network.newLine().setId("lzi")
                .setVoltageLevel1("vl2").setBus1("b2").setConnectableBus1("b2")
                .setVoltageLevel2("vl8").setBus2("b8").setConnectableBus2("b8")
                .setR(0.0).setX(0.0).setG1(0).setB1(0).setG2(0).setB2(0).add();
        return network;
    }

    private static double[] run(Network network, boolean gpu) {
        LoadFlowParameters lfp = new LoadFlowParameters().setDistributedSlack(true).setUseReactiveLimits(true);
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.create(lfp).setVoltageRemoteControl(true);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(network, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        if (gpu) {
            acp.setSolverFactory(new GpuNewtonRaphsonFactory(), lfp);
        }
        LfNetwork lf = LfNetwork.load(network, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);
        try (AcLoadFlowContext context = new AcLoadFlowContext(lf, acp)) {
            AcLoadFlowResult r = acp.getSolverFactory().createLoadFlowEngine(context).run();
            assertEquals(AcSolverStatus.CONVERGED, r.getSolverStatus(),
                    (gpu ? "GPU" : "CPU") + " engine run must converge");
            return lf.getBuses().stream().mapToDouble(b -> b.getV()).toArray();
        }
    }

    @org.junit.jupiter.api.Test
    void pinnedRemoteMemberRunStaysOnTheDevice() {
        org.junit.jupiter.api.Assumptions.assumeTrue(
                com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        double[] cpu = run(zeroImpedanceMemberNetwork(), false);
        long fallbacksBefore = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        double[] gpu = run(zeroImpedanceMemberNetwork(), true);
        assertEquals(fallbacksBefore, GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the pinned remote gen-VC run must need no CPU fallback: its re-extracted qPct is OLF's");
        double worst = 0;
        for (int i = 0; i < cpu.length; i++) {
            worst = Math.max(worst, Math.abs(cpu[i] - gpu[i]));
        }
        assertTrue(worst < 1e-10, "the on-device run must match the CPU engine run (worst |dV| = " + worst + " pu)");
    }
}
