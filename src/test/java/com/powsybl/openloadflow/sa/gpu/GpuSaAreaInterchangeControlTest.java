/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.GeneratorContingency;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
import com.powsybl.openloadflow.network.MultiAreaNetworkFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * AreaInterchangeControl in the batched security analysis: the device's AREA_INTERCHANGE_CONTROL phase balances each
 * post-contingency area on its own generators, from the initial targets as OLF does. A branch outage, a boundary's
 * included, batches; a contingency that loses an injection (OLF first spreads the loss over the whole network) goes to
 * the CPU. Every result must match plain OLF: statuses, violations, distributed active power at 1e-9 MW and bus voltages.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaAreaInterchangeControlTest extends AbstractGpuSaTest {

    GpuSaAreaInterchangeControlTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.isAvailable(), "GPU lib required");
    }

    /** OLF's AreaInterchangeControlTest settings, with the interchange band at {@code areaInterchangePMaxMismatch} MW. */
    private static LoadFlowParameters parameters(double areaInterchangePMaxMismatch) {
        LoadFlowParameters parameters = new LoadFlowParameters();
        OpenLoadFlowParameters.create(parameters)
                .setAreaInterchangeControl(true)
                .setSlackBusPMaxMismatch(1e-3)
                .setAreaInterchangePMaxMismatch(areaInterchangePMaxMismatch);
        return parameters;
    }

    /**
     * OLF's two areas with an X node, every line doubled so that no single outage leaves an island: a1 (b1, b2) and a2 (b3)
     * exchange through the X node bx1, which is in no area, over two boundaries on each side. Every line has 2 ohm of
     * resistance: an outage then moves the areas' losses, so the post-contingency check finds interchange mismatches of a
     * few kW to balance (OLF distributes -0.004 to 0.07 MW per contingency with a 1 kW band), where OLF's lossless lines
     * leave every interchange on its target and the loop with nothing to do.
     */
    private static Network meshedTwoAreas() {
        Network network = MultiAreaNetworkFactory.createTwoAreasWithXNode();
        network.getArea("a1").setInterchangeTarget(-40);
        network.getArea("a2").setInterchangeTarget(40);
        network.newLine().setId("l12b").setBus1("b1").setBus2("b2").setR(0).setX(1).add();
        network.newLine().setId("l23_A1b").setBus1("b2").setBus2("bx1").setR(0).setX(2).add();
        network.newLine().setId("l23_A2b").setBus1("bx1").setBus2("b3").setR(0).setX(2).add();
        network.getArea("a1").newAreaBoundary().setTerminal(network.getLine("l23_A1b").getTerminal2()).setAc(true).add();
        network.getArea("a2").newAreaBoundary().setTerminal(network.getLine("l23_A2b").getTerminal1()).setAc(true).add();
        network.getLineStream().forEach(line -> line.setR(2));
        return network;
    }

    /** Every line, a boundary or not, in turn: each batches. */
    @Test
    void branchOutagesOfTwoMeshedAreasMatchCpu() {
        Network network = meshedTwoAreas();
        List<Contingency> contingencies = network.getLineStream()
                .map(line -> new Contingency(line.getId(), new BranchContingency(line.getId())))
                .toList();
        assertGpuMatchesCpu(network, contingencies, 1e-12, parameters(1e-3));
        assertEquals(contingencies.size(), GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(),
                "every branch outage must solve on the device: refused " + GpuBatchScopeCensus.planRefusedIds());
    }

    /** OLF's ten areas on a ring of tie lines: each tie line outage keeps the ring connected, and the two areas it joined
     *  read their interchange on their remaining tie line half, at its boundary point. OLF reaches its iteration limit on
     *  key_5 and key_8; the device reports the same from its own solve (the status check is per contingency). */
    @Test
    void tieLineOutagesOfTenAreasMatchCpu() {
        Network network = MultiAreaNetworkFactory.createTenAreas();
        LoadFlowParameters parameters = parameters(1e-1);
        OpenLoadFlowParameters.get(parameters).setLowImpedanceBranchMode(OpenLoadFlowParameters.LowImpedanceBranchMode.REPLACE_BY_MIN_IMPEDANCE_LINE);
        List<Contingency> contingencies = IntStream.range(0, 10)
                .mapToObj(i -> new Contingency("key_" + i, new BranchContingency("key_" + i)))
                .toList();
        assertGpuMatchesCpu(network, contingencies, 1e-12, parameters);
        assertEquals(java.util.Set.of(), GpuBatchScopeCensus.planRefusedIds(), "every tie line outage on the device");
        assertEquals(8, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "converged on the device: all but key_5 and key_8");
    }

    /** A generator outage loses active power, which OLF spreads over the whole network before balancing the areas: that
     *  contingency goes to the CPU, the branch outages beside it stay on the device. */
    @Test
    void generatorOutageGoesToTheCpuBesideBatchedBranchOutages() {
        Network network = meshedTwoAreas();
        List<Contingency> contingencies = List.of(
                new Contingency("l12", new BranchContingency("l12")),
                new Contingency("gen3", new GeneratorContingency("gen3")),
                new Contingency("l23_A2", new BranchContingency("l23_A2")));
        assertGpuMatchesCpu(network, contingencies, 1e-12, parameters(1e-3));
        assertEquals(2, GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get(), "the two branch outages on the device");
        assertEquals(java.util.Set.of("gen3"), GpuBatchScopeCensus.planRefusedIds(), "the generator outage on the CPU");
    }
}
