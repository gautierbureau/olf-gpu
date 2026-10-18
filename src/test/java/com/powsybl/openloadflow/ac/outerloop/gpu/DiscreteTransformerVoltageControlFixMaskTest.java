/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.TopologyKind;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.VoltageControlNetworkFactory;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@code fixTransformerVoltageControls} is a POST-CONTINGENCY check on the CPU — OLF cuts the
 * contingency's branches before it looks for a PV bus on the not-controlled side — so the device's
 * connectivity masks are a property of the SCENARIO, not of the run. This pins that down on the
 * builder itself rather than through a whole security analysis, where the same scenario can be routed
 * to the CPU for unrelated reasons and hide the difference.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class DiscreteTransformerVoltageControlFixMaskTest {

    /** BUS_2 is the not-controlled side of both transformers and reaches the only generator, at BUS_1,
     *  over LINE_12 alone. A 33 kV loop back to BUS_1 through a NON-regulating transformer keeps the
     *  pocket energized when that line opens, so the outage changes the component instead of darkening
     *  it. The base-topology mask sees the generator; the scenario's does not. */
    @Test
    void contingencyChangesTheNotControlledSideComponent() {
        Network iidm = VoltageControlNetworkFactory.createNetworkWith2T2wt();
        var t1 = iidm.getTwoWindingsTransformer("T2wT1");
        var t2 = iidm.getTwoWindingsTransformer("T2wT2");
        t1.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(3)
                .setRegulationTerminal(t1.getTerminal2()).setTargetV(34.0);
        t2.getRatioTapChanger().setTargetDeadband(0).setRegulating(true).setTapPosition(0)
                .setRegulationTerminal(t2.getTerminal2()).setTargetV(34.0);
        var s1 = iidm.getSubstation("SUBSTATION1");
        var vl9 = s1.newVoltageLevel().setId("VL_9").setNominalV(33.0).setTopologyKind(TopologyKind.BUS_BREAKER).add();
        vl9.getBusBreakerView().newBus().setId("BUS_9").add();
        s1.newTwoWindingsTransformer().setId("T_LOOP")
                .setRatedU1(132.0).setRatedU2(33.0).setR(17.0).setX(10.0)
                .setVoltageLevel1("VL_1").setBus1("BUS_1").setConnectableBus1("BUS_1")
                .setVoltageLevel2("VL_9").setBus2("BUS_9").setConnectableBus2("BUS_9").add();
        iidm.newLine().setId("LINE_93")
                .setVoltageLevel1("VL_9").setBus1("BUS_9").setConnectableBus1("BUS_9")
                .setVoltageLevel2("VL_3").setBus2("BUS_3").setConnectableBus2("BUS_3")
                .setR(1.0).setX(3.0).add();

        LoadFlowParameters lfp = new LoadFlowParameters().setTransformerVoltageControlOn(true);
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.create(lfp)
                .setTransformerVoltageControlMode(OpenLoadFlowParameters.TransformerVoltageControlMode.AFTER_GENERATOR_VOLTAGE_CONTROL);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);

        List<LfBus> pvBuses = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            if (bus.isGeneratorVoltageControlled()) {
                pvBuses.add(bus);
            }
        }
        assertEquals(1, pvBuses.size(), "only BUS_1 carries a generator voltage control here");
        List<Integer> controllerBranches = new ArrayList<>();
        for (LfBranch branch : network.getBranches()) {
            if (branch.isVoltageController()) {
                controllerBranches.add(branch.getNum());
            }
        }
        assertEquals(2, controllerBranches.size(), "both transformers control BUS_3");

        int words = 1;                                      // one PV bus
        var builder = new DiscreteTransformerVoltageControl.FixMaskBuilder(network,
                controllerBranches.stream().mapToInt(Integer::intValue).toArray(), pvBuses, words);
        int nm = controllerBranches.size();

        int[] baseMin = new int[nm * words];
        int[] baseMax = new int[nm * words];
        builder.fill(List.of(), baseMin, 0, baseMax, 0);
        for (int m = 0; m < nm; m++) {
            assertTrue(baseMin[m] != 0, "at base the not-controlled side reaches BUS_1 over LINE_12");
        }

        LfBranch line12 = network.getBranchById("LINE_12");
        int[] ctgMin = new int[nm * words];
        int[] ctgMax = new int[nm * words];
        builder.fill(List.of(line12), ctgMin, 0, ctgMax, 0);
        for (int m = 0; m < nm; m++) {
            assertEquals(0, ctgMin[m], "with LINE_12 out the not-controlled side holds no generator bus");
            assertTrue(ctgMax[m] != 0, "it still reaches BUS_1 through the transformers OLF may not cut");
        }
        assertFalse(java.util.Arrays.equals(baseMin, ctgMin),
                "a base-topology mask cannot answer this scenario — that is the whole point");
    }
}
