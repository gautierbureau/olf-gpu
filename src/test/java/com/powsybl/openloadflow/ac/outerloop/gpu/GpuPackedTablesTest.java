/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.PiModel;
import com.powsybl.openloadflow.network.PiModelArray;
import com.powsybl.openloadflow.network.impl.Networks;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * What the packers read from OLF's array tables is what they read from the network objects before, to the last bit:
 * this runs without a GPU.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuPackedTablesTest {

    @Test
    void tapRowsAreTheTapsModels() {
        Network network = EurostagTutorialExample1Factory.create();
        LfNetwork lfNetwork = Networks.load(network, new LfNetworkParameters()).getFirst();
        int checked = 0;
        for (LfBranch branch : lfNetwork.getBranches()) {
            if (!(branch.getPiModel() instanceof PiModelArray array)) {
                assertEquals(0, GpuTransformerVoltageControlOuterLoop.tapModels(branch.getPiModel()).length);
                continue;
            }
            double[][] rows = GpuTransformerVoltageControlOuterLoop.tapModels(array);
            int low = array.getTapPositionRange().getMinimum();
            assertEquals(array.getTapPositionRange().getMaximum() - low + 1, rows.length);
            for (int i = 0; i < rows.length; i++) {
                PiModel m = array.getModel(low + i);
                double[] expected = {m.getY(), m.getG1(), m.getG2(), m.getB1(), m.getB2(), m.getKsi(), m.getR1(), m.getA1()};
                for (int c = 0; c < expected.length; c++) {
                    assertEquals(Double.doubleToLongBits(expected[c]), Double.doubleToLongBits(rows[i][c]), branch.getId() + " tap " + i + " column " + c);
                }
            }
            checked++;
        }
        assertTrue(checked > 0);
    }

    @Test
    void sectionConductancesAreTheController() {
        List<Double> sectionsB = List.of(0.0, 0.1, 0.2);
        List<Double> sectionsG = List.of(0.0, 0.001, 0.002);
        LfShunt.Controller controller = new LfShunt.Controller("c", sectionsB, sectionsG, 0, 0);
        for (int pos = 0; pos < sectionsG.size(); pos++) {
            assertEquals(sectionsG.get(pos), GpuIncrementalShuntVoltageControlOuterLoop.shuntSectionG(controller, pos));
        }
        assertEquals(0, GpuIncrementalShuntVoltageControlOuterLoop.shuntSectionG(controller, 3));
    }

    /** The curve points the packers read from OLF are IIDM's, in IIDM's order (increasing P), and only for a curve. */
    @Test
    void curvePointsAreIidms() {
        Network network = EurostagTutorialExample1Factory.create();
        network.getGenerator("GEN").newReactiveCapabilityCurve()
                .beginPoint().setP(200).setMinQ(-80).setMaxQ(90).endPoint()
                .beginPoint().setP(0).setMinQ(-100).setMaxQ(100).endPoint()
                .beginPoint().setP(600).setMinQ(-40).setMaxQ(50).endPoint()
                .add();
        LfNetwork lfNetwork = Networks.load(network, new LfNetworkParameters()).getFirst();
        com.powsybl.openloadflow.network.ReactiveCapabilityCurvePoints points =
                lfNetwork.getGeneratorById("GEN").getReactiveCapabilityCurvePoints().orElseThrow();
        List<com.powsybl.iidm.network.ReactiveCapabilityCurve.Point> iidm = List.copyOf(
                network.getGenerator("GEN").getReactiveLimits(com.powsybl.iidm.network.ReactiveCapabilityCurve.class).getPoints());
        assertEquals(iidm.size(), points.getP().length);
        for (int k = 0; k < iidm.size(); k++) {
            assertEquals(iidm.get(k).getP(), points.getP()[k]);
            assertEquals(iidm.get(k).getMinQ(), points.getMinQ()[k]);
            assertEquals(iidm.get(k).getMaxQ(), points.getMaxQ()[k]);
        }
        assertArrayEquals(new double[] {0, 200, 600}, points.getP());
        assertTrue(ReactiveCapabilityCurves.hasCurveGenerators(lfNetwork));
        network.getGenerator("GEN").newMinMaxReactiveLimits().setMinQ(-10).setMaxQ(10).add();
        LfNetwork minMax = Networks.load(network, new LfNetworkParameters()).getFirst();
        assertTrue(minMax.getGeneratorById("GEN").getReactiveCapabilityCurvePoints().isEmpty());
        assertFalse(ReactiveCapabilityCurves.hasCurveGenerators(minMax));
    }

    /** The automation systems' tables, built from OLF's table, are the systems the network holds, in their order. */
    @Test
    void automationSystemTablesAreTheNetworks() {
        Network network = com.powsybl.openloadflow.network.AutomationSystemNetworkFactory.createWithBranchTripping();
        LfNetwork lfNetwork = Networks.load(network, new LfNetworkParameters().setSimulateAutomationSystems(true)).getFirst();
        var mapper = new com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper(lfNetwork);
        GpuAutomationSystemOuterLoop.Tables tables = GpuAutomationSystemOuterLoop.build(lfNetwork, mapper);
        assertNotNull(tables);
        List<com.powsybl.openloadflow.network.LfOverloadManagementSystem> systems = lfNetwork.getOverloadManagementSystems();
        assertFalse(systems.isEmpty());
        assertEquals(systems.size(), tables.monitoredElement().length);
        int t = 0;
        for (int m = 0; m < systems.size(); m++) {
            var system = systems.get(m);
            assertEquals(mapper.getElementIndex(system.getMonitoredBranch()), tables.monitoredElement()[m]);
            assertEquals(system.getMonitoredSide() == com.powsybl.iidm.network.TwoSides.ONE
                    ? com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides.ONE
                    : com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides.TWO, tables.monitoredSide()[m]);
            assertEquals(t, tables.trippingStart()[m]);
            for (var tripping : system.getBranchTrippingList()) {
                assertEquals(mapper.getElementIndex(tripping.branchToOperate()), tables.operatedMaskSlot()[t]);
                assertEquals(tripping.branchOpen() ? 1 : 0, tables.openAction()[t]);
                assertEquals(tripping.threshold(), tables.threshold()[t]);
                assertSame(tripping.branchToOperate(), tables.operatedBranches().get(t));
                t++;
            }
        }
        assertEquals(t, tables.trippingStart()[systems.size()]);
    }

    /** The AC emulation rows packed from OLF's table are each line's parameters, in the network's order. */
    @Test
    void hvdcRowsAreTheLines() {
        Network network = com.powsybl.openloadflow.network.HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        com.powsybl.loadflow.LoadFlowParameters parameters = new com.powsybl.loadflow.LoadFlowParameters().setHvdcAcEmulation(true);
        var acParameters = com.powsybl.openloadflow.OpenLoadFlowParameters.createAcParameters(network, parameters,
                com.powsybl.openloadflow.OpenLoadFlowParameters.create(parameters), new com.powsybl.math.matrix.SparseMatrixFactory(),
                new com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        LfNetwork lfNetwork = Networks.load(network, acParameters.getNetworkParameters()).getFirst();
        try (var context = new com.powsybl.openloadflow.ac.AcLoadFlowContext(lfNetwork, acParameters)) {
            new com.powsybl.openloadflow.ac.AcloadFlowEngine(context).run();
            var data = com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor.extract(lfNetwork, context.getEquationSystem());
            double[] packed = data.hvdcAcEmulation().parameters();
            List<com.powsybl.openloadflow.network.LfHvdc> lines = lfNetwork.getHvdcs().stream()
                    .filter(h -> h.getBus1() != null && h.getBus2() != null && h.isAcEmulation()).toList();
            assertFalse(lines.isEmpty());
            assertEquals(5 * lines.size(), packed.length);
            for (int h = 0; h < lines.size(); h++) {
                var line = lines.get(h);
                double[] expected = {line.getAcEmulationControl().getP0(), line.getAcEmulationControl().getDroop() * 180 / Math.PI,
                    line.getConverterStation1().getLossFactor() / 100, line.getConverterStation2().getLossFactor() / 100, line.getR()};
                for (int c = 0; c < 5; c++) {
                    assertEquals(Double.doubleToLongBits(expected[c]), Double.doubleToLongBits(packed[5 * h + c]), "line " + h + " column " + c);
                }
            }
        }
    }
}
