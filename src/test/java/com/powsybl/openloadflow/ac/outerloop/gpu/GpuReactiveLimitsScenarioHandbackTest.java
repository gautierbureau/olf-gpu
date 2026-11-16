/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuQLimitType;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link GpuReactiveLimitsOuterLoop#applyScenarioToNetwork}: a batched scenario's PV/PQ decisions given to OLF's buses,
 * and the equations they switch.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuReactiveLimitsScenarioHandbackTest {

    private LfNetwork lfNetwork;
    private AcLoadFlowContext context;
    private GpuAcData data;
    private LfBus bus2;
    private int entry2;

    /** IEEE 14 solved: every generator holds its voltage. */
    @BeforeEach
    void setUp() {
        Network network = IeeeCdfNetworkFactory.create14();
        LoadFlowParameters loadFlowParameters = new LoadFlowParameters();
        AcLoadFlowParameters acParameters = OpenLoadFlowParameters.createAcParameters(network, loadFlowParameters,
                OpenLoadFlowParameters.create(loadFlowParameters), new SparseMatrixFactory(),
                new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        lfNetwork = LfNetwork.load(network, new LfNetworkLoaderImpl(), acParameters.getNetworkParameters(), ReportNode.NO_OP).get(0);
        context = new AcLoadFlowContext(lfNetwork, acParameters);
        new AcloadFlowEngine(context).run();
        data = GpuAcDataExtractor.extract(lfNetwork, context.getEquationSystem());
        bus2 = lfNetwork.getBusById("VL2_0");
        entry2 = entryOf(bus2);
    }

    @AfterEach
    void tearDown() {
        context.close();
    }

    private int entryOf(LfBus bus) {
        int[] busNum = data.reactiveLimits().busNum();
        for (int i = 0; i < busNum.length; i++) {
            if (busNum[i] == bus.getNum()) {
                return i;
            }
        }
        throw new IllegalStateException(bus.getId() + " is no reactive-limit entry");
    }

    private int[] allPv() {
        return new int[data.reactiveLimits().busNum().length];
    }

    private boolean targetVActive(LfBus bus) {
        return context.getEquationSystem().getEquation(bus.getNum(), AcEquationType.BUS_TARGET_V).orElseThrow().isActive();
    }

    @Test
    void aSwitchToPqFreezesTheTargetTheScenarioFroze() {
        int[] finalState = allPv();
        finalState[entry2] = GpuQLimitType.MAX_Q;
        double[] frozenRowTargetQ = new double[finalState.length];
        java.util.Arrays.fill(frozenRowTargetQ, Double.NaN);
        frozenRowTargetQ[entry2] = 0.3;
        assertTrue(targetVActive(bus2));
        assertTrue(GpuReactiveLimitsOuterLoop.applyScenarioToNetwork(lfNetwork, data, finalState, frozenRowTargetQ));
        assertFalse(bus2.isGeneratorVoltageControlEnabled());
        assertFalse(targetVActive(bus2), "the bus's voltage equation leaves the system");
        assertEquals(Optional.of(LfBus.QLimitType.MAX_Q), bus2.getQLimitType());
        assertEquals(0.3 + data.reactiveLimits().loadQ()[entry2], bus2.getGenerationTargetQ(), 1e-12);
    }

    @Test
    void withoutAnExportedTargetTheSwitchFreezesThePackedLimit() {
        int[] finalState = allPv();
        finalState[entry2] = GpuQLimitType.MIN_Q;
        assertTrue(GpuReactiveLimitsOuterLoop.applyScenarioToNetwork(lfNetwork, data, finalState, null));
        assertEquals(Optional.of(LfBus.QLimitType.MIN_Q), bus2.getQLimitType());
        assertEquals(data.reactiveLimits().minQ()[entry2], bus2.getGenerationTargetQ(), 1e-12);
    }

    @Test
    void aSwitchBackGivesTheBusItsVoltageControlAgain() {
        bus2.freezeGenerationTargetQAndDisableGeneratorVoltageControl(0.5);
        bus2.setQLimitType(LfBus.QLimitType.MAX_Q);
        assertFalse(targetVActive(bus2));
        assertTrue(GpuReactiveLimitsOuterLoop.applyScenarioToNetwork(lfNetwork, data, allPv(), null));
        assertTrue(bus2.isGeneratorVoltageControlEnabled());
        assertTrue(targetVActive(bus2));
        assertEquals(Optional.empty(), bus2.getQLimitType());
    }

    /** A bus PQ without a limit (robust mode's realistic-voltage freeze here) is no switch back: left as it is. */
    @Test
    void aRealisticVoltageFreezeIsNeverUndone() {
        bus2.freezeGenerationTargetQAndDisableGeneratorVoltageControl(0.5);
        bus2.setQLimitType(LfBus.QLimitType.MAX_REALISTIC_V);
        assertTrue(GpuReactiveLimitsOuterLoop.applyScenarioToNetwork(lfNetwork, data, allPv(), null));
        assertFalse(bus2.isGeneratorVoltageControlEnabled());
        assertEquals(Optional.of(LfBus.QLimitType.MAX_REALISTIC_V), bus2.getQLimitType());
    }

    /** All or nothing: an entry frozen by the generator voltage control manager refuses the scenario before any switch. */
    @Test
    void aFrozenEntryLeavesTheNetworkAsItWas() {
        LfBus bus3 = lfNetwork.getBusById("VL3_0");
        int[] finalState = allPv();
        finalState[entry2] = GpuQLimitType.MAX_Q;
        finalState[entryOf(bus3)] = GpuQLimitType.FROZEN;
        assertFalse(GpuReactiveLimitsOuterLoop.applyScenarioToNetwork(lfNetwork, data, finalState, null));
        assertTrue(bus2.isGeneratorVoltageControlEnabled());
        assertTrue(targetVActive(bus2));
        assertEquals(Optional.empty(), bus2.getQLimitType());
    }
}
