/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.StaticVarCompensator;
import com.powsybl.iidm.network.extensions.StandbyAutomatonAdder;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.HvdcNetworkFactory;
import com.powsybl.openloadflow.network.LfContingency;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfTopoConfig;
import com.powsybl.openloadflow.network.VoltageControlNetworkFactory;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import com.powsybl.openloadflow.network.impl.PropagatedContingency;
import com.powsybl.openloadflow.network.impl.PropagatedContingencyCreationParameters;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;

/**
 * {@link GpuPostContingencyBatch}: which outer loops a batch hands back, and a run without the device.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuPostContingencyBatchTest {

    private record Loaded(LfNetwork lfNetwork, AcLoadFlowParameters acParameters, LoadFlowParameters loadFlowParameters) {
    }

    private static Loaded load(Network network, Consumer<LoadFlowParameters> configure) {
        LoadFlowParameters loadFlowParameters = new LoadFlowParameters();
        OpenLoadFlowParameters parametersExt = OpenLoadFlowParameters.create(loadFlowParameters);
        configure.accept(loadFlowParameters);
        AcLoadFlowParameters acParameters = OpenLoadFlowParameters.createAcParameters(network, loadFlowParameters, parametersExt,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        LfNetwork lfNetwork = LfNetwork.load(network, new LfNetworkLoaderImpl(), acParameters.getNetworkParameters(), ReportNode.NO_OP).get(0);
        return new Loaded(lfNetwork, acParameters, loadFlowParameters);
    }

    private static List<String> loopNames(AcLoadFlowParameters acParameters) {
        return acParameters.getOuterLoops().stream().map(AcOuterLoop::getName).toList();
    }

    /** OLF's default loops: distributed slack and reactive limits handed back, voltage monitoring and the HVDC AC emulation
     *  limits with nothing to decide on a network without a monitoring SVC or an HVDC. */
    @Test
    void theDefaultLoopsAreHandedBack() {
        Loaded loaded = load(IeeeCdfNetworkFactory.create14(), p -> { });
        assertEquals(List.of("DistributedSlack", "AcHvdcAcEmulationLimits", "VoltageMonitoring", "ReactiveLimits"),
                loopNames(loaded.acParameters()));
        assertEquals(List.of(), GpuPostContingencyBatch.loopsNotHandedBack(loaded.lfNetwork(), loaded.acParameters()));
    }

    @Test
    void anHvdcInAcEmulationKeepsItsLoopFromTheBatch() {
        Loaded loaded = load(HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork(), p -> { });
        assertEquals(List.of("AcHvdcAcEmulationLimits"), GpuPostContingencyBatch.loopsNotHandedBack(loaded.lfNetwork(), loaded.acParameters()));
    }

    @Test
    void aMonitoringSvcKeepsVoltageMonitoringFromTheBatch() {
        Network network = VoltageControlNetworkFactory.createWithStaticVarCompensator();
        StaticVarCompensator svc = network.getStaticVarCompensator("svc1");
        svc.setVoltageSetpoint(385).setRegulationMode(StaticVarCompensator.RegulationMode.VOLTAGE).setRegulating(true);
        svc.newExtension(StandbyAutomatonAdder.class)
                .withHighVoltageThreshold(400).withLowVoltageThreshold(388)
                .withLowVoltageSetpoint(389).withHighVoltageSetpoint(395)
                .withB0(0f).withStandbyStatus(true).add();
        Loaded loaded = load(network, p -> { });
        assertEquals(List.of("VoltageMonitoring"), GpuPostContingencyBatch.loopsNotHandedBack(loaded.lfNetwork(), loaded.acParameters()));
    }

    @Test
    void transformerVoltageControlIsNotHandedBack() {
        Loaded loaded = load(IeeeCdfNetworkFactory.create14(), p -> p.setTransformerVoltageControlOn(true));
        List<String> notHandedBack = GpuPostContingencyBatch.loopsNotHandedBack(loaded.lfNetwork(), loaded.acParameters());
        assertEquals(1, notHandedBack.size(), "only the transformer voltage control: " + notHandedBack);
        assertTrue(notHandedBack.get(0).contains("TransformerVoltageControl"), notHandedBack.get(0));
    }

    /** Without the device nothing is batched, and every contingency is left to its own load flow. */
    @Test
    void withoutTheDeviceNothingIsBatched() {
        assumeFalse(GpuAcNewtonSolver.isAvailable(), "the GPU library is available");
        Network network = IeeeCdfNetworkFactory.create14();
        Loaded loaded = load(network, p -> { });
        List<PropagatedContingency> contingencies = PropagatedContingency.createList(network,
                List.of(new Contingency("L1-2-1", new BranchContingency("L1-2-1"))), new LfTopoConfig(),
                new PropagatedContingencyCreationParameters());
        try (AcLoadFlowContext context = new AcLoadFlowContext(loaded.lfNetwork(), loaded.acParameters())) {
            new AcloadFlowEngine(context).run();
            try (GpuPostContingencyBatch batch = GpuPostContingencyBatch.solve(network, loaded.lfNetwork(), context, contingencies,
                    loaded.loadFlowParameters(), false, ReportNode.NO_OP)) {
                assertTrue(batch.notBatchedReason().orElseThrow().contains("did not run"), batch.notBatchedReason().orElseThrow());
                assertEquals(0, batch.size());
                LfContingency lfContingency = contingencies.get(0).toLfContingency(loaded.lfNetwork()).orElseThrow();
                assertFalse(batch.bringToConvergedState(context, lfContingency));
            }
        }
    }
}
