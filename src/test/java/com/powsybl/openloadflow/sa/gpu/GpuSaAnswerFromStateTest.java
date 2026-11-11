/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.BusbarSectionContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Line;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.PhaseTapChanger;
import com.powsybl.iidm.network.TwoWindingsTransformer;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.network.PhaseControlFactory;
import com.powsybl.openloadflow.network.SlackBusSelectionMode;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The contingencies the device answers from their state array, without writing the state into OLF's objects (P1 of
 * {@code docs/device-final-results-plan.md}), decide exactly as the written-back path does. With the comparison on
 * ({@link GpuStateViewCheck#compareInTests}), each one is also written back and detected the old way, and its
 * realistic-voltage status and its violation list must be the same, field by field, doubles by their bits: a difference
 * throws inside the run. The shared helper also holds the results to OLF's CPU arm and monitors every voltage level,
 * which these contingencies serve by writing the bus voltages alone. Each fixture makes violations through one input
 * path of {@link GpuStateView}: closed branches with current, active and apparent power limits, branches opened at one
 * side by a busbar section outage, a phase shifter whose tap the device moves, bus voltage limits.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaAnswerFromStateTest extends AbstractGpuSaTest {

    GpuSaAnswerFromStateTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @BeforeEach
    void compareWithTheWrittenBackPath() {
        Assumptions.assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        GpuStateViewCheck.compareInTests(true);
    }

    @AfterEach
    void stopComparing() {
        GpuStateViewCheck.compareInTests(false);
    }

    private void assertAnsweredAsWrittenBack(Network network, List<Contingency> contingencies, LoadFlowParameters lfp) {
        long compared = GpuStateViewCheck.DECISIONS_COMPARED.sum();
        long mismatches = GpuStateViewCheck.DECISION_MISMATCHES.sum();
        assertGpuMatchesCpu(network, contingencies, 1e-12, lfp);
        assertTrue(GpuBatchedAcSecurityAnalysis.LAST_ANSWERED_FROM_STATE_COUNT.get() > 0,
                "no contingency was answered from its state: the comparison proves nothing");
        assertTrue(GpuStateViewCheck.DECISIONS_COMPARED.sum() > compared, "no violation list was compared");
        assertEquals(mismatches, GpuStateViewCheck.DECISION_MISMATCHES.sum());
    }

    @Test
    void currentActiveAndApparentPowerLimits() {
        Network network = EurostagTutorialExample1Factory.createWithFixedCurrentLimits();
        for (String id : List.of("NHV1_NHV2_1", "NHV1_NHV2_2")) {
            Line line = network.getLine(id);
            line.getOrCreateSelectedOperationalLimitsGroup1().newActivePowerLimits().setPermanentLimit(250).add();
            line.getOrCreateSelectedOperationalLimitsGroup2().newApparentPowerLimits().setPermanentLimit(250).add();
        }
        assertAnsweredAsWrittenBack(network, List.of(new Contingency("NHV1_NHV2_1", new BranchContingency("NHV1_NHV2_1")),
                new Contingency("NHV1_NHV2_2", new BranchContingency("NHV1_NHV2_2"))), new LoadFlowParameters());
    }

    @Test
    void busbarSectionOutageOpensBranchesAtOneSide() {
        Network network = GpuSaTestNetworks.middleSectionLossNetwork(false);
        network.getLineStream().forEach(line -> {
            line.getOrCreateSelectedOperationalLimitsGroup1().newCurrentLimits().setPermanentLimit(1).add();
            line.getOrCreateSelectedOperationalLimitsGroup2().newCurrentLimits().setPermanentLimit(1).add();
        });
        LoadFlowParameters lfp = new LoadFlowParameters();
        OpenLoadFlowParameters.create(lfp)                   // the slack its own tests pin (breakers mode: bus/breaker ids)
                .setSlackBusSelectionMode(SlackBusSelectionMode.NAME).setSlackBusesIds(List.of("vl0_b"));
        assertAnsweredAsWrittenBack(network, List.of(new Contingency("bus-mid", new BusbarSectionContingency("bbs1"))), lfp);
    }

    @Test
    void phaseShifterWhoseTapTheDeviceMoves() {
        Network network = PhaseControlFactory.createNetworkWithT2wt();
        TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().setRegulationMode(PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL)
                .setTargetDeadband(1).setRegulating(true).setTapPosition(1)
                .setRegulationTerminal(ps1.getTerminal1()).setRegulationValue(83);
        Line l1 = network.getLine("L1");
        network.newLine().setId("L1_BIS")
                .setVoltageLevel1(l1.getTerminal1().getVoltageLevel().getId()).setBus1(l1.getTerminal1().getBusBreakerView().getBus().getId())
                .setVoltageLevel2(l1.getTerminal2().getVoltageLevel().getId()).setBus2(l1.getTerminal2().getBusBreakerView().getBus().getId())
                .setR(l1.getR() * 2).setX(l1.getX() * 2)
                .setG1(l1.getG1()).setB1(l1.getB1()).setG2(l1.getG2()).setB2(l1.getB2()).add();
        network.getLineStream().forEach(line ->
                line.getOrCreateSelectedOperationalLimitsGroup1().newCurrentLimits().setPermanentLimit(50).add());
        ps1.getOrCreateSelectedOperationalLimitsGroup1().newCurrentLimits().setPermanentLimit(50).add();
        LoadFlowParameters lfp = new LoadFlowParameters().setPhaseShifterRegulationOn(true);
        OpenLoadFlowParameters.create(lfp).setPhaseShifterControlMode(OpenLoadFlowParameters.PhaseShifterControlMode.INCREMENTAL);
        assertAnsweredAsWrittenBack(network, List.of(new Contingency("L1_BIS", new BranchContingency("L1_BIS")),
                new Contingency("L1", new BranchContingency("L1"))), lfp);
    }

    /** Every bus outside a band of zero width, so nearly every bus reports its voltage: the reported value is
     *  {@code LfBus.getV()}'s round trip through kV, which differs from the state's value on about one bus in ten. */
    @Test
    void busVoltageLimits() {
        Network network = IeeeCdfNetworkFactory.create118();
        network.getVoltageLevelStream().forEach(vl ->
                vl.setLowVoltageLimit(vl.getNominalV()).setHighVoltageLimit(vl.getNominalV()));
        List<Contingency> contingencies = network.getLineStream().limit(3)
                .map(line -> new Contingency(line.getId(), new BranchContingency(line.getId()))).toList();
        assertAnsweredAsWrittenBack(network, contingencies, new LoadFlowParameters());
    }
}
