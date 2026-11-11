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
import com.powsybl.contingency.ContingencyContext;
import com.powsybl.contingency.violations.LimitViolation;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Line;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.PhaseTapChanger;
import com.powsybl.iidm.network.PhaseTapChangerAdder;
import com.powsybl.iidm.network.TwoWindingsTransformer;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.PhaseControlFactory;
import com.powsybl.openloadflow.network.SlackBusSelectionMode;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.monitor.StateMonitor;
import com.powsybl.security.results.PostContingencyResult;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

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
 * <p>Without state monitors the same contingencies are answered without applying them to the network (P2), their
 * topology read from the contingency: with the comparison on, each one is answered the usual way too and the two results
 * must be the same, field by field; without it, the answer is held to OLF's CPU arm.
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
        long screens = GpuStateViewCheck.SCREENS_COMPARED.sum();
        assertGpuMatchesCpu(network, contingencies, 1e-12, lfp);
        assertTrue(GpuBatchedAcSecurityAnalysis.LAST_ANSWERED_FROM_STATE_COUNT.get() > 0,
                "no contingency was answered from its state: the comparison proves nothing");
        assertTrue(GpuStateViewCheck.DECISIONS_COMPARED.sum() > compared, "no violation list was compared");
        assertEquals(mismatches, GpuStateViewCheck.DECISION_MISMATCHES.sum());
        assertTrue(GpuStateViewCheck.SCREENS_COMPARED.sum() > screens, "no screened detection was compared with a full scan");
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

    /** PS1 regulating its active power over 21 phase steps, each with its own impedance: both outages make the device
     *  move its tap, and the moved tap's row (another y and ksi) is what the view and the screen must read. */
    @Test
    void phaseShifterWhoseTapTheDeviceMoves() {
        Network network = PhaseControlFactory.createNetworkWithT2wt();
        TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        PhaseTapChangerAdder tapChanger = ps1.newPhaseTapChanger().setLowTapPosition(-10).setTapPosition(0)
                .setRegulationMode(PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL)
                .setTargetDeadband(1).setRegulating(true).setRegulationTerminal(ps1.getTerminal1()).setRegulationValue(83);
        for (int t = -10; t <= 10; t++) {
            tapChanger.beginStep().setAlpha(t).setRho(1.0).setR(0.3 * (t + 11)).setX(3.0 * (t + 11)).setG(0).setB(0).endStep();
        }
        tapChanger.add();
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
        long rows = GpuStateViewCheck.SCREEN_ROWS_COMPARED.sum();
        GpuDeviceTaps.CHANGES.set(0);
        List<Contingency> contingencies = List.of(new Contingency("L1_BIS", new BranchContingency("L1_BIS")),
                new Contingency("L1", new BranchContingency("L1")));
        assertAnsweredAsWrittenBack(network, contingencies, lfp);
        assertTrue(GpuDeviceTaps.CHANGES.get() > 0, "the device moved no tap: the fixture proves nothing about moved taps");
        assertAnsweredFromRows(network, contingencies, lfp, false);  // the moved taps' branches among the rows (P4)
        // the screen read the moved tap's row, the one the device taps leave on the pi model (checked while comparing)
        assertTrue(GpuStateViewCheck.SCREEN_ROWS_COMPARED.sum() > rows, "no retapped branch's screen row was compared");
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

    /** The contingencies answered without the network, compared with the usual path, then on their own against OLF. */
    private void assertAnsweredWithoutNetwork(Network network, List<Contingency> contingencies, LoadFlowParameters lfp) {
        assertAnsweredFromRows(network, contingencies, lfp, true);
    }

    /**
     * Without state monitors, every contingency is answered from the state rows gathered on the device alone (P4): with
     * the comparison on, each answer is also given from its full state, then on their own against OLF. When
     * {@code withoutNetwork}, they must also be answered without the network (P2), compared with the usual path.
     */
    private void assertAnsweredFromRows(Network network, List<Contingency> contingencies, LoadFlowParameters lfp,
                                        boolean withoutNetwork) {
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, List.of(), sap);
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(commonTestConfig.matrixFactory(),
                new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try {
            long compared = GpuStateViewCheck.RESULTS_COMPARED.sum();
            long decisions = GpuStateViewCheck.DECISIONS_COMPARED.sum();
            runSecurityAnalysis(network, contingencies, List.of(), sap);
            assertTrue(GpuStateViewCheck.DECISIONS_COMPARED.sum() > decisions, "no answer from the rows was compared");
            if (withoutNetwork) {
                assertTrue(GpuStateViewCheck.RESULTS_COMPARED.sum() > compared, "no result answered without the network was compared");
            }
            GpuStateViewCheck.compareInTests(false);
            SecurityAnalysisResult gpu = runSecurityAnalysis(network, contingencies, List.of(), sap);
            if (withoutNetwork) {
                assertTrue(GpuBatchedAcSecurityAnalysis.LAST_ANSWERED_WITHOUT_NETWORK_COUNT.get() > 0,
                        "no contingency was answered without the network: the comparison proves nothing");
            }
            assertTrue(GpuBatchedAcSecurityAnalysis.LAST_ANSWERED_FROM_ROWS_COUNT.get() > 0,
                    "no contingency was answered from its gathered rows alone");
            assertMatchesCpu(cpu, gpu);
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(commonTestConfig.matrixFactory(),
                    new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
    }

    /** Statuses, distributed active power and violation lists (values to 1e-9, relative: the two arms' states differ in
     *  the last bits). */
    private static void assertMatchesCpu(SecurityAnalysisResult cpu, SecurityAnalysisResult gpu) {
        assertEquals(cpu.getPostContingencyResults().size(), gpu.getPostContingencyResults().size());
        for (int i = 0; i < cpu.getPostContingencyResults().size(); i++) {
            PostContingencyResult c = cpu.getPostContingencyResults().get(i);
            PostContingencyResult g = gpu.getPostContingencyResults().get(i);
            String id = c.getContingency().getId();
            assertEquals(c.getStatus(), g.getStatus(), "status of " + id);
            assertEquals(c.getDistributedActivePower(), g.getDistributedActivePower(), 1e-9, "distributed active power of " + id);
            List<LimitViolation> cv = c.getLimitViolationsResult().getLimitViolations();
            List<LimitViolation> gv = g.getLimitViolationsResult().getLimitViolations();
            assertEquals(cv.size(), gv.size(), "violation count of " + id);
            for (int v = 0; v < cv.size(); v++) {
                LimitViolation a = cv.get(v);
                LimitViolation b = gv.get(v);
                String what = id + " violation " + v;
                assertEquals(a.getSubjectId(), b.getSubjectId(), what);
                assertEquals(a.getLimitType(), b.getLimitType(), what);
                assertEquals(a.getSide(), b.getSide(), what);
                assertEquals(a.getLimitName(), b.getLimitName(), what);
                assertEquals(a.getValue(), b.getValue(), 1e-9 * Math.max(1, Math.abs(a.getValue())), what);
            }
        }
    }

    @Test
    void currentActiveAndApparentPowerLimitsWithoutTheNetwork() {
        Network network = EurostagTutorialExample1Factory.createWithFixedCurrentLimits();
        for (String id : List.of("NHV1_NHV2_1", "NHV1_NHV2_2")) {
            Line line = network.getLine(id);
            line.getOrCreateSelectedOperationalLimitsGroup1().newActivePowerLimits().setPermanentLimit(250).add();
            line.getOrCreateSelectedOperationalLimitsGroup2().newApparentPowerLimits().setPermanentLimit(250).add();
        }
        assertAnsweredWithoutNetwork(network, List.of(new Contingency("NHV1_NHV2_1", new BranchContingency("NHV1_NHV2_1")),
                new Contingency("NHV1_NHV2_2", new BranchContingency("NHV1_NHV2_2"))), new LoadFlowParameters());
    }

    /** The busbar section outage opens branches at one side and loses a bus: the topology the result reads comes from the
     *  contingency's disabled network, with nothing applied. Every bus outside a zero-width band just above the nominal
     *  voltage, so a lost bus read as in service would report its voltage (the device leaves a lost bus flat at 1 pu). */
    @Test
    void busbarSectionOutageWithoutTheNetwork() {
        Network network = GpuSaTestNetworks.middleSectionLossNetwork(false);
        network.getLineStream().forEach(line -> {
            line.getOrCreateSelectedOperationalLimitsGroup1().newCurrentLimits().setPermanentLimit(1).add();
            line.getOrCreateSelectedOperationalLimitsGroup2().newCurrentLimits().setPermanentLimit(1).add();
        });
        network.getVoltageLevelStream().forEach(vl ->
                vl.setLowVoltageLimit(vl.getNominalV() * 1.0001).setHighVoltageLimit(vl.getNominalV() * 1.0001));
        LoadFlowParameters lfp = new LoadFlowParameters();
        OpenLoadFlowParameters.create(lfp)
                .setSlackBusSelectionMode(SlackBusSelectionMode.NAME).setSlackBusesIds(List.of("vl0_b"));
        assertAnsweredWithoutNetwork(network, List.of(new Contingency("bus-mid", new BusbarSectionContingency("bbs1"))), lfp);
    }

    @Test
    void busVoltageLimitsWithoutTheNetwork() {
        Network network = IeeeCdfNetworkFactory.create118();
        network.getVoltageLevelStream().forEach(vl ->
                vl.setLowVoltageLimit(vl.getNominalV()).setHighVoltageLimit(vl.getNominalV()));
        List<Contingency> contingencies = network.getLineStream().limit(3)
                .map(line -> new Contingency(line.getId(), new BranchContingency(line.getId()))).toList();
        assertAnsweredWithoutNetwork(network, contingencies, new LoadFlowParameters());
    }

    /** The busbar section outage opens l1 at its vl12 end; with a long line's charging, its vl3 end then carries more
     *  than in the base, over a limit set between the two (OLF's own security analysis on the network itself). The host
     *  no longer scans a side-opened branch in full: the screen alone finds that violation, evaluating the live end
     *  with OLF's open-branch formula. */
    @Test
    void lineOpenedAtOneSideViolatesAtItsLiveEnd() {
        Network network = GpuSaTestNetworks.middleSectionLossNetwork(false);
        Line l1 = network.getLine("l1");
        l1.setB1(1.5e-3).setB2(1.5e-3);
        LoadFlowParameters lfp = new LoadFlowParameters();
        OpenLoadFlowParameters.create(lfp)                   // the slack its own tests pin (breakers mode: bus/breaker ids)
                .setSlackBusSelectionMode(SlackBusSelectionMode.NAME).setSlackBusesIds(List.of("vl0_b"));
        List<Contingency> contingencies = List.of(new Contingency("bus-mid", new BusbarSectionContingency("bbs1")));
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        SecurityAnalysisResult cpu = runSecurityAnalysis(network, contingencies, List.of(new StateMonitor(ContingencyContext.all(),
                Set.of("l1"), Set.of(), Set.of())), sap);
        double base = cpu.getPreContingencyResult().getNetworkResult().getBranchResult("l1").getI2();
        double opened = cpu.getPostContingencyResults().get(0).getNetworkResult().getBranchResult("l1").getI2();
        assertTrue(opened > 1.2 * base, "the fixture needs l1's live end above its base current: " + opened + " A against "
                + base + " A");
        l1.getOrCreateSelectedOperationalLimitsGroup2().newCurrentLimits().setPermanentLimit((base + opened) / 2).add();
        assertAnsweredAsWrittenBack(network, contingencies, lfp);
        assertAnsweredWithoutNetwork(network, contingencies, lfp);
    }

    /** OLF's own voltage-angle fixture: with NHV1_NHV2_2 out, the angle across NHV1_NHV2_1 reaches -7.5 degrees, below
     *  one limit's low value and, read from the other end, above another's high value. The angles are read from the
     *  state, with the contingency applied (P1) and without it (P2). */
    @Test
    void voltageAngleLimits() {
        Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        Line line = network.getLine("NHV1_NHV2_1");
        network.newVoltageAngleLimit().setId("val").from(line.getTerminal1()).to(line.getTerminal2())
                .setHighLimit(7.0).setLowLimit(-7.0).add();
        network.newVoltageAngleLimit().setId("valReverse").from(line.getTerminal2()).to(line.getTerminal1())
                .setHighLimit(7.0).add();
        List<Contingency> contingencies = List.of(new Contingency("NHV1_NHV2_2", new BranchContingency("NHV1_NHV2_2")));
        assertAnsweredAsWrittenBack(network, contingencies, new LoadFlowParameters());
        assertAnsweredWithoutNetwork(network, contingencies, new LoadFlowParameters());
    }
}
