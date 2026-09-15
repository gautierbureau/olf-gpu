/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.*;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.iidm.network.test.PhaseShifterTestCaseFactory;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.loadflow.LoadFlowRunParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.dc.equations.DcApproximationType;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.HvdcNetworkFactory;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The GPU load flow provider's DC results against OLF's on the same network and parameters: every
 * bus angle, every terminal active power, every generator dispatch (the distributed slack,
 * bit for bit), and the component results — statuses, texts, distributed power, slack mismatch.
 *
 * <p>Bound on the flows: 1e-8 MW (1e-10 pu), two direct solvers on one matrix; angles 1e-7 degree.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuDcLoadFlowProviderTest {

    private static final double TOL_MW = 1e-8;
    private static final double TOL_DEG = 1e-7;

    record Case(String name, Supplier<Network> network, Supplier<LoadFlowParameters> parameters) {
    }

    private static LoadFlowParameters dc() {
        LoadFlowParameters p = new LoadFlowParameters().setDc(true);
        OpenLoadFlowParameters.create(p);
        return p;
    }

    private static Network phaseShifter(boolean regulating) {
        Network network = PhaseShifterTestCaseFactory.create();
        TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().getStep(0).setAlpha(-5);
        ps1.getPhaseTapChanger().getStep(2).setAlpha(5);
        ps1.getPhaseTapChanger().setTapPosition(2);
        if (regulating) {
            ps1.getPhaseTapChanger().setTargetDeadband(10);
            ps1.getPhaseTapChanger().setRegulationMode(PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL);
            ps1.getPhaseTapChanger().setRegulationValue(-80);
            ps1.getPhaseTapChanger().setRegulating(true);
        }
        return network;
    }

    private static Network ieee57LargeGeneratorsExcluded() {
        return IeeeCdfNetworkFactory.create57();
    }

    static List<Case> cases() {
        List<Case> cases = new ArrayList<>();
        cases.add(new Case("ieee14", IeeeCdfNetworkFactory::create14, GpuDcLoadFlowProviderTest::dc));
        for (LoadFlowParameters.BalanceType bt : LoadFlowParameters.BalanceType.values()) {
            cases.add(new Case("ieee14 distributed " + bt, IeeeCdfNetworkFactory::create14,
                    () -> dc().setDistributedSlack(true).setBalanceType(bt)));
        }
        cases.add(new Case("ieee14 distributed P_MAX no limits", IeeeCdfNetworkFactory::create14, () -> dc().setDistributedSlack(true)
                .setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX).setUseReactiveLimits(false)));
        cases.add(new Case("ieee14 three slack buses", IeeeCdfNetworkFactory::create14, () -> {
            LoadFlowParameters p = dc();
            OpenLoadFlowParameters.get(p).setMaxSlackBusCount(3);
            return p;
        }));
        cases.add(new Case("eurostag", () -> EurostagFactory.fix(EurostagTutorialExample1Factory.create()), GpuDcLoadFlowProviderTest::dc));
        cases.add(new Case("eurostag IGNORE_G", () -> {
            Network n = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
            Line line1 = n.getLine("NHV1_NHV2_1");
            line1.setR(line1.getR() * 1.1);
            line1.setX(line1.getX() * 1.05);
            return n;
        }, () -> {
            LoadFlowParameters p = dc();
            OpenLoadFlowParameters.get(p).setDcApproximationType(DcApproximationType.IGNORE_G);
            return p;
        }));
        cases.add(new Case("phase shifter fixed", () -> phaseShifter(false), GpuDcLoadFlowProviderTest::dc));
        cases.add(new Case("phase shifter regulating (outer loop)", () -> phaseShifter(true), () -> dc().setPhaseShifterRegulationOn(true)));
        cases.add(new Case("hvdc ac emulation", HvdcNetworkFactory::createWithHvdcInAcEmulation, () -> {
            LoadFlowParameters p = dc();
            p.setHvdcAcEmulation(true);
            return p;
        }));
        cases.add(new Case("zero impedance line", () -> {
            Network n = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
            n.getLine("NHV1_NHV2_1").setR(0).setX(0);
            return n;
        }, GpuDcLoadFlowProviderTest::dc));
        cases.add(new Case("ieee9 residual mismatch left on slack", IeeeCdfNetworkFactory::create9, () -> {
            LoadFlowParameters p = dc();
            OpenLoadFlowParameters.get(p).setSlackDistributionFailureBehavior(OpenLoadFlowParameters.SlackDistributionFailureBehavior.LEAVE_ON_SLACK_BUS)
                    .setPlausibleActivePowerLimit(5000);
            return p;
        }));
        for (OpenLoadFlowParameters.SlackDistributionFailureBehavior behavior : List.of(
                OpenLoadFlowParameters.SlackDistributionFailureBehavior.LEAVE_ON_SLACK_BUS,
                OpenLoadFlowParameters.SlackDistributionFailureBehavior.FAIL,
                OpenLoadFlowParameters.SlackDistributionFailureBehavior.DISTRIBUTE_ON_REFERENCE_GENERATOR)) {
            cases.add(new Case("ieee57 distribution failure " + behavior, GpuDcLoadFlowProviderTest::ieee57LargeGeneratorsExcluded, () -> {
                LoadFlowParameters p = dc().setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P);
                OpenLoadFlowParameters.get(p).setSlackDistributionFailureBehavior(behavior).setPlausibleActivePowerLimit(5000);
                return p;
            }));
        }
        cases.add(new Case("ieee118 distributed P_MAX", IeeeCdfNetworkFactory::create118,
                () -> dc().setDistributedSlack(true).setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX)));
        cases.add(new Case("ieee300 distributed P", IeeeCdfNetworkFactory::create300,
                () -> dc().setDistributedSlack(true).setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P)));
        return cases;
    }

    @TestFactory
    Stream<DynamicTest> matchesStock() {
        return cases().stream().map(c -> DynamicTest.dynamicTest(c.name(), () -> check(c)));
    }

    private static LoadFlowResult run(Network network, LoadFlowParameters parameters, boolean gpu) {
        var provider = gpu ? new GpuLoadFlowProvider(new SparseMatrixFactory()) : new OpenLoadFlowProvider(new SparseMatrixFactory());
        var runParameters = LoadFlowRunParameters.getDefault().setParameters(parameters)
                .setReportNode(ReportNode.newRootReportNode().withMessageTemplate("test").build());
        return new LoadFlow.Runner(provider).run(network, runParameters);
    }

    private static void check(Case c) {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network cpuNetwork = c.network().get();
        Network gpuNetwork = c.network().get();
        LoadFlowResult cpu = run(cpuNetwork, c.parameters().get(), false);
        GpuLoadFlowProvider.LAST_GPU_COMPONENT_COUNT.set(-1);
        LoadFlowResult gpu = run(gpuNetwork, c.parameters().get(), true);
        assertTrue(GpuLoadFlowProvider.LAST_GPU_COMPONENT_COUNT.get() >= 1, c.name() + ": the device solved a component");
        assertSameResults(c.name(), cpu, gpu);
        assertSameNetworkState(c.name(), cpuNetwork, gpuNetwork);
    }

    static void assertSameResults(String where, LoadFlowResult cpu, LoadFlowResult gpu) {
        assertEquals(cpu.isFullyConverged(), gpu.isFullyConverged(), where + ": converged");
        assertEquals(cpu.getComponentResults().size(), gpu.getComponentResults().size(), where + ": component count");
        for (int i = 0; i < cpu.getComponentResults().size(); i++) {
            LoadFlowResult.ComponentResult a = cpu.getComponentResults().get(i);
            LoadFlowResult.ComponentResult b = gpu.getComponentResults().get(i);
            String w = where + " component " + i;
            assertEquals(a.getStatus(), b.getStatus(), w + ": status");
            assertEquals(a.getStatusText(), b.getStatusText(), w + ": status text");
            assertEquals(a.getDistributedActivePower(), b.getDistributedActivePower(), 1e-9, w + ": distributed active power");
            assertEquals(a.getReferenceBusId(), b.getReferenceBusId(), w + ": reference bus");
            assertEquals(a.getSlackBusResults().size(), b.getSlackBusResults().size(), w + ": slack buses");
            for (int k = 0; k < a.getSlackBusResults().size(); k++) {
                assertEquals(a.getSlackBusResults().get(k).getId(), b.getSlackBusResults().get(k).getId(), w + ": slack bus");
                assertEquals(a.getSlackBusResults().get(k).getActivePowerMismatch(), b.getSlackBusResults().get(k).getActivePowerMismatch(), 1e-9, w + ": slack mismatch");
            }
        }
    }

    static void assertSameNetworkState(String where, Network cpu, Network gpu) {
        assertSameNetworkState(where, cpu, gpu, 0.0);
    }

    /**
     * @param generatorTol tolerance on the generator and load dispatch. Zero (bit-identical) whenever
     *                     the dispatch comes from the slack distribution alone: the same operations in
     *                     the same order on both arms. A loop that distributes FROM THE SOLVED STATE —
     *                     area interchange control — feeds it a state that differs by roundoff between
     *                     two direct solvers, so its dispatch does too.
     */
    static void assertSameNetworkState(String where, Network cpu, Network gpu, double generatorTol) {
        for (Bus bus : cpu.getBusView().getBuses()) {
            Bus other = gpu.getBusView().getBus(bus.getId());
            assertNotNull(other, where + ": bus " + bus.getId());
            assertSameDouble(bus.getAngle(), other.getAngle(), TOL_DEG, where + ": angle of " + bus.getId());
        }
        for (Branch<?> branch : cpu.getBranches()) {
            Branch<?> other = gpu.getBranch(branch.getId());
            assertSameDouble(branch.getTerminal1().getP(), other.getTerminal1().getP(), TOL_MW, where + ": p1 of " + branch.getId());
            assertSameDouble(branch.getTerminal2().getP(), other.getTerminal2().getP(), TOL_MW, where + ": p2 of " + branch.getId());
        }
        for (Generator g : cpu.getGenerators()) {
            Generator other = gpu.getGenerator(g.getId());
            assertSameDouble(g.getTerminal().getP(), other.getTerminal().getP(), generatorTol, where + ": p of generator " + g.getId());
        }
        for (Load l : cpu.getLoads()) {
            assertSameDouble(l.getTerminal().getP(), gpu.getLoad(l.getId()).getTerminal().getP(), generatorTol, where + ": p of load " + l.getId());
        }
        for (TwoWindingsTransformer t : cpu.getTwoWindingsTransformers()) {
            if (t.hasPhaseTapChanger()) {
                TwoWindingsTransformer other = gpu.getTwoWindingsTransformer(t.getId());
                assertEquals(t.getPhaseTapChanger().getTapPosition(), other.getPhaseTapChanger().getTapPosition(), where + ": tap of " + t.getId());
                assertEquals(t.getPhaseTapChanger().getSolvedTapPosition(), other.getPhaseTapChanger().getSolvedTapPosition(), where + ": solved tap of " + t.getId());
            }
        }
    }

    private static void assertSameDouble(double expected, double actual, double tol, String what) {
        if (Double.isNaN(expected) || Double.isNaN(actual)) {
            assertEquals(Double.isNaN(expected), Double.isNaN(actual), what + ": NaN " + expected + " vs " + actual);
            return;
        }
        if (tol == 0.0) {
            assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(actual), what + ": " + expected + " vs " + actual);
        } else {
            assertEquals(expected, actual, tol, what);
        }
    }

    @Test
    void throwBehaviorThrowsLikeStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Supplier<LoadFlowParameters> parameters = () -> {
            LoadFlowParameters p = dc().setBalanceType(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P);
            OpenLoadFlowParameters.get(p).setSlackDistributionFailureBehavior(OpenLoadFlowParameters.SlackDistributionFailureBehavior.THROW)
                    .setPlausibleActivePowerLimit(5000);
            return p;
        };
        var cpu = assertThrows(Exception.class, () -> run(IeeeCdfNetworkFactory.create57(), parameters.get(), false));
        var gpu = assertThrows(Exception.class, () -> run(IeeeCdfNetworkFactory.create57(), parameters.get(), true));
        Throwable cpuCause = rootCause(cpu);
        Throwable gpuCause = rootCause(gpu);
        assertInstanceOf(PowsyblException.class, cpuCause);
        assertEquals(cpuCause.getClass(), gpuCause.getClass());
        assertEquals(cpuCause.getMessage(), gpuCause.getMessage());
    }

    private static Throwable rootCause(Throwable t) {
        while (t.getCause() != null && t.getCause() != t) {
            t = t.getCause();
        }
        return t;
    }

    @Test
    void providerIsRegistered() {
        assertTrue(LoadFlow.find(GpuLoadFlowProvider.NAME).getName().equals(GpuLoadFlowProvider.NAME));
    }
}
