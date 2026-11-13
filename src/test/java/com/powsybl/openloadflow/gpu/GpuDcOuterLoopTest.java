/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.iidm.network.HvdcConverterStation;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.PhaseTapChanger;
import com.powsybl.iidm.network.TwoWindingsTransformer;
import com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControl;
import com.powsybl.iidm.network.test.PhaseShifterTestCaseFactory;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.loadflow.LoadFlowRunParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.dc.gpu.GpuDcLoadFlowEngine;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.openloadflow.network.HvdcNetworkFactory;
import com.powsybl.openloadflow.network.MultiAreaNetworkFactory;
import org.junit.jupiter.api.Test;

import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The DC outer loops as device kernels, against OLF's own loops.
 *
 * <p>Every case is a scenario where the loop MUST fire — a phase shifter that has to move taps, an
 * HVDC that has to saturate, one that has to come back to linear, one that has to flip saturation
 * side — and the assertions are in two parts: the result equals the stock provider's (taps, flows,
 * angles, dispatch), AND the device decision counter says the kernel actually made the decisions.
 * A kernel that silently decided nothing would leave the taps and the AC-emulation statuses at
 * their initial values and fail the comparison, not just the counter.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuDcOuterLoopTest {

    private static LoadFlowParameters dc() {
        LoadFlowParameters p = new LoadFlowParameters().setDc(true);
        OpenLoadFlowParameters.create(p);
        return p;
    }

    private static LoadFlowResult run(Network network, LoadFlowParameters parameters, boolean gpu) {
        var provider = gpu ? new GpuLoadFlowProvider(new SparseMatrixFactory()) : new OpenLoadFlowProvider(new SparseMatrixFactory());
        var runParameters = LoadFlowRunParameters.getDefault().setParameters(parameters)
                .setReportNode(ReportNode.newRootReportNode().withMessageTemplate("test").build());
        return new LoadFlow.Runner(provider).run(network, runParameters);
    }

    /** Both arms on their own copy of the network, then the full state comparison. */
    private static Network[] runBoth(Supplier<Network> factory, Supplier<LoadFlowParameters> parameters, String where) {
        return runBoth(factory, parameters, where, 0.0);
    }

    private static Network[] runBoth(Supplier<Network> factory, Supplier<LoadFlowParameters> parameters, String where, double generatorTol) {
        Network cpu = factory.get();
        Network gpu = factory.get();
        LoadFlowResult cpuResult = run(cpu, parameters.get(), false);
        GpuLoadFlowProvider.LAST_GPU_COMPONENT_COUNT.set(-1);
        GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.set(-1);
        GpuDcLoadFlowEngine.LAST_DEVICE_HVDC_SWITCHES.set(-1);
        LoadFlowResult gpuResult = run(gpu, parameters.get(), true);
        assertTrue(GpuLoadFlowProvider.LAST_GPU_COMPONENT_COUNT.get() >= 1, where + ": the device solved a component");
        GpuDcLoadFlowProviderTest.assertSameResults(where, cpuResult, gpuResult);
        GpuDcLoadFlowProviderTest.assertSameNetworkState(where, cpu, gpu, generatorTol);
        return new Network[] {cpu, gpu};
    }

    private static Network phaseShifterRegulating(double regulationValue) {
        Network network = PhaseShifterTestCaseFactory.create();
        TwoWindingsTransformer ps1 = network.getTwoWindingsTransformer("PS1");
        ps1.getPhaseTapChanger().getStep(0).setAlpha(-5);
        ps1.getPhaseTapChanger().getStep(2).setAlpha(5);
        ps1.getPhaseTapChanger().setTargetDeadband(10);
        ps1.getPhaseTapChanger().setRegulationMode(PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL);
        ps1.getPhaseTapChanger().setRegulationValue(regulationValue);
        ps1.getPhaseTapChanger().setRegulating(true);
        return network;
    }

    @Test
    void phaseControlMovesTheSameTaps() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        // OLF's own DcLoadFlowTest.outerLoopPhaseShifterTest scenario: tap 1 -> 2
        Network[] networks = runBoth(() -> phaseShifterRegulating(-80), () -> dc().setPhaseShifterRegulationOn(true), "phase control");
        PhaseTapChanger cpu = networks[0].getTwoWindingsTransformer("PS1").getPhaseTapChanger();
        PhaseTapChanger gpu = networks[1].getTwoWindingsTransformer("PS1").getPhaseTapChanger();
        assertEquals(2, cpu.getSolvedTapPosition(), "the stock loop moved the tap");
        assertEquals(cpu.getSolvedTapPosition(), gpu.getSolvedTapPosition(), "same solved tap");
        assertNotEquals(gpu.getTapPosition(), gpu.getSolvedTapPosition(), "the tap actually moved");
        assertTrue(GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.get() >= 1,
                "the device decided the tap changes: " + GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.get());
    }

    @Test
    void phaseControlDirectionRatchetMatches() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        // a target the phase shifter cannot reach: the loop keeps moving taps until the direction
        // ratchet (MAX_DIRECTION_CHANGE) or the tap range stops it — the state OLF keeps per controller
        runBoth(() -> phaseShifterRegulating(-10), () -> dc().setPhaseShifterRegulationOn(true), "phase control unreachable target");
        assertTrue(GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.get() >= 1, "the device decided the tap changes");
    }

    @Test
    void hvdcSaturatesLikeStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        // OLF's own AcLoadFlowVscTest.testAcEmuAndPMax (isDc = true): without the limit p = 195
        Supplier<Network> factory = () -> {
            Network network = HvdcNetworkFactory.createHvdcLinkedByTwoLinesAndSwitch(HvdcConverterStation.HvdcType.VSC);
            network.getHvdcLine("hvdc23").setMaxP(170);
            return network;
        };
        Network[] networks = runBoth(factory, () -> dc().setHvdcAcEmulation(true), "hvdc saturation");
        assertEquals(170, networks[0].getHvdcConverterStation("cs2").getTerminal().getP(), 1e-3, "the stock loop saturated the link");
        assertEquals(networks[0].getHvdcConverterStation("cs2").getTerminal().getP(),
                networks[1].getHvdcConverterStation("cs2").getTerminal().getP(), 1e-8, "same saturated flow");
        assertTrue(GpuDcLoadFlowEngine.LAST_DEVICE_HVDC_SWITCHES.get() >= 1,
                "the device decided the AC-emulation switch: " + GpuDcLoadFlowEngine.LAST_DEVICE_HVDC_SWITCHES.get());
    }

    @Test
    void hvdcSaturatesOnTheOtherSideLikeStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Supplier<Network> factory = () -> {
            Network network = HvdcNetworkFactory.createHvdcLinkedByTwoLinesAndSwitch(HvdcConverterStation.HvdcType.VSC);
            network.getHvdcLine("hvdc23").setMaxP(170);
            HvdcAngleDroopActivePowerControl control = network.getHvdcLine("hvdc23").getExtension(HvdcAngleDroopActivePowerControl.class);
            control.setP0(-control.getP0());
            return network;
        };
        Network[] networks = runBoth(factory, () -> dc().setHvdcAcEmulation(true), "hvdc saturation, inverted");
        assertEquals(170, networks[0].getHvdcConverterStation("cs3").getTerminal().getP(), 1e-3, "the stock loop saturated the other way");
        assertEquals(networks[0].getHvdcConverterStation("cs3").getTerminal().getP(),
                networks[1].getHvdcConverterStation("cs3").getTerminal().getP(), 1e-8, "same saturated flow");
        assertTrue(GpuDcLoadFlowEngine.LAST_DEVICE_HVDC_SWITCHES.get() >= 1, "the device decided the AC-emulation switch");
    }

    @Test
    void hvdcAndPhaseControlTogether() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        // OLF's own AcLoadFlowVscTest.testAcEmulationSaturationToLinearChange network: the HVDC
        // saturates and the phase shifter moves — the two loops interleaved, which is the DC engine's
        // nested sweep over the loop list (OLF's own AC case uses maxP 51; in DC the link carries
        // 49.7 MW and never reaches it, so the limit that makes the DC loops fire is 20 MW)
        Supplier<Network> factory = () -> {
            Network network = HvdcNetworkFactory.createHvdcInAcEmulationAndPSTInSymetricNetwork();
            network.getHvdcLine("hvdc12").setMaxP(20);
            return network;
        };
        Network[] networks = runBoth(factory, () -> dc().setHvdcAcEmulation(true).setPhaseShifterRegulationOn(true), "hvdc + phase control");
        assertEquals(20, networks[0].getHvdcConverterStation("cs1").getTerminal().getP(), 1e-3, "the stock loop saturated the link");
        assertEquals(0, networks[0].getTwoWindingsTransformer("pst").getPhaseTapChanger().getSolvedTapPosition(), "the stock loop moved the tap");
        assertTrue(GpuDcLoadFlowEngine.LAST_DEVICE_HVDC_SWITCHES.get() >= 1, "the device decided AC-emulation switches");
        assertTrue(GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.get() >= 1, "the device decided tap changes");
    }

    @Test
    void hvdcSaturationFlipsSideLikeStock() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        // OLF's own AcLoadFlowVscTest.testAcEmulationSaturationToOppositeChange (isDc = true)
        Supplier<Network> factory = () -> {
            Network network = HvdcNetworkFactory.createHvdcInAcEmulationAndPSTInSymetricNetwork();
            network.getHvdcLine("hvdc12").setMaxP(1.5);
            network.getGenerator("g1").setTargetP(15);
            network.getLoad("l1").setP0(10);
            network.getGenerator("g2").setTargetP(10);
            network.getLoad("l2").setP0(15);
            network.getTwoWindingsTransformer("pst").getPhaseTapChanger().setRegulationValue(20);
            return network;
        };
        runBoth(factory, () -> dc().setHvdcAcEmulation(true).setPhaseShifterRegulationOn(true), "hvdc saturation side flip");
        assertTrue(GpuDcLoadFlowEngine.LAST_DEVICE_HVDC_SWITCHES.get() >= 2, "the device decided both switches");
    }

    @Test
    void areaInterchangeControlStaysOnTheHost() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        // OLF's own DcLoadFlowTest.multipleOuterLoopsTest: the area interchange loop has no device
        // model and runs through OLF, interleaved with the device phase control
        Supplier<LoadFlowParameters> parameters = () -> {
            LoadFlowParameters p = dc().setPhaseShifterRegulationOn(true);
            OpenLoadFlowParameters.get(p).setAreaInterchangeControl(true).setAreaInterchangePMaxMismatch(1);
            return p;
        };
        // the area interchange loop distributes FROM THE SOLVED STATE, so its dispatch carries the two
        // solvers' roundoff (1e-13 MW here) instead of being bit-identical like the pre-solve distribution
        Network[] networks = runBoth(MultiAreaNetworkFactory::createTwoAreasWithPhaseShifter, parameters,
                "area interchange + phase control", 1e-9);
        assertEquals(networks[0].getArea("A1").getInterchangeTarget().orElseThrow(), networks[1].getArea("A1").getInterchange(), 1,
                "the area reached its interchange target on the device arm");
        assertTrue(GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.get() >= 0, "the device loop tables were built");
    }

    /**
     * The A/B of the kernels: the same runs with {@link GpuDcLoadFlowEngine#HOST_OUTER_LOOPS} set take OLF's
     * own loop checks and must land on the same result — the device decisions are the loops', not an
     * approximation of them.
     */
    @Test
    void hostLoopsGiveTheSameResult() {
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        Network device = HvdcNetworkFactory.createHvdcInAcEmulationAndPSTInSymetricNetwork();
        device.getHvdcLine("hvdc12").setMaxP(20);
        Network host = HvdcNetworkFactory.createHvdcInAcEmulationAndPSTInSymetricNetwork();
        host.getHvdcLine("hvdc12").setMaxP(20);
        LoadFlowParameters parameters = dc().setHvdcAcEmulation(true).setPhaseShifterRegulationOn(true);
        GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.set(-1);
        LoadFlowResult deviceResult = run(device, parameters, true);
        assertTrue(GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.get() >= 1, "the kernels ran");
        GpuDcLoadFlowEngine.HOST_OUTER_LOOPS.set(true);
        LoadFlowResult hostResult;
        try {
            GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.set(-1);
            hostResult = run(host, parameters, true);
            assertEquals(-1, GpuDcLoadFlowEngine.LAST_DEVICE_TAP_CHANGES.get(), "the kernels did not run");
        } finally {
            GpuDcLoadFlowEngine.HOST_OUTER_LOOPS.set(false);
        }
        GpuDcLoadFlowProviderTest.assertSameResults("device loops vs host loops", hostResult, deviceResult);
        GpuDcLoadFlowProviderTest.assertSameNetworkState("device loops vs host loops", host, device);
    }
}
