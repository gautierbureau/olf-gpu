/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Line;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.RatioTapChanger;
import com.powsybl.iidm.network.Switch;
import com.powsybl.iidm.network.ThreeSides;
import com.powsybl.iidm.network.TwoSides;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuAutomationSystemOuterLoop;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.network.AutomationSystemNetworkFactory;
import com.powsybl.openloadflow.network.VoltageControlNetworkFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * AutomationSystem driven on the device ({@code GpuOuterLoopKind.AUTOMATION_SYSTEM}) against plain OLF, on the networks of
 * OLF's own AutomationSystemTest: a line an overload opens, a line one closes, a system that never trips. Each run must
 * agree with OLF on the status, the iteration count, the bus voltages and angles, the branch currents and every terminal
 * and switch state; where the device owns the loop it must switch the branch itself, with no fallback. A switch to operate
 * is a zero-impedance coupler, which the device does not switch: OLF's loop stays on the host and the results still match.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuAutomationSystemTest {

    private static final double TOLERANCE = 1e-9;

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    private static LoadFlowParameters parameters() {
        LoadFlowParameters parameters = new LoadFlowParameters();
        OpenLoadFlowParameters.create(parameters).setSimulateAutomationSystems(true);
        return parameters;
    }

    /** Run plain OLF and the GPU on two copies and compare; {@code deviceSwitches}: the branches the device must switch. */
    private static void compare(String label, Supplier<Network> networkSupplier, int deviceSwitches) {
        compare(label, networkSupplier, GpuAutomationSystemTest::parameters, deviceSwitches);
    }

    private static void compare(String label, Supplier<Network> networkSupplier, Supplier<LoadFlowParameters> parametersSupplier,
                                int deviceSwitches) {
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));
        Network cpuNetwork = networkSupplier.get();
        LoadFlowResult cpuResult = runner.run(cpuNetwork, parametersSupplier.get());

        Network gpuNetwork = networkSupplier.get();
        LoadFlowParameters gpuParameters = parametersSupplier.get();
        OpenLoadFlowParameters.get(gpuParameters).setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        long fallbacksBefore = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        long switchedBefore = GpuAutomationSystemOuterLoop.DEVICE_SWITCHED_BRANCHES.sum();
        LoadFlowResult gpuResult = runner.run(gpuNetwork, gpuParameters);
        long fallbacks = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get() - fallbacksBefore;
        long switched = GpuAutomationSystemOuterLoop.DEVICE_SWITCHED_BRANCHES.sum() - switchedBefore;

        var cpuComponent = cpuResult.getComponentResults().get(0);
        var gpuComponent = gpuResult.getComponentResults().get(0);
        assertEquals(cpuComponent.getStatus(), gpuComponent.getStatus(), label + ": status");
        assertEquals(cpuComponent.getIterationCount(), gpuComponent.getIterationCount(), label + ": iteration count");
        double worstV = 0;
        double worstAngle = 0;
        for (Bus cpuBus : cpuNetwork.getBusView().getBuses()) {
            Bus gpuBus = gpuNetwork.getBusView().getBus(cpuBus.getId());
            if (gpuBus == null || Double.isNaN(cpuBus.getV())) {
                continue;
            }
            worstV = Math.max(worstV, Math.abs(cpuBus.getV() - gpuBus.getV()));
            worstAngle = Math.max(worstAngle, Math.abs(cpuBus.getAngle() - gpuBus.getAngle()));
        }
        double worstI = 0;
        for (Branch<?> cpuBranch : cpuNetwork.getBranches()) {
            Branch<?> gpuBranch = gpuNetwork.getBranch(cpuBranch.getId());
            for (TwoSides side : TwoSides.values()) {
                assertEquals(cpuBranch.getTerminal(side).isConnected(), gpuBranch.getTerminal(side).isConnected(),
                        label + ": " + cpuBranch.getId() + " side " + side + " connection");
                double cpuI = cpuBranch.getTerminal(side).getI();
                double gpuI = gpuBranch.getTerminal(side).getI();
                assertEquals(Double.isNaN(cpuI), Double.isNaN(gpuI), label + ": " + cpuBranch.getId() + " side " + side + " current set");
                if (!Double.isNaN(cpuI)) {
                    worstI = Math.max(worstI, Math.abs(cpuI - gpuI));
                }
            }
        }
        for (Switch cpuSwitch : cpuNetwork.getSwitches()) {
            assertEquals(cpuSwitch.isOpen(), gpuNetwork.getSwitch(cpuSwitch.getId()).isOpen(), label + ": switch " + cpuSwitch.getId());
        }
        System.out.printf("AutomationSystem GPU vs CPU [%s]: %s, %d iterations, max dV %.2e kV, dAngle %.2e deg, dI %.2e A,"
                        + " fallbacks %d, branches switched on device %d%n",
                label, gpuComponent.getStatus(), gpuComponent.getIterationCount(), worstV, worstAngle, worstI, fallbacks, switched);
        assertTrue(worstV < TOLERANCE, label + ": bus voltages (max dV = " + worstV + " kV)");
        assertTrue(worstAngle < TOLERANCE, label + ": bus angles (max dAngle = " + worstAngle + " deg)");
        assertTrue(worstI < TOLERANCE, label + ": branch currents (max dI = " + worstI + " A)");
        assertEquals(0, fallbacks, label + ": no fallback to the CPU");
        assertEquals(deviceSwitches, switched, label + ": branches the device switched");
    }

    /** l34 overloaded: its system opens l33p, and l34's current drops below the threshold. */
    @Test
    void branchTripping() {
        compare("l33p opened", AutomationSystemNetworkFactory::createWithBranchTripping, 1);
    }

    /** l12 overloaded: its system closes l33p, open in the base. */
    @Test
    void branchClosing() {
        compare("l33p closed", AutomationSystemNetworkFactory::createWithBranchTripping2, 1);
    }

    /** A system whose monitored line never reaches its threshold: nothing switches. */
    @Test
    void noTripping() {
        compare("no tripping", () -> {
            Network network = AutomationSystemNetworkFactory.createWithBadAutomationSystems();
            network.getSubstation("s1").newOverloadManagementSystem()
                    .setId("l34_opens_l12")
                    .setEnabled(true)
                    .setMonitoredElementId("l34")
                    .setMonitoredElementSide(ThreeSides.ONE)
                    .newBranchTripping()
                    .setKey("key")
                    .setBranchToOperateId("l33p")
                    .setSideToOperate(TwoSides.TWO)
                    .setCurrentLimit(2000.)
                    .add()
                    .add();
            return network;
        }, 0);
    }

    /** br1 is a switch, a zero-impedance coupler in breaker mode: OLF's loop opens it on the host, and the run's next solve
     *  stays on the device (the opened coupler's dummy variables packed base-open, the distribution starting at what the
     *  first solve placed). */
    @Test
    void switchTrippingStaysOnTheHost() {
        compare("br1 opened by OLF's loop", AutomationSystemNetworkFactory::create, 0);
    }

    @Test
    void switchClosingStaysOnTheHost() {
        compare("br1 closed by OLF's loop", AutomationSystemNetworkFactory::createWithSwitchToClose, 0);
    }

    /** The same switch tripping with a second generator sharing the slack mismatch: the distribution then has participants
     *  besides the slack bus's own generator, and the run's second solve starts from what the first gave each of them. */
    @Test
    void switchTrippingWithTwoParticipatingGenerators() {
        compare("br1 opened by OLF's loop, two participants", () -> {
            Network network = AutomationSystemNetworkFactory.create();
            network.getVoltageLevel("b4_vl").newGenerator()
                    .setId("g4")
                    .setBus("b4")
                    .setConnectableBus("b4")
                    .setMinP(0)
                    .setMaxP(100)
                    .setTargetP(20)
                    .setTargetQ(0)
                    .setVoltageRegulatorOn(false)
                    .add();
            return network;
        }, 0);
    }

    /** A tripping whose opening leaves b5 alone: OLF disables the island it makes, which the device does not, so OLF's loop
     *  stays on the host for the run. */
    @Test
    void trippingThatIslandsABusStaysOnTheHost() {
        compare("l45 opened by OLF's loop", GpuAutomationSystemTest::radialBusTrippedByAnOverload, 0);
    }

    /** OLF's branch tripping network with b5 fed by l45 from b4 alone, and l34's overload opening l45 instead of l33p. */
    private static Network radialBusTrippedByAnOverload() {
        Network network = AutomationSystemNetworkFactory.createWithBranchTripping();
        var vl5 = network.getSubstation("s2").newVoltageLevel().setId("vl5").setNominalV(63)
                .setTopologyKind(com.powsybl.iidm.network.TopologyKind.BUS_BREAKER).add();
        vl5.getBusBreakerView().newBus().setId("b5").add();
        vl5.newLoad().setId("ld5").setBus("b5").setP0(2).setQ0(1).add();
        network.newLine().setId("l45").setBus1("b4").setBus2("b5").setR(0.1).setX(3).add();
        network.getOverloadManagementSystem("l34_opens_l12").remove();
        network.getSubstation("s1").newOverloadManagementSystem()
                .setId("l34_opens_l45")
                .setEnabled(true)
                .setMonitoredElementId("l34")
                .setMonitoredElementSide(ThreeSides.ONE)
                .newBranchTripping()
                .setKey("l45 key")
                .setBranchToOperateId("l45")
                .setSideToOperate(TwoSides.TWO)
                .setCurrentLimit(200.)
                .add()
                .add();
        return network;
    }

    /**
     * OLF's createNetworkWithT2wt with LINE_12 doubled and T2wT's ratio tap changer regulating BUS_3 (1 kV deadband), its
     * steps 1 to 3 at twice the transformer's reactance; T2wT's system opens LINE_12b above 100 A at its side 2.
     */
    private static Network regulatedTransformer() {
        Network network = VoltageControlNetworkFactory.createNetworkWithT2wt();
        Line line = network.getLine("LINE_12");
        network.newLine().setId("LINE_12b").setBus1("BUS_1").setConnectableBus1("BUS_1").setBus2("BUS_2").setConnectableBus2("BUS_2")
                .setR(line.getR()).setX(line.getX()).setG1(line.getG1()).setB1(line.getB1()).setG2(line.getG2()).setB2(line.getB2())
                .add();
        RatioTapChanger tapChanger = network.getTwoWindingsTransformer("T2wT").getRatioTapChanger();
        tapChanger.setTargetDeadband(1.0).setRegulating(true);
        for (int step = 1; step <= 3; step++) {
            tapChanger.getStep(step).setX(100);
        }
        network.getSubstation("SUBSTATION").newOverloadManagementSystem()
                .setId("T2wT_opens_LINE_12b")
                .setEnabled(true)
                .setMonitoredElementId("T2wT")
                .setMonitoredElementSide(ThreeSides.TWO)
                .newBranchTripping()
                .setKey("LINE_12b key")
                .setBranchToOperateId("LINE_12b")
                .setSideToOperate(TwoSides.ONE)
                .setCurrentLimit(100.)
                .add()
                .add();
        return network;
    }

    /**
     * OLF's incremental voltage control moves T2wT from tap 0 to tap 2, and its system then reads 88.6 A at that tap: below
     * 100 A, LINE_12b stays closed. The moved tap reaches the device as a tap override; read at the base tap's impedance,
     * the current came out above 110 A and the device opened the line.
     */
    @Test
    void monitoredTransformerAtTheTapItsVoltageControlMoved() {
        compare("T2wT at its moved tap", GpuAutomationSystemTest::regulatedTransformer, () -> {
            LoadFlowParameters parameters = parameters().setTransformerVoltageControlOn(true);
            OpenLoadFlowParameters.get(parameters)
                    .setTransformerVoltageControlMode(OpenLoadFlowParameters.TransformerVoltageControlMode.INCREMENTAL_VOLTAGE_CONTROL);
            return parameters;
        }, 0);
    }
}
