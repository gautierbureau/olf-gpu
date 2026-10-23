/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Area;
import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Generator;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuAreaInterchangeControlOuterLoop;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.network.FourBusNetworkFactory;
import com.powsybl.openloadflow.network.MultiAreaNetworkFactory;
import com.powsybl.openloadflow.network.SlackBusSelectionMode;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CompletionException;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * AreaInterchangeControl driven on the device ({@code GpuOuterLoopKind.AREA_INTERCHANGE_CONTROL}) against plain OLF, on the
 * networks and settings of OLF's own AreaInterchangeControlTest: the areas' interchanges balanced on their own generators
 * with their share of the slack bus mismatch, the slack mismatch on the buses without area, the remaining slack spread
 * over the areas by their margins, boundaries on branches, unpaired boundary lines and tie lines, and the failure
 * behaviours. Every run must agree with OLF on the status, the iteration count, the bus voltages and angles, the generators'
 * active power, the areas' interchanges and the distributed active power; a run OLF converges must not fall back to the
 * CPU, and the device must be the one that moved the generators.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuAreaInterchangeControlTest {

    private static final double TOLERANCE = 1e-9;

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    /** OLF's AreaInterchangeControlTest settings. */
    private static LoadFlowParameters parameters(Consumer<LoadFlowParameters> customize) {
        LoadFlowParameters parameters = new LoadFlowParameters();
        OpenLoadFlowParameters.create(parameters)
                .setAreaInterchangeControl(true)
                .setSlackBusPMaxMismatch(1e-3)
                .setAreaInterchangePMaxMismatch(1e-1);
        customize.accept(parameters);
        return parameters;
    }

    private static Supplier<Network> twoAreas(Supplier<Network> factory, double interchangeTarget1, double interchangeTarget2) {
        return () -> {
            Network network = factory.get();
            network.getArea("a1").setInterchangeTarget(interchangeTarget1);
            network.getArea("a2").setInterchangeTarget(interchangeTarget2);
            return network;
        };
    }

    /** OLF's prepareAicRemainingSlackDistribution: the slack bus in a1, with margins of 0.9 MW for a1 and 0.01 MW for a2. */
    private static Network remainingSlackDistributionNetwork() {
        Network network = MultiAreaNetworkFactory.createTwoAreasWithXNode();
        network.getGenerator("g1").setTargetP(70.4);
        network.getGenerator("gen3").setTargetP(40);
        network.getArea("a1").setInterchangeTarget(-10);
        network.getArea("a2").setInterchangeTarget(9.51);
        return network;
    }

    private static void remainingSlackDistributionParameters(LoadFlowParameters parameters) {
        parameters.setReadSlackBus(true);
        OpenLoadFlowParameters.get(parameters)
                .setSlackBusSelectionMode(SlackBusSelectionMode.NAME)
                .setSlackBusId("vl1_0")
                .setAreaInterchangePMaxMismatch(0.5);
    }

    /**
     * Run plain OLF and the GPU on two copies of the network and compare. {@code onDevice}: the device must solve it (no CPU
     * fallback) and move generators itself; false for a run whose distribution leaves a remaining mismatch OLF's failure
     * behaviour decides, which the device hands to the CPU.
     */
    private static void compare(String label, Supplier<Network> networkSupplier, Consumer<LoadFlowParameters> customize, boolean onDevice) {
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));
        Network cpuNetwork = networkSupplier.get();
        LoadFlowResult cpuResult = runner.run(cpuNetwork, parameters(customize));

        Network gpuNetwork = networkSupplier.get();
        LoadFlowParameters gpuParameters = parameters(customize);
        OpenLoadFlowParameters.get(gpuParameters).setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        long fallbacksBefore = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        long distributionsBefore = GpuAreaInterchangeControlOuterLoop.DEVICE_DISTRIBUTIONS.sum();
        LoadFlowResult gpuResult = runner.run(gpuNetwork, gpuParameters);
        long fallbacks = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get() - fallbacksBefore;
        long distributions = GpuAreaInterchangeControlOuterLoop.DEVICE_DISTRIBUTIONS.sum() - distributionsBefore;

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
        double worstP = 0;
        for (Generator cpuGenerator : cpuNetwork.getGenerators()) {
            double cpuP = cpuGenerator.getTerminal().getP();
            if (!Double.isNaN(cpuP)) {
                worstP = Math.max(worstP, Math.abs(cpuP - gpuNetwork.getGenerator(cpuGenerator.getId()).getTerminal().getP()));
            }
        }
        double worstInterchange = 0;
        for (Area cpuArea : cpuNetwork.getAreas()) {
            worstInterchange = Math.max(worstInterchange, Math.abs(cpuArea.getInterchange() - gpuNetwork.getArea(cpuArea.getId()).getInterchange()));
        }
        double distributedDifference = Math.abs(cpuComponent.getDistributedActivePower() - gpuComponent.getDistributedActivePower());
        System.out.printf("AreaInterchangeControl GPU vs CPU [%s]: %s, %d iterations, max dV %.2e kV, dAngle %.2e deg, dP %.2e MW,"
                        + " dInterchange %.2e MW, dDistributed %.2e MW, fallbacks %d, device distributions %d%n",
                label, gpuComponent.getStatus(), gpuComponent.getIterationCount(), worstV, worstAngle, worstP, worstInterchange,
                distributedDifference, fallbacks, distributions);
        assertTrue(worstV < TOLERANCE, label + ": bus voltages (max dV = " + worstV + " kV)");
        assertTrue(worstAngle < TOLERANCE, label + ": bus angles (max dAngle = " + worstAngle + " deg)");
        assertTrue(worstP < TOLERANCE, label + ": generator active power (max dP = " + worstP + " MW)");
        assertTrue(worstInterchange < TOLERANCE, label + ": area interchanges (max = " + worstInterchange + " MW)");
        assertTrue(distributedDifference < TOLERANCE, label + ": distributed active power (CPU " + cpuComponent.getDistributedActivePower()
                + " MW, GPU " + gpuComponent.getDistributedActivePower() + " MW)");
        if (onDevice) {
            assertEquals(0, fallbacks, label + ": the device must solve it");
            assertTrue(distributions > 0, label + ": the device's AreaInterchangeControl phase must move the generators");
        }
    }

    @Test
    void twoAreasWithXNode() {
        compare("two areas, X node", twoAreas(MultiAreaNetworkFactory::createTwoAreasWithXNode, -40, 40), p -> { }, true);
    }

    @Test
    void twoAreasWithUnpairedBoundaryLine() {
        // a1's boundary is an unpaired boundary line with P0 = 20 MW
        compare("two areas, unpaired boundary line", twoAreas(MultiAreaNetworkFactory::createTwoAreasWithBoundaryLine, -60, 40), p -> { }, true);
    }

    @Test
    void twoAreasWithTieLine() {
        compare("two areas, tie line", twoAreas(MultiAreaNetworkFactory::createTwoAreasWithTieLine, -40, 40), p -> { }, true);
    }

    @Test
    void twoAreasWithUnconsideredTieLine() {
        compare("two areas, tie line outside the boundaries", twoAreas(MultiAreaNetworkFactory::createTwoAreasWithUnconsideredTieLine, -40, 40),
                p -> { }, true);
    }

    @Test
    void halfNetworkWithoutArea() {
        compare("half the network without area", () -> {
            Network network = MultiAreaNetworkFactory.createTwoAreasWithTieLine();
            network.getArea("a2").remove();
            return network;
        }, p -> { }, true);
    }

    /** No area: OLF's AreaInterchangeControl is its DistributedSlackOuterLoop, at its own place in the loop order. */
    @Test
    void networkWithoutAreas() {
        compare("no area", FourBusNetworkFactory::createBaseNetwork, p -> p.setDistributedSlack(false), true);
    }

    @Test
    void slackBusOnBoundaryBus() {
        compare("slack on a boundary bus", twoAreas(MultiAreaNetworkFactory::createTwoAreasWithTwoXNodes, -15, 15),
                p -> OpenLoadFlowParameters.get(p).setSlackBusSelectionMode(SlackBusSelectionMode.NAME).setSlackBusId("bx1_vl_0"), true);
    }

    @Test
    void slackBusOnIgnoredBoundaryBus() {
        compare("slack on a boundary bus outside the interchange", twoAreas(MultiAreaNetworkFactory::createTwoAreasWithTwoXNodes, -15, 15),
                p -> OpenLoadFlowParameters.get(p).setSlackBusSelectionMode(SlackBusSelectionMode.NAME).setSlackBusId("bx2_vl_0"), true);
    }

    @Test
    void tenAreas() {
        compare("ten areas", MultiAreaNetworkFactory::createTenAreas,
                p -> OpenLoadFlowParameters.get(p).setLowImpedanceBranchMode(OpenLoadFlowParameters.LowImpedanceBranchMode.REPLACE_BY_MIN_IMPEDANCE_LINE), true);
    }

    /** distributeRemainingSlackMismatch: once both areas are in their band, the slack mismatch goes to the areas by margin. */
    @Test
    void remainingSlackDistribution() {
        compare("remaining slack over the areas", MultiAreaNetworkFactory::createTwoAreasWithXNodeHighZ,
                p -> OpenLoadFlowParameters.get(p).setAreaInterchangePMaxMismatch(0.5), true);
    }

    @Test
    void remainingSlackDistributionShares() {
        compare("remaining slack shares", GpuAreaInterchangeControlTest::remainingSlackDistributionNetwork,
                GpuAreaInterchangeControlTest::remainingSlackDistributionParameters, true);
    }

    /** a1 can take 0.1 MW of its share only: it drops out and a2 takes the rest. */
    @Test
    void remainingSlackDistributionSharesOneAreaDistributionFails() {
        compare("remaining slack, one area saturates", () -> {
            Network network = remainingSlackDistributionNetwork();
            network.getGenerator("g1").setMinP(70.3);
            return network;
        }, p -> {
            remainingSlackDistributionParameters(p);
            OpenLoadFlowParameters.get(p).setAreaInterchangePMaxMismatch(1);
        }, true);
    }

    /** Both areas saturate: OLF's FAIL behaviour fails the run, which the device leaves to the CPU. */
    @Test
    void remainingSlackDistributionSharesAllAreasDistributionFails() {
        compare("remaining slack, every area saturates", () -> {
            Network network = remainingSlackDistributionNetwork();
            network.getGenerator("g1").setMinP(70.2);
            network.getGenerator("gen3").setMinP(39.9);
            return network;
        }, GpuAreaInterchangeControlTest::remainingSlackDistributionParameters, false);
    }

    /** One area that cannot reach its target: LEAVE_ON_SLACK_BUS leaves the rest on the slack bus and stays on the device;
     *  the other behaviours are OLF's decision, made on the CPU. */
    @ParameterizedTest(name = "{0}")
    @EnumSource(OpenLoadFlowParameters.SlackDistributionFailureBehavior.class)
    void slackDistributionFailureBehaviors(OpenLoadFlowParameters.SlackDistributionFailureBehavior behavior) {
        Supplier<Network> network = () -> {
            Network n = MultiAreaNetworkFactory.createOneAreaBase();
            n.getGenerator("g1").setMinP(90);                   // it would have to go down to 70 MW to meet the target
            return n;
        };
        Consumer<LoadFlowParameters> customize = p -> OpenLoadFlowParameters.get(p).setSlackDistributionFailureBehavior(behavior);
        if (behavior == OpenLoadFlowParameters.SlackDistributionFailureBehavior.THROW) {
            LoadFlowParameters gpuParameters = parameters(customize);
            OpenLoadFlowParameters.get(gpuParameters).setAcSolverType(GpuNewtonRaphsonFactory.NAME);
            LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));
            CompletionException thrown = assertThrows(CompletionException.class, () -> runner.run(network.get(), gpuParameters));
            assertEquals("Failed to distribute active power mismatch", thrown.getCause().getMessage());
            return;
        }
        compare("one area, " + behavior, network, customize, behavior == OpenLoadFlowParameters.SlackDistributionFailureBehavior.LEAVE_ON_SLACK_BUS);
    }

    @ParameterizedTest(name = "{0}")
    @EnumSource(value = LoadFlowParameters.BalanceType.class, names = {"PROPORTIONAL_TO_GENERATION_P", "PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN"})
    void balanceTypes(LoadFlowParameters.BalanceType balanceType) {
        compare("two areas, X node, " + balanceType, twoAreas(MultiAreaNetworkFactory::createTwoAreasWithXNode, -40, 40),
                p -> p.setBalanceType(balanceType), true);
    }
}
