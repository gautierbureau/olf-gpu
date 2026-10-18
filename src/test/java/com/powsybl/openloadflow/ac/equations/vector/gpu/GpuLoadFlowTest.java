/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControlAdder;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.HvdcNetworkFactory;
import com.powsybl.openloadflow.network.ShuntNetworkFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * RUNG 3: the GPU Newton-Raphson behind a PLAIN {@code LoadFlow.run} — selected through
 * the AC solver SPI with
 * {@code OpenLoadFlowParameters.create(parameters).setAcSolverType(GpuNewtonRaphsonFactory.NAME)}.
 * Default load-flow parameters (distributed slack ON → the outer loop reruns the solver
 * with adjusted targets, exercising the per-run re-extraction). Each network runs twice
 * — default CPU Newton-Raphson vs GPU — and every bus voltage/angle must agree.
 *
 * Skipped unless the native lib is available (see {@link GpuFullNewtonTest}).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuLoadFlowTest extends com.powsybl.openloadflow.network.AbstractLoadFlowNetworkFactory {

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    private void compare(Supplier<Network> networkSupplier, String label) {
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));

        Network cpuNetwork = networkSupplier.get();
        LoadFlowResult cpuResult = runner.run(cpuNetwork, new LoadFlowParameters());
        assertTrue(cpuResult.isFullyConverged(), "CPU reference load flow must converge");

        Network gpuNetwork = networkSupplier.get();
        LoadFlowParameters gpuParameters = new LoadFlowParameters();
        OpenLoadFlowParameters.create(gpuParameters).setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        LoadFlowResult gpuResult = runner.run(gpuNetwork, gpuParameters);
        assertTrue(gpuResult.isFullyConverged(), "GPU load flow must converge");

        double worstV = 0;
        double worstA = 0;
        int compared = 0;
        for (Bus cpuBus : cpuNetwork.getBusView().getBuses()) {
            Bus gpuBus = gpuNetwork.getBusView().getBus(cpuBus.getId());
            if (gpuBus == null || Double.isNaN(cpuBus.getV()) || Double.isNaN(gpuBus.getV())) {
                continue;
            }
            worstV = Math.max(worstV, Math.abs(cpuBus.getV() - gpuBus.getV()));
            worstA = Math.max(worstA, Math.abs(cpuBus.getAngle() - gpuBus.getAngle()));
            compared++;
        }
        System.out.printf("LoadFlow.run GPU vs CPU [%s]: %d buses, CPU %d iters / GPU %d iters, "
                        + "max dV = %.2e kV, max dAngle = %.2e deg%n",
                label, compared,
                cpuResult.getComponentResults().get(0).getIterationCount(),
                gpuResult.getComponentResults().get(0).getIterationCount(),
                worstV, worstA);
        // With the same OLF stopping criterion wired on both sides, the production LoadFlow.run
        // path should now land on the same operating point up to small floating-point noise.
        assertTrue(compared > 0, "must compare at least one bus");
        assertTrue(worstV < 1e-3, "bus voltages must match (max dV = " + worstV + " kV)");
        assertTrue(worstA < 1e-3, "bus angles must match (max dAngle = " + worstA + " deg)");
        // The distributed active power the result reports (a GPU-driven security analysis base reported 0 —
        // assessment L10; a plain LoadFlow.run keeps OLF's own distributed-slack loop and must match too).
        double cpuDistributed = cpuResult.getComponentResults().get(0).getDistributedActivePower();
        double gpuDistributed = gpuResult.getComponentResults().get(0).getDistributedActivePower();
        assertTrue(Math.abs(cpuDistributed - gpuDistributed) < 1e-3,
                "distributed active power must match: CPU " + cpuDistributed + " MW, GPU " + gpuDistributed + " MW");
    }

    @Test
    void eurostagClosedBranches() {
        compare(() -> EurostagFactory.fix(EurostagTutorialExample1Factory.create()), "eurostag closed-branch");
    }

    @Test
    void eurostagWithOpenBranch() {
        compare(() -> {
            Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
            network.getLine("NHV1_NHV2_1").getTerminal2().disconnect();
            return network;
        }, "eurostag + open branch");
    }

    @Test
    void shuntCompensator() {
        compare(ShuntNetworkFactory::create, "shunt compensator");
    }

    /** Sequential-solver ZIP/exponential loads (the batch-of-1 route runs the vdl fills): the
     *  Eurostag network with a ZIP load solves ON the GPU (fallback counter guarded) and matches
     *  the CPU exactly. */
    @Test
    void zipLoadModel() {
        long before = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        compareWithParams(() -> {
            Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
            network.getVoltageLevel("VLLOAD").newLoad()
                    .setId("ZIPLOAD").setBus("NLOAD").setP0(50).setQ0(30)
                    .newZipModel()
                        .setC0p(0.5).setC0q(0.55)
                        .setC1p(0.3).setC1q(0.35)
                        .setC2p(0.2).setC2q(0.1)
                        .add()
                    .add();
            return network;
        }, "zip load model", p -> OpenLoadFlowParameters.create(p).setUseLoadModel(true), 1e-9);
        org.junit.jupiter.api.Assertions.assertEquals(before,
                com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the ZIP solve must run ON the GPU, not silently fall back");
    }

    /** Sequential-solver PER_EQUATION_TYPE stopping criteria (the batch-of-1 route installs the
     *  per-row thresholds via setBatchRowCriteria): the device stops on the SAME decision as OLF's
     *  PerEquationTypeStoppingCriteria — identical iteration counts and a machine-precision state,
     *  where the old uniform-floor path converged DEEPER and broke exact parity. */
    @Test
    void perEquationTypeCriteria() {
        long before = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        compareWithParams(() -> EurostagFactory.fix(EurostagTutorialExample1Factory.create()),
                "per-equation-type criteria",
                p -> OpenLoadFlowParameters.create(p).setNewtonRaphsonStoppingCriteriaType(
                        com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteriaType.PER_EQUATION_TYPE_CRITERIA),
                1e-9);
        org.junit.jupiter.api.Assertions.assertEquals(before,
                com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the per-eq-criteria solve must run ON the GPU, not silently fall back");
    }

    /** Sequential-solver CONTROLLABLE shunt (SHUNT_B as an in-Newton variable): shunt voltage
     *  control ON in continuous mode — the batch-of-1 route carries the SHUNT_B rows. */
    @Test
    void controllableShunt() {
        long before = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        compareWithParams(ShuntNetworkFactory::create, "controllable shunt",
                p -> p.setShuntCompensatorVoltageControlOn(true), 1e-9);
        org.junit.jupiter.api.Assertions.assertEquals(before,
                com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the controllable-shunt solve must run ON the GPU, not silently fall back");
    }

    /** Sequential-solver MONITORING SVC (stand-by automaton) activating IN THE BASE load flow:
     *  the low-voltage threshold sits above the standing-by voltage, so OLF's VoltageMonitoring
     *  loop activates the automaton mid-run — the activation re-extract must carry the flipped
     *  row modes onto the device (the last cpuBaseOnly guard besides frozen HVDC). */
    @Test
    void monitoringSvcActivatesInBase() {
        long before = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        compareWithParams(() -> {
            Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithStaticVarCompensator();
            com.powsybl.iidm.network.StaticVarCompensator svc1 = network.getStaticVarCompensator("svc1");
            svc1.setVoltageSetpoint(385).setRegulationMode(com.powsybl.iidm.network.StaticVarCompensator.RegulationMode.VOLTAGE)
                    .setRegulating(true);
            svc1.newExtension(com.powsybl.iidm.network.extensions.StandbyAutomatonAdder.class)
                    .withHighVoltageThreshold(400).withLowVoltageThreshold(393)
                    .withLowVoltageSetpoint(394).withHighVoltageSetpoint(395)
                    .withB0(-0.001f).withStandbyStatus(true).add();
            return network;
        }, "monitoring svc base activation",
                p -> com.powsybl.openloadflow.OpenLoadFlowParameters.create(p).setSvcVoltageMonitoring(true), 1e-9);
        org.junit.jupiter.api.Assertions.assertEquals(before,
                com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the monitoring-SVC base solve must run ON the GPU, not silently fall back");
        // prove the automaton genuinely ACTIVATED mid-base (else the test is hollow): activation
        // makes the SVC hold its bus at the LOW setpoint (394), far from the standing-by state
        Network check = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithStaticVarCompensator();
        com.powsybl.iidm.network.StaticVarCompensator svc = check.getStaticVarCompensator("svc1");
        svc.setVoltageSetpoint(385).setRegulationMode(com.powsybl.iidm.network.StaticVarCompensator.RegulationMode.VOLTAGE)
                .setRegulating(true);
        svc.newExtension(com.powsybl.iidm.network.extensions.StandbyAutomatonAdder.class)
                .withHighVoltageThreshold(400).withLowVoltageThreshold(393)
                .withLowVoltageSetpoint(394).withHighVoltageSetpoint(395)
                .withB0(-0.001f).withStandbyStatus(true).add();
        LoadFlowParameters cp = new LoadFlowParameters();
        com.powsybl.openloadflow.OpenLoadFlowParameters.create(cp).setSvcVoltageMonitoring(true)
                .setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory())).run(check, cp);
        org.junit.jupiter.api.Assertions.assertEquals(394.0,
                svc.getTerminal().getBusView().getBus().getV(), 1e-6,
                "the stand-by automaton must have ACTIVATED at the low setpoint in the base");
    }

    /** compare() with a parameter customizer applied to BOTH arms and a configurable tolerance. */
    /** OLF_GPU_DBG_RL: OLF's own reactive-limit decisions (TRACE) as CPU_RL lines — the same hook the rte6515
     *  harness installs, so a fixture's CPU arm can be read decision by decision. */
    private static void installRlTrace() {
        if (System.getenv("OLF_GPU_DBG_RL") == null) {
            return;
        }
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
        ch.qos.logback.classic.encoder.PatternLayoutEncoder enc = new ch.qos.logback.classic.encoder.PatternLayoutEncoder();
        enc.setContext(root.getLoggerContext());
        enc.setPattern("CPU_RL %msg%n");
        enc.start();
        ch.qos.logback.core.ConsoleAppender<ch.qos.logback.classic.spi.ILoggingEvent> ca = new ch.qos.logback.core.ConsoleAppender<>();
        ca.setContext(root.getLoggerContext());
        ca.setEncoder(enc);
        ca.start();
        for (String n : new String[] {"com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop",
                                      "com.powsybl.openloadflow.ac.AcloadFlowEngine"}) {
            ch.qos.logback.classic.Logger lg = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger(n);
            lg.setLevel(ch.qos.logback.classic.Level.TRACE);
            lg.setAdditive(false);
            lg.addAppender(ca);
        }
    }

    private void compareWithParams(Supplier<Network> networkSupplier, String label,
                                   java.util.function.Consumer<LoadFlowParameters> customize, double tolKv) {
        installRlTrace();
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));

        Network cpuNetwork = networkSupplier.get();
        LoadFlowParameters cpuParameters = new LoadFlowParameters();
        customize.accept(cpuParameters);
        LoadFlowResult cpuResult = runner.run(cpuNetwork, cpuParameters);
        assertTrue(cpuResult.isFullyConverged(), "CPU reference load flow must converge");

        Network gpuNetwork = networkSupplier.get();
        LoadFlowParameters gpuParameters = new LoadFlowParameters();
        customize.accept(gpuParameters);
        OpenLoadFlowParameters.get(gpuParameters).setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        LoadFlowResult gpuResult = runner.run(gpuNetwork, gpuParameters);
        assertTrue(gpuResult.isFullyConverged(), "GPU load flow must converge");

        org.junit.jupiter.api.Assertions.assertEquals(
                cpuResult.getComponentResults().get(0).getIterationCount(),
                gpuResult.getComponentResults().get(0).getIterationCount(),
                "same stopping criteria must stop on the same iteration");
        double worstV = 0;
        int compared = 0;
        for (Bus cpuBus : cpuNetwork.getBusView().getBuses()) {
            Bus gpuBus = gpuNetwork.getBusView().getBus(cpuBus.getId());
            if (gpuBus == null || Double.isNaN(cpuBus.getV()) || Double.isNaN(gpuBus.getV())) {
                continue;
            }
            worstV = Math.max(worstV, Math.abs(cpuBus.getV() - gpuBus.getV()));
            compared++;
        }
        System.out.printf("LoadFlow.run GPU vs CPU [%s]: %d buses, CPU %d iters / GPU %d iters, max dV = %.2e kV%n",
                label, compared, cpuResult.getComponentResults().get(0).getIterationCount(),
                gpuResult.getComponentResults().get(0).getIterationCount(), worstV);
        assertTrue(compared > 0, "must compare at least one bus");
        assertTrue(worstV < tolKv, "bus voltages must match (max dV = " + worstV + " kV)");
    }

    /** LF-level startWithFrozenACEmulation: the base runs the FreezingHvdcACEmulation loop — the
     *  first solve has every AC-emu HVDC FROZEN (coupling terms deactivated on the CPU, injections
     *  in the target vector), the unfreeze re-solve is linear. The batch-of-1 route mirrors the
     *  per-solve statuses with the hv element disabled bits (was the LAST cpu-base-only condition
     *  in the SA provider). Same iterations, no fallback. */
    @Test
    void hvdcFrozenStartMatchesCpu() {
        long before = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        compareWithParams(HvdcNetworkFactory::createHvdcInAcEmulationInSymetricNetwork, "hvdc frozen start",
                p -> {
                    p.setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
                    OpenLoadFlowParameters.create(p).setStartWithFrozenACEmulation(true);
                }, 1e-9);
        org.junit.jupiter.api.Assertions.assertEquals(before,
                com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the frozen-start solves must stay on the GPU (no CPU fallback)");
    }

    /**
     * Frozen start on a network whose angles are KNOWN (a second load flow on a solved network): OLF's
     * FreezingHvdcACEmulation.initialize freezes the emulation at that flow and its first check unfreezes it and
     * re-solves. On a fresh network (NaN angles, {@link #hvdcFrozenStartMatchesCpu}) it freezes nothing. The driver
     * starts from that decision: it neither freezes what OLF left linear nor keeps OLF's frozen emulation frozen
     * as if a contingency had outaged it — each of which cost iterations and 3e-10 against the CPU.
     */
    @Test
    void hvdcFrozenStartOnSolvedNetworkMatchesCpu() {
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));
        Network cpu = HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        Network gpu = HvdcNetworkFactory.createHvdcInAcEmulationInSymetricNetwork();
        assertTrue(runner.run(cpu, frozenStartParameters(false)).isFullyConverged());
        assertTrue(runner.run(gpu, frozenStartParameters(true)).isFullyConverged());
        for (Network network : List.of(cpu, gpu)) {                   // something to solve, from known angles
            var load = network.getLoadStream().findFirst().orElseThrow();
            load.setP0(load.getP0() * 1.1);
        }
        long fallbacksBefore = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        LoadFlowResult cpuResult = runner.run(cpu, frozenStartParameters(false));
        LoadFlowResult gpuResult = runner.run(gpu, frozenStartParameters(true));
        assertEquals(fallbacksBefore, com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the frozen-start solve must stay on the GPU");
        assertEquals(cpuResult.getComponentResults().get(0).getIterationCount(),
                gpuResult.getComponentResults().get(0).getIterationCount(), "OLF's freeze, unfreeze and re-solve");
        for (Bus bus : cpu.getBusView().getBuses()) {
            Bus gpuBus = gpu.getBusView().getBus(bus.getId());
            assertEquals(bus.getV(), gpuBus.getV(), 1e-12, "V @ " + bus.getId());
            assertEquals(bus.getAngle(), gpuBus.getAngle(), 1e-12, "angle @ " + bus.getId());
        }
    }

    private static LoadFlowParameters frozenStartParameters(boolean gpu) {
        LoadFlowParameters p = new LoadFlowParameters().setHvdcAcEmulation(true).setDistributedSlack(true).setUseReactiveLimits(true);
        OpenLoadFlowParameters ext = OpenLoadFlowParameters.create(p).setStartWithFrozenACEmulation(true);
        if (gpu) {
            ext.setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        }
        return p;
    }

    /** A REMOTE generator voltage-control group whose CONTROLLED (pilot) bus is ALSO one of its
     *  controllers — a generator regulating its own bus merged with remote controllers. The device
     *  maps each equation onto its variable's row, so the pilot's BUS_TARGET_V and the
     *  pilot-member's DISTR_Q both claimed that bus's V row: one silently overwrote the other and
     *  the group's reactive sharing was never enforced, while the FREE controller's row (which
     *  hosts the pilot's reactive equation in every other form) sat unused. The device then
     *  "converged" on its own norm while OLF's re-test of the same state failed, so the load flow
     *  reported MAX_ITERATION_REACHED with no fallback. Found on a richer-modelling RTE-like
     *  network, where it made the base LF — and therefore every contingency — fail. */
    @Test
    void remoteVoltageControlWithPilotAsController() {
        long before = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        compareWithParams(GpuLoadFlowTest::pilotIsControllerNetwork, "remote VC, pilot is also a controller",
                p -> {
                    p.setDistributedSlack(true).setUseReactiveLimits(true);
                    OpenLoadFlowParameters.create(p).setVoltageRemoteControl(true);
                }, 1e-10);
        org.junit.jupiter.api.Assertions.assertEquals(before,
                com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the pilot-is-controller group must solve ON the GPU (no CPU fallback)");
    }

    /** A zero-impedance chain of generator buses (every bus a controller of the merged group, the pilot
     *  one of them): the LoadFlow path, so the first fill can be read against OLF's f(x0)
     *  (OLF_GPU_F0_TRACE + OLF_GPU_VALS_DUMP_ANY) when the two disagree. */
    @Test
    void zeroImpedanceControllerChain() {
        long before = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        compareWithParams(() -> ziControllerChain(3), "zero-impedance controller chain of 3",
                p -> p.setDistributedSlack(true).setUseReactiveLimits(true), 1e-10);
        org.junit.jupiter.api.Assertions.assertEquals(before,
                com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the chain must solve ON the GPU (no CPU fallback)");
    }

    private static Network ziControllerChain(int couplers) {
        Network network = Network.create("zi-chain-" + couplers, "code");
        com.powsybl.iidm.network.Bus b0 = createBus(network, "s", "b0");
        createLoad(b0, "ld0", couplers, couplers / 2.0);
        com.powsybl.iidm.network.Bus prev = createBus(network, "s", "b1");
        createGenerator(prev, "g1", 1, 1.0);
        createLine(network, b0, prev, "l01", 0.1);
        for (int i = 2; i <= couplers + 1; i++) {
            com.powsybl.iidm.network.Bus b = createBus(network, "s", "b" + i);
            createGenerator(b, "g" + i, 1, 1.0);
            createLine(network, prev, b, "c" + i, 0.0, 0.0);
            prev = b;
        }
        return network;
    }

    /** A pilot-self group whose second controller sits on a dead-end STUB behind a zero-impedance coupler
     *  (rte6515 VL-5997: sections _0 and _1 coupled), and whose stub generator PINS at a non-zero limit in
     *  the base load flow. The stub's V row hosts the coupler's ZERO_V equation; its reactive balance lives
     *  on the coupler's DUMMY_Q row — a pin written on the V row turns "v_stub − v_pilot = 0" into
     *  "= limit − loadQ" (BUS-6088_BBS in the [5000,6000) batch, 30 iterations and a bad solve). */
    @Test
    void pilotSelfGroupWithStubMemberPinning() {
        long before = com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
        compareWithParams(GpuLoadFlowTest::pilotSelfStubMemberNetwork, "pilot-self group, stub member pins",
                p -> {
                    p.setDistributedSlack(true).setUseReactiveLimits(true);
                    OpenLoadFlowParameters.create(p).setVoltageRemoteControl(true);
                }, 1e-10);
        org.junit.jupiter.api.Assertions.assertEquals(before,
                com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.CPU_FALLBACK_COUNT.get(),
                "the stub-member pin must solve ON the GPU (no CPU fallback)");
    }

    private static Network pilotSelfStubMemberNetwork() {
        Network network = Network.create("pilot-self-stub", "code");
        com.powsybl.iidm.network.Bus b0 = createBus(network, "s0", "b0");
        com.powsybl.iidm.network.Bus b1 = createBus(network, "s1", "b1");   // the pilot: its own generator
        com.powsybl.iidm.network.Bus b2 = createBus(network, "s1", "b2");   // the stub: coupled to b1, no other branch
        com.powsybl.iidm.network.Bus b3 = createBus(network, "s0", "b3");
        createGenerator(b0, "g0", 3, 1.0);                                   // slack side
        createLoad(b3, "ld3", 4, 3.0);                                      // a reactive load the group must feed
        createLoad(b0, "ld0", 1, 0.5);
        createLine(network, b0, b1, "l01", 0.1);
        createLine(network, b1, b3, "l13", 0.1);
        createLine(network, b0, b3, "l03", 0.1);
        createLine(network, b1, b2, "c12", 0.0, 0.0);                        // the zero-impedance coupler
        createGenerator(b1, "g1", 1, 1.02);                                  // pilot controls its own bus
        network.getGenerator("g1").newMinMaxReactiveLimits().setMinQ(-100).setMaxQ(100).add();
        var g2 = b2.getVoltageLevel().newGenerator()
                .setId("g2").setBus("b2").setConnectableBus("b2")
                .setTargetP(1).setMinP(0).setMaxP(10).setVoltageRegulatorOn(true)
                .setTargetV(1.02 * b1.getVoltageLevel().getNominalV())
                .setRegulatingTerminal(network.getGenerator("g1").getTerminal())
                .add();
        g2.newMinMaxReactiveLimits().setMinQ(-0.5).setMaxQ(0.5).add();      // pins at maxQ = 0.005 pu, not 0
        return network;
    }

    /** g1/g2/g3 (b1/b2/b3) remotely control b4, plus g4 AT b4 controlling b4 — so the group's
     *  controlled bus is also one of its controllers. */
    private static Network pilotIsControllerNetwork() {
        Network network = com.powsybl.openloadflow.network.VoltageControlNetworkFactory.createWithGeneratorRemoteControl();
        var vl4 = network.getBusBreakerView().getBus("b4").getVoltageLevel();
        vl4.newGenerator().setId("g4").setBus("b4").setConnectableBus("b4")
                .setMinP(0).setMaxP(200).setTargetP(50).setTargetV(413.4)
                .setVoltageRegulatorOn(true).add();
        network.getGenerator("g4").newMinMaxReactiveLimits().setMinQ(-1000).setMaxQ(1000).add();
        return network;
    }

    @Test
    void hvdcAcEmulation() {
        compare(() -> {
            Network network = HvdcNetworkFactory.createWithHvdcInAcEmulation();
            network.getHvdcLine("hvdc34").newExtension(HvdcAngleDroopActivePowerControlAdder.class)
                    .withDroop(180)
                    .withP0(0.f)
                    .withEnabled(true)
                    .add();
            return network;
        }, "hvdc ac emulation");
    }
}
