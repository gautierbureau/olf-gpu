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

import java.util.function.Supplier;

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
class GpuLoadFlowTest {

    static {
        GpuTestPaths.init();
    }

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
    private void compareWithParams(Supplier<Network> networkSupplier, String label,
                                   java.util.function.Consumer<LoadFlowParameters> customize, double tolKv) {
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
