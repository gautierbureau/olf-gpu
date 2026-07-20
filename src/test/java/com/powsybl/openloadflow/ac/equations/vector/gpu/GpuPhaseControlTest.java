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
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.solver.GpuNewtonRaphsonFactory;
import com.powsybl.openloadflow.network.PhaseControlFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.function.BiConsumer;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Transformer PHASE control (PST) on the sequential GPU inner solver.
 *
 * <p>A REGULATING phase shifter makes OLF derive A1 as a solver variable — {@code isDeriveA1}
 * keys on {@code branch.isPhaseController()} alone, never on the control mode, so BOTH modes
 * create the {@code BRANCH_ALPHA1} column. What differs is which ROW pairs with it:
 * <ul>
 *   <li>INCREMENTAL: the loop never calls {@code setPhaseControlEnabled}, so
 *       {@code BRANCH_TARGET_ALPHA1} (the a1−target identity) stays active for the whole run —
 *       a1 is a variable PINNED to its tap value and the outer loop moves TAPS between solves.
 *       That is the row this test exercises on the GPU.</li>
 *   <li>CONTINUOUS_WITH_DISCRETISATION with control enabled: {@code BRANCH_TARGET_P} on the
 *       CONTROLLED branch instead — a branch-owned power row the GPU packs no family for, so
 *       the extractor throws and the solver falls back to CPU. Asserted here too: the run must
 *       still produce CPU-identical results, never a silently dropped control.</li>
 * </ul>
 *
 * <p>Everything else was already in place before this test: the closed-branch kernel reads a1
 * from the state vector when the branch derives it ({@code cbIdx[8]}) and {@code CB_SCATTER}
 * already emits dp1da1/dq1da1/dp2da1/dq2da1 into the branch-owned alpha column.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuPhaseControlTest {

    static {
        GpuTestPaths.init();
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    /**
     * Runs the same network + parameters on CPU and on the GPU inner solver and asserts the states
     * agree. Also asserts the PST ended on the SAME TAP: a phase control that silently did nothing
     * on the GPU would still match on voltages in a lightly loaded case, but not on the tap.
     */
    private void compare(Supplier<Network> networkSupplier,
                         BiConsumer<LoadFlowParameters, OpenLoadFlowParameters> customizer, String label) {
        LoadFlow.Runner runner = new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory()));

        Network cpuNetwork = networkSupplier.get();
        LoadFlowParameters cpuParameters = new LoadFlowParameters();
        // Same stopping criterion on BOTH arms. With OLF's 1e-4 pu default the CPU stops well inside
        // the basin while the GPU iterates on, and the states differ by ~1e-8 pu (4.5e-3 kV on 400 kV)
        // for reasons that have nothing to do with phase control — tightening beats loosening the
        // assertion, which would hide a real term error of the same magnitude.
        OpenLoadFlowParameters cpuOlf = OpenLoadFlowParameters.create(cpuParameters)
                .setNewtonRaphsonConvEpsPerEq(1e-8);
        customizer.accept(cpuParameters, cpuOlf);
        LoadFlowResult cpuResult = runner.run(cpuNetwork, cpuParameters);
        assertTrue(cpuResult.isFullyConverged(), "CPU reference load flow must converge");

        Network gpuNetwork = networkSupplier.get();
        LoadFlowParameters gpuParameters = new LoadFlowParameters();
        OpenLoadFlowParameters gpuOlf = OpenLoadFlowParameters.create(gpuParameters)
                .setNewtonRaphsonConvEpsPerEq(1e-8)
                .setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        customizer.accept(gpuParameters, gpuOlf);
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
        int cpuTap = cpuNetwork.getTwoWindingsTransformer("PS1").getPhaseTapChanger().getTapPosition();
        int gpuTap = gpuNetwork.getTwoWindingsTransformer("PS1").getPhaseTapChanger().getTapPosition();
        System.out.printf("PST GPU vs CPU [%s]: %d buses, max dV = %.2e kV, max dAngle = %.2e deg, tap %d/%d%n",
                label, compared, worstV, worstA, cpuTap, gpuTap);
        assertTrue(compared > 0, "must compare at least one bus");
        assertEquals(cpuTap, gpuTap, "the PST must end on the same tap (a dropped phase control would not)");
        assertTrue(worstV < 1e-3, "bus voltages must match (max dV = " + worstV + " kV)");
        assertTrue(worstA < 1e-3, "bus angles must match (max dAngle = " + worstA + " deg)");
    }

    @org.junit.jupiter.api.Disabled("does NOT yet exercise the alpha column — see the comment below")
    @Test
    void incrementalActivePowerControl() {
        // UNPROVEN. This test passes, but it does not validate the BRANCH_TARGET_ALPHA1 packing:
        // the PST never acts (tap stays at 1 on both arms even with the regulation value moved far
        // off the natural flow), and GPU-vs-CPU comes out BIT-IDENTICAL (0.00e+00) instead of the
        // ~1e-14 the genuinely GPU-solved no-control case shows. Both facts say the equation system
        // here has no alpha column at all, so the identity row under test is never activated.
        // Setting olf.gpu.cpuFallback=false does not fail it either, which only confirms nothing
        // unsupported is being packed. Before trusting the a1 path, the test must FIRST assert that
        // the built equation system actually contains a BRANCH_ALPHA1 variable (build the LfNetwork +
        // AcEquationSystemCreator directly, as GpuTransformerVoltageControlTest does for rho) and
        // that the tap MOVES; only then is a state comparison meaningful.
        // INCREMENTAL: BRANCH_TARGET_ALPHA1 identity row active all run — the row this change packs.
        compare(GpuPhaseControlTest::regulatingPst,
            (p, olf) -> {
                p.setPhaseShifterRegulationOn(true);
                olf.setPhaseShifterControlMode(OpenLoadFlowParameters.PhaseShifterControlMode.INCREMENTAL);
            }, "incremental");
    }

    @Test
    void continuousWithDiscretisationFallsBackButMatches() {
        // CONTINUOUS: BRANCH_TARGET_P is activated on the controlled branch, the extractor throws,
        // the solver falls back to CPU NR. The point of the assertion is that the RESULT is still
        // CPU-identical — the control is never silently dropped.
        compare(GpuPhaseControlTest::regulatingPst,
            (p, olf) -> {
                p.setPhaseShifterRegulationOn(true);
                olf.setPhaseShifterControlMode(
                        OpenLoadFlowParameters.PhaseShifterControlMode.CONTINUOUS_WITH_DISCRETISATION);
            }, "continuous");
    }

    @Test
    void nonRegulatingPstPacksAsPlainBranch() {
        // Control OFF: no TransformerPhaseControl, no alpha column at all — the network packs as
        // plain branches. This is the shape every run has today (phaseShifterRegulationOn defaults
        // to false and the RTE parameter file does not set it).
        compare(PhaseControlFactory::createNetworkWithT2wt,
                (p, olf) -> p.setPhaseShifterRegulationOn(false), "no control");
    }

    private static Network regulatingPst() {
        Network network = PhaseControlFactory.createNetworkWithT2wt();
        network.getTwoWindingsTransformer("PS1").getPhaseTapChanger()
                .setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL)
                .setRegulationValue(83)
                .setTargetDeadband(1)
                .setRegulating(true);
        return network;
    }
}
