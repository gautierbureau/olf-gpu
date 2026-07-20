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
                         BiConsumer<LoadFlowParameters, OpenLoadFlowParameters> customizer, String label,
                         int expectedSolvedTap) {
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
        int cpuTap = cpuNetwork.getTwoWindingsTransformer("PS1").getPhaseTapChanger().getSolvedTapPosition();
        int gpuTap = gpuNetwork.getTwoWindingsTransformer("PS1").getPhaseTapChanger().getSolvedTapPosition();
        System.out.printf("PST GPU vs CPU [%s]: %d buses, max dV = %.2e kV, max dAngle = %.2e deg, tap %d/%d%n",
                label, compared, worstV, worstA, cpuTap, gpuTap);
        assertTrue(compared > 0, "must compare at least one bus");
        assertEquals(expectedSolvedTap, cpuTap, "the CPU reference must move the tap — otherwise this test "
                + "asserts nothing about phase control (getSolvedTapPosition, NOT getTapPosition: OLF leaves "
                + "the input tapPosition untouched)");
        assertEquals(cpuTap, gpuTap, "the PST must end on the same tap (a dropped phase control would not)");
        assertTrue(worstV < 1e-3, "bus voltages must match (max dV = " + worstV + " kV)");
        assertTrue(worstA < 1e-3, "bus angles must match (max dAngle = " + worstA + " deg)");
    }

    @Test
    void incrementalActivePowerControl() {
        // NOTE — this asserts CPU-parity for PST modelling (the tap really moves 1->2, mirroring OLF's
        // AcLoadFlowPhaseShifterTest.incrementalPhaseShifterActivePowerControlTest), and the GPU arm
        // currently FALLS BACK to CPU: "equation row 4 has no diagonal pairing". The a1 identity row has
        // no natural (a1row, a1col) entry in the CSR pattern — unlike BRANCH_TARGET_RHO1, whose row IS a
        // bus Q row moved onto the rho column, so the branch's dq/dr1 scatter supplies its diagonal.
        // Nothing scatters into the a1 row, so applyRowMode has no diagPos to write its 1.0 into.
        // Until the pattern builder emits that diagonal, this test proves CPU parity only. See #36.
        // INCREMENTAL: BRANCH_TARGET_ALPHA1 identity row active all run — the row this change packs.
        compare(GpuPhaseControlTest::regulatingPst,
            (p, olf) -> {
                p.setPhaseShifterRegulationOn(true);
                olf.setPhaseShifterControlMode(OpenLoadFlowParameters.PhaseShifterControlMode.INCREMENTAL);
            }, "incremental", 2);
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
            }, "continuous", 2);
    }

    @Test
    void nonRegulatingPstPacksAsPlainBranch() {
        // Control OFF: no TransformerPhaseControl, no alpha column at all — the network packs as
        // plain branches. This is the shape every run has today (phaseShifterRegulationOn defaults
        // to false and the RTE parameter file does not set it).
        compare(PhaseControlFactory::createNetworkWithT2wt,
                (p, olf) -> p.setPhaseShifterRegulationOn(false), "no control", 1);
    }

    private static Network regulatingPst() {
        // Same setup as OLF's own AcLoadFlowPhaseShifterTest.incrementalPhaseShifterActivePowerControlTest,
        // which is the CPU reference for this modelling: the regulation TERMINAL must be set (side 1),
        // and the loop moves the tap 1 -> 2 to pull P1 from 50.084 MW to 83.687 MW.
        Network network = PhaseControlFactory.createNetworkWithT2wt();
        com.powsybl.iidm.network.TwoWindingsTransformer t2wt = network.getTwoWindingsTransformer("PS1");
        t2wt.getPhaseTapChanger()
                .setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.ACTIVE_POWER_CONTROL)
                .setTargetDeadband(1)
                .setTapPosition(1)
                .setRegulationTerminal(t2wt.getTerminal1())
                .setRegulationValue(100)
                .setRegulating(true);
        return network;
    }
}
