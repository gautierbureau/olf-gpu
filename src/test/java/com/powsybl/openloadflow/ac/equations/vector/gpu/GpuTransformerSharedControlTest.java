/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Network;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.ac.AcTargetVector;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.equations.StateVector;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.VoltageControl;
import com.powsybl.openloadflow.network.VoltageControlNetworkFactory;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * RUNG 2 for transformer ratio distribution (DISTR_RHO): several transformers share control of one bus.
 * The pilot's reactive balance moves onto the FREE controller's BRANCH_RHO1 row; each other controller's
 * BRANCH_RHO1 row carries a linear DISTR_RHO equation ((1/count)·Σ r1 − r1_k) that the GPU assembles. OLF's
 * own {@link JacobianMatrix} Newton on the SAME equation system is the reference.
 *
 * <p>Skipped unless the native lib is available (built by native/build-gpu.sh with cuDSS).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuTransformerSharedControlTest {

    private static final int MAX_ITER = 30;
    private static final double TOL = 1e-8;

    static {
        GpuTestPaths.init();
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    private static double[] cpuReference(EquationSystem<AcVariableType, AcEquationType> es,
                                         AcTargetVector target, EquationVector<AcVariableType, AcEquationType> eq,
                                         StateVector sv) {
        try (JacobianMatrix<AcVariableType, AcEquationType> j = new JacobianMatrix<>(es, new DenseMatrixFactory())) {
            eq.minus(target);
            for (int it = 0; it < MAX_ITER && maxAbs(eq.getArray()) >= TOL; it++) {
                double[] dx = eq.getArray().clone();
                j.solveTransposed(dx);
                sv.minus(dx);
                eq.minus(target);
            }
        }
        return sv.get().clone();
    }

    /** A2 (plan-group-member-dynamics): at a control-DISABLED base (the AFTER_GENERATOR shape) the
     *  DISTR_RHO pack must STILL carry the group's all-enabled coupling (existence-keyed — the
     *  e82290f audit found the activation-keyed pack silently empty), the packed rows must exclude
     *  the FREE member (first enabled), and the toggled-ALL-ON activation must extract as: pilot V
     *  row identity, non-free member rho rows mode 1 with the DISTR_RHO targets, the pilot's Q
     *  remapped onto the free member's rho row. Host-only — no device needed. */
    @Test
    void sharedGroupPackAtDisabledBaseAndOnActivation() {
        Network iidm = VoltageControlNetworkFactory.createWithTransformerSharedRemoteControl();
        LfNetworkParameters params = new LfNetworkParameters()
                .setSlackBusSelector(new FirstSlackBusSelector())
                .setTransformerVoltageControl(true);
        LfNetwork network = Networks.load(iidm, params).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        java.util.List<LfBranch> members = network.<LfBranch>getControllerElements(VoltageControl.Type.TRANSFORMER)
                .stream().filter(LfBranch::isConnectedAtBothSides).toList();
        org.junit.jupiter.api.Assertions.assertEquals(2, members.size(), "two-member shared group expected");
        // control DISABLED (the AFTER_GENERATOR base): the pack must still carry the coupling
        members.forEach(b -> b.setVoltageControlEnabled(false));
        GpuAcData d = GpuAcDataExtractor.extract(network, es);
        org.junit.jupiter.api.Assertions.assertEquals(2, d.drRow().length,
                "one non-free member row x 2 columns expected in the DISTR_RHO pack at a disabled base");
        int n = es.getIndex().getSortedVariablesToFind().size();
        int rhoColFree = es.getVariableSet().getVariable(members.get(0).getNum(), AcVariableType.BRANCH_RHO1).getRow();
        int rhoColOther = es.getVariableSet().getVariable(members.get(1).getNum(), AcVariableType.BRANCH_RHO1).getRow();
        for (int k = 0; k < d.drRow().length; k++) {
            org.junit.jupiter.api.Assertions.assertEquals(rhoColOther, d.drRow()[k],
                    "DISTR_RHO rows must sit on the NON-free member's rho column");
        }
        org.junit.jupiter.api.Assertions.assertEquals(-0.5, d.drCoef()[java.util.List.of(d.drCol()[0], d.drCol()[1]).indexOf(rhoColOther)],
                1e-15, "self coefficient 1/2 - 1");
        // toggled ALL-ON: activation must express the group state
        members.forEach(b -> b.setVoltageControlEnabled(true));
        network.fixTransformerVoltageControls();
        GpuAcActivation on = GpuAcActivationExtractor.extractActivation(network, es, n);
        LfBus pilot = members.get(0).getVoltageControl().orElseThrow().getControlledBus();
        int pilotVCol = es.getVariableSet().getVariable(pilot.getNum(), AcVariableType.BUS_V).getRow();
        org.junit.jupiter.api.Assertions.assertEquals(0, on.rowMode()[pilotVCol], "pilot V row = identity (V target)");
        org.junit.jupiter.api.Assertions.assertEquals(1, on.rowMode()[rhoColOther], "non-free rho row = DISTR_RHO power row");
        org.junit.jupiter.api.Assertions.assertEquals(1, on.rowMode()[rhoColFree], "free rho row = relocated pilot Q");
        int pilotQEq = es.getEquation(pilot.getNum(), AcEquationType.BUS_TARGET_Q).orElseThrow().getColumn();
        org.junit.jupiter.api.Assertions.assertEquals(pilotQEq, on.targetMap()[rhoColFree],
                "the pilot's Q target must ride the FREE member's rho row");
        members.forEach(b -> b.setVoltageControlEnabled(false));
    }

    @Test
    void transformerSharedControl() {
        Network iidm = VoltageControlNetworkFactory.createWithTransformerSharedRemoteControl();
        LfNetworkParameters params = new LfNetworkParameters()
                .setSlackBusSelector(new FirstSlackBusSelector())
                .setTransformerVoltageControl(true);
        LfNetwork network = Networks.load(iidm, params).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        for (LfBranch b : network.<LfBranch>getControllerElements(VoltageControl.Type.TRANSFORMER)) {
            if (b.isConnectedAtBothSides()) {
                b.setVoltageControlEnabled(true);
            }
        }
        network.fixTransformerVoltageControls();

        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        StateVector sv = es.getStateVector();
        AcTargetVector target = new AcTargetVector(network, es);
        EquationVector<AcVariableType, AcEquationType> eq = new EquationVector<>(es);
        int n = es.getIndex().getSortedVariablesToFind().size();

        double[] x0 = sv.get().clone();
        GpuAcData d = GpuAcDataExtractor.extract(network, es);
        GpuAcActivation activation = GpuAcActivationExtractor.extractActivation(network, es, n);
        double[] stableTargets = activation.applyTargets(target.getArray());

        double[] out = GpuAcNewtonSolver.solve(x0, stableTargets, activation.rowMode(), MAX_ITER, TOL,
                d.cbIn(), d.cbIdx(), d.obIn(), d.obIdx(), d.shIn(), d.shIdx(), d.hvIn(), d.hvIdx(),
                d.dqElem(), d.dqSide(), d.dqWeight(), d.dqRow(), d.dqKind(),
                d.ziIn(), d.ziIdx(),
                d.drRow(), d.drCol(), d.drCoef(), d.dbRow(), d.dbCol(), d.dbCoef(), d.qrPtr(), d.qrRows(), d.obQAlt(), d.ziQAlt());
        int iters = (int) out[n];
        double norm = out[n + 1];

        double[] xRef = cpuReference(es, target, eq, sv);    // mutates sv AFTER the GPU ran on x0

        double worst = 0;
        for (int i = 0; i < n; i++) {
            worst = Math.max(worst, Math.abs(out[i] - xRef[i]));
        }
        System.out.printf("GPU full Newton [transformer shared control / DISTR_RHO]: n=%d, %d dr contributions, "
                + "%d iters, ||F||inf %.2e, max|x_gpu - x_olf| = %.2e%n",
                n, d.getDistrRhoContributionCount(), iters, norm, worst);
        assertTrue(d.getDistrRhoContributionCount() > 0, "the test must exercise DISTR_RHO contributions");
        assertTrue(norm < TOL, "GPU Newton must converge (||F||inf = " + norm + ")");
        assertTrue(worst < 1e-6, "GPU state must match OLF's Newton (max diff = " + worst + ")");
    }

    private static double maxAbs(double[] a) {
        double m = 0;
        for (double v : a) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
    }
}
