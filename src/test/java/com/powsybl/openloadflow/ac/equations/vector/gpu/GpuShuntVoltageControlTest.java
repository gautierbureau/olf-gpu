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
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.ShuntNetworkFactory;
import com.powsybl.openloadflow.network.VoltageControl;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Continuous SHUNT voltage control on the GPU: the SHUNT_B susceptance is a Newton state variable, and the
 * controlled bus's reactive balance is paired with that SHUNT_B column (its BUS_V row takes the v−target
 * identity) — the exact analogue of the transformer BRANCH_RHO1 case. The full Newton runs on the GPU behind
 * one JNI call ({@link GpuAcNewtonSolver}); OLF's own {@link JacobianMatrix}-driven Newton on the SAME equation
 * system is the reference — if the GPU mis-placed the b column, the b row, or the moved Q row, the states diverge.
 *
 * <p>Skipped unless the native lib is available (built by native/build-gpu.sh with cuDSS).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuShuntVoltageControlTest {

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

    /** OLF's own Newton (JacobianMatrix over the SAME equation system) — the reference. */
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

    @Test
    void shuntContinuousVoltageControl() {
        Network iidm = ShuntNetworkFactory.create();

        LfNetworkParameters params = new LfNetworkParameters()
                .setSlackBusSelector(new FirstSlackBusSelector())
                .setShuntVoltageControl(true);
        LfNetwork network = Networks.load(iidm, params).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();

        // enable the CONTINUOUS shunt voltage control (as ShuntVoltageControlOuterLoop.initialize does):
        // SHUNT_B becomes a free Newton variable, BUS_TARGET_V activates at the controlled bus, and the
        // controlled bus's reactive balance moves onto the SHUNT_B row.
        for (LfShunt shunt : network.<LfShunt>getControllerElements(VoltageControl.Type.SHUNT)) {
            shunt.setVoltageControlEnabled(true);
        }

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
                d.drRow(), d.drCol(), d.drCoef(), d.dbRow(), d.dbCol(), d.dbCoef());
        int iters = (int) out[n];
        double norm = out[n + 1];

        double[] xRef = cpuReference(es, target, eq, sv);    // mutates sv AFTER the GPU ran on x0

        double worst = 0;
        for (int i = 0; i < n; i++) {
            worst = Math.max(worst, Math.abs(out[i] - xRef[i]));
        }
        System.out.printf("GPU full Newton [shunt continuous VC]: %d sh, n=%d, %d iters, "
                + "||F||inf %.2e, max|x_gpu - x_olf| = %.2e%n",
                d.getShuntCount(), n, iters, norm, worst);
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
