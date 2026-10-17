/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Network;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.ac.AcTargetVector;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.equations.StateVector;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Verifies the {@code convEpsPerEq} plumbing on the batched solver: when set {@code > 0}, the device inner
 * Newton stops on OLF's EXACT criterion ({@code DefaultNewtonRaphsonStoppingCriteria}:
 * {@code ‖F‖₂ < sqrt(convEpsPerEq² · n)}) instead of the device's infinity-norm floor — so a full GPU SA can
 * be launched with the SAME stopping decision as the CPU loadflow, for an apples-to-apples GPU-vs-CPU timing
 * comparison.
 *
 * <p>Proof that the criterion is actually honored end-to-end: the loose OLF criterion (convEpsPerEq=1e-4, the
 * OLF default) must stop at a LOOSER final mismatch and in NO MORE iterations than the tight one
 * (convEpsPerEq=1e-12), and the tight one must reach machine precision.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchedOlfCriterionTest {

    private static final int MAX_ITER = 40;
    private static final double TOL = 1e-12;
    private static final String LINE_1_2 = "L1-2-1";


    @Test
    void convEpsPerEqSelectsOlfStoppingCriterion() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU batched native lib not available (build it with native/build-gpu.sh + cuDSS)");

        LfNetwork net = load(IeeeCdfNetworkFactory.create14()).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(net).create();
        AcSolverUtil.initStateVector(net, es, new UniformValueVoltageInitializer());
        StateVector sv = es.getStateVector();
        AcTargetVector target = new AcTargetVector(net, es);
        EquationVector<AcVariableType, AcEquationType> eq = new EquationVector<>(es);
        newton(es, target, eq, sv);
        double[] baseState = sv.get().clone();

        List<BatchContingency> conts = List.of(new BatchContingency(LINE_1_2, net.getBranchById(LINE_1_2)));

        BatchScenarioResult loose = solve(net, baseState, conts, es, target, 1e-4);   // OLF default criterion
        BatchScenarioResult tight = solve(net, baseState, conts, es, target, 1e-12);  // tight OLF criterion

        System.out.printf("convEpsPerEq=1e-4 : iters=%d ||F||inf=%.3e converged=%b%n",
                loose.iterations(), loose.finalMismatch(), loose.hasConverged());
        System.out.printf("convEpsPerEq=1e-12: iters=%d ||F||inf=%.3e converged=%b%n",
                tight.iterations(), tight.finalMismatch(), tight.hasConverged());

        assertTrue(loose.hasConverged() && tight.hasConverged(), "both criterion runs must converge");
        // The tight criterion reaches machine precision (the true root).
        assertTrue(tight.finalMismatch() < 1e-8, "the tight criterion must converge to the true root (||F||inf="
                + tight.finalMismatch() + ")");
        // CORRECTED behavior (2026-08-29): OLF re-solves only after an UNSTABLE outer-loop check — a STABLE
        // phase transition re-CHECKS against the same state and never re-runs Newton. The device driver now
        // matches that exactly (the solvePass skip). Under it, the loose (1e-4) run legitimately retires at
        // its criterion — one warm step from the outage lands under the 2-norm threshold
        // sqrt(convEpsPerEq^2 * n) — instead of being pushed to the root by a spurious extra forced
        // iteration on the DS->RL phase transition (the pre-fix behavior this assertion used to encode:
        // it made the loose and tight runs coincide at machine precision, which the CPU never does).
        assertTrue(loose.finalMismatch() > tight.finalMismatch(),
                "the loose criterion retires at its threshold, NOT at the root (loose=" + loose.finalMismatch()
                        + " tight=" + tight.finalMismatch() + ")");
        assertTrue(loose.iterations() < tight.iterations(),
                "the looser criterion must stop in FEWER iterations (loose=" + loose.iterations()
                        + " tight=" + tight.iterations() + ")");
    }

    private static BatchScenarioResult solve(LfNetwork net, double[] baseState, List<BatchContingency> conts,
                                             EquationSystem<AcVariableType, AcEquationType> es,
                                             AcTargetVector target, double convEpsPerEq) {
        List<BatchScenarioResult> r = GpuBatchedSecurityAnalysisSolver.solveBatchN1(net, baseState, conts, es, target,
                new NewtonRaphsonParameters().setMaxIterations(MAX_ITER), 1e-10, true, true,
                GpuBatchedSecurityAnalysisSolver.DEFAULT_REACTIVE_LIMIT_TOL,
                GpuBatchedSecurityAnalysisSolver.DEFAULT_SLACK_DIST_TOL, convEpsPerEq, ReportNode.NO_OP);
        BatchScenarioResult res = r.get(0);
        assumeTrue(!res.needsCpuFallback(), "L1-2-1 must batch on the device");
        return res;
    }

    private static List<LfNetwork> load(Network iidm) {
        return Networks.load(iidm, new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector()));
    }

    private static void newton(EquationSystem<AcVariableType, AcEquationType> es,
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
    }

    private static double maxAbs(double[] a) {
        double m = 0;
        for (double v : a) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
    }
}
