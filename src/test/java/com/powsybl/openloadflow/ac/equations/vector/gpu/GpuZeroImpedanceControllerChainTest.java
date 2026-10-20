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
import com.powsybl.openloadflow.network.AbstractLoadFlowNetworkFactory;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A zero-impedance cluster whose sections are ALL voltage controllers. rte6515's VL-1113 / VL-132: four
 * busbar sections, a generator on each, merged into one voltage control by the three couplers. The
 * sections behind the pilot have NO impedant branch, so their DISTR_Q rows carry only the couplers'
 * dummy-Q (and the pilot's flows) — never their own V: a structurally ZERO diagonal that the
 * dead-end-stub relocation does not cover (generators), and that the batched cuDSS path (no pivoting,
 * no matching in uniform-batch mode) turns into a wrong Newton STEP. The trajectory still reconverges,
 * which is why it only showed where OLF stops early (BUS-1196_BBS, 5.157e-02 kV at BUS-3462_BBS).
 *
 * <p>So this compares the STEP, not the fixed point: one Newton iteration from the flat start on both
 * arms must agree to machine precision.
 *
 * <pre>
 *   ld0
 *    |
 *   b0 --- l01 --- b1 === c12 === b2 === c23 === b3 === c34 === b4
 *        (x=0.1)   g1            g2            g3            g4
 * </pre>
 */
class GpuZeroImpedanceControllerChainTest extends AbstractLoadFlowNetworkFactory {

    private static final double TOL = 1e-10;

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    private static Network controllerChain() {
        Network network = Network.create("zi-controller-chain", "code");
        Bus b0 = createBus(network, "s", "b0");
        Bus b1 = createBus(network, "s", "b1");
        Bus b2 = createBus(network, "s", "b2");
        Bus b3 = createBus(network, "s", "b3");
        Bus b4 = createBus(network, "s", "b4");
        createLoad(b0, "ld0", 4);
        createGenerator(b1, "g1", 1, 1.0);
        createGenerator(b2, "g2", 1, 1.0);
        createGenerator(b3, "g3", 1, 1.0);
        createGenerator(b4, "g4", 1, 1.0);
        createLine(network, b0, b1, "l01", 0.1);
        createLine(network, b1, b2, "c12", 0.0, 0.0);
        createLine(network, b2, b3, "c23", 0.0, 0.0);
        createLine(network, b3, b4, "c34", 0.0, 0.0);
        return network;
    }

    /** OLF's own Newton, dense: {@code k} steps from the current state vector. */
    private static double[] cpuSteps(EquationSystem<AcVariableType, AcEquationType> es, AcTargetVector target,
                                     EquationVector<AcVariableType, AcEquationType> eq, StateVector sv, int k) {
        try (JacobianMatrix<AcVariableType, AcEquationType> j = new JacobianMatrix<>(es, new DenseMatrixFactory())) {
            for (int it = 0; it < k; it++) {
                eq.minus(target);
                double[] dx = eq.getArray().clone();
                j.solveTransposed(dx);
                sv.minus(dx);
            }
        }
        return sv.get().clone();
    }

    private static double[] gpuSteps(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                     AcTargetVector target, double[] x0, int k, int n) {
        GpuAcData d = GpuAcDataExtractor.extract(network, es);
        GpuAcActivation activation = GpuAcActivationExtractor.extractActivation(network, es, n);
        double[] stableTargets = activation.applyTargets(target.getArray());
        // maxIter = k with a tolerance nothing meets: exactly k steps
        return GpuAcNewtonSolver.solve(x0, stableTargets, activation.rowMode(), k, 0.0,
                d.closedBranch().parameters(), d.closedBranch().indices(), d.openBranch().parameters(),
                d.openBranch().indices(), d.shunt().parameters(), d.shunt().indices(), d.hvdcAcEmulation().parameters(),
                d.hvdcAcEmulation().indices(),
                d.distrQ().element(), d.distrQ().side(), d.distrQ().weight(), d.distrQ().row(), d.distrQ().kind(),
                d.zeroImpedanceBranch().parameters(), d.zeroImpedanceBranch().indices(),
                d.distrRho().row(), d.distrRho().column(), d.distrRho().coefficient(), d.distrShuntB().row(), d.distrShuntB().column(), d.distrShuntB().coefficient(),
                // the group tables + ob/zi alternates travel with the packs: a pilot-self group enrolled
                // in the candidate table (rotation) encodes -2-g in the q alternates, refused without them
                d.reactiveRelocation().groupStart(), d.reactiveRelocation().candidateRows(), d.openBranch().alternateReactiveDestination(), d.zeroImpedanceBranch().alternateReactiveDestination());
    }

    @Test
    void firstNewtonStepMatchesOlfOnAnAllControllerCluster() {
        LfNetwork network = Networks.load(controllerChain(),
                new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        StateVector sv = es.getStateVector();
        AcTargetVector target = new AcTargetVector(network, es);
        EquationVector<AcVariableType, AcEquationType> eq = new EquationVector<>(es);
        int n = es.getIndex().getSortedVariablesToFind().size();
        double[] x0 = sv.get().clone();

        // The fixture must produce the shape: three merged controllers with a DISTR_Q row and no
        // impedant branch — or the test proves nothing.
        long ziOnlyDistrQ = network.getBuses().stream()
                .filter(b -> GpuAcRowIndexing.row(es, b.getNum(), AcEquationType.DISTR_Q) >= 0)
                .filter(b -> b.getBranches().stream().allMatch(br -> br.isZeroImpedance(
                        com.powsybl.openloadflow.network.LoadFlowModel.AC)))
                .count();
        assertEquals(3, ziOnlyDistrQ, "three controller sections must live behind couplers only");

        double[] gpu1 = gpuSteps(network, es, target, x0, 1, n);
        double[] cpu1 = cpuSteps(es, target, eq, sv, 1);
        double worst1 = 0;
        for (int i = 0; i < n; i++) {
            worst1 = Math.max(worst1, Math.abs(gpu1[i] - cpu1[i]));
        }
        System.out.printf("GPU vs OLF first Newton step [all-controller zi cluster]: n=%d, max|dx diff| = %.3e%n",
                n, worst1);
        assertTrue(worst1 < TOL, "the first Newton step must be OLF's step (max diff = " + worst1 + ")");

        // and the fixed point, from the same start, after enough steps for both to converge
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        double[] gpu8 = gpuSteps(network, es, target, x0, 8, n);
        double[] cpu8 = cpuSteps(es, target, eq, sv, 8);
        double worst8 = 0;
        for (int i = 0; i < n; i++) {
            worst8 = Math.max(worst8, Math.abs(gpu8[i] - cpu8[i]));
        }
        assertTrue(worst8 < TOL, "the converged states must agree (max diff = " + worst8 + ")");
    }
}
