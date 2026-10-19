/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.commons.config.PlatformConfig;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.ac.AcTargetVector;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuLoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuMatchingAlgorithm;
import com.powsybl.openloadflow.ac.solver.gpu.GpuPivotType;
import com.powsybl.openloadflow.ac.solver.gpu.GpuReorderingAlgorithm;
import com.powsybl.openloadflow.ac.solver.gpu.NewtonSettingsNative;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.equations.StateVector;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A batched solve called directly, as the benchmarks and many tests do, runs with the GPU solver parameters of its
 * own {@link BatchConfig}: the ones it is given, or the configuration module's. It never inherits the settings a
 * previous run left on the device.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchedSolverParametersTest {

    private static final String LINE_1_2 = "L1-2-1";

    private static GpuLoadFlowParameters nonDefault() {
        return new GpuLoadFlowParameters()
                .setReorderingAlgorithm(GpuReorderingAlgorithm.NESTED_DISSECTION)
                .setMatchingAlgorithm(GpuMatchingAlgorithm.MAX_DIAG_SUM)
                .setPivotType(GpuPivotType.GLOBAL_ROW)
                .setPivotThreshold(0.25)
                .setIterativeRefinementBaseSteps(2)
                .setIterativeRefinementMaxSteps(5)
                .setIterativeRefinementTolerance(1e-11);
    }

    @AfterEach
    void restoreDefaults() {
        if (GpuAcNewtonSolver.isAvailable()) {
            new GpuLoadFlowParameters().install();
        }
    }

    @Test
    void aDirectSolveInstallsTheParametersItIsGiven() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        assertTrue(solve(BatchConfig.DEVICE_DEFAULTS.withGpuParameters(nonDefault())).hasConverged());
        assertArrayEquals(new double[] {4, 4, 3, 0.25, 2, 5, 1e-11}, NewtonSettingsNative.readLinearSolverSettings());
    }

    @Test
    void aDirectSolveDoesNotInheritAPreviousRunsParameters() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        GpuLoadFlowParameters.load(PlatformConfig.defaultConfig()).install();
        double[] configured = NewtonSettingsNative.readLinearSolverSettings();

        nonDefault().install();                                  // what a previous run left on the device
        assertTrue(solve(BatchConfig.DEVICE_DEFAULTS).hasConverged());
        assertArrayEquals(configured, NewtonSettingsNative.readLinearSolverSettings());
    }

    private static BatchScenarioResult solve(BatchConfig config) {
        LfNetwork net = Networks.load(IeeeCdfNetworkFactory.create14(),
                new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(net).create();
        AcSolverUtil.initStateVector(net, es, new UniformValueVoltageInitializer());
        StateVector sv = es.getStateVector();
        AcTargetVector target = new AcTargetVector(net, es);
        EquationVector<AcVariableType, AcEquationType> eq = new EquationVector<>(es);
        try (JacobianMatrix<AcVariableType, AcEquationType> j = new JacobianMatrix<>(es, new DenseMatrixFactory())) {
            eq.minus(target);
            for (int it = 0; it < 20; it++) {
                double[] dx = eq.getArray().clone();
                j.solveTransposed(dx);
                sv.minus(dx);
                eq.minus(target);
            }
        }
        List<BatchContingency> conts = List.of(new BatchContingency(LINE_1_2, net.getBranchById(LINE_1_2)));
        return GpuBatchedSecurityAnalysisSolver.solveBatchN1(
                BatchSolveRequest.of(net, sv.get().clone(), conts, es, target,
                        new NewtonRaphsonParameters().setMaxIterations(40), ReportNode.NO_OP)
                        .withConfig(config)).get(0);
    }
}
