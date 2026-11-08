/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.google.common.jimfs.Configuration;
import com.google.common.jimfs.Jimfs;
import com.powsybl.commons.config.InMemoryPlatformConfig;
import com.powsybl.commons.config.MapModuleConfig;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuLinearSolveMode;
import com.powsybl.openloadflow.network.EurostagFactory;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.FileSystem;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuLoadFlowParametersTest {

    @Test
    void defaultsAreTheDevicesHistoricSettings() {
        GpuLoadFlowParameters p = new GpuLoadFlowParameters();
        assertEquals(GpuLinearSolver.CUDSS, p.getLinearSolver());
        assertEquals(GpuReorderingAlgorithm.AUTO, p.getReorderingAlgorithm());
        assertEquals(GpuMatchingAlgorithm.AUTO, p.getMatchingAlgorithm());
        assertEquals(GpuPivotType.BY_ORDERING, p.getPivotType());
        assertTrue(Double.isNaN(p.getPivotThreshold()));
        assertEquals(1, p.getIterativeRefinementBaseSteps());
        assertEquals(4, p.getIterativeRefinementMaxSteps());
        assertEquals(1e-12, p.getIterativeRefinementTolerance());
    }

    @Test
    void loadsFromTheConfigurationModule() throws IOException {
        try (FileSystem fs = Jimfs.newFileSystem(Configuration.unix())) {
            InMemoryPlatformConfig config = new InMemoryPlatformConfig(fs);
            MapModuleConfig module = config.createModuleConfig(GpuLoadFlowParameters.MODULE);
            module.setStringProperty(GpuLoadFlowParameters.LINEAR_SOLVER_PARAM_NAME, "BATCHED_LU");
            module.setStringProperty(GpuLoadFlowParameters.REORDERING_ALGORITHM_PARAM_NAME, "NESTED_DISSECTION");
            module.setStringProperty(GpuLoadFlowParameters.PIVOT_TYPE_PARAM_NAME, "GLOBAL_ROW");
            module.setStringProperty(GpuLoadFlowParameters.PIVOT_THRESHOLD_PARAM_NAME, "0.1");
            module.setStringProperty(GpuLoadFlowParameters.ITERATIVE_REFINEMENT_MAX_STEPS_PARAM_NAME, "6");
            GpuLoadFlowParameters p = GpuLoadFlowParameters.load(config);
            assertEquals(GpuLinearSolver.BATCHED_LU, p.getLinearSolver());
            assertEquals(GpuReorderingAlgorithm.NESTED_DISSECTION, p.getReorderingAlgorithm());
            assertEquals(GpuPivotType.GLOBAL_ROW, p.getPivotType());
            assertEquals(0.1, p.getPivotThreshold());
            assertEquals(6, p.getIterativeRefinementMaxSteps());
            assertEquals(GpuMatchingAlgorithm.AUTO, p.getMatchingAlgorithm(), "unset keeps the default");
        }
    }

    /** The linear solver is the mode the batch contexts are built in. */
    @Test
    void linearSolverPicksTheBatchMode() {
        assertEquals(GpuLinearSolveMode.UBATCH, GpuBatchContextCache.resolveBatchMode(new GpuLoadFlowParameters()));
        assertEquals(GpuLinearSolveMode.BATCHED_LU, GpuBatchContextCache.resolveBatchMode(
                new GpuLoadFlowParameters().setLinearSolver(GpuLinearSolver.BATCHED_LU)));
        assertEquals(GpuLinearSolver.BATCHED_LU, GpuLoadFlowParameters.load(java.util.Map.of(
                GpuLoadFlowParameters.LINEAR_SOLVER_PARAM_NAME, "BATCHED_LU")).getLinearSolver());
    }

    @Test
    void rejectsInvalidValues() {
        GpuLoadFlowParameters p = new GpuLoadFlowParameters();
        assertThrows(IllegalArgumentException.class, () -> p.setPivotThreshold(1.5));
        assertThrows(IllegalArgumentException.class, () -> p.setIterativeRefinementBaseSteps(-1));
        assertThrows(IllegalArgumentException.class, () -> p.setIterativeRefinementTolerance(0));
    }

    /** A plain load flow with the GPU solver installs the run's GPU parameters on the device, and converges. */
    @Test
    void aLoadFlowInstallsItsParametersOnTheDevice() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        LoadFlowParameters parameters = new LoadFlowParameters();
        OpenLoadFlowParameters.create(parameters).setAcSolverType(GpuNewtonRaphsonFactory.NAME);
        GpuLoadFlowParameters.create(parameters)
                .setReorderingAlgorithm(GpuReorderingAlgorithm.NESTED_DISSECTION)
                .setBatchedLuOrdering(GpuBatchedLuOrdering.FEWEST_LEVELS)
                .setMatchingAlgorithm(GpuMatchingAlgorithm.MAX_DIAG_SUM)
                .setPivotType(GpuPivotType.GLOBAL_ROW)
                .setPivotThreshold(0.25)
                .setIterativeRefinementBaseSteps(2)
                .setIterativeRefinementMaxSteps(5)
                .setIterativeRefinementTolerance(1e-11);
        Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        try {
            assertTrue(new LoadFlow.Runner(new OpenLoadFlowProvider(new DenseMatrixFactory())).run(network, parameters)
                    .isFullyConverged());
            assertArrayEquals(new double[] {4, 4, 3, 0.25, 2, 5, 1e-11, 2}, NewtonSettingsNative.readLinearSolverSettings());
        } finally {
            new GpuLoadFlowParameters().install();               // the next tests run with the defaults
        }
        assertArrayEquals(new double[] {-1, 6, -1, Double.NaN, 1, 4, 1e-12, 0}, NewtonSettingsNative.readLinearSolverSettings());
    }
}
