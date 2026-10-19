/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.network.LfNetwork;

import java.util.List;

/**
 * One batched security analysis solve: the base case and its contingencies, and how the device runs them.
 *
 * <ul>
 * <li>{@code tol}: the infinity-norm mismatch floor of the device's inner Newton; {@code tol <= 0} selects the
 * fixed-iteration mode (every scenario runs {@code parameters.getMaxIterations()} steps on the device, with no
 * per-iteration copy back and no convergence retirement).</li>
 * <li>{@code distributedSlack} / {@code reactiveLimits}: the outer loops the device runs, which must be the ones OLF
 * applied to the base case.</li>
 * <li>{@code reactiveLimitTol} / {@code slackDistTol}: the device outer-loop tolerances (pu), the run's
 * maxReactivePowerMismatch and slackBusPMaxMismatch so the device stops at the CPU's outer-loop fixed point.</li>
 * <li>{@code convEpsPerEq}: when {@code > 0}, the device's inner Newton stops on OLF's
 * DefaultNewtonRaphsonStoppingCriteria ({@code |F|2 < sqrt(convEpsPerEq^2 n)}) instead of the {@code tol}
 * floor.</li>
 * <li>{@code config}: the run's outer-loop parameterization.</li>
 * </ul>
 *
 * <p>{@link #of} gives the defaults: the GPU tolerance floor, both outer loops on, OLF's default outer-loop
 * tolerances, the floor stopping criterion and the device's default configuration.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record BatchSolveRequest(LfNetwork network, double[] baseState, List<BatchContingency> contingencies,
                                EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                TargetVector<AcVariableType, AcEquationType> targetVector,
                                NewtonRaphsonParameters parameters, double tol, boolean distributedSlack,
                                boolean reactiveLimits, double reactiveLimitTol, double slackDistTol,
                                double convEpsPerEq, ReportNode reportNode, BatchConfig config) {

    public static BatchSolveRequest of(LfNetwork network, double[] baseState, List<BatchContingency> contingencies,
                                       EquationSystem<AcVariableType, AcEquationType> equationSystem,
                                       TargetVector<AcVariableType, AcEquationType> targetVector,
                                       NewtonRaphsonParameters parameters, ReportNode reportNode) {
        return new BatchSolveRequest(network, baseState, contingencies, equationSystem, targetVector, parameters,
                GpuNewtonRaphson.GPU_TOLERANCE, true, true, GpuBatchedSecurityAnalysisSolver.DEFAULT_REACTIVE_LIMIT_TOL,
                GpuBatchedSecurityAnalysisSolver.DEFAULT_SLACK_DIST_TOL, 0.0, reportNode, BatchConfig.DEVICE_DEFAULTS);
    }

    public BatchSolveRequest withTol(double tol) {
        return new BatchSolveRequest(network, baseState, contingencies, equationSystem, targetVector, parameters, tol,
                distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol, convEpsPerEq, reportNode, config);
    }

    public BatchSolveRequest withOuterLoops(boolean distributedSlack, boolean reactiveLimits) {
        return new BatchSolveRequest(network, baseState, contingencies, equationSystem, targetVector, parameters, tol,
                distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol, convEpsPerEq, reportNode, config);
    }

    public BatchSolveRequest withOuterLoopTolerances(double reactiveLimitTol, double slackDistTol) {
        return new BatchSolveRequest(network, baseState, contingencies, equationSystem, targetVector, parameters, tol,
                distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol, convEpsPerEq, reportNode, config);
    }

    public BatchSolveRequest withConvEpsPerEq(double convEpsPerEq) {
        return new BatchSolveRequest(network, baseState, contingencies, equationSystem, targetVector, parameters, tol,
                distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol, convEpsPerEq, reportNode, config);
    }

    public BatchSolveRequest withConfig(BatchConfig config) {
        return new BatchSolveRequest(network, baseState, contingencies, equationSystem, targetVector, parameters, tol,
                distributedSlack, reactiveLimits, reactiveLimitTol, slackDistTol, convEpsPerEq, reportNode, config);
    }
}
