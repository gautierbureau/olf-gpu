/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.google.auto.service.AutoService;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.ac.solver.AcSolver;
import com.powsybl.openloadflow.ac.solver.AcSolverParameters;
import com.powsybl.openloadflow.ac.solver.AcSolverFactory;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonFactory;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.network.LfNetwork;

/**
 * Selects the GPU Newton-Raphson ({@link GpuNewtonRaphson}) as the AC solver:
 * {@code OpenLoadFlowParameters.create(parameters).setAcSolverType(GpuNewtonRaphsonFactory.NAME)}.
 * Reuses the Newton-Raphson parameters (max iterations, stopping criteria, ...).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
@AutoService(AcSolverFactory.class)
public class GpuNewtonRaphsonFactory extends NewtonRaphsonFactory {

    public static final String NAME = "GPU_NEWTON_RAPHSON";

    @Override
    public String getName() {
        return NAME;
    }

    /** OLF's Newton-Raphson parameters, plus the run's {@link GpuLoadFlowParameters}. */
    @Override
    public AcSolverParameters createParameters(LoadFlowParameters parameters) {
        NewtonRaphsonParameters nr = (NewtonRaphsonParameters) super.createParameters(parameters);
        GpuNewtonRaphsonParameters gpu = new GpuNewtonRaphsonParameters();
        gpu.setStoppingCriteria(nr.getStoppingCriteria())
                .setMaxIterations(nr.getMaxIterations())
                .setStateVectorScalingMode(nr.getStateVectorScalingMode())
                .setLineSearchStateVectorScalingMaxIteration(nr.getLineSearchStateVectorScalingMaxIteration())
                .setLineSearchStateVectorScalingStepFold(nr.getLineSearchStateVectorScalingStepFold())
                .setMaxVoltageChangeStateVectorScalingMaxDv(nr.getMaxVoltageChangeStateVectorScalingMaxDv())
                .setMaxVoltageChangeStateVectorScalingMaxDphi(nr.getMaxVoltageChangeStateVectorScalingMaxDphi())
                .setAlwaysUpdateNetwork(nr.isAlwaysUpdateNetwork());
        return gpu.setGpuParameters(GpuLoadFlowParameters.get(parameters));
    }

    @Override
    public AcSolver create(LfNetwork network, AcLoadFlowParameters parameters, EquationSystem<AcVariableType, AcEquationType> equationSystem,
                           JacobianMatrix<AcVariableType, AcEquationType> j, TargetVector<AcVariableType, AcEquationType> targetVector,
                           EquationVector<AcVariableType, AcEquationType> equationVector) {
        // A batch of one is still a batch: the device driver runs OLF's outer loops for it as it does for a
        // 600-scenario chunk, with the bands of the run's OWN configured loops, so it decides at OLF's thresholds.
        double[] d = GpuNewtonRaphson.driverSettings(parameters);
        // NOTE: the loops the driver takes over must ALSO stop running in OLF's engine, or they run
        // twice — the device distributes into its own targets while the network still holds the
        // pre-distribution ones, so OLF's DistributedSlack measures the amount just distributed as a
        // fresh mismatch. Removing them from parameters.getOuterLoops() HERE is the wrong layer: the
        // GPU SA derives its own batch configuration (batchDistributedSlack, the qualification gates)
        // from that same list, so the run stopped batching entirely (share 100% -> 0%). The suppression
        // belongs in the engine (GpuDriverOuterLoopEngine, see createLoadFlowEngine), not in the
        // shared parameters object.
        return new GpuNewtonRaphson(network, (NewtonRaphsonParameters) parameters.getAcSolverParameters(), equationSystem,
                j, targetVector, equationVector, parameters.isDetailedReport(),
                d[0], d[1], (int) d[2], (int) d[3], parameters.getOuterLoops(), parameters);
    }

    /** The device driver's engine: a load flow with this solver is a batch of one, outer loops included. */
    @Override
    public AcloadFlowEngine createLoadFlowEngine(AcLoadFlowContext context) {
        return new GpuDriverOuterLoopEngine(context);
    }
}
