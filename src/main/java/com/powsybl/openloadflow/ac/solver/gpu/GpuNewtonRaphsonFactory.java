/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.google.auto.service.AutoService;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.AcSolver;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop;
import com.powsybl.openloadflow.ac.solver.AcSolverFactory;
import java.util.List;
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

    private static final org.slf4j.Logger LOGGER = org.slf4j.LoggerFactory.getLogger(GpuNewtonRaphsonFactory.class);

    public static final String NAME = "GPU_NEWTON_RAPHSON";

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public AcSolver create(LfNetwork network, AcLoadFlowParameters parameters, EquationSystem<AcVariableType, AcEquationType> equationSystem,
                           JacobianMatrix<AcVariableType, AcEquationType> j, TargetVector<AcVariableType, AcEquationType> targetVector,
                           EquationVector<AcVariableType, AcEquationType> equationVector) {
        // A batch of one is still a batch: if the device driver runs OLF's outer loops for a 600-scenario
        // chunk it runs them for nS == 1 too. olf.gpu.driverOuterLoops hands them over; the bands come
        // from the run's OWN configured loops so the device decides at OLF's thresholds.
        boolean driverLoops = Boolean.getBoolean("olf.gpu.driverOuterLoops")
                || "1".equals(System.getenv("OLF_GPU_DRIVER_OUTER_LOOPS"));
        if (System.getenv("OLF_GPU_DRIVER_OUTER_LOOPS") != null) {
            LOGGER.info("GPU base solver: driverOuterLoops={} (DS/ReactiveLimits move from OLF's engine "
                    + "into the device driver, batch-of-one)", driverLoops);
        }
        double[] d = GpuNewtonRaphson.driverSettings(parameters);
        // NOTE: the loops the driver takes over must ALSO stop running in OLF's engine, or they run
        // twice — the device distributes into its own targets while the network still holds the
        // pre-distribution ones, so OLF's DistributedSlack measures the amount just distributed as a
        // fresh mismatch. Removing them from parameters.getOuterLoops() HERE is the wrong layer: the
        // GPU SA derives its own batch configuration (batchDistributedSlack, the qualification gates)
        // from that same list, so the run stopped batching entirely (share 100% -> 0%). The suppression
        // belongs in the engine the SA builds (AcSecurityAnalysis.createLoadFlowEngine), not in the
        // shared parameters object.
        return new GpuNewtonRaphson(network, (NewtonRaphsonParameters) parameters.getAcSolverParameters(), equationSystem,
                j, targetVector, equationVector, parameters.isDetailedReport(),
                driverLoops, d[0], d[1], (int) d[2], (int) d[3]);
    }
}
