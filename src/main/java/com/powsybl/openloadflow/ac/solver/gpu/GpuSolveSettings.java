/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.outerloop.gpu.IncrementalTransformerVoltageControl;
import com.powsybl.openloadflow.ac.solver.DefaultNewtonRaphsonStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;

/**
 * What the run's OLF configuration means for the device: the outer-loop bands read from the run's own loops, the
 * convergence epsilon of its stopping criteria, and the native host-loop kind of an outer loop.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuSolveSettings {

    private GpuSolveSettings() {
    }

    /** Read the bands out of the run's OWN configured loops, so the device decides at OLF's threshold. */
    static double[] driverSettings(com.powsybl.openloadflow.ac.AcLoadFlowParameters p) {
        double slackTol = Double.MAX_VALUE;
        double rlTol = 0;
        int maxSwitch = 0;
        for (var ol : p.getOuterLoops()) {
            if (ol instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop ds) {
                slackTol = ds.getSlackBusPMaxMismatch() / com.powsybl.openloadflow.util.PerUnit.SB;
            } else if (ol instanceof com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop rl) {
                // OLF builds the loop with its EFFECTIVE band, already in per unit
                // (AbstractAcOuterLoopConfig: convEpsPerEq under UNIFORM_CRITERIA, maxReactivePowerMismatch / SB
                // under PER_EQUATION_TYPE_CRITERIA), the same value GpuSaRunScope derives for the batched SA.
                // Dividing it by SB again made the per-equation band 100 times too tight.
                rlTol = rl.getMaxReactivePowerMismatch();
                maxSwitch = rl.getMaxPqPvSwitch();
            }
        }
        return new double[] {slackTol, rlTol, p.getMaxOuterLoopIterations(), maxSwitch};
    }

    static double deviceConvEpsPerEq(NewtonRaphsonParameters parameters) {
        if (parameters.getStoppingCriteria() instanceof DefaultNewtonRaphsonStoppingCriteria criteria) {
            return criteria.getConvEpsPerEq();
        }
        return 0.0;
    }

    /** Map an OLF outer loop name to the native {@code LOOP_HOST_*} kind, or -1 if it is an
     *  in-kernel driver phase (DS, RL, HVDC, Monitoring) or unknown. */
    public static int loopKindFor(String name) {
        return switch (name) {
            case "SecondaryVoltageControl" -> GpuAcNewtonSolver.HOST_LOOP_SVC;
            case "PhaseControl", "IncrementalPhaseControl" -> GpuAcNewtonSolver.HOST_LOOP_PHASE;
            case "TransformerVoltageControl", "SimpleTransformerVoltageControl",
                 "IncrementalTransformerVoltageControl" -> GpuAcNewtonSolver.HOST_LOOP_TVC;
            case "IncrementalTransformerReactivePowerControl" -> GpuAcNewtonSolver.HOST_LOOP_TRPC;
            case "ShuntVoltageControl", "IncrementalShuntVoltageControl" -> GpuAcNewtonSolver.HOST_LOOP_SHUNT;
            default -> -1;
        };
    }
}
