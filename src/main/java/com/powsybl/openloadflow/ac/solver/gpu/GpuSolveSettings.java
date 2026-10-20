/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

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
        double maxReactivePowerMismatch = 0;
        int maxSwitch = 0;
        for (var outerLoop : p.getOuterLoops()) {
            if (outerLoop instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop distributedSlackOuterLoop) {
                slackTol = distributedSlackOuterLoop.getSlackBusPMaxMismatch() / com.powsybl.openloadflow.util.PerUnit.SB;
            } else if (outerLoop instanceof com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop reactiveLimitsOuterLoop) {
                // OLF builds the loop with its EFFECTIVE band, already in per unit
                // (AbstractAcOuterLoopConfig: convEpsPerEq under UNIFORM_CRITERIA, maxReactivePowerMismatch / SB
                // under PER_EQUATION_TYPE_CRITERIA), the same value GpuSaRunScope derives for the batched SA.
                // Dividing it by SB again made the per-equation band 100 times too tight.
                maxReactivePowerMismatch = reactiveLimitsOuterLoop.getMaxReactivePowerMismatch();
                maxSwitch = reactiveLimitsOuterLoop.getMaxPqPvSwitch();
            }
        }
        return new double[] {slackTol, maxReactivePowerMismatch, p.getMaxOuterLoopIterations(), maxSwitch};
    }

    static double deviceConvEpsPerEq(NewtonRaphsonParameters parameters) {
        if (parameters.getStoppingCriteria() instanceof DefaultNewtonRaphsonStoppingCriteria criteria) {
            return criteria.getConvEpsPerEq();
        }
        return 0.0;
    }

}
