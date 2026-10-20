/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * How the device computes each distributed-slack participant's factor ({@code distributedSlackFactorMode}). OLF's
 * GenerationActivePowerDistributionStep.ParticipationType has four values; the device only needs to tell
 * REMAINING_MARGIN, whose factor depends on the direction and on what the participant already took, from every
 * other balance type, whose factor is fixed for the run. Mirrored by
 * native/cuda/common/codes/gpu_distributed_slack_factor_mode.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuDistributedSlackFactorMode {

    /** The factor is fixed for the run (distributedSlackBusFactor): MAX, TARGET, PARTICIPATION_FACTOR and the load balance types. */
    public static final int STATIC = 0;
    /** OLF's ParticipationType.REMAINING_MARGIN: the factor is the room left in the direction of the mismatch. */
    public static final int REMAINING_MARGIN = 1;

    private GpuDistributedSlackFactorMode() {
    }

    public static int of(boolean remainingMargin) {
        return remainingMargin ? REMAINING_MARGIN : STATIC;
    }
}
