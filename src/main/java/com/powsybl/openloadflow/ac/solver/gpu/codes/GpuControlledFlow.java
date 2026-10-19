/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * The branch flow a flow-observed controller acts on, which is also its kind (setBatchFlowControllers kind[], the
 * observable of flcFlowAndGrad):
 * <ul>
 *   <li>REACTIVE_POWER: IncrementalTransformerReactivePowerControlOuterLoop (rho taps, closest r1);</li>
 *   <li>ACTIVE_POWER: AcIncrementalPhaseControlOuterLoop, TransformerPhaseControl.Mode.CONTROLLER (a1 taps,
 *       closest a1);</li>
 *   <li>CURRENT: AcIncrementalPhaseControlOuterLoop, TransformerPhaseControl.Mode.LIMITER (a1 taps, one-sided).</li>
 * </ul>
 * The driver selects kinds with a mask of {@code 1 << kind}. Mirrored by
 * native/cuda/jni/common/codes/gpu_controlled_flow.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuControlledFlow {

    public static final int REACTIVE_POWER = 0;
    public static final int ACTIVE_POWER = 1;
    public static final int CURRENT = 2;

    private GpuControlledFlow() {
    }
}
