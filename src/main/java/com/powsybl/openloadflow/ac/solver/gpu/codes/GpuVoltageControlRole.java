/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * The role of a reactive-limit entry's bus in its generator voltage control (reactiveLimitVoltageControlRole): a local controller, a
 * member of a remote voltage control, or the controlled bus of a remote control that is also one of its own
 * controllers (pilot-self, docs/gpu-vs-olf.md section 6). A remote member splits in two on OLF's
 * voltageRemoteControlRobustMode rule (ReactiveLimitsOuterLoop: realistic-voltage pins of remote controllers),
 * which exempts a member whose own control is local. Mirrored by
 * native/cuda/common/codes/gpu_voltage_control_role.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuVoltageControlRole {

    /** A local controller (OLF's GeneratorVoltageControl.isLocalControl). */
    public static final int LOCAL = 0;
    /** A member of a remote voltage control that OLF's robust mode rule applies to. */
    public static final int REMOTE = 1;
    /** The controlled bus of a remote control that is also one of its own controllers. */
    public static final int PILOT_SELF = 2;
    /** A member of a remote voltage control that OLF's robust mode rule exempts (its own control is local). */
    public static final int REMOTE_ROBUST_MODE_EXEMPT = 3;

    private GpuVoltageControlRole() {
    }
}
