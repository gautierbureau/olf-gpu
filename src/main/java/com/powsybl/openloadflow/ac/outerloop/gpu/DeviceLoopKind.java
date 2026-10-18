/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

/**
 * The outer-loop kinds the device driver can leave to the host: the constants match the native
 * {@code DeviceLoopKind} enum, and the host-loop mask passed to {@code GpuAcNewtonSolver.setBatchHostLoop} has one
 * bit per kind from {@link #HOST_LOOP_SVC} on.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DeviceLoopKind {

    public static final int HOST_LOOP_SVC = 5;
    public static final int HOST_LOOP_PHASE = 6;
    public static final int HOST_LOOP_TVC = 7;
    public static final int HOST_LOOP_TRPC = 8;
    public static final int HOST_LOOP_SHUNT = 9;

    private DeviceLoopKind() {
    }

    /** The host-loop mask bit of a {@code HOST_LOOP_*} kind (bit 0 = SecondaryVoltageControl). */
    public static int hostMaskBit(int kind) {
        return 1 << (kind - HOST_LOOP_SVC);
    }

    /** Map an OLF outer loop name to the native {@code LOOP_HOST_*} kind, or -1 if it is an
     *  in-kernel driver phase (DS, RL, HVDC, Monitoring) or unknown. */
    public static int loopKindFor(String name) {
        return switch (name) {
            case "SecondaryVoltageControl" -> HOST_LOOP_SVC;
            case "PhaseControl", "IncrementalPhaseControl" -> HOST_LOOP_PHASE;
            case "TransformerVoltageControl", "SimpleTransformerVoltageControl",
                 "IncrementalTransformerVoltageControl" -> HOST_LOOP_TVC;
            case "IncrementalTransformerReactivePowerControl" -> HOST_LOOP_TRPC;
            case "ShuntVoltageControl", "IncrementalShuntVoltageControl" -> HOST_LOOP_SHUNT;
            default -> -1;
        };
    }
}
