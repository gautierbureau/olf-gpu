/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * The outer loops the device driver runs, named after OLF's loops. A {@code HOST_*} kind is a loop the driver leaves
 * to OLF: it pauses at the loop's slot and OLF's own loop runs there; the host-loop mask has one bit per host kind
 * ({@link #hostMaskBit}). Mirrored by native/cuda/jni/common/codes/gpu_outer_loop_kind.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuOuterLoopKind {

    /** No loop: no restart sweep start (native), no loop triggered the re-solve, or OLF's loop has no host kind. */
    public static final int NONE = -1;
    /** What readBatchDriverPaused answers when the driver did not pause (a pause is always at a HOST_* kind, so 0
     *  cannot be mistaken for DISTRIBUTED_SLACK there). To be unified with NONE once a test pauses the driver. */
    public static final int NOT_PAUSED = 0;
    /** DistributedSlack. */
    public static final int DISTRIBUTED_SLACK = 0;
    /** FreezingHvdcACEmulation. */
    public static final int FREEZING_HVDC_AC_EMULATION = 1;
    /** AcHvdcAcEmulationLimits. */
    public static final int HVDC_AC_EMULATION_LIMITS = 2;
    /** VoltageMonitoring (MonitoringVoltageOuterLoop). */
    public static final int MONITORING_VOLTAGE = 3;
    /** ReactiveLimits. */
    public static final int REACTIVE_LIMITS = 4;
    /** SecondaryVoltageControl, run by OLF at a driver pause. */
    public static final int HOST_SECONDARY_VOLTAGE_CONTROL = 5;
    /** PhaseControl / IncrementalPhaseControl, run by OLF at a driver pause. */
    public static final int HOST_PHASE_CONTROL = 6;
    /** the transformer voltage control loops, run by OLF at a driver pause. */
    public static final int HOST_TRANSFORMER_VOLTAGE_CONTROL = 7;
    /** IncrementalTransformerReactivePowerControl, run by OLF at a driver pause. */
    public static final int HOST_TRANSFORMER_REACTIVE_POWER_CONTROL = 8;
    /** the shunt voltage control loops, run by OLF at a driver pause. */
    public static final int HOST_SHUNT_VOLTAGE_CONTROL = 9;
    /** IncrementalTransformerVoltageControl. */
    public static final int INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL = 10;
    /** IncrementalShuntVoltageControl. */
    public static final int INCREMENTAL_SHUNT_VOLTAGE_CONTROL = 11;
    /** IncrementalPhaseControl (CONTROLLER + LIMITER). */
    public static final int INCREMENTAL_PHASE_CONTROL = 12;
    /** IncrementalTransformerReactivePowerControl. */
    public static final int INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL = 13;
    /** TransformerVoltageControl (AFTER_GENERATOR) and SimpleTransformerVoltageControl (WITH_GENERATOR). */
    public static final int TRANSFORMER_VOLTAGE_CONTROL = 14;
    /** ShuntVoltageControl (WITH_GENERATOR). */
    public static final int SHUNT_VOLTAGE_CONTROL = 15;
    /** SecondaryVoltageControl. */
    public static final int SECONDARY_VOLTAGE_CONTROL = 16;
    /** number of kinds. */
    public static final int COUNT = 17;

    private GpuOuterLoopKind() {
    }

    /** The host-loop mask bit of a {@code HOST_*} kind (bit 0 = HOST_SECONDARY_VOLTAGE_CONTROL). */
    public static int hostMaskBit(int kind) {
        return 1 << (kind - HOST_SECONDARY_VOLTAGE_CONTROL);
    }

    /** The host kind OLF's loop runs as when the driver leaves it to OLF, from OLF's loop name, or NONE if the loop
     *  is an in-kernel driver phase only (DS, RL, HVDC, monitoring) or unknown. */
    public static int hostKindOf(String olfLoopName) {
        return switch (olfLoopName) {
            case "SecondaryVoltageControl" -> HOST_SECONDARY_VOLTAGE_CONTROL;
            case "PhaseControl", "IncrementalPhaseControl" -> HOST_PHASE_CONTROL;
            case "TransformerVoltageControl", "SimpleTransformerVoltageControl",
                 "IncrementalTransformerVoltageControl" -> HOST_TRANSFORMER_VOLTAGE_CONTROL;
            case "IncrementalTransformerReactivePowerControl" -> HOST_TRANSFORMER_REACTIVE_POWER_CONTROL;
            case "ShuntVoltageControl", "IncrementalShuntVoltageControl" -> HOST_SHUNT_VOLTAGE_CONTROL;
            default -> NONE;
        };
    }
}
