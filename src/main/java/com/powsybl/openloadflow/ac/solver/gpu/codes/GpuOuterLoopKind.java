/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.codes;

/**
 * The outer loops the device driver runs, named after OLF's loops. Mirrored by
 * native/cuda/common/codes/gpu_outer_loop_kind.cuh.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuOuterLoopKind {

    /** No loop: no loop triggered the re-solve. */
    public static final int NONE = -1;
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
    /** IncrementalTransformerVoltageControl. */
    public static final int INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL = 5;
    /** IncrementalShuntVoltageControl. */
    public static final int INCREMENTAL_SHUNT_VOLTAGE_CONTROL = 6;
    /** IncrementalPhaseControl (CONTROLLER + LIMITER). */
    public static final int INCREMENTAL_PHASE_CONTROL = 7;
    /** IncrementalTransformerReactivePowerControl. */
    public static final int INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL = 8;
    /** TransformerVoltageControl (AFTER_GENERATOR) and SimpleTransformerVoltageControl (WITH_GENERATOR). */
    public static final int TRANSFORMER_VOLTAGE_CONTROL = 9;
    /** ShuntVoltageControl (WITH_GENERATOR). */
    public static final int SHUNT_VOLTAGE_CONTROL = 10;
    /** SecondaryVoltageControl. */
    public static final int SECONDARY_VOLTAGE_CONTROL = 11;
    /** AreaInterchangeControl. */
    public static final int AREA_INTERCHANGE_CONTROL = 12;
    /** number of kinds. */
    public static final int COUNT = 13;

    private GpuOuterLoopKind() {
    }
}
