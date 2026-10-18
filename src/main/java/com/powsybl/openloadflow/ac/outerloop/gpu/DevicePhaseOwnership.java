/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.outerloop.gpu.ContinuousPhaseControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteShuntVoltageControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.DiscreteTransformerVoltageControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.IncrementalFlowControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.IncrementalShuntVoltageControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.IncrementalTransformerVoltageControl;
import com.powsybl.openloadflow.ac.outerloop.gpu.SecondaryVoltageControlLoop;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSkips;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The one rule both routes apply to decide whether the device OWNS an outer loop: all of it, or none.
 *
 * <p>A phase builder walks OLF's controllers and skips the ones it cannot express. An EMPTY list is
 * benign — the kernel declines and the base load flow's host-pause slot runs OLF's own loop, while the
 * batched security analysis refuses the run. A PARTIAL list is the dangerous one: the kernel takes the
 * loop, the pause slot is cleared, and the controllers the builder skipped are silently dropped, so the
 * device drives a loop on a subset of what OLF drives and every equation still converges.
 *
 * <p>{@link GpuBatchSkips} counts those skips; this turns the count into the decision. It is here, and
 * not in either caller, because the base load flow and the security analysis kept drifting apart on
 * exactly this kind of question — the secondary voltage control phase ran on one route and paused on
 * the other for a whole release.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DevicePhaseOwnership {

    private static final Logger LOGGER = LoggerFactory.getLogger(DevicePhaseOwnership.class);

    private DevicePhaseOwnership() {
    }

    /**
     * Hand back any loop whose representation is incomplete: clear the kernel's tables for it and
     * restore its host-pause bit, so OLF's own loop runs at its own slot instead of the device running
     * part of it. Returns the corrected mask.
     */
    public static int enforceComplete(long handle, int hostMask) {
        int mask = hostMask;
        if (!GpuBatchSkips.complete(GpuBatchSkips.Slot.TVC)) {
            IncrementalTransformerVoltageControl.uploadToDevice(handle, null, 1);
            DiscreteTransformerVoltageControl.uploadToDevice(handle, null, 0, false);
            DiscreteTransformerVoltageControl.uploadFreeze(handle, null);
            mask |= DeviceLoopKind.hostMaskBit(DeviceLoopKind.HOST_LOOP_TVC);
        }
        if (!GpuBatchSkips.complete(GpuBatchSkips.Slot.SHUNT)) {
            IncrementalShuntVoltageControl.uploadToDevice(handle, null, 3);
            DiscreteShuntVoltageControl.uploadToDevice(handle, null, true);
            mask |= DeviceLoopKind.hostMaskBit(DeviceLoopKind.HOST_LOOP_SHUNT);
        }
        if (!GpuBatchSkips.complete(GpuBatchSkips.Slot.PHASE) || !GpuBatchSkips.complete(GpuBatchSkips.Slot.RPC)) {
            IncrementalFlowControl.uploadToDevice(handle, null);
            ContinuousPhaseControl.uploadToDevice(handle, null);
            mask |= DeviceLoopKind.hostMaskBit(DeviceLoopKind.HOST_LOOP_PHASE);
            mask |= DeviceLoopKind.hostMaskBit(DeviceLoopKind.HOST_LOOP_TRPC);
        }
        if (!GpuBatchSkips.complete(GpuBatchSkips.Slot.SVC)) {
            SecondaryVoltageControlLoop.uploadToDevice(handle, null);
            mask |= 1;                                        // bit 0 = HOST_LOOP_SVC
        }
        if (mask != hostMask) {
            LOGGER.warn("GPU base load flow: handing loops back to OLF's engine, the phase builders could not "
                    + "represent every controller ({})", GpuBatchSkips.summary());
        }
        return mask;
    }

    /**
     * The batched security analysis has no host-pause slot for anything but secondary voltage control,
     * so a partially represented loop there can only mean the whole run goes to the CPU.
     */
    public static boolean batchable() {
        if (GpuBatchSkips.summary().isEmpty()) {
            return true;
        }
        LOGGER.warn("GPU batched SA: routing the run to the CPU, the phase builders could not represent every "
                + "controller ({}) and the device would drive those loops on a SUBSET of what OLF drives",
                GpuBatchSkips.summary());
        return false;
    }
}
