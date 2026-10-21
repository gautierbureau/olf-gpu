/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.commons.PowsyblException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The one rule both routes apply to decide whether the device OWNS an outer loop: all of it, or none.
 *
 * <p>A phase builder walks OLF's controllers and skips the ones it cannot express. A PARTIAL list is the dangerous
 * one: the kernel takes the loop and the controllers the builder skipped are silently dropped, so the device drives
 * a loop on a subset of what OLF drives and every equation still converges.
 *
 * <p>{@link GpuBatchSkips} counts those skips; this turns the count into the decision: the batched security analysis
 * sends the whole run to the CPU, the base load flow's solve goes to the CPU Newton-Raphson, where OLF's own loops
 * run. It is here, and not in either caller, because the base load flow and the security analysis kept drifting
 * apart on exactly this kind of question — the secondary voltage control phase ran on one route and paused on the
 * other for a whole release.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DevicePhaseOwnership {

    private static final Logger LOGGER = LoggerFactory.getLogger(DevicePhaseOwnership.class);

    private DevicePhaseOwnership() {
    }

    /**
     * The base load flow: a loop whose representation is incomplete fails the device solve, which goes to the CPU
     * Newton-Raphson with OLF's own loops, as {@link #batchable()} sends the security analysis run to the CPU.
     */
    public static void requireComplete() {
        if (!GpuBatchSkips.summary().isEmpty()) {
            throw new PowsyblException("GPU driver: the phase builders could not represent every controller ("
                    + GpuBatchSkips.summary() + ") and the device would drive those loops on a SUBSET of what OLF drives; "
                    + "the solve goes to the CPU");
        }
    }

    /**
     * The batched security analysis: a partially represented loop sends the whole run to the CPU.
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
