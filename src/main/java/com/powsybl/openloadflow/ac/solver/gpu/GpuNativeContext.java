/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

import java.util.concurrent.atomic.AtomicLong;

/**
 * The cached native solver context (superset CSR pattern, device buffers, cuDSS
 * analysis) + the listener deregistration, released when the owning {@link GpuNewtonRaphson}
 * becomes unreachable (the AcSolver SPI has no close hook). Registered as a
 * {@link java.lang.ref.Cleaner} action, so it must NOT reference the solver itself.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuNativeContext implements Runnable {

    final AtomicLong handle = new AtomicLong();

    Runnable removeListeners;

    @Override
    public void run() {
        BatchSolveNative.destroyBatchContext(handle.getAndSet(0));
        if (removeListeners != null) {
            removeListeners.run();                       // dead-flags only — never mutates listener lists
        }
    }
}
