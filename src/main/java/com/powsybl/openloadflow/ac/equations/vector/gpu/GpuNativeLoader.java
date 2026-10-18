/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import org.scijava.nativelib.NativeLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads the OLF GPU native library (libolfgpu) for {@link GpuAcNewtonSolver}, from the classpath
 * ({@code natives/<arch>/libolfgpu}).
 * <p>libolfgpu is OUR wrapper (MPL) and is bundled in the jar, built location-independent (no RPATH); what is
 * NOT redistributable is NVIDIA's cuDSS and cuBLAS, which the wrapper references dynamically
 * ({@code NEEDED libcudss.so.0}, {@code libcublas.so}). The system loader resolves them like any native
 * dependency: {@code LD_LIBRARY_PATH} or the ldconfig cache must name the cuDSS and CUDA library directories.
 * When they are missing the library does not load and {@link GpuAcNewtonSolver#isAvailable()} is false.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuNativeLoader {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuNativeLoader.class);

    private GpuNativeLoader() {
    }

    static boolean load() {
        try {
            NativeLoader.loadLibrary("olfgpu");
            LOGGER.info("OLF GPU Newton-Raphson library loaded from classpath (natives/<arch>/libolfgpu)");
            return true;
        } catch (Throwable t) {
            LOGGER.debug("OLF GPU Newton-Raphson library not available: {}", t.toString());
            return false;
        }
    }
}
