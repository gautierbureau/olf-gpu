/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

/**
 * The GPU AC Newton-Raphson's native library, libolfgpu: loaded once, by the first class that touches this one
 * (every {@code *Native} class does in its static initializer). The library (native/cuda/, JNI entry points in
 * native/cuda/jni/, built by the root CMakeLists, see native/build-gpu.sh) keeps the fixed-pattern CSR Jacobian,
 * the mismatch and the state resident on the device: the generated fused kernels fill J and F each iteration,
 * cuDSS refactorizes and solves, a kernel applies the state update.
 *
 * <p>Distribution: libolfgpu is OUR wrapper (MPL) and is bundled in the jar, built
 * location-independent (no RPATH); what is NOT redistributable is NVIDIA's cuDSS, which
 * the wrapper only references dynamically ({@code NEEDED libcudss.so.0}): the system loader
 * resolves it ({@code LD_LIBRARY_PATH} or ldconfig, see GpuNativeLoader). {@link #isAvailable()}
 * is false when the library or its cuDSS dependency cannot be loaded, so CUDA-free builds are
 * unaffected.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcNewtonSolver {

    private static final boolean LOADED = GpuNativeLoader.load();

    private GpuAcNewtonSolver() {
    }

    public static boolean isAvailable() {
        return LOADED;
    }
}
