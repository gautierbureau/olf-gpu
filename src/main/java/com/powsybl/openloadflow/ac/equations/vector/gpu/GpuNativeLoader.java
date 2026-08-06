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

import java.io.File;
import java.util.regex.Pattern;

/**
 * Loads the OLF GPU Newton-Raphson native library (libolfgpu) for {@link GpuAcNewtonSolver}.
 *
 * <p>libolfgpu is OUR wrapper (MPL) and is bundled in the jar, built location-independent
 * (no RPATH); what is NOT redistributable is NVIDIA's cuDSS, which the wrapper only
 * references dynamically ({@code NEEDED libcudss.so.0}). At load time this therefore
 * PRELOADS the NVIDIA chain (libcublasLt → libcublas → libcudss) from the user's install —
 * {@code olf.cudss.path} system property or {@code CUDSS_ROOT} env for cuDSS,
 * {@code olf.cuda.path} property or {@code CUDA_HOME}/{@code CUDA_PATH}/{@code CUDA_TOOLKIT}
 * env for the toolkit libs — so the dynamic linker resolves the bundled wrapper's
 * dependency from the already-loaded sonames, with no RPATH and no LD_LIBRARY_PATH. If
 * cuDSS is already resolvable (ldconfig, LD_LIBRARY_PATH, or a local RPATH-embedded build),
 * the preload is simply unnecessary and load succeeds anyway.
 *
 * <p>The {@code olf.gpu.lib} system property overrides the classpath lookup with an
 * explicit path (dev builds).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuNativeLoader {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuNativeLoader.class);

    private GpuNativeLoader() {
    }

    static boolean load() {
        String path = System.getProperty("olf.gpu.lib");
        if (path != null) {
            try {
                System.load(path);
                LOGGER.info("OLF GPU Newton-Raphson library loaded from olf.gpu.lib: {}", path);
                return true;
            } catch (Throwable t) {
                LOGGER.warn("Failed to load OLF GPU Newton-Raphson library from olf.gpu.lib={}: {}", path, t.toString());
                return false;
            }
        }
        preloadNvidiaChain();
        try {
            NativeLoader.loadLibrary("olfgpu");
            LOGGER.info("OLF GPU Newton-Raphson library loaded from classpath (natives/<arch>/libolfgpu)");
            return true;
        } catch (Throwable t) {
            LOGGER.debug("OLF GPU Newton-Raphson library not available: {}", t.toString());
            return false;
        }
    }

    /**
     * Best-effort preload of libolfgpu's dynamic NVIDIA dependencies, in dependency
     * order (libcublasLt → libcublas → libcudss), from the configured install dirs. A
     * library that is missing here may still resolve through the system loader paths,
     * so failures are only logged — the libolfgpu load is the real verdict.
     */
    private static void preloadNvidiaChain() {
        File cuda = firstExistingDir(
                System.getProperty("olf.cuda.path"),
                subdir(System.getenv("CUDA_HOME"), "lib64"), subdir(System.getenv("CUDA_HOME"), "lib"),
                subdir(System.getenv("CUDA_PATH"), "lib64"), subdir(System.getenv("CUDA_PATH"), "lib"),
                subdir(System.getenv("CUDA_TOOLKIT"), "lib64"), subdir(System.getenv("CUDA_TOOLKIT"), "lib"));
        File cudss = firstExistingDir(
                System.getProperty("olf.cudss.path"),
                subdir(System.getenv("CUDSS_ROOT"), "lib"));
        preloadOne("libcublasLt", cuda, cudss);
        preloadOne("libcublas", cuda, cudss);
        preloadOne("libcudss", cudss, cuda);
    }

    private static String subdir(String root, String sub) {
        return root == null ? null : root + File.separator + sub;
    }

    private static File firstExistingDir(String... paths) {
        for (String p : paths) {
            if (p != null) {
                File d = new File(p);
                if (d.isDirectory()) {
                    return d;
                }
            }
        }
        return null;
    }

    /** Load {@code <base>.so.<major>} (the soname form) from the first dir that has it. */
    private static void preloadOne(String base, File... dirs) {
        Pattern soname = Pattern.compile(Pattern.quote(base) + "\\.so\\.\\d+");
        for (File dir : dirs) {
            if (dir == null) {
                continue;
            }
            File[] candidates = dir.listFiles((d, name) -> soname.matcher(name).matches());
            File lib = candidates != null && candidates.length > 0 ? candidates[0]
                    : new File(dir, base + ".so");                  // unversioned symlink fallback
            if (lib.isFile()) {
                try {
                    System.load(lib.getAbsolutePath());
                    LOGGER.debug("Preloaded {} for libolfgpu", lib);
                    return;
                } catch (Throwable t) {
                    LOGGER.debug("Preload of {} failed: {}", lib, t.toString());
                }
            }
        }
    }
}
