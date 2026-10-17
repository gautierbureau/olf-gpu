/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import java.nio.file.Path;

/**
 * Where the large external benchmark cases live, for the GPU benchmark tests.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuTestPaths {

    private GpuTestPaths() {
    }

    /**
     * Directory holding the large external benchmark cases (e.g. {@code case9241pegase}),
     * configurable via {@code -Dolf.gpu.caseDir} or the {@code OLF_GPU_CASE_DIR} env var and
     * defaulting to {@code target/gpu-cases}. The perf/benchmark tests skip when the case
     * file is absent, so no case data ships with the repo.
     */
    public static Path caseDir() {
        String dir = System.getProperty("olf.gpu.caseDir");
        if (dir == null) {
            dir = System.getenv().getOrDefault("OLF_GPU_CASE_DIR", "target/gpu-cases");
        }
        return Path.of(dir);
    }
}
