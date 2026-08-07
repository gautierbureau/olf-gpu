/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson;
import org.junit.jupiter.api.extension.AfterEachCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

/**
 * Per-test GPU fallback audit: reports, per test, how many solves silently bailed out of the GPU path
 * onto the CPU Newton-Raphson.
 *
 * <p>Why this exists: a GPU test that only compares against CPU results CANNOT detect that the GPU
 * never ran. CPU-identical results are the fallback CONTRACT, the fallback warning goes to a logger
 * the test logback config attaches no appender to, and {@code olf.gpu.cpuFallback=false} only covers
 * throws reaching that one catch. {@code GpuPhaseControlTest} passed for a whole session while the
 * GPU never executed once.
 *
 * <p>This extension used to ALSO reset the batched solver's JVM-global config before each test, because
 * that config leaked between a provider-driven run and a test driving the solver directly (the pegase
 * full-suite failure — see {@code BatchConfig}). That leak is now gone
 * by construction: the config is a per-call {@code BatchConfig}, so there is nothing to reset.
 *
 * <p>Auto-registered via {@code META-INF/services/org.junit.jupiter.api.extension.Extension}, with
 * autodetection enabled for surefire in the pom, so it applies to every test. The audit covers only
 * the SEQUENTIAL inner solver — the batched SA path has its own per-scenario CPU fallback (status −2),
 * which this does not see.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuTestIsolationExtension implements BeforeEachCallback, AfterEachCallback {

    private long before;

    @Override
    public void beforeEach(ExtensionContext context) {
        before = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get();
    }

    @Override
    public void afterEach(ExtensionContext context) {
        long fallbacks = GpuNewtonRaphson.CPU_FALLBACK_COUNT.get() - before;
        if (fallbacks > 0) {
            System.err.printf("GPUAUDIT %s#%s fallbacks=%d%n",
                    context.getRequiredTestClass().getSimpleName(),
                    context.getRequiredTestMethod().getName(), fallbacks);
        }
    }
}
