/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.openloadflow.ac.solver.GpuBatchedSecurityAnalysisSolver;
import com.powsybl.openloadflow.ac.solver.GpuNewtonRaphson;
import org.junit.jupiter.api.extension.AfterEachCallback;
import org.junit.jupiter.api.extension.BeforeEachCallback;
import org.junit.jupiter.api.extension.ExtensionContext;

/**
 * Per-test GPU isolation + fallback audit. Two jobs:
 *
 * <p>1. ISOLATION — resets {@link GpuBatchedSecurityAnalysisSolver}'s JVM-global configuration before
 * each test. That config (dsRejectResidual, maxOuterIterations, maxPqPvSwitch, transformer control,
 * state-vector scaling) is set by the SA PROVIDER on every provider-driven run, but tests that drive the
 * batched solver DIRECTLY never set it and so inherit the previous test's. A run parameterized from the
 * RTE file (slackDistributionFailureBehavior=FAIL) leaves dsRejectResidual set, and the next direct
 * solver use then rejects every scenario it saturates: that is exactly how
 * GpuBatchedPegaseFullLoadflowTest passed alone and failed in a full suite run, all 40 scenarios
 * reporting needsCpuFallback.
 *
 * <p>2. AUDIT — reports, per test, how many solves silently bailed out of the GPU path onto the CPU
 * Newton-Raphson.
 *
 * <p>Why this exists: a GPU test that only compares against CPU results CANNOT detect that the GPU
 * never ran. CPU-identical results are the fallback CONTRACT, the fallback warning goes to a logger
 * the test logback config attaches no appender to, and {@code olf.gpu.cpuFallback=false} only covers
 * throws reaching that one catch. {@code GpuPhaseControlTest} passed for a whole session while the
 * GPU never executed once.
 *
 * <p>Auto-registered via {@code META-INF/services/org.junit.jupiter.api.extension.Extension}, with
 * autodetection enabled for surefire in the pom, so both jobs apply to every test. The audit covers only
 * the SEQUENTIAL inner solver — the batched SA path has its own per-scenario CPU fallback (status −2),
 * which this does not see.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuTestIsolationExtension implements BeforeEachCallback, AfterEachCallback {

    private long before;

    @Override
    public void beforeEach(ExtensionContext context) {
        GpuBatchedSecurityAnalysisSolver.resetGlobalConfig();
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
