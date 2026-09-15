/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sensi;

import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.dc.gpu.GpuDcSolver;
import com.powsybl.sensitivity.SensitivityAnalysis;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * OLF's own {@code DcSensitivityAnalysisContingenciesTest}, rerun against the GPU sensitivity provider. Every
 * expected value in it was written for the CPU engine, so this is the parity gate for the device
 * sensitivity path: same factors, same contingencies, same tolerances, OLF's assertions unchanged.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuDcSensitivityAnalysisContingenciesTest extends DcSensitivityAnalysisContingenciesTest {

    GpuDcSensitivityAnalysisContingenciesTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @Override
    @BeforeEach
    public void setUp() throws IOException {
        super.setUp();
        assumeTrue(GpuDcSolver.isAvailable(), "GPU native library not available");
        sensiProvider = new GpuSensitivityAnalysisProvider(commonTestConfig.matrixFactory());
        sensiRunner = new SensitivityAnalysis.Runner(sensiProvider);
    }

    /**
     * Skipped, and NOT because of the device: this one writes a debug dump into the platform
     * configuration directory, which this module does not vendor. Verified by running it against the
     * STOCK provider from here, where it fails the same way ("Cannot write to debug directory as no
     * configuration directory has been defined").
     */
    @Override
    @Test
    void testDebug() {
        assumeTrue(false, "needs a platform configuration directory; fails identically on the stock provider here");
    }
}
