/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysis;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.json.JsonSecurityAnalysisParameters;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSecurityAnalysisProviderTest {

    @Test
    void isFoundByName() {
        assertEquals(GpuSecurityAnalysisProvider.NAME, SecurityAnalysis.find(GpuSecurityAnalysisProvider.NAME).getName());
    }

    /**
     * The JSON form of security analysis parameters writes and reads with this provider registered beside OLF's: it adds
     * no serializer of its own under the name of OLF's extension (a duplicate key for powsybl, which failed every JSON
     * write and read of security analysis parameters).
     */
    @Test
    void securityAnalysisParametersStillWriteAndReadAsJson() {
        SecurityAnalysisParameters parameters = new SecurityAnalysisParameters();
        parameters.addExtension(OpenSecurityAnalysisParameters.class, new OpenSecurityAnalysisParameters().setThreadCount(3));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        JsonSecurityAnalysisParameters.write(parameters, out);
        SecurityAnalysisParameters read = JsonSecurityAnalysisParameters.read(new ByteArrayInputStream(out.toByteArray()));
        assertEquals(3, read.getExtension(OpenSecurityAnalysisParameters.class).getThreadCount());
        assertTrue(new GpuSecurityAnalysisProvider().getSpecificParametersSerializer().isEmpty());
    }
}
