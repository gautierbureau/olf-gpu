/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu.dc;

import com.powsybl.computation.local.LocalComputationManager;
import com.powsybl.iidm.network.ImportConfig;
import com.powsybl.iidm.network.Network;

import java.nio.file.Path;
import java.util.Properties;

/**
 * The DC probes' case files, read as the AC comparisons read them: a reactive capability curve with a point whose max Q
 * is below its min Q is repaired by the importer (powsybl-core 7.4 rejects such a curve otherwise; network.xiidm's
 * battery JPXU has one). Every arm of a comparison reads the same network.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class DcCases {

    private DcCases() {
    }

    static Network read(Path file) {
        Properties parameters = new Properties();
        parameters.setProperty("iidm.import.repair-invalid-reactive-curve-limits", "true");
        return Network.read(file, LocalComputationManager.getDefault(), ImportConfig.load(), parameters);
    }
}
