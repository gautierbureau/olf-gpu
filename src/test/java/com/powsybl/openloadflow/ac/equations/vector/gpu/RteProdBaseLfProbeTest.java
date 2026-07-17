/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Diagnostic probe: WHY does the case9241pegase BASE load flow fail under the RTE production
 * parameters (rte-loadflow-default-parameters.json)? Prints the per-component status text,
 * iteration count and slack results of one plain LoadFlow run. CPU-only, machine-local.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class RteProdBaseLfProbeTest {

    private static final Path CASES = GpuTestPaths.caseDir();
    private static final String CASE = "case9241pegase.xiidm.gz";

    static {
        GpuTestPaths.init();
    }

    @Test
    void probeBaseLoadFlow() {
        assumeTrue(Files.exists(CASES.resolve(CASE)), CASE + " not found");
        Path paramsFile = Path.of("rte-loadflow-default-parameters.json");
        assumeTrue(Files.exists(paramsFile), "params json not found");
        Network network = Network.read(CASES.resolve(CASE));
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(paramsFile);
        long t0 = System.nanoTime();
        LoadFlowResult r = LoadFlow.run(network, lfp);
        long ms = (System.nanoTime() - t0) / 1_000_000;
        System.out.printf("PROBE base LF: ok=%s | %d ms%n", r.isFullyConverged(), ms);
        for (LoadFlowResult.ComponentResult c : r.getComponentResults()) {
            System.out.printf("PROBE comp cc=%d sc=%d: %s | statusText='%s' | iters=%d | slackDistributed=%.1f MW%n",
                    c.getConnectedComponentNum(), c.getSynchronousComponentNum(), c.getStatus(),
                    c.getStatusText(), c.getIterationCount(), c.getDistributedActivePower());
        }
    }
}
