/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlow;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.LoadFlowResult;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.OpenLoadFlowProvider;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Decisive comparison: does the real CPU AC LoadFlow "converge" the stressed outage LINE-26-4231 by
 * driving the slack mismatch to ~0, or by HITTING the outer-loop iteration cap and leaving a large
 * residual mismatch on the slack bus (inner solve still healthy)? The hybrid batch prototype crawls DS
 * at ~0.05pu/round and then diverges the inner solve past ~round 24 — this probe tells us whether CPU
 * simply STOPS earlier (cap = maxOuterLoopIterations, default 20) with the inner solve still healthy.
 */
class GpuCpuOuterLoopTrajectoryProbe {

    private static final Path CASES = GpuTestPaths.caseDir();
    private static final String CASE = "case9241pegase.xiidm.gz";


    @Test
    void cpuOuterLoopTrajectory() throws Exception {
        assumeTrue(Files.exists(CASES.resolve(CASE)), CASE + " not found");
        assumeTrue(Files.exists(Path.of("rte-loadflow-default-parameters.json")), "rte params not found");

        String branchId = System.getenv().getOrDefault("OLF_GPU_DISC_CTG", "LINE-26-4231");
        Network net = Network.read(CASES.resolve(CASE));
        Branch<?> br = net.getBranch(branchId);
        br.getTerminal1().disconnect();
        br.getTerminal2().disconnect();

        LoadFlowParameters lfp = JsonLoadFlowParameters.read(Path.of("rte-loadflow-default-parameters.json"));
        if ("1".equals(System.getenv("OLF_GPU_NO_DIST_SLACK"))) {
            lfp.setDistributedSlack(false);   // slack absorbs the entire raw imbalance -> reveals the true contingency mismatch
        }
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.get(lfp);
        System.out.printf("%n=== CPU LF %s | maxOuterLoopIterations=%d slackBusPMaxMismatch=%.4f MW distSlack=%b%n",
                branchId, olfp.getMaxOuterLoopIterations(), olfp.getSlackBusPMaxMismatch(),
                lfp.isDistributedSlack());

        LoadFlowResult r = new LoadFlow.Runner(new OpenLoadFlowProvider(new SparseMatrixFactory())).run(net, lfp);
        System.out.printf("  fullyConverged=%b status=%s%n", r.isFullyConverged(), r.getComponentResults().isEmpty()
                ? "n/a" : r.getComponentResults().get(0).getStatus());
        for (LoadFlowResult.ComponentResult cr : r.getComponentResults()) {
            System.out.printf("  component %d: status=%s iterationCount=%d distributedActivePower=%.4f MW slackResults=%d%n",
                    cr.getConnectedComponentNum(), cr.getStatus(), cr.getIterationCount(),
                    cr.getDistributedActivePower(), cr.getSlackBusResults().size());
            for (LoadFlowResult.SlackBusResult sbr : cr.getSlackBusResults()) {
                System.out.printf("    slackBus %s : activePowerMismatch=%.4f MW%n",
                        sbr.getId(), sbr.getActivePowerMismatch());
            }
        }
    }
}
