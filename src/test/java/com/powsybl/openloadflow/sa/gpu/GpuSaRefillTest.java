/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.action.Action;
import com.powsybl.action.LoadActionBuilder;
import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.GeneratorContingency;
import com.powsybl.contingency.strategy.OperatorStrategy;
import com.powsybl.ieeecdf.converter.IeeeCdfNetworkFactory;
import com.powsybl.iidm.network.Load;
import com.powsybl.iidm.network.Network;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchedSecurityAnalysisSolver;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-batched security analysis against the CPU with the bank refilled from the queue while its other scenarios still
 * solve: a bank narrower than the contingencies, each slot handed back for the next scenario as soon as its own ends.
 * Every scenario must still match the CPU, its strategy pairs too, and the device must solve the scenarios it solves
 * when the bank holds them all: a scenario that found its slot's previous occupant's state, or whose install disturbed
 * a scenario still solving, would differ from the CPU or be given up on by the device - and a scenario the device gives
 * up on is replayed on the CPU, which the comparison alone would not see.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaRefillTest extends AbstractGpuSaTest {

    GpuSaRefillTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** A bank of 2: generator and branch outages interleaved, so a slot holds a generator loss (distributed slack
     *  overrides, a loss pre-distribution) and then a branch outage, installed while the other slot's scenario solves. */
    @Test
    void eachSlotRefilledAsSoonAsItsScenarioEndsMatchesCpu() {
        assertRefilledScenariosMatchCpu(2);
    }

    /** A bank of 3: each new scenario goes into a slot an earlier one used while two others solve. */
    @Test
    void scenariosInstalledWhileOthersSolveMatchCpu() {
        assertRefilledScenariosMatchCpu(3);
    }

    private void assertRefilledScenariosMatchCpu(int bankSize) {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU lib required");
        Network network = IeeeCdfNetworkFactory.create14();
        List<Contingency> contingencies = List.of(
                new Contingency("B2-G", new GeneratorContingency("B2-G")),
                new Contingency("B3-G", new GeneratorContingency("B3-G")),
                new Contingency("L1-2-1", new BranchContingency("L1-2-1")),
                new Contingency("B6-G", new GeneratorContingency("B6-G")),
                new Contingency("L2-3-1", new BranchContingency("L2-3-1")),
                new Contingency("B8-G", new GeneratorContingency("B8-G")));
        Load load12 = network.getLoadStream()
                .filter(l -> "B12".equals(l.getTerminal().getBusBreakerView().getBus().getId()))
                .findFirst().orElseThrow();
        List<Action> actions = List.of(
                new LoadActionBuilder().withId("shed12-half").withLoadId(load12.getId())
                        .withRelativeValue(false).withActivePowerValue(load12.getP0() / 2).build(),
                new LoadActionBuilder().withId("shed12-quarter").withLoadId(load12.getId())
                        .withRelativeValue(false).withActivePowerValue(load12.getP0() * 3 / 4).build());
        List<OperatorStrategy> strategies = new ArrayList<>(strategiesOf("L2-3-1", "shed12-half"));
        strategies.addAll(strategiesOf("B3-G", "shed12-quarter"));
        int refills;
        try {
            // a bank that holds every scenario: nothing to refill, the scenarios the device solves
            assertStrategyMatchesCpu(network, contingencies, strategies, actions, -1, true);
            Set<String> onDevice = new HashSet<>(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_IDS);
            assertTrue(onDevice.size() >= 5, "the device must solve most scenarios: " + onDevice.size());
            GpuBatchContextCache.forceChunkSizeForTests(bankSize);
            GpuBatchedSecurityAnalysisSolver.refillAtForTests(1);
            assertStrategyMatchesCpu(network, contingencies, strategies, actions, -1, true);
            assertEquals(onDevice, new HashSet<>(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_IDS),
                    "the device must solve the same scenarios in the refilled bank");
            refills = GpuBatchedSecurityAnalysisSolver.refillsForTests();
        } finally {
            GpuBatchedSecurityAnalysisSolver.refillAtForTests(null);
            GpuBatchContextCache.forceChunkSizeForTests(null);
        }
        assertTrue(refills >= 2, "scenarios must be installed while others still solve: " + refills);
    }
}
