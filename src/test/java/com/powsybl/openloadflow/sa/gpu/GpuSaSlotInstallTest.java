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
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU-batched security analysis against the CPU with the scenarios installed into chosen slots of a bank wider than the
 * chunk: each chunk goes into slots earlier chunks used, at other local positions, the slots it does not take keep what
 * their last scenarios left, and only its own scenarios solve. Every scenario must still match the CPU, its strategy
 * pairs too, and the device must solve the scenarios it solves in a bank of the chunk's size: a seed, a reset or a
 * readback that took a scenario's local slot for its bank slot would hand it another scenario's state, and a scenario
 * the device then gave up on would be replayed on the CPU, which the comparison alone would not see.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuSaSlotInstallTest extends AbstractGpuSaTest {

    GpuSaSlotInstallTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    /** Chunks of 2 scattered over a bank of 5: generator and branch outages interleaved, so a slot holds a generator loss
     *  (distributed slack overrides, a loss pre-distribution) in one chunk and a branch outage in the next. */
    @Test
    void chunksScatteredOverAWiderBankMatchCpu() {
        assertScenariosMatchCpu(2, 3, 3);
    }

    /** Chunks of 1 in a bank of 3: every scenario from the fourth on goes alone into a slot an earlier one used, and
     *  solves while the two other slots keep what their scenarios left. */
    @Test
    void eachScenarioAloneInAUsedSlotMatchesCpu() {
        assertScenariosMatchCpu(1, 2, 6);
    }

    private void assertScenariosMatchCpu(int chunkSize, int extraSlots, int minChunks) {
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
        AtomicInteger chunks = new AtomicInteger();
        GpuBatchContextCache.forceChunkSizeForTests(chunkSize);
        GpuBatchedSecurityAnalysisSolver.InstallLayout scattered = new GpuBatchedSecurityAnalysisSolver.InstallLayout() {
            @Override
            public int bankSize(int size) {
                return size + extraSlots;
            }

            // Chunk c's scenario j in bank slot (2c + 2j) mod bankSize: the bank is odd, so a chunk's slots are distinct,
            // and a slot comes back to a later chunk at another local position.
            @Override
            public int[] bankSlots(int chunk, int size, int bankSize) {
                chunks.incrementAndGet();
                int[] slots = new int[size];
                for (int j = 0; j < size; j++) {
                    slots[j] = (2 * chunk + 2 * j) % bankSize;
                }
                return slots;
            }
        };
        try {
            // the same chunks in a bank of their own size: the scenarios the device solves
            assertStrategyMatchesCpu(network, contingencies, strategies, actions, -1, true);
            Set<String> onDevice = new HashSet<>(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_IDS);
            assertTrue(onDevice.size() >= 5, "the device must solve most scenarios: " + onDevice.size());
            GpuBatchedSecurityAnalysisSolver.installChunksForTests(scattered);
            assertStrategyMatchesCpu(network, contingencies, strategies, actions, -1, true);
            assertEquals(onDevice, new HashSet<>(GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_IDS),
                    "the device must solve the same scenarios in the wider bank");
        } finally {
            GpuBatchedSecurityAnalysisSolver.installChunksForTests(null);
            GpuBatchContextCache.forceChunkSizeForTests(null);
        }
        assertTrue(chunks.get() >= minChunks, "the scenarios must span several chunks laid out in the bank: " + chunks.get());
    }
}
