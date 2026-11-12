/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.BaseOpenBranches;
import com.powsybl.openloadflow.ac.equations.vector.gpu.ClosedBranchPack;
import com.powsybl.openloadflow.ac.equations.vector.gpu.DistrQMembers;
import com.powsybl.openloadflow.ac.equations.vector.gpu.DistrQTerms;
import com.powsybl.openloadflow.ac.equations.vector.gpu.DistributedSlackParticipants;
import com.powsybl.openloadflow.ac.equations.vector.gpu.ElementPack;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
import com.powsybl.openloadflow.ac.equations.vector.gpu.LinearDistributionTerms;
import com.powsybl.openloadflow.ac.equations.vector.gpu.OpenBranchPack;
import com.powsybl.openloadflow.ac.equations.vector.gpu.ReactiveLimitEntries;
import com.powsybl.openloadflow.ac.equations.vector.gpu.RemoteMemberReactiveTerms;
import com.powsybl.openloadflow.ac.equations.vector.gpu.StructureNative;
import com.powsybl.openloadflow.ac.equations.vector.gpu.VoltageRowParameters;
import com.powsybl.openloadflow.ac.equations.vector.gpu.ZeroImpedanceBranchPack;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocationCandidates;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuLinearSolveMode;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * CPU-only checks of the GPU batch chunk-size resolution, which bounds the per-chunk device-memory
 * footprint. A size forced by the test hook is honoured; otherwise the chunk is auto-sized from free device
 * memory by a native query,
 * falling back to the default when no GPU is present. The auto path (and the native nnz /
 * cudaMemGetInfo estimate) is exercised by the device tests.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchedSecurityAnalysisSolverTest {

    private static final GpuAcData EMPTY = new GpuAcData(ClosedBranchPack.empty(), OpenBranchPack.empty(),
            ElementPack.empty(), ElementPack.empty(), DistrQTerms.empty(), DistrQMembers.empty(),
            ZeroImpedanceBranchPack.empty(), LinearDistributionTerms.empty(), LinearDistributionTerms.empty(),
            ReactiveLimitEntries.empty(), RemoteMemberReactiveTerms.empty(), VoltageRowParameters.empty(),
            DistributedSlackParticipants.empty(), BaseOpenBranches.empty(), ElementPack.empty(),
            ReactiveRelocationCandidates.empty());

    @Test
    void explicitCapIsHonouredAndClampedToCount() {
        GpuBatchContextCache.forceChunkSizeForTests(64);
        try {
            assertEquals(64, GpuBatchContextCache.resolveChunkSize(1000, 10, EMPTY, GpuLinearSolveMode.UBATCH));
            assertEquals(10, GpuBatchContextCache.resolveChunkSize(10, 10, EMPTY, GpuLinearSolveMode.UBATCH),
                    "chunk is clamped to the scenario count");
        } finally {
            GpuBatchContextCache.forceChunkSizeForTests(null);
        }
    }

    @Test
    void autoSizeFallsBackToDefaultWhenNoGpu() {
        assumeFalse(GpuAcNewtonSolver.isAvailable(), "auto-size queries the device when a GPU is present");
        // No GPU: recommendBatchSize is unlinked, so resolveChunkSize falls back to the default.
        GpuBatchContextCache.forceChunkSizeForTests(null);
        assertEquals(GpuBatchContextCache.DEFAULT_MAX_BATCH,
                GpuBatchContextCache.resolveChunkSize(1000, 10, EMPTY, GpuLinearSolveMode.UBATCH));
    }

    @Test
    void autoSizedBankIsPricedPerMode() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "the pricing runs on a device structure");
        LfNetwork network = Networks.load(EurostagFactory.fix(EurostagTutorialExample1Factory.create()), new FirstSlackBusSelector()).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        int n = es.getIndex().getSortedVariablesToFind().size();
        GpuAcData data = GpuAcDataExtractor.extract(network, es);
        // our batched LU prices a slot from its own plan, cuDSS from its estimate of one matrix: a structure this small
        // fits the cap in either
        long structure = GpuAcStructures.handle(n, data);
        assertEquals(4096, StructureNative.recommendBatchSize(structure, 4096, GpuLinearSolveMode.BATCHED_LU));
        assertEquals(4096, StructureNative.recommendBatchSize(structure, 4096, GpuLinearSolveMode.UBATCH));
        // the recommendation is remembered per extraction and mode: a run in the other mode is priced for its own slots
        AtomicInteger pricings = new AtomicInteger();
        GpuBatchContextCache.forceChunkSizeForTests(null);
        GpuBatchContextCache.clearContextCache();
        GpuBatchContextCache.priceChunksForTests(pricings::incrementAndGet);
        try {
            GpuBatchContextCache.resolveChunkSize(20_000, n, data, GpuLinearSolveMode.BATCHED_LU);
            GpuBatchContextCache.resolveChunkSize(20_000, n, data, GpuLinearSolveMode.BATCHED_LU);
            assertEquals(1, pricings.get(), "an extraction is priced once per mode");
            GpuBatchContextCache.resolveChunkSize(20_000, n, data, GpuLinearSolveMode.UBATCH);
            assertEquals(2, pricings.get(), "another mode is priced for its own slots");
        } finally {
            GpuBatchContextCache.priceChunksForTests(null);
            GpuBatchContextCache.clearContextCache();
        }
    }

    @Test
    void chunkSizeAlwaysClampedToScenarioCount() {
        // Holds for every path (explicit cap, auto-size, or fallback): 1 <= chunk <= nb.
        GpuBatchContextCache.forceChunkSizeForTests(null);
        int chunk = GpuBatchContextCache.resolveChunkSize(5, 10, EMPTY, GpuLinearSolveMode.UBATCH);
        assertTrue(chunk >= 1 && chunk <= 5, "chunk must be in [1, nb], was " + chunk);

        GpuBatchContextCache.forceChunkSizeForTests(0);   // invalid -> auto/fallback
        try {
            int c2 = GpuBatchContextCache.resolveChunkSize(5, 10, EMPTY, GpuLinearSolveMode.UBATCH);
            assertTrue(c2 >= 1 && c2 <= 5, "chunk must be in [1, nb], was " + c2);
        } finally {
            GpuBatchContextCache.forceChunkSizeForTests(null);
        }
    }
}
