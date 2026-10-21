/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
import com.powsybl.openloadflow.ac.equations.vector.gpu.StructureNative;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The batched context cache keeps ONE context, built on the shared device structure of its extraction, and
 * reuses it whenever a request's extraction has the same content. The structure underneath is shared with
 * the base load flow, whose VALUES path refreshes the structure's device values in place and re-keys it
 * ({@code GpuAcStructures.rebind}): after that the cached context solves the refreshed values while its
 * stored extraction still describes the old ones, and a request for the old extraction - the same network
 * state as before, say the next security analysis after a load flow whose outer loops moved a tap - would be
 * handed a context on the wrong system. A context is reusable only while its extraction still resolves to
 * the structure it was built on.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchContextCacheTest {

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    @Test
    void contextIsRebuiltWhenItsStructureWasRefreshedAndReKeyed() {
        LfNetwork network = Networks.load(EurostagFactory.fix(EurostagTutorialExample1Factory.create()),
                new FirstSlackBusSelector()).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        int n = es.getIndex().getSortedVariablesToFind().size();
        int mode = GpuBatchContextCache.resolveBatchMode();
        GpuBatchContextCache.clearContextCache();

        GpuAcData before = GpuAcDataExtractor.extract(network, es);
        long builds = GpuBatchContextCache.buildCount();
        GpuBatchContextCache.acquireBatchContext(before, 1, mode, n);
        assertEquals(builds + 1, GpuBatchContextCache.buildCount(), "first request: a context is built");
        GpuBatchContextCache.acquireBatchContext(before, 1, mode, n);
        assertEquals(builds + 1, GpuBatchContextCache.buildCount(), "same extraction: the context is reused");

        // the base load flow's VALUES path on the same shared structure: a tap move keeps the pattern, the
        // structure's device values are refreshed in place and the key moves with them
        LfBranch trafo = network.getBranchById("NHV2_NLOAD");
        trafo.getPiModel().setR1(trafo.getPiModel().getR1() * 1.1);
        GpuAcData after = GpuAcDataExtractor.extract(network, es);
        assertTrue(before.sameStructureAs(after));
        assertFalse(before.sameAs(after));
        StructureNative.refreshStructureValues(GpuAcStructures.handle(n, before),
                after.closedBranch().parameters(), after.openBranch().parameters(), after.shunt().parameters(), after.hvdcAcEmulation().parameters(), after.zeroImpedanceBranch().parameters());
        GpuAcStructures.rebind(before, after);

        // the old extraction again (the network state restored): its structure is no longer the one the
        // cached context was built on, so that context must not be handed out for it
        GpuBatchContextCache.acquireBatchContext(before, 1, mode, n);
        assertEquals(builds + 2, GpuBatchContextCache.buildCount(),
                "a context whose structure now carries other values must be rebuilt for the old extraction");
        GpuBatchContextCache.clearContextCache();
    }
}
