/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.openloadflow.ac.AcTargetVector;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The per-extraction shared-structure cache ({@link GpuAcStructures}): two extraction FLAVORS of one
 * equation-system state — the base load flow always extracts DS_RL, an SA run with distributed slack /
 * reactive limits disabled extracts another flavor, so they are DIFFERENT {@link GpuAcData} instances —
 * must still share ONE device-resident structure, because the flavor flags only select the outer-loop
 * RL/DS packs, which are batch-context inputs and not structure inputs
 * ({@link GpuAcData#sameStructureInputsAs}). A context built on the cross-flavor-aliased handle must
 * solve normally.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuAcStructuresTest {


    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(),
                "GPU full-NR native lib not available (build it with native/build-gpu.sh + cuDSS)");
    }

    @Test
    void crossFlavorExtractionsShareOneStructure() {
        LfNetwork network = Networks.load(EurostagFactory.fix(EurostagTutorialExample1Factory.create()),
                new FirstSlackBusSelector()).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        AcTargetVector target = new AcTargetVector(network, es);
        int n = es.getIndex().getSortedVariablesToFind().size();

        // Two flavors of the SAME equation-system state: distinct instances, distinct RL/DS packs
        // (eurostag's PV generator gives the DS_RL flavor a non-empty reactive-limits pack), but
        // identical structure inputs.
        GpuAcData dsRl = GpuAcDataExtractor.extract(network, es, true, true);
        GpuAcData none = GpuAcDataExtractor.extract(network, es, false, false);
        assertNotSame(dsRl, none);
        assertTrue(dsRl.getReactiveLimitCount() > 0, "eurostag's PV generator must fill the DS_RL flavor's RL pack");
        assertEquals(0, none.getReactiveLimitCount(), "the NONE flavor must have an empty RL pack");
        assertTrue(dsRl.sameStructureInputsAs(none),
                "the flavor flags must not change the structure inputs (only the RL/DS packs)");

        long first = GpuAcStructures.handle(n, dsRl);
        long builds = GpuAcStructures.buildCount();
        long hits = GpuAcStructures.hitCount();

        // Cross-flavor CONTENT match: same handle, no new build.
        long aliased = GpuAcStructures.handle(n, none);
        assertEquals(first, aliased, "a different flavor of the same state must alias the existing structure");
        assertEquals(builds, GpuAcStructures.buildCount(), "the cross-flavor lookup must not build a structure");
        assertEquals(hits + 1, GpuAcStructures.hitCount(), "the cross-flavor lookup must count as a hit");

        // Same-instance lookups keep hitting (both flavors are now keys).
        assertEquals(first, GpuAcStructures.handle(n, dsRl));
        assertEquals(first, GpuAcStructures.handle(n, none));
        assertEquals(hits + 3, GpuAcStructures.hitCount());

        // A context built on the aliased handle solves normally.
        GpuAcActivation act = GpuAcActivationExtractor.extractActivation(network, es, n);
        double[] targets = act.applyTargets(target.getArray());
        long ctx = GpuAcNewtonSolver.createContext(aliased);
        try {
            double[] out = GpuAcNewtonSolver.solveContext(ctx, es.getStateVector().get().clone(), targets,
                    act.rowMode(), 30, 1e-10, 0.0, 0, 0, 0.0, 0.0, 0.0, new int[n]);
            assertTrue(out[n + 1] < 1e-10, "the solve on the cross-flavor-shared structure must converge");
        } finally {
            GpuAcNewtonSolver.destroyContext(ctx);
        }
    }
}
