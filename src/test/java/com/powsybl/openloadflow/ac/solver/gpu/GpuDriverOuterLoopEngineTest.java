/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcOuterLoopContext;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.lf.outerloop.OuterLoopResult;
import com.powsybl.openloadflow.lf.outerloop.OuterLoopStatus;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.MostMeshedSlackBusSelector;
import com.powsybl.openloadflow.network.impl.Networks;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuDriverOuterLoopEngineTest {

    /** OLF's own loop, counting the calls the engine makes on it. */
    private static final class CountingLoop implements AcOuterLoop {
        int initializations;
        int checks;

        @Override
        public String getName() {
            return "DistributedSlack";
        }

        @Override
        public void initialize(AcOuterLoopContext context) {
            initializations++;
        }

        @Override
        public OuterLoopResult check(AcOuterLoopContext context, ReportNode reportNode) {
            checks++;
            return new OuterLoopResult(this, OuterLoopStatus.STABLE);
        }
    }

    /**
     * OLF initializes every loop once, when the engine creates them (createOuterLoopsAndContexts), before this
     * engine wraps the ones the device drives. A check the device did not drive (the solve went to OLF's CPU
     * Newton-Raphson) runs OLF's loop as it stands: initializing it a second time there reset its state — and
     * re-froze, in FreezingHvdcACEmulation's case, the AC emulations the run had already unfrozen.
     */
    @Test
    void aCheckAfterACpuSolveDoesNotInitializeTheLoopAgain() {
        Network network = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        LfNetwork lfNetwork = Networks.load(network, new MostMeshedSlackBusSelector()).get(0);
        LoadFlowParameters parameters = new LoadFlowParameters();
        AcLoadFlowParameters acParameters = OpenLoadFlowParameters.createAcParameters(network, parameters,
                OpenLoadFlowParameters.create(parameters), new DenseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>());
        try (AcLoadFlowContext context = new AcLoadFlowContext(lfNetwork, acParameters)) {
            GpuDriverOuterLoopEngine engine = new GpuDriverOuterLoopEngine(context);
            CountingLoop real = new CountingLoop();
            AcOuterLoopContext loopContext = com.powsybl.openloadflow.ac.AcloadFlowEngine
                    .createOuterLoopsAndContexts(context, false).getFirst().getRight();
            real.initialize(loopContext);                    // what createOuterLoopsAndContexts did
            var wrapped = new GpuDriverOuterLoopEngine.DeviceDrivenLoop(engine, real);

            wrapped.check(loopContext, ReportNode.NO_OP);    // no solver yet: the device did not drive it
            wrapped.cleanup(loopContext);

            assertEquals(1, real.initializations, "OLF initializes its loop once");
            assertEquals(1, real.checks);
        }
    }
}
