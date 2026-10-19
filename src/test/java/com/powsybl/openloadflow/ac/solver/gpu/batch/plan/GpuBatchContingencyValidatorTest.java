/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;

import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchContingencyValidator.Disqualification;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchContingencyValidator.Qualification;
import com.powsybl.openloadflow.network.EurostagFactory;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.Networks;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Validates {@link GpuBatchContingencyValidator} qualification — a CPU-only graph check,
 * so this runs in the normal test suite (no GPU needed). Eurostag is NGEN—NHV1—(two
 * parallel lines)—NHV2—NLOAD: the parallel lines are meshed (removing one keeps the main
 * component intact → qualifies for the batch), while the two transformers are bridges
 * (removing either islands a bus → rejected, route to the CPU path).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuBatchContingencyValidatorTest extends com.powsybl.openloadflow.network.AbstractLoadFlowNetworkFactory {

    private LfNetwork eurostag() {
        Network iidm = EurostagFactory.fix(EurostagTutorialExample1Factory.create());
        return Networks.load(iidm, new FirstSlackBusSelector()).get(0);
    }

    private static BatchContingency contingency(LfNetwork network, String branchId) {
        LfBranch branch = network.getBranchById(branchId);
        return new BatchContingency(branchId, branch);
    }

    @Test
    void parallelLineOutageQualifies() {
        LfNetwork network = eurostag();
        GpuAcElementMapper mapper = new GpuAcElementMapper(network);
        List<Qualification> q = GpuBatchContingencyValidator.qualify(network,
                List.of(contingency(network, "NHV1_NHV2_1")), mapper, false, null);

        assertEquals(1, q.size());
        assertTrue(q.get(0).qualified(), "removing one of two parallel lines keeps the network connected");
    }

    @Test
    void bridgeOutageIsRejected() {
        LfNetwork network = eurostag();
        GpuAcElementMapper mapper = new GpuAcElementMapper(network);
        // NGEN_NHV1 is the only path to the generator bus NGEN — a bridge. With islanding handling OFF
        // (fixed-iter mode) it is rejected to the CPU path.
        List<Qualification> q = GpuBatchContingencyValidator.qualify(network,
                List.of(contingency(network, "NGEN_NHV1")), mapper, false, null);

        assertEquals(1, q.size());
        assertFalse(q.get(0).qualified(), "removing a bridge transformer islands a bus");
        assertEquals(Disqualification.FRAGMENTS_NETWORK, q.get(0).reason());
    }

    @Test
    void mixedListPartitionsCorrectly() {
        LfNetwork network = eurostag();
        GpuAcElementMapper mapper = new GpuAcElementMapper(network);
        List<Qualification> q = GpuBatchContingencyValidator.qualify(network, List.of(
                contingency(network, "NHV1_NHV2_1"),    // meshed → qualify
                contingency(network, "NGEN_NHV1"),       // bridge → reject
                contingency(network, "NHV1_NHV2_2"),     // meshed → qualify
                contingency(network, "NHV2_NLOAD")),     // bridge → reject
                mapper, false, null);

        assertEquals(4, q.size());
        assertTrue(q.get(0).qualified());
        assertFalse(q.get(1).qualified());
        assertEquals(Disqualification.FRAGMENTS_NETWORK, q.get(1).reason());
        assertTrue(q.get(2).qualified());
        assertFalse(q.get(3).qualified());
        assertEquals(Disqualification.FRAGMENTS_NETWORK, q.get(3).reason());

        // The connectivity must be fully restored after qualification (temporary changes undone):
        // re-qualifying yields the same verdicts.
        List<Qualification> again = GpuBatchContingencyValidator.qualify(network,
                List.of(contingency(network, "NHV1_NHV2_1")), mapper, false, null);
        assertTrue(again.get(0).qualified(), "connectivity must be restored after each qualification");
    }

    /** The slack s hangs on the rest by one line l0; the rest is the hub m, its coupler c1 to n and the leaves a, c. */
    private static Network slackOnAStub() {
        Network network = Network.create("slack-on-a-stub", "code");
        com.powsybl.iidm.network.Bus s = createBus(network, "x", "s");
        createGenerator(s, "gs", 3, 1.0);
        com.powsybl.iidm.network.Bus m = createBus(network, "y", "m");
        createLoad(m, "lm", 0.5, 0.2);
        com.powsybl.iidm.network.Bus n = createBus(network, "y", "n");
        createLoad(n, "ln", 0.5, 0.2);
        com.powsybl.iidm.network.Bus a = createBus(network, "y", "a");
        createLoad(a, "la", 0.5, 0.2);
        com.powsybl.iidm.network.Bus c = createBus(network, "y", "c");
        createLoad(c, "lc", 0.5, 0.2);
        createLine(network, s, m, "l0", 0.1);
        createLine(network, m, n, "c1", 0.0, 0.0);
        createLine(network, m, a, "lA", 0.1);
        createLine(network, n, c, "lC", 0.1);
        return network;
    }

    private static java.util.Set<String> lostWhenCut(LfNetwork network, String branchId) {
        var connectivity = network.getConnectivity();
        connectivity.startTemporaryChanges();
        try {
            connectivity.removeEdge(network.getBranchById(branchId));
            return connectivity.getVerticesRemovedFromMainComponent().stream()
                    .map(com.powsybl.openloadflow.network.LfBus::getId).collect(java.util.stream.Collectors.toSet());
        } finally {
            connectivity.undoTemporaryChanges();
        }
    }

    /** Qualifying an outage that isolates the slack anchors the islanding query on the relocated slack, as OLF
     *  does. OLF's anchor goes with its network state; the validator's must go too: setMainComponentVertex also
     *  sets the connectivity's DEFAULT main-component vertex, which every later query without an anchor of its
     *  own starts from - OLF's toLfContingency for the contingencies the CPU replays, and the tree replay. Left on
     *  the relocated slack, a later contingency losing that bus reads the whole network as lost. */
    @Test
    void slackRelocationLeavesTheConnectivityAnchorWhereItFoundIt() {
        LfNetwork network = Networks.load(slackOnAStub(), new FirstSlackBusSelector()).get(0);
        java.util.Set<String> otherSide = java.util.Set.of("m_vl_0", "n_vl_0", "a_vl_0", "c_vl_0");
        assertEquals(otherSide, lostWhenCut(network, "l0"), "before: the main component is the slack's");
        GpuAcElementMapper mapper = new GpuAcElementMapper(network);
        List<Qualification> q = GpuBatchContingencyValidator.qualify(network, List.of(contingency(network, "l0")), mapper, true,
                new SlackRelocation(new FirstSlackBusSelector(), 1));
        assertTrue(q.get(0).qualified(), "the slack is relocated inside the largest component");
        assertEquals("m_vl_0", q.get(0).relocatedSlack().getId());
        assertEquals(otherSide, lostWhenCut(network, "l0"), "after: the main component is still the slack's");
    }

    @Test
    void bridgeOutageWithIslandingHandlingQualifies() {
        LfNetwork network = eurostag();
        GpuAcElementMapper mapper = new GpuAcElementMapper(network);
        // NHV2_NLOAD bridges the load bus NLOAD (a leaf, not the slack). With islanding handling ON
        // (full-loadflow mode) the outage QUALIFIES and carries its islanded bus(es), to be frozen
        // per scenario rather than routed to the CPU.
        List<Qualification> q = GpuBatchContingencyValidator.qualify(network,
                List.of(contingency(network, "NHV2_NLOAD")), mapper, true, null);

        assertEquals(1, q.size());
        assertTrue(q.get(0).qualified(), "with islanding handling on, a fragmenting outage is batched on-GPU");
        assertFalse(q.get(0).islandedBuses().isEmpty(), "the islanded bus set must be reported for freezing");
    }
}
