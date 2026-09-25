/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.contingency.violations.LimitViolation;
import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.LfNetworkLoaderImpl;
import com.powsybl.openloadflow.sa.LimitReductionManager;
import com.powsybl.openloadflow.sa.LimitViolationManager;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The flattened limit table must reproduce {@code LimitViolationManager.detectViolations} decision
 * for decision, on the same network state.
 *
 * <p>This is the regression half of that guarantee — the other half runs on real data, where
 * {@code OLF_GPU_VIOL_CROSSCHECK=1} demands the same agreement on every contingency of an rte6515
 * census (three-winding legs, tie lines, several operational limits groups, temporary limits). Here
 * the point is that the agreement is checked by the SUITE, without a device and without a case file.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuLimitTableTest {

    /** The base case violates nothing, so this is the no-false-positive direction: a flattened table
     *  that got a threshold or a scale wrong would report a violation where OLF reports none. */
    @Test
    void reportsNothingWhereStockDetectionReportsNothing() {
        Network iidm = EurostagTutorialExample1Factory.createWithFixedCurrentLimits();
        LoadFlowParameters lfp = new LoadFlowParameters();
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.create(lfp);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);
        new AcloadFlowEngine(new AcLoadFlowContext(network, acp)).run();

        List<LimitViolation> stock = detectStock(network);
        List<LimitViolation> table = detectTable(network);

        assertTrue(stock.isEmpty(), "the Eurostag base case violates nothing — that is the point here");
        assertEquals(describe(stock), describe(table));
    }

    /** A contingency changes which branches are disabled, which is the one thing the table does NOT
     *  bake in — so the agreement has to hold with a branch out too. */
    @Test
    void reproducesStockDetectionWithABranchDisabled() {
        Network iidm = EurostagTutorialExample1Factory.createWithFixedCurrentLimits();
        LoadFlowParameters lfp = new LoadFlowParameters();
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.create(lfp);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);
        network.getBranchById("NHV1_NHV2_1").setDisabled(true);
        new AcloadFlowEngine(new AcLoadFlowContext(network, acp)).run();

        List<LimitViolation> stock = detectStock(network);
        List<LimitViolation> table = detectTable(network);

        assertFalse(stock.isEmpty(), "the surviving line should be overloaded once its parallel line is out");
        assertEquals(describe(stock), describe(table));
        // and the disabled branch must not be reported by either
        assertFalse(describe(table).contains("NHV1_NHV2_1"), "a disabled branch reports no violation");
    }

    /**
     * A branch whose side a contingency opened is evaluated by the screen in the BASE topology, closed — so its
     * candidates can be wrong both ways. Too high and the group's cut stops at a limit the host re-test rejects,
     * hiding the lower limit the stock detection reports (rte6515 BUS-1601_BBS: LINE-2771-1601 lost its 47.6 MVA
     * apparent-power violation). The scanner must therefore ignore such a branch's candidates and scan it in full,
     * in place. Here the screen result is forged to what an over-estimating device emits: only each group's highest
     * limit, which the true flow does not reach.
     */
    @Test
    void scansASideOpenedBranchInFullInsteadOfTrustingItsCandidates() {
        Network iidm = EurostagTutorialExample1Factory.createWithFixedCurrentLimits();
        // temporary limits ABOVE the ~1048 A flow the outage leaves on side TWO: the violation stays the permanent
        // one, and a candidate at the 1' limit is one the host re-test rejects
        iidm.getLine("NHV1_NHV2_2").getOrCreateSelectedOperationalLimitsGroup2().newCurrentLimits()
                .setPermanentLimit(500)
                .beginTemporaryLimit().setName("20'").setAcceptableDuration(1200).setValue(1100).endTemporaryLimit()
                .beginTemporaryLimit().setName("1'").setAcceptableDuration(60).setValue(1500).endTemporaryLimit()
                .add();
        LoadFlowParameters lfp = new LoadFlowParameters();
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.create(lfp);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(iidm, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        LfNetwork network = LfNetwork.load(iidm, new LfNetworkLoaderImpl(), acp.getNetworkParameters(), ReportNode.NO_OP).get(0);
        network.getBranchById("NHV1_NHV2_1").setDisabled(true);
        new AcloadFlowEngine(new AcLoadFlowContext(network, acp)).run();
        List<LimitViolation> stock = detectStock(network);
        assertFalse(stock.isEmpty(), "the surviving line should be overloaded once its parallel line is out");

        GpuLimitTable table = GpuLimitTable.build(network, LimitReductionManager.create(List.of()));
        int b = -1;
        for (int k = 0; k < table.branches().length; k++) {
            if (table.branches()[k].getId().equals("NHV1_NHV2_2")) {
                b = k;
            }
        }
        // every group of the overloaded branch emits only its highest-threshold limit: what a device reading the
        // branch far above every limit sends once the cut stops the group there
        java.util.List<Integer> forged = new java.util.ArrayList<>();
        for (int g = table.groupStart(b); g < table.groupEnd(b); g++) {
            int highest = -1;
            for (int l = table.limitStart(g); l < table.limitEnd(g); l++) {
                if (highest < 0 || table.threshold(l) > table.threshold(highest)) {
                    highest = l;
                }
            }
            if (highest >= 0) {
                forged.add(highest);
            }
        }
        int[] candidates = forged.stream().mapToInt(Integer::intValue).toArray();
        java.util.BitSet sideOpened = new java.util.BitSet();
        sideOpened.set(b);

        var trusted = new LimitViolationManager(List.of());
        new GpuLimitScanner(table).detectFromCandidates(network, trusted, candidates, null, new int[0], null);
        var scanned = new LimitViolationManager(List.of());
        new GpuLimitScanner(table).detectFromCandidates(network, scanned, candidates, null, new int[0], sideOpened);

        // the forged list carries no bus record, so only the branch is compared
        String stockBranch = describe(stock.stream().filter(v -> v.getSubjectId().equals("NHV1_NHV2_2")).toList());
        assertFalse(stockBranch.isEmpty());
        assertFalse(stockBranch.equals(describe(trusted.getLimitViolations())),
                "the forged candidates must actually lose a violation, or this test proves nothing");
        assertEquals(stockBranch, describe(scanned.getLimitViolations()));
    }

    private static List<LimitViolation> detectStock(LfNetwork network) {
        var manager = new LimitViolationManager(List.of());
        manager.detectViolations(network);
        return manager.getLimitViolations();
    }

    private static List<LimitViolation> detectTable(LfNetwork network) {
        var manager = new LimitViolationManager(List.of());
        new GpuLimitScanner(GpuLimitTable.build(network, LimitReductionManager.create(List.of())))
                .detectViolations(network, manager);
        return manager.getLimitViolations();
    }

    /** Every reported field, in order — a wrong scale or a wrong temporary limit is a difference,
     *  not only a wrong verdict. */
    private static String describe(List<LimitViolation> violations) {
        StringBuilder sb = new StringBuilder();
        for (LimitViolation v : violations) {
            sb.append(v.getSubjectId()).append('|').append(v.getSide()).append('|').append(v.getLimitType())
                    .append('|').append(v.getOperationalLimitsGroupId()).append('|').append(v.getLimitName())
                    .append('|').append(v.getAcceptableDuration()).append('|').append(v.getLimit())
                    .append('|').append(v.getLimitReduction()).append('|').append(v.getValue()).append('\n');
        }
        return sb.toString();
    }
}
