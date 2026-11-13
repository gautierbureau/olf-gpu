/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.Contingency;
import com.powsybl.contingency.ContingencyElement;
import com.powsybl.contingency.ContingencyElementType;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.impl.LfBoundaryLineBranch;
import com.powsybl.openloadflow.network.impl.LfBranchImpl;
import com.powsybl.openloadflow.network.impl.LfTieLineBranch;
import com.powsybl.openloadflow.sa.PreContingencyNetworkResult;
import com.powsybl.openloadflow.util.PerUnit;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.monitor.StateMonitor;
import com.powsybl.security.monitor.StateMonitorIndex;
import com.powsybl.security.results.BranchResult;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * The monitored branches' post-contingency results of {@code PostContingencyNetworkResult.update}, with the
 * monitored branches found ONCE per run instead of once per contingency.
 *
 * <p>OLF's {@code AbstractNetworkResult.addResults} walks every branch of the network for every contingency to find
 * the monitored ones (a branch is monitored when one of its original ids is, the first match wins), then builds their
 * results. On network.xiidm that walk is 12,000 branches for 100 monitored ones, 9 of the GPU security analysis' 17 s.
 * Here the walk runs once, in the network's branch order, and each contingency builds the results of those branches
 * only - the contingency's disabled branches skipped, each result built by OLF's own {@code createBranchResult} and
 * kept under OLF's pre/post modification threshold, exactly as {@code createBranchResults} does it.
 *
 * <p>It covers the all-state monitor's branches: a contingency with its own monitor, monitored voltage levels or
 * three-winding transformers, and zero-impedance monitored branches (whose flows OLF computes over the spanning
 * tree) keep OLF's own update ({@link #appliesTo}).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuMonitoredBranchResults {

    private final StateMonitorIndex monitorIndex;
    private final boolean applies;
    private final List<LfBranch> monitored;
    /** Every monitored branch's result made of its six evaluables, non-zero impedance and both buses present: a line
     *  or a two-winding transformer (LfBranchImpl), a tie line (LfTieLineBranch: the line's and its halves'), a boundary
     *  line (LfBoundaryLineBranch: no flow transfer). Then the results can be read from a state (branchResultsFromState). */
    private final boolean readableFromState;
    private final double[] currentScale1;
    private final double[] currentScale2;
    private final String[] half1Id;                          // a tie line's halves, null otherwise
    private final String[] half2Id;
    private final boolean[] boundary;

    private GpuMonitoredBranchResults(StateMonitorIndex monitorIndex, boolean applies, List<LfBranch> monitored) {
        this.monitorIndex = monitorIndex;
        this.applies = applies;
        this.monitored = monitored;
        this.readableFromState = monitored.stream().allMatch(b -> (b instanceof LfBranchImpl || b instanceof LfTieLineBranch
                || b instanceof LfBoundaryLineBranch) && !b.isZeroImpedance(LoadFlowModel.DC) && b.getBus1() != null && b.getBus2() != null);
        this.currentScale1 = new double[monitored.size()];
        this.currentScale2 = new double[monitored.size()];
        this.half1Id = new String[monitored.size()];
        this.half2Id = new String[monitored.size()];
        this.boundary = new boolean[monitored.size()];
        if (readableFromState) {
            for (int k = 0; k < monitored.size(); k++) {
                LfBranch branch = monitored.get(k);
                if (branch instanceof LfTieLineBranch tieLine) {
                    // LfTieLineBranch.createBranchResult: each half's terminal level
                    currentScale1[k] = PerUnit.ib(tieLine.getHalf1().getTerminal().getVoltageLevel().getNominalV());
                    currentScale2[k] = PerUnit.ib(tieLine.getHalf2().getTerminal().getVoltageLevel().getNominalV());
                    half1Id[k] = tieLine.getHalf1().getId();
                    half2Id[k] = tieLine.getHalf2().getId();
                } else if (branch instanceof LfBoundaryLineBranch) {
                    // LfBoundaryLineBranch.createBranchResult: the boundary line's terminal level (its network side, bus 1), both sides
                    currentScale1[k] = PerUnit.ib(branch.getBus1().getNominalV());
                    currentScale2[k] = currentScale1[k];
                    boundary[k] = true;
                } else {
                    // LfBranchImpl.createBranchResult's scales: the nominal voltage of each terminal's level, its bus's
                    currentScale1[k] = PerUnit.ib(branch.getBus1().getNominalV());
                    currentScale2[k] = PerUnit.ib(branch.getBus2().getNominalV());
                }
            }
        }
    }

    /** Whether {@link #branchResultsFromState} can answer this run's contingencies (no result extension asked). */
    boolean readableFromState(boolean createResultExtension) {
        return readableFromState && !createResultExtension;
    }

    static GpuMonitoredBranchResults build(LfNetwork network, StateMonitorIndex monitorIndex, StateMonitorIndex zeroImpedanceMonitorIndex) {
        StateMonitor all = monitorIndex.getAllStateMonitor();
        StateMonitor zeroImpedance = zeroImpedanceMonitorIndex.getAllStateMonitor();
        boolean applies = all.getVoltageLevelIds().isEmpty() && all.getThreeWindingsTransformerIds().isEmpty()
                && zeroImpedance.getBranchIds().isEmpty() && zeroImpedance.getThreeWindingsTransformerIds().isEmpty()
                && zeroImpedanceMonitorIndex.getSpecificStateMonitors().isEmpty();
        List<LfBranch> monitored = new ArrayList<>();
        if (applies && !all.getBranchIds().isEmpty()) {
            for (LfBranch branch : network.getBranches()) {
                for (String originalId : branch.getOriginalIds()) {
                    if (all.getBranchIds().contains(originalId)) {
                        monitored.add(branch);
                        break;
                    }
                }
            }
        }
        return new GpuMonitoredBranchResults(monitorIndex, applies, monitored);
    }

    /** Whether this contingency's network result is the all-state monitor's branches only. */
    boolean appliesTo(String contingencyId) {
        return applies && !monitorIndex.getSpecificStateMonitors().containsKey(contingencyId);
    }

    /** {@code PostContingencyNetworkResult.update(isBranchDisabled).getBranchResults()} for a contingency it applies to. */
    List<BranchResult> branchResults(Contingency contingency, Predicate<LfBranch> isBranchDisabled, PreContingencyNetworkResult preContingency,
                                     boolean createResultExtension, SecurityAnalysisParameters.ModifiedMonitoredElementsParameters modified) {
        List<BranchResult> out = new ArrayList<>();
        if (monitored.isEmpty()) {
            return out;
        }
        double preContingencyBranchOfContingencyP1 = preContingencyBranchOfContingencyP1(contingency, preContingency);
        Map<String, LfBranch.LfBranchResults> zeroImpedanceFlows = new HashMap<>();
        double threshold = modified.getPowerModificationThreshold();
        for (LfBranch branch : monitored) {
            if (isBranchDisabled.test(branch)) {
                continue;
            }
            BranchResult pre = preContingency.getBranchResult(branch.getId());
            double preP1 = pre != null ? pre.getP1() : Double.NaN;
            List<BranchResult> results = branch.createBranchResult(preP1, preContingencyBranchOfContingencyP1, createResultExtension,
                    zeroImpedanceFlows, LoadFlowModel.DC);
            if (changed(pre, results.getFirst(), threshold)) {
                out.addAll(results);
            }
        }
        return out;
    }

    /**
     * {@link #branchResults} without the state loaded into OLF's model: each monitored branch's
     * {@code LfBranchImpl.createBranchResult} (no extension), its evaluables read from {@code state} through
     * {@code reader}: {@code AbstractImpedantLfBranch.buildBranchResult}, operation for operation.
     */
    List<BranchResult> branchResultsFromState(Contingency contingency, Predicate<LfBranch> isBranchDisabled, PreContingencyNetworkResult preContingency,
                                              SecurityAnalysisParameters.ModifiedMonitoredElementsParameters modified, GpuDcStateReader reader,
                                              double[] state, double dcPowerFactor) {
        List<BranchResult> out = new ArrayList<>();
        if (monitored.isEmpty()) {
            return out;
        }
        double preContingencyBranchOfContingencyP1 = preContingencyBranchOfContingencyP1(contingency, preContingency);
        double threshold = modified.getPowerModificationThreshold();
        for (int k = 0; k < monitored.size(); k++) {
            LfBranch branch = monitored.get(k);
            if (isBranchDisabled.test(branch)) {
                continue;
            }
            BranchResult pre = preContingency.getBranchResult(branch.getId());
            double preP1 = boundary[k] || pre == null ? Double.NaN : pre.getP1();
            double preOfContingencyP1 = boundary[k] ? Double.NaN : preContingencyBranchOfContingencyP1;
            double flowP1 = reader.p(state, branch, GpuTwoSides.ONE) * PerUnit.SB;
            double flowQ1 = branch.getQ1().eval() * PerUnit.SB;
            double flowP2 = reader.p(state, branch, GpuTwoSides.TWO) * PerUnit.SB;
            double flowQ2 = branch.getQ2().eval() * PerUnit.SB;
            double currentI1 = reader.i(state, branch, GpuTwoSides.ONE, dcPowerFactor) * currentScale1[k];
            double currentI2 = reader.i(state, branch, GpuTwoSides.TWO, dcPowerFactor) * currentScale2[k];
            double flowTransfer = Double.NaN;
            if (!Double.isNaN(preP1) && !Double.isNaN(preOfContingencyP1)) {
                double flowChange = flowP1 - preP1;
                if (Math.abs(preOfContingencyP1) > P_EPSILON_SI) {
                    flowTransfer = flowChange / preOfContingencyP1;
                } else {
                    flowTransfer = 0.;
                }
            }
            BranchResult result = new BranchResult(branch.getId(), flowP1, flowQ1, currentI1, flowP2, flowQ2, currentI2, flowTransfer);
            if (changed(pre, result, threshold)) {
                out.add(result);
                if (half1Id[k] != null) {
                    // the tie line first, then its halves (LfTieLineBranch.createBranchResult)
                    out.add(new BranchResult(half1Id[k], flowP1, flowQ1, currentI1, Double.NaN, Double.NaN, Double.NaN, flowTransfer));
                    out.add(new BranchResult(half2Id[k], flowP2, flowQ2, currentI2, Double.NaN, Double.NaN, Double.NaN, flowTransfer));
                }
            }
        }
        return out;
    }

    /** AbstractImpedantLfBranch.P_EPSILON_SI: below it, the contingency's pre-contingency flow is taken as zero. */
    private static final double P_EPSILON_SI = 1e-6;

    private static double preContingencyBranchOfContingencyP1(Contingency contingency, PreContingencyNetworkResult preContingency) {
        if (contingency.getElements().size() == 1) {
            ContingencyElement element = contingency.getElements().getFirst();
            if (element.getType() == ContingencyElementType.BRANCH
                    || element.getType() == ContingencyElementType.LINE
                    || element.getType() == ContingencyElementType.BOUNDARY_LINE
                    || element.getType() == ContingencyElementType.TWO_WINDINGS_TRANSFORMER) {
                BranchResult result = preContingency.getBranchResult(element.getId());
                if (result != null) {
                    return result.getP1();
                }
            }
        }
        return Double.NaN;
    }

    private static boolean changed(double pre, double post, double threshold) {
        return Double.isNaN(pre) || Double.isNaN(post) || Math.abs(pre - post) > threshold;
    }

    /** {@code PostContingencyNetworkResult.changed(BranchResult, BranchResult)}. */
    private static boolean changed(BranchResult pre, BranchResult post, double threshold) {
        if (pre == null) {
            return true;
        }
        return threshold <= 0
                || changed(pre.getP1(), post.getP1(), threshold)
                || changed(pre.getQ1(), post.getQ1(), threshold)
                || changed(pre.getP2(), post.getP2(), threshold)
                || changed(pre.getQ2(), post.getQ2(), threshold);
    }

    /** The check hook: the fast results against OLF's, field for field and bit for bit; returns the differences. */
    static int compare(List<BranchResult> fast, List<BranchResult> olf) {
        if (fast.size() != olf.size()) {
            return Math.max(1, Math.abs(fast.size() - olf.size()));
        }
        int differences = 0;
        for (int k = 0; k < fast.size(); k++) {
            BranchResult a = fast.get(k);
            BranchResult b = olf.get(k);
            if (!a.getBranchId().equals(b.getBranchId())
                    || !same(a.getP1(), b.getP1()) || !same(a.getQ1(), b.getQ1()) || !same(a.getI1(), b.getI1())
                    || !same(a.getP2(), b.getP2()) || !same(a.getQ2(), b.getQ2()) || !same(a.getI2(), b.getI2())
                    || !same(a.getFlowTransfer(), b.getFlowTransfer())
                    || a.getExtensions().size() != b.getExtensions().size()) {
                differences++;
            }
        }
        return differences;
    }

    private static boolean same(double a, double b) {
        return Double.doubleToLongBits(a) == Double.doubleToLongBits(b);
    }
}
