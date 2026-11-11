/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.violations.LimitViolation;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
import com.powsybl.security.results.ConnectivityResult;
import com.powsybl.security.results.PostContingencyResult;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

/**
 * {@code OLF_GPU_STATE_VIEW_CHECK}: after a device-answered contingency's state is written back and its violations
 * detected, every quantity the detection can read is recomputed by {@link GpuStateView} from the device state array and
 * compared, bit for bit ({@code Double.doubleToLongBits}), with what OLF's objects return:
 * <ul>
 *   <li>every enabled branch of the limit table: I, P and apparent power at both sides;</li>
 *   <li>every enabled bus: the voltage {@code getV()} and the angle {@code getAngle()};</li>
 *   <li>every voltage-angle limit between enabled buses: the angle difference OLF compares.</li>
 * </ul>
 * The lines name elements by number only. A channel only prints: the check reads the objects after the detection and
 * changes nothing; it makes the write-back refresh every branch's flows ({@code flowReadSet}), which only costs time.
 *
 * <p>A contingency answered from the state without the write-back (P1 of {@code docs/device-final-results-plan.md}) is
 * also written back and detected the old way while the channel is on, and the two decisions are compared: the
 * realistic-voltage status, and the violation lists field by field, doubles by their bits ({@link #compareStatus},
 * {@link #compareViolations}). The answered result stays the state's. The tests turn that comparison on in code
 * ({@link #compareInTests}), where a mismatch fails the test.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuStateViewCheck {

    static final LongAdder CONTINGENCIES = new LongAdder();
    static final LongAdder COMPARED = new LongAdder();
    static final LongAdder MISMATCHES = new LongAdder();
    /** Bus voltages whose {@code setV}/{@code getV} round trip differs from the state's value: what the view's round
     *  trip is for. */
    static final LongAdder ROUND_TRIP_DIFFERS = new LongAdder();
    /** Contingencies answered from the state and also detected the old way, and the decisions that differed. */
    static final LongAdder DECISIONS_COMPARED = new LongAdder();
    static final LongAdder DECISION_MISMATCHES = new LongAdder();
    /** Contingencies answered without applying them, whose result was compared with the usual path's. */
    static final LongAdder RESULTS_COMPARED = new LongAdder();
    private static volatile boolean comparedInTests;
    private static final AtomicInteger SCENARIOS = new AtomicInteger();
    private static final AtomicInteger PRINTED = new AtomicInteger();
    private static final int PRINTED_MAX = 20;
    private static final String[] QUANTITIES = {"I", "P", "S"};

    private GpuStateViewCheck() {
    }

    /** True when a contingency answered from the state is also written back and detected the old way, to compare. */
    static boolean comparesDecisions() {
        return comparedInTests || GpuTrace.STATE_VIEW_CHECK.isOn();
    }

    /** Tests only: compare every answer from the state with the old way's, and throw on a difference. */
    static void compareInTests(boolean on) {
        comparedInTests = on;
    }

    /** The realistic-voltage status from the state and from the written-back objects. */
    static void compareStatus(boolean unrealisticFromState, boolean unrealisticFromObjects) {
        if (unrealisticFromState != unrealisticFromObjects) {
            decisionMismatch("realistic-voltage status: state " + !unrealisticFromState + ", objects " + !unrealisticFromObjects);
        }
    }

    /** The violation lists from the state and from the written-back objects, field by field, doubles by their bits. */
    static void compareViolations(List<LimitViolation> fromState, List<LimitViolation> fromObjects) {
        DECISIONS_COMPARED.increment();
        if (fromState.size() != fromObjects.size()) {
            decisionMismatch("violation count: state " + fromState.size() + ", objects " + fromObjects.size());
            return;
        }
        for (int i = 0; i < fromState.size(); i++) {
            String field = differingField(fromState.get(i), fromObjects.get(i));
            if (field != null) {
                decisionMismatch("violation #" + i + " of " + fromState.size() + " differs in " + field + " (type "
                        + fromObjects.get(i).getLimitType() + ")");
                return;
            }
        }
    }

    /**
     * A contingency's result answered without applying it to the network and the result of the usual path (applied,
     * then answered from the state or written back), on the same device state: the statuses, the violation lists field
     * by field, the distributed active power and the connectivity result, doubles by their bits.
     */
    static void compareResults(PostContingencyResult withoutNetwork, PostContingencyResult usual) {
        RESULTS_COMPARED.increment();
        if (withoutNetwork.getStatus() != usual.getStatus()) {
            decisionMismatch("without the network: status " + withoutNetwork.getStatus() + ", usual " + usual.getStatus());
            return;
        }
        if (!same(withoutNetwork.getDistributedActivePower(), usual.getDistributedActivePower())) {
            decisionMismatch("without the network: distributed active power " + Double.toHexString(withoutNetwork.getDistributedActivePower())
                    + ", usual " + Double.toHexString(usual.getDistributedActivePower()));
            return;
        }
        ConnectivityResult a = withoutNetwork.getConnectivityResult();
        ConnectivityResult b = usual.getConnectivityResult();
        if (a.getCreatedSynchronousComponentCount() != b.getCreatedSynchronousComponentCount()
                || a.getCreatedConnectedComponentCount() != b.getCreatedConnectedComponentCount()
                || !same(a.getDisconnectedLoadActivePower(), b.getDisconnectedLoadActivePower())
                || !same(a.getDisconnectedGenerationActivePower(), b.getDisconnectedGenerationActivePower())
                || !a.getDisconnectedElements().equals(b.getDisconnectedElements())) {
            decisionMismatch("without the network: the connectivity result differs");
            return;
        }
        if (withoutNetwork.getNetworkResult().getBranchResults().size() != usual.getNetworkResult().getBranchResults().size()
                || withoutNetwork.getNetworkResult().getBusResults().size() != usual.getNetworkResult().getBusResults().size()
                || withoutNetwork.getNetworkResult().getThreeWindingsTransformerResults().size()
                    != usual.getNetworkResult().getThreeWindingsTransformerResults().size()) {
            decisionMismatch("without the network: the network result differs");
            return;
        }
        List<LimitViolation> fromState = withoutNetwork.getLimitViolationsResult().getLimitViolations();
        List<LimitViolation> fromUsual = usual.getLimitViolationsResult().getLimitViolations();
        if (fromState.size() != fromUsual.size()) {
            decisionMismatch("without the network: violation count " + fromState.size() + ", usual " + fromUsual.size());
            return;
        }
        for (int i = 0; i < fromState.size(); i++) {
            String field = differingField(fromState.get(i), fromUsual.get(i));
            if (field != null) {
                decisionMismatch("without the network: violation #" + i + " of " + fromState.size() + " differs in " + field
                        + " (type " + fromUsual.get(i).getLimitType() + ")");
                return;
            }
        }
    }

    /** The first field two violations differ in, or null; names and ids are compared, never printed. */
    private static String differingField(LimitViolation a, LimitViolation b) {
        if (!Objects.equals(a.getSubjectId(), b.getSubjectId())) {
            return "subject";
        }
        if (a.getLimitType() != b.getLimitType()) {
            return "type";
        }
        if (!Objects.equals(a.getLimitName(), b.getLimitName()) || !Objects.equals(a.getOperationalLimitsGroupId(), b.getOperationalLimitsGroupId())) {
            return "limit name or group";
        }
        if (a.getAcceptableDuration() != b.getAcceptableDuration()) {
            return "acceptable duration";
        }
        if (!same(a.getLimit(), b.getLimit()) || !same(a.getLimitReduction(), b.getLimitReduction())) {
            return "limit or reduction";
        }
        if (!same(a.getValue(), b.getValue())) {
            return "value (state " + Double.toHexString(a.getValue()) + ", objects " + Double.toHexString(b.getValue()) + ")";
        }
        if (!Objects.equals(a.getSide(), b.getSide())) {
            return "side";
        }
        return null;
    }

    private static void decisionMismatch(String what) {
        DECISION_MISMATCHES.increment();
        if (comparedInTests) {
            throw new AssertionError("STATE_VIEW_CHECK the decision from the state differs from the written-back one: " + what);
        }
        if (PRINTED.getAndIncrement() < PRINTED_MAX) {
            System.err.println("STATE_VIEW_CHECK DECISION MISMATCH " + what);
        }
    }

    static void check(GpuStateView view, double[] state, GpuLimitTable table, LfNetwork network) {
        int scenario = SCENARIOS.getAndIncrement();
        long compared = 0;
        LfBranch[] branches = table.branches();
        for (int b = 0; b < branches.length; b++) {
            LfBranch branch = branches[b];
            if (table.groupStart(b) == table.groupEnd(b) || branch.isDisabled()) {
                continue;
            }
            for (int side = GpuTwoSides.ONE; side <= GpuTwoSides.TWO; side++) {
                for (byte quantity = GpuLimitTable.QTY_I; quantity <= GpuLimitTable.QTY_S; quantity++) {
                    compared++;
                    double fromView = view.branch(state, branch, side, quantity);
                    double fromOlf = olfValue(branch, side, quantity);
                    if (!same(fromView, fromOlf)) {
                        mismatch(scenario, "branch=" + branch.getNum() + " side=" + (side + 1) + " quantity="
                                + QUANTITIES[quantity] + " status=" + status(branch), fromView, fromOlf);
                    }
                }
            }
        }
        for (LfBus bus : network.getBuses()) {
            if (bus.isDisabled()) {
                continue;
            }
            compared += 2;
            double v = view.busV(state, bus);
            if (!same(v, bus.getV())) {
                mismatch(scenario, "bus=" + bus.getNum() + " quantity=V", v, bus.getV());
            }
            if (bus.getV() != view.stateV(state, bus)) {
                ROUND_TRIP_DIFFERS.increment();
            }
            double angle = view.busAngle(state, bus);
            if (!same(angle, bus.getAngle())) {
                mismatch(scenario, "bus=" + bus.getNum() + " quantity=ANGLE", angle, bus.getAngle());
            }
        }
        for (LfNetwork.LfVoltageAngleLimit limit : network.getVoltageAngleLimits()) {
            if (limit.getFrom().isDisabled() || limit.getTo().isDisabled()) {
                continue;
            }
            compared++;
            double fromView = view.busAngle(state, limit.getTo()) - view.busAngle(state, limit.getFrom());
            double fromOlf = limit.getTo().getAngle() - limit.getFrom().getAngle();
            if (!same(fromView, fromOlf)) {
                mismatch(scenario, "angleLimitFrom=" + limit.getFrom().getNum() + " to=" + limit.getTo().getNum(),
                        fromView, fromOlf);
            }
        }
        CONTINGENCIES.increment();
        COMPARED.add(compared);
    }

    /** The run's totals: one line, numbers only. */
    static String summary() {
        return "STATE_VIEW_CHECK contingencies=" + CONTINGENCIES.sum() + " quantities=" + COMPARED.sum()
                + " mismatches=" + MISMATCHES.sum() + " busVoltagesWhoseRoundTripDiffers=" + ROUND_TRIP_DIFFERS.sum()
                + " decisionsCompared=" + DECISIONS_COMPARED.sum() + " resultsWithoutTheNetworkCompared=" + RESULTS_COMPARED.sum()
                + " decisionMismatches=" + DECISION_MISMATCHES.sum();
    }

    /** What the stock detection and {@code GpuLimitScanner} read for this side and quantity. */
    private static double olfValue(LfBranch branch, int side, byte quantity) {
        if (side == GpuTwoSides.ONE) {
            return switch (quantity) {
                case GpuLimitTable.QTY_I -> branch.getI1().eval();
                case GpuLimitTable.QTY_P -> branch.getP1().eval();
                default -> branch.computeApparentPower1();
            };
        }
        return switch (quantity) {
            case GpuLimitTable.QTY_I -> branch.getI2().eval();
            case GpuLimitTable.QTY_P -> branch.getP2().eval();
            default -> branch.computeApparentPower2();
        };
    }

    private static String status(LfBranch branch) {
        boolean live1 = branch.getBus1() != null && branch.isConnectedSide1();
        boolean live2 = branch.getBus2() != null && branch.isConnectedSide2();
        if (live1 && live2) {
            return branch.getClosedP1() != null ? "closed" : "closed-no-terms";
        }
        return live1 ? "side2-open" : live2 ? "side1-open" : "both-open";
    }

    private static boolean same(double a, double b) {
        return Double.doubleToLongBits(a) == Double.doubleToLongBits(b);
    }

    private static void mismatch(int scenario, String what, double fromView, double fromOlf) {
        MISMATCHES.increment();
        if (PRINTED.getAndIncrement() < PRINTED_MAX) {
            System.err.println("STATE_VIEW_CHECK MISMATCH scenario=" + scenario + " " + what + " view="
                    + Double.toHexString(fromView) + " olf=" + Double.toHexString(fromOlf));
        }
    }
}
