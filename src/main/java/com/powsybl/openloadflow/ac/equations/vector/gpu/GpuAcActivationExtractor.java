/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CouplerTreeRelocation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.ReactiveRelocation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.StableRowIndexing;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcVariableType;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.SingleEquation;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.VoltageControl;
import com.powsybl.openloadflow.network.gpu.GeneratorVoltageControlRoles;

/**
 * Extracts what ACTIVATION decides — the per-row power-equation/identity mode, the
 * OLF-column→stable-row target mapping ({@link GpuAcActivation}) and the per-row variable
 * kind used by the MAX_VOLTAGE_CHANGE device step scaling. Rebuilt only when an activation
 * event fired; the extracted {@link GpuAcData} structure stays valid across PV→PQ switching
 * (see {@link StableRowIndexing} for the stable-row policy this reuses).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcActivationExtractor {

    private GpuAcActivationExtractor() {
    }

    /**
     * Assign one equation to one device row, refusing to overwrite an assignment already made.
     *
     * <p>{@code targetMap} carries at most one equation per row: the row IS the constraint's slot in
     * the batched system. A second write does not add a constraint, it silently DROPS the first —
     * the device then solves a system one equation short and converges happily on a state that
     * violates the lost one, with a clean residual, because the residual is only ever computed for
     * the equations the map still names. Nothing downstream can detect that, so it has to fail here.
     *
     * <p>This class of bug has occurred twice (a remote gen-VC pilot that is also a controller put
     * BUS_TARGET_V and DISTR_Q on the same row), which is why the check is unconditional rather than
     * behind a debug switch. It must stay ALLOCATION-FREE on the happy path: no remembered description per
     * row (String[n]) and no eagerly built message per call.
     */
    private static void assign(EquationSystem<AcVariableType, AcEquationType> es, int[] targetMap,
                               int row, int column, AcEquationType type, int elementNum) {
        if (targetMap[row] >= 0) {
            throw new PowsyblException("GPU AC solver: device row " + row + " assigned twice — "
                    + describeColumn(es, targetMap[row]) + " then " + type + " elem " + elementNum
                    + " (column " + column + "). One of the two constraints would be silently dropped.");
        }
        targetMap[row] = column;
    }

    /** The equation occupying {@code column}, for the collision message. Walks the equation list, which
     *  is fine because it runs only on the way to throwing — the alternative, remembering a description
     *  per row, costs a String[n] allocation on EVERY extraction. */
    private static String describeColumn(EquationSystem<AcVariableType, AcEquationType> es, int column) {
        for (SingleEquation<AcVariableType, AcEquationType> e : es.getIndex().getSortedSingleEquationsToSolve()) {
            if (e.getColumn() == column) {
                return e.getType() + " elem " + e.getElementNum() + " (column " + column + ")";
            }
        }
        return "column " + column;
    }

    /**
     * Extract what ACTIVATION decides — the per-row power-equation/identity mode and the
     * OLF-column→stable-row target mapping. Rebuilt only when an activation event fired
     * (the structure stays valid across PV→PQ switching). Walks every active equation, so
     * an unsupported equation type ACTIVATED at runtime (e.g. zero-impedance) fails loudly
     * here even though the full structural validation does not re-run.
     */
    public static GpuAcActivation extractActivation(LfNetwork network,
                                                    EquationSystem<AcVariableType, AcEquationType> es, int n) {
        int[] mode = new int[n];
        java.util.Arrays.fill(mode, GpuRowMode.EQUATION);
        int[] targetMap = new int[n];
        java.util.Arrays.fill(targetMap, -1);
        int nBranches = network.getBranches().size();
        int[] branchRhoCol = StableRowIndexing.branchVarCols(es, nBranches, AcVariableType.BRANCH_RHO1);
        int[] shuntBCol = StableRowIndexing.branchVarCols(es, network.getShunts().size(), AcVariableType.SHUNT_B);
        int[] vCol = new int[network.getBuses().size()];
        for (LfBus bus : network.getBuses()) {
            vCol[bus.getNum()] = StableRowIndexing.col(es, bus.getNum(), AcVariableType.BUS_V);
        }
        int[] qRow = ReactiveRelocation.computeQRow(network, es, vCol, branchRhoCol, shuntBCol);
        int[] phiCol = new int[network.getBuses().size()];
        for (LfBus bus : network.getBuses()) {
            phiCol[bus.getNum()] = StableRowIndexing.col(es, bus.getNum(), AcVariableType.BUS_PHI);
        }
        // DEAD-END STUB relocation, the other half of the one in GpuAcFamilyPacker: both sides must
        // name the SAME row for each equation. assign() refuses a double write, so a disagreement
        // fails loudly here instead of silently dropping a constraint.
        int[] dummyPCol = StableRowIndexing.branchVarCols(es, nBranches, AcVariableType.DUMMY_P);
        int[] dummyQCol = StableRowIndexing.branchVarCols(es, nBranches, AcVariableType.DUMMY_Q);
        // The coupler-tree relocation, from the same helper ExtractionColumns uses, so the scatter destinations and
        // this target map name the same rows: a child bus's power equations on its parent coupler's dummy columns,
        // that coupler's zero equations on the child's own PHI / V.
        int[][] ziTree = CouplerTreeRelocation.spanningTreeRelocation(network, es, phiCol, vCol, qRow, dummyPCol, dummyQCol);
        int[] stubBranchOfBus = ziTree[0];
        int[] stubBusOfBranch = ziTree[1];
        for (LfBus bus : network.getBuses()) {
            int stubBr = stubBranchOfBus[bus.getNum()];
            int p = StableRowIndexing.row(es, bus.getNum(), AcEquationType.BUS_TARGET_P);
            if (p >= 0) {
                assign(es, targetMap, stubBr >= 0 ? dummyPCol[stubBr] : phiCol[bus.getNum()], p,
                        AcEquationType.BUS_TARGET_P, bus.getNum());
            }
            int q = StableRowIndexing.row(es, bus.getNum(), AcEquationType.BUS_TARGET_Q);
            if (q >= 0) {
                // Q row = V row, the controlling rho row, or the DUMMY_Q row on a dead-end stub
                int targetQRow = stubBr >= 0 ? dummyQCol[stubBr] : qRow[bus.getNum()];
                GeneratorVoltageControl mainVc = GeneratorVoltageControlRoles.mainGeneratorVoltageControl(bus);
                if (stubBr < 0 && mainVc != null && ReactiveRelocation.hasCandidateTable(mainVc, es)) {
                    // A group with a candidate table is extracted CANONICALLY (pin-state independent: every
                    // controller's frozen q row suppressed, the pilot's on the FIRST member), but this target map
                    // is per scenario, so the row is the one the in-kernel mode scan picks in THIS state: a pinned
                    // member's own V row (mode 4 below, its alternate dest); the pilot's balance on the first
                    // ENABLED member's V row (the carrier), or on its own V row once every controller is pinned
                    // (its V identity gone, the row is a power row). Measured: the canonical row collided with the
                    // pinned first member's own Q (deviceRemoteFreeMemberSwitchedToPqInBaseMatchesCpu & co., 12 tests).
                    targetQRow = ReactiveRelocation.stateQRow(bus, vCol, stubBranchOfBus, dummyQCol);
                }
                assign(es, targetMap, targetQRow, q, AcEquationType.BUS_TARGET_Q, bus.getNum());
                // ACTIVE own-Q balance at a (non-free) remote gen-VC member = the RL-PINNED state:
                // row mode 4, so the family fills route the member's own q here (the frozen dest is
                // canonically SUPPRESSED) and the DISTR_Q emissions at this row gate off (plan-B2).
                // The SAME predicate computeQRow used to hand this bus its own Q row. It used
                // isRemoteGeneratorVoltageControlControllerElement, which INCLUDES a controller that is also the
                // group's controlled bus (its javadoc says so); this used isRemoteGenVcMember,
                // which is controlledBus != bus and therefore EXCLUDES the pilot. For a PINNED
                // pilot-that-is-also-a-controller the two disagreed: computeQRow gave it its own Q
                // row, this left the row at mode 1, so the mode-4 gate in fillDistrQBatch never
                // fired and the existence-keyed DISTR_Q kind-3 terms stayed live ALONGSIDE the
                // zero-impedance family's direct +-DUMMY_Q into the same row. The two cancel: on
                // rte6515 BUS-24_COUPLER_1 the kind-3 pair summed to +1.0 against the zero-impedance coupling's
                // -1.0, leaving the coupler's DUMMY_Q in NO equation. 45 such columns at 2000
                // busbar sections, which is what collapsed batching above ~1000.
                // A pilot that is one of its OWN controllers gets mode 4 whenever its own Q balance
                // is active, free member or not: the free-member exception exists for the member
                // whose row carries the pilot's Q in a pilot-is-not-a-controller group, which is not
                // this shape (the pilot's q feeds the DISTR_Q rows and the row's mode-4 alternate is
                // its own row). Left at mode 1, the base seeding (which keys a remote pin on mode 4)
                // read an all-pinned group as "pilot unpinned" (all-pinned-group fixture).
                boolean pilotSelf = bus.getGeneratorVoltageControl()
                        .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                        .map(c -> c.getControlledBus() == bus).orElse(false);
                // ...EXCEPT a pilot-self pilot ENROLLED in the candidate table (rotation): its pinned balance is
                // a POWER row — the carrier's row while a sibling controls, its own row once the group lost
                // control — routed to by the in-kernel mode scan, and mode 4 there is what the scan SKIPS. The
                // same predicate ExtractionColumns enrols with. Measured on rte6515's [5000,6000) busbar batch:
                // the carriers of VL-5997 and VL-6112 came in at mode 4, the scan fell back to the identity pilot
                // rows, the raw q and the couplers' DUMMY_Q vanished (dead columns 13171/13207) and every
                // scenario failed its first linear solve (976 x LINEAR_SOLVE_FAILED).
                boolean enrolled = pilotSelf
                        && bus.getGeneratorVoltageControl()
                                .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                                .map(c -> c.getMergedControllerElements().size() >= 2
                                        && c.getMergedControllerElements().stream().allMatch(m -> vCol[m.getNum()] >= 0))
                                .orElse(false);
                // `enrolled` requires pilotSelf, so for a MEMBER mode4 reduces to
                // !isFreeRemoteGeneratorVoltageControlMember, and "free" is the FIRST ENABLED member — a group whose members
                // are all pinned has no free member and every member's row is stamped mode 4 into
                // baseRowMode, which every scenario without an explicit per-scenario mode inherits. That is
                // CORRECT for a genuine Q-limit pin (the 20 base-pinned fixtures rely on it). What used to
                // reach here wrongly was a TRANSIENT MIN_REALISTIC_V freeze the device should never have
                // made: see RemoteMemberQPacker.isOlfRobustRemoteController (tasks #66), fixed at the SOURCE
                // by exempting a merged member from robust mode as OLF's own predicate does, not here.
                boolean mode4 = GeneratorVoltageControlRoles.isRemoteGeneratorVoltageControlControllerElement(bus)
                        && (pilotSelf || !GeneratorVoltageControlRoles.isFreeRemoteGeneratorVoltageControlMember(bus)) && !enrolled;
                if (mode4) {
                    mode[targetQRow] = GpuRowMode.BUS_TARGET_Q;
                }
            }
        }
        for (SingleEquation<AcVariableType, AcEquationType> e : es.getIndex().getSortedSingleEquationsToSolve()) {
            // POWER-equation rows (mode stays 1) whose stable row is a non-bus-P/Q variable — assembled by
            // the GPU DISTR_Q / zero-impedance passes; just map the target onto that variable's row.
            if (e.getType() == AcEquationType.DISTR_Q) {
                // the controller's BUS_V row — or, when the controller IS the group's controlled bus
                // (its own row already carries the pilot's V identity), the FREE controller's row
                int distrRow = ReactiveRelocation.distrQRow(es, network.getBus(e.getElementNum()),
                        stubBranchOfBus, dummyQCol);
                assign(es, targetMap, distrRow, e.getColumn(), AcEquationType.DISTR_Q, e.getElementNum());
                continue;
            }
            // ZERO_V / ZERO_PHI move off the dummy rows onto the STUB's own v / phi rows when this
            // branch feeds a dead end — the other two legs of the same four-way relocation.
            if ((e.getType() == AcEquationType.ZERO_V || e.getType() == AcEquationType.ZERO_PHI)
                    && stubBusOfBranch[e.getElementNum()] >= 0) {
                int stub = stubBusOfBranch[e.getElementNum()];
                assign(es, targetMap, e.getType() == AcEquationType.ZERO_V ? vCol[stub] : phiCol[stub],
                        e.getColumn(), e.getType(), e.getElementNum());
                continue;
            }
            AcVariableType powerRowVar = switch (e.getType()) {
                case DISTR_RHO -> AcVariableType.BRANCH_RHO1; // controller's BRANCH_RHO1 row
                case DISTR_SHUNT_B -> AcVariableType.SHUNT_B; // controller's SHUNT_B row
                // STOCK placement, crosswise (native zeroVRowOf / zeroPhiRowOf): zero-imp v1−rho·v2 sits on the
                // DUMMY_Q row and phi1−phi2 on the DUMMY_P row, so that each row's numeric repair from a donor
                // that carries its dummy column (a Q row, a P row) keeps an entry of its own in the donor's
                // diagonal column. The target travels with the equation: ZERO_PHI's is the branch's phase shift.
                case ZERO_V -> AcVariableType.DUMMY_Q;
                case ZERO_PHI -> AcVariableType.DUMMY_P;
                default -> null;
            };
            if (powerRowVar != null) {
                assign(es, targetMap, StableRowIndexing.col(es, e.getElementNum(), powerRowVar),
                        e.getColumn(), e.getType(), e.getElementNum());
                continue;
            }
            if (e.getType() == AcEquationType.BRANCH_TARGET_P) {
                // CONTINUOUS phase control enabled: the controlled branch's P-flow equation sits at the
                // CONTROLLER's a1-paired row (row mode 3 — the cb family re-emits the P outputs there,
                // closedBranchPhaseControlSide carrying the controller's a1 column, LOCAL and REMOTE alike).
                var branch = network.getBranchById(network.getBranches().stream()
                        .filter(b -> b.getNum() == e.getElementNum()).findFirst().orElseThrow().getId());
                var phaseControl = branch.getPhaseControl().orElseThrow();
                int a1r = StableRowIndexing.col(es, phaseControl.getControllerBranch().getNum(), AcVariableType.BRANCH_ALPHA1);
                mode[a1r] = GpuRowMode.BRANCH_TARGET_P;
                assign(es, targetMap, a1r, e.getColumn(), AcEquationType.BRANCH_TARGET_P, e.getElementNum());
                continue;
            }
            // IDENTITY rows (mode 0): the device row-mode mask writes diag 1 and F = x − target.
            AcVariableType vt = switch (e.getType()) {
                case BUS_TARGET_V -> AcVariableType.BUS_V;
                case BUS_TARGET_PHI -> AcVariableType.BUS_PHI;
                case BRANCH_TARGET_RHO1 -> AcVariableType.BRANCH_RHO1;   // ratio held constant: r1−target identity
                // Phase shift held constant: a1−target identity. This is the ONLY phase-control row the
                // GPU packs, and it is the one INCREMENTAL mode keeps active for the whole run (that loop
                // never calls setPhaseControlEnabled, so a1 stays a variable PINNED to its tap value and
                // the taps move between solves). CONTINUOUS mode with control ON activates BRANCH_TARGET_P
                // on the CONTROLLED branch instead — a branch-owned power row the GPU has no family for,
                // and it is NOT in this switch, so it hits the default throw below and the solver falls
                // back to CPU for that solve. Safe by construction, never silently dropped.
                case BRANCH_TARGET_ALPHA1 -> AcVariableType.BRANCH_ALPHA1;
                case SHUNT_TARGET_B -> AcVariableType.SHUNT_B;           // susceptance held constant: b−target identity
                case DUMMY_TARGET_P -> AcVariableType.DUMMY_P;           // non-tree zero-imp: dummyP held = 0
                case DUMMY_TARGET_Q -> AcVariableType.DUMMY_Q;           // non-tree zero-imp: dummyQ held = 0
                case BUS_TARGET_P, BUS_TARGET_Q -> null;     // power rows, mapped above
                default -> throw new PowsyblException("GPU AC solver: unsupported equation type "
                        + e.getType() + " activated");
            };
            if (vt != null) {
                int r = StableRowIndexing.col(es, e.getElementNum(), vt);
                // SLOPE-SVC V row (V + slope*sumQ = target): the natural reactive fill STAYS in the row,
                // scaled by the slope and completed with the +v diagonal (row mode 2) — not an identity.
                boolean slopeRow = e.getType() == AcEquationType.BUS_TARGET_V
                        && network.getBus(e.getElementNum()).hasGeneratorsWithSlope();
                mode[r] = slopeRow ? GpuRowMode.BUS_TARGET_V_WITH_SLOPE : GpuRowMode.FIXED_VARIABLE;
                assign(es, targetMap, r, e.getColumn(), e.getType(), e.getElementNum());
            }
        }
        return new GpuAcActivation(mode, targetMap);
    }

    /** PER_EQUATION_TYPE stopping thresholds per paired row, from the VARIABLE at each column (the
     *  batch pairs every equation with its dominant variable): {@code [0][r]} = the identity/slope-row
     *  threshold, {@code [1][r]} = the power-row threshold — mirroring OLF's
     *  PerEquationTypeStoppingCriteria type switch (power values MW/MVAr -> pu /100; dummy columns:
     *  identity = DUMMY_TARGET_P/Q power thresholds, coupled = ZERO_PHI/ZERO_V angle/voltage ones). */
    @SuppressWarnings("java:S107")
    public static double[][] extractRowCriteria(EquationSystem<AcVariableType, AcEquationType> es, int n,
                                                double maxP, double maxQ, double maxV, double maxAngle,
                                                double maxRatio, double maxSusceptance, double eps) {
        double[] fixedVariableRowMaxMismatch = new double[n];
        double[] equationRowMaxMismatch = new double[n];
        java.util.Arrays.fill(fixedVariableRowMaxMismatch, eps);
        java.util.Arrays.fill(equationRowMaxMismatch, eps);
        double sb = 100.0;
        for (var v : es.getIndex().getSortedVariablesToFind()) {
            int r = v.getRow();
            switch (v.getType()) {
                case BUS_V -> {
                    fixedVariableRowMaxMismatch[r] = maxV;                      // BUS_TARGET_V (incl. slope rows)
                    equationRowMaxMismatch[r] = maxQ / sb;                 // BUS_TARGET_Q / DISTR_Q
                }
                case BUS_PHI -> {
                    fixedVariableRowMaxMismatch[r] = maxAngle;                  // BUS_TARGET_PHI (slack)
                    equationRowMaxMismatch[r] = maxP / sb;                 // BUS_TARGET_P / BUS_DISTR_SLACK_P
                }
                case BRANCH_ALPHA1 -> {
                    fixedVariableRowMaxMismatch[r] = maxAngle;                  // BRANCH_TARGET_ALPHA1
                    equationRowMaxMismatch[r] = maxP / sb;                 // BRANCH_TARGET_P (row mode 3)
                }
                case BRANCH_RHO1 -> {
                    fixedVariableRowMaxMismatch[r] = maxRatio;                  // BRANCH_TARGET_RHO1 / DISTR_RHO
                    equationRowMaxMismatch[r] = maxRatio;
                }
                case SHUNT_B -> {
                    fixedVariableRowMaxMismatch[r] = maxSusceptance;            // SHUNT_TARGET_B / DISTR_SHUNT_B
                    equationRowMaxMismatch[r] = maxSusceptance;
                }
                case DUMMY_P -> {
                    fixedVariableRowMaxMismatch[r] = maxP / sb;                 // DUMMY_TARGET_P (identity, pinned 0)
                    equationRowMaxMismatch[r] = maxAngle;                  // ZERO_PHI coupling (mode 1)
                }
                case DUMMY_Q -> {
                    fixedVariableRowMaxMismatch[r] = maxQ / sb;                 // DUMMY_TARGET_Q
                    equationRowMaxMismatch[r] = maxV;                      // ZERO_V coupling
                }
                default -> {
                    // convEpsPerEq for anything else — matches OLF's default case
                }
            }
        }
        return new double[][] {fixedVariableRowMaxMismatch, equationRowMaxMismatch};
    }

    /** OLF's PerEquationTypeStoppingCriteria threshold for an equation TYPE (its checkEquation switch, power values
     *  in per unit). */
    @SuppressWarnings("java:S107")
    static double thresholdOf(AcEquationType type, double maxP, double maxQ, double maxV, double maxAngle,
                              double maxRatio, double maxSusceptance, double eps) {
        double sb = 100.0;
        return switch (type) {
            case BRANCH_TARGET_P, BUS_TARGET_P, DUMMY_TARGET_P, BUS_DISTR_SLACK_P -> maxP / sb;
            case BRANCH_TARGET_Q, BUS_TARGET_Q, DISTR_Q, DUMMY_TARGET_Q -> maxQ / sb;
            case BUS_TARGET_V, ZERO_V -> maxV;
            case BRANCH_TARGET_RHO1, DISTR_RHO -> maxRatio;
            case DISTR_SHUNT_B, SHUNT_TARGET_B -> maxSusceptance;
            case BUS_TARGET_PHI, ZERO_PHI, BRANCH_TARGET_ALPHA1 -> maxAngle;
            default -> eps;
        };
    }

    /**
     * PER_EQUATION_TYPE thresholds per device row, from the EQUATION each row hosts rather than from the variable at
     * its column: the variable-based defaults of {@link #extractRowCriteria(EquationSystem, int, double, double,
     * double, double, double, double, double)} for the slot a row does not use in the base, and, for the slot it does
     * use (identity for row mode 0 and 2, power for 1, 3 and 4), the threshold of the equation the activation placed
     * there. A relocated row then carries its own equation's threshold: a coupler-tree child's P balance on its
     * parent's DUMMY_P row (maxP, not the ZERO_PHI angle), the coupler's ZERO_V on the child's V row (maxV, not maxQ),
     * a transformer- or shunt-controlled bus's Q balance on the control variable's row (maxQ, not the ratio or
     * susceptance one).
     *
     * <p>Two placements change per scenario and get an ALTERNATE threshold with a SELECTOR row: a relocated child whose
     * parent coupler opens comes back HOME (its PHI / V rows hold its own balances again, the selector being the
     * coupler's dummy row, an identity in that scenario), and a cross identity (row mode 6: a child's voltage target on
     * its parent's DUMMY_Q row) is a voltage equation. Returns {fixedVariableRowMaxMismatch, equationRowMaxMismatch, alternateRowMaxMismatch, alternateRowMaxMismatchSelector (as double, -1 when
     * none)}.
     */
    @SuppressWarnings("java:S107")
    public static double[][] extractRowCriteria(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                GpuAcActivation activation, int n,
                                                double maxP, double maxQ, double maxV, double maxAngle,
                                                double maxRatio, double maxSusceptance, double eps) {
        double[][] base = extractRowCriteria(es, n, maxP, maxQ, maxV, maxAngle, maxRatio, maxSusceptance, eps);
        double[] fixedVariableRowMaxMismatch = base[0];
        double[] equationRowMaxMismatch = base[1];
        double[] alternateRowMaxMismatch = new double[n];
        double[] alternateRowMaxMismatchSelector = new double[n];
        java.util.Arrays.fill(alternateRowMaxMismatchSelector, -1);
        java.util.Arrays.fill(alternateRowMaxMismatch, eps);
        java.util.Map<Integer, AcEquationType> typeOfColumn = new java.util.HashMap<>();
        for (SingleEquation<AcVariableType, AcEquationType> e : es.getIndex().getSortedSingleEquationsToSolve()) {
            typeOfColumn.put(e.getColumn(), e.getType());
        }
        int[] mode = activation.rowMode();
        int[] targetMap = activation.targetMap();
        for (int r = 0; r < n; r++) {
            AcEquationType type = targetMap[r] >= 0 ? typeOfColumn.get(targetMap[r]) : null;
            if (type == null) {
                continue;
            }
            double thr = thresholdOf(type, maxP, maxQ, maxV, maxAngle, maxRatio, maxSusceptance, eps);
            switch (mode[r]) {
                case 0, 2 -> fixedVariableRowMaxMismatch[r] = thr;
                case 6 -> alternateRowMaxMismatch[r] = thr;
                default -> equationRowMaxMismatch[r] = thr;
            }
        }
        // The coupler-tree relocation, per scenario: a child's own rows host the coupler's ZERO equations while the
        // coupler lives and the child's balances once it opens (the dummy rows are then DUMMY_TARGET identities,
        // whose identity thresholds are already the dummy variables' defaults).
        ExtractionColumns cols = ExtractionColumns.of(network, es);
        for (LfBus bus : network.getBuses()) {
            int b = bus.getNum();
            int pe = cols.spanningTreeParentBranch()[b];
            if (pe < 0) {
                continue;
            }
            int dummyPRow = cols.dummyPCol()[pe];
            int dummyQRow = cols.dummyQCol()[pe];
            int phi = cols.phiCol()[b];
            int v = cols.vCol()[b];
            if (phi >= 0 && dummyPRow >= 0) {
                alternateRowMaxMismatchSelector[phi] = dummyPRow;
                alternateRowMaxMismatch[phi] = maxP / 100.0;                  // BUS_TARGET_P back home
            }
            if (v >= 0 && dummyQRow >= 0) {
                alternateRowMaxMismatchSelector[v] = dummyQRow;
                alternateRowMaxMismatch[v] = maxQ / 100.0;                    // its reactive balance back home (its voltage target,
            }                                                // an identity, keeps the V row's identity threshold)
            if (dummyQRow >= 0) {
                alternateRowMaxMismatch[dummyQRow] = maxV;                           // the child's voltage target written crosswise (mode 6)
                equationRowMaxMismatch[dummyQRow] = maxQ / 100.0;                 // its reactive balance there, shared or pinned
            }
            if (dummyPRow >= 0) {
                equationRowMaxMismatch[dummyPRow] = maxP / 100.0;
            }
        }
        return new double[][] {fixedVariableRowMaxMismatch, equationRowMaxMismatch, alternateRowMaxMismatch, alternateRowMaxMismatchSelector};
    }

    /**
     * Per-row variable type for the MAX_VOLTAGE_CHANGE device step scaling: OLF's {@link AcVariableType} ordinal of
     * each row's variable ({@link GpuAcVariableType}). The kernel clamps the BUS_V steps to maxDv and the BUS_PHI steps
     * to maxDphi; every other type (BRANCH_RHO1, ...) is scaled by the global step factor but never a clamp source,
     * as {@code MaxVoltageChangeStateVectorScaling}, which only inspects BUS_V / BUS_PHI. Indexed by the variable row,
     * which is the column index of the Newton step; a row with no variable holds -1.
     */
    public static int[] extractVarType(EquationSystem<AcVariableType, AcEquationType> es, int n) {
        int[] varType = new int[n];
        java.util.Arrays.fill(varType, -1);
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            varType[v.getRow()] = v.getType().ordinal();
        }
        return varType;
    }

    /**
     * The buses currently PINNED as remote generator-voltage-control members — exactly the set this extractor
     * gives row mode 4 above. A pin is a REACTIVE-LIMIT event: OLF renormalizes the group's reactive keys
     * ({@code updateReactiveKeys}) and fires NO equation-system event, so a cached extraction would keep its
     * pre-pin {@code qPct} and the device would distribute reactive power with stale weights. A caller that
     * caches an extraction across solves therefore watches this set and re-extracts when it changes: the fresh
     * pack's {@code distrQWeight} differs, which {@code GpuAcData.sameStructureAs} compares, so the structure is
     * REBUILT with OLF's renormalized keys instead of being value-refreshed in place.
     */
    public static java.util.BitSet pinnedRemoteMembers(LfNetwork network,
                                                      EquationSystem<AcVariableType, AcEquationType> es) {
        java.util.BitSet pinned = new java.util.BitSet();
        for (LfBus bus : network.getBuses()) {
            if (StableRowIndexing.row(es, bus.getNum(), AcEquationType.BUS_TARGET_Q) < 0) {
                continue;                                    // no active own-Q balance: not pinned
            }
            boolean pilotSelf = bus.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .map(c -> c.getControlledBus() == bus).orElse(false);
            if (GeneratorVoltageControlRoles.isRemoteGeneratorVoltageControlControllerElement(bus)
                    && (pilotSelf || !GeneratorVoltageControlRoles.isFreeRemoteGeneratorVoltageControlMember(bus))) {
                pinned.set(bus.getNum());
            }
        }
        return pinned;
    }
}
