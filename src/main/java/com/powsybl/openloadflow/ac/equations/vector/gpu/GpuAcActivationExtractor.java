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
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.SingleEquation;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.VoltageControl;
import com.powsybl.openloadflow.network.LfNetwork;

/**
 * Extracts what ACTIVATION decides — the per-row power-equation/identity mode, the
 * OLF-column→stable-row target mapping ({@link GpuAcActivation}) and the per-row variable
 * kind used by the MAX_VOLTAGE_CHANGE device step scaling. Rebuilt only when an activation
 * event fired; the extracted {@link GpuAcData} structure stays valid across PV→PQ switching
 * (see {@link GpuAcRowIndexing} for the stable-row policy this reuses).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcActivationExtractor {

    /** {@code OLF_GPU_DISTRQ_DUMP=<busIdPrefix>}: print the DISTR_Q row assignment for a group —
     *  which row each controller's equation lands on, its own BUS_V row, the group's controlled bus
     *  and which members are enabled. Lines carry the extraction pass number: OLF keys DISTR_Q on
     *  {@code (bus, type)}, so one bus has at most ONE such equation per pass and two lines for the
     *  same bus are two passes, not two equations. */
    private static final String DISTR_Q_DUMP = System.getenv("OLF_GPU_DISTRQ_DUMP");

    /** {@code OLF_GPU_QROW_TRACE=<row|busIdPrefix>[,…]}: see the claim trace below. Read once — the
     *  row loop runs per extraction over every bus, and the two traces must agree on the pass number. */
    private static final String QROW_TRACE = System.getenv("OLF_GPU_QROW_TRACE");

    private static final java.util.concurrent.atomic.AtomicInteger PASS = new java.util.concurrent.atomic.AtomicInteger();

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
        java.util.Arrays.fill(mode, 1);
        int[] targetMap = new int[n];
        java.util.Arrays.fill(targetMap, -1);
        // ONE pass number for both traces, taken before either writes, so a QROW_CLAIM and a
        // DISTR_Q line from the same extraction carry the same number. Comparing an EARLY pass's
        // claim against a LATE pass's DISTR_Q state (the mistake this entry made five times, across
        // two different runs) then shows up as the pass numbers differing rather than as a result.
        int pass = DISTR_Q_DUMP == null && QROW_TRACE == null ? 0 : PASS.incrementAndGet();
        int nBranches = network.getBranches().size();
        int[] branchRhoCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.BRANCH_RHO1);
        int[] shuntBCol = GpuAcRowIndexing.branchVarCols(es, network.getShunts().size(), AcVariableType.SHUNT_B);
        int[] vCol = new int[network.getBuses().size()];
        for (LfBus bus : network.getBuses()) {
            vCol[bus.getNum()] = GpuAcRowIndexing.col(es, bus.getNum(), AcVariableType.BUS_V);
        }
        int[] qRow = GpuAcRowIndexing.computeQRow(network, es, vCol, branchRhoCol, shuntBCol);
        int[] phiCol = new int[network.getBuses().size()];
        for (LfBus bus : network.getBuses()) {
            phiCol[bus.getNum()] = GpuAcRowIndexing.col(es, bus.getNum(), AcVariableType.BUS_PHI);
        }
        // DEAD-END STUB relocation, the other half of the one in GpuAcFamilyPacker: both sides must
        // name the SAME row for each equation. assign() refuses a double write, so a disagreement
        // fails loudly here instead of silently dropping a constraint.
        int[] dummyPCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.DUMMY_P);
        int[] dummyQCol = GpuAcRowIndexing.branchVarCols(es, nBranches, AcVariableType.DUMMY_Q);
        // The coupler-tree relocation, from the same helper ExtractionColumns uses, so the scatter destinations and
        // this target map name the same rows: a child bus's power equations on its parent coupler's dummy columns,
        // that coupler's zero equations on the child's own PHI / V.
        int[][] ziTree = GpuAcRowIndexing.ziTreeRelocation(network, es, phiCol, vCol, qRow, dummyPCol, dummyQCol);
        int[] stubBranchOfBus = ziTree[0];
        int[] stubBusOfBranch = ziTree[1];
        for (LfBus bus : network.getBuses()) {
            int stubBr = stubBranchOfBus[bus.getNum()];
            int p = GpuAcRowIndexing.row(es, bus.getNum(), AcEquationType.BUS_TARGET_P);
            if (p >= 0) {
                assign(es, targetMap, stubBr >= 0 ? dummyPCol[stubBr] : phiCol[bus.getNum()], p,
                        AcEquationType.BUS_TARGET_P, bus.getNum());
            }
            int q = GpuAcRowIndexing.row(es, bus.getNum(), AcEquationType.BUS_TARGET_Q);
            if (q >= 0) {
                // Q row = V row, the controlling rho row, or the DUMMY_Q row on a dead-end stub
                int qr = stubBr >= 0 ? dummyQCol[stubBr] : qRow[bus.getNum()];
                GeneratorVoltageControl mainVc = GpuAcRowIndexing.mainGenVc(bus);
                if (stubBr < 0 && mainVc != null && GpuAcRowIndexing.hasCandidateTable(mainVc, es)) {
                    // A group with a candidate table is extracted CANONICALLY (pin-state independent: every
                    // controller's frozen q row suppressed, the pilot's on the FIRST member), but this target map
                    // is per scenario, so the row is the one the in-kernel mode scan picks in THIS state: a pinned
                    // member's own V row (mode 4 below, its alternate dest); the pilot's balance on the first
                    // ENABLED member's V row (the carrier), or on its own V row once every controller is pinned
                    // (its V identity gone, the row is a power row). Measured: the canonical row collided with the
                    // pinned first member's own Q (deviceRemoteFreeMemberBasePinnedMatchesCpu & co., 12 tests).
                    qr = GpuAcRowIndexing.stateQRow(bus, vCol, stubBranchOfBus, dummyQCol);
                }
                assign(es, targetMap, qr, q, AcEquationType.BUS_TARGET_Q, bus.getNum());
                // ACTIVE own-Q balance at a (non-free) remote gen-VC member = the RL-PINNED state:
                // row mode 4, so the family fills route the member's own q here (the frozen dest is
                // canonically SUPPRESSED) and the DISTR_Q emissions at this row gate off (plan-B2).
                // The SAME predicate computeQRow used to hand this bus its own Q row. It used
                // isRemoteGenVcControllerElement, which INCLUDES a controller that is also the
                // group's controlled bus (its javadoc says so); this used isRemoteGenVcMember,
                // which is controlledBus != bus and therefore EXCLUDES the pilot. For a PINNED
                // pilot-that-is-also-a-controller the two disagreed: computeQRow gave it its own Q
                // row, this left the row at mode 1, so the mode-4 gate in fillDistrQBatch never
                // fired and the existence-keyed DISTR_Q kind-3 terms stayed live ALONGSIDE the
                // zero-impedance family's direct +-DUMMY_Q into the same row. The two cancel: on
                // rte6515 BUS-24_COUPLER_1 the kind-3 pair summed to +1.0 against the zi coupling's
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
                boolean enrolled = pilotSelf && GpuAcRowIndexing.pilotSelfRotationEnabled()
                        && bus.getGeneratorVoltageControl()
                                .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                                .map(c -> c.getMergedControllerElements().size() >= 2
                                        && c.getMergedControllerElements().stream().allMatch(m -> vCol[m.getNum()] >= 0))
                                .orElse(false);
                // `enrolled` requires pilotSelf, so for a MEMBER mode4 reduces to
                // !isFreeRemoteGenVcMember, and "free" is the FIRST ENABLED member — a group whose members
                // are all pinned has no free member and every member's row is stamped mode 4 into
                // baseRowMode, which every scenario without an explicit per-scenario mode inherits. That is
                // CORRECT for a genuine Q-limit pin (the 20 base-pinned fixtures rely on it). What used to
                // reach here wrongly was a TRANSIENT MIN_REALISTIC_V freeze the device should never have
                // made: see GpuAcOuterLoopPacks.isOlfRobustRemoteController (tasks #66), fixed at the SOURCE
                // by exempting a merged member from robust mode as OLF's own predicate does, not here.
                boolean mode4 = GpuAcRowIndexing.isRemoteGenVcControllerElement(bus)
                        && (pilotSelf || !GpuAcRowIndexing.isFreeRemoteGenVcMember(bus)) && !enrolled;
                // OLF_GPU_QROW_TRACE=<row|busIdPrefix>[,…]: WHO claims a reactive row, and on what grounds.
                // A relocated member's DISTR_Q is assigned to its parent coupler's DUMMY_Q row, and so is
                // any bus whose qRow maps there with an ACTIVE BUS_TARGET_Q — two equations, one row, and
                // the mode-4 own-Q wins, dropping the sharing (tasks #66). The gate below keys on an
                // ACTIVE BUS_TARGET_Q, and in OLF's base those are INACTIVE for the measured group, so
                // this prints the bus, the column it read, the target and the predicate to say which bus
                // and which extraction pass actually claimed the row.
                for (String tok : QROW_TRACE == null ? new String[0] : QROW_TRACE.split(",")) {
                    // A row number OR a bus-id prefix. The prefix form is the one to use for a
                    // busbar-section scope: which switches stay retained follows the contingency
                    // LIST, so a row number is only valid at the exact scope it was read at, and
                    // keying on it forces the full-scope run for every measurement.
                    String t = tok.trim();
                    boolean hit = t.chars().allMatch(Character::isDigit)
                            ? Integer.parseInt(t) == qr : bus.getId().startsWith(t);
                    if (hit) {
                        // mainVc is what the row logic resolves (the MERGED group); the bus's own control is
                        // a dependent. Printing both, plus whether the controller is ENABLED, says whether the
                        // device is reading an inconsistent BUS_TARGET_Q/DISTR_Q pair or a genuinely pinned one.
                        var ownVc = bus.getGeneratorVoltageControl().orElse(null);
                        // WHICH of OLF's three disable paths left enabled=false, because they need
                        // opposite fixes and the pin dumps cannot tell them apart:
                        //   frozenQ=true  -> freezeGenerationTargetQAndDisableGeneratorVoltageControl,
                        //                    i.e. a PV->PQ pin (ReactiveLimitsOuterLoop or the tap
                        //                    manager's disableGeneratorVoltageControlsUnderMax...).
                        //   monitoringV   -> a MONITORING_VOLTAGE generator (an SVC with a stand-by
                        //                    automaton): LfNetworkLoaderImpl creates the control
                        //                    DISABLED at LOAD and MonitoringVoltageOuterLoop enables
                        //                    it during the solve. Invisible to RL_SWITCH, which is why
                        //                    the arms' pin SETS matched while `enabled` did not (#66).
                        //   neither       -> LfContingency/NetworkCache, or a BusState restore.
                        boolean monitoringV = bus.getGenerators().stream().anyMatch(
                                g -> g.getGeneratorControlType() == com.powsybl.openloadflow.network.LfGenerator
                                        .GeneratorControlType.MONITORING_VOLTAGE);
                        System.err.printf("QROW_CLAIM pass=%d row=%d bus=%s busTargetQcol=%d distrQcol=%d "
                                + "mainVc=%s mainMembers=%d enabled=%b ownVc=%s ownMembers=%d ownMerge=%s "
                                + "frozenQ=%b monitoringV=%b qLimit=%s gens=%d candTable=%s "
                                + "remoteCtl=%b pilotSelf=%b freeMember=%b enrolled=%b -> mode4=%b%n",
                                pass, qr, bus.getId(), q,
                                GpuAcRowIndexing.row(es, bus.getNum(), AcEquationType.DISTR_Q),
                                mainVc == null ? "-" : mainVc.getControlledBus().getId(),
                                mainVc == null ? 0 : mainVc.getMergedControllerElements().size(),
                                mainVc != null && mainVc.isControllerEnabled(bus),
                                ownVc == null ? "-" : ownVc.getControlledBus().getId(),
                                ownVc == null ? 0 : ownVc.getMergedControllerElements().size(),
                                ownVc == null ? "-" : ownVc.getMergeStatus().toString(),
                                bus.isGenerationTargetQFrozen(), monitoringV,
                                bus.getQLimitType().map(Object::toString).orElse("-"),
                                bus.getGenerators().size(),
                                // WHY the group misses the canonical pin-state-independent path (#66)
                                GpuAcRowIndexing.candidateTableReason(mainVc, es),
                                GpuAcRowIndexing.isRemoteGenVcControllerElement(bus), pilotSelf,
                                GpuAcRowIndexing.isFreeRemoteGenVcMember(bus), enrolled, mode4);
                    }
                }
                if (mode4) {
                    mode[qr] = 4;
                }
            }
        }
        for (SingleEquation<AcVariableType, AcEquationType> e : es.getIndex().getSortedSingleEquationsToSolve()) {
            // POWER-equation rows (mode stays 1) whose stable row is a non-bus-P/Q variable — assembled by
            // the GPU DISTR_Q / zero-impedance passes; just map the target onto that variable's row.
            if (e.getType() == AcEquationType.DISTR_Q) {
                // the controller's BUS_V row — or, when the controller IS the group's controlled bus
                // (its own row already carries the pilot's V identity), the FREE controller's row
                int distrRow = GpuAcRowIndexing.distrQRow(es, network.getBus(e.getElementNum()),
                        stubBranchOfBus, dummyQCol);
                if (DISTR_Q_DUMP != null) {
                    LfBus cb = network.getBus(e.getElementNum());
                    if (cb.getId().startsWith(DISTR_Q_DUMP)) {
                        var vc = cb.getGeneratorVoltageControl().orElse(null);
                        System.err.printf("DISTRQ_ROW pass=%d controller=%s eqCol=%d assignedRow=%d ownVRow=%d "
                                + "controlledBus=%s enabled=%b qPct=%.9f members=%s%n",
                                pass, cb.getId(), e.getColumn(), distrRow,
                                GpuAcRowIndexing.col(es, cb.getNum(), AcVariableType.BUS_V),
                                vc == null ? "-" : vc.getControlledBus().getId(),
                                vc != null && vc.isControllerEnabled(cb),
                                cb.getRemoteControlReactivePercent(),
                                vc == null ? "-" : vc.getMergedControllerElements().stream()
                                        .map(b -> b.getId() + (vc.isControllerEnabled(b) ? "(on)" : "(off)")).toList());
                    }
                }
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
                // STOCK placement, crosswise (native ziZeroVRow / ziZeroPhiRow): zero-imp v1−rho·v2 sits on the
                // DUMMY_Q row and phi1−phi2 on the DUMMY_P row, so that each row's numeric repair from a donor
                // that carries its dummy column (a Q row, a P row) keeps an entry of its own in the donor's
                // diagonal column. The target travels with the equation: ZERO_PHI's is the branch's phase shift.
                case ZERO_V -> AcVariableType.DUMMY_Q;
                case ZERO_PHI -> AcVariableType.DUMMY_P;
                default -> null;
            };
            if (powerRowVar != null) {
                assign(es, targetMap, GpuAcRowIndexing.col(es, e.getElementNum(), powerRowVar),
                        e.getColumn(), e.getType(), e.getElementNum());
                continue;
            }
            if (e.getType() == AcEquationType.BRANCH_TARGET_P) {
                // CONTINUOUS phase control enabled: the controlled branch's P-flow equation sits at the
                // CONTROLLER's a1-paired row (row mode 3 — the cb family re-emits the P outputs there,
                // cbPcSide carrying the controller's a1 column, LOCAL and REMOTE alike).
                var branch = network.getBranchById(network.getBranches().stream()
                        .filter(b -> b.getNum() == e.getElementNum()).findFirst().orElseThrow().getId());
                var pc = branch.getPhaseControl().orElseThrow();
                int a1r = GpuAcRowIndexing.col(es, pc.getControllerBranch().getNum(), AcVariableType.BRANCH_ALPHA1);
                mode[a1r] = 3;
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
                int r = GpuAcRowIndexing.col(es, e.getElementNum(), vt);
                // SLOPE-SVC V row (V + slope*sumQ = target): the natural reactive fill STAYS in the row,
                // scaled by the slope and completed with the +v diagonal (row mode 2) — not an identity.
                boolean slopeRow = e.getType() == AcEquationType.BUS_TARGET_V
                        && network.getBus(e.getElementNum()).hasGeneratorsWithSlope();
                mode[r] = slopeRow ? 2 : 0;
                assign(es, targetMap, r, e.getColumn(), e.getType(), e.getElementNum());
            }
        }
        return new GpuAcActivation(mode, targetMap);
    }

    /**
     * Per-row variable kind for the MAX_VOLTAGE_CHANGE device step scaling: {@code 0} = BUS_PHI (clamp
     * the angle step to maxDphi), {@code 1} = BUS_V (clamp the magnitude step to maxDv), {@code 2} = any
     * other variable (BRANCH_RHO1, …) — scaled by the global step factor but never itself a clamp source,
     * mirroring {@code MaxVoltageChangeStateVectorScaling} (which only inspects BUS_V / BUS_PHI). Indexed by
     * the variable row, which is the column index of the Newton step.
     */
    /** PER_EQUATION_TYPE stopping thresholds per paired row, from the VARIABLE at each column (the
     *  batch pairs every equation with its dominant variable): {@code [0][r]} = the identity/slope-row
     *  threshold, {@code [1][r]} = the power-row threshold — mirroring OLF's
     *  PerEquationTypeStoppingCriteria type switch (power values MW/MVAr -> pu /100; dummy columns:
     *  identity = DUMMY_TARGET_P/Q power thresholds, coupled = ZERO_PHI/ZERO_V angle/voltage ones). */
    @SuppressWarnings("java:S107")
    public static double[][] extractRowCriteria(EquationSystem<AcVariableType, AcEquationType> es, int n,
                                                double maxP, double maxQ, double maxV, double maxAngle,
                                                double maxRatio, double maxSusceptance, double eps) {
        double[] thrIdent = new double[n];
        double[] thrPower = new double[n];
        java.util.Arrays.fill(thrIdent, eps);
        java.util.Arrays.fill(thrPower, eps);
        double sb = 100.0;
        for (var v : es.getIndex().getSortedVariablesToFind()) {
            int r = v.getRow();
            switch (v.getType()) {
                case BUS_V -> {
                    thrIdent[r] = maxV;                      // BUS_TARGET_V (incl. slope rows)
                    thrPower[r] = maxQ / sb;                 // BUS_TARGET_Q / DISTR_Q
                }
                case BUS_PHI -> {
                    thrIdent[r] = maxAngle;                  // BUS_TARGET_PHI (slack)
                    thrPower[r] = maxP / sb;                 // BUS_TARGET_P / BUS_DISTR_SLACK_P
                }
                case BRANCH_ALPHA1 -> {
                    thrIdent[r] = maxAngle;                  // BRANCH_TARGET_ALPHA1
                    thrPower[r] = maxP / sb;                 // BRANCH_TARGET_P (row mode 3)
                }
                case BRANCH_RHO1 -> {
                    thrIdent[r] = maxRatio;                  // BRANCH_TARGET_RHO1 / DISTR_RHO
                    thrPower[r] = maxRatio;
                }
                case SHUNT_B -> {
                    thrIdent[r] = maxSusceptance;            // SHUNT_TARGET_B / DISTR_SHUNT_B
                    thrPower[r] = maxSusceptance;
                }
                case DUMMY_P -> {
                    thrIdent[r] = maxP / sb;                 // DUMMY_TARGET_P (identity, pinned 0)
                    thrPower[r] = maxAngle;                  // ZERO_PHI coupling (mode 1)
                }
                case DUMMY_Q -> {
                    thrIdent[r] = maxQ / sb;                 // DUMMY_TARGET_Q
                    thrPower[r] = maxV;                      // ZERO_V coupling
                }
                default -> {
                    // convEpsPerEq for anything else — matches OLF's default case
                }
            }
        }
        return new double[][] {thrIdent, thrPower};
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
     * its parent's DUMMY_Q row) is a voltage equation. Returns {thrIdent, thrPower, thrAlt, thrSel (as double, -1 when
     * none)}.
     */
    @SuppressWarnings("java:S107")
    public static double[][] extractRowCriteria(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                GpuAcActivation activation, int n,
                                                double maxP, double maxQ, double maxV, double maxAngle,
                                                double maxRatio, double maxSusceptance, double eps) {
        double[][] base = extractRowCriteria(es, n, maxP, maxQ, maxV, maxAngle, maxRatio, maxSusceptance, eps);
        double[] thrIdent = base[0];
        double[] thrPower = base[1];
        double[] thrAlt = new double[n];
        double[] thrSel = new double[n];
        java.util.Arrays.fill(thrSel, -1);
        java.util.Arrays.fill(thrAlt, eps);
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
                case 0, 2 -> thrIdent[r] = thr;
                case 6 -> thrAlt[r] = thr;
                default -> thrPower[r] = thr;
            }
        }
        // The coupler-tree relocation, per scenario: a child's own rows host the coupler's ZERO equations while the
        // coupler lives and the child's balances once it opens (the dummy rows are then DUMMY_TARGET identities,
        // whose identity thresholds are already the dummy variables' defaults).
        ExtractionColumns cols = ExtractionColumns.of(network, es);
        for (LfBus bus : network.getBuses()) {
            int b = bus.getNum();
            int pe = cols.ziParentEdge()[b];
            if (pe < 0) {
                continue;
            }
            int dp = cols.dummyPCol()[pe];
            int dq = cols.dummyQCol()[pe];
            int phi = cols.phiCol()[b];
            int v = cols.vCol()[b];
            if (phi >= 0 && dp >= 0) {
                thrSel[phi] = dp;
                thrAlt[phi] = maxP / 100.0;                  // BUS_TARGET_P back home
            }
            if (v >= 0 && dq >= 0) {
                thrSel[v] = dq;
                thrAlt[v] = maxQ / 100.0;                    // its reactive balance back home (its voltage target,
            }                                                // an identity, keeps the V row's identity threshold)
            if (dq >= 0) {
                thrAlt[dq] = maxV;                           // the child's voltage target written crosswise (mode 6)
                thrPower[dq] = maxQ / 100.0;                 // its reactive balance there, shared or pinned
            }
            if (dp >= 0) {
                thrPower[dp] = maxP / 100.0;
            }
        }
        return new double[][] {thrIdent, thrPower, thrAlt, thrSel};
    }

    public static int[] extractVarType(EquationSystem<AcVariableType, AcEquationType> es, int n) {
        int[] varType = new int[n];
        java.util.Arrays.fill(varType, 2);
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            varType[v.getRow()] = switch (v.getType()) {
                case BUS_PHI -> 0;
                case BUS_V -> 1;
                default -> 2;
            };
        }
        return varType;
    }

    /**
     * The buses currently PINNED as remote generator-voltage-control members — exactly the set this extractor
     * gives row mode 4 above. A pin is a REACTIVE-LIMIT event: OLF renormalizes the group's reactive keys
     * ({@code updateReactiveKeys}) and fires NO equation-system event, so a cached extraction would keep its
     * pre-pin {@code qPct} and the device would distribute reactive power with stale weights. A caller that
     * caches an extraction across solves therefore watches this set and re-extracts when it changes: the fresh
     * pack's {@code dqWeight} differs, which {@code GpuAcData.sameStructureAs} compares, so the structure is
     * REBUILT with OLF's renormalized keys instead of being value-refreshed in place.
     */
    public static java.util.BitSet pinnedRemoteMembers(LfNetwork network,
                                                      EquationSystem<AcVariableType, AcEquationType> es) {
        java.util.BitSet pinned = new java.util.BitSet();
        for (LfBus bus : network.getBuses()) {
            if (GpuAcRowIndexing.row(es, bus.getNum(), AcEquationType.BUS_TARGET_Q) < 0) {
                continue;                                    // no active own-Q balance: not pinned
            }
            boolean pilotSelf = bus.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .map(c -> c.getControlledBus() == bus).orElse(false);
            if (GpuAcRowIndexing.isRemoteGenVcControllerElement(bus)
                    && (pilotSelf || !GpuAcRowIndexing.isFreeRemoteGenVcMember(bus))) {
                pinned.set(bus.getNum());
            }
        }
        return pinned;
    }
}
