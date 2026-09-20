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
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;

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
     * behind a debug switch. It must stay ALLOCATION-FREE on the happy path: the JAVA hybrid re-extracts
     * per scenario per outer-loop step, so a remembered description per row (String[n]) or an eagerly
     * built message per call is tens of thousands of array/string allocations on a 13040-row system.
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
     *  per row, costs a String[n] allocation on EVERY extraction, and this method runs per scenario per
     *  outer-loop step on the hybrid path. */
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
        int[] stubBranchOfBus = new int[network.getBuses().size()];
        int[] stubBusOfBranch = new int[nBranches];
        java.util.Arrays.fill(stubBranchOfBus, -1);
        java.util.Arrays.fill(stubBusOfBranch, -1);
        for (LfBranch br : network.getBranches()) {
            if (!br.isZeroImpedance(LoadFlowModel.AC)) {
                continue;
            }
            int side = GpuAcRowIndexing.ziDeadEndSide(es, br, phiCol, vCol, qRow);
            if (side != 0) {
                LfBus stub = side == 1 ? br.getBus1() : br.getBus2();
                stubBranchOfBus[stub.getNum()] = br.getNum();
                stubBusOfBranch[br.getNum()] = stub.getNum();
            }
        }
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
                assign(es, targetMap, qr, q, AcEquationType.BUS_TARGET_Q, bus.getNum());
                // ACTIVE own-Q balance at a (non-free) remote gen-VC member = the RL-PINNED state:
                // row mode 4, so the family fills route the member's own q here (the frozen dest is
                // canonically SUPPRESSED) and the DISTR_Q emissions at this row gate off (plan-B2).
                if (GpuAcRowIndexing.isRemoteGenVcMember(bus) && !GpuAcRowIndexing.isFreeRemoteGenVcMember(bus)) {
                    mode[qr] = 4;
                }
            }
        }
        int pass = DISTR_Q_DUMP == null ? 0 : PASS.incrementAndGet();
        for (SingleEquation<AcVariableType, AcEquationType> e : es.getIndex().getSortedSingleEquationsToSolve()) {
            // POWER-equation rows (mode stays 1) whose stable row is a non-bus-P/Q variable — assembled by
            // the GPU DISTR_Q / zero-impedance passes; just map the target onto that variable's row.
            if (e.getType() == AcEquationType.DISTR_Q) {
                // the controller's BUS_V row — or, when the controller IS the group's controlled bus
                // (its own row already carries the pilot's V identity), the FREE controller's row
                int distrRow = GpuAcRowIndexing.distrQRow(es, network.getBus(e.getElementNum()));
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
                case ZERO_V -> AcVariableType.DUMMY_P;       // zero-imp v1−rho·v2 sits on the DUMMY_P row
                case ZERO_PHI -> AcVariableType.DUMMY_Q;     // zero-imp phi1−phi2 sits on the DUMMY_Q row
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
}
