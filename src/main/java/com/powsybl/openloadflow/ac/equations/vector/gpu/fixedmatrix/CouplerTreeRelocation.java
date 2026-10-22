/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.VoltageControl;
import com.powsybl.openloadflow.network.gpu.GeneratorVoltageControlRoles;
import com.powsybl.openloadflow.util.gpu.GpuTrace;

import java.util.List;

/**
 * docs/gpu-vs-olf.md section 2.3 (zero-impedance branches: relocation): the base decision of which zero-impedance
 * spanning-tree coupler carries its child bus's power balances on its {@code DUMMY_P} / {@code DUMMY_Q} rows, the
 * coupler's {@code ZERO_V} / {@code ZERO_PHI} going onto the child's own rows, decided once per run from the base
 * tree; and its dead-end stub special case.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class CouplerTreeRelocation {

    private CouplerTreeRelocation() {
    }

    /**
     * The COUPLER-TREE RELOCATION, the general form of the dead-end case below: per zero-impedance spanning tree,
     * every non-root bus's two power equations move onto its PARENT coupler's {@code DUMMY_P} / {@code DUMMY_Q}
     * columns, and every coupler's {@code ZERO_PHI} / {@code ZERO_V} move onto its CHILD bus's {@code PHI} / {@code V}.
     * Each of those four pairings has a structurally non-zero diagonal (+-1, or +-rho for ZERO_V), and the root keeps
     * its own rows, whose diagonal comes from its impedant branches.
     *
     * <p>Why: with the stock pairing a coupler's two equations sit on its own dummy rows, which neither depends on,
     * so a chain of couplers chains zero diagonals. The numeric row repair (repairZeroDiagonalRowsBatch) rescues short chains but
     * not longer ones: measured against OLF's own Newton on an all-generator chain, the device's FIRST step is exact
     * at 3 couplers (2.8e-16) and wrong from 7 on (1.6e+04 at 7, 5.5e+04 at 15) - no zero diagonal, no tiny one and
     * no empty column at that point, just an unpivoted factorization losing the chain. rte6515's busbar scope (366
     * retained couplers) is the same defect at scale: the base load flow's factorization came back wrong
     * (||J.dx-B||inf / ||F||inf = 2.1e+08) and every scenario of a 1000-section window had to fall back to the CPU.
     *
     * @return {@code parentEdgeOfBus[busNum]} = the coupler whose dummy columns host that bus's power equations
     *         (-1 = a root, or not in a relocated tree), and {@code childBusOfEdge[branchNum]} = the bus whose
     *         {@code PHI} / {@code V} host that coupler's zero equations (-1 = not relocated).
     */

    public static int[][] spanningTreeRelocation(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                    int[] phiCol, int[] vCol, int[] qRow, int[] dummyPCol, int[] dummyQCol) {
        int nBuses = network.getBuses().size();
        int nBranches = network.getBranches().size();
        int[] parentEdgeOfBus = new int[nBuses];
        int[] childBusOfEdge = new int[nBranches];
        java.util.Arrays.fill(parentEdgeOfBus, -1);
        java.util.Arrays.fill(childBusOfEdge, -1);
        // Adjacency over the zero-impedance SPANNING-TREE edges only: a non-tree coupler carries no reactive flow
        // and no zero equations to relocate (OLF gives it no DUMMY variables either).
        java.util.Map<Integer, java.util.List<LfBranch>> treeEdges = new java.util.HashMap<>();
        for (LfBranch br : network.getBranches()) {
            if (!br.isZeroImpedance(LoadFlowModel.AC) || !br.isSpanningTreeEdge(LoadFlowModel.AC)
                    || br.getBus1() == null || br.getBus2() == null) {
                continue;
            }
            treeEdges.computeIfAbsent(br.getBus1().getNum(), k -> new java.util.ArrayList<>()).add(br);
            treeEdges.computeIfAbsent(br.getBus2().getNum(), k -> new java.util.ArrayList<>()).add(br);
        }
        // A child's own rows must be FREE for the pairing: its V row takes its coupler's ZERO_V once its reactive
        // equation moves to the parent's DUMMY_Q, so a bus whose V row hosts ANOTHER bus's reactive balance (a
        // group's free controller carries the pilot's) or whose reactive equation is relocated elsewhere (a
        // transformer- or shunt-controlled bus) keeps its rows, and the numeric repair covers it as before.
        boolean[] hostsAnotherReactiveRow = new boolean[nBuses];
        for (int other = 0; other < nBuses; other++) {
            if (qRow[other] < 0) {
                continue;
            }
            for (int bn = 0; bn < nBuses; bn++) {
                if (bn != other && qRow[other] == vCol[bn]) {
                    hostsAnotherReactiveRow[bn] = true;
                }
            }
        }
        // ROOT ORDER: a bus that cannot be a CHILD roots its tree when it can (an impedant branch gives its P row a
        // diagonal), the reference bus first: its PHI row holds BUS_TARGET_PHI, so a coupler's ZERO_PHI cannot take
        // it at all (made a child, it is a double row assignment and the whole extraction is refused). Then a bus
        // whose V row is an identity, hosts another bus's reactive balance, or does not hold its own. Rooted
        // anywhere else the tree is CUT at such a bus: the coupler between it and its parent keeps the stock
        // placement, two zero diagonals the numeric repair fills from a live balance row, and that borrowed pair is
        // an exactly singular 2x2 block an unpivoted factorization cannot eliminate in either order (network.xiidm
        // CKYD: the PV section CKYD_2 cut CLDK off a CKYD_0 root, and every inaccurate batched solve of a
        // 258-section window had its worst residual on CLDK's or its neighbours' rows).
        java.util.List<LfBus> rootOrder = new java.util.ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            if (treeEdges.containsKey(bus.getNum()) && hasImpedantBranch(bus) && hostsPhiIdentity(bus, es)) {
                rootOrder.add(bus);
            }
        }
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            if (treeEdges.containsKey(bn) && hasImpedantBranch(bus) && !hostsPhiIdentity(bus, es)
                    && (hostsVoltageIdentity(bus, es) || hostsAnotherReactiveRow[bn]
                        || !(qRow[bn] == vCol[bn] || GeneratorVoltageControlRoles.isRemoteGeneratorVoltageControlControllerElement(bus)))) {
                rootOrder.add(bus);
            }
        }
        rootOrder.addAll(network.getBuses());
        boolean[] seen = new boolean[nBuses];
        for (LfBus bus : rootOrder) {
            int rootNum = bus.getNum();
            if (seen[rootNum] || !treeEdges.containsKey(rootNum) || !hasImpedantBranch(bus)) {
                continue;                                    // the root's own P row needs a real dP/dPHI diagonal
            }
            // Breadth-first from the root: the edge that first reaches a bus is its parent, and the bus is that
            // edge's child. A tree, so every bus is reached once and the pairing is a permutation.
            java.util.Deque<LfBus> queue = new java.util.ArrayDeque<>();
            java.util.List<Integer> childrenOfThisTree = new java.util.ArrayList<>();
            seen[rootNum] = true;
            queue.add(bus);
            while (!queue.isEmpty()) {
                LfBus b = queue.poll();
                for (LfBranch br : treeEdges.getOrDefault(b.getNum(), List.of())) {
                    LfBus other = br.getBus1() == b ? br.getBus2() : br.getBus1();
                    int otherNum = other.getNum();
                    if (seen[otherNum] || phiCol[otherNum] < 0 || vCol[otherNum] < 0
                            || hostsVoltageIdentity(other, es) || hostsPhiIdentity(other, es)) {
                        continue;                            // a pilot's V row is an identity: ZERO_V cannot take it;
                    }                                        // nor ZERO_PHI the reference bus's PHI row
                    seen[otherNum] = true;
                    parentEdgeOfBus[otherNum] = br.getNum();
                    childBusOfEdge[br.getNum()] = otherNum;
                    childrenOfThisTree.add(otherNum);
                    queue.add(other);
                }
            }
            if (GpuTrace.ZITREE_TRACE.isOn()) {
                System.err.println("ZITREE root=" + bus.getId() + " couplers=" + childrenOfThisTree.size());
            }
        }
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            int parent = parentEdgeOfBus[bn];
            if (parent < 0) {
                continue;
            }
            // CANONICAL, i.e. independent of the PV/PQ state (the lesson of the canonical extraction): a remote gen-VC
            // controller element's q is SUPPRESSED at extraction (qRow = -1) and its own V row is where its balance
            // comes back — its DISTR_Q while it shares, its own Q once a limit switches it to PQ — so its row is the home of
            // its reactive balance whether it is PV or PQ. Reading the PV/PQ state here made the base and the
            // security analysis relocate DIFFERENTLY (measured: 44 of 47 couplers in the base extraction against
            // 41 in the SA's, the three differing ones exactly the pilot-self groups the base load flow switches to PQ), so
            // the two ran on different structures and every scenario failed.
            boolean ownReactiveOnItsVRow = qRow[bn] == vCol[bn] || GeneratorVoltageControlRoles.isRemoteGeneratorVoltageControlControllerElement(bus);
            if (dummyPCol[parent] < 0 || dummyQCol[parent] < 0 || hostsAnotherReactiveRow[bn]
                    || !ownReactiveOnItsVRow) {
                parentEdgeOfBus[bn] = -1;
                childBusOfEdge[parent] = -1;
            }
        }
        if (GpuTrace.ZITREE_TRACE.isOn()) {
            int tree = 0;
            int relocated = 0;
            for (LfBranch br : network.getBranches()) {
                if (br.isZeroImpedance(LoadFlowModel.AC) && br.isSpanningTreeEdge(LoadFlowModel.AC)
                        && br.getBus1() != null && br.getBus2() != null) {
                    tree++;
                    if (childBusOfEdge[br.getNum()] >= 0) {
                        relocated++;
                    } else {                                 // why each stock coupler stayed stock: both ends' flags
                        System.err.println("ZITREE_STOCK coupler=" + br.getId()
                                + " dummyCols=" + (dummyPCol[br.getNum()] >= 0 && dummyQCol[br.getNum()] >= 0)
                                + " " + spanningTreeTraceBus(br.getBus1(), es, parentEdgeOfBus, hostsAnotherReactiveRow, qRow, vCol)
                                + " " + spanningTreeTraceBus(br.getBus2(), es, parentEdgeOfBus, hostsAnotherReactiveRow, qRow, vCol));
                    }
                }
            }
            System.err.println("ZITREE_SUMMARY treeCouplers=" + tree + " relocated=" + relocated
                    + " stock=" + (tree - relocated));
        }
        return new int[][] {parentEdgeOfBus, childBusOfEdge};
    }

    /** OLF_GPU_ZITREE_TRACE: one coupler end's placement flags, the inputs of {@link #spanningTreeRelocation}'s decisions. */
    private static String spanningTreeTraceBus(LfBus bus, EquationSystem<AcVariableType, AcEquationType> es, int[] parentEdgeOfBus,
                                         boolean[] hostsAnotherReactiveRow, int[] qRow, int[] vCol) {
        int bn = bus.getNum();
        return bus.getId() + "{" + (parentEdgeOfBus[bn] >= 0 ? "child" : "root")
                + (hostsPhiIdentity(bus, es) ? ",phiId" : "")
                + (hostsVoltageIdentity(bus, es) ? ",vId" : "")
                + (hostsAnotherReactiveRow[bn] ? ",hostsOtherQ" : "")
                + (qRow[bn] == vCol[bn] ? ",ownQ" : GeneratorVoltageControlRoles.isRemoteGeneratorVoltageControlControllerElement(bus) ? ",remoteCtl" : ",qElsewhere")
                + (hasImpedantBranch(bus) ? ",imp" : ",noImp") + "}";
    }

    /**
     * True when this bus owns a {@code BUS_TARGET_V} identity row: it is where OLF puts the voltage equation of
     * a control it takes part in, so that row is not free for a coupler's {@code ZERO_V}.
     *
     * <p>INDEPENDENT of the PV/PQ state, which asking whether the equation is active right now is not — a pilot whose
     * controllers a reactive limit has switched to PQ has no active identity, yet the row is still its own and comes
     * back the moment they switch back to PV. The owner is OLF's own answer, {@link
     * VoltageControl#findMainVisibleControlledBus()}: of the controls merged over a zero-impedance component
     * exactly ONE is main and visible, and its controlled bus carries the single identity the component gets.
     * Asking instead whether the bus has a control AT ALL excluded every merged dependent controlled bus and
     * cost most of the coverage (measured on rte6515's full busbar scope: 74 of 366 couplers relocated against
     * 329, 793 zero diagonals left and the base's first factorization off by a ratio of 1.3e+10).
     */
    private static boolean hostsVoltageIdentity(LfBus bus, EquationSystem<AcVariableType, AcEquationType> es) {
        return StableRowIndexing.row(es, bus.getNum(), AcEquationType.BUS_TARGET_V) >= 0
                || identityOwner(bus.getGeneratorVoltageControl().orElse(null), bus)
                || identityOwner(bus.getTransformerVoltageControl().orElse(null), bus)
                || identityOwner(bus.getShuntVoltageControl().orElse(null), bus);
    }

    /** True when this bus's {@code PHI} row holds the reference angle ({@code BUS_TARGET_PHI}). */
    private static boolean hostsPhiIdentity(LfBus bus, EquationSystem<AcVariableType, AcEquationType> es) {
        return StableRowIndexing.row(es, bus.getNum(), AcEquationType.BUS_TARGET_PHI) >= 0;
    }

    private static boolean identityOwner(VoltageControl<?> voltageControl, LfBus bus) {
        return voltageControl != null && voltageControl.findMainVisibleControlledBus().map(b -> b == bus).orElse(false);
    }

    private static boolean hasImpedantBranch(LfBus bus) {
        for (LfBranch br : bus.getBranches()) {
            if (!br.isZeroImpedance(LoadFlowModel.AC)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Which side of a zero-impedance branch is a DEAD-END STUB that is SAFE to relocate: a bus whose
     * only branch is this one, with no shunt, no generator and no voltage-dependent load, and with a
     * plain P/Q balance of its own. A {@link GpuTwoSides}: ONE, TWO, or NONE for neither (or both, which would be
     * a two-bus island and is left alone).
     *
     * <p>Why relocate at all: a zero-impedance branch puts ZERO_V (v1−rho·v2) on the DUMMY_P row and
     * ZERO_PHI (ph1−ph2) on the DUMMY_Q row, and neither depends on its own dummy — two structurally
     * zero diagonals, harmless while isolated. A dead-end stub adds two more, because its P and Q
     * balances then carry ONLY ∓DUMMY_P / ∓DUMMY_Q, and the four zero pivots CHAIN. cuDSS's AMD
     * ordering is symmetric and assumes a non-zero diagonal; on that chain it perturbs the pivot and
     * propagates the reciprocal (measured DUMMY_P = -1.220703125e+09 and phi = -1.220703125e+22
     * against exact answers of -0.05 and 0). Pairing the same four equations with different columns
     * — the stub's P balance on the DUMMY_P row, its Q balance on the DUMMY_Q row, ZERO_V on the
     * stub's v row and ZERO_PHI on its phi row — makes every diagonal non-zero and the factorization
     * well behaved, with AMD retained.
     *
     * <p>WHY THE CONDITIONS ARE THIS STRICT. Moving a bus's P balance off {@code phiCol} is only safe
     * when NOTHING ELSE is keyed on that row, and several packs are: {@code distributedSlackBusPhiRow} writes each
     * distributed-slack participant's target at {@code phiCol[bn]}, the reactive-capability curve
     * pack has its own P rows, the HVDC family scatters into both ends' P rows, and a load model
     * emits a load model term at the bus's P and Q rows. Excluding generators covers DS, the curves and
     * HVDC at once, because OLF models a converter station as an {@code LfGenerator}. A first,
     * looser version of this test (branch count + shunt only) fired on rte6515 buses that carry
     * those packs and moved the base error from 3.9e-08 to 5.0e-05.
     */
    public static int zeroImpedanceDeadEndSide(EquationSystem<AcVariableType, AcEquationType> es, LfBranch br,
                             int[] phiCol, int[] vCol, int[] qRow) {
        if (!br.isSpanningTreeEdge(LoadFlowModel.AC)) {
            return GpuTwoSides.NONE;                         // no ZERO_V/ZERO_PHI to relocate
        }
        boolean s1 = isZeroImpedanceDeadEnd(es, br.getBus1(), br, phiCol, vCol, qRow);
        boolean s2 = isZeroImpedanceDeadEnd(es, br.getBus2(), br, phiCol, vCol, qRow);
        if (s1 == s2) {
            return GpuTwoSides.NONE;                         // neither, or a two-bus island
        }
        return s1 ? GpuTwoSides.ONE : GpuTwoSides.TWO;
    }

    private static boolean isZeroImpedanceDeadEnd(EquationSystem<AcVariableType, AcEquationType> es, LfBus bus,
                                       LfBranch br, int[] phiCol, int[] vCol, int[] qRow) {
        if (bus == null) {
            return false;
        }
        int bn = bus.getNum();
        // Exactly one incident branch, counted regardless of state: a base-open branch still packs an
        // open-branch view that scatters into this bus's Q row, and a contingency can reclose one,
        // while the STRUCTURE is built once.
        if (bus.getBranches().size() != 1 || bus.getBranches().get(0) != br) {
            return false;
        }
        if (bus.getShunt().isPresent() || bus.getControllerShunt().isPresent() || bus.getSvcShunt().isPresent()) {
            return false;                                    // a shunt scatters into this bus's Q row
        }
        if (!bus.getGenerators().isEmpty()) {
            return false;                                    // DS rows, curve P rows, HVDC stations
        }
        for (com.powsybl.openloadflow.network.LfLoad load : bus.getLoads()) {
            if (load.getLoadModel().isPresent()) {
                return false;                                // load model term at this bus's P and Q rows
            }
        }
        return phiCol[bn] >= 0 && vCol[bn] >= 0 && qRow[bn] == vCol[bn]
                && StableRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_P) >= 0
                && StableRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_Q) >= 0;
    }

    /**
     * Applies the coupler-tree relocation ({@link #spanningTreeRelocation}) to the row arrays, after the candidate
     * tables ({@link ReactiveRelocationTables#build}): each zero-impedance branch's dead-end side
     * ({@link #zeroImpedanceDeadEndSide}, read on the {@code qRow} the tables left), then each relocated child's P
     * row, Q row and plain own-row alternate onto its parent coupler's {@code DUMMY_P} / {@code DUMMY_Q}, then drops
     * from {@code spanningTreeEdges[1]} a coupler whose child's relocation did not survive the check. Mutates
     * {@code qRow}, {@code alternateQRow} and {@code spanningTreeEdges[1]}.
     */
    public static AppliedRows applyToRows(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                          int[] phiCol, int[] vCol, int[] dummyPCol, int[] dummyQCol,
                                          int[] qRow, int[] alternateQRow, int[][] spanningTreeEdges) {
        int nBranches = network.getBranches().size();
        // Per zero-impedance branch, which side is a DEAD-END STUB safe to relocate (a GpuTwoSides, NONE = none):
        // CouplerTreeRelocation#zeroImpedanceDeadEndSide. GpuAcActivationExtractor derives the same answer from the
        // same helper, so the scatter destinations and the target map name the same rows.
        int[] zeroImpedanceDeadEndSides = new int[nBranches];
        java.util.Arrays.fill(zeroImpedanceDeadEndSides, GpuTwoSides.NONE);
        for (LfBranch br : network.getBranches()) {
            if (br.isZeroImpedance(LoadFlowModel.AC)) {
                zeroImpedanceDeadEndSides[br.getNum()] = CouplerTreeRelocation.zeroImpedanceDeadEndSide(es, br, phiCol, vCol, qRow);
            }
        }
        int[] spanningTreeParentBranch = spanningTreeEdges[0];
        int[] spanningTreeChildBus = spanningTreeEdges[1];
        int[] pRow = phiCol.clone();
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            int parent = spanningTreeParentBranch[bn];
            if (parent < 0) {
                continue;
            }
            pRow[bn] = dummyPCol[parent];
            if (qRow[bn] == vCol[bn]) {                       // a DISTR_Q child moves through distrQRow instead
                qRow[bn] = dummyQCol[parent];
            }
            // ...and so does the ALTERNATE row, the one every family writes the bus's OWN q to when a reactive
            // limit puts it in mode 4. For a relocated child that switch to PQ lands on this coupler's DUMMY_Q row (the
            // native pass diverts it there through reactiveLimitReactivePowerRow, since that is where its balance lives), so leaving the
            // alternate on the bus's own BUS_V row sent its own q onto the row now holding the coupler's ZERO_V
            // and left the DUMMY_Q column held by nothing but the DISTR_Q terms the mode-4 flip gates off —
            // a DEAD COLUMN. Measured on rte6515's full busbar scope: 9 mode-4 rows on relocated couplers of
            // VL-5997/VL-6003/VL-6194/VL-6229, deadCols=2 (first 14417 = DUMMY_Q(BUS-6003_COUPLER_2)),
            // tinyDiag=7, and 6616 of 6876 scenarios rejecting their second iteration at a ratio of 1e+10.
            // Only the plain own-row alternate moves: a candidate-table pilot's is encoded as -2-group.
            if (alternateQRow[bn] == vCol[bn] && dummyQCol[parent] >= 0) {
                alternateQRow[bn] = dummyQCol[parent];
            }
        }
        for (int br = 0; br < nBranches; br++) {
            int child = spanningTreeChildBus[br];
            if (child >= 0 && spanningTreeParentBranch[child] != br) {
                spanningTreeChildBus[br] = -1;                          // its child's relocation did not survive the check
            }
        }
        return new AppliedRows(pRow, zeroImpedanceDeadEndSides);
    }

    /**
     * The rows {@link #applyToRows} computes: each bus's P-balance row ({@code pRow}) and each zero-impedance
     * branch's dead-end side ({@code zeroImpedanceDeadEndSides}, a GpuTwoSides, NONE = none).
     */
    public record AppliedRows(int[] pRow, int[] zeroImpedanceDeadEndSides) {
    }
}
