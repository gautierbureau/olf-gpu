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
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.ShuntVoltageControl;
import com.powsybl.openloadflow.network.TransformerVoltageControl;
import com.powsybl.openloadflow.network.VoltageControl;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * The outer-loop packs of the GPU extraction — the parts the batched device solver gates on a
 * non-empty pack (an empty pack means "do not run that outer loop", mirroring OLF's configured
 * outer-loop list): the DISTR_Q remote-reactive-distribution contributions, the DISTR_RHO ratio-
 * distribution contributions, the reactive-limits (PV↔PQ) pack and the distributed-slack pack.
 * All read the precomputed {@link ExtractionColumns} (and DISTR_Q the family element maps).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuAcOuterLoopPacks {

    private GpuAcOuterLoopPacks() {
    }

    /** DISTR_Q contributions in the flat device layout: {@code {elem, side, dqRow, kind}} per row + weight. */
    record DqPack(int[] dqElem, int[] dqSide, double[] dqWeight, int[] dqRow, int[] dqKind) {
    }

    /** DISTR_RHO contributions in the flat device layout: {@code {drRow, drCol}} per row + coefficient. */
    record DrPack(int[] drRow, int[] drCol, double[] drCoef) {
    }

    /** DISTR_SHUNT_B contributions in the flat device layout: {@code {dbRow, dbCol}} per row + coefficient —
     *  the shunt-susceptance analogue of {@link DrPack} (structurally identical linear distribution rows). */
    record DsbPack(int[] dbRow, int[] dbCol, double[] dbCoef) {
    }

    /** Reactive-limits (PV↔PQ) pack: one entry per voltage-capable controller bus. */
    record RlPack(int[] rlVrow, double[] rlMinQ, double[] rlMaxQ, double[] rlLoadQ, double[] rlVtarget) {
    }

    /** Distributed-slack pack: per-participating-bus factor/headroom + the slack bus's own share.
     *  {@code dsMode} 0 = static factors ({@code dsFactor}); 1 = REMAINING_MARGIN (factors are the
     *  direction-dependent per-bus margins {@code dsMarginUp}/{@code dsMarginDown} + the slack's own
     *  {@code dsSlackMarginUp}/{@code Down}, normalized in-kernel per scenario and pass). */
    record DsPack(int[] dsPhiRow, double[] dsFactor, int dsSlackPhiRow, double dsSlackTargetP,
                  double[] dsBaseTargetP, double[] dsMaxDelta, double[] dsMinDelta,
                  int dsMode, double[] dsMarginUp, double[] dsMarginDown,
                  double dsSlackMarginUp, double dsSlackMarginDown) {
    }

    static DqPack distrQPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                             ExtractionColumns columns, GpuAcFamilyPacker.Packs family) {
        List<int[]> dq = new ArrayList<>();      // {elem, side, dqRow, kind} per DISTR_Q contribution
        List<Double> dqW = new ArrayList<>();    // weight per contribution
        buildDistrQContributions(network, es, columns.vCol(), family.cbElemOf(), family.shElemOf(),
                family.obElemOf(), columns.dummyQCol(), dq, dqW);
        int[] dqElem = new int[dq.size()];
        int[] dqSide = new int[dq.size()];
        int[] dqRow = new int[dq.size()];
        // dqKind: 0 = closed-branch q1/q2, 1 = shunt q, 2 = open-branch q,
        // 3 = zero-impedance DUMMY_Q (linear, elem holds the var column)
        int[] dqKind = new int[dq.size()];
        double[] dqWeight = new double[dq.size()];
        for (int i = 0; i < dq.size(); i++) {
            dqElem[i] = dq.get(i)[0];
            dqSide[i] = dq.get(i)[1];
            dqRow[i] = dq.get(i)[2];
            dqKind[i] = dq.get(i)[3];
            dqWeight[i] = dqW.get(i);
        }
        return new DqPack(dqElem, dqSide, dqWeight, dqRow, dqKind);
    }

    static DrPack distrRhoPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                               ExtractionColumns columns) {
        List<int[]> dr = new ArrayList<>();      // {drRow, drCol} per DISTR_RHO contribution
        List<Double> drC = new ArrayList<>();    // coefficient per contribution
        buildDistrRhoContributions(network, es, columns.branchRhoCol(), dr, drC);
        int[] drRow = new int[dr.size()];
        int[] drCol = new int[dr.size()];
        double[] drCoef = new double[dr.size()];
        for (int i = 0; i < dr.size(); i++) {
            drRow[i] = dr.get(i)[0];
            drCol[i] = dr.get(i)[1];
            drCoef[i] = drC.get(i);
        }
        return new DrPack(drRow, drCol, drCoef);
    }

    static DsbPack distrShuntBPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                   ExtractionColumns columns) {
        List<int[]> db = new ArrayList<>();      // {dbRow, dbCol} per DISTR_SHUNT_B contribution
        List<Double> dbC = new ArrayList<>();    // coefficient per contribution
        buildDistrShuntBContributions(network, es, columns.shuntBCol(), db, dbC);
        int[] dbRow = new int[db.size()];
        int[] dbCol = new int[db.size()];
        double[] dbCoef = new double[db.size()];
        for (int i = 0; i < db.size(); i++) {
            dbRow[i] = db.get(i)[0];
            dbCol[i] = db.get(i)[1];
            dbCoef[i] = dbC.get(i);
        }
        return new DsbPack(dbRow, dbCol, dbCoef);
    }

    static RlPack reactiveLimitsPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                     ExtractionColumns columns, boolean reactiveLimits) {
        int[] vCol = columns.vCol();
        // Reactive-limits (PV↔PQ) pack: every LOCAL voltage-capable controller bus. Two kinds are packed so the
        // device outer loop can move a bus in EITHER direction (matching OLF, which re-derives the PV/PQ set for
        // each contingency from scratch):
        //  - base PV (BUS_TARGET_V active, BUS_TARGET_Q inactive): its branch q scatters to its own BUS_V row;
        //    the outer loop computes its generator Q and flips the row PV→PQ on a limit violation.
        //  - base PQ (BUS_TARGET_Q active, BUS_TARGET_V inactive) but voltage-CAPABLE (a generator voltage
        //    control that OLF disabled during the base loadflow because it hit a Q limit): MUST be packed too, so
        //    a contingency that RELIEVES its reactive loading can switch it BACK to PV. Omitting these froze such
        //    a bus at its base-case limit for every scenario — a real ~1e-3 error vs OLF when a contingency eased
        //    it (e.g. losing a nearby transformer). The device seeds the pinned limit type from the base state
        //    (v below target ⇒ maxQ-pinned, above ⇒ minQ-pinned) so the switch-back fires.
        // No remote control here (the batch path rejects DISTR_Q), so every voltage-controlled bus is local.
        List<Integer> rlVrow = new ArrayList<>();
        List<Double> rlMinQ = new ArrayList<>();
        List<Double> rlMaxQ = new ArrayList<>();
        List<Double> rlLoadQ = new ArrayList<>();
        List<Double> rlVtarget = new ArrayList<>();
        if (reactiveLimits) {                                   // empty pack when the RL outer loop is disabled
            for (LfBus bus : network.getBuses()) {
                int bn = bus.getNum();
                boolean vActive = GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_V) >= 0;
                boolean qActive = GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_Q) >= 0;
                boolean basePv = vActive && !qActive;          // voltage-controlling in the base case
                boolean basePqCapable = qActive && !vActive    // switched to PQ in the base case but still capable
                        && !bus.isGeneratorVoltageControlEnabled();
                if (vCol[bn] >= 0 && bus.getGeneratorVoltageControl().isPresent() && (basePv || basePqCapable)) {
                    rlVrow.add(vCol[bn]);
                    rlMinQ.add(bus.getMinQ());
                    rlMaxQ.add(bus.getMaxQ());
                    rlLoadQ.add(bus.getLoadTargetQ());
                    rlVtarget.add(bus.getGeneratorVoltageControl().orElseThrow().getTargetValue());
                }
            }
        }
        return new RlPack(GpuAcArrays.toIntArray(rlVrow), GpuAcArrays.toDoubleArray(rlMinQ),
                GpuAcArrays.toDoubleArray(rlMaxQ), GpuAcArrays.toDoubleArray(rlLoadQ),
                GpuAcArrays.toDoubleArray(rlVtarget));
    }

    static DsPack distributedSlackPack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                       ExtractionColumns columns, boolean distributedSlack, GpuDsParams dsParams) {
        int[] phiCol = columns.phiCol();
        // Distributed-slack pack: the single slack/reference bus (BUS_TARGET_PHI active, BUS_TARGET_P
        // inactive — its P is free) provides the mismatch; every OTHER bus with participating generators
        // takes a share of it into its BUS_TARGET_P target. Default balance type PROPORTIONAL_TO_GENERATION_
        // P_MAX: per-bus factor = Σ (maxP / droop) over its participating generators, normalized to sum 1.
        //
        // PARTICIPATING SLACK: if the slack/reference bus ITSELF has participating generators, OLF gives them
        // a share too (it resets the reference generator and redistributes the full mismatch over ALL
        // participants incl. the slack). The GPU has no BUS_TARGET_P at the slack to retarget, but it does not
        // need one: by including the slack's factor in the NORMALIZATION DENOMINATOR (without emitting a row),
        // the non-slack shares shrink to f_i/Σ_all and the slack bus naturally retains the remaining
        // f_slack/Σ_all of the mismatch — exactly OLF's allocation.
        //
        // SATURATION: per-bus headroom/footroom (Σ over participating gens of maxTargetP−targetP / minTargetP−
        // targetP, with OLF's no-sign-change rule, useActiveLimits assumed true = the default) bound how much a
        // bus may absorb. The batch kernel clamps each bus's CUMULATIVE delta to [footroom, headroom]; an
        // over-allocation is clamped and its residual reappears at the slack, redistributed next outer pass to
        // unsaturated buses ∝ their fixed factors — converging to OLF's water-filling fixed point.
        int dsSlackPhiRow = -1;
        double dsSlackTargetP = 0;
        double dsSlackFactor = 0;                                       // slack's own participation (denominator only)
        double dsSlackMarginUp = 0;                                     // REMAINING_MARGIN: the slack's own margins
        double dsSlackMarginDown = 0;
        List<Integer> dsPhiRow = new ArrayList<>();
        List<Double> dsFactorRaw = new ArrayList<>();
        List<Double> dsBaseTargetP = new ArrayList<>();                 // each bus's initial BUS_TARGET_P
        List<Double> dsMaxDelta = new ArrayList<>();                    // headroom ≥ 0
        List<Double> dsMinDelta = new ArrayList<>();                    // footroom ≤ 0
        List<Double> dsMarginUp = new ArrayList<>();                    // REMAINING_MARGIN per-bus margins
        List<Double> dsMarginDown = new ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            if (!distributedSlack) {                                    // empty pack when the DS outer loop is disabled
                break;
            }
            int bn = bus.getNum();
            if (GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_PHI) >= 0) {       // the slack/reference bus
                dsSlackPhiRow = phiCol[bn];
                dsSlackTargetP = bus.getTargetP();
                dsSlackFactor = dsParams.busFactor(bus);
                if (dsParams.isRemainingMargin()) {
                    double[] m = dsParams.busMargins(bus);
                    dsSlackMarginUp = m[0];
                    dsSlackMarginDown = m[1];
                }
                continue;
            }
            if (GpuAcRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_P) < 0) {
                continue;                                               // no active P balance to retarget
            }
            double factor = dsParams.busFactor(bus);
            double headroom = 0;
            double footroom = 0;
            for (LfGenerator g : bus.getGenerators()) {
                if (dsParams.factor(g) != 0) {
                    double minTp = g.getMinTargetP();
                    double maxTp = g.getMaxTargetP();
                    double tp = g.getTargetP();
                    if (tp < 0) {                                       // OLF: distribution never flips a gen's sign
                        maxTp = Math.min(maxTp, 0);
                    } else {
                        minTp = Math.max(minTp, 0);
                    }
                    headroom += maxTp - tp;
                    footroom += minTp - tp;
                }
            }
            if (!dsParams.useActiveLimits() || dsParams.isLoadBased()) {
                headroom = Double.POSITIVE_INFINITY;                    // OLF: +-Double.MAX_VALUE bounds (or loads,
                footroom = Double.NEGATIVE_INFINITY;                    // which OLF rescales unbounded) — the
            }                                                           // water-fill clamp never saturates
            if (factor > 0) {
                dsPhiRow.add(phiCol[bn]);
                dsFactorRaw.add(factor);
                dsBaseTargetP.add(bus.getTargetP());
                dsMaxDelta.add(headroom);
                dsMinDelta.add(footroom);
                if (dsParams.isRemainingMargin()) {
                    double[] m = dsParams.busMargins(bus);
                    dsMarginUp.add(m[0]);
                    dsMarginDown.add(m[1]);
                }
            }
        }
        double dsSum = dsFactorRaw.stream().mapToDouble(Double::doubleValue).sum() + dsSlackFactor;
        double[] dsFactor = new double[dsPhiRow.size()];
        for (int i = 0; i < dsFactor.length; i++) {
            dsFactor[i] = dsFactorRaw.get(i) / dsSum;
        }
        // A pack with participants but no slack row is unusable; one with NEITHER participants nor a
        // participating slack means DS never fires. But a PARTICIPATING SLACK ALONE is kept (nds == 0,
        // dsSlackPhiRow >= 0): OLF then distributes the whole mismatch into the slack generator's own
        // target — no flow change, but the outer loop reports UNSTABLE and re-solves, and the device
        // must take that same extra polish run (slackOnlyDsCheck) to match the CPU trajectory.
        if (dsSlackPhiRow < 0 || dsPhiRow.isEmpty() && dsSlackFactor <= 0) {
            dsPhiRow.clear();
            dsFactor = new double[0];
            dsBaseTargetP.clear();
            dsMaxDelta.clear();
            dsMinDelta.clear();
            dsMarginUp.clear();
            dsMarginDown.clear();
            dsSlackPhiRow = -1;
        }
        return new DsPack(GpuAcArrays.toIntArray(dsPhiRow), dsFactor, dsSlackPhiRow, dsSlackTargetP,
                GpuAcArrays.toDoubleArray(dsBaseTargetP), GpuAcArrays.toDoubleArray(dsMaxDelta),
                GpuAcArrays.toDoubleArray(dsMinDelta),
                dsParams.isRemainingMargin() ? 1 : 0,
                GpuAcArrays.toDoubleArray(dsMarginUp), GpuAcArrays.toDoubleArray(dsMarginDown),
                dsSlackMarginUp, dsSlackMarginDown);
    }

    /**
     * Build the weighted closed-branch reactive contributions of every active DISTR_Q row (remote
     * generator voltage control). For a control group, the row at controller {@code Gk}'s BUS_V row is
     * {@code DISTR_Q_k = qPct_k·Σ_j q_Gj − q_Gk}, so each controller {@code Gj}'s reactive injection
     * (the q1/q2 flow of each branch incident to {@code Gj}) contributes with weight
     * {@code qPct_k − (j==k ? 1 : 0)}. A controller's reactive injection is its closed-branch q1/q2 (kind 0),
     * a fixed shunt's q (kind 1), a half-open incident branch's connected-side q (kind 2), or a zero-impedance
     * spanning-tree branch's linear DUMMY_Q variable term (kind 3) — mirroring the terms {@code createReactiveTerms}
     * sums. The pilot bus's own reactive balance is handled by qRow (routed to the free controller's BUS_V row);
     * the controllers' own per-bus Q scatter is suppressed there too.
     */
    private static void buildDistrQContributions(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                 int[] vCol, int[] cbElemOf, int[] shElemOf, int[] obElemOf,
                                                 int[] dummyQCol, List<int[]> out, List<Double> weights) {
        for (LfBus pilot : network.getBuses()) {
            GeneratorVoltageControl gvc = pilot.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> pilot.isGeneratorVoltageControlled() && !c.isLocalControl())
                    .orElse(null);
            if (gvc == null) {
                continue;
            }
            List<LfBus> controllers = gvc.getMergedControllerElements();
            for (LfBus gk : controllers) {
                if (GpuAcRowIndexing.row(es, gk.getNum(), AcEquationType.DISTR_Q) < 0) {
                    continue;                                // the free controller carries the pilot's Q, no DISTR_Q
                }
                int dqRow = vCol[gk.getNum()];
                double qpk = gk.getRemoteControlReactivePercent();
                for (LfBus gj : controllers) {
                    double weight = qpk - (gj == gk ? 1.0 : 0.0);
                    if (gj.getControllerShunt().isPresent()) {
                        throw new PowsyblException("GPU AC solver: DISTR_Q controller " + gj.getId()
                                + " has a voltage-controlling shunt (SHUNT_B) reactive contribution (not supported)");
                    }
                    if (gj.getShunt().isPresent()) {         // a fixed shunt's reactive joins the q sum
                        out.add(new int[] {shElemOf[gj.getNum()], 0, dqRow, 1});
                        weights.add(weight);
                    }
                    for (LfBranch brj : gj.getBranches()) {
                        if (brj.isDisabled()) {
                            continue;
                        }
                        int side = brj.getBus1() == gj ? 0 : 1;
                        int e = cbElemOf[brj.getNum()];
                        if (e >= 0) {                        // closed branch: weighted q1/q2 (kind 0)
                            out.add(new int[] {e, side, dqRow, 0});
                            weights.add(weight);
                            continue;
                        }
                        int oe = obElemOf[brj.getNum()];
                        if (oe >= 0) {                       // half-open branch incident on the controller:
                            // its connected-side reactive flow (shunt + series) joins the q sum (kind 2)
                            out.add(new int[] {oe, side, dqRow, 2});
                            weights.add(weight);
                            continue;
                        }
                        if (brj.isZeroImpedance(LoadFlowModel.AC)) {
                            // a zero-impedance coupler: only a spanning-tree edge carries reactive flow, as the
                            // linear DUMMY_Q variable (+1 if the controller is bus1, −1 if bus2) — exactly the
                            // term createReactiveTerms adds. Non-tree edges contribute nothing (matches OLF).
                            if (!brj.isSpanningTreeEdge(LoadFlowModel.AC)) {
                                continue;
                            }
                            int dqCol = dummyQCol[brj.getNum()];
                            if (dqCol < 0) {
                                throw new PowsyblException("GPU AC solver: DISTR_Q controller " + gj.getId()
                                        + " has a zero-impedance branch " + brj.getId() + " without a DUMMY_Q variable");
                            }
                            out.add(new int[] {dqCol, side, dqRow, 3});  // elem slot holds the DUMMY_Q var column
                            weights.add(weight * (side == 0 ? 1.0 : -1.0));
                            continue;
                        }
                        throw new PowsyblException("GPU AC solver: DISTR_Q controller " + gj.getId()
                                + " has an unsupported branch " + brj.getId() + " reactive contribution");
                    }
                }
            }
        }
    }

    /**
     * Build the LINEAR DISTR_RHO contributions (ratio distribution when several transformers control one
     * bus). A controller branch {@code Gk} with an active DISTR_RHO equation gets, on its BRANCH_RHO1 row,
     * {@code DISTR_RHO_k = (1/count)·Σ_j r1_j − r1_k}, i.e. each controller {@code Gj}'s ratio column
     * contributes with coefficient {@code 1/count − (j==k ? 1 : 0)}, where {@code count} is the number of
     * enabled controllers. The free controller (no DISTR_RHO) carries the pilot's Q balance via qRow.
     */
    private static void buildDistrRhoContributions(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                   int[] branchRhoCol, List<int[]> out, List<Double> coefs) {
        for (LfBus pilot : network.getBuses()) {
            TransformerVoltageControl tvc = pilot.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> pilot.isTransformerVoltageControlled())
                    .orElse(null);
            if (tvc == null) {
                continue;
            }
            List<LfBranch> controllers = tvc.getMergedControllerElements();
            long count = controllers.stream().filter(c -> !c.isDisabled()).count();
            for (LfBranch gk : controllers) {
                if (GpuAcRowIndexing.row(es, gk.getNum(), AcEquationType.DISTR_RHO) < 0) {
                    continue;                                // the free controller carries the pilot's Q, no DISTR_RHO
                }
                int drRow = branchRhoCol[gk.getNum()];
                for (LfBranch gj : controllers) {
                    if (gj.isDisabled()) {
                        continue;
                    }
                    double coef = 1.0 / count - (gj == gk ? 1.0 : 0.0);
                    out.add(new int[] {drRow, branchRhoCol[gj.getNum()]});
                    coefs.add(coef);
                }
            }
        }
    }

    /**
     * DISTR_SHUNT_B linear-distribution rows — the shunt-susceptance analogue of
     * {@link #buildDistrRhoContributions}. For each shunt-voltage-controlled pilot bus, every controller shunt
     * with an ACTIVE DISTR_SHUNT_B equation gets a row {@code b_k = (1/count)·Σ_j b_j} rewritten as
     * {@code 0 = (1/count − 1)·b_k + Σ_{j≠k} b_j/count} over the {@code count} enabled controllers. The free
     * controller (no DISTR_SHUNT_B) carries the pilot's Q balance via qRow.
     */
    private static void buildDistrShuntBContributions(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                                      int[] shuntBCol, List<int[]> out, List<Double> coefs) {
        for (LfBus pilot : network.getBuses()) {
            ShuntVoltageControl svc = pilot.getShuntVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> pilot.isShuntVoltageControlled())
                    .orElse(null);
            if (svc == null) {
                continue;
            }
            List<LfShunt> controllers = svc.getMergedControllerElements();
            long count = controllers.stream().filter(c -> !c.isDisabled()).count();
            for (LfShunt gk : controllers) {
                if (GpuAcRowIndexing.row(es, gk.getNum(), AcEquationType.DISTR_SHUNT_B) < 0) {
                    continue;                                // the free controller carries the pilot's Q, no DISTR_SHUNT_B
                }
                int dbRow = shuntBCol[gk.getNum()];
                for (LfShunt gj : controllers) {
                    if (gj.isDisabled()) {
                        continue;
                    }
                    double coef = 1.0 / count - (gj == gk ? 1.0 : 0.0);
                    out.add(new int[] {dbRow, shuntBCol[gj.getNum()]});
                    coefs.add(coef);
                }
            }
        }
    }

    /**
     * The transformers that participate in a ratio-distribution (DISTR_RHO) group. Outaging one of these
     * REFORMS the {@code 1/count} coefficients (OLF redistributes the ratio across the remaining controllers),
     * which the fixed batch pattern cannot represent — so the batched N-1 path rejects such a contingency to
     * the CPU. Returns the controllers of every transformer-voltage-controlled pilot whose group has more than
     * one enabled member (a single local controller is item 4, not DISTR_RHO, and is mask-safe).
     */
    public static Set<LfBranch> distrRhoControllerBranches(LfNetwork network,
                                                           EquationSystem<AcVariableType, AcEquationType> es) {
        Set<LfBranch> controllers = new java.util.HashSet<>();
        for (LfBus pilot : network.getBuses()) {
            TransformerVoltageControl tvc = pilot.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> pilot.isTransformerVoltageControlled())
                    .orElse(null);
            if (tvc == null) {
                continue;
            }
            List<LfBranch> group = tvc.getMergedControllerElements();
            boolean distrRho = group.stream().anyMatch(c -> GpuAcRowIndexing.row(es, c.getNum(), AcEquationType.DISTR_RHO) >= 0);
            if (distrRho) {
                controllers.addAll(group);
            }
        }
        return controllers;
    }
}
