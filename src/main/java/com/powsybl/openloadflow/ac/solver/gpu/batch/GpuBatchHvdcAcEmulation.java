/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.AbstractHvdcAcEmulationFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.HvdcAcEmulationSide1ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.HvdcAcEmulationSide2ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfNetwork;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of OLF's HVDC AC-emulation outer loops with HVDCs actually present:
 *
 * <ul>
 *   <li>{@code FreezingHvdcACEmulationOuterloop}: initialize() FREEZES every AC-emulation HVDC at the
 *       warm-start flows BEFORE the first solve (VSC injections pinned, coupling off), and its first
 *       check UNFREEZES them all → one linear re-solve;</li>
 *   <li>{@code AcHvdcAcEmulationLimitsOuterLoop}: linear ↔ saturation transitions — a linear HVDC
 *       whose |P| overpasses pMax saturates at pMax (with converter/line losses), a saturated one
 *       back inside its limits returns to linear, and an over-reversed one switches saturation side.</li>
 * </ul>
 *
 * <p>Device mechanics per (scenario, hvdc): a NON-LINEAR status sets the hv element's per-scenario
 * DISABLED bit (fillHvdcBatch drops the coupling terms — same channel the JAVA hybrid uses) and the
 * VSC injections land as CONVERTER-BUS target overrides (the stations' targetP is 0 in the base
 * linear extraction, so the bus-target delta IS the station targetP, generator convention). Flow
 * evaluations call OLF's OWN p1/p2 formulas directly on the scenario's state array (see
 * {@link Ctl#p1At}) — the equation terms read a shared {@code StateVector} whose every load
 * recomputes the whole network's flows and derivatives, which is unaffordable per scenario.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchHvdcAcEmulation {

    private GpuBatchHvdcAcEmulation() {
    }

    static final int LINEAR = 0;
    static final int SAT_1_TO_2 = 1;
    static final int SAT_2_TO_1 = 2;
    static final int FROZEN = 3;

    /** One packed AC-emulation HVDC: the hv element index (per-scenario disabled bit), the two
     *  converter buses' P-target rows, OLF's own linear p1/p2 terms, the emulation limits, and the
     *  BASE frame — the base status plus the base VSC station targetP contributions ALREADY inside
     *  baseTargets (0 for a linear base; a saturated/frozen base carries its injections there, so
     *  every status override composes as {@code baseTargets - baseD + d(status)}). */
    record Ctl(int hvElemIndex, int bus1PRow, int bus2PRow, double p0, double k,
               double pMax12, double pMax21, double loss1, double loss2, double r,
               int baseStatus, double baseD1, double baseD2) {

        /** OLF's own {@code HvdcAcEmulationSide1ActiveFlowEquationTerm.p1} at a scenario's state.
         *  Called directly on the state array rather than through the equation term: loading a
         *  state into the shared {@code StateVector} notifies {@code AcNetworkVector}, which
         *  recomputes flows AND derivatives for every branch in the network — a whole-network pass
         *  per scenario per check round, which measured 6 s of a 10 s orchestrator on a 9k-branch
         *  case. The row IS the bus's BUS_PHI row (one device row per variable). */
        double p1At(double[] x) {
            return HvdcAcEmulationSide1ActiveFlowEquationTerm.p1(p0, k, loss1, loss2, r, x[bus1PRow], x[bus2PRow]);
        }

        /** OLF's own {@code HvdcAcEmulationSide2ActiveFlowEquationTerm.p2}; see {@link #p1At}. */
        double p2At(double[] x) {
            return HvdcAcEmulationSide2ActiveFlowEquationTerm.p2(p0, k, loss1, loss2, r, x[bus1PRow], x[bus2PRow]);
        }
    }

    static List<Ctl> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                      int nElements) {
        Map<Integer, Integer> phiRows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_PHI) {
                phiRows.put(v.getElementNum(), v.getRow());
            }
        }
        List<LfHvdc> packed = network.getHvdcs().stream()
                .filter(h -> h.getBus1() != null && h.getBus2() != null && h.isAcEmulation())
                .toList();                                   // the packer's exact enumeration order
        int hvOffset = nElements - packed.size();
        List<Ctl> out = new ArrayList<>();
        for (int h = 0; h < packed.size(); h++) {
            LfHvdc hvdc = packed.get(h);
            var ac = hvdc.getAcEmulationControl();
            int baseStatus = switch (ac.getAcEmulationStatus()) {
                case LINEAR_MODE -> LINEAR;
                case SATURATION_MODE_FROM_CS1_TO_CS2 -> SAT_1_TO_2;
                case SATURATION_MODE_FROM_CS2_TO_CS1 -> SAT_2_TO_1;
                case FROZEN -> FROZEN;
            };
            out.add(new Ctl(hvOffset + h,
                    phiRows.get(hvdc.getBus1().getNum()), phiRows.get(hvdc.getBus2().getNum()),
                    ac.getP0(), ac.getDroop() * 180 / Math.PI,
                    ac.getPMaxFromCS1toCS2(), ac.getPMaxFromCS2toCS1(),
                    hvdc.getConverterStation1().getLossFactor() / 100,
                    hvdc.getConverterStation2().getLossFactor() / 100, hvdc.getR(),
                    baseStatus,
                    baseStatus == LINEAR ? 0 : hvdc.getConverterStation1().getTargetP(),
                    baseStatus == LINEAR ? 0 : hvdc.getConverterStation2().getTargetP()));
        }
        return out;
    }

    /** The packed hv element index of every AC-emulation HVDC (the packer's exact enumeration order) —
     *  for per-scenario outage masking of HVDC contingencies. */
    static java.util.Map<LfHvdc, Integer> elementIndices(LfNetwork network, int nElements) {
        List<LfHvdc> packed = network.getHvdcs().stream()
                .filter(h -> h.getBus1() != null && h.getBus2() != null && h.isAcEmulation())
                .toList();
        java.util.Map<LfHvdc, Integer> out = new java.util.LinkedHashMap<>();
        int hvOffset = nElements - packed.size();
        for (int h = 0; h < packed.size(); h++) {
            out.put(packed.get(h), hvOffset + h);
        }
        return out;
    }

    /** True when every AC-emulation HVDC starts LINEAR (the device pack's extraction assumption). */
    public static boolean allLinearAtBase(LfNetwork network) {
        return network.getHvdcs().stream()
                .filter(LfHvdc::isAcEmulation)
                .allMatch(h -> h.getAcEmulationControl().getAcEmulationStatus()
                        == LfHvdc.AcEmulationControl.AcEmulationStatus.LINEAR_MODE);
    }

    /** OLF setVscTargetP: {@code [bus1TargetDelta, bus2TargetDelta]} for a saturation at pController
     *  flowing from the controller station (generator-convention deltas over the linear base). */
    private static double[] saturationDeltas(Ctl c, boolean from1to2) {
        if (from1to2) {
            double pAbs = AbstractHvdcAcEmulationFlowEquationTerm.getAbsActivePowerWithLosses(c.pMax12(), c.loss1(), c.loss2(), c.r());
            return new double[] {-c.pMax12(), pAbs};
        }
        double pAbs = AbstractHvdcAcEmulationFlowEquationTerm.getAbsActivePowerWithLosses(c.pMax21(), c.loss2(), c.loss1(), c.r());
        return new double[] {pAbs, -c.pMax21()};
    }

    /** OLF switchToFrozenState at the given (warm-start) evaluation point: pMax-clamped when the
     *  linear flows overpass the limits, else pinned at the current flows. */
    private static double[] frozenDeltas(Ctl c, double p1, double p2) {
        if (p1 > c.pMax12()) {
            return saturationDeltas(c, true);
        } else if (p2 > c.pMax21()) {
            return saturationDeltas(c, false);
        }
        return new double[] {-p1, -p2};
    }

    /** OLF checkAcEmulationMode: the next status (or the current one if stable) at the given flows. */
    private static int nextStatus(Ctl c, int status, double p1, double p2) {
        switch (status) {
            case LINEAR -> {
                if (p1 >= 0) {
                    if (p1 > c.pMax12()) {
                        return SAT_1_TO_2;
                    }
                } else if (p2 > c.pMax21()) {
                    return SAT_2_TO_1;
                }
                return LINEAR;
            }
            case SAT_1_TO_2, SAT_2_TO_1 -> {
                if (p1 > 0 && p1 < c.pMax12() || p2 > 0 && p2 < c.pMax21()) {
                    return LINEAR;
                } else if (p1 < 0 && status == SAT_1_TO_2) {
                    return SAT_2_TO_1;
                } else if (p2 < 0 && status == SAT_2_TO_1) {
                    return SAT_1_TO_2;
                }
                return status;
            }
            default -> {
                return status;                               // FROZEN transitions are the freezing loop's
            }
        }
    }

    /** Pre-solve freeze (OLF FreezingHvdcACEmulation.initialize): every live hvdc frozen at its
     *  scenario's WARM-START flows — hv disabled bits set, converter-bus targets pinned. Returns the
     *  per-(scenario, hvdc) status matrix seeded FROZEN (or LINEAR where the freeze cannot apply). */
    static int[][] applyFreeze(int chunkSize, int realCount, int n, List<Ctl> ctls,
                               double[] baseState, double[] x0PerScenario, boolean[][] disabledMask,
                               double[] baseTargets, List<Integer> ovIndexList, List<Double> ovValueList) {
        return applyFreeze(chunkSize, realCount, n, ctls, baseState, x0PerScenario, disabledMask,
                baseTargets, ovIndexList, ovValueList, null, null);
    }

    /** As above, additionally recording the converter-row delta each freeze APPLIED (relative to the
     *  base-status delta), which is what the in-kernel loops need to keep applying transitions
     *  incrementally instead of rebuilding the override list. */
    static int[][] applyFreeze(int chunkSize, int realCount, int n, List<Ctl> ctls,
                               double[] baseState, double[] x0PerScenario, boolean[][] disabledMask,
                               double[] baseTargets, List<Integer> ovIndexList, List<Double> ovValueList,
                               double[] appliedD1, double[] appliedD2) {
        int[][] status = new int[chunkSize][ctls.size()];
        java.util.Map<Integer, Integer> posOf = indexOf(ovIndexList);
        double[] warm = new double[n];
        for (int s = 0; s < realCount; s++) {
            if (x0PerScenario != null) {
                System.arraycopy(x0PerScenario, s * n, warm, 0, n);
            } else {
                System.arraycopy(baseState, 0, warm, 0, n);
            }
            for (int h = 0; h < ctls.size(); h++) {
                Ctl c = ctls.get(h);
                if (disabledMask[s][c.hvElemIndex()]) {
                    continue;                                // outaged hvdc — stays as the contingency left it
                }
                double[] d = frozenDeltas(c, c.p1At(warm), c.p2At(warm));
                if (appliedD1 != null) {
                    appliedD1[s * ctls.size() + h] = d[0] - c.baseD1();
                    appliedD2[s * ctls.size() + h] = d[1] - c.baseD2();
                }
                status[s][h] = FROZEN;
                disabledMask[s][c.hvElemIndex()] = true;
                // base frame: baseTargets (or the row's existing override — e.g. a lost load on the
                // converter bus) already carry the base injections
                mergeOverride(ovIndexList, ovValueList, posOf, s * n + c.bus1PRow(),
                        baseTargets[c.bus1PRow()], c.baseD1(), d[0]);
                mergeOverride(ovIndexList, ovValueList, posOf, s * n + c.bus2PRow(),
                        baseTargets[c.bus2PRow()], c.baseD2(), d[1]);
            }
        }
        return status;
    }

    /** Compose one converter-bus status override ONTO any override already present at the row
     *  (e.g. a lost-load target on the converter bus — a row must NEVER carry two overrides, the
     *  parallel scatter has no ordering): value = (existing override | baseTargets) - baseD + d. */
    private static void mergeOverride(List<Integer> ovIdx, List<Double> ovVal,
                                      java.util.Map<Integer, Integer> posOf, int flat,
                                      double baseTarget, double baseD, double d) {
        Integer at = posOf.get(flat);
        double from = at != null ? ovVal.get(at) : baseTarget;
        if (at != null) {
            ovVal.set(at, from - baseD + d);
        } else {
            posOf.put(flat, ovIdx.size());
            ovIdx.add(flat);
            ovVal.add(from - baseD + d);
        }
    }

    /** flat-row → position index over an override list (the lookup mergeOverride needs; a scan
     *  would be quadratic in scenarios × hvdcs over a list that grows with both). */
    private static java.util.Map<Integer, Integer> indexOf(List<Integer> ovIdx) {
        java.util.Map<Integer, Integer> posOf = new java.util.HashMap<>(2 * ovIdx.size() + 16);
        for (int k = 0; k < ovIdx.size(); k++) {
            posOf.put(ovIdx.get(k), k);
        }
        return posOf;
    }

    /** Seed the NON-LINEAR-at-base statuses (a saturated/frozen base — the extraction gate used to
     *  route such runs whole-run CPU): entries the freeze did not claim keep the BASE status with
     *  the hv bit set and NO target override (baseTargets already carry the base injections), and
     *  PADDING slots get the base bits so they still converge at iteration 0. An entry whose bit is
     *  already set while its status is LINEAR is a contingency outage — left alone. */
    static void seedBaseStatuses(int chunkSize, int realCount, List<Ctl> ctls,
                                 int[][] status, boolean[][] disabledMask) {
        for (int h = 0; h < ctls.size(); h++) {
            Ctl c = ctls.get(h);
            if (c.baseStatus() == LINEAR) {
                continue;
            }
            for (int s = 0; s < chunkSize; s++) {
                if (s >= realCount) {
                    disabledMask[s][c.hvElemIndex()] = true;
                } else if (status[s][h] == LINEAR && !disabledMask[s][c.hvElemIndex()]) {
                    status[s][h] = c.baseStatus();
                    disabledMask[s][c.hvElemIndex()] = true;
                }
            }
        }
    }

    /** Post-solve driver: the unfreeze pass (all FROZEN → LINEAR, one re-solve) then the AC-emulation
     *  limit rounds (linear ↔ saturation) until stable. Restores every hv disabled bit at exit. */
    @SuppressWarnings({"java:S107", "java:S3776"})
    static double[][] orchestrate(long handle, int chunkSize, int realCount, int n, int nrl, int[] rlState,
            boolean[] rlSolved,
            double[][] chunkResults, List<Ctl> ctls, int[][] status, boolean limitsActive,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            List<Integer> baseOvIndexList, List<Double> baseOvValueList,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            int[] dsPhiRow, double[] dsFactor,
            int[] r1OvSlotElem, double[] r1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        int nh = ctls.size();
        int maxRounds = config.maxOuterIterations() > 0 ? config.maxOuterIterations() : 20;
        boolean unfreezePending = false;
        for (int s = 0; s < realCount; s++) {
            for (int h = 0; h < nh; h++) {
                unfreezePending |= status[s][h] == FROZEN;
            }
        }
        // Contingency-outaged hvdcs: disabled at entry while still LINEAR (a freeze would have marked
        // FROZEN) — their bit must survive every status-driven mask rebuild.
        boolean[][] ctgOut = new boolean[realCount][nh];
        for (int s = 0; s < realCount; s++) {
            for (int h = 0; h < nh; h++) {
                ctgOut[s][h] = status[s][h] == LINEAR && disabledMask[s][ctls.get(h).hvElemIndex()];
            }
        }
        // DS target carry (same three coupled pieces as GpuBatchTransformerControl): the MAIN
        // solve's slack distribution must ride into every freeze/saturation re-solve, or the
        // kernel re-derives it from the re-uploaded baseTargets along a trajectory the CPU
        // never takes (drift confined to slack-distributing scenarios).
        int nds = dsPhiRow != null ? dsPhiRow.length : 0;
        double[] dsAnchorCarry = nds > 0 ? new double[chunkSize * nds] : null;
        double[] dsSlackCarry = nds > 0 ? new double[chunkSize] : null;
        double[] dsSeed = nds > 0 ? new double[chunkSize * nds] : null;
        if (nds > 0) {
            GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed);
        }
        try {
            for (int round = 0; round < maxRounds; round++) {
                GpuBatchPhaseStats.bump("HvdcAcEmulation.rounds");
                boolean[] affected = new boolean[chunkSize];
                boolean any = false;
                if (unfreezePending) {                       // OLF's UNFREEZE step: all frozen back to linear
                    for (int s = 0; s < realCount; s++) {
                        for (int h = 0; h < nh; h++) {
                            if (status[s][h] == FROZEN) {
                                status[s][h] = LINEAR;
                                affected[s] = true;
                                any = true;
                            }
                        }
                    }
                    unfreezePending = false;
                } else if (limitsActive) {                   // AcHvdcAcEmulationLimits transitions
                    for (int s = 0; s < realCount; s++) {
                        double[] r = chunkResults[s];
                        if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                            continue;
                        }
                        for (int h = 0; h < nh; h++) {
                            Ctl c = ctls.get(h);
                            if (status[s][h] == LINEAR && disabledMask[s][c.hvElemIndex()]) {
                                continue;                    // outaged by the contingency itself
                            }
                            int next = nextStatus(c, status[s][h], c.p1At(r), c.p2At(r));
                            if (next != status[s][h]) {
                                status[s][h] = next;
                                affected[s] = true;
                                any = true;
                            }
                        }
                    }
                } else {
                    break;
                }
                if (!any) {
                    break;
                }
                // ---- rebuild the per-scenario hv bits + converter-bus targets from the statuses ----
                long tPrep = System.nanoTime();
                List<Integer> ovIdx = new ArrayList<>(baseOvIndexList);
                List<Double> ovVal = new ArrayList<>(baseOvValueList);
                GpuBatchTransformerControl.seedDsTargets(nds, dsPhiRow, dsSeed, chunkResults, realCount, n, ovIdx, ovVal);
                java.util.Map<Integer, Integer> posOf = indexOf(ovIdx);
                double[] x0Full = new double[chunkSize * n];
                for (int s = 0; s < chunkSize; s++) {
                    double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
                    System.arraycopy(src, 0, x0Full, s * n, n);
                }
                for (int s = 0; s < realCount; s++) {
                    for (int h = 0; h < nh; h++) {
                        Ctl c = ctls.get(h);
                        if (ctgOut[s][h]) {
                            continue;                        // outaged by the contingency — the bit stays set
                        }
                        if (status[s][h] == LINEAR) {
                            disabledMask[s][c.hvElemIndex()] = false;
                            if (c.baseStatus() != LINEAR) {  // strip the base injections from the targets
                                mergeOverride(ovIdx, ovVal, posOf, s * n + c.bus1PRow(),
                                        baseTargets[c.bus1PRow()], c.baseD1(), 0);
                                mergeOverride(ovIdx, ovVal, posOf, s * n + c.bus2PRow(),
                                        baseTargets[c.bus2PRow()], c.baseD2(), 0);
                            }
                        } else {
                            disabledMask[s][c.hvElemIndex()] = true;
                            double[] d = status[s][h] == SAT_1_TO_2 ? saturationDeltas(c, true)
                                    : saturationDeltas(c, false);
                            mergeOverride(ovIdx, ovVal, posOf, s * n + c.bus1PRow(),
                                    baseTargets[c.bus1PRow()], c.baseD1(), d[0]);
                            mergeOverride(ovIdx, ovVal, posOf, s * n + c.bus2PRow(),
                                    baseTargets[c.bus2PRow()], c.baseD2(), d[1]);
                        }
                    }
                }
                if (GpuBatchPhaseStats.ENABLED) {
                    int na = 0;
                    for (int s = 0; s < realCount; s++) {
                        if (affected[s]) {
                            na++;
                        }
                    }
                    GpuBatchPhaseStats.add("HvdcAcEmulation.affectedSum", na);
                    GpuBatchPhaseStats.add("HvdcAcEmulation.realCountSum", realCount);
                }
                GpuBatchPhaseStats.addNs("HvdcAcEmulation.prepNs", System.nanoTime() - tPrep);
                GpuBatchPhaseStats.bump("HvdcAcEmulation.solves");
                long tDev = System.nanoTime();
                // Carry the reactive-limits PINNING across the freeze/saturation re-solves and out to the
                // loops that run after this one. The RL loop is stateful on the CPU too — OLF re-enters it
                // with the pinning it already established — and a solve handed the pinning cold has to
                // re-derive it, which measured 4 Newton iterations instead of 1 in the next orchestrator.
                int[] rlStateOut = rlState != null ? new int[chunkSize * nrl] : null;
                double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                        rowModePerScenario, baseRowMode, disabledMask,
                        ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                        ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                        // The distribution rides in on the TARGET SEED (seedDsTargets, above); the anchor
                        // must NOT move with it. OLF's ActivePowerDistribution.run always resets to the
                        // initial targets and re-derives the WHOLE distribution from
                        // mismatch + previousMismatch — within a contingency the anchor never moves (it is
                        // re-anchored only by NetworkState.save, i.e. per contingency, and before an
                        // operator strategy leg). Carrying an anchor here made the device INCREMENT from the
                        // current distribution where OLF re-derives, and clamped water-fills do not compose:
                        // measured on LINE-1938-692 as a pass entered with absorbedNow=0 and rem=-0.0167
                        // instead of the CPU's full ~13.2 re-derivation.
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, null, null, null, null,
                        parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                        config.lsMode(), config.maxDv(), config.maxDphi(),
                        config.lsMode() == 2 ? varType : null, x0Full,
                        shOvSlotElem, shOvGb, rlState, rlStateOut,
                        r1OvSlotElem, r1OvVal,
                        config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0,
                        nziBase, ziOnSlotElem, ziOffSlotElem);
                GpuBatchPhaseStats.addNs("HvdcAcEmulation.deviceNs", System.nanoTime() - tDev);
                if (rlStateOut != null) {
                    System.arraycopy(rlStateOut, 0, rlState, 0, rlStateOut.length);
                }
                if (GpuBatchPhaseStats.ENABLED) {
                    int itMax = 0;
                    for (int s = 0; s < realCount; s++) {
                        itMax = Math.max(itMax, (int) r2[s][n]);
                    }
                    GpuBatchPhaseStats.add("HvdcAcEmulation.iterMax", itMax);
                }
                for (int s = 0; s < realCount; s++) {
                    if (affected[s]) {                       // untouched scenarios KEEP their converged state
                        chunkResults[s] = r2[s];
                        if (rlSolved != null) {
                            rlSolved[s] = true;              // this solve ran the RL phase for the scenario
                        }
                    }
                }
                if (nds > 0) {                               // this solve may have distributed further
                    long tCap = System.nanoTime();
                    GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                            dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed, affected);
                    GpuBatchPhaseStats.addNs("HvdcAcEmulation.captureNs", System.nanoTime() - tCap);
                }
            }
        } finally {
            // the hv disabled bits are chunk-local scratch: restore them to the BASE configuration
            // (set only for a non-LINEAR-at-base hvdc) so the next chunk's slot cleanup (which only
            // tracks contingency-disabled elements) starts from a clean mask
            for (int s = 0; s < realCount; s++) {
                for (int h = 0; h < nh; h++) {
                    Ctl c = ctls.get(h);
                    if (ctgOut[s][h]) {
                        continue;                            // outage bit belongs to the contingency
                    }
                    disabledMask[s][c.hvElemIndex()] = c.baseStatus() != LINEAR;
                }
            }
        }
        return chunkResults;
    }
}
