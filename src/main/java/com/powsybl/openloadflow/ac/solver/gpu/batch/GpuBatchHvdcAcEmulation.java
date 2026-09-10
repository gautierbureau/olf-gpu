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

    /**
     * Re-derive the converter-bus target overrides and the hv element bits from a set of statuses,
     * onto {@code ovIdx}/{@code ovVal} seeded with the PRE-FREEZE list. This is the orchestrator's
     * per-round rebuild, extracted so the in-kernel loops can leave the HOST's mirrors consistent
     * with what they did on the device — otherwise the orchestrators that run next re-upload a
     * pre-freeze override list and re-apply a freeze the kernel already lifted.
     */
    static void applyStatusOverrides(int chunkSize, int realCount, int n, List<Ctl> ctls,
                                     int[][] status, byte[] ctgOut, boolean[][] disabledMask,
                                     double[] baseTargets, List<Integer> ovIdx, List<Double> ovVal,
                                     double[] appliedD1, double[] appliedD2) {
        java.util.Map<Integer, Integer> posOf = indexOf(ovIdx);
        int nh = ctls.size();
        for (int s = 0; s < realCount; s++) {
            for (int h = 0; h < nh; h++) {
                Ctl c = ctls.get(h);
                if (ctgOut[s * nh + h] != 0) {
                    continue;                                // the contingency owns that bit
                }
                if (status[s][h] == LINEAR) {
                    disabledMask[s][c.hvElemIndex()] = false;
                    if (c.baseStatus() != LINEAR) {          // strip the base injections from the targets
                        mergeOverride(ovIdx, ovVal, posOf, s * n + c.bus1PRow(),
                                baseTargets[c.bus1PRow()], c.baseD1(), 0);
                        mergeOverride(ovIdx, ovVal, posOf, s * n + c.bus2PRow(),
                                baseTargets[c.bus2PRow()], c.baseD2(), 0);
                    }
                } else {
                    disabledMask[s][c.hvElemIndex()] = true;
                    // A STILL-FROZEN entry (its scenario retired before the unfreeze phase reached
                    // it) must keep the delta the pre-solve freeze applied — recomputing it here is
                    // impossible (it was taken at the warm-start flows), but it is exactly what the
                    // applied-delta arrays hold. Seeding the list PRE-freeze and skipping this entry
                    // silently dropped its override.
                    double[] d = status[s][h] == SAT_1_TO_2 ? saturationDeltas(c, true)
                            : status[s][h] == SAT_2_TO_1 ? saturationDeltas(c, false)
                            : new double[] {appliedD1[s * nh + h] + c.baseD1(),
                                            appliedD2[s * nh + h] + c.baseD2()};
                    mergeOverride(ovIdx, ovVal, posOf, s * n + c.bus1PRow(),
                            baseTargets[c.bus1PRow()], c.baseD1(), d[0]);
                    mergeOverride(ovIdx, ovVal, posOf, s * n + c.bus2PRow(),
                            baseTargets[c.bus2PRow()], c.baseD2(), d[1]);
                }
            }
        }
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

}
