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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.util.Evaluable;

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
 * evaluations reuse OLF's OWN p1/p2 equation terms via es-state loading — same pattern as
 * {@link GpuBatchIncrementalFlowControl}.
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
     *  converter buses' P-target rows, OLF's own linear p1/p2 terms, and the emulation limits. */
    record Ctl(int hvElemIndex, int bus1PRow, int bus2PRow, Evaluable p1, Evaluable p2,
               double pMax12, double pMax21, double loss1, double loss2, double r) {
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
            out.add(new Ctl(hvOffset + h,
                    phiRows.get(hvdc.getBus1().getNum()), phiRows.get(hvdc.getBus2().getNum()),
                    hvdc.getP1(), hvdc.getP2(),
                    ac.getPMaxFromCS1toCS2(), ac.getPMaxFromCS2toCS1(),
                    hvdc.getConverterStation1().getLossFactor() / 100,
                    hvdc.getConverterStation2().getLossFactor() / 100, hvdc.getR()));
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
                               GpuBatchIncrementalFlowControl.EsContext esCtx,
                               double[] baseState, double[] x0PerScenario, boolean[][] disabledMask,
                               double[] baseTargets, List<Integer> ovIndexList, List<Double> ovValueList) {
        int[][] status = new int[chunkSize][ctls.size()];
        double[] warm = new double[n];
        for (int s = 0; s < realCount; s++) {
            if (x0PerScenario != null) {
                System.arraycopy(x0PerScenario, s * n, warm, 0, n);
            } else {
                System.arraycopy(baseState, 0, warm, 0, n);
            }
            esCtx.stateVector().set(warm.clone());
            for (int h = 0; h < ctls.size(); h++) {
                Ctl c = ctls.get(h);
                if (disabledMask[s][c.hvElemIndex()]) {
                    continue;                                // outaged hvdc — stays as the contingency left it
                }
                double[] d = frozenDeltas(c, c.p1().eval(), c.p2().eval());
                status[s][h] = FROZEN;
                disabledMask[s][c.hvElemIndex()] = true;
                ovIndexList.add(s * n + c.bus1PRow());
                ovValueList.add(baseTargets[c.bus1PRow()] + d[0]);
                ovIndexList.add(s * n + c.bus2PRow());
                ovValueList.add(baseTargets[c.bus2PRow()] + d[1]);
            }
        }
        return status;
    }

    /** Post-solve driver: the unfreeze pass (all FROZEN → LINEAR, one re-solve) then the AC-emulation
     *  limit rounds (linear ↔ saturation) until stable. Restores every hv disabled bit at exit. */
    @SuppressWarnings({"java:S107", "java:S3776"})
    static double[][] orchestrate(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<Ctl> ctls, int[][] status, boolean limitsActive,
            GpuBatchIncrementalFlowControl.EsContext esCtx,
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
                        esCtx.stateVector().set(java.util.Arrays.copyOf(r, n));
                        for (int h = 0; h < nh; h++) {
                            Ctl c = ctls.get(h);
                            if (status[s][h] == LINEAR && disabledMask[s][c.hvElemIndex()]) {
                                continue;                    // outaged by the contingency itself
                            }
                            int next = nextStatus(c, status[s][h], c.p1().eval(), c.p2().eval());
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
                List<Integer> ovIdx = new ArrayList<>(baseOvIndexList);
                List<Double> ovVal = new ArrayList<>(baseOvValueList);
                GpuBatchTransformerControl.seedDsTargets(nds, dsPhiRow, dsSeed, chunkResults, realCount, n, ovIdx, ovVal);
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
                        } else {
                            disabledMask[s][c.hvElemIndex()] = true;
                            double[] d = status[s][h] == SAT_1_TO_2 ? saturationDeltas(c, true)
                                    : saturationDeltas(c, false);
                            ovIdx.add(s * n + c.bus1PRow());
                            ovVal.add(baseTargets[c.bus1PRow()] + d[0]);
                            ovIdx.add(s * n + c.bus2PRow());
                            ovVal.add(baseTargets[c.bus2PRow()] + d[1]);
                        }
                    }
                }
                double[][] r2 = GpuAcNewtonSolver.solveBatchContext(handle, baseState, baseTargets,
                        rowModePerScenario, baseRowMode, disabledMask,
                        ovIdx.stream().mapToInt(Integer::intValue).toArray(),
                        ovVal.stream().mapToDouble(Double::doubleValue).toArray(),
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, dsOvMaxDelta, dsOvMinDelta, dsAnchorCarry, dsSlackCarry, null, null,
                        parameters.getMaxIterations(), tol, reactiveLimitTol, slackDistTol, convEpsPerEq,
                        config.lsMode(), config.maxDv(), config.maxDphi(),
                        config.lsMode() == 2 ? varType : null, x0Full,
                        shOvSlotElem, shOvGb, null, null,
                        r1OvSlotElem, r1OvVal,
                        config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0,
                        nziBase, ziOnSlotElem, ziOffSlotElem);
                for (int s = 0; s < realCount; s++) {
                    if (affected[s]) {                       // untouched scenarios KEEP their converged state
                        chunkResults[s] = r2[s];
                    }
                }
                if (nds > 0) {                               // this solve may have distributed further
                    GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                            dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed);
                }
            }
        } finally {
            // the hv disabled bits are chunk-local scratch: restore them so the next chunk's slot
            // cleanup (which only tracks contingency-disabled elements) starts from a clean mask
            for (int s = 0; s < realCount; s++) {
                for (int h = 0; h < nh; h++) {
                    if (status[s][h] != LINEAR) {
                        disabledMask[s][ctls.get(h).hvElemIndex()] = false;
                    }
                }
            }
        }
        return chunkResults;
    }
}
