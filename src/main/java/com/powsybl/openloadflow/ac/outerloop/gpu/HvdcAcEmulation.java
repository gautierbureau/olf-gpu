/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.AbstractHvdcAcEmulationFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.HvdcAcEmulationSide1ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.equations.HvdcAcEmulationSide2ActiveFlowEquationTerm;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchOverrides;
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
 * DISABLED bit (fillHvdcBatch drops the coupling terms) and the
 * VSC injections land as CONVERTER-BUS target overrides (the stations' targetP is 0 in the base
 * linear extraction, so the bus-target delta IS the station targetP, generator convention). Flow
 * evaluations call OLF's OWN p1/p2 formulas directly on the scenario's state array (see
 * {@link Ctl#p1At}) — the equation terms read a shared {@code StateVector} whose every load
 * recomputes the whole network's flows and derivatives, which is unaffordable per scenario.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class HvdcAcEmulation {

    private HvdcAcEmulation() {
    }

    public static final int LINEAR = 0;
    public static final int SAT_1_TO_2 = 1;
    public static final int SAT_2_TO_1 = 2;
    public static final int FROZEN = 3;

    /** One packed AC-emulation HVDC: the hv element index (per-scenario disabled bit), the two
     *  converter buses' P-target rows, OLF's own linear p1/p2 terms, the emulation limits, and the
     *  BASE frame — the base status plus the base VSC station targetP contributions ALREADY inside
     *  baseTargets (0 for a linear base; a saturated/frozen base carries its injections there, so
     *  every status override composes as {@code baseTargets - baseD + d(status)}). */
    public record Ctl(int hvElemIndex, int bus1PRow, int bus2PRow, int bus1PhiCol, int bus2PhiCol,
               double p0, double k,
               double pMax12, double pMax21, double loss1, double loss2, double r,
               int baseStatus, double baseD1, double baseD2) {

        // TWO DISTINCT ROLES, and they are NOT the same row under the coupler-tree relocation:
        //   bus{1,2}PRow   = where the P TARGET is written (the bus's P equation). For a relocated
        //                    coupler child that is the parent's DUMMY_P row, NOT the bus's own PHI row,
        //                    which hosts the coupler's ZERO_PHI (target 0). See GpuAcData.relocatedPRow
        //                    and tasks #68: conflating the two put the freeze setpoints on all ten
        //                    VSC-station breakers' ZERO_PHI rows.
        //   bus{1,2}PhiCol = the ANGLE COLUMN the emulation formula reads. Always the bus's OWN BUS_PHI
        //                    column, relocation or not — a relocation moves EQUATIONS, never variables.
        // A converter bus whose BUS_TARGET_P is INACTIVE (the slack) keeps the ~ encoding on PRow: it must
        // never receive a P target — OLF's slack simply absorbs the station's targetP. Writing there pinned
        // the base LF's slack angle at the freeze delta (BUS_TARGET_PHI residual 0.014, base MAX_ITER).
        public int bus1PhiRow() {
            return bus1PhiCol;
        }

        public int bus2PhiRow() {
            return bus2PhiCol;
        }

        public boolean bus1PActive() {
            return bus1PRow >= 0;
        }

        /** The row the side-1 P target is written to THIS scenario: the relocated row, or the bus's own PHI row
         *  once the scenario has opened the parent coupler (that dummy row is then a mode-0 identity and the
         *  balance is back home — the stock placement the kernels follow through relocOwn). */
        public int bus1WriteRow(int[] rowMode) {
            return bus1PRow >= 0 && rowMode != null && bus1PRow != bus1PhiCol && rowMode[bus1PRow] == 0 ? bus1PhiCol : bus1PRow;
        }

        public int bus2WriteRow(int[] rowMode) {
            return bus2PRow >= 0 && rowMode != null && bus2PRow != bus2PhiCol && rowMode[bus2PRow] == 0 ? bus2PhiCol : bus2PRow;
        }

        public boolean bus2PActive() {
            return bus2PRow >= 0;
        }

        /** OLF's own {@code HvdcAcEmulationSide1ActiveFlowEquationTerm.p1} at a scenario's state.
         *  Called directly on the state array rather than through the equation term: loading a
         *  state into the shared {@code StateVector} notifies {@code AcNetworkVector}, which
         *  recomputes flows AND derivatives for every branch in the network — a whole-network pass
         *  per scenario per check round, which measured 6 s of a 10 s orchestrator on a 9k-branch
         *  case. The row IS the bus's BUS_PHI row (one device row per variable). */
        public double p1At(double[] x) {
            return HvdcAcEmulationSide1ActiveFlowEquationTerm.p1(p0, k, loss1, loss2, r, x[bus1PhiRow()], x[bus2PhiRow()]);
        }

        /** OLF's own {@code HvdcAcEmulationSide2ActiveFlowEquationTerm.p2}; see {@link #p1At}. */
        public double p2At(double[] x) {
            return HvdcAcEmulationSide2ActiveFlowEquationTerm.p2(p0, k, loss1, loss2, r, x[bus1PhiRow()], x[bus2PhiRow()]);
        }
    }

    public static List<Ctl> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                      int nElements,
                                      com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData gpuData) {
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
            // The bus's own BUS_PHI column (the angle the emulation reads) and, SEPARATELY, the row its
            // P equation actually lives on — the parent coupler's DUMMY_P when the relocation moved it
            // (tasks #68). Resolved through the zero-impedance pack so it cannot drift from the kernels.
            int phi1 = phiRows.get(hvdc.getBus1().getNum());
            int phi2 = phiRows.get(hvdc.getBus2().getNum());
            // The P-target row IS the relocated DUMMY_P row (GpuAcData.relocatedPRow) — the row the bus's P
            // balance lives on under the coupler-tree relocation; the bus's own PHI row hosts the coupler's
            // ZERO_PHI there (target 0). The 2026-10-02 attempt at exactly this was "catastrophic" (108 ->
            // 13194 bad solves) because the kernel read the ANGLE at the same index it wrote the target to, so
            // a relocated bus's angle was read off a DUMMY_P variable. The two roles are separate arrays now
            // (bus{1,2}PhiCol for the reads, bus{1,2}PRow for the writes), and the kernel falls back to the own
            // row in a scenario that kills the parent coupler (relocOwn, the stock placement). Measured as the
            // only device failure on an OLF-converging busbar contingency, BUS-1219_BBS: the saturation
            // delta of SYN_HVDC_4 landed on a ZERO_PHI row, reversed the emulated flow and the limits loop
            // toggled SAT_2_TO_1 <-> SAT_1_TO_2 to its 32-pass budget (tasks #68 / #69).
            int r1 = gpuData.relocatedPRow(phi1);
            int r2 = gpuData.relocatedPRow(phi2);
            boolean p1Active = es.getEquation(hvdc.getBus1().getNum(), AcEquationType.BUS_TARGET_P)
                    .filter(com.powsybl.openloadflow.equations.Equation::isActive).isPresent();
            boolean p2Active = es.getEquation(hvdc.getBus2().getNum(), AcEquationType.BUS_TARGET_P)
                    .filter(com.powsybl.openloadflow.equations.Equation::isActive).isPresent();
            if (HVDC_TRACE_ROW != -1) {
                System.err.printf("HVDC_BUILD h=%d id=%s elem=%d olfStatus=%s st1TargetP=%.9f st2TargetP=%.9f p1Active=%b%n",
                        h, hvdc.getId(), hvOffset + h, ac.getAcEmulationStatus(), hvdc.getConverterStation1().getTargetP(),
                        hvdc.getConverterStation2().getTargetP(), p1Active);
            }
            out.add(new Ctl(hvOffset + h,
                    p1Active ? r1 : ~r1, p2Active ? r2 : ~r2,
                    phi1, phi2,
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
    public static java.util.Map<LfHvdc, Integer> elementIndices(LfNetwork network, int nElements) {
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
    private static final int HVDC_TRACE_ROW = System.getenv("OLF_GPU_HVDC_TRACE") == null ? -1
            : "all".equals(System.getenv("OLF_GPU_HVDC_TRACE")) ? -2 : Integer.parseInt(System.getenv("OLF_GPU_HVDC_TRACE"));

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
    public static int[][] applyFreeze(int chunkSize, int realCount, int n, List<Ctl> ctls,
                               double[] baseState, double[] x0PerScenario, boolean[][] disabledMask,
                               double[] baseTargets, List<Integer> ovIndexList, List<Double> ovValueList,
                               double[] appliedD1, double[] appliedD2) {
        GpuBatchOverrides overrides = new GpuBatchOverrides();   // the base load flow's plain lists, same order
        for (int k = 0; k < ovIndexList.size(); k++) {
            overrides.add(ovIndexList.get(k), ovValueList.get(k));
        }
        int[][] status = applyFreeze(chunkSize, realCount, n, ctls, baseState, x0PerScenario, null, null, disabledMask,
                baseTargets, overrides, appliedD1, appliedD2);
        ovIndexList.clear();
        ovValueList.clear();
        for (int k = 0; k < overrides.size(); k++) {
            ovIndexList.add(overrides.indexAt(k));
            ovValueList.add(overrides.valueAt(k));
        }
        return status;
    }

    /**
     * As above, on the batch's own override list (its index is kept, not rebuilt per call). A scenario's warm start is
     * {@code x0PerScenario}'s slice when given; otherwise the base state with the scenario's seeds written over it
     * ({@code seedIdx} flat {@code s * n + row}, the start state the device builds from the same seeds) — applied to
     * one shared copy and put back after, so a chunk never materializes {@code chunkSize * n} doubles for it.
     */
    static int[][] applyFreeze(int chunkSize, int realCount, int n, List<Ctl> ctls,
                               double[] baseState, double[] x0PerScenario, int[] seedIdx, double[] seedVal,
                               boolean[][] disabledMask, double[] baseTargets, GpuBatchOverrides overrides,
                               double[] appliedD1, double[] appliedD2) {
        return applyFreeze(chunkSize, realCount, n, ctls, baseState, x0PerScenario, seedIdx, seedVal, disabledMask,
                baseTargets, overrides, appliedD1, appliedD2, null);
    }

    /** @param rowModePerScenario the per-scenario row modes (null = base layout everywhere): a converter bus whose
     *  parent coupler the scenario opens takes its P target on its OWN row (see Ctl.bus1WriteRow). */
    public static int[][] applyFreeze(int chunkSize, int realCount, int n, List<Ctl> ctls,
                               double[] baseState, double[] x0PerScenario, int[] seedIdx, double[] seedVal,
                               boolean[][] disabledMask, double[] baseTargets, GpuBatchOverrides overrides,
                               double[] appliedD1, double[] appliedD2, int[][] rowModePerScenario) {
        int[][] status = new int[chunkSize][ctls.size()];
        double[] warm = baseState.clone();
        int[][] seedsOf = x0PerScenario == null && seedIdx != null ? seedsByScenario(seedIdx, realCount, n) : null;
        for (int s = 0; s < realCount; s++) {
            int[] rm = rowModePerScenario == null ? null : rowModePerScenario[s];
            if (x0PerScenario != null) {
                System.arraycopy(x0PerScenario, s * n, warm, 0, n);
            } else if (seedsOf != null) {
                for (int k : seedsOf[s]) {
                    warm[seedIdx[k] - s * n] = seedVal[k];
                }
            }
            for (int h = 0; h < ctls.size(); h++) {
                Ctl c = ctls.get(h);
                if (HVDC_TRACE_ROW != -1) {
                    System.err.printf("HVDC_CTL s=%d h=%d elem=%d baseStatus=%d baseD1=%.9f baseD2=%.9f disabled=%b p1warm=%.9f p2warm=%.9f pMax12=%.6f baseT1=%.9f%n",
                            s, h, c.hvElemIndex(), c.baseStatus(), c.baseD1(), c.baseD2(), disabledMask[s][c.hvElemIndex()],
                            c.p1At(warm), c.p2At(warm), c.pMax12(), c.bus1PRow() >= 0 ? baseTargets[c.bus1PRow()] : Double.NaN);
                }
                if (disabledMask[s][c.hvElemIndex()]) {
                    continue;                                // outaged hvdc — stays as the contingency left it
                }
                double[] d = frozenDeltas(c, c.p1At(warm), c.p2At(warm));
                if (HVDC_TRACE_ROW == -2 || (HVDC_TRACE_ROW >= 0 && (c.bus1PhiRow() == HVDC_TRACE_ROW || c.bus2PhiRow() == HVDC_TRACE_ROW))) {
                    // OLF_GPU_HVDC_TRACE=<phiRow>: what this scenario's freeze fixed the hvdc at - the
                    // flows read at the warm-start state and the deltas that go into the two converter
                    // rows. OLF freezes each contingency at its own restored base angles; a scenario
                    // whose warm start is not the base freezes at different flows and nothing else
                    // in the residual will say so (the frozen P IS the equation).
                    System.err.printf("HVDC_FREEZE s=%d h=%d row1=%d row2=%d warmIsBase=%b p1=%.9f p2=%.9f d1=%.9f d2=%.9f"
                            + " baseD1=%.9f baseD2=%.9f pMax12=%.6f pMax21=%.6f%n", s, h, c.bus1PhiRow(), c.bus2PhiRow(),
                            x0PerScenario == null, c.p1At(warm), c.p2At(warm), d[0], d[1], c.baseD1(), c.baseD2(),
                            c.pMax12(), c.pMax21());
                }
                if (appliedD1 != null) {
                    appliedD1[s * ctls.size() + h] = d[0] - c.baseD1();
                    appliedD2[s * ctls.size() + h] = d[1] - c.baseD2();
                }
                status[s][h] = FROZEN;
                disabledMask[s][c.hvElemIndex()] = true;
                // base frame: baseTargets (or the row's existing override — e.g. a lost load on the
                // converter bus) already carry the base injections
                if (c.bus1PActive()) {
                    overrides.merge(s * n + c.bus1WriteRow(rm), baseTargets[c.bus1WriteRow(rm)], c.baseD1(), d[0]);
                }
                if (c.bus2PActive()) {
                    overrides.merge(s * n + c.bus2WriteRow(rm), baseTargets[c.bus2WriteRow(rm)], c.baseD2(), d[1]);
                }
            }
            if (seedsOf != null) {
                for (int k : seedsOf[s]) {                   // back to the base state for the next scenario
                    int row = seedIdx[k] - s * n;
                    warm[row] = baseState[row];
                }
            }
        }
        return status;
    }

    /** The positions in {@code seedIdx} of each scenario's seeds. */
    private static int[][] seedsByScenario(int[] seedIdx, int realCount, int n) {
        int[] count = new int[realCount];
        for (int flat : seedIdx) {
            count[flat / n]++;
        }
        int[][] seedsOf = new int[realCount][];
        for (int s = 0; s < realCount; s++) {
            seedsOf[s] = new int[count[s]];
            count[s] = 0;
        }
        for (int k = 0; k < seedIdx.length; k++) {
            int s = seedIdx[k] / n;
            seedsOf[s][count[s]++] = k;
        }
        return seedsOf;
    }

    /**
     * Re-derive the converter-bus target overrides and the hv element bits from a set of statuses,
     * onto {@code ovIdx}/{@code ovVal} seeded with the PRE-FREEZE list. This is the orchestrator's
     * per-round rebuild, extracted so the in-kernel loops can leave the HOST's mirrors consistent
     * with what they did on the device — otherwise the orchestrators that run next re-upload a
     * pre-freeze override list and re-apply a freeze the kernel already lifted.
     */
    public static void applyStatusOverrides(int chunkSize, int realCount, int n, List<Ctl> ctls,
                                     int[][] status, byte[] ctgOut, boolean[][] disabledMask,
                                     double[] baseTargets, GpuBatchOverrides overrides,
                                     double[] appliedD1, double[] appliedD2, int[][] rowModePerScenario) {
        int nh = ctls.size();
        for (int s = 0; s < realCount; s++) {
            int[] rm = rowModePerScenario == null ? null : rowModePerScenario[s];
            for (int h = 0; h < nh; h++) {
                Ctl c = ctls.get(h);
                if (ctgOut[s * nh + h] != 0) {
                    continue;                                // the contingency owns that bit
                }
                if (status[s][h] == LINEAR) {
                    disabledMask[s][c.hvElemIndex()] = false;
                    if (c.baseStatus() != LINEAR) {          // strip the base injections from the targets
                        if (c.bus1PActive()) {
                            overrides.merge(s * n + c.bus1WriteRow(rm), baseTargets[c.bus1WriteRow(rm)], c.baseD1(), 0);
                        }
                        if (c.bus2PActive()) {
                            overrides.merge(s * n + c.bus2WriteRow(rm), baseTargets[c.bus2WriteRow(rm)], c.baseD2(), 0);
                        }
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
                    if (c.bus1PActive()) {
                        overrides.merge(s * n + c.bus1WriteRow(rm), baseTargets[c.bus1WriteRow(rm)], c.baseD1(), d[0]);
                    }
                    if (c.bus2PActive()) {
                        overrides.merge(s * n + c.bus2WriteRow(rm), baseTargets[c.bus2WriteRow(rm)], c.baseD2(), d[1]);
                    }
                }
            }
        }
    }

    /** Seed the NON-LINEAR-at-base statuses (a saturated/frozen base — the extraction gate used to
     *  route such runs whole-run CPU): entries the freeze did not claim keep the BASE status with
     *  the hv bit set and NO target override (baseTargets already carry the base injections), and
     *  PADDING slots get the base bits so they still converge at iteration 0. An entry whose bit is
     *  already set while its status is LINEAR is a contingency outage — left alone. */
    public static void seedBaseStatuses(int chunkSize, int realCount, List<Ctl> ctls,
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

    /** Install the two AC-emulation driver phases (freeze, limits) on a batch context: the controller
     *  tables, each scenario's starting status, the "outaged by the contingency" bits and the deltas the
     *  host-side freeze already applied. Shared by the batched SA's chunk contexts and the base load
     *  flow's batch-of-one. Returns the flat ctgOut bits (null when there is nothing to install — the
     *  context's HVDC loops are then cleared). */
    public static byte[] installDriverLoops(long handle, List<Ctl> ctls, int chunkSize, int[][] hvdcStatus,
                                            boolean[][] disabledMask, double[] appliedD1, double[] appliedD2,
                                            boolean limits) {
        if (ctls == null || ctls.isEmpty()) {
            HvdcNative.setBatchHvdcLoops(handle, null, null, null, null, null, null, null, null, null, null, 0);
            return null;
        }
        int nh = ctls.size();
        int[] hvElem = new int[nh];
        int[] hvB1 = new int[nh];
        int[] hvB2 = new int[nh];
        int[] hvPhi1 = new int[nh];
        int[] hvPhi2 = new int[nh];
        double[] hvIn = new double[nh * 9];
        for (int h = 0; h < nh; h++) {
            Ctl c = ctls.get(h);
            hvElem[h] = c.hvElemIndex();          // GLOBAL element index
            hvB1[h] = c.bus1PRow();               // where the P target is WRITTEN (relocated row, ~ when inactive)
            hvB2[h] = c.bus2PRow();
            hvPhi1[h] = c.bus1PhiCol();           // where the ANGLE is READ (always the bus's own column)
            hvPhi2[h] = c.bus2PhiCol();
            hvIn[h * 9] = c.p0();
            hvIn[h * 9 + 1] = c.k();
            hvIn[h * 9 + 2] = c.pMax12();
            hvIn[h * 9 + 3] = c.pMax21();
            hvIn[h * 9 + 4] = c.loss1();
            hvIn[h * 9 + 5] = c.loss2();
            hvIn[h * 9 + 6] = c.r();
            hvIn[h * 9 + 7] = c.baseD1();
            hvIn[h * 9 + 8] = c.baseD2();
        }
        byte[] ctgOut = new byte[chunkSize * nh];
        int[] flatStatus = new int[chunkSize * nh];
        for (int s2 = 0; s2 < chunkSize; s2++) {
            for (int h = 0; h < nh; h++) {
                flatStatus[s2 * nh + h] = hvdcStatus[s2][h];
                // "the contingency outaged it" is the bit set on a LINEAR entry
                ctgOut[s2 * nh + h] = (byte) (hvdcStatus[s2][h] == LINEAR
                        && disabledMask[s2][ctls.get(h).hvElemIndex()] ? 1 : 0);
            }
        }
        HvdcNative.setBatchHvdcLoops(handle, hvElem, hvB1, hvB2, hvPhi1, hvPhi2, hvIn, ctgOut,
                flatStatus, appliedD1, appliedD2, limits ? 1 : 0);
        return ctgOut;
    }

    /** Give the driver's final AC-emulation statuses (scenario 0) back to the network the way OLF's own
     *  loops do — the same {@code AcEmulationControl} switches, which set the converter stations'
     *  targetP and re-activate/deactivate the emulation terms through OLF's listeners. */
    public static void writeBack(LfNetwork network, List<Ctl> ctls, int[] deviceStatus, int nElements) {
        if (ctls == null || ctls.isEmpty() || deviceStatus == null) {
            return;
        }
        java.util.Map<Integer, LfHvdc> byElem = new java.util.HashMap<>();
        elementIndices(network, nElements).forEach((h, e) -> byElem.put(e, h));
        for (int h = 0; h < ctls.size() && h < deviceStatus.length; h++) {
            LfHvdc hvdc = byElem.get(ctls.get(h).hvElemIndex());
            if (hvdc == null) {
                continue;
            }
            var control = hvdc.getAcEmulationControl();
            var now = control.getAcEmulationStatus();
            switch (deviceStatus[h]) {
                case LINEAR -> {
                    if (now != LfHvdc.AcEmulationControl.AcEmulationStatus.LINEAR_MODE) {
                        control.switchToLinearMode();
                    }
                }
                case SAT_1_TO_2 -> {
                    if (now != LfHvdc.AcEmulationControl.AcEmulationStatus.SATURATION_MODE_FROM_CS1_TO_CS2) {
                        control.switchToSaturationFromCS1toCS2(true);
                    }
                }
                case SAT_2_TO_1 -> {
                    if (now != LfHvdc.AcEmulationControl.AcEmulationStatus.SATURATION_MODE_FROM_CS2_TO_CS1) {
                        control.switchToSaturationFromCS2toCS1(true);
                    }
                }
                case FROZEN -> {
                    if (now != LfHvdc.AcEmulationControl.AcEmulationStatus.FROZEN) {
                        control.switchToFrozenState(true);
                    }
                }
                default -> { }
            }
        }
    }
}
