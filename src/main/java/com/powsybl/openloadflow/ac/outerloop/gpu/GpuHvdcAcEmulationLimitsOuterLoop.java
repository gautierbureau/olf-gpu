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
import com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix.CouplerTreeReHoming;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcEmulationStatus;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.util.gpu.GpuTrace;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of OLF's {@code AcHvdcAcEmulationLimitsOuterLoop}: linear ↔ saturation transitions — a linear
 * HVDC whose |P| overpasses pMax saturates at pMax (with converter/line losses), a saturated one back inside its
 * limits returns to linear, and an over-reversed one switches saturation side. The transitions run in the kernel
 * (hvdc_ac_emulation.cu); this class holds the per-(scenario, hvdc) status machinery they drive, which
 * {@link GpuFreezingHvdcACEmulationOuterloop} also uses: the controllers, the statuses (FROZEN included), the
 * converter-bus overrides, the driver install and applyToNetwork.
 *
 * <p>Device mechanics per (scenario, hvdc): a NON-LINEAR status sets the hv element's per-scenario
 * DISABLED bit (fillHvdcBatch drops the coupling terms) and the
 * VSC injections land as CONVERTER-BUS target overrides (the stations' targetP is 0 in the base
 * linear extraction, so the bus-target delta IS the station targetP, generator convention). Flow
 * evaluations call OLF's OWN p1/p2 formulas directly on the scenario's state array (see
 * {@link Controller#p1At}) — the equation terms read a shared {@code StateVector} whose every load
 * recomputes the whole network's flows and derivatives, which is unaffordable per scenario.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuHvdcAcEmulationLimitsOuterLoop {

    private GpuHvdcAcEmulationLimitsOuterLoop() {
    }

    static final int HVDC_TRACE_ROW = !GpuTrace.HVDC_TRACE.isOn() ? -1
            : "all".equals(GpuTrace.HVDC_TRACE.value()) ? -2 : Integer.parseInt(GpuTrace.HVDC_TRACE.value());

    /** One packed AC-emulation HVDC: the hv element index (per-scenario disabled bit), the two
     *  converter buses' P-target rows, OLF's own linear p1/p2 terms, the emulation limits, and the
     *  BASE frame — the base status plus the base VSC station targetP contributions ALREADY inside
     *  baseTargets (0 for a linear base; a saturated/frozen base carries its injections there, so
     *  every status override composes as {@code baseTargets - baseD + d(status)}). */
    public record Controller(int hvdcElementIndex, int bus1PRow, int bus2PRow, int bus1PhiCol, int bus2PhiCol,
               double p0, double k,
               double pMax12, double pMax21, double loss1, double loss2, double r,
               int baseStatus, double baseD1, double baseD2) {

        // TWO DISTINCT ROLES, and they are NOT the same row under the coupler-tree relocation:
        //   bus{1,2}PRow   = where the P TARGET is written (the bus's P equation). For a relocated
        //                    coupler child that is the parent's DUMMY_P row, NOT the bus's own PHI row,
        //                    which hosts the coupler's ZERO_PHI (target 0). See CouplerTreeReHoming.relocatedPRow
        //                    and tasks #68: conflating the two put the freeze setpoints on all ten
        //                    VSC-station breakers' ZERO_PHI rows.
        //   bus{1,2}PhiCol = the ANGLE COLUMN the emulation formula reads. Always the bus's OWN BUS_PHI
        //                    column, relocation or not — a relocation moves EQUATIONS, never variables.
        // A converter bus whose BUS_TARGET_P is INACTIVE (the slack) keeps the ~ encoding on PRow: it must
        // never receive a P target — OLF's slack simply absorbs the station's targetP. Writing there fixed
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
         *  balance is back home — the stock placement the kernels follow through relocatedChildOwnRow). */
        public int bus1WriteRow(int[] rowMode) {
            return bus1PRow >= 0 && rowMode != null && bus1PRow != bus1PhiCol && rowMode[bus1PRow] == GpuRowMode.FIXED_VARIABLE ? bus1PhiCol : bus1PRow;
        }

        public int bus2WriteRow(int[] rowMode) {
            return bus2PRow >= 0 && rowMode != null && bus2PRow != bus2PhiCol && rowMode[bus2PRow] == GpuRowMode.FIXED_VARIABLE ? bus2PhiCol : bus2PRow;
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

    public static List<Controller> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
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
        int hvdcAcEmulationOffset = nElements - packed.size();
        List<Controller> out = new ArrayList<>();
        for (int h = 0; h < packed.size(); h++) {
            LfHvdc hvdc = packed.get(h);
            var ac = hvdc.getAcEmulationControl();
            int baseStatus = GpuAcEmulationStatus.of(ac.getAcEmulationStatus());
            // The bus's own BUS_PHI column (the angle the emulation reads) and, SEPARATELY, the row its
            // P equation actually lives on — the parent coupler's DUMMY_P when the relocation moved it
            // (tasks #68). Resolved through the zero-impedance pack so it cannot drift from the kernels.
            int phi1 = phiRows.get(hvdc.getBus1().getNum());
            int phi2 = phiRows.get(hvdc.getBus2().getNum());
            // The P-target row IS the relocated DUMMY_P row (CouplerTreeReHoming.relocatedPRow) — the row the bus's P
            // balance lives on under the coupler-tree relocation; the bus's own PHI row hosts the coupler's
            // ZERO_PHI there (target 0). The 2026-10-02 attempt at exactly this was "catastrophic" (108 ->
            // 13194 bad solves) because the kernel read the ANGLE at the same index it wrote the target to, so
            // a relocated bus's angle was read off a DUMMY_P variable. The two roles are separate arrays now
            // (bus{1,2}PhiCol for the reads, bus{1,2}PRow for the writes), and the kernel falls back to the own
            // row in a scenario that kills the parent coupler (relocatedChildOwnRow, the stock placement). Measured as the
            // only device failure on an OLF-converging busbar contingency, BUS-1219_BBS: the saturation
            // delta of SYN_HVDC_4 landed on a ZERO_PHI row, reversed the emulated flow and the limits loop
            // toggled SAT_2_TO_1 <-> SAT_1_TO_2 to its 32-pass budget (tasks #68 / #69).
            int r1 = CouplerTreeReHoming.relocatedPRow(gpuData.zeroImpedanceBranch().indices(), phi1);
            int r2 = CouplerTreeReHoming.relocatedPRow(gpuData.zeroImpedanceBranch().indices(), phi2);
            boolean p1Active = es.getEquation(hvdc.getBus1().getNum(), AcEquationType.BUS_TARGET_P)
                    .filter(com.powsybl.openloadflow.equations.Equation::isActive).isPresent();
            boolean p2Active = es.getEquation(hvdc.getBus2().getNum(), AcEquationType.BUS_TARGET_P)
                    .filter(com.powsybl.openloadflow.equations.Equation::isActive).isPresent();
            if (HVDC_TRACE_ROW != -1) {
                System.err.printf("HVDC_BUILD h=%d id=%s elem=%d olfStatus=%s st1TargetP=%.9f st2TargetP=%.9f p1Active=%b%n",
                        h, hvdc.getId(), hvdcAcEmulationOffset + h, ac.getAcEmulationStatus(), hvdc.getConverterStation1().getTargetP(),
                        hvdc.getConverterStation2().getTargetP(), p1Active);
            }
            out.add(new Controller(hvdcAcEmulationOffset + h,
                    p1Active ? r1 : ~r1, p2Active ? r2 : ~r2,
                    phi1, phi2,
                    ac.getP0(), ac.getDroop() * 180 / Math.PI,
                    ac.getPMaxFromCS1toCS2(), ac.getPMaxFromCS2toCS1(),
                    hvdc.getConverterStation1().getLossFactor() / 100,
                    hvdc.getConverterStation2().getLossFactor() / 100, hvdc.getR(),
                    baseStatus,
                    baseStatus == GpuAcEmulationStatus.LINEAR_MODE ? 0 : hvdc.getConverterStation1().getTargetP(),
                    baseStatus == GpuAcEmulationStatus.LINEAR_MODE ? 0 : hvdc.getConverterStation2().getTargetP()));
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
        int hvdcAcEmulationOffset = nElements - packed.size();
        for (int h = 0; h < packed.size(); h++) {
            out.put(packed.get(h), hvdcAcEmulationOffset + h);
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
    static double[] saturationDeltas(Controller c, boolean from1to2) {
        if (from1to2) {
            double pAbs = AbstractHvdcAcEmulationFlowEquationTerm.getAbsActivePowerWithLosses(c.pMax12(), c.loss1(), c.loss2(), c.r());
            return new double[] {-c.pMax12(), pAbs};
        }
        double pAbs = AbstractHvdcAcEmulationFlowEquationTerm.getAbsActivePowerWithLosses(c.pMax21(), c.loss2(), c.loss1(), c.r());
        return new double[] {pAbs, -c.pMax21()};
    }

    /** Seed the NON-LINEAR-at-base statuses (a saturated/frozen base — the extraction gate used to
     *  route such runs whole-run CPU): entries the freeze did not claim keep the BASE status with
     *  the hv bit set and NO target override (baseTargets already carry the base injections), and
     *  PADDING slots get the base bits so they still converge at iteration 0. An entry whose bit is
     *  already set while its status is LINEAR is a contingency outage — left alone. */
    public static void seedBaseStatuses(int chunkSize, int realCount, List<Controller> controllers,
                                 int[][] status, boolean[][] disabledMask) {
        for (int h = 0; h < controllers.size(); h++) {
            Controller c = controllers.get(h);
            if (c.baseStatus() == GpuAcEmulationStatus.LINEAR_MODE) {
                continue;
            }
            for (int s = 0; s < chunkSize; s++) {
                if (s >= realCount) {
                    disabledMask[s][c.hvdcElementIndex()] = true;
                } else if (status[s][h] == GpuAcEmulationStatus.LINEAR_MODE && !disabledMask[s][c.hvdcElementIndex()]) {
                    status[s][h] = c.baseStatus();
                    disabledMask[s][c.hvdcElementIndex()] = true;
                }
            }
        }
    }

    /** Install the two AC-emulation driver phases (freeze, limits) on a batch context: the controller
     *  tables, each scenario's starting status, the "outaged by the contingency" bits and the deltas the
     *  host-side freeze already applied. Shared by the batched SA's chunk contexts and the base load
     *  flow's batch-of-one. Without controllers the context's HVDC loops are cleared. */
    public static void installDriverLoops(long handle, List<Controller> controllers, int chunkSize, int[][] hvdcStatus,
                                            boolean[][] disabledMask, double[] appliedDelta1, double[] appliedDelta2,
                                            boolean limits) {
        if (controllers == null || controllers.isEmpty()) {
            HvdcNative.setBatchHvdcLoops(handle, null, null, null, null, null, null, null, null, null, null, 0);
            return;
        }
        int nControllers = controllers.size();
        int[] hvdcElement = new int[nControllers];
        int[] hvdcBus1PRows = new int[nControllers];
        int[] hvdcBus2PRows = new int[nControllers];
        int[] hvdcBus1PhiColumns = new int[nControllers];
        int[] hvdcBus2PhiColumns = new int[nControllers];
        double[] hvdcAcEmulationParameters = new double[nControllers * 9];
        for (int h = 0; h < nControllers; h++) {
            Controller c = controllers.get(h);
            hvdcElement[h] = c.hvdcElementIndex();          // GLOBAL element index
            hvdcBus1PRows[h] = c.bus1PRow();               // where the P target is WRITTEN (relocated row, ~ when inactive)
            hvdcBus2PRows[h] = c.bus2PRow();
            hvdcBus1PhiColumns[h] = c.bus1PhiCol();           // where the ANGLE is READ (always the bus's own column)
            hvdcBus2PhiColumns[h] = c.bus2PhiCol();
            hvdcAcEmulationParameters[h * 9] = c.p0();
            hvdcAcEmulationParameters[h * 9 + 1] = c.k();
            hvdcAcEmulationParameters[h * 9 + 2] = c.pMax12();
            hvdcAcEmulationParameters[h * 9 + 3] = c.pMax21();
            hvdcAcEmulationParameters[h * 9 + 4] = c.loss1();
            hvdcAcEmulationParameters[h * 9 + 5] = c.loss2();
            hvdcAcEmulationParameters[h * 9 + 6] = c.r();
            hvdcAcEmulationParameters[h * 9 + 7] = c.baseD1();
            hvdcAcEmulationParameters[h * 9 + 8] = c.baseD2();
        }
        byte[] ctgOut = new byte[chunkSize * nControllers];
        int[] flatStatus = new int[chunkSize * nControllers];
        for (int s2 = 0; s2 < chunkSize; s2++) {
            for (int h = 0; h < nControllers; h++) {
                flatStatus[s2 * nControllers + h] = hvdcStatus[s2][h];
                // "the contingency outaged it" is the bit set on a LINEAR entry
                ctgOut[s2 * nControllers + h] = (byte) (hvdcStatus[s2][h] == GpuAcEmulationStatus.LINEAR_MODE
                        && disabledMask[s2][controllers.get(h).hvdcElementIndex()] ? 1 : 0);
            }
        }
        HvdcNative.setBatchHvdcLoops(handle, hvdcElement, hvdcBus1PRows, hvdcBus2PRows, hvdcBus1PhiColumns, hvdcBus2PhiColumns, hvdcAcEmulationParameters, ctgOut,
                flatStatus, appliedDelta1, appliedDelta2, limits ? 1 : 0);
    }

    /** Give the driver's final AC-emulation statuses (scenario 0) back to the network the way OLF's own
     *  loops do — the same {@code AcEmulationControl} switches, which set the converter stations'
     *  targetP and re-activate/deactivate the emulation terms through OLF's listeners. */
    public static void applyToNetwork(LfNetwork network, List<Controller> controllers, int[] deviceStatus, int nElements) {
        if (controllers == null || controllers.isEmpty() || deviceStatus == null) {
            return;
        }
        java.util.Map<Integer, LfHvdc> byElem = new java.util.HashMap<>();
        elementIndices(network, nElements).forEach((h, e) -> byElem.put(e, h));
        for (int h = 0; h < controllers.size() && h < deviceStatus.length; h++) {
            LfHvdc hvdc = byElem.get(controllers.get(h).hvdcElementIndex());
            if (hvdc == null) {
                continue;
            }
            var control = hvdc.getAcEmulationControl();
            var now = control.getAcEmulationStatus();
            switch (deviceStatus[h]) {
                case GpuAcEmulationStatus.LINEAR_MODE -> {
                    if (now != LfHvdc.AcEmulationControl.AcEmulationStatus.LINEAR_MODE) {
                        control.switchToLinearMode();
                    }
                }
                case GpuAcEmulationStatus.SATURATION_MODE_FROM_CS1_TO_CS2 -> {
                    if (now != LfHvdc.AcEmulationControl.AcEmulationStatus.SATURATION_MODE_FROM_CS1_TO_CS2) {
                        control.switchToSaturationFromCS1toCS2(true);
                    }
                }
                case GpuAcEmulationStatus.SATURATION_MODE_FROM_CS2_TO_CS1 -> {
                    if (now != LfHvdc.AcEmulationControl.AcEmulationStatus.SATURATION_MODE_FROM_CS2_TO_CS1) {
                        control.switchToSaturationFromCS2toCS1(true);
                    }
                }
                case GpuAcEmulationStatus.FROZEN -> {
                    if (now != LfHvdc.AcEmulationControl.AcEmulationStatus.FROZEN) {
                        control.switchToFrozenState(true);
                    }
                }
                default -> { }
            }
        }
    }
}
