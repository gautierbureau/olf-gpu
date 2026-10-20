/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSkips;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of OLF's {@code IncrementalShuntVoltageControlOuterLoop}, sharing the incremental-family
 * machinery with {@link GpuIncrementalTransformerVoltageControlOuterLoop}: batched in-kernel dV/dB sensitivities (unit RHS at
 * each controller shunt's pinned {@code SHUNT_TARGET_B} identity row), then OLF's section decisions
 * replicated verbatim per scenario — the |B|-magnitude-descending shunt ordering, the per-round
 * {@code maxSectionShift} cap (OLF's {@code > maxSectionShift} guard, checked BEFORE the increment),
 * one-section-at-a-time moves via the closest-B search, and the {@code AllowedDirection} ratchet per
 * SUB-controller (an aggregated LfShunt can carry several). No sensitivity filter exists in this loop
 * (unlike the transformer one). Section moves land as per-scenario {@code SHUNT_TARGET_B} target
 * overrides + a B state seed (the pinned identity value) and additive (dG, dB) shunt-admittance
 * overrides (the static G follows the section; the static B is shadowed by the SHUNT_B variable);
 * re-solve warm and iterate until every scenario is in deadband or nothing moves.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuIncrementalShuntVoltageControlOuterLoop {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuIncrementalShuntVoltageControlOuterLoop.class);

    private static final int MAX_DIRECTION_CHANGE = 3;       // OLF IncrementalShuntVoltageControlOuterLoop
    private static final double MIN_TARGET_DEADBAND_KV = 0.1;

    private GpuIncrementalShuntVoltageControlOuterLoop() {
    }

    /** One SUB-controller (OLF LfShunt.Controller): per-section B/G tables + the ratchet identity. */
    public record Sub(double[] sectionsB, double[] sectionsG, int basePos, int minPos) {
        public double b(int pos) {
            return sectionsB[pos];
        }

        double g(int pos) {
            return sectionsG[pos];
        }
    }

    /** One controller shunt: the SHUNT_B variable row (pinned identity — unit-RHS row, target index and
     *  state index coincide), the controlled bus V row, and its sub-controllers. */
    public record Controller(int bRow, int vRow, int ctrlBusNum, double targetV, double halfDeadband,
                      List<Sub> subs, double bMagnitude, int shuntAdmittanceOverrideElement, int elemIndexAbs, int shuntNum) {
    }

    private static final class Ratchet {
        int dirChangeCount;
        int currentDirection;
        int allowedDirection;

        void update(int direction) {                         // OLF ControllerContext.updateAllowedDirection
            if (dirChangeCount < MAX_DIRECTION_CHANGE) {
                if (currentDirection != 0 && currentDirection != direction) {
                    dirChangeCount++;
                }
                currentDirection = direction;
            } else {
                allowedDirection = direction;
            }
        }
    }

    public static List<Controller> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                             GpuAcElementMapper elementMapper, int shuntPackOffset) {
        List<Controller> out = new ArrayList<>();
        Map<Integer, Integer> busVRows = new LinkedHashMap<>();
        Map<Integer, Integer> shuntBRows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVRows.put(v.getElementNum(), v.getRow());
            } else if (v.getType() == AcVariableType.SHUNT_B) {
                shuntBRows.put(v.getElementNum(), v.getRow());
            }
        }
        if (System.getenv("OLF_GPU_SHUNT_TRACE") != null) {
            System.err.println("SHUNT_TRACE build: controllerElements="
                    + network.getControllerElements(com.powsybl.openloadflow.network.VoltageControl.Type.SHUNT).size()
                    + " shuntBRows=" + shuntBRows.size());
        }
        for (LfShunt shunt : network.<LfShunt>getControllerElements(com.powsybl.openloadflow.network.VoltageControl.Type.SHUNT)) {
            var vcOpt = shunt.getVoltageControl();
            if (vcOpt.isEmpty() || shunt.getControllers().isEmpty()) {
                continue;
            }
            var vc = vcOpt.orElseThrow();
            Integer bRow = shuntBRows.get(shunt.getNum());
            Integer vRow = busVRows.get(vc.getControlledBus().getNum());
            // The element mapper packs only the per-bus FIXED shunt aggregate — a CONTROLLER shunt
            // is a separate LfShunt and legitimately maps to -1. Its b rides the SHUNT_B state
            // variable (target override at bRow + x0 seed), so the element index is only needed for
            // STATIC-G deltas (shuntAdmittanceOverrides) and the per-scenario outage mask: accept -1 when the sections
            // carry no G variation, and guard those two uses. (Before this, EVERY incremental shunt
            // controller of this shape was silently skipped and the loop never ran on the batch —
            // the dedicated test was hollow: its 5 kV deadband never moved a section post-outage.)
            int elemIndex = elementMapper.getElementIndex(shunt);
            if (bRow == null || vRow == null) {
                GpuBatchSkips.record(GpuBatchSkips.Slot.SHUNT, LOGGER,
                        "Incremental shunt controller {} not representable on the GPU batch — skipped", shunt.getId());
                continue;
            }
            List<Sub> subs = new ArrayList<>();
            double magnitude = 0;
            boolean gVaries = false;
            for (LfShunt.Controller c : shunt.getControllers()) {
                double[] sb = c.getSectionsB().stream().mapToDouble(Double::doubleValue).toArray();
                double[] sg = new double[sb.length];
                for (int i = 0; i < sb.length; i++) {
                    sg[i] = shuntSectionG(c, i);
                    gVaries |= sg[i] != sg[0];
                }
                subs.add(new Sub(sb, sg, c.getPosition(), c.getMinPosition()));
                magnitude += c.getBMagnitude();
            }
            if (elemIndex < 0 && gVaries) {
                LOGGER.info("Incremental shunt controller {} has section-G variation but no packed element — skipped", shunt.getId());
                continue;
            }
            double halfDeadband = vc.getTargetDeadband()
                    .orElse(MIN_TARGET_DEADBAND_KV / vc.getControlledBus().getNominalV()) / 2;
            out.add(new Controller(bRow, vRow, vc.getControlledBus().getNum(), vc.getTargetValue(), halfDeadband,
                    subs, magnitude, elemIndex - shuntPackOffset, elemIndex, shunt.getNum()));
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    /** Install the controllers for the in-kernel loop (OLF order 10): groups per controlled bus in
     *  OLF's |b|-descending order, sub-controller section tables, the run's maxSectionShift. Returns
     *  true when the kernel loop OWNS them (the host orchestrator must then stay off). */
    public static boolean uploadToDevice(long handle, List<Controller> ctrls, int maxSectionShift) {
        if (ctrls == null || ctrls.isEmpty()) {
            ShuntVoltageControlNative.setBatchShuntControllers(handle,
                    null, null, null, null, null, null, null, null, null, null, null, null, null, null, maxSectionShift);
            return false;
        }
        int nc = ctrls.size();
        Map<Integer, List<Integer>> byBus = new LinkedHashMap<>();
        for (int c = 0; c < nc; c++) {
            byBus.computeIfAbsent(ctrls.get(c).ctrlBusNum(), k -> new ArrayList<>()).add(c);
        }
        int[] groupPtr = new int[byBus.size() + 1];
        int[] groupCtrl = new int[nc];
        int g = 0;
        int k = 0;
        for (List<Integer> group : byBus.values()) {
            group.sort(Comparator.comparingDouble((Integer ci) -> ctrls.get(ci).bMagnitude()).reversed());
            groupPtr[g++] = k;
            for (int ci : group) {
                groupCtrl[k++] = ci;
            }
        }
        groupPtr[g] = k;
        int[] bRow = new int[nc];
        int[] vRow = new int[nc];
        double[] targetV = new double[nc];
        double[] halfDb = new double[nc];
        int[] elemAbs = new int[nc];
        int[] shuntAdmittanceOverrideElement = new int[nc];
        int[] subPtr = new int[nc + 1];
        List<Integer> secPtr = new ArrayList<>();
        List<Double> secB = new ArrayList<>();
        List<Double> secG = new ArrayList<>();
        List<Integer> subMin = new ArrayList<>();
        List<Integer> subBase = new ArrayList<>();
        for (int c = 0; c < nc; c++) {
            Controller ctl = ctrls.get(c);
            bRow[c] = ctl.bRow();
            vRow[c] = ctl.vRow();
            targetV[c] = ctl.targetV();
            halfDb[c] = ctl.halfDeadband();
            elemAbs[c] = ctl.elemIndexAbs();
            shuntAdmittanceOverrideElement[c] = ctl.elemIndexAbs() >= 0 ? ctl.shuntAdmittanceOverrideElement() : -1;
            subPtr[c] = subMin.size();
            for (Sub sub : ctl.subs()) {
                secPtr.add(secB.size());
                for (int i = 0; i < sub.sectionsB().length; i++) {
                    secB.add(sub.sectionsB()[i]);
                    secG.add(sub.sectionsG()[i]);
                }
                subMin.add(sub.minPos());
                subBase.add(sub.basePos());
            }
        }
        subPtr[nc] = subMin.size();
        secPtr.add(secB.size());
        boolean owned = ShuntVoltageControlNative.setBatchShuntControllers(handle,
                bRow, vRow, targetV, halfDb, elemAbs, shuntAdmittanceOverrideElement, groupPtr, groupCtrl, subPtr,
                secPtr.stream().mapToInt(Integer::intValue).toArray(),
                secB.stream().mapToDouble(Double::doubleValue).toArray(),
                secG.stream().mapToDouble(Double::doubleValue).toArray(),
                subMin.stream().mapToInt(Integer::intValue).toArray(),
                subBase.stream().mapToInt(Integer::intValue).toArray(), maxSectionShift) != 0;
        LOGGER.info("Uploaded {} incremental shunt controllers to device ({} groups, {} subs): {}",
                nc, byBus.size(), subMin.size(), owned ? "in-kernel loop" : "host orchestrator");
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("SHUNT_UPLOAD nCtrl=%d nGroups=%d nSubs=%d kernel=%b%n", nc, byBus.size(), subMin.size(), owned);
        }
        return owned;
    }

    /** Give the driver's final sections (scenario 0) back to the network — what OLF's own loop leaves
     *  through LfShunt.Controller.updateSectionB; the shunt's b follows through OLF's listeners. */
    public static void applyToNetwork(LfNetwork network, List<Controller> ctrls, int[] positions) {
        if (ctrls == null || ctrls.isEmpty() || positions == null) {
            return;
        }
        int u = 0;
        for (Controller ctl : ctrls) {
            LfShunt shunt = network.getShunt(ctl.shuntNum());
            List<LfShunt.Controller> subs = shunt != null ? shunt.getControllers() : List.of();
            for (int k = 0; k < ctl.subs().size(); k++, u++) {
                if (u < positions.length && k < subs.size() && subs.get(k).getPosition() != positions[u]) {
                    subs.get(k).updateSectionB(positions[u]);
                }
            }
        }
    }

    @SuppressWarnings("checkstyle:IllegalCatchError") // no section G table: the sections are pure B
    public static double shuntSectionG(LfShunt.Controller c, int pos) {
        // no public accessor for the per-section G table (only getG() at the current position) —
        // read-only reflection, same pattern as GpuTransformerVoltageControlOuterLoop.tapModels
        try {
            java.lang.reflect.Field fld = LfShunt.Controller.class.getDeclaredField("sectionsG");
            fld.setAccessible(true);
            List<Double> sg = (List<Double>) fld.get(c);
            return pos < sg.size() ? sg.get(pos) : 0;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return 0;                                        // no G table — sections are pure-B
        }
    }

}
