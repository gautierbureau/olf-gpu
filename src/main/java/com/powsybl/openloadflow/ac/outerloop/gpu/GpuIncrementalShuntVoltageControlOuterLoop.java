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
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
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
 * each controller shunt's {@code SHUNT_TARGET_B} identity row), then OLF's section decisions
 * replicated verbatim per scenario — the |B|-magnitude-descending shunt ordering, the per-round
 * {@code maxSectionShift} cap (OLF's {@code > maxSectionShift} guard, checked BEFORE the increment),
 * one-section-at-a-time moves via the closest-B search, and the {@code AllowedDirection} ratchet per
 * SUB-controller (an aggregated LfShunt can carry several). No sensitivity filter exists in this loop
 * (unlike the transformer one). Section moves land as per-scenario {@code SHUNT_TARGET_B} target
 * overrides + a B state seed (the value the identity fixes) and additive (dG, dB) shunt-admittance
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

    /** One controller shunt: the SHUNT_B variable row (fixed by its identity — unit-RHS row, target index and
     *  state index coincide), the controlled bus V row, and its sub-controllers. */
    public record Controller(int bRow, int vRow, int controlledBusNum, double targetV, double halfDeadband,
                      List<Sub> subs, double bMagnitude, int shuntAdmittanceOverrideElement, int elemIndexAbs, int shuntNum) {
    }

    private static final class ControllerContext {
        int directionChangeCount;
        int currentDirection;
        int allowedDirection;

        void update(int direction) {                         // OLF ControllerContext.updateAllowedDirection
            if (directionChangeCount < MAX_DIRECTION_CHANGE) {
                if (currentDirection != 0 && currentDirection != direction) {
                    directionChangeCount++;
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
        if (GpuTrace.SHUNT_TRACE.isOn()) {
            System.err.println("SHUNT_TRACE build: controllerElements="
                    + network.getControllerElements(com.powsybl.openloadflow.network.VoltageControl.Type.SHUNT).size()
                    + " shuntBRows=" + shuntBRows.size());
        }
        for (LfShunt shunt : network.<LfShunt>getControllerElements(com.powsybl.openloadflow.network.VoltageControl.Type.SHUNT)) {
            var optionalVoltageControl = shunt.getVoltageControl();
            if (optionalVoltageControl.isEmpty() || shunt.getControllers().isEmpty()) {
                continue;
            }
            var voltageControl = optionalVoltageControl.orElseThrow();
            Integer bRow = shuntBRows.get(shunt.getNum());
            Integer vRow = busVRows.get(voltageControl.getControlledBus().getNum());
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
            double halfDeadband = voltageControl.getTargetDeadband()
                    .orElse(MIN_TARGET_DEADBAND_KV / voltageControl.getControlledBus().getNominalV()) / 2;
            out.add(new Controller(bRow, vRow, voltageControl.getControlledBus().getNum(), voltageControl.getTargetValue(), halfDeadband,
                    subs, magnitude, elemIndex - shuntPackOffset, elemIndex, shunt.getNum()));
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    /** Install the controllers for the in-kernel loop (OLF order 10): groups per controlled bus in
     *  OLF's |b|-descending order, sub-controller section tables, the run's maxSectionShift. Returns
     *  true when the kernel loop OWNS them (the host orchestrator must then stay off). */
    public static boolean uploadToDevice(long handle, List<Controller> controllers, int maxSectionShift) {
        if (controllers == null || controllers.isEmpty()) {
            ShuntVoltageControlNative.setBatchIncrementalShuntVoltageControl(handle,
                    null, null, null, null, null, null, null, null, null, null, null, null, null, null, maxSectionShift);
            return false;
        }
        int nControllers = controllers.size();
        Map<Integer, List<Integer>> byBus = new LinkedHashMap<>();
        for (int c = 0; c < nControllers; c++) {
            byBus.computeIfAbsent(controllers.get(c).controlledBusNum(), k -> new ArrayList<>()).add(c);
        }
        int[] groupControllerStart = new int[byBus.size() + 1];
        int[] groupController = new int[nControllers];
        int g = 0;
        int k = 0;
        for (List<Integer> group : byBus.values()) {
            group.sort(Comparator.comparingDouble((Integer ci) -> controllers.get(ci).bMagnitude()).reversed());
            groupControllerStart[g++] = k;
            for (int ci : group) {
                groupController[k++] = ci;
            }
        }
        groupControllerStart[g] = k;
        int[] bRow = new int[nControllers];
        int[] vRow = new int[nControllers];
        double[] targetV = new double[nControllers];
        double[] halfTargetDeadband = new double[nControllers];
        int[] controllerElement = new int[nControllers];
        int[] shuntAdmittanceOverrideElement = new int[nControllers];
        int[] shuntCompensatorStart = new int[nControllers + 1];
        List<Integer> sectionStart = new ArrayList<>();
        List<Double> sectionB = new ArrayList<>();
        List<Double> sectionG = new ArrayList<>();
        List<Integer> shuntCompensatorMinPosition = new ArrayList<>();
        List<Integer> shuntCompensatorBasePosition = new ArrayList<>();
        for (int c = 0; c < nControllers; c++) {
            Controller controller = controllers.get(c);
            bRow[c] = controller.bRow();
            vRow[c] = controller.vRow();
            targetV[c] = controller.targetV();
            halfTargetDeadband[c] = controller.halfDeadband();
            controllerElement[c] = controller.elemIndexAbs();
            shuntAdmittanceOverrideElement[c] = controller.elemIndexAbs() >= 0 ? controller.shuntAdmittanceOverrideElement() : -1;
            shuntCompensatorStart[c] = shuntCompensatorMinPosition.size();
            for (Sub sub : controller.subs()) {
                sectionStart.add(sectionB.size());
                for (int i = 0; i < sub.sectionsB().length; i++) {
                    sectionB.add(sub.sectionsB()[i]);
                    sectionG.add(sub.sectionsG()[i]);
                }
                shuntCompensatorMinPosition.add(sub.minPos());
                shuntCompensatorBasePosition.add(sub.basePos());
            }
        }
        shuntCompensatorStart[nControllers] = shuntCompensatorMinPosition.size();
        sectionStart.add(sectionB.size());
        boolean owned = ShuntVoltageControlNative.setBatchIncrementalShuntVoltageControl(handle,
                bRow, vRow, targetV, halfTargetDeadband, controllerElement, shuntAdmittanceOverrideElement, groupControllerStart, groupController, shuntCompensatorStart,
                sectionStart.stream().mapToInt(Integer::intValue).toArray(),
                sectionB.stream().mapToDouble(Double::doubleValue).toArray(),
                sectionG.stream().mapToDouble(Double::doubleValue).toArray(),
                shuntCompensatorMinPosition.stream().mapToInt(Integer::intValue).toArray(),
                shuntCompensatorBasePosition.stream().mapToInt(Integer::intValue).toArray(), maxSectionShift) != 0;
        LOGGER.info("Uploaded {} incremental shunt controllers to device ({} groups, {} subs): {}",
                nControllers, byBus.size(), shuntCompensatorMinPosition.size(), owned ? "in-kernel loop" : "host orchestrator");
        if (GpuTrace.BATCH_TRACE.isOn()) {
            System.err.printf("SHUNT_UPLOAD nControllers=%d nGroups=%d nSubs=%d kernel=%b%n", nControllers, byBus.size(), shuntCompensatorMinPosition.size(), owned);
        }
        return owned;
    }

    /** Give the driver's final sections (scenario 0) back to the network — what OLF's own loop leaves
     *  through LfShunt.Controller.updateSectionB; the shunt's b follows through OLF's listeners. */
    public static void applyToNetwork(LfNetwork network, List<Controller> controllers, int[] positions) {
        if (controllers == null || controllers.isEmpty() || positions == null) {
            return;
        }
        int u = 0;
        for (Controller controller : controllers) {
            LfShunt shunt = network.getShunt(controller.shuntNum());
            List<LfShunt.Controller> subs = shunt != null ? shunt.getControllers() : List.of();
            for (int k = 0; k < controller.subs().size(); k++, u++) {
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
