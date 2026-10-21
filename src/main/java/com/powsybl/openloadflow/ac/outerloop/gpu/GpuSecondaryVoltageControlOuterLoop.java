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
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveTermKind;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuTwoSides;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfSecondaryVoltageControl;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of OLF's {@code SecondaryVoltageControlOuterLoop} (pilot-point zone regulation): the tables of the
 * in-kernel phase (native/cuda/outerloop/secondary_voltage_control.cu), which runs the whole check at OLF's position in
 * the sweep — tryToReEnableHelpfulControllerBuses, the per-zone trigger, the sensitivities on the driver's own factors
 * (unit right-hand side at each controlled bus's BUS_TARGET_V row), the dense zone system and the moved targets.
 *
 * <p>Per-scenario mechanics: a moved generator voltage control target lands on the CONTROLLED bus's V row; in a run with
 * OLF's ReactiveLimitsOuterLoop a controller is keyed by its reactive-limit entry, whose pin the re-enable step clears
 * and whose switch-back compares against the MOVED target. Scope: LOCAL and REMOTE generator voltage control with
 * DISTINCT controlled buses (a controller<->controlled bijection — see {@link #supported}); a remote controller's
 * reactive-limit entry is the reactiveLimitVoltageControlRole member entry.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuSecondaryVoltageControlOuterLoop {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuSecondaryVoltageControlOuterLoop.class);

    /** Controlled-bus target voltages this JVM has given back to a network from the device phase (OLF's "Adjust target
     *  voltage of controlled bus"). Test-visible: a run whose loop OLF ran instead leaves the very same voltages, so only
     *  this tells a test that the device drove the zone. */
    public static final java.util.concurrent.atomic.LongAdder ADJUSTED_CONTROLLED_BUS_TARGETS = new java.util.concurrent.atomic.LongAdder();

    private GpuSecondaryVoltageControlOuterLoop() {
    }

    /** One Q observable of a controller bus, as the DEVICE sees it: {@code kind} a {@link GpuReactiveTermKind},
     *  CLOSED_BRANCH_REACTIVE_FLOW = the closed branch's reactive flow on {@code side} (a {@link GpuTwoSides}), whose
     *  gradient comes from the codegen'd updateClosedBranch (the very function the fill scatters);
     *  SHUNT_COMPENSATOR_REACTIVE_FLOW = a SHUNT's q = -b*v^2, i.e. OLF's ShuntCompensatorReactiveFlowEquationTerm
     *  (dq/dv = -2bv, dq/db = -v^2), side NONE. {@code elem} is the PACK-relative element index; {@code bRow} a
     *  derived shunt's SHUNT_B row, -1 when b is fixed. */
    public record ReactiveTerm(int kind, int side, int elem, int bRow) {
    }

    /** One (controller, controlled) PAIR — a bijection under {@link #supported}: {@code vRow} is
     *  the CONTROLLER bus's V row (keys its RL entry — for a REMOTE controller that is the
     *  reactiveLimitVoltageControlRole member entry; -1 in a run without OLF's ReactiveLimitsOuterLoop, where no controller
     *  is ever pinned), {@code controlledBusVRow} the CONTROLLED bus's pinned BUS_TARGET_V row (the
     *  sensitivity unit row and the row the moved zone target lands on — OLF adjusts the
     *  CONTROLLED bus's VC target). Local control collapses the two. */
    public record ControllerBus(int vRow, int controlledBusVRow, int controlledBusNum, int reactiveLimitEntry, double baseTargetV, double minQ, double maxQ,
                  double loadTargetQ, List<ReactiveTerm> reactiveTerms) {
    }

    public record Zone(int pilotVRow, double pilotTargetV, int[] controllerBuses) {
    }

    public record Controllers(List<ControllerBus> controllerBuses, List<Zone> zones,
                       double minPlausibleTargetV, double maxPlausibleTargetV) {
    }

    /** True if every enabled secondary-voltage-control zone is representable: LOCAL or REMOTE
     *  generator VC with DISTINCT controlled buses (controllers <-> controlled is a BIJECTION —
     *  OLF's own zone system requires it: b = A*Jk^T is nControllers x nControlled and its LU
     *  needs a square matrix, so a shared controlled bus is outside OLF's effective scope too),
     *  and every REMOTE controller carries a real Q range (its K divides by it, and the reactive-limit
     *  pack gives a remote controller an entry only then). */
    public static boolean supported(LfNetwork network) {
        java.util.Set<LfBus> controlled = new java.util.HashSet<>();
        for (LfSecondaryVoltageControl secondaryVoltageControl : network.getEnabledSecondaryVoltageControls()) {
            for (LfBus controller : secondaryVoltageControl.getControllerBuses()) {
                var voltageControl = controller.getGeneratorVoltageControl().orElse(null);
                if (voltageControl == null) {
                    return false;
                }
                if (!controlled.add(voltageControl.getControlledBus())) {
                    return false;                            // shared controlled bus — not bijective
                }
                if (voltageControl.getControlledBus() != controller && !(controller.getMinQ() < controller.getMaxQ())) {
                    return false;                            // remote controller without an RL entry
                }
            }
        }
        return true;
    }

    public static Controllers build(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                             int[] reactiveLimitVoltageRow, boolean reactiveLimitsOuterLoop,
                             double minPlausibleTargetV, double maxPlausibleTargetV,
                             GpuAcElementMapper elementMapper) {
        if (!supported(network)) {
            GpuBatchSkips.record(GpuBatchSkips.Slot.SVC, LOGGER,
                    "Secondary VC zones not representable on the GPU batch (a shared controlled bus, or a remote controller without a reactive range)");
            return new Controllers(List.of(), List.of(), minPlausibleTargetV, maxPlausibleTargetV);
        }
        Map<Integer, Integer> busVRows = new LinkedHashMap<>();
        Map<Integer, Integer> shuntBRows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVRows.put(v.getElementNum(), v.getRow());
            } else if (v.getType() == AcVariableType.SHUNT_B) {
                shuntBRows.put(v.getElementNum(), v.getRow());
            }
        }
        Map<Integer, Integer> reactiveLimitEntryByVoltageRow = new LinkedHashMap<>();
        for (int i = 0; i < reactiveLimitVoltageRow.length; i++) {
            reactiveLimitEntryByVoltageRow.put(reactiveLimitVoltageRow[i], i);
        }
        List<ControllerBus> controllerBuses = new ArrayList<>();
        List<Zone> zones = new ArrayList<>();
        for (LfSecondaryVoltageControl secondaryVoltageControl : network.getEnabledSecondaryVoltageControls()) {
            List<Integer> zoneControllerBuses = new ArrayList<>();
            for (LfBus controller : secondaryVoltageControl.getControllerBuses()) {
                var voltageControl = controller.getGeneratorVoltageControl().orElseThrow();
                Integer vRow = busVRows.get(controller.getNum());
                Integer controlledBusVRow = busVRows.get(voltageControl.getControlledBus().getNum());
                // a controller is keyed by its reactive-limit entry, whose pin state the phase reads, only in a run with
                // OLF's ReactiveLimitsOuterLoop: without it OLF never switches a controller PV -> PQ, so every controller
                // keeps its voltage control and the phase moves its target on the BUS_TARGET_V row alone (-1: no entry)
                Integer reactiveLimitEntry = -1;
                if (reactiveLimitsOuterLoop) {
                    reactiveLimitEntry = vRow != null ? reactiveLimitEntryByVoltageRow.get(vRow) : null;
                }
                if (vRow == null || controlledBusVRow == null || reactiveLimitEntry == null) {
                    GpuBatchSkips.record(GpuBatchSkips.Slot.SVC, LOGGER,
                            "Secondary VC controller {} not representable on the GPU batch — skipped zone {}",
                            controller.getId(), secondaryVoltageControl.getZoneName());
                    return new Controllers(List.of(), List.of(), minPlausibleTargetV, maxPlausibleTargetV);
                }
                // OLF calculateSensiQ: the controller bus's branch Q flows (side by attachment) + shunts, addressed the
                // way the DEVICE reaches them — pack-relative element indices, so the in-kernel phase evaluates the
                // identical quantities without re-deriving them.
                List<ReactiveTerm> reactiveTerms = new ArrayList<>();
                for (LfBranch branch : controller.getBranches()) {
                    if (!branch.isDisabled() && branch.isConnectedAtBothSides()) {
                        int elem = elementMapper.getElementIndex(branch);
                        if (branch.getBus1() == controller) {
                            reactiveTerms.add(new ReactiveTerm(GpuReactiveTermKind.CLOSED_BRANCH_REACTIVE_FLOW, GpuTwoSides.ONE, elem, -1));
                        }
                        if (branch.getBus2() == controller) {
                            reactiveTerms.add(new ReactiveTerm(GpuReactiveTermKind.CLOSED_BRANCH_REACTIVE_FLOW, GpuTwoSides.TWO, elem, -1));
                        }
                    }
                }
                for (var optionalShunt : List.of(controller.getShunt(), controller.getControllerShunt(), controller.getSvcShunt())) {
                    if (optionalShunt.isPresent()) {
                        LfShunt shunt = optionalShunt.get();
                        Integer bRow = shuntBRows.get(shunt.getNum());
                        reactiveTerms.add(new ReactiveTerm(GpuReactiveTermKind.SHUNT_COMPENSATOR_REACTIVE_FLOW, GpuTwoSides.NONE,
                                elementMapper.getElementIndex(shunt), bRow != null ? bRow : -1));
                    }
                }
                zoneControllerBuses.add(controllerBuses.size());
                controllerBuses.add(new ControllerBus(vRow, controlledBusVRow, voltageControl.getControlledBus().getNum(), reactiveLimitEntry, voltageControl.getTargetValue(),
                        controller.getMinQ(), controller.getMaxQ(),
                        controller.getLoadTargetQ(), reactiveTerms));
            }
            if (zoneControllerBuses.isEmpty()) {
                continue;                                    // OLF leaves a zone without controller buses alone
            }
            Integer pilotVRow = busVRows.get(secondaryVoltageControl.getPilotBus().getNum());
            if (pilotVRow == null) {
                GpuBatchSkips.record(GpuBatchSkips.Slot.SVC, LOGGER, "Secondary VC pilot bus {} not representable on the GPU batch — skipped zone {}",
                        secondaryVoltageControl.getPilotBus().getId(), secondaryVoltageControl.getZoneName());
                return new Controllers(List.of(), List.of(), minPlausibleTargetV, maxPlausibleTargetV);
            }
            zones.add(new Zone(pilotVRow, secondaryVoltageControl.getTargetValue(),
                    zoneControllerBuses.stream().mapToInt(Integer::intValue).toArray()));
        }
        return new Controllers(controllerBuses, zones, minPlausibleTargetV, maxPlausibleTargetV);
    }

    /**
     * Give the in-kernel zone regulation back to the network — the batch-of-one's analogue of what
     * OLF's {@code SecondaryVoltageControlOuterLoop} does as a side effect of running: it moves each
     * controller's generator voltage control TARGET, and without this OLF's world keeps the
     * pre-regulation targets while the device solved at the moved ones.
     *
     * <p>The moved target comes from the phase's OWN state, not from the controlled bus's row. Reading
     * the row looks right and is not: that row carries a Q target the moment the bus is pinned or is a
     * shared-distribution member, and writing 0.165 pu in as a target voltage puts the control outside
     * the plausible band and OLF drops it.
     */
    public static void applyToNetwork(LfNetwork network, Controllers controllers, double[] movedTargetV) {
        if (controllers == null || movedTargetV == null) {
            return;
        }
        List<ControllerBus> controllerBuses = controllers.controllerBuses();
        for (int c = 0; c < controllerBuses.size() && c < movedTargetV.length; c++) {
            double moved = movedTargetV[c];                  // batch of one: scenario 0
            if (Double.isNaN(moved) || moved <= 0) {
                continue;                                    // the phase never adjusted this controller
            }
            LfBus controlledBus = network.getBus(controllerBuses.get(c).controlledBusNum());
            if (controlledBus == null) {
                continue;
            }
            final double v = moved;
            if (moved != controllerBuses.get(c).baseTargetV()) {
                ADJUSTED_CONTROLLED_BUS_TARGETS.increment();
            }
            controlledBus.getGeneratorVoltageControl().ifPresent(voltageControl -> voltageControl.setTargetValue(v));
        }
    }

    @SuppressWarnings({"java:S107", "java:S3776"})
    /** Install the zones for the IN-KERNEL SecondaryVoltageControl phase (OLF order 4). Returns true when
     *  the kernel owns the loop: the whole check — the re-enable step, the per-zone trigger, the
     *  sensitivities on the driver's own factors, the dense zone system and the moved targets — then runs
     *  inside the sweep at OLF's position, with no host round trip and no orchestrator after it. */
    public static boolean uploadToDevice(long handle, Controllers controllers) {
        if (controllers == null || controllers.controllerBuses().isEmpty() || controllers.zones().isEmpty()) {
            SecondaryVoltageControlNative.setBatchSecondaryVoltageControl(handle, null, null, null, null, null, null, null,
                    null, null, null, null, null, null, null, null, null, 0, 0);
            return false;
        }
        List<ControllerBus> controllerBuses = controllers.controllerBuses();
        int nControllerBuses = controllerBuses.size();
        int[] controlledBusVRow = new int[nControllerBuses];
        int[] busVRow = new int[nControllerBuses];
        int[] reactiveLimitEntry = new int[nControllerBuses];
        double[] baseTargetV = new double[nControllerBuses];
        double[] minQ = new double[nControllerBuses];
        double[] maxQ = new double[nControllerBuses];
        double[] loadQ = new double[nControllerBuses];
        int[] qPtr = new int[nControllerBuses + 1];
        List<Integer> reactiveTermKind = new ArrayList<>();
        List<Integer> reactiveTermSide = new ArrayList<>();
        List<Integer> reactiveTermElement = new ArrayList<>();
        List<Integer> reactiveTermShuntBRow = new ArrayList<>();
        for (int c = 0; c < nControllerBuses; c++) {
            ControllerBus controllerBus = controllerBuses.get(c);
            controlledBusVRow[c] = controllerBus.controlledBusVRow();
            busVRow[c] = controllerBus.vRow();
            reactiveLimitEntry[c] = controllerBus.reactiveLimitEntry();
            baseTargetV[c] = controllerBus.baseTargetV();
            minQ[c] = controllerBus.minQ();
            maxQ[c] = controllerBus.maxQ();
            loadQ[c] = controllerBus.loadTargetQ();
            qPtr[c] = reactiveTermKind.size();
            for (ReactiveTerm o : controllerBus.reactiveTerms()) {
                reactiveTermKind.add(o.kind());
                reactiveTermSide.add(o.side());
                reactiveTermElement.add(o.elem());
                reactiveTermShuntBRow.add(o.bRow());
            }
        }
        qPtr[nControllerBuses] = reactiveTermKind.size();
        int nZones = controllers.zones().size();
        int[] zoneStart = new int[nZones + 1];
        List<Integer> zoneControllerBus = new ArrayList<>();
        int[] zonePilotBusVRow = new int[nZones];
        double[] zoneTargetV = new double[nZones];
        for (int z = 0; z < nZones; z++) {
            Zone zone = controllers.zones().get(z);
            zoneStart[z] = zoneControllerBus.size();
            for (int c : zone.controllerBuses()) {
                zoneControllerBus.add(c);
            }
            zonePilotBusVRow[z] = zone.pilotVRow();
            zoneTargetV[z] = zone.pilotTargetV();
        }
        zoneStart[nZones] = zoneControllerBus.size();
        boolean owned = SecondaryVoltageControlNative.setBatchSecondaryVoltageControl(handle, controlledBusVRow, busVRow, reactiveLimitEntry, baseTargetV,
                minQ, maxQ, loadQ, qPtr,
                reactiveTermKind.stream().mapToInt(Integer::intValue).toArray(),
                reactiveTermSide.stream().mapToInt(Integer::intValue).toArray(),
                reactiveTermElement.stream().mapToInt(Integer::intValue).toArray(),
                reactiveTermShuntBRow.stream().mapToInt(Integer::intValue).toArray(),
                zoneStart, zoneControllerBus.stream().mapToInt(Integer::intValue).toArray(), zonePilotBusVRow, zoneTargetV,
                controllers.minPlausibleTargetV(), controllers.maxPlausibleTargetV()) != 0;
        LOGGER.info("Uploaded {} secondary voltage-control zones ({} controllers) to device: {}",
                nZones, nControllerBuses, owned ? "in-kernel loop" : "refused");
        if (GpuTrace.BATCH_TRACE.isOn()) {
            System.err.printf("SVC_UPLOAD nZones=%d nControllerBuses=%d kernel=%b%n", nZones, nControllerBuses, owned);
        }
        return owned;
    }
}
