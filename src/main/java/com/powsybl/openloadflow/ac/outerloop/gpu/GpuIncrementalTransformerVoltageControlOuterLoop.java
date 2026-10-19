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
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchSkips;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuPiModelColumn;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DEVICE port of OLF's {@code IncrementalTransformerVoltageControlOuterLoop} (the DEFAULT transformer
 * voltage control mode), run as a per-scenario Java orchestrator around batched device solves:
 *
 * <ol>
 *   <li>after a converged batch pass, find the (scenario, controlled bus) pairs whose |targetV − V|
 *       exceeds the half deadband;</li>
 *   <li>batched SENSITIVITY solve on the device ({@link GpuAcNewtonSolver#solveBatchSensitivities}):
 *       J at the converged states, unit RHS at each out-of-deadband controller's pinned
 *       {@code BRANCH_TARGET_RHO1} identity row, dV/dρ read at the controlled bus's V column — the
 *       exact device analogue of the loop's {@code SensitivityContext} (OLF's {@code solveTransposed}
 *       is the forward direction in powsybl-math's transposed storage);</li>
 *   <li>OLF's tap decisions replicated verbatim per scenario: {@code deltaR1 = diffV / sensi},
 *       closest-tap search within {@code maxTapShift}, the {@code AllowedDirection} ratchet
 *       (MAX_DIRECTION_CHANGE = 3), the {@code MIN_SENSI_FILTER} (0.05) insensitivity cut, and the
 *       one-tap-at-a-time round-robin when several controllers hold one bus;</li>
 *   <li>tap moves land as per-scenario rho-row TARGET overrides (the identity row's pinned value),
 *       full 8-double pi-model row overrides (per-step R/X/G/B tables), and a rho state seed at the
 *       new tap (the CPU's PreviousValueVoltageInitializer re-reads r1 from the pi model);
 *       re-solve warm and iterate until every scenario is in deadband or nothing moves.</li>
 * </ol>
 *
 * <p>The insensitivity cut is re-evaluated every round instead of cached across consecutive
 * iterations (OLF caches it only until another outer loop runs — the device re-solve stabilizes
 * DS/RL every pass, which is the cache-reset condition, so fresh evaluation matches).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuIncrementalTransformerVoltageControlOuterLoop {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuIncrementalTransformerVoltageControlOuterLoop.class);

    private static final int MAX_DIRECTION_CHANGE = 3;       // OLF IncrementalTransformerVoltageControlOuterLoop
    private static final double MIN_SENSI_FILTER = 0.05;
    private static final double MIN_TARGET_DEADBAND_KV = 0.1;

    private GpuIncrementalTransformerVoltageControlOuterLoop() {
    }

    /** One incremental controller: the rho variable row (pinned identity — unit-RHS row, target index
     *  and state index all coincide in the device's paired layout), the controlled bus V row, targets,
     *  and the discrete tap table ([y g1 g2 b1 b2 ksi r1 a1] per tap). */
    public record Controller(int rhoRow, int vRow, int ctrlBusNum, int branchNum, double targetV, double halfDeadband,
                      double[][] tapRows, int baseTapIdx, int elemIndex) {
        double r1(int tap) {
            return tapRows[tap][GpuPiModelColumn.R1];
        }
    }

    /** Per-(scenario, controller) OLF ControllerContext ratchet state. */
    private static final class Ratchet {
        int dirChangeCount;
        int currentDirection;                                // 0 = none, +1 = increase, -1 = decrease
        int allowedDirection;                                // 0 = both, +1 = increase-only, -1 = decrease-only

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

    /** Collect the network's incremental transformer voltage controllers (INCREMENTAL mode keeps the
     *  branch VC DISABLED — rho stays a variable pinned by its BRANCH_TARGET_RHO1 identity row). */
    public static List<Controller> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                             GpuAcElementMapper elementMapper) {
        List<Controller> out = new ArrayList<>();
        Map<Integer, Integer> busVRows = new LinkedHashMap<>();
        Map<Integer, Integer> branchRhoRows = new LinkedHashMap<>();
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == AcVariableType.BUS_V) {
                busVRows.put(v.getElementNum(), v.getRow());
            } else if (v.getType() == AcVariableType.BRANCH_RHO1) {
                branchRhoRows.put(v.getElementNum(), v.getRow());
            }
        }
        // Grouped by the MAIN control (OLF's checkAndAdjustControlledBus iterates MAIN controlled
        // buses and adjusts their MERGED controller elements together): every member — including a
        // zero-impedance-merged DEPENDENT's — keys on the MAIN controlled bus and reads target,
        // deadband and V from it, so the orchestrator's byBus grouping reconstructs OLF's
        // one-tap-at-a-time round-robin sets exactly (elements stay in merged order — LfNetwork's
        // getControllerElements flat-maps the same lists in the same bus-index order).
        for (com.powsybl.openloadflow.network.LfBus bus
                : network.getControlledBuses(com.powsybl.openloadflow.network.VoltageControl.Type.TRANSFORMER)) {
            var vc = bus.getTransformerVoltageControl().orElseThrow();
            Integer vRow = busVRows.get(bus.getNum());
            double halfDeadband = vc.getTargetDeadband()
                    .orElse(MIN_TARGET_DEADBAND_KV / bus.getNominalV()) / 2;
            for (LfBranch branch : vc.getMergedControllerElements()) {
                if (branch.isDisabled() || !branch.isConnectedAtBothSides()) {
                    continue;
                }
                Integer rhoRow = branchRhoRows.get(branch.getNum());
                int elemIndex = elementMapper.getElementIndex(branch);
                double[][] tapRows = GpuTransformerVoltageControlOuterLoop.tapModels(branch.getPiModel());
                if (rhoRow == null || vRow == null || elemIndex < 0 || tapRows.length == 0) {
                    GpuBatchSkips.record(GpuBatchSkips.Slot.TVC, LOGGER,
                            "Incremental transformer controller {} not representable on the GPU batch — skipped", branch.getId());
                    continue;
                }
                double r1Now = branch.getPiModel().getR1();
                int baseTap = 0;
                for (int t = 1; t < tapRows.length; t++) {
                    if (Math.abs(tapRows[t][GpuPiModelColumn.R1] - r1Now) < Math.abs(tapRows[baseTap][GpuPiModelColumn.R1] - r1Now)) {
                        baseTap = t;
                    }
                }
                out.add(new Controller(rhoRow, vRow, bus.getNum(), branch.getNum(), vc.getTargetValue(), halfDeadband,
                        tapRows, baseTap, elemIndex));
            }
        }
        return out;
    }

    /** Install the controllers for the in-kernel loop, flattened into the CSR-packed arrays
     *  {@link TransformerVoltageControlNative#setBatchTvcControllers} expects. Returns true when the kernel loop OWNS them
     *  (shared controlled buses included), false when there are none. */
    public static boolean uploadToDevice(long handle, List<Controller> ctrls, int maxTapShift) {
        if (ctrls == null || ctrls.isEmpty()) {
            TransformerVoltageControlNative.setBatchTvcControllers(
                    handle, null, null, null, null, null, null, null, null, null, null, maxTapShift);   // clears a cached context
            return false;
        }
        int nc = ctrls.size();
        int[] rhoRow = new int[nc];
        int[] vRow = new int[nc];
        double[] targetV = new double[nc];
        double[] halfDb = new double[nc];
        int[] elemIndex = new int[nc];
        int[] baseTap = new int[nc];
        // CSR tap table: tapPtr[c] = offset into flat tapData for controller c
        int[] tapPtr = new int[nc + 1];
        int totalTaps = 0;
        for (int c = 0; c < nc; c++) {
            Controller ctl = ctrls.get(c);
            rhoRow[c] = ctl.rhoRow();
            vRow[c] = ctl.vRow();
            targetV[c] = ctl.targetV();
            halfDb[c] = ctl.halfDeadband();
            elemIndex[c] = ctl.elemIndex();
            baseTap[c] = ctl.baseTapIdx();
            tapPtr[c] = totalTaps;
            totalTaps += ctl.tapRows().length;
        }
        tapPtr[nc] = totalTaps;
        // Flat tap data: totalTaps pi model rows (GpuPiModelColumn), row-major
        double[] tapData = new double[totalTaps * GpuPiModelColumn.COUNT];
        int off = 0;
        for (Controller ctl : ctrls) {
            for (double[] row : ctl.tapRows()) {
                System.arraycopy(row, 0, tapData, off, GpuPiModelColumn.COUNT);
                off += GpuPiModelColumn.COUNT;
            }
        }
        // Group structure: controllers sharing a controlled bus (keyed by ctrlBusNum)
        java.util.Map<Integer, java.util.List<Integer>> byBus = new java.util.LinkedHashMap<>();
        for (int c = 0; c < nc; c++) {
            byBus.computeIfAbsent(ctrls.get(c).ctrlBusNum(), k -> new java.util.ArrayList<>()).add(c);
        }
        int nGroups = byBus.size();
        int[] groupPtr = new int[nGroups + 1];
        int[] groupCtrl = new int[nc];
        int gi = 0;
        int ci = 0;
        for (var group : byBus.values()) {
            groupPtr[gi] = ci;
            for (int idx : group) {
                groupCtrl[ci++] = idx;
            }
            gi++;
        }
        groupPtr[nGroups] = ci;
        boolean owned = TransformerVoltageControlNative.setBatchTvcControllers(
                handle, rhoRow, vRow, targetV, halfDb, elemIndex, tapPtr, tapData, baseTap,
                groupPtr, groupCtrl, maxTapShift) != 0;
        LOGGER.info("Uploaded {} incremental TVC controllers to device ({} taps, {} groups): {}",
                nc, totalTaps, nGroups, owned ? "in-kernel loop" : "host orchestrator");
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.printf("TVC_UPLOAD nCtrl=%d totalTaps=%d nGroups=%d kernel=%b%n", nc, totalTaps, nGroups, owned);
        }
        return owned;
    }

    /** Give the driver's final taps (scenario 0) back to the network — the tap position OLF's own loop
     *  would have left through updateTapPositionToReachNewR1; the pi model's listeners refresh r1. */
    public static void applyToNetwork(LfNetwork network, List<Controller> ctrls, int[] taps) {
        if (ctrls == null || ctrls.isEmpty() || taps == null) {
            return;
        }
        for (int c = 0; c < ctrls.size() && c < taps.length; c++) {
            Controller ctl = ctrls.get(c);
            if (taps[c] == ctl.baseTapIdx()) {
                continue;
            }
            LfBranch branch = network.getBranch(ctl.branchNum());
            if (branch == null) {
                continue;
            }
            var pm = branch.getPiModel();
            pm.setTapPosition(pm.getTapPositionRange().getMinimum() + taps[c]);
        }
    }
}
