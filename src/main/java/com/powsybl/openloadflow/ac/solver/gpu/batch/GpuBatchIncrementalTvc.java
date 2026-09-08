/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
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
final class GpuBatchIncrementalTvc {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuBatchIncrementalTvc.class);

    private static final int MAX_DIRECTION_CHANGE = 3;       // OLF IncrementalTransformerVoltageControlOuterLoop
    private static final double MIN_SENSI_FILTER = 0.05;
    private static final double MIN_TARGET_DEADBAND_KV = 0.1;

    private GpuBatchIncrementalTvc() {
    }

    /** One incremental controller: the rho variable row (pinned identity — unit-RHS row, target index
     *  and state index all coincide in the device's paired layout), the controlled bus V row, targets,
     *  and the discrete tap table ([y g1 g2 b1 b2 ksi r1 a1] per tap). */
    record Controller(int rhoRow, int vRow, int ctrlBusNum, double targetV, double halfDeadband,
                      double[][] tapRows, int baseTapIdx, int elemIndex) {
        double r1(int tap) {
            return tapRows[tap][6];
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
    static List<Controller> buildControllers(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                             GpuBatchElementMapper elementMapper) {
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
                double[][] tapRows = GpuBatchTransformerControl.tapModels(branch.getPiModel());
                if (rhoRow == null || vRow == null || elemIndex < 0 || tapRows.length == 0) {
                    LOGGER.info("Incremental transformer controller {} not representable on the GPU batch — skipped", branch.getId());
                    continue;
                }
                double r1Now = branch.getPiModel().getR1();
                int baseTap = 0;
                for (int t = 1; t < tapRows.length; t++) {
                    if (Math.abs(tapRows[t][6] - r1Now) < Math.abs(tapRows[baseTap][6] - r1Now)) {
                        baseTap = t;
                    }
                }
                out.add(new Controller(rhoRow, vRow, bus.getNum(), vc.getTargetValue(), halfDeadband,
                        tapRows, baseTap, elemIndex));
            }
        }
        return out;
    }

    /** OLF PiModelArray.updateTapPositionToReachNewR1: closest tap to r1(cur)+deltaR1 within
     *  maxTapShift and the ratchet's allowed range. Returns 0 (no move) or the direction. */
    private static int moveTap(Controller c, int[] tap, double deltaR1, int maxTapShift, Ratchet ratchet) {
        int cur = tap[0];
        double newR1 = c.r1(cur) + deltaR1;
        int lo = ratchet.allowedDirection > 0 ? cur : 0;
        int hi = ratchet.allowedDirection < 0 ? cur : c.tapRows().length - 1;
        int best = cur;
        double bestDist = Math.abs(newR1 - c.r1(cur));
        for (int i = lo; i <= hi; i++) {
            if (Math.abs(i - cur) > maxTapShift) {
                continue;
            }
            double dist = Math.abs(newR1 - c.r1(i));
            if (dist < bestDist) {
                best = i;
                bestDist = dist;
            }
        }
        if (best == cur) {
            return 0;
        }
        tap[0] = best;
        int direction = best > cur ? 1 : -1;
        ratchet.update(direction);
        return direction;
    }

    /** OLF's per-(scenario, controlled-bus) tap decisions for one round: adjustWithOneController
     *  (full diffV, maxTapShift) when a single live controller holds the bus, else the one-tap-at-a-
     *  time round-robin over the merged controllers (adjustWithSeveralControllers: remaining diffV
     *  updated by each move's r1-delta x sensitivity, sweeps repeat until nothing moves). */
    @SuppressWarnings("java:S107")
    private static boolean adjustPairs(List<int[]> outOfDb, Map<Integer, List<Integer>> byBus, List<Integer> busKeys,
            List<Controller> ctrls, double[][] chunkResults, boolean[][] disabledMask,
            double[] sensi, int[] unitOf, int ku, int m, int[][] tap, Ratchet[][] ratchets,
            int maxTapShift, boolean[] affectedSlots) {
        boolean anyMoved = false;
        for (int[] sg : outOfDb) {
            int slot = sg[0];
            int g = sg[1];
            List<Integer> group = byBus.get(busKeys.get(g));
            Controller c0 = ctrls.get(group.get(0));
            double diffV = c0.targetV() - chunkResults[slot][c0.vRow()];
            List<Integer> live = group.stream().filter(ci -> !disabledMask[slot][ctrls.get(ci).elemIndex()]).toList();
            java.util.function.IntToDoubleFunction sensiOf = ci -> sensi[slot * ku * m + unitOf[ci] * m + g];
            if (System.getenv("OLF_GPU_INC_TRACE") != null) {
                StringBuilder sb = new StringBuilder("INC_TRACE slot=" + slot + " g=" + g + " diffV=" + diffV + " live=" + live);
                for (int ci : live) {
                    sb.append(String.format(" s[%d]=%.6f tap=%d", ci, sensiOf.applyAsDouble(ci), tap[slot][ci]));
                }
                System.err.println(sb);
            }
            if (live.size() == 1) {
                int ci = live.get(0);
                double sv = sensiOf.applyAsDouble(ci);
                if (Math.abs(sv) < MIN_SENSI_FILTER) {
                    continue;                                // insensitive — no control performed (OLF filter)
                }
                int[] holder = {tap[slot][ci]};
                if (moveTap(ctrls.get(ci), holder, diffV / sv, maxTapShift, ratchets[slot][ci]) != 0) {
                    tap[slot][ci] = holder[0];
                    anyMoved = true;
                    affectedSlots[slot] = true;
                }
            } else if (!live.isEmpty()) {
                // several controllers: one tap at a time round-robin (OLF adjustWithSeveralControllers)
                double remaining = diffV;
                boolean changed = true;
                while (changed) {
                    changed = false;
                    for (int ci : live) {
                        if (Math.abs(remaining) > c0.halfDeadband()) {
                            double sv = sensiOf.applyAsDouble(ci);
                            if (Math.abs(sv) < MIN_SENSI_FILTER) {
                                continue;
                            }
                            double prevR1 = ctrls.get(ci).r1(tap[slot][ci]);
                            int[] holder = {tap[slot][ci]};
                            if (moveTap(ctrls.get(ci), holder, remaining / sv, 1, ratchets[slot][ci]) != 0) {
                                tap[slot][ci] = holder[0];
                                remaining -= (ctrls.get(ci).r1(tap[slot][ci]) - prevR1) * sv;
                                if (System.getenv("OLF_GPU_INC_TRACE") != null) {
                                    System.err.println("INC_TRACE   move slot=" + slot + " ctrl=" + ci + " ->tap=" + tap[slot][ci] + " remaining=" + remaining);
                                }
                                changed = true;
                                anyMoved = true;
                                affectedSlots[slot] = true;
                            }
                        }
                    }
                }
            }
        }
        return anyMoved;
    }

    @SuppressWarnings({"java:S107", "java:S3776"})
    static double[][] orchestrate(long handle, int chunkSize, int realCount, int n,
            double[][] chunkResults, List<Controller> ctrls,
            int[][] rowModePerScenario, int[] baseRowMode, boolean[][] disabledMask,
            int[] targetOverrideIndex, double[] targetOverrideValue,
            int[] dsOvBus, double[] dsOvFactorDelta, double[] dsOvBaseDelta, double[] dsOvMaxDelta, double[] dsOvMinDelta,
            int[] dsPhiRow, double[] dsFactor,
            int[] baseR1OvSlotElem, double[] baseR1OvVal, int[] shOvSlotElem, double[] shOvGb,
            int nziBase, int[] ziOnSlotElem, int[] ziOffSlotElem,
            double[] baseState, double[] baseTargets, NewtonRaphsonParameters parameters,
            double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, int[] varType,
            BatchConfig config) {
        int nc = ctrls.size();
        int[][] tap = new int[chunkSize][nc];                // per-scenario tap index
        Ratchet[][] ratchets = new Ratchet[chunkSize][nc];
        for (int s = 0; s < chunkSize; s++) {
            for (int c = 0; c < nc; c++) {
                tap[s][c] = ctrls.get(c).baseTapIdx();
                ratchets[s][c] = new Ratchet();
            }
        }
        // controllers grouped by controlled bus (round-robin when several hold one bus)
        Map<Integer, List<Integer>> byBus = new LinkedHashMap<>();
        for (int c = 0; c < nc; c++) {
            byBus.computeIfAbsent(ctrls.get(c).ctrlBusNum(), k -> new ArrayList<>()).add(c);
        }
        int maxTapShift = config.incTvcMaxTapShift();
        int maxRounds = config.maxOuterIterations() > 0 ? config.maxOuterIterations() : 20;
        // DS target carry (the same three coupled pieces as GpuBatchTransformerControl): every
        // re-solve re-uploads baseTargets, but the CPU's targets are STATEFUL — the distribution
        // applied by the MAIN chunk solve (and by previous rounds) must ride along, or the re-solve
        // re-derives it from scratch along a trajectory the CPU never takes (surfaced by the
        // shared-group member outage: the only scenario distributing slack drifted 3.9e-3 kV).
        int nds = dsPhiRow != null ? dsPhiRow.length : 0;
        double[] dsAnchorCarry = nds > 0 ? new double[chunkSize * nds] : null;
        double[] dsSlackCarry = nds > 0 ? new double[chunkSize] : null;
        double[] dsSeed = nds > 0 ? new double[chunkSize * nds] : null;
        if (nds > 0) {
            GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                    dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed);
        }

        for (int round = 0; round < maxRounds; round++) {
            // ---- which (scenario, bus) pairs are out of deadband on the current converged states ----
            List<int[]> outOfDb = new ArrayList<>();         // {slot, busGroupKey-index into byBus order}
            List<Integer> busKeys = new ArrayList<>(byBus.keySet());
            boolean[] ctrlNeeded = new boolean[nc];
            for (int slot = 0; slot < realCount; slot++) {
                double[] r = chunkResults[slot];
                if (r == null || (int) r[n + 2] != GpuBatchStatus.STATUS_CONVERGED) {
                    continue;
                }
                final int slotL = slot;
                for (int g = 0; g < busKeys.size(); g++) {
                    List<Integer> group = byBus.get(busKeys.get(g));
                    Controller c0 = ctrls.get(group.get(0));
                    boolean anyLive = group.stream().anyMatch(ci -> !disabledMask[slotL][ctrls.get(ci).elemIndex()]);
                    if (anyLive && Math.abs(c0.targetV() - r[c0.vRow()]) > c0.halfDeadband()) {
                        outOfDb.add(new int[] {slot, g});
                        for (int ci : group) {
                            ctrlNeeded[ci] = true;
                        }
                    }
                }
            }
            if (outOfDb.isEmpty()) {
                break;
            }
            // ---- batched device sensitivities dV/drho for the needed controllers ----
            int[] unit = new int[nc];
            int[] obs = new int[busKeys.size()];
            int ku = 0;
            int[] unitOf = new int[nc];
            for (int c = 0; c < nc; c++) {
                if (ctrlNeeded[c]) {
                    unitOf[c] = ku;
                    unit[ku++] = ctrls.get(c).rhoRow();
                } else {
                    unitOf[c] = -1;
                }
            }
            for (int g = 0; g < busKeys.size(); g++) {
                obs[g] = ctrls.get(byBus.get(busKeys.get(g)).get(0)).vRow();
            }
            double[] statesFlat = new double[chunkSize * n];
            for (int s = 0; s < chunkSize; s++) {
                double[] src = s < realCount && chunkResults[s] != null ? chunkResults[s] : baseState;
                System.arraycopy(src, 0, statesFlat, s * n, n);
            }
            boolean[] affectedSlots = new boolean[chunkSize];
            int[] unitTrim = java.util.Arrays.copyOf(unit, ku);
            double[] sensi = GpuAcNewtonSolver.solveBatchSensitivities(handle, statesFlat, unitTrim, obs);
            int m = obs.length;

            boolean anyMoved = adjustPairs(outOfDb, byBus, busKeys, ctrls, chunkResults, disabledMask,
                    sensi, unitOf, ku, m, tap, ratchets, maxTapShift, affectedSlots);
            if (!anyMoved) {
                break;
            }
            // ---- apply the moved taps: rho targets + pi rows + rho state seeds, then re-solve warm ----
            List<Integer> ovIdx = new ArrayList<>();
            List<Double> ovVal = new ArrayList<>();
            for (int i = 0; i < targetOverrideIndex.length; i++) {
                ovIdx.add(targetOverrideIndex[i]);
                ovVal.add(targetOverrideValue[i]);
            }
            GpuBatchTransformerControl.seedDsTargets(nds, dsPhiRow, dsSeed, chunkResults, realCount, n, ovIdx, ovVal);
            List<Integer> r1Slot = new ArrayList<>();
            List<Double> r1Val = new ArrayList<>();
            for (int i = 0; i < baseR1OvSlotElem.length; i += 2) {
                r1Slot.add(baseR1OvSlotElem[i]);
                r1Slot.add(baseR1OvSlotElem[i + 1]);
            }
            for (double v : baseR1OvVal) {
                r1Val.add(v);
            }
            double[] x0Full = statesFlat;                     // warm from the converged states
            for (int s = 0; s < chunkSize; s++) {
                for (int c = 0; c < nc; c++) {
                    if (tap[s][c] != ctrls.get(c).baseTapIdx()) {
                        Controller ctl = ctrls.get(c);
                        double[] row = ctl.tapRows()[tap[s][c]];
                        ovIdx.add(s * n + ctl.rhoRow());
                        ovVal.add(row[6]);                    // the identity row pins rho at the tap's r1
                        r1Slot.add(s);
                        r1Slot.add(ctl.elemIndex());
                        for (double d : row) {
                            r1Val.add(d);                     // full 8-double pi row: per-step R/X/G/B follow the tap
                        }
                        x0Full[s * n + ctl.rhoRow()] = row[6];  // CPU re-reads r1 from the pi model on re-solve
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
                    r1Slot.stream().mapToInt(Integer::intValue).toArray(),
                    r1Val.stream().mapToDouble(Double::doubleValue).toArray(),
                    config.maxOuterIterations(), config.maxPqPvSwitch(), config.dsRejectResidualFlag(), 0,
                    nziBase, ziOnSlotElem, ziOffSlotElem);
            for (int s = 0; s < realCount; s++) {
                if (affectedSlots[s]) {                      // untouched scenarios KEEP their converged state
                    chunkResults[s] = r2[s];                 // (no extra Newton polish the CPU never runs)
                }
            }
            if (nds > 0) {                                   // this solve may have distributed further
                GpuBatchTransformerControl.captureDsCarry(handle, chunkSize, realCount, n, dsPhiRow, dsFactor, baseTargets,
                        dsOvBus, dsOvFactorDelta, dsOvBaseDelta, chunkResults, dsAnchorCarry, dsSlackCarry, dsSeed);
            }
        }
        return chunkResults;
    }
}
