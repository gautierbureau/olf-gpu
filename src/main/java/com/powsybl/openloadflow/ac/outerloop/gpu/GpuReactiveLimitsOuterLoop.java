/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuQLimitType;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuReactiveLimitsState;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRowMode;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuVoltageControlRole;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;

/**
 * DEVICE port of OLF's {@code ReactiveLimitsOuterLoop}, Java side: the loop runs in the kernel (reactive_limits.cu);
 * this class applies its pins to the network, the way OLF's loop leaves a bus: generation Q frozen at the limit and
 * voltage control disabled, or a released pin undone.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuReactiveLimitsOuterLoop {

    private GpuReactiveLimitsOuterLoop() {
    }

    /** ReactiveLimits: the device pins a controller at its limit (row mode V→Q, target the limit) where
     *  OLF's loop freezes the bus's generation Q at that limit and disables its voltage control — the
     *  same two calls, so OLF's equation system swaps BUS_TARGET_V for BUS_TARGET_Q on its own and the
     *  target vector picks the limit up. A pin the device released (a PQ→PV switch-back) is undone the
     *  way OLF undoes it. Limit types 3/4 are robust mode's MIN/MAX_REALISTIC_V freezes: the pin is the
     *  same, only the recorded type differs (and OLF never re-freezes those at a moved Q limit). */
    public static void applyToNetwork(LfNetwork network, int[] reactiveLimitsFinalState, GpuAcData data, double[] deviceTargets, int[] devModes) {
        if (reactiveLimitsFinalState == null || data == null || data.reactiveLimits().busNum() == null) {
            return;
        }
        int nReactiveLimitEntries = Math.min(reactiveLimitsFinalState.length, data.reactiveLimits().busNum().length);
        for (int i = 0; i < nReactiveLimitEntries; i++) {
            int limitType = GpuReactiveLimitsState.limitType(reactiveLimitsFinalState[i]);
            LfBus bus = network.getBus(data.reactiveLimits().busNum()[i]);
            if (bus == null) {
                continue;
            }
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null && (limitType != GpuQLimitType.NONE || GpuReactiveLimitsState.pvPqSwitchCount(reactiveLimitsFinalState[i]) != 0)) {
                System.err.println("WB_RL bus=" + bus.getId() + " limitType=" + limitType + " switchCount=" + GpuReactiveLimitsState.pvPqSwitchCount(reactiveLimitsFinalState[i])
                        + " vcEnabled=" + bus.isGeneratorVoltageControlEnabled() + " qLimitType=" + bus.getQLimitType().map(Object::toString).orElse("-")
                        + " remote=" + data.reactiveLimits().voltageControlRole()[i]);
            }
            if (limitType == GpuQLimitType.FROZEN) {
                // Frozen by the generator voltage control manager, not pinned at a reactive limit:
                // applyFreezeToNetwork owns that bus. A solve normally unfreezes at COMPLETE, so this only
                // shows up when one ended mid-phase.
                continue;
            }
            if (limitType != GpuQLimitType.NONE && bus.isGeneratorVoltageControlEnabled()) {
                // The limit the device pinned at — its row target is limit − loadQ, evaluated at the LIVE
                // voltage for an SVC (OLF freezes at bus.getMinQ()/getMaxQ() read at the check, the same
                // live value); the packed base-V aggregate is 1.3e-4 pu off on rte6515's VL-6226_0.
                int vRow = data.reactiveLimits().voltageRow()[i];
                // A pilot-self pilot enrolled in the candidate table keeps BUS_TARGET_V on its own row while
                // pinned: its pinned balance rides the first enabled member's row (the carrier), or its own
                // row only once the group lost control (flipped to a power row). Read the limit where the
                // device put it — its own row's target here is the VOLTAGE target (measured: 1.055 + loadQ
                // 1.218 = a 2.273 "limit" frozen on VL-6112_0 against the real 0.774, and the base judged
                // MAX_ITERATION_REACHED on that residual).
                int limitRow = pinnedBalanceRow(data, i, vRow, devModes);
                double qLimit = deviceTargets != null && limitRow >= 0 && limitRow < deviceTargets.length
                        ? deviceTargets[limitRow] + data.reactiveLimits().loadQ()[i]
                        : (limitType == GpuQLimitType.MIN_Q ? data.reactiveLimits().minQ()[i]
                           : limitType == GpuQLimitType.MAX_Q ? data.reactiveLimits().maxQ()[i] : data.reactiveLimits().initialTargetQ()[i]);
                bus.freezeGenerationTargetQAndDisableGeneratorVoltageControl(qLimit);
                bus.setQLimitType(GpuQLimitType.toOlf(limitType));
            } else if (limitType == GpuQLimitType.NONE && !bus.isGeneratorVoltageControlEnabled() && bus.getQLimitType().isPresent()
                    && bus.getQLimitType().get() != LfBus.QLimitType.MIN_REALISTIC_V
                    && bus.getQLimitType().get() != LfBus.QLimitType.MAX_REALISTIC_V) {
                bus.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(true);
                bus.setQLimitType(null);
            }
        }
    }

    /** The row whose target holds RL entry {@code i}'s pinned balance ({@code target + loadQ = the limit}).
     *  Its own V row, except for a pilot-self pilot enrolled in the candidate table (its V row is
     *  candidate 0 of a group and still a V identity, mode 0, while pinned): then the same pick the
     *  device's {@code chooseCarrierRow} makes — the first OTHER candidate whose row is neither identity nor
     *  pinned (the carrier) — and its own row when none is live (the group lost control and the device
     *  flipped that row to a power row, mode 1). */
    private static int pinnedBalanceRow(GpuAcData data, int i, int vRowIn, int[] devModes) {
        int vRow = vRowIn;
        // First, the COUPLER-TREE RELOCATION: a relocated child's reactive balance — and so the target the pin
        // wrote — lives on its parent coupler's DUMMY_Q row, which is exactly where the native pass put it
        // (reactiveLimitReactivePowerRow). Reading the bus's own row instead read the coupler's ZERO_V target as a reactive limit and
        // froze that on the bus. Mode 0 at the relocated row means the coupler is open here, so the equations
        // are back home and the own row is right (the same per-scenario test the device makes).
        int relocated = relocatedPinRow(data, vRow);
        if (relocated != vRow && devModes != null && relocated < devModes.length && devModes[relocated] != GpuRowMode.FIXED_VARIABLE) {
            vRow = relocated;
        }
        byte[] role = data.reactiveLimits().voltageControlRole();
        if (devModes == null || role == null || role[i] != GpuVoltageControlRole.PILOT_SELF || data.reactiveRelocation().groupStart() == null
                || vRow < 0 || vRow >= devModes.length || devModes[vRow] != GpuRowMode.FIXED_VARIABLE) {
            return vRow;
        }
        int[] reactiveRelocationGroupStart = data.reactiveRelocation().groupStart();
        int[] reactiveRelocationCandidateRows = data.reactiveRelocation().candidateRows();
        for (int g = 0; g + 1 < reactiveRelocationGroupStart.length; g++) {
            if (reactiveRelocationCandidateRows[reactiveRelocationGroupStart[g]] != vRow) {
                continue;
            }
            for (int c = reactiveRelocationGroupStart[g] + 1; c < reactiveRelocationGroupStart[g + 1]; c++) {
                int r = reactiveRelocationCandidateRows[c];
                if (r == vRow) {
                    continue;
                }
                int m = devModes[r];
                if (m != GpuRowMode.FIXED_VARIABLE && m != GpuRowMode.BUS_TARGET_Q) {
                    return r;
                }
            }
            return vRow;
        }
        return vRow;
    }

    /** The relocated {@code DUMMY_Q} row that hosts the reactive balance of the bus whose own {@code BUS_V} row
     *  is {@code vRow}, or {@code vRow} itself when that bus is not a relocated coupler child. The Java twin of
     *  the native {@code reactiveLimitReactivePowerRow} map, derived from the same zero-impedance pack. */
    private static int relocatedPinRow(GpuAcData data, int vRow) {
        int[] zeroImpedanceBranchIndices = data.zeroImpedanceBranch().indices();
        if (zeroImpedanceBranchIndices == null || vRow < 0) {
            return vRow;
        }
        for (int e = 0; e < zeroImpedanceBranchIndices.length / 11; e++) {
            int o = e * 11;
            int childSide = (zeroImpedanceBranchIndices[o + 10] >> 1) & 3;
            int ownV = childSide == 1 ? zeroImpedanceBranchIndices[o + 6] : (childSide == 2 ? zeroImpedanceBranchIndices[o + 7] : -1);
            if (ownV == vRow && zeroImpedanceBranchIndices[o + 5] >= 0) {
                return zeroImpedanceBranchIndices[o + 5];
            }
        }
        return vRow;
    }
}
