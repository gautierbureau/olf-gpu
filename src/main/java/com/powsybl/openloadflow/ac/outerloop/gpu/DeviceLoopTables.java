/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.ReactiveLimitEntries;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuQLimitType;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuRealisticVoltageFixingLoop;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuVoltageControlRole;

/**
 * The run-constant outer-loop tables EVERY batch context needs before its driver runs — the batched
 * SA's chunk contexts and the base load flow's batch-of-one alike: the reactive-capability curve
 * channel, the voltage-dependent SVC limits, the remote-member RL kinds and the per-generator
 * DistributedSlack participants. One place, so a batch of one never drives a different loop than a
 * batch of five hundred. The base load flow used to skip these: without the remote kinds the driver
 * read every remote member's DISTR_Q share row as a base PQ pin (limit type seeded at solve start,
 * never re-checked, switched back on the pilot's voltage) — three of rte6515's remote controllers
 * came out pinned where OLF's loop had never touched them.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class DeviceLoopTables {

    /** ReactiveLimitsOuterLoop.REALISTIC_VOLTAGE_MARGIN. */
    private static final double REALISTIC_VOLTAGE_MARGIN = 1.02;

    /** {@code AcloadFlowEngine.runOuterLoops}: under {@code voltageRemoteControlRobustMode} the
     *  realistic-voltage check is DEFERRED until the last configured loop that declares
     *  {@code canFixUnrealisticState()} has run, so that loop gets its chance to repair the state
     *  instead of the run failing on it. Returns which loop that is, as the device's hint: a
     *  GpuRealisticVoltageFixingLoop (NONE: every solve is checked). */
    public static int realisticFixingLoop(com.powsybl.openloadflow.ac.AcLoadFlowParameters p) {
        if (!p.isVoltageRemoteControlRobustMode()) {
            return GpuRealisticVoltageFixingLoop.NONE;
        }
        com.powsybl.openloadflow.ac.outerloop.AcOuterLoop last = null;
        for (var ol : p.getOuterLoops()) {
            if (ol.canFixUnrealisticState()) {
                last = ol;
            }
        }
        if (last == null) {
            return GpuRealisticVoltageFixingLoop.NONE;
        }
        return last instanceof com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop
                ? GpuRealisticVoltageFixingLoop.REACTIVE_LIMITS : GpuRealisticVoltageFixingLoop.TRANSFORMER_VOLTAGE_CONTROL;
    }

    private DeviceLoopTables() {
    }

    /**
     * OLF's forced-PV election keys, installed on a batch context the moment it is CREATED.
     *
     * <p>Every context carrying reactive-limit entries drives the RL phase — a full load flow does so
     * even when OLF, not the device, owns the outer loops (the device stabilizes the loop internally
     * and OLF's own loop then finds nothing to switch). {@code switchPvPq}'s network-wide rule that
     * the strongest bus stays PV when EVERY PV bus would switch is part of that phase, so its keys
     * belong to the context's construction and not to the driver's optional table install. They used
     * to be installed only for a driver run, and a context without them fell back to a silent
     * per-scenario rejection to the CPU; installing them here makes that rejection unreachable, which
     * is why {@code solveBatchContext} now refuses a context that reaches it without them.
     */
    public static void installRlElection(long handle, GpuAcData gpuData) {
        ReactiveLimitsNative.setBatchRlElection(handle, gpuData.reactiveLimits().nominalV(), gpuData.reactiveLimits().activePowerRow(),
                gpuData.reactiveLimits().targetP(), gpuData.reactiveLimits().idRank());
    }

    public static void install(long handle, GpuAcData gpuData, ReactiveCapabilityCurves.Channel rlCurves,
                               ForceTargetQInReactiveLimits.Channel forcedTargetQ,
                               boolean rlRobustMode, double minRealisticVoltage, double maxRealisticVoltage,
                               double minNominalVoltageRealisticVoltageCheck, int realisticFixingLoop) {
        // AcloadFlowEngine's realistic-voltage check, applied to the states INSIDE the driver's sweep.
        // The row carries its bus's nominal voltage (0 = a row the check never looks at, which is
        // every row that is not a non-fictitious bus's BUS_V), so the run's threshold is applied here.
        double[] rowNominalV = gpuData.voltageRows().nominalV();
        byte[] rowChecked = new byte[rowNominalV.length];
        for (int r = 0; r < rowNominalV.length; r++) {
            rowChecked[r] = rowNominalV[r] > 0 && rowNominalV[r] >= minNominalVoltageRealisticVoltageCheck
                    ? (byte) 1 : (byte) 0;
        }
        ReactiveLimitsNative.setBatchRealisticCheck(handle, rowChecked, minRealisticVoltage, maxRealisticVoltage, 1,
                realisticFixingLoop);
        // Curve limits at runtime P: install (or clear, for cached contexts) the per-gen curve channel.
        ReactiveLimitsNative.setBatchRlCurves(handle,
                rlCurves != null ? rlCurves.cvIdx() : null,
                rlCurves != null ? rlCurves.cvIn() : null,
                rlCurves != null ? rlCurves.cvPts() : null,
                rlCurves != null ? rlCurves.cvHv() : null,
                rlCurves != null && rlCurves.extrapolate() ? 1 : 0);
        // forceTargetQInReactiveLimits: the clamped generators' Q targets follow their P (or the channel is cleared).
        ForceTargetQInReactiveLimits.install(handle, forcedTargetQ);
        // Voltage-dependent SVC reactive limits: install the quadratic coefficients (zero on
        // SVC-less entries — the kernel's limit(v) then degenerates to the packed static limit).
        ReactiveLimitsNative.setBatchRlSvcLimits(handle,
                gpuData.reactiveLimits().svcMinB(), gpuData.reactiveLimits().svcMaxB(), gpuData.reactiveLimits().baseVSquared());
        // OLF's global forced-PV election keys — the RL pass keeps the strongest bus PV when every
        // PV bus would switch, instead of rejecting the scenario to the CPU.
        ReactiveLimitEntries rl = gpuData.reactiveLimits();
        ReactiveLimitsNative.setBatchRlElection(handle, rl.nominalV(), rl.activePowerRow(), rl.targetP(), rl.idRank());
        // voltageRemoteControlRobustMode: install (or clear, for a cached context whose run has the
        // mode off) the realistic-V criteria the RL pass applies to REMOTE voltage controllers.
        // REALISTIC_VOLTAGE_MARGIN = 1.02, straight out of ReactiveLimitsOuterLoop.
        ReactiveLimitsNative.setBatchRlRobust(handle, gpuData.reactiveLimits().initialTargetQ(),
                minRealisticVoltage * REALISTIC_VOLTAGE_MARGIN, maxRealisticVoltage / REALISTIC_VOLTAGE_MARGIN,
                rlRobustMode ? 1 : 0);
        // OLF's own base QLimitTypes, so a realistic-voltage switch is not re-seeded as a Q-limit one.
        byte[] baseLimitTypes = gpuData.reactiveLimits().baseQLimitType();
        boolean anyBaseLt = false;
        for (byte b : baseLimitTypes) {
            anyBaseLt |= b != GpuQLimitType.NONE;
        }
        if (System.getenv("OLF_GPU_RL_TRACE") != null) {
            // Which install carries which kinds: the base solve's context and the batch's both come
            // through here, with packs that may differ (the batch's is extracted after the base
            // solve's applyToNetwork). A kind the batch never receives cannot be what seeds it.
            StringBuilder ks = new StringBuilder();
            for (int i = 0; i < baseLimitTypes.length; i++) {
                if (baseLimitTypes[i] != GpuQLimitType.NONE) {
                    ks.append(' ').append(i).append(':').append(baseLimitTypes[i]);
                }
            }
            System.err.println("GPU_RL_KINDS install handle=" + handle + " nrl=" + gpuData.reactiveLimits().voltageRow().length
                    + " anyBaseLt=" + anyBaseLt + " kinds:" + ks + " caller="
                    + Thread.currentThread().getStackTrace()[2].getClassName().replaceAll(".*\\.", ""));
        }
        ReactiveLimitsNative.setBatchRlBaseLimitType(handle, anyBaseLt ? baseLimitTypes : null);
        // Remote-member RL entries (plan-B1): install (or clear, on a cached context) the kinds.
        boolean anyRlRemote = false;
        for (byte b : gpuData.reactiveLimits().voltageControlRole()) {
            anyRlRemote |= b != GpuVoltageControlRole.LOCAL;
        }
        ReactiveLimitsNative.setBatchRlRemote(handle,
                anyRlRemote ? gpuData.reactiveLimits().voltageControlRole() : null,
                anyRlRemote ? gpuData.reactiveLimits().comparedVoltageRow() : null,
                anyRlRemote ? gpuData.remoteMemberReactiveTerms().row() : null,
                anyRlRemote ? gpuData.remoteMemberReactiveTerms().element() : null,
                anyRlRemote ? gpuData.remoteMemberReactiveTerms().side() : null,
                anyRlRemote ? gpuData.remoteMemberReactiveTerms().kind() : null,
                anyRlRemote ? gpuData.remoteMemberReactiveTerms().weight() : null,
                anyRlRemote ? gpuData.distrQ().rowMember() : null,
                anyRlRemote ? gpuData.distrQ().injectionMember() : null,
                anyRlRemote ? gpuData.distrQMembers().key() : null,
                anyRlRemote ? gpuData.distrQMembers().reactiveLimitEntry() : null,
                anyRlRemote ? gpuData.distrQMembers().groupStart() : null,
                anyRlRemote ? gpuData.distrQMembers().groupCount() : null,
                anyRlRemote ? gpuData.distrQMembers().row() : null,
                anyRlRemote ? gpuData.distrQMembers().hasRow() : null,
                anyRlRemote ? gpuData.distrQMembers().relocationGroup() : null,
                anyRlRemote ? gpuData.distrQMembers().targetQ() : null,
                anyRlRemote ? gpuData.distrQMembers().targetPeerStart() : null,
                anyRlRemote ? gpuData.distrQMembers().targetPeer() : null);
        // Per-generator DistributedSlack participants (OLF's water-fill saturates per generator).
        DistributedSlackNative.setBatchDsGenerators(handle, gpuData.distributedSlack().generatorBus(), gpuData.distributedSlack().generatorFactor(),
                gpuData.distributedSlack().generatorMaxDelta(), gpuData.distributedSlack().generatorMinDelta(),
                gpuData.distributedSlack().generatorMarginUp(), gpuData.distributedSlack().generatorMarginDown(),
                gpuData.distributedSlack().busGeneratorStart(), gpuData.distributedSlack().busGeneratorEnd());
        DistributedSlackNative.setBatchDsSlackRooms(handle, gpuData.distributedSlack().slackRoomUp(), gpuData.distributedSlack().slackRoomDown());
    }
}
