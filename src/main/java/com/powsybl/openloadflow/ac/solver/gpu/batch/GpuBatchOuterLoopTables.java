/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;

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
public final class GpuBatchOuterLoopTables {

    /** ReactiveLimitsOuterLoop.REALISTIC_VOLTAGE_MARGIN. */
    private static final double REALISTIC_VOLTAGE_MARGIN = 1.02;

    private GpuBatchOuterLoopTables() {
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
        GpuAcNewtonSolver.setBatchRlElection(handle, gpuData.rlNominalV(), gpuData.rlPRow(),
                gpuData.rlTargetP(), gpuData.rlIdRank());
    }

    public static void install(long handle, GpuAcData gpuData, GpuBatchRlCurves.Channel rlCurves,
                               boolean rlRobustMode, double minRealisticVoltage, double maxRealisticVoltage) {
        // Curve limits at runtime P: install (or clear, for cached contexts) the per-gen curve channel.
        GpuAcNewtonSolver.setBatchRlCurves(handle,
                rlCurves != null ? rlCurves.cvIdx() : null,
                rlCurves != null ? rlCurves.cvIn() : null,
                rlCurves != null ? rlCurves.cvPts() : null,
                rlCurves != null && rlCurves.extrapolate() ? 1 : 0);
        // Voltage-dependent SVC reactive limits: install the quadratic coefficients (zero on
        // SVC-less entries — the kernel's limit(v) then degenerates to the packed static limit).
        GpuAcNewtonSolver.setBatchRlSvcLimits(handle,
                gpuData.rlSvcBmin(), gpuData.rlSvcBmax(), gpuData.rlV2Base());
        // OLF's global forced-PV election keys — the RL pass keeps the strongest bus PV when every
        // PV bus would switch, instead of rejecting the scenario to the CPU.
        GpuAcNewtonSolver.setBatchRlElection(handle, gpuData.rlNominalV(), gpuData.rlPRow(), gpuData.rlTargetP(), gpuData.rlIdRank());
        // voltageRemoteControlRobustMode: install (or clear, for a cached context whose run has the
        // mode off) the realistic-V criteria the RL pass applies to REMOTE voltage controllers.
        // REALISTIC_VOLTAGE_MARGIN = 1.02, straight out of ReactiveLimitsOuterLoop.
        GpuAcNewtonSolver.setBatchRlRobust(handle, gpuData.rlInitTargetQ(),
                minRealisticVoltage * REALISTIC_VOLTAGE_MARGIN, maxRealisticVoltage / REALISTIC_VOLTAGE_MARGIN,
                rlRobustMode ? 1 : 0);
        // Remote-member RL entries (plan-B1): install (or clear, on a cached context) the kinds.
        boolean anyRlRemote = false;
        for (byte b : gpuData.rlRemote()) {
            anyRlRemote |= b != 0;
        }
        GpuAcNewtonSolver.setBatchRlRemote(handle,
                anyRlRemote ? gpuData.rlRemote() : null,
                anyRlRemote ? gpuData.rlCmpVrow() : null,
                anyRlRemote ? gpuData.rmqRow() : null,
                anyRlRemote ? gpuData.rmqElem() : null,
                anyRlRemote ? gpuData.rmqSide() : null,
                anyRlRemote ? gpuData.rmqKind() : null,
                anyRlRemote ? gpuData.rmqWeight() : null,
                anyRlRemote ? gpuData.dqKMember() : null,
                anyRlRemote ? gpuData.dqJMember() : null,
                anyRlRemote ? gpuData.dqMemberKey() : null,
                anyRlRemote ? gpuData.dqMemberRlEntry() : null,
                anyRlRemote ? gpuData.dqMemberGroupStart() : null,
                anyRlRemote ? gpuData.dqMemberGroupCount() : null,
                anyRlRemote ? gpuData.dqMemberRow() : null,
                anyRlRemote ? gpuData.dqMemberHasRow() : null,
                anyRlRemote ? gpuData.dqMemberQrGroup() : null,
                anyRlRemote ? gpuData.dqMemberTargetQ() : null);
        // Per-generator DistributedSlack participants (OLF's water-fill saturates per generator).
        GpuAcNewtonSolver.setBatchDsGenerators(handle, gpuData.dgBus(), gpuData.dgFactor(),
                gpuData.dgMaxDelta(), gpuData.dgMinDelta(),
                gpuData.dgMarginUp(), gpuData.dgMarginDown(),
                gpuData.busGenStart(), gpuData.busGenEnd());
    }
}
