// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See solve_seed.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_q_limit_type.cuh"
#include "../common/codes/gpu_reactive_limits_state.cuh"
#include "../outerloop/area_interchange_control.cuh"
#include "../outerloop/distributed_slack.cuh"
#include "../outerloop/forced_target_q.cuh"
#include "../outerloop/incremental_flow_control.cuh"
#include "../outerloop/incremental_shunt_voltage_control.cuh"
#include "../outerloop/incremental_transformer_voltage_control.cuh"
#include "../outerloop/phase_control.cuh"
#include "../outerloop/pilot_voltage.cuh"
#include "../outerloop/reactive_capability_curves.cuh"
#include "../outerloop/reactive_limits_seed.cuh"
#include "../outerloop/remote_voltage_control.cuh"
#include "../outerloop/secondary_voltage_control.cuh"
#include "../outerloop/shunt_voltage_control.cuh"
#include "../outerloop/transformer_voltage_control.cuh"
#include "../outerloop/voltage_monitoring.cuh"
#include "driver_steps.cuh"
#include "solve_seed.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

// The per-solve seed of a fresh (non-resumed) full load flow: the generators' slack moves cleared, the pilot Q targets
// captured, every reactive-limit PV -> PQ switch seeded (from the base or a previous solve) and stamped on the rows, and every
// in-kernel loop reset to its entry state.
void seedSolve(BatchSolverContext* ctx, DriverPass& drv, const int* reactiveLimitsSeedState, const double* slackBusAnchorShift) {
    const int& n = drv.n;
    const int& nS = drv.nS;
    const bool& usePerScenarioReactiveLimits = drv.usePerScenarioReactiveLimits;
    DeviceBuf& dBaseMode = drv.dBaseMode;
    // Each generator's own slack move is per-solve state, cleared before ANYTHING reads it: the seed-time curve
    // limits below evaluate a curve at base + distributedSlackGeneratorMove wherever distributedSlackGeneratorMoveValid is set. Cleared only after the seed,
    // a reused context evaluated scenario slot s at the move of the contingency the PREVIOUS chunk solved in
    // that slot. applyToNetwork also reads it when the DS phase never fires (NaN targetP otherwise).
    if (ctx->nDistributedSlackGenerators > 0 && ctx->dDistributedSlackGeneratorMove.p) {
        CUDA_CHECK(cudaMemsetAsync(ctx->dDistributedSlackGeneratorMove.p, 0, static_cast<size_t>(ctx->nDistributedSlackGenerators) * nS * sizeof(double), ctx->stream));
        CUDA_CHECK(cudaMemsetAsync(ctx->dDistributedSlackGeneratorMoveValid.p, 0, (size_t) nS * sizeof(char), ctx->stream));
    }
    if (ctx->hasDistrQRelocationGroups && ctx->structure->nReactiveRelocationGroups > 0) {        // plan-B4: capture each group's pilot-Q target from
        size_t need = static_cast<size_t>(ctx->structure->nReactiveRelocationGroups) * nS * sizeof(double);   // the BASE carrier's row,
        if (!ctx->dReactiveRelocationPilotTargetQ.p) ctx->dReactiveRelocationPilotTargetQ.alloc(need);   // BEFORE any RL mode seeding rewrites it
        if (!ctx->dReactiveRelocationSavedPilotTargetV.p) ctx->dReactiveRelocationSavedPilotTargetV.alloc(need);
        if (!ctx->dReactiveRelocationPilotTargetVDeactivated.p) ctx->dReactiveRelocationPilotTargetVDeactivated.alloc(static_cast<size_t>(ctx->structure->nReactiveRelocationGroups) * nS * sizeof(char));
        CUDA_CHECK(cudaMemsetAsync(ctx->dReactiveRelocationPilotTargetVDeactivated.p, 0,      // per-solve state, like limitType
                static_cast<size_t>(ctx->structure->nReactiveRelocationGroups) * nS * sizeof(char), ctx->stream));
        capturePilotQBatch<<<grid(ctx->structure->nReactiveRelocationGroups * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->nReactiveRelocationGroups, nS, n,
                ctx->structure->dReactiveRelocationGroupStart.as<int>(), ctx->structure->dReactiveRelocationCandidateRows.as<int>(), dBaseMode.as<int>(),
                ctx->dTargetBase.as<double>(), ctx->dReactiveRelocationPilotTargetQ.as<double>(),
                ctx->structure->dReactiveRelocationBaseTargetRow.p ? ctx->structure->dReactiveRelocationBaseTargetRow.as<int>() : nullptr);
        if (ctx->reactiveRelocationPilotTargetQDeltas.entries() > 0) {
            SlotEntries pilotQDeltas = ctx->reactiveRelocationPilotTargetQDeltas.view();
            applyPilotQDeltasBatch<<<grid(nS * pilotQDeltas.capacity), BLOCK, 0, ctx->stream>>>(nS, pilotQDeltas, ctx->structure->nReactiveRelocationGroups, n,
                    ctx->structure->dReactiveRelocationGroupStart.as<int>(), ctx->structure->dReactiveRelocationCandidateRows.as<int>(), ctx->dMode.as<int>(),
                    ctx->hasDistrQSubgroups ? ctx->dDistrQSubgroup.as<char>() : nullptr, ctx->nDistrQMembers,
                    ctx->hasDistrQSubgroups ? ctx->dReactiveRelocationDistrQMemberStart.as<int>() : nullptr,
                    ctx->dReactiveRelocationPilotTargetQ.as<double>(), ctx->dTarget.as<double>());
        }
        if (!ctx->dReactiveRelocationPilotReactiveLimitEntry.p) {
            // Per group, the pilot's own LOCAL reactive-limit entry: a pilot whose group lost control
            // in the BASE (every member switched to PQ there) enters the pack as a local entry with an empty
            // reactive range, and that entry carries the group's V target — the value the pilot row
            // takes back when a member switches back to PV (updatePilotTargetVEquationBatch). Built once per context.
            int nReactiveRelocationGroups = ctx->structure->nReactiveRelocationGroups;
            std::vector<int> ptr(nReactiveRelocationGroups + 1);
            CUDA_CHECK(cudaMemcpy(ptr.data(), ctx->structure->dReactiveRelocationGroupStart.p, ptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<int> rows(std::max(ptr[nReactiveRelocationGroups], 1));
            CUDA_CHECK(cudaMemcpy(rows.data(), ctx->structure->dReactiveRelocationCandidateRows.p, static_cast<size_t>(ptr[nReactiveRelocationGroups]) * sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<int> vrow(std::max(ctx->nReactiveLimitEntries, 1));
            std::vector<char> rem(std::max(ctx->nReactiveLimitEntries, 1), 0);
            if (ctx->nReactiveLimitEntries > 0) {
                CUDA_CHECK(cudaMemcpy(vrow.data(), ctx->dReactiveLimitVoltageRow.p, static_cast<size_t>(ctx->nReactiveLimitEntries) * sizeof(int), cudaMemcpyDeviceToHost));
                if (ctx->dReactiveLimitVoltageControlRole.p) {
                    CUDA_CHECK(cudaMemcpy(rem.data(), ctx->dReactiveLimitVoltageControlRole.p, static_cast<size_t>(ctx->nReactiveLimitEntries), cudaMemcpyDeviceToHost));
                }
            }
            std::vector<int> pe(std::max(nReactiveRelocationGroups, 1), -1);
            for (int g = 0; g < nReactiveRelocationGroups; g++) {
                int pilotRow = rows[ptr[g]];
                for (int i = 0; i < ctx->nReactiveLimitEntries; i++) {
                    if (!rem[i] && vrow[i] == pilotRow) { pe[g] = i; break; }
                }
            }
            ctx->dReactiveRelocationPilotReactiveLimitEntry.upload(pe.data(), pe.size() * sizeof(int));
        }
        if (nS == 1 && traceOn(Trace::BATCH_TRACE)) {
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            int nReactiveRelocationGroups = ctx->structure->nReactiveRelocationGroups;
            std::vector<int> ptr(nReactiveRelocationGroups + 1);
            CUDA_CHECK(cudaMemcpy(ptr.data(), ctx->structure->dReactiveRelocationGroupStart.p, ptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<int> rows(ptr[nReactiveRelocationGroups]);
            CUDA_CHECK(cudaMemcpy(rows.data(), ctx->structure->dReactiveRelocationCandidateRows.p, rows.size() * sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<double> pq(nReactiveRelocationGroups), tg(n);
            std::vector<int> md(n);
            CUDA_CHECK(cudaMemcpy(pq.data(), ctx->dReactiveRelocationPilotTargetQ.p, nReactiveRelocationGroups * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(tg.data(), ctx->dTarget.p, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(md.data(), dBaseMode.p, n * sizeof(int), cudaMemcpyDeviceToHost));
            for (int g = 0; g < nReactiveRelocationGroups; g++) {
                std::fprintf(stderr, "PILOTQ_CAPTURE group=%d captured=%.9f candidates:", g, pq[g]);
                for (int c = ptr[g]; c < ptr[g + 1]; c++) {
                    std::fprintf(stderr, " row%d(mode=%d,target=%.9f)", rows[c], md[rows[c]], tg[rows[c]]);
                }
                std::fprintf(stderr, "\n");
            }
        }
    }
    // PV -> PQ switches exist only for a run with OLF's ReactiveLimitsOuterLoop: without it every entry stays NONE (all PV), as OLF,
    // whatever the packed entries or a caller's seed state hold.
    if (reactiveLimitsSeedState != nullptr && ctx->reactiveLimitsOuterLoop && ctx->nReactiveLimitEntries > 0) {
        // WARM CONTINUATION: seed each (controller, scenario)'s limit type (switched to PQ or not) AND switch
        // count from the state the caller passes. A re-solve inside one load flow passes both; a
        // strategy pair passes its parent's PV -> PQ switches with zero counts, since OLF's reactive-limit
        // loop counts afresh in every load flow.
        std::vector<char> qLimitType(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS), sc(static_cast<size_t>(ctx->nReactiveLimitEntries) * nS);
        for (size_t k = 0; k < qLimitType.size(); k++) {
            qLimitType[k] = static_cast<char>(reactiveLimitsStateQLimitType(reactiveLimitsSeedState[k]));
            sc[k] = static_cast<char>(reactiveLimitsStatePvPqSwitchCount(reactiveLimitsSeedState[k]));
        }
        CUDA_CHECK(cudaMemcpyAsync(ctx->dReactiveLimitQLimitType.p, qLimitType.data(), qLimitType.size(), cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpy(ctx->dReactiveLimitPvPqSwitchCount.p, sc.data(), sc.size(), cudaMemcpyHostToDevice));
        // The limits the continued PV -> PQ switches are STAMPED at: OLF restores the generation target Q of each bus switched to PQ from
        // the parent's end, and a GeneratorAction only invalidates the bus's target P, so the first solve keeps
        // the limit at the parent's P (distributedSlackGeneratorSeedAnchor) and the reactive-limit check refreshes it afterwards.
        if (usePerScenarioReactiveLimits) {                            // curve limits at the CURRENT (seeded) targets,
            reactiveCapabilityCurveLimitsBatch<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS, ctx->nReactiveCapabilityCurves, n, ctx->reactiveCapabilityCurveExtrapolate,
                    ctx->dReactiveCapabilityCurveIndices.as<int>(), ctx->dReactiveCapabilityCurveParameters.as<double>(), ctx->dReactiveCapabilityCurvePoints.as<double>(),
                    ctx->dReactiveLimitEntryCurveStart.as<int>(), ctx->dReactiveLimitEntryCurves.as<int>(), curveHvdcAcEmulationView(ctx),
                    ctx->dReactiveLimitMinQ.as<double>(), ctx->dReactiveLimitMaxQ.as<double>(), ctx->dReactiveLimitLoadQ.as<double>(),
                    ctx->dTarget.as<double>(),
                    ctx->reactiveLimitDeltaOverrides.view(), ctx->reactiveCapabilityCurveOverrides.view(),
                    ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMove.as<double>() : nullptr,
                    ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMoveValid.as<char>() : nullptr, ctx->hasDistributedSlackGeneratorSeedAnchor ? ctx->dDistributedSlackGeneratorSeedAnchor.as<double>()
                            : (ctx->hasDistributedSlackGeneratorAnchor ? ctx->dDistributedSlackGeneratorAnchor.as<double>() : nullptr),
                    ctx->hasDistributedSlackGeneratorInitialMove ? ctx->dDistributedSlackGeneratorInitialMove.as<double>() : nullptr, ctx->nDistributedSlackGenerators,
                    ctx->nDistributedSlackBuses > 0 ? ctx->dDistributedSlackBusPhiRow.as<int>() : nullptr,
                    ctx->nDistributedSlackBusRowOverrides > 0 ? ctx->dDistributedSlackBusRowOverride.as<int>() : nullptr, ctx->nDistributedSlackBuses,
                    ctx->nDistributedSlackGenerators > 0 ? ctx->distributedSlackGeneratorBus.as<int>() : nullptr,
                    ctx->dReactiveLimitMinQPerScenario.as<double>(), ctx->dReactiveLimitMaxQPerScenario.as<double>(),
                    ctx->dReactiveLimitLoadQPerScenario.as<double>(), nullptr);
            launchForcedTargetQLimits(ctx, nS, n);
        }
        seedReactiveLimitRowModeFromState<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS, n,
                ctx->dReactiveLimitVoltageRow.as<int>(), ctx->dReactiveLimitReactivePowerRow.p ? ctx->dReactiveLimitReactivePowerRow.as<int>() : nullptr,
                usePerScenarioReactiveLimits ? ctx->dReactiveLimitMinQPerScenario.as<double>() : ctx->dReactiveLimitMinQ.as<double>(),
                usePerScenarioReactiveLimits ? ctx->dReactiveLimitMaxQPerScenario.as<double>() : ctx->dReactiveLimitMaxQ.as<double>(),
                usePerScenarioReactiveLimits ? ctx->dReactiveLimitLoadQPerScenario.as<double>() : ctx->dReactiveLimitLoadQ.as<double>(), ctx->dReactiveLimitQLimitType.as<char>(),
                ctx->dMode.as<int>(), ctx->dTarget.as<double>(), usePerScenarioReactiveLimits ? 1 : 0,
                ctx->nReactiveLimitRemoteEntries > 0 ? ctx->dReactiveLimitVoltageControlRole.as<char>() : nullptr,
                ctx->dReactiveLimitSavedTarget.as<double>(), ctx->dReactiveLimitSavedRowMode.as<char>(),
                ctx->hasReactiveLimitBaseQLimitType ? ctx->dReactiveLimitBaseQLimitType.as<char>() : nullptr,
                pilotSelfReactiveLimitEntryOrNull(ctx), ctx->structure->nReactiveRelocationGroups,
                ctx->dReactiveLimitTargetV.as<double>(), ctx->reactiveLimitTargetVOverrides.view(), ctx->nSecondaryVoltageControlControllerBuses > 0 ? ctx->dSecondaryVoltageControlReactiveLimitTargetV.as<double>() : nullptr,
                ctx->nReactiveLimitLocalOverrides > 0 ? ctx->dReactiveLimitLocalOverride.as<char>() : nullptr,
                ctx->nReactiveLimitLocalOverrides > 0 ? ctx->dReactiveLimitLocalTargetV.as<double>() : nullptr,
                ctx->hasReactiveLimitSeedTargetQ ? ctx->dReactiveLimitSeedTargetQ.as<double>() : nullptr,
                ctx->nReactiveLimitIslanded > 0 ? ctx->dReactiveLimitIslanded.as<char>() : nullptr);
    } else {
        CUDA_CHECK(cudaMemsetAsync(ctx->dReactiveLimitPvPqSwitchCount.p, 0, static_cast<size_t>(ctx->nReactiveLimitEntries) * nS * sizeof(char), ctx->stream));
        if (!ctx->reactiveLimitsOuterLoop && ctx->nReactiveLimitEntries > 0) {   // no ReactiveLimits loop: no PV -> PQ switch, a base-PQ
            CUDA_CHECK(cudaMemsetAsync(ctx->dReactiveLimitQLimitType.p, GpuQLimitType::NONE,   // controller included
                    static_cast<size_t>(ctx->nReactiveLimitEntries) * nS * sizeof(char), ctx->stream));
        } else if (ctx->nReactiveLimitEntries > 0) {                       // seed the limit type of base-PQ controllers (else all PV)
            initReactiveLimitQLimitTypeBatch<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS, n, ctx->dReactiveLimitVoltageRow.as<int>(),
                    ctx->dReactiveLimitReactivePowerRow.p ? ctx->dReactiveLimitReactivePowerRow.as<int>() : nullptr,
                    ctx->dReactiveLimitTargetV.as<double>(), ctx->dReactiveLimitFlags.as<char>(), dBaseMode.as<int>(), ctx->dX.as<double>(),
                    ctx->dReactiveLimitQLimitType.as<char>(),
                    ctx->reactiveLimitTargetVOverrides.view(),
                    ctx->nSecondaryVoltageControlControllerBuses > 0 ? ctx->dSecondaryVoltageControlReactiveLimitTargetV.as<double>() : nullptr,
                    ctx->nReactiveLimitRemoteEntries > 0 ? ctx->dReactiveLimitVoltageControlRole.as<char>() : nullptr,
                    ctx->nReactiveLimitRemoteEntries > 0 ? ctx->dReactiveLimitComparedVoltageRow.as<int>() : nullptr,
                    ctx->dTarget.as<double>(), ctx->dReactiveLimitMinQ.as<double>(), ctx->dReactiveLimitMaxQ.as<double>(),
                    ctx->dReactiveLimitLoadQ.as<double>(),
                    ctx->hasReactiveLimitBaseQLimitType ? ctx->dReactiveLimitBaseQLimitType.as<char>() : nullptr,
                    pilotSelfReactiveLimitEntryOrNull(ctx), ctx->structure->nReactiveRelocationGroups,
                    ctx->nReactiveLimitGeneratorsLost > 0 ? ctx->dReactiveLimitGeneratorsLost.as<char>() : nullptr);
        }
    }
    // The seeded PV/PQ states must reach the ROWS before the first fill: the balance of an enrolled pilot switched to PQ
    // goes on the carrier's target (its DISTR_Q target otherwise), and a group that lost control in the
    // base has its pilot row flipped already. Both kernels are idempotent; the RL pass runs them again.
    // Every scenario takes the seed stamps: updateDistrQTargetsBatch returns on !outerLoopCheckActive[s], and outerLoopCheckActive is
    // only written by the outer-loop passes - at seed time it still held the context's zeros (or the previous
    // chunk's last pass), so the carrier stamp below never ran before the FIRST solve. A pilot-self pilot that
    // lost every generator with two live survivors then solved its carrier row against the plan's renormalized
    // DISTR_Q seed (0.5 x its -0.6 balance = -0.3) instead of its balance: measured as 0.258 kV at b2 on
    // devicePilotControllerLosingItsOwnGeneratorMatchesCpu (OLF's BUS_TARGET_Q of the pilot 0.3 pu off at the
    // device state), and rte6515's 50 refused pilot outages. One survivor hid it: its seed degenerates to the
    // pilot's own target. The passes upload their own outerLoopCheckActive before they run.
    {
        std::vector<char> allOn(nS, 1);
        CUDA_CHECK(cudaMemcpyAsync(ctx->dOuterLoopCheckActive.p, allOn.data(), nS, cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }
    // The dynamic reactive limits BEFORE the seed stamps, on the cold path too. They were first computed in the
    // reactive-limit pass, after the first solve, so the stamps read the arrays unfilled: a gen-lost pilot's
    // carrier target came out 0 - 0 instead of 0 - loadQ, and a survivor's sharing target summed a switched-to-PQ
    // peer's (limit - loadQ) as 0 - 0 - measured on devicePilotControllerLosingItsOwnGeneratorMatchesCpu
    // as rows 2 and 4 both at target 0 during the solve (OLF: -0.6 and -0.3). The warm path already ran
    // this before seeding; here it runs after initReactiveLimitQLimitTypeBatch has set the scenario's limit types.
    if (usePerScenarioReactiveLimits) {                            // curve limits at the CURRENT (seeded) targets,
        reactiveCapabilityCurveLimitsBatch<<<grid(ctx->nReactiveLimitEntries * nS), BLOCK, 0, ctx->stream>>>(ctx->nReactiveLimitEntries, nS, ctx->nReactiveCapabilityCurves, n, ctx->reactiveCapabilityCurveExtrapolate,
                ctx->dReactiveCapabilityCurveIndices.as<int>(), ctx->dReactiveCapabilityCurveParameters.as<double>(), ctx->dReactiveCapabilityCurvePoints.as<double>(),
                    ctx->dReactiveLimitEntryCurveStart.as<int>(), ctx->dReactiveLimitEntryCurves.as<int>(), curveHvdcAcEmulationView(ctx),
                ctx->dReactiveLimitMinQ.as<double>(), ctx->dReactiveLimitMaxQ.as<double>(), ctx->dReactiveLimitLoadQ.as<double>(),
                ctx->dTarget.as<double>(),
                ctx->reactiveLimitDeltaOverrides.view(), ctx->reactiveCapabilityCurveOverrides.view(),
                ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMove.as<double>() : nullptr,
                ctx->nDistributedSlackGenerators > 0 ? ctx->dDistributedSlackGeneratorMoveValid.as<char>() : nullptr, ctx->hasDistributedSlackGeneratorAnchor ? ctx->dDistributedSlackGeneratorAnchor.as<double>() : nullptr, ctx->hasDistributedSlackGeneratorInitialMove ? ctx->dDistributedSlackGeneratorInitialMove.as<double>() : nullptr, ctx->nDistributedSlackGenerators,
                ctx->nDistributedSlackBuses > 0 ? ctx->dDistributedSlackBusPhiRow.as<int>() : nullptr,
                ctx->nDistributedSlackBusRowOverrides > 0 ? ctx->dDistributedSlackBusRowOverride.as<int>() : nullptr, ctx->nDistributedSlackBuses,
                ctx->nDistributedSlackGenerators > 0 ? ctx->distributedSlackGeneratorBus.as<int>() : nullptr,
                ctx->dReactiveLimitMinQPerScenario.as<double>(), ctx->dReactiveLimitMaxQPerScenario.as<double>(),
                ctx->dReactiveLimitLoadQPerScenario.as<double>(), nullptr);
        launchForcedTargetQLimits(ctx, nS, n);
    }
    launchReMergedVoltageTargetRowUpdate(ctx, drv, true);                       // a main seeded switched to PQ (a continued solve)
    launchPilotTargetVEquationUpdate(ctx, drv, true, true);                     // seed: base limits, or the previous solve's flip
    launchDistrQTargetsInner(ctx, drv);
    if (ctx->distributedSlackOuterLoop && ctx->nDistributedSlackBuses > 0) {   // the run's loop, not the packed table
        distributedSlackResetSolve(ctx, drv);
    }
    if (ctx->areaInterchangeControlOuterLoop) {
        areaInterchangeControlResetSolve(ctx, drv);
    }
    if (ctx->nMonitoringVoltageControllers > 0) {
        monitoringVoltageResetSolve(ctx, drv);
    }
    if (ctx->nIncrementalTransformerVoltageControlControllerBranches > 0) {
        incrementalTransformerVoltageControlResetSolve(ctx, drv);
    }
    if (ctx->nPhaseControlControllerBranches > 0) {
        phaseControlResetSolve(ctx, drv);
    }
    if (ctx->nTransformerVoltageControlMembers > 0) {
        transformerVoltageControlResetSolve(ctx, drv);
    }
    if (ctx->nSecondaryVoltageControlControllerBuses > 0) {
        secondaryVoltageControlResetSolve(ctx, drv);
    }
    if (ctx->nShuntVoltageControlMembers > 0) {
        shuntVoltageControlResetSolve(ctx, drv);
    }
    if (ctx->nIncrementalFlowControlControllerBranches > 0) {
        incrementalFlowControlResetSolve(ctx, drv);
    }
    if (ctx->nIncrementalShuntVoltageControlControllerShunts > 0) {
        incrementalShuntVoltageControlResetSolve(ctx, drv);
    }
    if (ctx->distributedSlackOuterLoop && ctx->slackOnlyDistribution) {
        slackOnlyDistributionResetSolve(ctx, drv, slackBusAnchorShift);
    }
}

}  // namespace olfgpu
