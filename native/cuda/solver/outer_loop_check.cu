// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See outer_loop_check.cuh.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../common/check.cuh"
#include "../common/launch.cuh"
#include "../common/stats.cuh"
#include "../common/codes/gpu_ac_solver_status.cuh"
#include "../common/codes/gpu_controlled_flow.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"
#include "../diagnostics/driver_traces.cuh"
#include "../equations/branch_fill.cuh"
#include "../equations/fill_layout.cuh"
#include "../equations/bus_injection_fill.cuh"
#include "../equations/zero_impedance_fill.cuh"
#include "../outerloop/area_interchange_control.cuh"
#include "../outerloop/automation_system.cuh"
#include "../outerloop/distributed_slack.cuh"
#include "../outerloop/hvdc_ac_emulation.cuh"
#include "../outerloop/incremental_flow_control.cuh"
#include "../outerloop/incremental_shunt_voltage_control.cuh"
#include "../outerloop/incremental_transformer_voltage_control.cuh"
#include "../outerloop/phase_control.cuh"
#include "../outerloop/reactive_limits.cuh"
#include "../outerloop/remote_voltage_control.cuh"
#include "../outerloop/secondary_voltage_control.cuh"
#include "../outerloop/shunt_voltage_control.cuh"
#include "../outerloop/transformer_voltage_control.cuh"
#include "../outerloop/voltage_monitoring.cuh"
#include "driver_steps.cuh"
#include "outer_loop_check.cuh"
#include "../common/trace.cuh"
#include "scenario_override_tables.cuh"

namespace olfgpu {

void checkOuterLoop(BatchSolverContext* ctx, DriverPass& drv, std::vector<char>& outerLoopUnstable) {
    const int& n = drv.n;
    const int& nnz = drv.nnz;
    const int& nS = drv.nS;
    const size_t& nTot = drv.nTot;
    const int& outer = drv.outer;
    const int& loopIdx = drv.loopIdx;
    const int& nLoops = drv.nLoops;
    const int (&loopOrder)[MAX_DEVICE_LOOPS] = drv.loopOrder;
    const double& reactiveLimitTol = drv.reactiveLimitTol;
    const bool& rejectSlackDistributionFailure = drv.rejectSlackDistributionFailure;
    const bool& reactiveLimitsAuditOn = drv.reactiveLimitsAuditOn;
    std::vector<std::vector<std::string>>& auditLog = drv.reactiveLimitsAuditLog;
    std::vector<char>& outerLoopCheckActive = drv.outerLoopCheckActive;
    std::vector<char>& loopRejected = drv.loopRejected;
    std::vector<char>& active = drv.active;
    std::vector<char>& live = drv.live;
    BatchResult& res = drv.res;
    gCheckRounds++;
    int nChecked = 0;
    for (int s = 0; s < nS; s++) {
        if (outerLoopCheckActive[s]) {
            ctx->outerLoopChecks[s]++;
            nChecked++;
        }
    }
    CUDA_CHECK(cudaMemcpy(ctx->dOuterLoopCheckActive.p, outerLoopCheckActive.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
    // generator Q per controller = (Σ branch q + shunt q) at its BUS_V row + load Q: qBuf[row] is the raw bus balance
    // (no −target, no applyRowModeBatch) on the FROZEN routing - the relocation only concerns the Newton residual. The
    // closed branches' part is dQbufClosed: every Newton fill zeroes it on its active scenarios and adds the closed
    // branches' p and q from the very outputs it puts in F (fillClosedBranchBatch), so a scenario holds them at the
    // state of its last fill - the state its convergence test passed on, and the state every check reads: a check that
    // changes a scenario sends it to the next pass's solve, and a STABLE one moves it on to its next loop against the
    // SAME state (a pass checks only the scenarios it solved), so no state moves between a scenario's last fill and its
    // checks (a retired scenario's step is exactly 0). It used to be recomputed by a pass of its own at every check, the
    // same expressions on the same state: 0.71 s of 3.2 s of kernels on rte6515 call 0, 74 of its 106 passes checking
    // an unchanged batch. The smaller families add theirs for the checked scenarios; disabled elements are skipped
    // (outage); they write no Jacobian (vals null): a retired scenario keeps its identity system.
    if (ctx->layout.interleaved) {
        copyToScenarioMajor(n, ctx->layout, ctx->dQbufClosed.as<double>(), ctx->dQbuf.as<double>(), nullptr, ctx->stream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(ctx->dQbuf.p, ctx->dQbufClosed.p, nTot * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
    }
    if (ctx->structure->openBranch.elementCount) fillOpenBranchBatch<<<grid(ctx->structure->openBranch.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->openBranch.elementCount, nS, n, nnz, ctx->nElements, ctx->openBranchOffset, ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(), ctx->structure->openBranch.destination.as<int>(), ctx->dX.as<double>(), nullptr, ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(), nullptr, nullptr, nullptr, 0, ReactiveRelocationTables{}, scenarioMajorLayout(nS));
    if (ctx->structure->shunt.elementCount) fillShuntBatch<<<grid(ctx->structure->shunt.elementCount * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->shunt.elementCount, nS, n, nnz, ctx->nElements, ctx->shuntOffset, ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(), ctx->structure->shunt.destination.as<int>(), ctx->dX.as<double>(), nullptr, ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(), shuntAdmittanceOverrides(ctx), nullptr, nullptr, nullptr, 0, ReactiveRelocationTables{}, scenarioMajorLayout(nS));
    if (ctx->structure->nLoadModelTerms) fillLoadModelBatch<<<grid(ctx->structure->nLoadModelTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->structure->nLoadModelTerms, nS, n, nnz, ctx->structure->dLoadModelParameters.as<double>(), ctx->structure->dLoadModelIndices.as<int>(), ctx->structure->dLoadModelDestination.as<int>(), ctx->dX.as<double>(), nullptr, ctx->dQbuf.as<double>(), ctx->dOuterLoopCheckActive.as<char>(), nullptr, nullptr, nullptr, scenarioMajorLayout(nS));
    // Zero-impedance dummy transfers MUST be in the Q/P buffer too: the distributed-slack retain
    // reads the slack row's balance from dQbuf — without the coupler's dummy-P the DS sees a
    // phantom mismatch equal to the through-flow and distributes it FOREVER (measured: +392 MW
    // runaway on the first switch-close scenario until the per-scenario budget rejected it).
    if (ctx->nZeroImpedanceBranch) fillZeroImpBatch<<<grid(ctx->nZeroImpedanceBranch * nS), BLOCK, 0, ctx->stream>>>(ctx->nZeroImpedanceBranch, nS, n, nnz, ctx->structure->dZeroImpedanceBranchParameters.as<double>(), ctx->structure->dZeroImpedanceBranchIndices.as<int>(), ctx->structure->dZeroImpedanceBranchDestination.as<int>(), ctx->dX.as<double>(), nullptr, ctx->dQbuf.as<double>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->nBaseClosedZeroImpedanceBranches, ctx->nZeroImpedanceBranchCloseOverrides, ctx->dZeroImpedanceBranchCloseScenarioElement.as<int>(), ctx->nZeroImpedanceBranchOpenOverrides, ctx->dZeroImpedanceBranchOpenScenarioElement.as<int>(), nullptr, nullptr, nullptr, 0, ReactiveRelocationTables{}, ctx->nSpanningTreeOverrides, ctx->nSpanningTreeOverrides ? ctx->dSpanningTreeOverrideScenarioElementValue.as<int>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dZeroImpedanceBranchOpenMask.as<char>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dZeroImpedanceBranchCloseMask.as<char>() : nullptr, ctx->hasZeroImpedanceBranchMasks ? ctx->dSpanningTreeOverrideMask.as<char>() : nullptr, scenarioMajorLayout(nS));
    // Remote-member OWN q (plan-B3): fill each member's own reactive injection into its (suppressed,
    // hence free) qBuf row so the RL kernel evaluates remote entries like local ones.
    if (ctx->nRemoteMemberReactiveTerms > 0) {
        fillRemoteMemberQBatch<<<grid(ctx->nRemoteMemberReactiveTerms * nS), BLOCK, 0, ctx->stream>>>(ctx->nRemoteMemberReactiveTerms, nS, n,
                ctx->nElements, ctx->closedBranchOffset, ctx->openBranchOffset,
                ctx->dRemoteMemberReactiveTermRow.as<int>(), ctx->dRemoteMemberReactiveTermElement.as<int>(), ctx->dRemoteMemberReactiveTermSide.as<int>(),
                ctx->dRemoteMemberReactiveTermKind.as<int>(), ctx->dRemoteMemberReactiveTermWeight.as<double>(),
                ctx->structure->closedBranch.parameters.as<double>(), ctx->structure->closedBranch.indices.as<int>(), ctx->structure->shunt.parameters.as<double>(), ctx->structure->shunt.indices.as<int>(),
                ctx->structure->openBranch.parameters.as<double>(), ctx->structure->openBranch.indices.as<int>(),
                ctx->dX.as<double>(), ctx->dQbuf.as<double>(), ctx->dDisabled.as<char>(), ctx->dOuterLoopCheckActive.as<char>(),
                closedBranchParameterOverrides(ctx),
                shuntAdmittanceOverrides(ctx));
    }
    // a link on the slack bus, as OLF's slack mismatch sees it: read by the DistributedSlack check alone
    if (ctx->distributedSlackOuterLoop && ctx->nHvdcAcEmulation > 0 && ctx->slackBusPhiRow >= 0) {
        hvdcSlackBalanceBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, ctx->nHvdcAcEmulation, n, ctx->nElements, ctx->slackBusPhiRow,
                ctx->hasSlackBusRowOverride ? ctx->dSlackBusRowOverride.as<int>() : nullptr,
                ctx->dHvdcBus1PRow.as<int>(), ctx->dHvdcBus2PRow.as<int>(), ctx->dHvdcBus1PhiColumn.as<int>(), ctx->dHvdcBus2PhiColumn.as<int>(),
                ctx->dHvdcElement.as<int>(), ctx->dHvdcAcEmulationParameters.as<double>(), ctx->dHvdcAcEmulationStatus.as<int>(),
                ctx->dHvdcAppliedDelta1.as<double>(), ctx->dHvdcAppliedDelta2.as<double>(), ctx->dHvdcContingencyOutage.as<char>(),
                ctx->dDisabled.as<char>(), ctx->dX.as<double>(), ctx->dOuterLoopCheckActive.as<char>(), ctx->dQbuf.as<double>());
    }
    CUDA_CHECK(cudaMemsetAsync(ctx->dOuterLoopUnstable.p, 0, nS * sizeof(char), ctx->stream));
    if (ctx->loopTraceSlot >= 0 && ctx->loopTraceSlot < nS && outerLoopCheckActive[ctx->loopTraceSlot]
            && ctx->outerLoopChecks[ctx->loopTraceSlot] == 1) {
        // OLF_GPU_SLOT_DUMP, second point: the traced scenario's state and bus balances (qBuf) at the FIRST
        // outer-loop check, i.e. after the first solve - "<prefix>.<k>.pass0", "<row> <x> <qBuf>".
        static const char* slotDump0 = traceValue(Trace::SLOT_DUMP);
        static int slotDumps0 = 0;
        if (slotDump0 != nullptr) {
            const size_t off = static_cast<size_t>(ctx->loopTraceSlot) * n;
            std::vector<double> hx(n), hq(n);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(hx.data(), ctx->dX.as<double>() + off, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hq.data(), ctx->dQbuf.as<double>() + off, n * sizeof(double), cudaMemcpyDeviceToHost));
            std::string path = std::string(slotDump0) + "." + std::to_string(slotDumps0++) + ".pass0";
            if (FILE* f = std::fopen(path.c_str(), "w")) {
                for (int r = 0; r < n; r++) {
                    std::fprintf(f, "%d %.17g %.17g\n", r, hx[r], hq[r]);
                }
                std::fclose(f);
            }
        }
    }
    // Apply ONLY the current phase's outer loop this pass (sequential, like OLF) — never both at once.
    bool inDistributedSlackPhase = (loopOrder[loopIdx] == GpuOuterLoopKind::DISTRIBUTED_SLACK);
    traceBatchPass(ctx, drv, nLoops, nChecked);
    // ---- IncrementalTransformerVoltageControl as a DRIVER PHASE (OLF order 8). OLF's own rule, per
    //      controlled bus: if |targetV - V| > halfDeadband, take dV/drho from the Jacobian at the converged
    //      state (unit RHS at the rho identity row, read the bus V entry), deltaR1 = diffV / sensitivity, move to
    //      the tap whose r1 is closest to r1 + deltaR1 within maxTapShift and the ratchet's allowed
    //      direction (MAX_DIRECTION_CHANGE 3), skip controllers with |sensitivity| < MIN_SENSI_FILTER (0.05).
    //      The sensitivity is the same forward solve GpuBatchIncrementalTvc used to make through
    //      solveBatchSensitivities — one host round trip per round — done here against the factors the
    //      driver already holds, with no host copy of the states. Single controller per controlled bus
    //      (the JNI upload refuses shared buses, which stay on the host orchestrator). ----
    // ---- IncrementalShuntVoltageControl as a DRIVER PHASE (OLF order 10): deadband per controlled bus,
    //      dV/dB from a unit RHS at each needed controller's SHUNT_B identity row (the driver's own
    //      factors), OLF's adjustB round-robin per group, the section total held at the identity row
    //      and the static (dG, dB) through the shunt-override list. Groups with several controller
    //      shunts ARE handled (the round-robin runs in the group kernel). ----
    // ---- the FLOW-OBSERVED incremental loops as DRIVER PHASES: kindMask selects the phase controllers
    //      (OLF order 7: kinds 1+2) or the RPC controllers (OLF order 9: kind 0). Same skeleton as the
    //      voltage-observed phases; the observable is the controlled branch's flow gradient dotted with
    //      the unit-RHS solution (OLF's EquationTerm.calculateSensi), evaluated by the codegen'd branch
    //      functions on the device. ----
    // ---- CONTINUOUS PhaseControl as a DRIVER PHASE (OLF order 7), see phaseControlCheckBatch. ----
    // ---- the DISCRETE voltage-control loops as DRIVER PHASES: TransformerVoltageControl (AFTER_GENERATOR /
    //      WITH_GENERATOR, OLF order 8) and ShuntVoltageControl (WITH_GENERATOR, OLF order 10). No sensitivities:
    //      OLF's decisions read the converged state only (deadband, clamp bounds, closest tap / section). ----
    // ---- SecondaryVoltageControl as a DRIVER PHASE (OLF order 4). The whole loop: the re-enable step and
    //      the per-zone trigger on the converged state's own qBuf, one unit-RHS solve per controlled bus on the
    //      driver's own factors for the K and pilot sensitivities, the small dense zone system per scenario,
    //      and the moved targets applied to the controlled buses' V rows (plus the switch-back comparison
    //      target for a controller switched to PQ). Nothing goes back to the host, so the loop sits in OLF's order
    //      instead of after the sweep. ----
    int loopKind = loopOrder[loopIdx];
    if (loopKind == GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_VOLTAGE_CONTROL) {
        incrementalTransformerVoltageControlStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::INCREMENTAL_SHUNT_VOLTAGE_CONTROL) {
        incrementalShuntVoltageControlStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::INCREMENTAL_PHASE_CONTROL) {
        if (ctx->nPhaseControlControllerBranches > 0) phaseControlStep(ctx, drv); else incrementalFlowControlStep(ctx, drv, controlledFlowMaskBit(GpuControlledFlow::ACTIVE_POWER) | controlledFlowMaskBit(GpuControlledFlow::CURRENT));  // continuous, or incremental phase
    } else if (loopKind == GpuOuterLoopKind::INCREMENTAL_TRANSFORMER_REACTIVE_POWER_CONTROL) {
        incrementalFlowControlStep(ctx, drv, controlledFlowMaskBit(GpuControlledFlow::REACTIVE_POWER));
    } else if (loopKind == GpuOuterLoopKind::TRANSFORMER_VOLTAGE_CONTROL) {
        transformerVoltageControlStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::SHUNT_VOLTAGE_CONTROL) {
        shuntVoltageControlStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::SECONDARY_VOLTAGE_CONTROL) {
        secondaryVoltageControlStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::MONITORING_VOLTAGE) {
        monitoringVoltageStep(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::FREEZING_HVDC_AC_EMULATION || loopKind == GpuOuterLoopKind::HVDC_AC_EMULATION_LIMITS) {
        hvdcStep(ctx, drv, loopKind);
    } else if (loopKind == GpuOuterLoopKind::AREA_INTERCHANGE_CONTROL) {
        areaInterchangeControlCheck(ctx, drv);
    } else if (loopKind == GpuOuterLoopKind::AUTOMATION_SYSTEM) {
        automationSystemCheck(ctx, drv);
    } else if (inDistributedSlackPhase) {                                // distributed-slack outer loop: spread the slack mismatch
        distributedSlackCheck(ctx, drv);
    } else {                                               // reactive-limits outer loop: switch PV->PQ over Q limit
        reactiveLimitsCheck(ctx, drv, loopKind);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(outerLoopUnstable.data(), ctx->dOuterLoopUnstable.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
    for (int s = 0; s < nS; s++) {
        if (loopRejected[s]) {                              // a device loop routed the scenario to the CPU fallback
            traceReject(s, "device-loop-routed", outer, 0);
            res.status[s] = GpuAcSolverStatus::REJECTED;
            live[s] = 0;
            active[s] = 0;
            outerLoopUnstable[s] = 0;
            loopRejected[s] = 0;
        }
    }
    // OLF_GPU_SW_TRACE=<slot>: every controller this RL pass CHANGED for one scenario, with the
    // quantities its decision turned on. A scenario that ends one outer pass short of the CPU's
    // trajectory did so because one of these decisions went the other way; without the list there is
    // no way to name WHICH controller, and the margin of that one decision is the whole question.
    static const char* reactiveLimitsTraceEnv = traceValue(Trace::SW_TRACE);
    static const int reactiveLimitsTraceScenarioEnv = reactiveLimitsTraceEnv != nullptr ? std::atoi(reactiveLimitsTraceEnv) : -1;
    // the contingency-id slot of OLF_GPU_LOOP_TRACE_CTG wins (chunk composition moves fixed slots)
    const int reactiveLimitsTraceScenario = ctx->loopTraceSlot >= 0 ? ctx->loopTraceSlot : reactiveLimitsTraceScenarioEnv;
    traceReactiveLimitsSwitches(ctx, drv, auditLog, reactiveLimitsAuditOn, reactiveLimitsTraceScenario);
    traceReactiveLimitsState(ctx, drv, inDistributedSlackPhase, reactiveLimitTol, outerLoopUnstable);
    bool inDistributionPhase = inDistributedSlackPhase || loopKind == GpuOuterLoopKind::AREA_INTERCHANGE_CONTROL;
    if (inDistributionPhase && (ctx->nDistributedSlackBuses > 0 || ctx->slackOnlyDistribution) && rejectSlackDistributionFailure) {
        // slackDistributionFailureBehavior != LEAVE_ON_SLACK_BUS: a scenario whose participants all
        // saturated must NOT be reported converged with the residual absorbed at the slack — route it
        // to the CPU fallback (status -2), which replays it with the run's real failure behavior.
        std::vector<char> slackDistributionFailure(nS);
        CUDA_CHECK(cudaMemcpy(slackDistributionFailure.data(), ctx->dSlackDistributionFailure.p, nS * sizeof(char), cudaMemcpyDeviceToHost));
        for (int s = 0; s < nS; s++) {
            if (slackDistributionFailure[s] && live[s]) {
                traceReject(s, "ds-residual-saturated", outer, 0);
                res.status[s] = GpuAcSolverStatus::REJECTED;
                active[s] = 0;
                live[s] = 0;
                outerLoopUnstable[s] = 0;
            }
        }
    }
}

}  // namespace olfgpu
