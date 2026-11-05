// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// See newton_iteration.cuh.
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
#include "../common/codes/gpu_state_vector_scaling_mode.cuh"
#include "../common/codes/gpu_ac_solver_status.cuh"
#include "../common/codes/gpu_linear_solve_mode.cuh"
#include "../diagnostics/driver_traces.cuh"
#include "../diagnostics/matrix_scan.cuh"
#include "../equations/fill_launch.cuh"
#include "compact_solve.cuh"
#include "convergence.cuh"
#include "driver_steps.cuh"
#include "iterative_refinement.cuh"
#include "linear_solver_settings.cuh"
#include "batched_lu_solve.cuh"
#include "realistic_voltage_check.cuh"
#include "state_vector_scaling.cuh"
#include "../fixedmatrix/zero_diagonal_repair.cuh"
#include "newton_iteration.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

// The inner Newton-Raphson of one outer pass on every active scenario: fill, factorize, solve, refine, scale and
// update until each scenario converges, diverges or exhausts its iterations (OLF NewtonRaphson.run).
void newtonSolve(BatchSolverContext* ctx, DriverPass& drv) {
    const int& n = drv.n;
    const int& nnz = drv.nnz;
    const int& nS = drv.nS;
    const size_t& nTot = drv.nTot;
    const int& outer = drv.outer;
    const double& tol = drv.tol;
    const int& maxIter = drv.maxIter;
    const int& stateVectorScalingMode = drv.stateVectorScalingMode;
    const double& convEpsPerEq = drv.convEpsPerEq;
    const bool& deviceFixedIter = drv.deviceFixedIter;
    const bool& outerLoopTiming = drv.outerLoopTiming;
    int& outerLoopPassSolvedScenarios = drv.outerLoopPassSolvedScenarios;
    int& nActive = drv.nActive;
    std::vector<char>& active = drv.active;
    std::vector<char>& live = drv.live;
    std::vector<char>& wasActive = drv.wasActive;
    std::vector<double>& hF = drv.hF;
    BatchResult& res = drv.res;

    // LINE_SEARCH per-scenario backtracking state (lastNorm2 / haveLast, per pass — a phase re-entry retargets the system, voiding the step relation)
    // Per scenario, the linear-solve residual ratio of this pass's LAST solve — scored only when the pass runs
    // its Newton budget out (see the it == maxIter branch), so a healthy run never pays for it.
    std::vector<double> lastSolveRatio(nS, 0.0);
    std::vector<double> lineSearchLastSumOfSquares(stateVectorScalingMode == GpuStateVectorScalingMode::LINE_SEARCH ? nS : 0, 0.0);
    std::vector<char> lineSearchHasLastTestResult(stateVectorScalingMode == GpuStateVectorScalingMode::LINE_SEARCH ? nS : 0, 0);
    std::vector<double> hMu(stateVectorScalingMode == GpuStateVectorScalingMode::LINE_SEARCH ? nS : 0, -1.0);

    for (int it = 0; it <= maxIter && nActive > 0; it++) {
        gLastPassIters = it;
        gNewtonIters++;
        bool fOnly = it == maxIter;
        fillActiveBatch(ctx, drv, fOnly);

        // ---- residual check + retirement ----
        // Normal mode (tol > 0): every iteration copies the batched mismatch to the host (the
        // only thing crossing PCIe) and retires converged/diverged scenarios.
        // Device fixed-iteration mode (tol <= 0): NO per-iteration host sync and NO retirement
        // — every scenario runs exactly maxIter steps on the device; the mismatch is read back
        // ONCE, on the final pass, only to report ||F||inf. This drops the per-iteration D->H
        // copy of the full batched mismatch. Tradeoff: a scenario going non-finite is not
        // isolated, so it can poison a uniform-batched factorization — use tol > 0 if
        // scenarios may diverge.
        if (!deviceFixedIter || it == maxIter) {
            std::vector<double> hNorms;
            if (!deviceFixedIter) {
                // Device-side reduction: read back nS*4 norms, not the 70-MB mismatch.
                readNormsBatch(ctx, drv, hNorms);
                // ---- LINE_SEARCH backtrack (per scenario): the full step applied at it-1 may have
                //      RAISED that scenario's ||F||2 — walk it back to x = xPrev + mu*dx with
                //      mu = stepFold^-k until the 2-norm improves, mirroring
                //      OLF's LineSearchStateVectorScaling. Judged on the 2-norm
                //      (ssq, monotonic-equivalent); a non-finite mismatch backtracks too. Scenarios
                //      not backtracking keep their state (mu sentinel -1 masks the step kernel). ----
                if (stateVectorScalingMode == GpuStateVectorScalingMode::LINE_SEARCH) {
                    auto needsBt = [&](int s2) {
                        return active[s2] && lineSearchHasLastTestResult[s2]
                                && (hNorms[s2 * 4 + 2] == 0.0 || hNorms[s2 * 4 + 1] >= lineSearchLastSumOfSquares[s2]);
                    };
                    bool anyBt = false;
                    for (int s2 = 0; s2 < nS; s2++) {
                        if (needsBt(s2)) anyBt = true;
                    }
                    for (int k = 1; anyBt && k <= ctx->lineSearchStateVectorScalingMaxIteration; k++) {
                        double mu = 1.0 / std::pow(ctx->lineSearchStateVectorScalingStepFold, k);
                        for (int s2 = 0; s2 < nS; s2++) hMu[s2] = needsBt(s2) ? mu : -1.0;
                        CUDA_CHECK(cudaMemcpyAsync(ctx->dStepFactor.p, hMu.data(), nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
                        stepFromPrevBatchMasked<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS,
                                ctx->dLineSearchPreviousX.as<double>(), ctx->dDx.as<double>(), ctx->dStepFactor.as<double>(), ctx->dX.as<double>());
                        fillActiveBatch(ctx, drv, false);              // full fill: J must sit at the accepted x (like the seq path)
                        readNormsBatch(ctx, drv, hNorms);
                        anyBt = false;
                        for (int s2 = 0; s2 < nS; s2++) {
                            if (needsBt(s2)) anyBt = true;
                        }
                    }
                    for (int s2 = 0; s2 < nS; s2++) {
                        if (active[s2]) {
                            lineSearchLastSumOfSquares[s2] = hNorms[s2 * 4 + 1];
                            lineSearchHasLastTestResult[s2] = 1;
                        }
                    }
                }
            } else {
                CUDA_CHECK(cudaMemcpy(hF.data(), ctx->dF.p, nTot * sizeof(double), cudaMemcpyDeviceToHost));
            }
            // A pass about to report exhaustion: score its LAST linear solve (dVals/dDx/dB still hold it - the
            // it == maxIter fill is F-only). A wrong factorization sits at 1e5..1e13, a sound solve at 1e-9, so this
            // separates "OLF's budget, spent moving" from "the step was junk" without the per-pass cost of scoring
            // every pass (measured: 34 s on a 19 s rte6515 run).
            if (it == maxIter && !deviceFixedIter && nActive > 0) {
                if (!ctx->dLinRp.p) {
                    ctx->dLinRp.alloc((size_t) n * nS * sizeof(double));
                    ctx->dLinRm.alloc((size_t) n * nS * sizeof(double));
                    ctx->dLinSc.alloc((size_t) 3 * nS * sizeof(double));
                }
                launchLinResidRows(ctx->stream, n, nS, nnz, ctx->structure->dRowPtr.as<int>(),
                        ctx->structure->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->layout, ctx->dDx.as<double>(),
                        ctx->dB.as<double>(), ctx->dLinRp.as<double>(), ctx->dLinRm.as<double>());
                linResidScoreBatch<<<nS, LANE_NORM_BLOCK, 0, ctx->stream>>>(n, nS, ctx->dLinRp.as<double>(),
                        ctx->dLinRm.as<double>(), ctx->dF.as<double>(), ctx->dLinSc.as<double>());
                std::vector<double> hsc((size_t) 3 * nS);
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                CUDA_CHECK(cudaMemcpy(hsc.data(), ctx->dLinSc.p, (size_t) 3 * nS * sizeof(double), cudaMemcpyDeviceToHost));
                for (int s2 = 0; s2 < nS; s2++) {
                    lastSolveRatio[s2] = hsc[3 * s2 + 2];
                }
            }
            bool anyChanged = false;
            // Inner-Newton convergence test. Default (convEpsPerEq <= 0): the device infinity-norm floor
            // ‖F‖∞ < tol (stricter than OLF's usual criterion). When convEpsPerEq > 0: OLF's EXACT criterion
            // — DefaultNewtonRaphsonStoppingCriteria stops at ‖F‖₂ < sqrt(convEpsPerEq² · n) — so a batched SA
            // can be run with the SAME stopping decision as the CPU loadflow for an apples-to-apples timing
            // comparison. ‖F‖∞ is still recorded in res.norm[s] for reporting either way. n counts the padded
            // system rows; for batched scenarios (no islanding — those go to the CPU) that equals OLF's active
            // equation count, since disabled-branch rows keep the bus equations and identity pads contribute 0.
            const double olf2NormThreshold = convEpsPerEq > 0.0 ? std::sqrt(convEpsPerEq * convEpsPerEq * n) : 0.0;
            auto norm2ThresholdOf = [&](int sIdx) {
                return convEpsPerEq > 0.0 && !ctx->hNActive.empty()
                        ? std::sqrt(convEpsPerEq * convEpsPerEq * ctx->hNActive[sIdx]) : olf2NormThreshold;
            };
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                double norm = 0.0;
                double ssq = 0.0;
                bool finite = true;
                double maxRatio = 0.0;
                if (!deviceFixedIter) {
                    norm = hNorms[(size_t) s * 4];
                    ssq = hNorms[(size_t) s * 4 + 1];
                    finite = hNorms[(size_t) s * 4 + 2] != 0.0;
                    maxRatio = hNorms[(size_t) s * 4 + 3];
                } else {
                    for (int i = 0; i < n; i++) {
                        double v = hF[(size_t) s * n + i];
                        if (!std::isfinite(v)) {
                            finite = false;
                            break;
                        }
                        norm = std::max(norm, std::fabs(v));
                        ssq += v * v;
                    }
                }
                // PER_EQUATION_TYPE (per-row thresholds set on the context): OLF's
                // PerEquationTypeStoppingCriteria — stop when EVERY row clears its type threshold.
                bool converged = ctx->perRowCriteria ? (finite && maxRatio < 1.0)
                        : convEpsPerEq > 0.0 ? (std::sqrt(ssq) < norm2ThresholdOf(s)) : (norm < tol);
                static const char* sdEnv = traceValue(Trace::STATE_DUMP);
                if (sdEnv != nullptr && s == 0 && it <= std::atoi(sdEnv) && n <= 64) {
                    stateDumpBatch<<<1, 1, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dMode.as<int>(),
                            ctx->dTarget.as<double>(), ctx->dX.as<double>(), s, it);
                    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                }
                if (traceOn(Trace::BATCH_TRACE)) {
                    std::fprintf(stderr, "BATCH_TRACE s=%d it=%d infnorm=%.17e norm2=%.17e thr=%.17e conv=%d\n",
                                 s, it, norm, std::sqrt(ssq), norm2ThresholdOf(s), converged ? 1 : 0);
                }
                if (s == ctx->loopTraceSlot) {
                    // The traced scenario's Newton step, host-side: the norms the convergence test saw,
                    // the MAX_VOLTAGE_CHANGE factor the step was scaled by, and the row that carries the
                    // largest residual with what sits on it. OLF's NR_EXIT prints the same norms after
                    // each of ITS iterations; a different iteration COUNT on the same system is where
                    // two converged answers part (network.xiidm/IZUZ: OLF 1 iteration, device 2).
                    std::vector<double> hF(n), hT(n), hX(n);
                    std::vector<int> hM(n);
                    double fac = 1.0;
                    CUDA_CHECK(cudaMemcpy(hF.data(), ctx->dF.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(hT.data(), ctx->dTarget.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(hX.data(), ctx->dX.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(hM.data(), ctx->dMode.as<int>() + (size_t) s * n, n * sizeof(int), cudaMemcpyDeviceToHost));
                    if (ctx->clampStep && ctx->dStepFactor.p) {
                        CUDA_CHECK(cudaMemcpy(&fac, ctx->dStepFactor.as<double>() + s, sizeof(double), cudaMemcpyDeviceToHost));
                    }
                    int wr = 0;
                    for (int r = 1; r < n; r++) if (std::fabs(hF[r]) > std::fabs(hF[wr])) wr = r;
                    // ...and the RAW step's largest entry (dDx still holds it), which is what the
                    // MAX_VOLTAGE_CHANGE factor was computed from
                    std::vector<double> hDx(n);
                    CUDA_CHECK(cudaMemcpy(hDx.data(), ctx->dDx.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    int wd = 0;
                    for (int r = 1; r < n; r++) if (std::fabs(hDx[r]) > std::fabs(hDx[wd])) wd = r;
                    std::fprintf(stderr, "SLOT_TRACE s=%d it=%d infnorm=%.17e norm2=%.17e thr=%.17e conv=%d stepFactor=%.9f worstRow=%d F=%.12e mode=%d target=%.9f x=%.9f maxDxRow=%d dx=%.9f\n",
                                 s, it, norm, std::sqrt(ssq), norm2ThresholdOf(s), converged ? 1 : 0, fac, wr, hF[wr], hM[wr], hT[wr], hX[wr], wd, hDx[wd]);
                }
                res.iters[s] = it;
                res.norm[s] = norm;
                if (deviceFixedIter) {
                    res.status[s] = finite ? GpuAcSolverStatus::MAX_ITERATION_REACHED : GpuAcSolverStatus::DIVERGED;  // ran maxIter; no convergence test
                } else if (!finite) {                        // a diverged scenario must not poison the batch
                    res.status[s] = GpuAcSolverStatus::DIVERGED;
                    active[s] = 0;
                    live[s] = 0;                             // failed: stop iterating its outer loops
                    nActive--;
                    anyChanged = true;
                } else if (converged && it > 0 && it < maxIter) {
                    // Force >=1 iteration, like CPU NewtonRaphson: a warm-started
                    // state may already meet the global 2-norm threshold yet leave a just-flipped identity row
                    // (PV target) unmet — one Newton step meets it. Retiring at it==0 skips that solve.
                    // And NOT at the budget: OLF's NewtonRaphson.run counts a step before testing it and
                    // reports MAX_ITERATION_REACHED whenever iterations >= maxIterations, so a test that
                    // passes after the LAST allowed step is not CONVERGED there — the it == maxIter branch
                    // below reports it at OLF's budget (with a sound last solve).
                    res.status[s] = GpuAcSolverStatus::CONVERGED;
                    active[s] = 0;
                    nActive--;
                    anyChanged = true;
                    CUDA_CHECK(cudaMemcpyAsync(ctx->dXLastConv.as<double>() + (size_t) s * n, ctx->dX.as<double>() + (size_t) s * n,
                                               n * sizeof(double), cudaMemcpyDeviceToDevice, ctx->stream));
                } else if (it == maxIter) {
                    // exhausted iterations. With fixed inner iterations those are the K steps asked for, not OLF's
                    // Newton budget, so the scenario is REJECTED (the CPU replays it) rather than reported
                    // MAX_ITERATION_REACHED from the device.
                    //
                    // A BAD LAST SOLVE is a reject, not OLF's answer: the pass ran its whole budget on steps the
                    // factorization got wrong (rte6515 busbar sections [5000,6000): 885 scenarios sat at
                    // ||F||inf ~ 56 for 30 iterations, reported as MAX_ITERATION_REACHED, 10,067 violations
                    // against the CPU's 71,553). Reported max_iter has to mean "OLF's budget, spent on sound
                    // steps". On the BATCH OF ONE (the base load flow) -3 takes the pivoted device retry first;
                    // a chunk scenario replays on the CPU either way (needsCpuFallback covers both).
                    bool badSolve = !(lastSolveRatio[s] <= LIN_SOLVE_FAIL_RATIO);
                    if (badSolve) {
                        traceReject(s, "max-iter-bad-solve", outer, it);
                        if (traceOn(Trace::REJECT_TRACE)) {
                            std::fprintf(stderr, "GPU_REJECT slot=%d reason=max-iter-bad-solve ratio=%.3e norm=%.6e\n",
                                         s, lastSolveRatio[s], norm);
                        }
                    }
                    res.status[s] = badSolve ? (nS == 1 ? GpuAcSolverStatus::LINEAR_SOLVE_FAILED : GpuAcSolverStatus::REJECTED)
                                            : GpuAcSolverStatus::MAX_ITERATION_REACHED;
                    active[s] = 0;
                    live[s] = 0;                             // did not converge: stop iterating its outer loops
                    nActive--;
                    anyChanged = true;
                }
            }
            if (!deviceFixedIter && anyChanged) {
                CUDA_CHECK(cudaMemcpy(ctx->dActive.p, active.data(), nS * sizeof(char), cudaMemcpyHostToDevice));
            }
        }
        if (deviceFixedIter) {
            if (it == maxIter) {
                break;                                       // final residual recorded; done
            }
        } else if (nActive == 0) {
            break;
        }

        // Scenarios still being FACTORIZED this iteration. Summed over iterations this is the batch's
        // real work measure: the CPU pays each contingency its own iteration count, so a batch is only
        // as efficient as its retirement — a batch that keeps every scenario alive until the last
        // straggler converges pays the WORST case for all of them.
        gActiveIterSum += nActive;
        if (stateVectorScalingMode == GpuStateVectorScalingMode::LINE_SEARCH) {                                   // LINE_SEARCH: remember the pre-step iterates
            CUDA_CHECK(cudaMemcpyAsync(ctx->dLineSearchPreviousX.p, ctx->dX.p, static_cast<size_t>(n) * nS * sizeof(double),
                                       cudaMemcpyDeviceToDevice, ctx->stream));
        }
        if (ctx->mode == GpuLinearSolveMode::UBATCH || ctx->mode == GpuLinearSolveMode::BATCHED_LU) {
        // ---- UBATCH solve: ANALYZE one block once (cheap), then a linear solve over all nS systems
        //      via CUDSS_CONFIG_UBATCH_SIZE on the single per-block matrix, FACTORIZED every iteration
        //      (honest Newton). Our batched LU (mode 7) factorizes the active scenarios only; the inactive ones
        //      get the same identity systems, so every buffer holds what it holds under cuDSS. ----
        writeIdentityForInactive<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->structure->dRowPtr.as<int>(),
                ctx->structure->dDiagPos.as<int>(), ctx->dActive.as<char>(), ctx->dIdentityLane.as<char>(),
                ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->layout);
        markIdentityLanes<<<grid(nS), BLOCK, 0, ctx->stream>>>(nS, ctx->dActive.as<char>(), ctx->dIdentityLane.as<char>());
        negateIntoBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dF.as<double>(), ctx->dB.as<double>());
        repairZeroDiagonalRightHandSide(ctx, nS, ctx->dB.as<double>());
        if (ctx->mode == GpuLinearSolveMode::BATCHED_LU) {
            batchedLuNewtonSolve(ctx, drv, it);
        } else {
        if (!ctx->analyzed) {
            bool timeIt = traceOn(Trace::TRACE);
            long long t0 = 0;
            if (timeIt) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                t0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
            }
            gAnalysesMain++;
            { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_ANALYSIS, ctx->solver.config, ctx->solver.data, ctx->uniformBatchSystemMatrix, ctx->uniformBatchSolutionMatrix, ctx->uniformBatchRightHandSideMatrix)); }
            double ms = -1.0;
            if (timeIt) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                ms = (std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count() - t0) / 1.0e6;
            }
            traceLuNnz(ctx->solver.handle, ctx->solver.data, "ubatch", ctx->nScenarios, ms);
            ctx->analyzed = true;
        }
        // ONE cudssExecute per iteration (phases are OR-able bit flags): halves the per-iteration
        // host-side phase overhead vs separate FACTORIZATION + SOLVE calls.
        int ubPhase = CUDSS_PHASE_SOLVE | CUDSS_PHASE_FACTORIZATION;
        // ---- solve-only COMPACTION: when few scenarios remain active, gather their vals/B into
        //      compact buffers and solve a NARROWER ubatch on the dedicated dsCompacted context, then
        //      scatter dx back. mode 3 only (factorizes every iteration — stateless across width
        //      changes); width quantized to power-of-2 tiers so the cheap single-block re-ANALYSIS
        //      runs only on tier change. Pad slots duplicate the first active scenario (keeps the
        //      batch non-singular) and are discarded by the scatter. dDx is zeroed first so
        //      inactive scenarios' frozen states are not perturbed by stale solutions (the
        //      full-width update kernel adds dx over ALL scenarios). ----
        // Compact once at most half of the chunk is active. Above it every pass factorizes the full width, padded
        // lanes included (network.xiidm: 278,144 lane-iterations for 234,088 useful). Compact widths are powers of
        // two, so a tier change re-analyzes rarely.
        constexpr double COMPACT_FRAC = 0.5;
        const int compactLimit = std::max(1, std::min(nS, static_cast<int>(COMPACT_FRAC * nS)));
        bool compactSolve = !deviceFixedIter && ctx->mode == GpuLinearSolveMode::UBATCH && nActive > 0 && nActive <= compactLimit
                && nActive < nS;
        if (!compactSolve) {
            gWidthIterSum += nS;                             // full-width factorization this iteration
        }
        if (compactSolve) {
            if (ctx->compactedCapacity == 0) {                       // lazy first-engage setup
                ctx->compactedCapacity = std::max(1, compactLimit);
                ctx->dValsCompacted.alloc(static_cast<size_t>(nnz) * ctx->compactedCapacity * sizeof(double));
                ctx->dBCompacted.alloc(static_cast<size_t>(n) * ctx->compactedCapacity * sizeof(double));
                ctx->dDxCompacted.alloc(static_cast<size_t>(n) * ctx->compactedCapacity * sizeof(double));
                ctx->dCompactedScenarioMap.alloc(static_cast<size_t>(ctx->compactedCapacity) * sizeof(int));
                CUDSS_CHECK(cudssCreate(&ctx->compactedSolver.handle));
                CUDSS_CHECK(cudssSetStream(ctx->compactedSolver.handle, ctx->stream));
                CUDSS_CHECK(cudssConfigCreate(&ctx->compactedSolver.config));
                applyReorderAlg(ctx->compactedSolver.config, ctx->structure->nZeroImpedanceBranch > 0);
                int compactedIterativeRefinementSteps = ctx->structure->nZeroImpedanceBranch > 0 ? configuredIterativeRefinementBaseSteps() : ITERATIVE_REFINEMENT_STEPS_WITHOUT_COUPLERS;
                CUDSS_CHECK(cudssConfigSet(ctx->compactedSolver.config, CUDSS_CONFIG_IR_N_STEPS, &compactedIterativeRefinementSteps, sizeof(compactedIterativeRefinementSteps)));
                CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->compactedUniformBatchSystemMatrix, n, n, nnz,
                        ctx->structure->dRowPtr.as<int>(), nullptr, ctx->structure->dColIdx.as<int>(), ctx->dValsCompacted.as<double>(),
                        OLF_CUDSS_CSR_TYPES, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->compactedUniformBatchRightHandSideMatrix, n, 1, n, ctx->dBCompacted.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->compactedUniformBatchSolutionMatrix, n, 1, n, ctx->dDxCompacted.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
                ctx->dRCompacted.alloc(static_cast<size_t>(n) * ctx->compactedCapacity * sizeof(double));
                ctx->dDeltaCompacted.alloc(static_cast<size_t>(n) * ctx->compactedCapacity * sizeof(double));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->compactedUniformBatchResidualMatrix, n, 1, n, ctx->dRCompacted.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->compactedUniformBatchDeltaMatrix, n, 1, n, ctx->dDeltaCompacted.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
            }
            std::vector<int> cmap;
            cmap.reserve(ctx->compactedCapacity);
            for (int cs = 0; cs < nS; cs++) {
                if (active[cs]) {
                    cmap.push_back(cs);
                }
            }
            int nReal = static_cast<int>(cmap.size());
            int w = 1;
            while (w < nReal) {
                w <<= 1;
            }
            if (w > ctx->compactedCapacity) {
                w = ctx->compactedCapacity;
            }
            while (static_cast<int>(cmap.size()) < w) {
                cmap.push_back(cmap[0]);
            }
            gWidthIterSum += w;                              // compacted width actually factorized
            gCompactIters++;
            CUDA_CHECK(cudaMemcpyAsync(ctx->dCompactedScenarioMap.p, cmap.data(), w * sizeof(int), cudaMemcpyHostToDevice, ctx->stream));
            gatherScenarios<<<grid(nnz * w), BLOCK, 0, ctx->stream>>>(nnz, w, ctx->dCompactedScenarioMap.as<int>(), ctx->dVals.as<double>(), ctx->dValsCompacted.as<double>());
            gatherScenarios<<<grid(n * w), BLOCK, 0, ctx->stream>>>(n, w, ctx->dCompactedScenarioMap.as<int>(), ctx->dB.as<double>(), ctx->dBCompacted.as<double>());
            // A new batch size needs a FRESH data object and a fresh ANALYSIS. Setting
            // CUDSS_CONFIG_UBATCH_SIZE on the existing data and skipping the analysis is accepted by
            // cuDSS and even looks faster on a bench -- measured 2026-09-16 at -11% warm with exact
            // violation parity -- and then fails with an illegal memory access on a bigger network:
            // the data object's buffers are sized for the batch size it was analyzed at. The analysis
            // is cheap (~64 ms) and costs the same at any batch size, but its STATE belongs to one.
            cudssData_t compactData = nullptr;
            {
                // Tier change to a width analyzed before: its own data object, the batch size set back on the
                // (shared) config. A new width: a fresh data object and its single-block ANALYSIS, kept.
                cudssData_t& cached = ctx->compactData[w];
                if (cached == nullptr) {
                    CUDSS_CHECK(cudssDataCreate(ctx->compactedSolver.handle, &cached));
                    CUDSS_CHECK(cudssConfigSet(ctx->compactedSolver.config, CUDSS_CONFIG_UBATCH_SIZE, &w, sizeof(w)));
                    gAnalysesCompact++;
                    { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->compactedSolver.handle, CUDSS_PHASE_ANALYSIS, ctx->compactedSolver.config, cached, ctx->compactedUniformBatchSystemMatrix, ctx->compactedUniformBatchSolutionMatrix, ctx->compactedUniformBatchRightHandSideMatrix)); }
                } else if (w != ctx->compactWidth) {
                    CUDSS_CHECK(cudssConfigSet(ctx->compactedSolver.config, CUDSS_CONFIG_UBATCH_SIZE, &w, sizeof(w)));
                }
                ctx->compactWidth = w;
                compactData = cached;
            }
            { AnalysisTimer at_(__LINE__, 'F'); CUDSS_CHECK(cudssExecute(ctx->compactedSolver.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, ctx->compactedSolver.config, compactData, ctx->compactedUniformBatchSystemMatrix, ctx->compactedUniformBatchSolutionMatrix, ctx->compactedUniformBatchRightHandSideMatrix)); }
            if (ctx->structure->nZeroImpedanceBranch > 0 && ctx->dRCompacted.p != nullptr && ctx->dIterativeRefinementScore.p != nullptr && ctx->dIterativeRefinementScore.cap >= static_cast<size_t>(w) * sizeof(double)) {
                // adaptive refinement on the compacted lanes (all w are real or a duplicate of a real one)
                refineLanes(ctx, w, n, nnz, ctx->dValsCompacted.as<double>(), ctx->dDxCompacted.as<double>(), ctx->dBCompacted.as<double>(),
                        ctx->dRCompacted.as<double>(), ctx->dDeltaCompacted.as<double>(), nullptr, configuredIterativeRefinementBaseSteps(), [&]() {
                    AnalysisTimer at_(__LINE__, 'S');
                    CUDSS_CHECK(cudssExecute(ctx->compactedSolver.handle, CUDSS_PHASE_SOLVE, ctx->compactedSolver.config, compactData, ctx->compactedUniformBatchSystemMatrix, ctx->compactedUniformBatchDeltaMatrix, ctx->compactedUniformBatchResidualMatrix));
                });
            }
            CUDA_CHECK(cudaMemsetAsync(ctx->dDx.p, 0, static_cast<size_t>(n) * nS * sizeof(double), ctx->stream));
            scatterScenarios<<<grid(n * nReal), BLOCK, 0, ctx->stream>>>(n, nReal, ctx->dCompactedScenarioMap.as<int>(), ctx->dDxCompacted.as<double>(), ctx->dDx.as<double>());
        } else {
            { AnalysisTimer at_(__LINE__, ((ubPhase) & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) ? 'G' : ((ubPhase) & CUDSS_PHASE_ANALYSIS) ? 'A' : 'S'); CUDSS_CHECK(cudssExecute(ctx->solver.handle, ubPhase, ctx->solver.config, ctx->solver.data, ctx->uniformBatchSystemMatrix, ctx->uniformBatchSolutionMatrix, ctx->uniformBatchRightHandSideMatrix)); }
            if (ctx->structure->nZeroImpedanceBranch > 0 && ctx->dR.p != nullptr && ctx->uniformBatchResidualMatrix != nullptr && ctx->dIterativeRefinementScore.p != nullptr) {
                // adaptive refinement on the full-width UBATCH solve (see refineLanes)
                refineLanes(ctx, nS, n, nnz, ctx->dVals.as<double>(), ctx->dDx.as<double>(), ctx->dB.as<double>(),
                        ctx->dR.as<double>(), ctx->dDelta.as<double>(), ctx->dActive.as<char>(), configuredIterativeRefinementBaseSteps(), [&]() {
                    AnalysisTimer at_(__LINE__, 'S');
                    CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_SOLVE, ctx->solver.config, ctx->solver.data, ctx->uniformBatchSystemMatrix, ctx->uniformBatchDeltaMatrix, ctx->uniformBatchResidualMatrix));
                });
            }
        }
        }   // cuDSS
        // LINEAR-SOLVE GUARD, first iteration of the first pass of every batched solve: a
        // factorization the UNPIVOTED uniform batch got wrong shows as a linear residual ratio of
        // 1e6..1e13 (rte6515's full busbar scope, 366 retained couplers: min pivots of 1e-18 under
        // any unpivoted ordering) while a sound one sits at 1e-9 (repaired rows, IR=2) and the hardest
        // measured case at 4e-4 (IR=0). Newton then never moves and the scenario surfaced as an
        // honest-looking MAX_ITERATION_REACHED; for the batch of ONE that is the BASE load flow of a
        // whole security analysis, which then reported NOTHING (6876 sections, GPU={}). Reject it
        // here (status -2): the base throws into the CPU fallback, a chunk scenario replays on the
        // CPU - never wrong, only un-accelerated. One SpMV per scenario, once per solve. Checked at
        // the SECOND iteration of the first pass: a first step from the flat start can be garbage
        // on a row that the first refactorization then heals (deviceRemoteSingletonMemberSwitchToPq:
        // ratio 0.17, dx 2e13, converged to 1e-12 all the same), while a wrong factorization stays
        // at 1e6..1e13 on every iteration.
        // Pass 0 only: two n*nS kernels and a sync per pass cost 34 s of a 19 s rte6515 run (measured), and a
        // factorization that goes wrong LATER - after a reactive-limits PV -> PQ switch rewrites its rows - is caught by the
        // no-progress reject at the end of that pass instead (see the it == maxIter branch).
        if (outer == 0 && it == 1 && !deviceFixedIter) {   // it=1: a flat-start first step may be garbage and heal (singleton-member fixture); a wrong factorization never moves
            if (!ctx->dLinRp.p) {
                ctx->dLinRp.alloc((size_t) n * nS * sizeof(double));
                ctx->dLinRm.alloc((size_t) n * nS * sizeof(double));
                ctx->dLinSc.alloc((size_t) 3 * nS * sizeof(double));
            }
            DeviceBuf& rp = ctx->dLinRp;
            DeviceBuf& rm = ctx->dLinRm;
            DeviceBuf& sc = ctx->dLinSc;
            launchLinResidRows(ctx->stream, n, nS, nnz, ctx->structure->dRowPtr.as<int>(),
                    ctx->structure->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->layout, ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), rp.as<double>(), rm.as<double>());
            linResidScoreBatch<<<nS, LANE_NORM_BLOCK, 0, ctx->stream>>>(n, nS, rp.as<double>(), rm.as<double>(),
                    ctx->dF.as<double>(), sc.as<double>());
            std::vector<double> hsc((size_t) 3 * nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(hsc.data(), sc.p, (size_t) 3 * nS * sizeof(double), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                double ratio = hsc[3 * s + 2];
                if (!(ratio <= LIN_SOLVE_FAIL_RATIO)) {         // NaN counts as failed
                    if (traceOn(Trace::REJECT_TRACE)) {
                        std::fprintf(stderr, "GPU_REJECT slot=%d reason=linear-solve ratio=%.3e ||J.dx-B||inf=%.3e ||F||inf=%.3e\n",
                                     s, ratio, hsc[3 * s], hsc[3 * s + 1]);
                    }
                    // OLF_GPU_DX_DUMP_ON_REJECT=<dir>: the rejected slot's system as the guard scored it, for a reject
                    // that only shows in a full run (the rte6515 busbar flap: one -3 per run on a different section
                    // each time, the identical ratio 3.06e+06 every time, never reproduced alone). The dump happens
                    // AFTER the verdict, so it cannot perturb what it catches.
                    static const char* rejectDumpDir = traceValue(Trace::DX_DUMP_ON_REJECT);
                    static int rejectDumps = 0;
                    if (rejectDumpDir != nullptr && rejectDumps < 8) {
                        std::string p = std::string(rejectDumpDir) + "/reject-" + std::to_string(rejectDumps++)
                                + "-slot" + std::to_string(s) + ".dump";
                        dumpSlotSystem(ctx, n, nnz, s, p.c_str());
                    }
                    traceReject(s, "linear-solve", outer, it);
                    res.status[s] = GpuAcSolverStatus::LINEAR_SOLVE_FAILED;  // the one reject a pivoted retry can fix
                    res.iters[s] = it;
                    active[s] = 0;
                    live[s] = 0;
                    nActive--;
                }
            }
        }
        if (ctx->clampStep) {
            maxVoltageChangeFactorBatch<<<nS, BLOCK, 0, ctx->stream>>>(n, ctx->dDx.as<double>(), ctx->dVarType.as<int>(), ctx->dActive.as<char>(), ctx->maxDv, ctx->maxDphi, ctx->dStepFactor.as<double>());
            axpy1BatchScaled<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS, ctx->dDx.as<double>(), ctx->dStepFactor.as<double>(), ctx->dX.as<double>());
        } else {
            axpy1Batch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dDx.as<double>(), ctx->dX.as<double>());
        }
        if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: no per-iter host sync (final readback syncs)
        } else {
        // ---- fallback: analyze once (shared pattern), refactorize + solve per scenario ----
        for (int s = 0; s < nS; s++) {
            if (!active[s]) continue;
            double* valsS = ctx->dVals.as<double>() + (size_t) s * nnz;
            double* bS = ctx->dB.as<double>() + (size_t) s * n;
            double* dxS = ctx->dDx.as<double>() + (size_t) s * n;
            double* fS = ctx->dF.as<double>() + (size_t) s * n;
            negateIntoOne<<<grid(n), BLOCK, 0, ctx->stream>>>(n, fS, bS);
            repairZeroDiagonalRightHandSide(ctx, 1, bS, s);
            CUDSS_CHECK(cudssMatrixSetValues(ctx->solver.systemMatrix, valsS));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->solver.rightHandSideMatrix, bS));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->solver.solutionMatrix, dxS));
            if (!ctx->analyzed) {
                gAnalysesMain++;
                { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_ANALYSIS, ctx->solver.config, ctx->solver.data, ctx->solver.systemMatrix, ctx->solver.solutionMatrix, ctx->solver.rightHandSideMatrix)); }
                traceLuNnz(ctx->solver.handle, ctx->solver.data, "per-scenario", ctx->nScenarios);
                ctx->analyzed = true;
            }
            { AnalysisTimer at_(__LINE__, 'F'); CUDSS_CHECK(cudssExecute(ctx->solver.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                    ctx->solver.config, ctx->solver.data, ctx->solver.systemMatrix, ctx->solver.solutionMatrix, ctx->solver.rightHandSideMatrix)); }
            double* xS = ctx->dX.as<double>() + (size_t) s * n;
            if (ctx->clampStep) {                            // OLF's MAX_VOLTAGE_CHANGE scaling, like every other mode:
                // unclamped, the robust mode took full Newton steps and its base load flow on network.xiidm's
                // busbar scope settled 0.32 kV away from OLF's
                maxVoltageChangeFactorBatch<<<1, BLOCK, 0, ctx->stream>>>(n, dxS, ctx->dVarType.as<int>(), nullptr,
                        ctx->maxDv, ctx->maxDphi, ctx->dStepFactor.as<double>() + s);
                axpy1BatchScaled<<<grid(n), BLOCK, 0, ctx->stream>>>(n, n, dxS, ctx->dStepFactor.as<double>() + s, xS);
            } else {
                axpy1One<<<grid(n), BLOCK, 0, ctx->stream>>>(n, dxS, xS);
            }
        }
        if (!deviceFixedIter) CUDA_CHECK(cudaStreamSynchronize(ctx->stream));   // fixed-iter: no per-iter host sync
        }
    }   // ---- end inner Newton loop ----
    if (outerLoopTiming) {
        int nSolved = 0;
        for (int s = 0; s < nS; s++) nSolved += wasActive[s] ? 1 : 0;
        outerLoopPassSolvedScenarios = nSolved;
    }
    for (int s = 0; s < nS; s++) {                           // per-scenario Newton work by pass (cheap, host)
        if (wasActive[s]) {
            if (ctx->outerLoopSolvePasses[s] == 0) ctx->outerLoopFirstSolveIterations[s] = res.iters[s];
            ctx->outerLoopIterations[s] += res.iters[s];
            ctx->outerLoopSolvePasses[s]++;
        }
    }
}

}  // namespace olfgpu
