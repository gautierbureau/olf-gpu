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
#include "../diagnostics/driver_traces.cuh"
#include "../diagnostics/matrix_scan.cuh"
#include "../equations/fill_launch.cuh"
#include "compact_solve.cuh"
#include "convergence.cuh"
#include "driver_steps.cuh"
#include "iterative_refinement.cuh"
#include "linear_solver_settings.cuh"
#include "realistic_voltage_check.cuh"
#include "state_vector_scaling.cuh"
#include "zero_diagonal_repair.cuh"
#include "newton_iteration.cuh"

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
    const int& lsMode = drv.lsMode;
    const double& convEpsPerEq = drv.convEpsPerEq;
    const bool& deviceFixedIter = drv.deviceFixedIter;
    const bool& iterLog = drv.iterLog;
    const bool& olTime = drv.olTime;
    int& olPassSolved = drv.olPassSolved;
    int& nActive = drv.nActive;
    std::vector<char>& active = drv.active;
    std::vector<char>& live = drv.live;
    std::vector<char>& wasActive = drv.wasActive;
    std::vector<double>& hF = drv.hF;
    std::vector<std::string>& iterLogLines = drv.iterLogLines;
    BatchResult& res = drv.res;

    // LINE_SEARCH per-scenario backtracking state (lastNorm2 / haveLast, per pass — a phase re-entry retargets the system, voiding the step relation)
    // Per scenario, the linear-solve residual ratio of this pass's LAST solve — scored only when the pass runs
    // its Newton budget out (see the it == maxIter branch), so a healthy run never pays for it.
    std::vector<double> lastSolveRatio(nS, 0.0);
    std::vector<double> lsLastSsq(lsMode == 1 ? nS : 0, 0.0);
    std::vector<char> lsHaveLast(lsMode == 1 ? nS : 0, 0);
    std::vector<double> hMu(lsMode == 1 ? nS : 0, -1.0);

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
                if (lsMode == 1) {
                    auto needsBt = [&](int s2) {
                        return active[s2] && lsHaveLast[s2]
                                && (hNorms[s2 * 4 + 2] == 0.0 || hNorms[s2 * 4 + 1] >= lsLastSsq[s2]);
                    };
                    bool anyBt = false;
                    for (int s2 = 0; s2 < nS; s2++) {
                        if (needsBt(s2)) anyBt = true;
                    }
                    for (int k = 1; anyBt && k <= ctx->lsMaxIter; k++) {
                        double mu = 1.0 / std::pow(ctx->lsStepFold, k);
                        for (int s2 = 0; s2 < nS; s2++) hMu[s2] = needsBt(s2) ? mu : -1.0;
                        CUDA_CHECK(cudaMemcpyAsync(ctx->dStepFactor.p, hMu.data(), nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
                        stepFromPrevBatchMasked<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS,
                                ctx->dXprevLs.as<double>(), ctx->dDx.as<double>(), ctx->dStepFactor.as<double>(), ctx->dX.as<double>());
                        fillActiveBatch(ctx, drv, false);              // full fill: J must sit at the accepted x (like the seq path)
                        readNormsBatch(ctx, drv, hNorms);
                        anyBt = false;
                        for (int s2 = 0; s2 < nS; s2++) {
                            if (needsBt(s2)) anyBt = true;
                        }
                    }
                    for (int s2 = 0; s2 < nS; s2++) {
                        if (active[s2]) {
                            lsLastSsq[s2] = hNorms[s2 * 4 + 1];
                            lsHaveLast[s2] = 1;
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
                linResidRowBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                        ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                        ctx->dB.as<double>(), ctx->dLinRp.as<double>(), ctx->dLinRm.as<double>());
                linResidScoreBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dLinRp.as<double>(),
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
            auto thr2Of = [&](int sIdx) {
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
                        : convEpsPerEq > 0.0 ? (std::sqrt(ssq) < thr2Of(s)) : (norm < tol);
                if (iterLog) {
                    char line[192];
                    std::snprintf(line, sizeof line, "ITER_LOG outer=%d s=%d it=%d norm2=%.17e thr2=%.17e conv=%d",
                                  outer, s, it, std::sqrt(ssq), thr2Of(s), converged ? 1 : 0);
                    iterLogLines.emplace_back(line);
                }
                // OLF_GPU_CONV_TRACE=<slot>|all: the stopping DECISION itself - the 2-norm, the
                // threshold it is held against and the active row count behind that threshold. Two
                // arms that agree on every residual and still stop one Newton step apart differ
                // here and nowhere else, and the exit norms alone cannot show it.
                static const char* convTraceEnv = std::getenv("OLF_GPU_CONV_TRACE");
                static const int convTraceSlot = convTraceEnv == nullptr ? -1
                        : (std::string(convTraceEnv) == "all" ? -2 : std::atoi(convTraceEnv));
                if (convTraceSlot == -2 || convTraceSlot == s) {
                    std::fprintf(stderr, "CONV_TRACE s=%d it=%d norm2=%.17e thr2=%.17e nActive=%d inf=%.17e perRow=%d maxRatio=%.6e converged=%d\n",
                                 s, it, std::sqrt(ssq), thr2Of(s), ctx->hNActive.empty() ? n : ctx->hNActive[s], norm,
                                 ctx->perRowCriteria ? 1 : 0, maxRatio, converged ? 1 : 0);
                    // the eight worst rows, by index: WHERE the residual sits, so a step that lands
                    // two orders of magnitude from the other arm's can be traced to its rows
                    std::vector<double> hFs(n);
                    CUDA_CHECK(cudaMemcpy(hFs.data(), ctx->dF.as<double>() + (size_t) s * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
                    std::vector<int> ord(n);
                    for (int i = 0; i < n; i++) ord[i] = i;
                    std::partial_sort(ord.begin(), ord.begin() + std::min(n, 8), ord.end(),
                                      [&](int a, int b) { return std::fabs(hFs[a]) > std::fabs(hFs[b]); });
                    std::fprintf(stderr, "CONV_TOP s=%d it=%d :", s, it);
                    for (int k = 0; k < std::min(n, 8); k++) {
                        std::fprintf(stderr, " [%d]=%.9e", ord[k], hFs[ord[k]]);
                    }
                    std::fprintf(stderr, "\n");
                }
                static const char* sdEnv = std::getenv("OLF_GPU_STATE_DUMP");
                if (sdEnv != nullptr && s == 0 && it <= std::atoi(sdEnv) && n <= 64) {
                    stateDumpBatch<<<1, 1, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dMode.as<int>(),
                            ctx->dTarget.as<double>(), ctx->dX.as<double>(), s, it);
                    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                }
                static const char* wrEnv = std::getenv("OLF_GPU_WORST_ROW");
                if (wrEnv != nullptr && s == 0 && it <= std::atoi(wrEnv)) {
                    worstRowBatch<<<1, 1, 0, ctx->stream>>>(n, ctx->dF.as<double>(), ctx->dMode.as<int>(),
                            ctx->dTarget.as<double>(), ctx->dX.as<double>(), s);
                    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                }
                if (std::getenv("OLF_GPU_BATCH_TRACE") != nullptr) {
                    std::fprintf(stderr, "BATCH_TRACE s=%d it=%d infnorm=%.17e norm2=%.17e thr=%.17e conv=%d\n",
                                 s, it, norm, std::sqrt(ssq), thr2Of(s), converged ? 1 : 0);
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
                                 s, it, norm, std::sqrt(ssq), thr2Of(s), converged ? 1 : 0, fac, wr, hF[wr], hM[wr], hT[wr], hX[wr], wd, hDx[wd]);
                }
                res.iters[s] = it;
                res.norm[s] = norm;
                if (deviceFixedIter) {
                    res.status[s] = finite ? 1 : -1;         // ran maxIter; no convergence test
                } else if (!finite) {                        // a diverged scenario must not poison the batch
                    res.status[s] = -1;
                    active[s] = 0;
                    live[s] = 0;                             // failed: stop iterating its outer loops
                    nActive--;
                    anyChanged = true;
                } else if (converged && it > 0 && it < maxIter) {
                    // Force >=1 iteration, like CPU NewtonRaphson: a warm-started
                    // state may already meet the global 2-norm threshold yet leave a just-flipped identity row
                    // (PV target) unpinned — one Newton step pins it. Retiring at it==0 skips that solve.
                    // And NOT at the budget: OLF's NewtonRaphson.run counts a step before testing it and
                    // reports MAX_ITERATION_REACHED whenever iterations >= maxIterations, so a test that
                    // passes after the LAST allowed step is not CONVERGED there — the it == maxIter branch
                    // below reports it at OLF's budget (with a sound last solve).
                    res.status[s] = 0;
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
                        if (std::getenv("OLF_GPU_REJECT_TRACE") != nullptr) {
                            std::fprintf(stderr, "GPU_REJECT slot=%d reason=max-iter-bad-solve ratio=%.3e norm=%.6e\n",
                                         s, lastSolveRatio[s], norm);
                        }
                    }
                    res.status[s] = badSolve ? (nS == 1 ? -3 : -2) : 1;
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
        if (lsMode == 1) {                                   // LINE_SEARCH: remember the pre-step iterates
            CUDA_CHECK(cudaMemcpyAsync(ctx->dXprevLs.p, ctx->dX.p, static_cast<size_t>(n) * nS * sizeof(double),
                                       cudaMemcpyDeviceToDevice, ctx->stream));
        }
        if (ctx->mode == 3) {
        // ---- UBATCH solve: ANALYZE one block once (cheap), then a linear solve over all nS systems
        //      via CUDSS_CONFIG_UBATCH_SIZE on the single per-block matrix, FACTORIZED every iteration
        //      (honest Newton). ----
        writeIdentityForInactive<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                ctx->st->dDiagPos.as<int>(), ctx->dActive.as<char>(), ctx->dVals.as<double>(), ctx->dF.as<double>());
        negateIntoBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dF.as<double>(), ctx->dB.as<double>());
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        if (!ctx->analyzed) {
            bool timeIt = std::getenv("OLF_GPU_TRACE") != nullptr;
            long long t0 = 0;
            if (timeIt) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                t0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
            }
            gAnalysesMain++;
            { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
            double ms = -1.0;
            if (timeIt) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                ms = (std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count() - t0) / 1.0e6;
            }
            traceLuNnz(ctx->ds.handle, ctx->ds.data, "ubatch", ctx->nScenarios, ms);
            ctx->analyzed = true;
        }
        // ONE cudssExecute per iteration (phases are OR-able bit flags): halves the per-iteration
        // host-side phase overhead vs separate FACTORIZATION + SOLVE calls.
        int ubPhase = CUDSS_PHASE_SOLVE | CUDSS_PHASE_FACTORIZATION;
        // ---- solve-only COMPACTION: when few scenarios remain active, gather their vals/B into
        //      compact buffers and solve a NARROWER ubatch on the dedicated dsC context, then
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
        bool compactSolve = !deviceFixedIter && ctx->mode == 3 && nActive > 0 && nActive <= compactLimit
                && nActive < nS;
        if (!compactSolve) {
            gWidthIterSum += nS;                             // full-width factorization this iteration
        }
        int iterLogWidth = nS;                               // OLF_GPU_ITER_LOG: the width this iteration factorized
        if (compactSolve) {
            if (ctx->compactCap == 0) {                       // lazy first-engage setup
                ctx->compactCap = std::max(1, compactLimit);
                ctx->dValsC.alloc(static_cast<size_t>(nnz) * ctx->compactCap * sizeof(double));
                ctx->dBC.alloc(static_cast<size_t>(n) * ctx->compactCap * sizeof(double));
                ctx->dDxC.alloc(static_cast<size_t>(n) * ctx->compactCap * sizeof(double));
                ctx->dMapC.alloc(static_cast<size_t>(ctx->compactCap) * sizeof(int));
                CUDSS_CHECK(cudssCreate(&ctx->dsC.handle));
                CUDSS_CHECK(cudssSetStream(ctx->dsC.handle, ctx->stream));
                CUDSS_CHECK(cudssConfigCreate(&ctx->dsC.config));
                applyReorderAlg(ctx->dsC.config, ctx->st->nzi > 0);
                int irC = ctx->st->nzi > 0 ? irBaseSteps() : IR_N_STEPS;
                CUDSS_CHECK(cudssConfigSet(ctx->dsC.config, CUDSS_CONFIG_IR_N_STEPS, &irC, sizeof(irC)));
                CUDSS_CHECK(cudssMatrixCreateCsr(&ctx->matUbC, n, n, nnz,
                        ctx->st->dRowPtr.as<int>(), nullptr, ctx->st->dColIdx.as<int>(), ctx->dValsC.as<double>(),
                        OLF_CUDSS_CSR_TYPES, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbCB, n, 1, n, ctx->dBC.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbCX, n, 1, n, ctx->dDxC.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
                ctx->dRC.alloc(static_cast<size_t>(n) * ctx->compactCap * sizeof(double));
                ctx->dDeltaC.alloc(static_cast<size_t>(n) * ctx->compactCap * sizeof(double));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbCR, n, 1, n, ctx->dRC.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
                CUDSS_CHECK(cudssMatrixCreateDn(&ctx->matUbCDelta, n, 1, n, ctx->dDeltaC.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
            }
            std::vector<int> cmap;
            cmap.reserve(ctx->compactCap);
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
            if (w > ctx->compactCap) {
                w = ctx->compactCap;
            }
            while (static_cast<int>(cmap.size()) < w) {
                cmap.push_back(cmap[0]);
            }
            gWidthIterSum += w;                              // compacted width actually factorized
            iterLogWidth = -w;
            gCompactIters++;
            CUDA_CHECK(cudaMemcpyAsync(ctx->dMapC.p, cmap.data(), w * sizeof(int), cudaMemcpyHostToDevice, ctx->stream));
            gatherScenarios<<<grid(nnz * w), BLOCK, 0, ctx->stream>>>(nnz, w, ctx->dMapC.as<int>(), ctx->dVals.as<double>(), ctx->dValsC.as<double>());
            gatherScenarios<<<grid(n * w), BLOCK, 0, ctx->stream>>>(n, w, ctx->dMapC.as<int>(), ctx->dB.as<double>(), ctx->dBC.as<double>());
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
                    CUDSS_CHECK(cudssDataCreate(ctx->dsC.handle, &cached));
                    CUDSS_CHECK(cudssConfigSet(ctx->dsC.config, CUDSS_CONFIG_UBATCH_SIZE, &w, sizeof(w)));
                    gAnalysesCompact++;
                    { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->dsC.handle, CUDSS_PHASE_ANALYSIS, ctx->dsC.config, cached, ctx->matUbC, ctx->matUbCX, ctx->matUbCB)); }
                } else if (w != ctx->compactWidth) {
                    CUDSS_CHECK(cudssConfigSet(ctx->dsC.config, CUDSS_CONFIG_UBATCH_SIZE, &w, sizeof(w)));
                }
                ctx->compactWidth = w;
                compactData = cached;
            }
            { AnalysisTimer at_(__LINE__, 'F'); CUDSS_CHECK(cudssExecute(ctx->dsC.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, ctx->dsC.config, compactData, ctx->matUbC, ctx->matUbCX, ctx->matUbCB)); }
            if (ctx->st->nzi > 0 && ctx->dRC.p != nullptr && ctx->dIrScore.p != nullptr && ctx->dIrScore.cap >= static_cast<size_t>(w) * sizeof(double)) {
                // adaptive refinement on the compacted lanes (all w are real or a duplicate of a real one)
                refineLanes(ctx, w, n, nnz, ctx->dValsC.as<double>(), ctx->dDxC.as<double>(), ctx->dBC.as<double>(),
                        ctx->dRC.as<double>(), ctx->dDeltaC.as<double>(), nullptr, [&]() {
                    AnalysisTimer at_(__LINE__, 'S');
                    CUDSS_CHECK(cudssExecute(ctx->dsC.handle, CUDSS_PHASE_SOLVE, ctx->dsC.config, compactData, ctx->matUbC, ctx->matUbCDelta, ctx->matUbCR));
                });
            }
            CUDA_CHECK(cudaMemsetAsync(ctx->dDx.p, 0, static_cast<size_t>(n) * nS * sizeof(double), ctx->stream));
            scatterScenarios<<<grid(n * nReal), BLOCK, 0, ctx->stream>>>(n, nReal, ctx->dMapC.as<int>(), ctx->dDxC.as<double>(), ctx->dDx.as<double>());
        } else {
            { AnalysisTimer at_(__LINE__, ((ubPhase) & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) ? 'G' : ((ubPhase) & CUDSS_PHASE_ANALYSIS) ? 'A' : 'S'); CUDSS_CHECK(cudssExecute(ctx->ds.handle, ubPhase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
            if (ctx->st->nzi > 0 && ctx->dR.p != nullptr && ctx->matUbR != nullptr && ctx->dIrScore.p != nullptr) {
                // adaptive refinement on the full-width UBATCH solve (see refineLanes)
                refineLanes(ctx, nS, n, nnz, ctx->dVals.as<double>(), ctx->dDx.as<double>(), ctx->dB.as<double>(),
                        ctx->dR.as<double>(), ctx->dDelta.as<double>(), ctx->dActive.as<char>(), [&]() {
                    AnalysisTimer at_(__LINE__, 'S');
                    CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_SOLVE, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbDelta, ctx->matUbR));
                });
            }
        }
        // OLF_GPU_DX_DUMP=<path>: the FIRST solve's first Newton step, written out whole — the
        // assembled Jacobian (CSR triplets), the mismatch F it was built at, and the step dx it
        // produced. The CPU arm writes the same three under OLF_NR_DX_DUMP, and the two are
        // comparable WITHOUT a row map: the device's COLUMN index is OLF's own variable row
        // (GpuBatchPrecomputer stores v.getRow() verbatim), so dx compares term by term, and a
        // per-variable-column fingerprint of |J| localises any Jacobian difference to a variable.
        // Only F needs the row map, and its sorted multiset is permutation-invariant anyway.
        {
            static const char* dxPath = std::getenv("OLF_GPU_DX_DUMP");   // read once, not per iteration
            static bool dumped = false;
            // With OLF_GPU_LOOP_TRACE_CTG set, the dump is that contingency's slot in its own chunk
            // (the host sets loopTraceSlot to -1 for the chunks that do not hold it); slot 0 of the
            // first solve otherwise, as before.
            const int dumpSlot = ctx->loopTraceSlot >= 0 ? ctx->loopTraceSlot : (std::getenv("OLF_GPU_LOOP_TRACE_CTG") != nullptr ? -1 : 0);
            // OLF_GPU_DX_DUMP_OUTER=<k>: the first step of outer pass k instead of the first solve's
            static const int dxOuter = std::getenv("OLF_GPU_DX_DUMP_OUTER") != nullptr ? std::atoi(std::getenv("OLF_GPU_DX_DUMP_OUTER")) : -1;
            if (dxPath != nullptr && !dumped && it == 0 && dumpSlot >= 0 && (dxOuter < 0 || outer == dxOuter)) {
                dumped = true;
                dumpSlotSystem(ctx, n, nnz, dumpSlot, dxPath);
            }
        }
        // OLF_GPU_ROW_PROBE=<row>: that row in EVERY scenario at EVERY iteration - its residual,
        // its own state, and its non-zero Jacobian entries by column. The DX dump answers "what is
        // the shared base structure" and the state dump only fits a toy system; neither says what
        // one row of a 13000-row SCENARIO looks like at convergence, which is the question when a
        // bus's voltage has not moved by a single bit from the base while the CPU's has.
        {
            static const char* rowProbeEnv = std::getenv("OLF_GPU_ROW_PROBE");
            static int rowProbe = rowProbeEnv != nullptr ? std::atoi(rowProbeEnv) : -1;
            // With OLF_GPU_LOOP_TRACE_CTG set, only the traced contingency's slot in the chunk that holds it: every
            // chunk has a slot of that number, so printing them all mixed other contingencies' rows into the trace
            // (rte6515 GEN-676: slot 275 of every chunk read as GEN-676's row, base weights and all).
            const bool probeCtg = std::getenv("OLF_GPU_LOOP_TRACE_CTG") != nullptr;
            if (rowProbe >= 0 && rowProbe < n && (!probeCtg || ctx->loopTraceSlot >= 0)) {
                CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
                std::vector<int> hrp(n + 1);
                CUDA_CHECK(cudaMemcpy(hrp.data(), ctx->st->dRowPtr.p, (size_t) (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
                int k0 = hrp[rowProbe];
                int k1 = hrp[rowProbe + 1];
                std::vector<int> hci(k1 - k0);
                CUDA_CHECK(cudaMemcpy(hci.data(), ctx->st->dColIdx.as<int>() + k0, (size_t) (k1 - k0) * sizeof(int), cudaMemcpyDeviceToHost));
                std::vector<int> hmode(nS);
                for (int s2 = probeCtg ? ctx->loopTraceSlot : 0; s2 < (probeCtg ? ctx->loopTraceSlot + 1 : nS); s2++) {
                    double fv = 0;
                    double xv = 0;
                    int mv = 0;
                    CUDA_CHECK(cudaMemcpy(&fv, ctx->dF.as<double>() + (size_t) s2 * n + rowProbe, sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(&xv, ctx->dX.as<double>() + (size_t) s2 * n + rowProbe, sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(&mv, ctx->dMode.as<int>() + (size_t) s2 * n + rowProbe, sizeof(int), cudaMemcpyDeviceToHost));
                    double tv = 0;
                    CUDA_CHECK(cudaMemcpy(&tv, ctx->dTarget.as<double>() + (size_t) s2 * n + rowProbe, sizeof(double), cudaMemcpyDeviceToHost));
                    std::vector<double> hv(k1 - k0);
                    CUDA_CHECK(cudaMemcpy(hv.data(), ctx->dVals.as<double>() + (size_t) s2 * nnz + k0, (size_t) (k1 - k0) * sizeof(double), cudaMemcpyDeviceToHost));
                    std::fprintf(stderr, "ROW_PROBE it=%d s=%d row=%d mode=%d F=%.12e x=%.12e target=%.12e nnz=%d :", it, s2, rowProbe, mv, fv, xv, tv, k1 - k0);
                    for (int k = 0; k < k1 - k0; k++) {
                        if (hv[k] != 0.0) {
                            std::fprintf(stderr, " [%d]=%.6e", hci[k], hv[k]);
                        }
                    }
                    std::fprintf(stderr, "\n");
                }
            }
        }
        static const bool linResidOn = std::getenv("OLF_GPU_LIN_RESID") != nullptr;
        if (linResidOn) {
            DeviceBuf rp;
            DeviceBuf rm;
            rp.alloc((size_t) n * nS * sizeof(double));
            rm.alloc((size_t) n * nS * sizeof(double));
            linResidRowBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                    ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), rp.as<double>(), rm.as<double>());   // B = the (repaired) RHS
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            std::vector<double> hrp((size_t) n * nS);
            std::vector<double> hrm((size_t) n * nS);
            std::vector<double> hf((size_t) n * nS);
            std::vector<double> hdx((size_t) n * nS);
            CUDA_CHECK(cudaMemcpy(hrp.data(), rp.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hrm.data(), rm.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hf.data(), ctx->dF.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hdx.data(), ctx->dDx.p, (size_t) n * nS * sizeof(double), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                double mp = 0;
                double mm = 0;
                double mf = 0;
                double mdx = 0;
                double ssq = 0;                              // ||F||2 too: OLF's default stopping test is on the 2-norm
                int argp = -1;
                int argm = -1;
                for (int i = 0; i < n; i++) {
                    size_t t = (size_t) s * n + i;
                    if (std::fabs(hrp[t]) > mp) { mp = std::fabs(hrp[t]); argp = i; }
                    if (std::fabs(hrm[t]) > mm) { mm = std::fabs(hrm[t]); argm = i; }
                    mf = std::max(mf, std::fabs(hf[t]));
                    ssq += hf[t] * hf[t];
                    mdx = std::max(mdx, std::fabs(hdx[t]));
                }
                double best = std::min(mp, mm);
                int argw = mp < mm ? argp : argm;
                std::fprintf(stderr, "LIN_RESID it=%d slot=%d n=%d ||J.dx-F||inf=%.6e ||F||inf=%.6e ||F||2=%.17e "
                             "||dx||inf=%.6e ratio=%.3e worstRow=%d\n",
                             it, s, n, best, mf, std::sqrt(ssq), mdx, mf > 0 ? best / mf : 0.0, argw);
            }
        }
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
        // on a row that the first refactorization then heals (deviceRemoteSingletonMemberPin:
        // ratio 0.17, dx 2e13, converged to 1e-12 all the same), while a wrong factorization stays
        // at 1e6..1e13 on every iteration.
        // Pass 0 only: two n*nS kernels and a sync per pass cost 34 s of a 19 s rte6515 run (measured), and a
        // factorization that goes wrong LATER - after a reactive-limits pin rewrites its rows - is caught by the
        // no-progress reject at the end of that pass instead (see the it == maxIter branch).
        if (outer == 0 && it == 1 && !deviceFixedIter) {   // it=1: a flat-start first step may be garbage and heal (singleton-pin fixture); a wrong factorization never moves
            if (!ctx->dLinRp.p) {
                ctx->dLinRp.alloc((size_t) n * nS * sizeof(double));
                ctx->dLinRm.alloc((size_t) n * nS * sizeof(double));
                ctx->dLinSc.alloc((size_t) 3 * nS * sizeof(double));
            }
            DeviceBuf& rp = ctx->dLinRp;
            DeviceBuf& rm = ctx->dLinRm;
            DeviceBuf& sc = ctx->dLinSc;
            linResidRowBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                    ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), rp.as<double>(), rm.as<double>());
            linResidScoreBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(n, nS, rp.as<double>(), rm.as<double>(),
                    ctx->dF.as<double>(), sc.as<double>());
            std::vector<double> hsc((size_t) 3 * nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(hsc.data(), sc.p, (size_t) 3 * nS * sizeof(double), cudaMemcpyDeviceToHost));
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                double ratio = hsc[3 * s + 2];
                if (!(ratio <= LIN_SOLVE_FAIL_RATIO)) {         // NaN counts as failed
                    if (std::getenv("OLF_GPU_REJECT_TRACE") != nullptr) {
                        std::fprintf(stderr, "GPU_REJECT slot=%d reason=linear-solve ratio=%.3e ||J.dx-B||inf=%.3e ||F||inf=%.3e\n",
                                     s, ratio, hsc[3 * s], hsc[3 * s + 1]);
                    }
                    // OLF_GPU_DX_DUMP_ON_REJECT=<dir>: the rejected slot's system as the guard scored it, for a reject
                    // that only shows in a full run (the rte6515 busbar flap: one -3 per run on a different section
                    // each time, the identical ratio 3.06e+06 every time, never reproduced alone). The dump happens
                    // AFTER the verdict, so it cannot perturb what it catches.
                    static const char* rejectDumpDir = std::getenv("OLF_GPU_DX_DUMP_ON_REJECT");
                    static int rejectDumps = 0;
                    if (rejectDumpDir != nullptr && rejectDumps < 8) {
                        std::string p = std::string(rejectDumpDir) + "/reject-" + std::to_string(rejectDumps++)
                                + "-slot" + std::to_string(s) + ".dump";
                        dumpSlotSystem(ctx, n, nnz, s, p.c_str());
                    }
                    traceReject(s, "linear-solve", outer, it);
                    res.status[s] = -3;                      // STATUS_LINEAR_SOLVE_FAILED: the one reject a pivoted retry can fix
                    res.iters[s] = it;
                    active[s] = 0;
                    live[s] = 0;
                    nActive--;
                }
            }
        }
        // OLF_GPU_ITER_LOG: every step's linear solve scored like the pass-0 guard above, on every pass
        std::vector<double> iterLogLin;
        if (iterLog && !deviceFixedIter) {
            if (!ctx->dLinRp.p) {
                ctx->dLinRp.alloc((size_t) n * nS * sizeof(double));
                ctx->dLinRm.alloc((size_t) n * nS * sizeof(double));
                ctx->dLinSc.alloc((size_t) 3 * nS * sizeof(double));
            }
            linResidRowBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->st->dRowPtr.as<int>(),
                    ctx->st->dColIdx.as<int>(), ctx->dVals.as<double>(), ctx->dDx.as<double>(),
                    ctx->dB.as<double>(), ctx->dLinRp.as<double>(), ctx->dLinRm.as<double>());
            linResidScoreBatch<<<grid(nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dLinRp.as<double>(), ctx->dLinRm.as<double>(),
                    ctx->dF.as<double>(), ctx->dLinSc.as<double>());
            iterLogLin.resize((size_t) 3 * nS);
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
            CUDA_CHECK(cudaMemcpy(iterLogLin.data(), ctx->dLinSc.p, (size_t) 3 * nS * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (ctx->clampStep) {
            maxVoltageChangeFactorBatch<<<nS, BLOCK, 0, ctx->stream>>>(n, ctx->dDx.as<double>(), ctx->dVarType.as<int>(), ctx->dActive.as<char>(), ctx->maxDv, ctx->maxDphi, ctx->dStepFactor.as<double>());
            traceStepFactor(ctx, nS, it, active);
            axpy1BatchScaled<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, n * nS, ctx->dDx.as<double>(), ctx->dStepFactor.as<double>(), ctx->dX.as<double>());
        } else {
            axpy1Batch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n * nS, ctx->dDx.as<double>(), ctx->dX.as<double>());
        }
        if (!iterLogLin.empty()) {
            std::vector<double> fac(nS, 1.0);
            if (ctx->clampStep) {
                CUDA_CHECK(cudaMemcpy(fac.data(), ctx->dStepFactor.p, nS * sizeof(double), cudaMemcpyDeviceToHost));
            }
            for (int s = 0; s < nS; s++) {
                if (!active[s]) continue;
                char line[256];
                std::snprintf(line, sizeof line, "LIN_LOG outer=%d s=%d it=%d ratio=%.6e resid=%.6e fmax=%.6e factor=%.9f width=%d",
                              outer, s, it, iterLogLin[3 * s + 2], iterLogLin[3 * s], iterLogLin[3 * s + 1], fac[s], iterLogWidth);
                iterLogLines.emplace_back(line);
                if (!(iterLogLin[3 * s + 2] <= 1e-4)) {
                    // WHERE a poor solve leaves its residual: the worst row under the sign convention the ratio
                    // used, its row mode, and whether it is a zero-diagonal hazard row (with this lane's donor)
                    std::vector<double> rp(n), rm(n);
                    std::vector<int> md(n);
                    CUDA_CHECK(cudaMemcpy(rp.data(), ctx->dLinRp.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(rm.data(), ctx->dLinRm.as<double>() + (size_t) s * n, n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(md.data(), ctx->dMode.as<int>() + (size_t) s * n, n * sizeof(int), cudaMemcpyDeviceToHost));
                    double mp = 0.0;
                    double mm = 0.0;
                    for (int i = 0; i < n; i++) {
                        mp = std::max(mp, std::fabs(rp[i]));
                        mm = std::max(mm, std::fabs(rm[i]));
                    }
                    const std::vector<double>& r = mp <= mm ? rp : rm;   // the sign convention the score took
                    std::vector<int> ord(n);
                    for (int i = 0; i < n; i++) ord[i] = i;
                    std::partial_sort(ord.begin(), ord.begin() + std::min(n, 4), ord.end(),
                                      [&](int a, int b) { return std::fabs(r[a]) > std::fabs(r[b]); });
                    const auto& hz = ctx->st->hHazRow;
                    std::vector<int> choice;
                    if (!hz.empty() && ctx->dHazChoice.p && ctx->hazCap >= nS) {
                        choice.resize(hz.size());
                        CUDA_CHECK(cudaMemcpy(choice.data(), ctx->dHazChoice.as<int>() + (size_t) s * hz.size(),
                                              hz.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    }
                    std::string worst = "LIN_WORST outer=" + std::to_string(outer) + " s=" + std::to_string(s)
                            + " it=" + std::to_string(it);
                    for (int k = 0; k < std::min(n, 4); k++) {
                        int row = ord[k];
                        int hi = -1;
                        for (size_t h = 0; h < hz.size(); h++) {
                            if (hz[h] == row) hi = static_cast<int>(h);
                        }
                        char b[128];
                        std::snprintf(b, sizeof b, " [row=%d r=%.3e mode=%d haz=%d donor=%d]", row, r[row], md[row], hi,
                                      hi >= 0 && !choice.empty() ? choice[hi] : -9);
                        worst += b;
                    }
                    iterLogLines.emplace_back(worst);
                }
            }
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
            hazRepairRightHandSide(ctx, 1, bS, s);
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matA, valsS));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matB, bS));
            CUDSS_CHECK(cudssMatrixSetValues(ctx->ds.matX, dxS));
            if (!ctx->analyzed) {
                gAnalysesMain++;
                { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB)); }
                traceLuNnz(ctx->ds.handle, ctx->ds.data, "per-scenario", ctx->nScenarios);
                ctx->analyzed = true;
            }
            { AnalysisTimer at_(__LINE__, 'F'); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                    ctx->ds.config, ctx->ds.data, ctx->ds.matA, ctx->ds.matX, ctx->ds.matB)); }
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
    if (olTime) {
        int nSolved = 0;
        for (int s = 0; s < nS; s++) nSolved += wasActive[s] ? 1 : 0;
        olPassSolved = nSolved;
    }
    for (int s = 0; s < nS; s++) {                           // per-scenario Newton work by pass (cheap, host)
        if (wasActive[s]) {
            if (ctx->olPasses[s] == 0) ctx->olFirst[s] = res.iters[s];
            ctx->olIters[s] += res.iters[s];
            ctx->olPasses[s]++;
        }
    }
}

}  // namespace olfgpu
