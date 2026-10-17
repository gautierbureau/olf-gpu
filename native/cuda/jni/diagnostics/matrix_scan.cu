// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Diagnostics of the assembled linear system, see matrix_scan.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "closed_branch_scatter.h"

#include "../common/check.cuh"
#include "matrix_scan.cuh"

using namespace powsybl::openloadflow;

namespace olfgpu {

// OLF_GPU_VALS_SCAN=1: is the MATRIX itself sound before cuDSS ever sees it? Every finiteness check on
// this path guards F (the mismatch); nothing guards the Jacobian values. A NaN or an astronomical entry
// arriving from a FILL explains a broken solve under every ordering and every mode at once, which no
// amount of reordering or pivoting can fix — so this separates "cuDSS cannot factor this matrix" from
// "the matrix handed to cuDSS was already wrong". Reports the FIRST non-finite entry with the row it
// sits on, that row's per-scenario mode and its diagonal, plus the largest |value| in the matrix.
// Diagnostic only: it copies values back, so it scans at most the first two scenario slots.
void scanValsFinite(BatchSolverContext* ctx, int nS, int n, int nnz, int outer) {
    static const bool on = std::getenv("OLF_GPU_VALS_SCAN") != nullptr;
    if (!on) {
        return;
    }
    static int seq = 0;
    int it = seq++;                                          // fill sequence: orders these against LIN_RESID
    int sMax = std::min(nS, 2);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<double> hv(static_cast<size_t>(nnz) * sMax);
    CUDA_CHECK(cudaMemcpy(hv.data(), ctx->dVals.p, hv.size() * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<int> rowPtr(n + 1);
    std::vector<int> colIdx(nnz);
    std::vector<int> diagPos(n);
    CUDA_CHECK(cudaMemcpy(rowPtr.data(), ctx->st->dRowPtr.p, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(colIdx.data(), ctx->st->dColIdx.p, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(diagPos.data(), ctx->st->dDiagPos.p, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> mode(static_cast<size_t>(n) * sMax);
    CUDA_CHECK(cudaMemcpy(mode.data(), ctx->dMode.p, mode.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int s = 0; s < sMax; s++) {
        const double* vs = hv.data() + static_cast<size_t>(s) * nnz;
        double maxAbs = 0.0;
        int maxRow = -1;
        int maxCol = -1;
        int bad = -1;
        for (int k = 0; k < nnz; k++) {
            double v = vs[k];
            if (bad < 0 && !std::isfinite(v)) {
                bad = k;
            }
            if (std::isfinite(v) && std::fabs(v) > maxAbs) {
                maxAbs = std::fabs(v);
                maxCol = colIdx[k];
                maxRow = static_cast<int>(std::upper_bound(rowPtr.begin(), rowPtr.end(), k) - rowPtr.begin()) - 1;
            }
        }
        if (bad >= 0) {
            int row = static_cast<int>(std::upper_bound(rowPtr.begin(), rowPtr.end(), bad) - rowPtr.begin()) - 1;
            std::fprintf(stderr, "VALS_NAN outer=%d it=%d slot=%d row=%d col=%d mode=%d value=%g diag=%g\n",
                         outer, it, s, row, colIdx[bad], mode[static_cast<size_t>(s) * n + row], vs[bad],
                         diagPos[row] >= 0 ? vs[diagPos[row]] : 0.0);
        }
        // Rows with no usable PIVOT on the real values: an unpivoted LU divides by these. nHaz is what the
        // structural repair thought needed fixing, so nHaz == 0 beside a nonzero count here means the
        // repair did not RECOGNIZE these rows — which is a detection bug, not a numerical limit.
        int noDiagSlot = 0;
        int tinyDiag = 0;
        int firstBad = -1;
        for (int r = 0; r < n; r++) {
            if (diagPos[r] < 0) {
                noDiagSlot++;
                if (firstBad < 0) {
                    firstBad = r;
                }
            } else if (std::fabs(vs[diagPos[r]]) < 1e-12) {
                tinyDiag++;
                if (firstBad < 0) {
                    firstBad = r;
                }
            }
        }
        // OLF_GPU_VALS_SCAN_TINY=1: NAME the tiny-pivot rows, not just count them, with what each row
        // actually holds. An EMPTY row (no nonzero at all) is an equation that was erased; a row with
        // off-diagonal entries but no diagonal is an equation placed on a column it does not depend on.
        // Those two point at different code, and the count alone cannot tell them apart.
        static const bool tinyList = std::getenv("OLF_GPU_VALS_SCAN_TINY") != nullptr;
        if (tinyList && tinyDiag > 0) {
            int shown = 0;
            for (int r = 0; r < n && shown < 30; r++) {
                if (diagPos[r] < 0 || std::fabs(vs[diagPos[r]]) >= 1e-12) {
                    continue;
                }
                int nz = 0;
                double rowMax = 0.0;
                int rowMaxCol = -1;
                for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
                    if (vs[k] != 0.0) {
                        nz++;
                        if (std::fabs(vs[k]) > rowMax) {
                            rowMax = std::fabs(vs[k]);
                            rowMaxCol = colIdx[k];
                        }
                    }
                }
                std::fprintf(stderr, "VALS_TINY outer=%d it=%d slot=%d row=%d mode=%d slots=%d nonzero=%d rowMax=%.3e at col=%d diag=%.3e\n",
                             outer, it, s, r, mode[static_cast<size_t>(s) * n + r], rowPtr[r + 1] - rowPtr[r],
                             nz, rowMax, rowMaxCol, vs[diagPos[r]]);
                shown++;
            }
        }
        // RANK, not scaling: when no ordering and no pivoting strategy helps a finite, well-scaled matrix
        // with no tiny diagonal, the remaining explanation is that it is SINGULAR — and pivoting cannot fix
        // rank deficiency. Two structural witnesses are cheap to check here: a row that DUPLICATES another
        // (the same equation scattered into two rows leaves one variable undetermined) and the per-mode row
        // census (an equation row that silently became an identity, or the reverse, changes rank).
        std::unordered_map<std::string, int> rowKey;
        int dupA = -1;
        int dupB = -1;
        for (int r = 0; r < n && dupA < 0; r++) {
            std::string key;
            key.reserve(static_cast<size_t>(rowPtr[r + 1] - rowPtr[r]) * 24);
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
                if (vs[k] != 0.0) {
                    char buf[48];
                    std::snprintf(buf, sizeof(buf), "%d:%.17g;", colIdx[k], vs[k]);
                    key += buf;
                }
            }
            if (key.empty()) {
                continue;                                    // an all-zero row is caught by tinyDiag above
            }
            auto hit = rowKey.emplace(key, r);
            if (!hit.second) {
                dupA = hit.first->second;
                dupB = r;
            }
        }
        // COLUMN witness, which the diagonal check above cannot give: the documented failure of this
        // structure is a variable left in NO equation when two contributions CANCEL (a DISTR_Q kind-3 pair
        // against the zero-impedance family's +-DUMMY_Q — "45 such columns at 2000 busbar sections, which
        // is what collapsed batching above ~1000", GpuAcActivationExtractor). A dead COLUMN makes the
        // matrix exactly singular, and no ordering or pivot strategy can rescue that.
        std::vector<double> colMax(n, 0.0);
        for (int k = 0; k < nnz; k++) {
            double a = std::fabs(vs[k]);
            if (std::isfinite(a) && a > colMax[colIdx[k]]) {
                colMax[colIdx[k]] = a;
            }
        }
        int deadCols = 0;
        int firstDeadCol = -1;
        for (int c = 0; c < n; c++) {
            if (colMax[c] < 1e-12) {
                deadCols++;
                if (firstDeadCol < 0) {
                    firstDeadCol = c;
                }
            }
        }
        int modeCount[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        char m4[256] = {0};
        int m4len = 0;
        for (int r = 0; r < n; r++) {
            int m = mode[static_cast<size_t>(s) * n + r];
            if (m >= 0 && m < 8) {
                modeCount[m]++;
            }
            if (m == 4 && m4len < 200) {                     // WHICH rows are pinned, so the fill that first
                m4len += std::snprintf(m4 + m4len, sizeof(m4) - m4len, " %d", r);   // moves one is visible
            }
        }
        // OLF_GPU_VALS_DUMP=<path>: write the FIRST matrix that carries a mode-4 row (the RL-pinned remote
        // gen-VC member whose fill is where this structure stops factoring) as plain CSR + rhs + modes, once.
        // Every local witness here comes back clean on it, so the open question is RANK — which is answered
        // offline in seconds (scipy / KLU on the same matrix) instead of by more 8-minute device runs.
        static bool dumped = false;
        const char* dumpPath = std::getenv("OLF_GPU_VALS_DUMP");
        // OLF_GPU_VALS_DUMP_ANY=1: dump the FIRST fill instead of waiting for a mode-4 row. The pinned pass is
        // the interesting one for a rank question, but a wrong VALUE is better read on the UNPINNED base, where
        // the matrix is well determined and nothing is confounded by the rank deficiency.
        static const bool dumpAny = std::getenv("OLF_GPU_VALS_DUMP_ANY") != nullptr;
        // OLF_GPU_VALS_DUMP_NTH=<n>: dump the n-th ELIGIBLE fill instead of the first, so a later context (the
        // contingency batch after the base) can be read; with DUMP_ANY every first fill of a solve is eligible.
        static const int dumpNth = std::getenv("OLF_GPU_VALS_DUMP_NTH") != nullptr ? std::atoi(std::getenv("OLF_GPU_VALS_DUMP_NTH")) : 1;
        static int eligibleSeen = 0;
        // `it` runs on across contexts (the contingency batch after the base starts at it=5, not 0), so the
        // first fill of a solve is "outer 0 after a scan that was not outer 0", not "it == 0".
        static int prevOuter = -1;
        bool firstOfSolve = outer == 0 && prevOuter != 0;
        bool firstOfOuter = outer != prevOuter;
        prevOuter = outer;
        // OLF_GPU_VALS_DUMP_OUTER=<outer>: dump the FIRST fill of that outer pass (the post-switch system of a
        // reactive-limits re-solve, say) instead of the first mode-4 / first-of-solve fill.
        static const char* dumpOuterEnv = std::getenv("OLF_GPU_VALS_DUMP_OUTER");
        static const int dumpOuter = dumpOuterEnv != nullptr ? std::atoi(dumpOuterEnv) : -1;
        bool eligible = dumpPath != nullptr && *dumpPath != '\0'
                && (dumpOuter >= 0 ? (outer == dumpOuter && firstOfOuter) : (dumpAny ? firstOfSolve : modeCount[4] > 0));
        if (eligible) eligibleSeen++;
        if (!dumped && eligible && eligibleSeen == dumpNth) {
            std::vector<double> hf(n);
            CUDA_CHECK(cudaMemcpy(hf.data(), static_cast<const double*>(ctx->dF.p) + static_cast<size_t>(s) * n,
                                  n * sizeof(double), cudaMemcpyDeviceToHost));
            if (std::FILE* f = std::fopen(dumpPath, "w")) {
                std::fprintf(f, "n %d nnz %d outer %d seq %d slot %d nS %d\n", n, nnz, outer, it, s, nS);
                for (int r = 0; r <= n; r++) {
                    std::fprintf(f, "%d%c", rowPtr[r], r == n ? '\n' : ' ');
                }
                for (int k = 0; k < nnz; k++) {
                    std::fprintf(f, "%d%c", colIdx[k], k == nnz - 1 ? '\n' : ' ');
                }
                for (int k = 0; k < nnz; k++) {
                    std::fprintf(f, "%.17g%c", vs[k], k == nnz - 1 ? '\n' : ' ');
                }
                for (int r = 0; r < n; r++) {
                    std::fprintf(f, "%.17g%c", hf[r], r == n - 1 ? '\n' : ' ');
                }
                for (int r = 0; r < n; r++) {
                    std::fprintf(f, "%d%c", mode[static_cast<size_t>(s) * n + r], r == n - 1 ? '\n' : ' ');
                }
                // lines 7 and 8: the STATE and the TARGET this fill was evaluated against. F = eval - target with
                // an identical Jacobian can differ through either, and only the vectors say which (the row-10
                // difference between a table-enrolled and a plain pilot-self group was exactly that question).
                {
                    std::vector<double> hx(n), ht(n);
                    CUDA_CHECK(cudaMemcpy(hx.data(), static_cast<const double*>(ctx->dX.p) + static_cast<size_t>(s) * n,
                                          n * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(ht.data(), static_cast<const double*>(ctx->dTarget.p) + static_cast<size_t>(s) * n,
                                          n * sizeof(double), cudaMemcpyDeviceToHost));
                    for (int r = 0; r < n; r++) {
                        std::fprintf(f, "%.17g%c", hx[r], r == n - 1 ? '\n' : ' ');
                    }
                    for (int r = 0; r < n; r++) {
                        std::fprintf(f, "%.17g%c", ht[r], r == n - 1 ? '\n' : ' ');
                    }
                }
                std::fclose(f);
                dumped = true;
                std::fprintf(stderr, "VALS_DUMP wrote %s (n=%d nnz=%d outer=%d seq=%d mode4=%d nS=%d nth=%d)\n",
                             dumpPath, n, nnz, outer, it, modeCount[4], nS, eligibleSeen);
                // OLF_GPU_VALS_DUMP_COL=<variable column>: WHICH PLACEMENT built this matrix, for the zero-
                // impedance element that owns that column. A dumped matrix alone cannot say: the base is
                // extracted more than once, with DIFFERENT qRow/qRowAlt per extraction (measured on rte6515's
                // VL-6112 pilot: qRow=12836/qRowAlt=-1 in one, qRow=-1/qRowAlt=12834 in the other), and each
                // placement is internally consistent, so reading a column's entries without knowing which one
                // is in force invites aiming a fix at a placement that was never live. The host holds the same
                // inputs the fill kernel reads, so REPLAY ziDynDest's two-way choice here rather than guess it.
                const char* colEnv = std::getenv("OLF_GPU_VALS_DUMP_COL");
                if (colEnv != nullptr && ctx->nzi > 0) {
                    int want = std::atoi(colEnv);
                    std::vector<int> zidx(static_cast<size_t>(ctx->nzi) * 11);
                    std::vector<int> zdst(static_cast<size_t>(ctx->nzi) * 14);
                    std::vector<int> zalt(static_cast<size_t>(ctx->nzi) * 8, -1);
                    std::vector<int> zsel(static_cast<size_t>(ctx->nzi) * 8, -1);
                    CUDA_CHECK(cudaMemcpy(zidx.data(), ctx->st->dZiIdx.p, zidx.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(zdst.data(), ctx->st->dZiDest.p, zdst.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    if (ctx->st->dZiAltDest.p) {
                        CUDA_CHECK(cudaMemcpy(zalt.data(), ctx->st->dZiAltDest.p, zalt.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    }
                    if (ctx->st->dZiSelRow.p) {
                        CUDA_CHECK(cudaMemcpy(zsel.data(), ctx->st->dZiSelRow.p, zsel.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    }
                    const int* ms = mode.data() + static_cast<size_t>(s) * n;
                    for (int e = 0; e < ctx->nzi; e++) {
                        const int* z = zidx.data() + static_cast<size_t>(e) * 11;
                        if (z[4] != want && z[5] != want) {
                            continue;                        // not this element's dummyP / dummyQ column
                        }
                        std::fprintf(stderr, "VALS_DUMP_COL col=%d zi=%d dummyP=%d dummyQ=%d"
                                     " frozenQ=[%d,%d] ziDyn=%d\n",
                                     want, e, z[4], z[5], z[1], z[3], ctx->st->ziDyn ? 1 : 0);
                        for (int slot = 0; slot < 4; slot++) {   // F1,F2 then J1,J2 of the dummyQ coupling
                            size_t o = static_cast<size_t>(e) * 8 + slot;
                            int frozen = zdst[static_cast<size_t>(e) * 14 + (slot < 2 ? 2 + slot : 4 + slot)];
                            int sr = zsel[o];
                            int selMode = sr >= 0 && sr < n ? ms[sr] : (sr <= -2 && (-2 - sr) < n ? ms[-2 - sr] : -99);
                            int resolved = frozen;
                            if (ctx->st->ziDyn) {
                                if (sr <= -2) {
                                    resolved = selMode == 0 ? zalt[o] : frozen;
                                } else if (sr >= 0) {
                                    resolved = (selMode == 0 || selMode == 4) ? zalt[o] : frozen;
                                }
                            }
                            std::fprintf(stderr, "    slot=%d(%s) frozen=%d alt=%d sel=%d selMode=%d -> %d\n",
                                         slot, slot < 2 ? "F" : "J", frozen, zalt[o], sr, selMode, resolved);
                        }
                    }
                    // ...and the DISTR_Q channel's kind-3 contributions on the SAME column, with each gate of
                    // fillDistrQBatch replayed. A dummy column carries +1 in ONE bus row and -1 in the dummy
                    // variable's own row when it is healthy; a THIRD entry means some channel emitted where it
                    // should not, and the zi replay above rules that channel out. Naming the writer by
                    // elimination cost two wrong diagnoses in this area, and the device-side printf in
                    // fillDistrQBatch (OLF_GPU_DQ_TRACE) never reaches this harness's log — hence this.
                    if (ctx->ndq > 0) {
                        std::vector<int> de(ctx->ndq), dk(ctx->ndq), dd(static_cast<size_t>(ctx->ndq) * 6);
                        std::vector<double> dw(ctx->ndq);
                        std::vector<int> jm(ctx->ndq, -1), mqg;
                        std::vector<char> pss;
                        CUDA_CHECK(cudaMemcpy(de.data(), ctx->st->dDqElem.p, de.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(dk.data(), ctx->st->dDqKind.p, dk.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(dw.data(), ctx->st->dDqWeight.p, dw.size() * sizeof(double), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(dd.data(), ctx->st->dDqDest.p, dd.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        if (ctx->dDqJMember.p) {
                            CUDA_CHECK(cudaMemcpy(jm.data(), ctx->dDqJMember.p, jm.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        }
                        if (ctx->hasDqQrGroups && ctx->dDqMemberQrGroup.p) {
                            mqg.assign(ctx->dDqMemberQrGroup.cap / sizeof(int), -1);
                            CUDA_CHECK(cudaMemcpy(mqg.data(), ctx->dDqMemberQrGroup.p, mqg.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        }
                        if (ctx->hasPilotSelf && ctx->dPsSkip.p) {
                            pss.assign(ctx->dPsSkip.cap / sizeof(char), 0);
                            CUDA_CHECK(cudaMemcpy(pss.data(), ctx->dPsSkip.p, pss.size() * sizeof(char), cudaMemcpyDeviceToHost));
                        }
                    // OLF_GPU_VALS_DUMP_ROW=<device row>: every REMOTE-MEMBER-Q contribution summing into
                    // that row's diagnostic q slot, which is what the reactive-limit check reads as
                    // `q = qBuf[row] + loadQ`. A term counted twice makes the check compare a doubled q
                    // against OLF's limits, and on rte6515's VL-6112_0 the device's q sits exactly one
                    // -0.3075 pu step past OLF's (0.603 against 0.9105 with loadQ 1.218), which is what a
                    // duplicated contribution looks like.
                    const char* rowEnv = std::getenv("OLF_GPU_VALS_DUMP_ROW");
                    if (rowEnv != nullptr && ctx->nRmq > 0) {
                        int wantRow = std::atoi(rowEnv);
                        std::vector<int> rr(ctx->nRmq), re(ctx->nRmq), rs(ctx->nRmq), rk(ctx->nRmq);
                        std::vector<double> rw(ctx->nRmq);
                        CUDA_CHECK(cudaMemcpy(rr.data(), ctx->dRmqRow.p, rr.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(re.data(), ctx->dRmqElem.p, re.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(rs.data(), ctx->dRmqSide.p, rs.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(rk.data(), ctx->dRmqKind.p, rk.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(rw.data(), ctx->dRmqWeight.p, rw.size() * sizeof(double), cudaMemcpyDeviceToHost));
                        std::vector<double> hx(n);
                        CUDA_CHECK(cudaMemcpy(hx.data(), static_cast<const double*>(ctx->dX.p) + static_cast<size_t>(s) * n,
                                              n * sizeof(double), cudaMemcpyDeviceToHost));
                        int cnt = 0;
                        for (int c = 0; c < ctx->nRmq; c++) {
                            if (rr[c] != wantRow) {
                                continue;
                            }
                            cnt++;
                            std::fprintf(stderr, "    RMQ row=%d c=%d kind=%d elem=%d side=%d w=%.9g%s\n",
                                         wantRow, c, rk[c], re[c], rs[c], rw[c],
                                         rk[c] == 3 ? (std::string(" x[elem]=") + std::to_string(hx[re[c]])).c_str() : "");
                        }
                        std::fprintf(stderr, "    RMQ row=%d contributions=%d\n", wantRow, cnt);
                    }
                        // The gate's OWN inputs for each pilot-self group, so "psSkip says unpinned" can be
                        // told from "the pin never booked a limitType". pilotSelfGateBatch sets
                        // psSkip = (limitType[psPilotEntry[g]] == 0), and the row mode is the other opinion.
                        if (ctx->hasPilotSelf && ctx->dPsPilotEntry.p && ctx->dRlLimitType.p && ctx->nrl > 0) {
                            std::vector<int> pe(ctx->st->nQr, -1);
                            std::vector<char> lt(static_cast<size_t>(ctx->nrl) * nS, 0);
                            CUDA_CHECK(cudaMemcpy(pe.data(), ctx->dPsPilotEntry.p, pe.size() * sizeof(int), cudaMemcpyDeviceToHost));
                            CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, lt.size() * sizeof(char), cudaMemcpyDeviceToHost));
                            for (int g = 0; g < ctx->st->nQr; g++) {
                                int e = pe[g];
                                if (e < 0) {
                                    continue;
                                }
                                int fallbackRow = ctx->st->hQrPtr.empty() ? -1 : ctx->st->hQrRows[ctx->st->hQrPtr[g]];
                                std::fprintf(stderr, "    PS g=%d pilotEntry=%d limitType=%d fallbackRow=%d"
                                             " rowMode=%d psSkip=%d\n", g, e, (int) lt[static_cast<size_t>(s) * ctx->nrl + e],
                                             fallbackRow, fallbackRow >= 0 && fallbackRow < n ? ms[fallbackRow] : -99,
                                             pss.empty() ? -1 : (int) pss[static_cast<size_t>(s) * ctx->st->nQr + g]);
                            }
                        }
                        for (int c = 0; c < ctx->ndq; c++) {
                            if (dk[c] != 3 || de[c] != want) {
                                continue;                    // kind 3 = zero-impedance DUMMY_Q, elem = its column
                            }
                            int d0 = dd[static_cast<size_t>(c) * 6];
                            int fRow = d0 <= -2 ? -2 - d0 : -1;
                            bool gateMode4 = fRow >= 0 && fRow < n && ms[fRow] == 4;
                            int g = !mqg.empty() && jm[c] >= 0 && jm[c] < static_cast<int>(mqg.size()) ? mqg[jm[c]] : -1;
                            bool psGateOff = !pss.empty() && g >= 0
                                    && pss[static_cast<size_t>(s) * ctx->st->nQr + g] != 0;
                            int carrier = -1;
                            if (g >= 0 && !ctx->st->hQrPtr.empty()) {
                                int p0 = ctx->st->hQrPtr[g];
                                int cnt = ctx->st->hQrPtr[g + 1] - p0;
                                int pick = 0;
                                for (int cc = 1; cc < cnt; cc++) {
                                    int m2 = ms[ctx->st->hQrRows[p0 + cc]];
                                    if (m2 != 0 && m2 != 4) { pick = cc; break; }
                                }
                                carrier = ctx->st->hQrRows[p0 + pick];
                            }
                            bool gateCarrier = g >= 0 && !psGateOff && fRow >= 0 && fRow == carrier;
                            std::fprintf(stderr, "    DQ c=%d kind=3 elem=%d w=%.6g fRow=%d fRowMode=%d"
                                         " jMember=%d qrGroup=%d carrier=%d psSkip=%d"
                                         " gateMode4=%d gateCarrier=%d -> %s\n",
                                         c, de[c], dw[c], fRow, fRow >= 0 ? ms[fRow] : -99, jm[c], g, carrier,
                                         psGateOff ? 1 : 0, gateMode4 ? 1 : 0, gateCarrier ? 1 : 0,
                                         (gateMode4 || gateCarrier) ? "SUPPRESSED" : "EMITTED");
                        }
                        // ...and EVERY DISTR_Q contribution whose residual lands in OLF_GPU_VALS_DUMP_ROW, all kinds,
                        // with the STATIC weight (the dynamic renormalization over survivors is applied in-kernel
                        // and is not replayed here — a row whose static weights are 1/3 with one member islanded
                        // should assemble as 1/2 on the device, which the dumped VALUES show).
                        const char* rowEnv2 = std::getenv("OLF_GPU_VALS_DUMP_ROW");
                        if (rowEnv2 != nullptr) {
                            int wantRow2 = std::atoi(rowEnv2);
                            for (int c = 0; c < ctx->ndq; c++) {
                                int d0 = dd[static_cast<size_t>(c) * 6];
                                int fRow = d0 <= -2 ? -2 - d0 : -1;
                                if (fRow != wantRow2) {
                                    continue;
                                }
                                int g = !mqg.empty() && jm[c] >= 0 && jm[c] < static_cast<int>(mqg.size()) ? mqg[jm[c]] : -1;
                                bool psGateOff = !pss.empty() && g >= 0 && pss[static_cast<size_t>(s) * ctx->st->nQr + g] != 0;
                                std::fprintf(stderr, "    DQROW row=%d c=%d kind=%d elem=%d w=%.6g jMember=%d qrGroup=%d psSkip=%d rowMode=%d\n",
                                             wantRow2, c, dk[c], de[c], dw[c], jm[c], g, psGateOff ? 1 : 0, ms[wantRow2]);
                            }
                        }
                    }
                }
            }
        }
        // OLF_GPU_VALS_DUMP_ROW=<row>, standalone (the replay above needs zero-impedance branches): every
        // DISTR_Q contribution whose residual lands in the row, with its static weight, dests, and the
        // DYNAMIC-key inputs the kernel renormalizes with (owner member, its RL entry and limit type, keys).
        // Printed on the same fills VALS_SCAN reports, once per fill for the first scenario.
        if (static const char* dqRowEnv = std::getenv("OLF_GPU_VALS_DUMP_ROW"); dqRowEnv != nullptr && ctx->ndq > 0 && s == 0) {
            int wantRow = std::atoi(dqRowEnv);
            std::vector<int> dd(static_cast<size_t>(ctx->ndq) * 6), dk(ctx->ndq), de(ctx->ndq), ds(ctx->ndq);
            std::vector<double> dw(ctx->ndq);
            CUDA_CHECK(cudaMemcpy(dd.data(), ctx->st->dDqDest.p, dd.size() * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(dk.data(), ctx->st->dDqKind.p, dk.size() * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(de.data(), ctx->st->dDqElem.p, de.size() * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ds.data(), ctx->st->dDqSide.p, ds.size() * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(dw.data(), ctx->st->dDqWeight.p, dw.size() * sizeof(double), cudaMemcpyDeviceToHost));
            std::vector<int> jm, km, mre;
            std::vector<double> mk;
            if (ctx->nDqDyn || ctx->hasDqQrGroups) {
                jm.resize(ctx->ndq);
                CUDA_CHECK(cudaMemcpy(jm.data(), ctx->dDqJMember.p, jm.size() * sizeof(int), cudaMemcpyDeviceToHost));
            }
            if (ctx->nDqDyn) {
                km.resize(ctx->ndq);
                CUDA_CHECK(cudaMemcpy(km.data(), ctx->dDqKMember.p, km.size() * sizeof(int), cudaMemcpyDeviceToHost));
                mre.resize(ctx->nDqMembers);
                mk.resize(ctx->nDqMembers);
                CUDA_CHECK(cudaMemcpy(mre.data(), ctx->dDqMemberRlEntry.p, mre.size() * sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(mk.data(), ctx->dDqMemberKey.p, mk.size() * sizeof(double), cudaMemcpyDeviceToHost));
            }
            std::vector<char> lt(static_cast<size_t>(std::max(ctx->nrl, 1)) * nS, 0);
            if (ctx->nrl > 0 && ctx->dRlLimitType.p) {
                CUDA_CHECK(cudaMemcpy(lt.data(), ctx->dRlLimitType.p, static_cast<size_t>(ctx->nrl) * nS, cudaMemcpyDeviceToHost));
            }
            for (int c = 0; c < ctx->ndq; c++) {
                int d0 = dd[static_cast<size_t>(c) * 6];
                if (d0 > -2 || -2 - d0 != wantRow) continue;
                int kM = km.empty() ? -1 : km[c];
                int kRe = kM >= 0 ? mre[kM] : -1;
                int jM = jm.empty() ? -1 : jm[c];
                int jRe = jM >= 0 && !mre.empty() ? mre[jM] : -1;
                std::fprintf(stderr, "    DQROW2 outer=%d it=%d row=%d c=%d kind=%d elem=%d side=%d w=%.6g dests=[%d %d %d %d %d %d]"
                             " jMember=%d jKey=%.6g jLt=%d kMember=%d kKey=%.6g kRlEntry=%d kLimitType=%d nDqDyn=%d nDqInact=%d rowMode=%d\n",
                             outer, it, wantRow, c, dk[c], de[c], ds[c], dw[c], dd[c * 6], dd[c * 6 + 1], dd[c * 6 + 2],
                             dd[c * 6 + 3], dd[c * 6 + 4], dd[c * 6 + 5], jM, jM >= 0 && !mk.empty() ? mk[jM] : 0.0,
                             jRe >= 0 ? (int) lt[static_cast<size_t>(s) * ctx->nrl + jRe] : -1, kM,
                             kM >= 0 ? mk[kM] : 0.0, kRe, kRe >= 0 ? (int) lt[static_cast<size_t>(s) * ctx->nrl + kRe] : -1, ctx->nDqDyn, ctx->nDqInact,
                             mode[static_cast<size_t>(s) * n + wantRow]);
            }
        }
        // OLF_GPU_VALS_DUMP_ROW: every closed-branch output whose frozen or ALTERNATE dest is the row, with the
        // per-scenario facts the scatter decides on (the select row's mode, the element's disabled flag, its
        // relocation group), for slot 0.
        if (static const char* cbRowEnv = std::getenv("OLF_GPU_VALS_DUMP_ROW"); cbRowEnv != nullptr && s == 0 && ctx->st->cb.ne > 0) {
            // comma-separated rows: "12488,12490" replays both (the pilot row and the carrier of one group)
            std::vector<int> wantRows;
            for (const char* q = cbRowEnv; *q != '\0';) {
                wantRows.push_back(std::atoi(q));
                while (*q != '\0' && *q != ',') q++;
                if (*q == ',') q++;
            }
            for (int wantRow : wantRows) {
            int slotDev = 0;
            size_t nOut = static_cast<size_t>(ctx->st->cb.ne) * CB_NOUT;
            std::vector<int> cdest(nOut), calt(nOut, -1), csel(nOut, -1), cgrp(nOut, -1);
            CUDA_CHECK(cudaMemcpy(cdest.data(), ctx->st->cb.dest.p, nOut * sizeof(int), cudaMemcpyDeviceToHost));
            if (ctx->st->cb.altDest.p) CUDA_CHECK(cudaMemcpy(calt.data(), ctx->st->cb.altDest.p, nOut * sizeof(int), cudaMemcpyDeviceToHost));
            if (ctx->st->cb.selRow.p) CUDA_CHECK(cudaMemcpy(csel.data(), ctx->st->cb.selRow.p, nOut * sizeof(int), cudaMemcpyDeviceToHost));
            if (ctx->st->cb.relocGroup.p) CUDA_CHECK(cudaMemcpy(cgrp.data(), ctx->st->cb.relocGroup.p, nOut * sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<char> dis(ctx->nElements, 0);
            if (ctx->dDisabled.p) CUDA_CHECK(cudaMemcpy(dis.data(), static_cast<const char*>(ctx->dDisabled.p) + static_cast<size_t>(slotDev) * ctx->nElements, ctx->nElements, cudaMemcpyDeviceToHost));
            std::vector<int> modeDev(n);
            CUDA_CHECK(cudaMemcpy(modeDev.data(), static_cast<const int*>(ctx->dMode.p) + static_cast<size_t>(slotDev) * n, n * sizeof(int), cudaMemcpyDeviceToHost));
            auto rowOfSlot = [&](int d) { if (d <= -2) return -2 - d; if (d < 0) return -1; int lo = 0, hi = n; while (hi - lo > 1) { int mid = (lo + hi) / 2; if (rowPtr[mid] <= d) lo = mid; else hi = mid; } return lo; };
            {
                std::fprintf(stderr, "    CBROW st=%p nQr=%d nS=%d groupsOf(%d):", (void*) ctx->st.get(), ctx->st->nQr, nS, wantRow);
                for (int g = 0; g < ctx->st->nQr; g++) {
                    for (int c = ctx->st->hQrPtr[g]; c < ctx->st->hQrPtr[g + 1]; c++) {
                        if (ctx->st->hQrRows[c] == wantRow) {
                            // the PILOT-SELF gate inputs of that group for the replayed scenario: the pilot entry's
                            // limit type (the gate is "unpinned = skip the raw q") and the gate value itself
                            int pe = -1; char plt = -1, psk = -1;
                            if (ctx->dPsPilotEntry.p) CUDA_CHECK(cudaMemcpy(&pe, static_cast<const int*>(ctx->dPsPilotEntry.p) + g, sizeof(int), cudaMemcpyDeviceToHost));
                            if (pe >= 0 && ctx->dRlLimitType.p) CUDA_CHECK(cudaMemcpy(&plt, static_cast<const char*>(ctx->dRlLimitType.p) + static_cast<size_t>(slotDev) * ctx->nrl + pe, 1, cudaMemcpyDeviceToHost));
                            if (ctx->dPsSkip.p) CUDA_CHECK(cudaMemcpy(&psk, static_cast<const char*>(ctx->dPsSkip.p) + static_cast<size_t>(slotDev) * ctx->st->nQr + g, 1, cudaMemcpyDeviceToHost));
                            std::fprintf(stderr, " g%d[cand %d pilotEntry=%d lt=%d psSkip=%d]", g, c - ctx->st->hQrPtr[g], pe, (int) plt, (int) psk);
                        }
                    }
                }
                std::fprintf(stderr, "\n");
            }
            std::vector<int> cix(static_cast<size_t>(ctx->st->cb.ne) * 12);
            CUDA_CHECK(cudaMemcpy(cix.data(), ctx->st->cb.ix.p, cix.size() * sizeof(int), cudaMemcpyDeviceToHost));
            for (size_t o = 0; o < nOut; o++) {
                int rf = rowOfSlot(cdest[o]), ra = rowOfSlot(calt[o]);
                if (rf != wantRow && ra != wantRow) continue;
                int e = static_cast<int>(o / CB_NOUT), k = static_cast<int>(o % CB_NOUT);
                int sr = csel[o];
                std::fprintf(stderr, "    CBROW row=%d slot=%d elem=%d out=%d frozenRow=%d altRow=%d sel=%d modeAtSel=%d disabled=%d group=%d ix=[%d %d %d %d]",
                             wantRow, slotDev, e, k, rf, ra, sr, sr >= 0 ? modeDev[sr] : (sr <= -2 ? modeDev[-2 - sr] : -99),
                             (int) dis[ctx->cbOffset + e], cgrp[o], cix[e * 12], cix[e * 12 + 1], cix[e * 12 + 2], cix[e * 12 + 3]);
                if (cgrp[o] >= 0) {
                    int g = cgrp[o];
                    std::fprintf(stderr, " groupRows=");
                    for (int c = ctx->st->hQrPtr[g]; c < ctx->st->hQrPtr[g + 1]; c++) std::fprintf(stderr, "%d(m%d),", ctx->st->hQrRows[c], modeDev[ctx->st->hQrRows[c]]);
                }
                std::fprintf(stderr, "\n");
            }
            }
        }
        // ...and WHICH equations this slot cannot satisfy: the three largest |F|. A scenario that runs its whole
        // Newton budget without converging is not a numerical mystery — it is a handful of rows whose residual
        // never falls, and naming them names the defect.
        char worstF[160] = "";
        {
            std::vector<double> hfv(n);
            CUDA_CHECK(cudaMemcpy(hfv.data(), static_cast<const double*>(ctx->dF.p) + (size_t) s * n,
                                  n * sizeof(double), cudaMemcpyDeviceToHost));
            int w[3] = {-1, -1, -1};
            for (int r = 0; r < n; r++) {
                for (int t = 0; t < 3; t++) {
                    if (w[t] < 0 || std::fabs(hfv[r]) > std::fabs(hfv[w[t]])) {
                        for (int u = 2; u > t; u--) w[u] = w[u - 1];
                        w[t] = r;
                        break;
                    }
                }
            }
            int off = 0;
            for (int t = 0; t < 3 && w[t] >= 0; t++) {
                off += std::snprintf(worstF + off, sizeof(worstF) - off, " %d:%.3e", w[t], hfv[w[t]]);
            }
        }
        std::fprintf(stderr, "VALS_SCAN worstF=[%s ]\n", worstF);
        std::fprintf(stderr, "VALS_SCAN outer=%d it=%d slot=%d nnz=%d finite=%s maxAbs=%.6e at row=%d col=%d"
                             " | noDiagSlot=%d tinyDiag=%d firstBadRow=%d nHaz=%d dupRows=%d,%d"
                             " modes[0..4]=%d,%d,%d,%d,%d deadCols=%d firstDeadCol=%d mode4Rows=[%s]\n",
                     outer, it, s, nnz, bad < 0 ? "yes" : "NO", maxAbs, maxRow, maxCol,
                     noDiagSlot, tinyDiag, firstBad, ctx->st->nHaz, dupA, dupB,
                     modeCount[0], modeCount[1], modeCount[2], modeCount[3], modeCount[4],
                     deadCols, firstDeadCol, m4);
    }
}

/** One scenario's linear system AS SOLVED, written whole: the row-repaired Jacobian (CSR triplets), the mismatch F it
 *  was built at, the repaired right-hand side B, the step dx the factorization returned and each row's mode. Read by
 *  the KLU re-solve probe and the structural-matching script (OLF_GPU_DX_DUMP / OLF_GPU_DX_DUMP_ON_REJECT). */
void dumpSlotSystem(BatchSolverContext* ctx, int n, int nnz, int slot, const char* path) {
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    std::vector<double> hv(nnz);
    std::vector<double> hd(n);
    std::vector<double> hff(n);
    std::vector<int> hrp(n + 1);
    std::vector<int> hci(nnz);
    std::vector<int> hmode(n);
    std::vector<double> hb(n);                               // the RHS as SOLVED (row-repaired, see hazRepairRhs)
    CUDA_CHECK(cudaMemcpy(hb.data(), ctx->dB.template as<double>() + (size_t) slot * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hv.data(), ctx->dVals.template as<double>() + (size_t) slot * nnz, (size_t) nnz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hd.data(), ctx->dDx.template as<double>() + (size_t) slot * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hff.data(), ctx->dF.template as<double>() + (size_t) slot * n, (size_t) n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hrp.data(), ctx->st->dRowPtr.p, (size_t) (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hci.data(), ctx->st->dColIdx.p, (size_t) nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hmode.data(), ctx->dMode.template as<int>() + (size_t) slot * n, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost));
    FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
        std::fprintf(stderr, "OLF_GPU_DX_DUMP cannot open %s\n", path);
        return;
    }
    std::fprintf(f, "N %d %d\n", n, nnz);
    for (int i = 0; i < n; i++) {
        std::fprintf(f, "F %d %.17e\n", i, hff[i]);
        std::fprintf(f, "DX %d %.17e\n", i, hd[i]);
        std::fprintf(f, "B %d %.17e\n", i, hb[i]);
        std::fprintf(f, "MODE %d %d\n", i, hmode[i]);
        for (int k = hrp[i]; k < hrp[i + 1]; k++) {
            std::fprintf(f, "J %d %d %.17e\n", i, hci[k], hv[k]);
        }
    }
    std::fclose(f);
    std::fprintf(stderr, "OLF_GPU_DX_DUMP wrote %s (n=%d nnz=%d slot=%d)\n", path, n, nnz, slot);
}

}  // namespace olfgpu
