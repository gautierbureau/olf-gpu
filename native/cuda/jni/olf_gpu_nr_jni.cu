// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// FULL Newton-Raphson AC power flow on the GPU, behind ONE JNI call. Java hands over
// the network data (per-family element parameters + the equation/variable indices the
// scatter tables resolve against), the initial state and the targets; the native side
// then keeps everything resident on the device:
//
//   - the fixed-pattern CSR Jacobian structure is built ONCE on the host from the
//     generated scatter tables (closed-branch, open-branch, shunt, HVDC AC emulation
//     + the slack/PV identity rows) — cuDSS analyses it once;
//   - each iteration, the four generated fused kernels (evalClosedBranch /
//     evalOpenBranch / evalShunt / evalHvdc) fill the CSR values AND the mismatch F
//     directly in device memory, reading the device state vector through the
//     precomputed index arrays (no host repack);
//   - cuDSS REFACTORIZATION + SOLVE produce dx on the device, a kernel applies
//     x += dx, and only the n-double mismatch crosses PCIe for the convergence check.
//
// Orientation: J is true CSR with row = equation column, col = variable row (OLF's
// indices), so cuDSS solves J dx = -F directly — no transposition games (contrast
// powsybl-math's CSC-as-CSR convention, which exists to serve solveTransposed).
//
// The host-built destination array folds the scatter ROLE in, so the kernels never
// index the constexpr scatter tables:  dest >= 0 -> J values slot;  dest == -1 ->
// skip;  dest <= -2 -> mismatch row (-2 - dest).
//
// Java: com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver
//   static native double[] solve(double[] x0, double[] target, int maxIter, double tol,
//       double[] cbIn, int[] cbIdx, double[] obIn, int[] obIdx,
//       double[] shIn, int[] shIdx, double[] hvIn, int[] hvIdx, int[] idIdx)
//     x0/target: length n (state by variable row / target by equation column)
//     cbIn nbr*8  (y g1 g2 b1 b2 ksi r1 a1)   cbIdx nbr*12 (P1 Q1 P2 Q2 ph1 v1 ph2 v2 a1col r1col qAlt1 qAlt2)
//     obIn nob*7  (y ksi g1 b1 g2 b2 r1)      obIdx nob*6 (P1 Q1 P2 Q2 v1 v2)
//     shIn nsh*2  (g b)                       shIdx nsh*3 (P Q v)
//     hvIn nhv*5  (p0 k lf1 lf2 r)            hvIdx nhv*4 (P1 P2 ph1 ph2)
//     idIdx nid*2 (eqRow varCol)              -1 = absent equation / variable
//     returns double[n + 2] = converged state, iterations, final ||F||inf
//
// This library links NVIDIA cuDSS DYNAMICALLY (never bundled, same licensing posture
// as powsybl-math-native's libmathcudss); cudart is linked statically.
#include <execinfo.h>
#include <jni.h>
#include <cuda_runtime.h>
#include <cudss.h>
#include <cublas_v2.h>

#include <algorithm>
#include <unordered_map>
#include <map>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <map>
#include <cstring>
#include <vector>

#include "closed_branch.cuh"
#include "closed_branch_scatter.h"
#include "open_branch.cuh"
#include "open_branch_scatter.h"
#include "shunt.cuh"
#include "shunt_scatter.h"
#include "hvdc.cuh"
#include "hvdc_scatter.h"
#include "vdl.cuh"

#include "common/loop_kinds.cuh"
#include "common/stats.cuh"

using namespace powsybl::openloadflow;

#include "common/check.cuh"
#include "common/device_buf.cuh"
#include "common/launch.cuh"
#include "diagnostics/driver_traces.cuh"
#include "diagnostics/matrix_scan.cuh"
#include "equations/branch_fill.cuh"
#include "equations/bus_injection_fill.cuh"
#include "equations/distribution_fill.cuh"
#include "equations/fill_launch.cuh"
#include "equations/scatter.cuh"
#include "equations/scenario_overrides.cuh"
#include "equations/zero_impedance_fill.cuh"
#include "outerloop/distributed_slack.cuh"
#include "outerloop/forced_target_q.cuh"
#include "outerloop/hvdc_ac_emulation.cuh"
#include "outerloop/incremental_flow_control.cuh"
#include "outerloop/incremental_shunt_voltage_control.cuh"
#include "outerloop/incremental_transformer_voltage_control.cuh"
#include "outerloop/phase_control.cuh"
#include "outerloop/pilot_voltage.cuh"
#include "outerloop/reactive_capability_curve.cuh"
#include "outerloop/reactive_capability_curves.cuh"
#include "outerloop/reactive_limits.cuh"
#include "outerloop/reactive_limits_seed.cuh"
#include "outerloop/remote_voltage_control.cuh"
#include "outerloop/secondary_voltage_control.cuh"
#include "outerloop/sensitivity.cuh"
#include "outerloop/shunt_voltage_control.cuh"
#include "outerloop/slack_participants.cuh"
#include "outerloop/transformer_voltage_control.cuh"
#include "outerloop/voltage_monitoring.cuh"
#include "solver/batch_context.cuh"
#include "solver/compact_solve.cuh"
#include "solver/convergence.cuh"
#include "solver/driver_pass.cuh"
#include "solver/driver_steps.cuh"
#include "solver/iterative_refinement.cuh"
#include "solver/linear_solver_settings.cuh"
#include "solver/newton_iteration.cuh"
#include "solver/realistic_voltage_check.cuh"
#include "solver/solve_seed.cuh"
#include "solver/state_vector_scaling.cuh"
#include "solver/zero_diagonal_repair.cuh"
#include "structure/csr_pattern.cuh"
#include "structure/shared_structure.cuh"
#include "solver/outer_loop_driver.cuh"

using namespace olfgpu;

extern "C" JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_createStructure(
        JNIEnv* env, jclass, jint n,
        jdoubleArray jcbIn, jintArray jcbIdx, jdoubleArray jobIn, jintArray jobIdx,
        jdoubleArray jshIn, jintArray jshIdx, jdoubleArray jhvIn, jintArray jhvIdx,
        jintArray jdqElem, jintArray jdqSide, jdoubleArray jdqWeight, jintArray jdqRow, jintArray jdqKind,
        jdoubleArray jziIn, jintArray jziIdx,
        jintArray jdrRow, jintArray jdrCol, jdoubleArray jdrCoef,
        jintArray jdbRow, jintArray jdbCol, jdoubleArray jdbCoef, jintArray jcbPcSide,
        jdoubleArray jvdlIn, jintArray jvdlIdx, jintArray jqrPtr, jintArray jqrRows,
        jintArray jobQAlt, jintArray jziQAlt, jintArray jqrDepBase, jintArray jqrBaseTargetRow, jintArray jdepGroupOfRow) {
    jdouble* cbIn = env->GetDoubleArrayElements(jcbIn, nullptr);
    jint* cbIdx = env->GetIntArrayElements(jcbIdx, nullptr);
    jdouble* obIn = env->GetDoubleArrayElements(jobIn, nullptr);
    jint* obIdx = env->GetIntArrayElements(jobIdx, nullptr);
    jdouble* shIn = env->GetDoubleArrayElements(jshIn, nullptr);
    jint* shIdx = env->GetIntArrayElements(jshIdx, nullptr);
    jdouble* hvIn = env->GetDoubleArrayElements(jhvIn, nullptr);
    jint* hvIdx = env->GetIntArrayElements(jhvIdx, nullptr);
    jint* dqElem = env->GetIntArrayElements(jdqElem, nullptr);
    jint* dqSide = env->GetIntArrayElements(jdqSide, nullptr);
    jdouble* dqWeight = env->GetDoubleArrayElements(jdqWeight, nullptr);
    jint* dqRow = env->GetIntArrayElements(jdqRow, nullptr);
    jint* dqKind = env->GetIntArrayElements(jdqKind, nullptr);
    jdouble* ziIn = env->GetDoubleArrayElements(jziIn, nullptr);
    jint* ziIdx = env->GetIntArrayElements(jziIdx, nullptr);
    jint* drRow = env->GetIntArrayElements(jdrRow, nullptr);
    jint* drCol = env->GetIntArrayElements(jdrCol, nullptr);
    jdouble* drCoef = env->GetDoubleArrayElements(jdrCoef, nullptr);
    jint* dbRow = env->GetIntArrayElements(jdbRow, nullptr);
    jint* dbCol = env->GetIntArrayElements(jdbCol, nullptr);
    jdouble* dbCoef = env->GetDoubleArrayElements(jdbCoef, nullptr);
    jint* cbPcSide = env->GetIntArrayElements(jcbPcSide, nullptr);
    jint* qrPtr = env->GetIntArrayElements(jqrPtr, nullptr);
    jint* qrRows = env->GetIntArrayElements(jqrRows, nullptr);
    int nQr = env->GetArrayLength(jqrPtr) - 1;               // groups; qrPtr = [0] when none
    int nQrRows = env->GetArrayLength(jqrRows);
    jdouble* vdlIn = env->GetDoubleArrayElements(jvdlIn, nullptr);
    jint* vdlIdx = env->GetIntArrayElements(jvdlIdx, nullptr);
    jint* obQAlt = env->GetIntArrayElements(jobQAlt, nullptr);
    jint* ziQAlt = env->GetIntArrayElements(jziQAlt, nullptr);
    jint* qrDepBase = env->GetIntArrayElements(jqrDepBase, nullptr);
    jint* qrBaseTargetRow = env->GetIntArrayElements(jqrBaseTargetRow, nullptr);
    jint* depGroupOfRow = env->GetIntArrayElements(jdepGroupOfRow, nullptr);

    int ncb = env->GetArrayLength(jcbIdx) / 12;             // cbIdx stride 12 (…a1col r1col qAlt1 qAlt2)
    int nob = env->GetArrayLength(jobIdx) / 6;
    int nsh = env->GetArrayLength(jshIdx) / 5;             // shIdx stride 5 (P Q v bcol qAlt)
    int nhv = env->GetArrayLength(jhvIdx) / 4;
    int ndq = env->GetArrayLength(jdqElem);
    int nzi = env->GetArrayLength(jziIdx) / 11;
    int ndr = env->GetArrayLength(jdrRow);
    int ndb = env->GetArrayLength(jdbRow);
    int nvdl = env->GetArrayLength(jvdlIdx) / 2;             // vdlIdx stride 2 (row, vCol)

    std::string error;
    SharedStructurePtr* st = nullptr;
    try {
        // DEFENSIVE input validation (an intermittent SIGSEGV at buildStructure+0xe43 — three
        // hs_err logs 2026-09-06 — is a one-past-the-end host read that only faults at a page
        // boundary; a named exception routes to the CPU fallback and identifies the bad input).
        auto checkRange = [&](const char* what, const jint* a, size_t len, size_t stride,
                              std::initializer_list<size_t> slots, int lo, int hi) {
            for (size_t e = 0; e < len; e++) {
                for (size_t sl : slots) {
                    int v = a[e * stride + sl];
                    if (v < lo || v >= hi) {
                        throw std::runtime_error(std::string("bad ") + what + " at elem " + std::to_string(e)
                                + " slot " + std::to_string(sl) + ": " + std::to_string(v)
                                + " (allowed [" + std::to_string(lo) + ", " + std::to_string(hi) + "))");
                    }
                }
            }
        };
        // rows/cols may be -1 = absent; anything else must be a valid row/col index < n
        checkRange("cbIdx", cbIdx, ncb, 12, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, -1, n);
        // qAlt slots ([10]/[11] cb, [4] sh) also carry the GROUP encoding -2-groupId (plan-A6)
        checkRange("cbIdx.qAlt", cbIdx, ncb, 12, {10, 11}, -1 - nQr, n);
        checkRange("obIdx", obIdx, nob, 6, {0, 1, 2, 3, 4, 5}, -1, n);
        checkRange("obQAlt", obQAlt, nob, 2, {0, 1}, -1 - nQr, n);
        checkRange("shIdx", shIdx, nsh, 5, {0, 1, 2, 3}, -1, n);
        checkRange("shIdx.qAlt", shIdx, nsh, 5, {4}, -1 - nQr, n);
        checkRange("qrRows", qrRows, nQrRows, 1, {0}, 0, n);
        for (int g = 0; g < nQr; g++) {
            if (qrPtr[g] < 0 || qrPtr[g + 1] <= qrPtr[g] + 1 || qrPtr[g + 1] > nQrRows) {
                throw std::runtime_error("bad qrPtr group " + std::to_string(g));
            }
        }
        checkRange("hvIdx", hvIdx, nhv, 4, {0, 1, 2, 3}, -1, n);
        checkRange("dqRow", dqRow, ndq, 1, {0}, 0, n);
        checkRange("ziIdx", ziIdx, nzi, 11, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, -1, n);
        // slot 10 is the FLAGS word, not a row: bit 0 spanning-tree, bits 1-2 the relocated child side,
        // bit 3 "the side came from the coupler-tree walk". Checking it against n rejected a valid
        // structure on any network with fewer than 14 rows once the relocation went on by default
        // ("bad ziIdx at elem 0 slot 10: 13 (allowed [-1, 10))", four fixtures).
        checkRange("ziIdx.flags", ziIdx, nzi, 11, {10}, 0, 16);
        checkRange("ziQAlt", ziQAlt, nzi, 2, {0, 1}, -1 - nQr, n);
        checkRange("drRow", drRow, ndr, 1, {0}, 0, n);
        checkRange("drCol", drCol, ndr, 1, {0}, 0, n);
        checkRange("dbRow", dbRow, ndb, 1, {0}, 0, n);
        checkRange("dbCol", dbCol, ndb, 1, {0}, 0, n);
        checkRange("vdlIdx", vdlIdx, nvdl, 2, {0, 1}, 0, n);
        for (int e = 0; e < ncb; e++) {                      // encoded (a1Col << 1) | side, or -1
            if (cbPcSide[e] >= 0 && (cbPcSide[e] >> 1) >= n) {
                throw std::runtime_error("bad cbPcSide at elem " + std::to_string(e)
                        + ": a1Col " + std::to_string(cbPcSide[e] >> 1) + " >= n " + std::to_string(n));
            }
        }
        st = new SharedStructurePtr(buildStructure(n, ncb, cbIn, cbIdx, nob, obIn, obIdx,
                                                   nsh, shIn, shIdx, nhv, hvIn, hvIdx,
                                                   ndq, dqElem, dqSide, dqWeight, dqRow, dqKind,
                                                   nzi, ziIn, ziIdx,
                                                   ndr, drRow, drCol, drCoef,
                                                   ndb, dbRow, dbCol, dbCoef, cbPcSide,
                                                   nvdl, vdlIn, vdlIdx, nQr, qrPtr, qrRows,
                                                   obQAlt, ziQAlt, qrDepBase, qrBaseTargetRow, depGroupOfRow));
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    env->ReleaseDoubleArrayElements(jcbIn, cbIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jcbIdx, cbIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jcbPcSide, cbPcSide, JNI_ABORT);
    env->ReleaseIntArrayElements(jqrPtr, qrPtr, JNI_ABORT);
    env->ReleaseIntArrayElements(jqrRows, qrRows, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jobIn, obIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jobIdx, obIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jshIn, shIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jshIdx, shIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jhvIn, hvIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jhvIdx, hvIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqElem, dqElem, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqSide, dqSide, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdqWeight, dqWeight, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqRow, dqRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdqKind, dqKind, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jziIn, ziIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jziIdx, ziIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jdrRow, drRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdrCol, drCol, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdrCoef, drCoef, JNI_ABORT);
    env->ReleaseIntArrayElements(jdbRow, dbRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdbCol, dbCol, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdbCoef, dbCoef, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jvdlIn, vdlIn, JNI_ABORT);
    env->ReleaseIntArrayElements(jvdlIdx, vdlIdx, JNI_ABORT);
    env->ReleaseIntArrayElements(jobQAlt, obQAlt, JNI_ABORT);
    env->ReleaseIntArrayElements(jziQAlt, ziQAlt, JNI_ABORT);
    env->ReleaseIntArrayElements(jqrDepBase, qrDepBase, JNI_ABORT);
    env->ReleaseIntArrayElements(jqrBaseTargetRow, qrBaseTargetRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jdepGroupOfRow, depGroupOfRow, JNI_ABORT);

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU structure creation failed: " + error).c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(st);
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_destroyStructure(
        JNIEnv*, jclass, jlong handle) {
    // Drops the Java-side owner reference only: contexts hold their own shared_ptr copies,
    // so the device-resident structure lives until the last context is destroyed.
    delete reinterpret_cast<SharedStructurePtr*>(handle);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_createBatchContext(
        JNIEnv* env, jclass, jlong structureHandle, jint nScenarios, jint mode,
        jintArray jrlVrow, jdoubleArray jrlMinQ, jdoubleArray jrlMaxQ, jdoubleArray jrlLoadQ,
        jdoubleArray jrlVtarget, jbyteArray jrlMonitored, jdoubleArray jrowSlope,
        jintArray jdsPhiRow, jdoubleArray jdsFactor,
        jint jdsSlackPhiRow, jdouble jdsSlackTargetP,
        jdoubleArray jdsBaseTargetP, jdoubleArray jdsMaxDelta, jdoubleArray jdsMinDelta,
        jint jdsMode, jdoubleArray jdsMarginUp, jdoubleArray jdsMarginDown,
        jdouble jdsSlackMarginUp, jdouble jdsSlackMarginDown) {
    auto* st = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    jint* rlVrow = env->GetIntArrayElements(jrlVrow, nullptr);
    jdouble* rlMinQ = env->GetDoubleArrayElements(jrlMinQ, nullptr);
    jdouble* rlMaxQ = env->GetDoubleArrayElements(jrlMaxQ, nullptr);
    jdouble* rlLoadQ = env->GetDoubleArrayElements(jrlLoadQ, nullptr);
    jdouble* rlVtarget = env->GetDoubleArrayElements(jrlVtarget, nullptr);
    jbyte* rlMonitored = env->GetByteArrayElements(jrlMonitored, nullptr);
    jdouble* rowSlope = env->GetDoubleArrayElements(jrowSlope, nullptr);
    jint* dsPhiRow = env->GetIntArrayElements(jdsPhiRow, nullptr);
    jdouble* dsFactor = env->GetDoubleArrayElements(jdsFactor, nullptr);
    jdouble* dsBaseTargetP = env->GetDoubleArrayElements(jdsBaseTargetP, nullptr);
    jdouble* dsMaxDelta = env->GetDoubleArrayElements(jdsMaxDelta, nullptr);
    jdouble* dsMinDelta = env->GetDoubleArrayElements(jdsMinDelta, nullptr);
    jdouble* dsMarginUp = env->GetDoubleArrayElements(jdsMarginUp, nullptr);
    jdouble* dsMarginDown = env->GetDoubleArrayElements(jdsMarginDown, nullptr);

    int nrl = env->GetArrayLength(jrlVrow);
    int nds = env->GetArrayLength(jdsPhiRow);

    std::string error;
    BatchSolverContext* ctx = nullptr;
    try {
        if (st == nullptr || !*st) {
            throw std::runtime_error("invalid structure handle");
        }
        ctx = createBatchContext(*st, nScenarios, mode,
                                 nrl, rlVrow, rlMinQ, rlMaxQ, rlLoadQ, rlVtarget, rlMonitored, rowSlope,
                                 nds, dsPhiRow, dsFactor, jdsSlackPhiRow, jdsSlackTargetP,
                                 dsBaseTargetP, dsMaxDelta, dsMinDelta,
                                 jdsMode, dsMarginUp, dsMarginDown, jdsSlackMarginUp, jdsSlackMarginDown);
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    env->ReleaseIntArrayElements(jrlVrow, rlVrow, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlMinQ, rlMinQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlMaxQ, rlMaxQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlLoadQ, rlLoadQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrlVtarget, rlVtarget, JNI_ABORT);
    env->ReleaseByteArrayElements(jrlMonitored, rlMonitored, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrowSlope, rowSlope, JNI_ABORT);
    env->ReleaseIntArrayElements(jdsPhiRow, dsPhiRow, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsFactor, dsFactor, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsBaseTargetP, dsBaseTargetP, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMaxDelta, dsMaxDelta, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMinDelta, dsMinDelta, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMarginUp, dsMarginUp, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jdsMarginDown, dsMarginDown, JNI_ABORT);

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU batched context creation failed: " + error).c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(ctx);
}

// OLF_GPU_PHASE_TIME: split one solveBatchContext JNI call into the three things it does —
// marshalling the Java arrays in, the solve itself, and packing the per-scenario result arrays out.
// The Java-side timer sees only the total, and on a 642-scenario chunk the total carries roughly a
// second of fixed cost that the Newton iterations do not explain.
static bool jniPhaseTime() {
    static const bool on = std::getenv("OLF_GPU_PHASE_TIME") != nullptr;
    return on;
}

static long long nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

static long long marshalNs = 0;
static long long solveNs = 0;
static long long packNs = 0;
static long long calls = 0;

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveBatchContext(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jx0Base, jdoubleArray jtargetBase,
        jobjectArray jrowMode, jintArray jbaseRowMode, jobjectArray jdisabledMask,
        jintArray jtargetOverrideIndex, jdoubleArray jtargetOverrideValue,
        jintArray jdsOverrideBus, jdoubleArray jdsOverrideFactorDelta, jdoubleArray jdsOverrideBaseDelta,
        jdoubleArray jdsOverrideMaxDelta, jdoubleArray jdsOverrideMinDelta, jdoubleArray jdsAnchorShift,
        jdoubleArray jdsSlackShift, jintArray jrlVtOvSlotIdx, jdoubleArray jrlVtOvVal,
        jintArray jrlQOvSlotEntry, jdoubleArray jrlQOvMinMax,
        jintArray jcvOvSlotEntry, jdoubleArray jcvOvStepScale, jint maxIter, jdouble tol,
        jdouble reactiveLimitTol, jdouble slackDistTol, jdouble convEpsPerEq,
        jint lsMode, jdouble maxDv, jdouble maxDphi, jintArray jvarType, jdoubleArray jx0Full,
        jintArray jShuntOvSlotElem, jdoubleArray jShuntOvGb,
        jintArray jRlStateIn, jintArray jRlStateOut,
        jintArray jR1OvSlotElem, jdoubleArray jR1OvVal,
        jint maxOuterProductive, jint rlMaxSwitch, jint dsRejectResidual, jint skipRlPhase,
        jint nziBase, jintArray jZiOnSlotElem, jintArray jZiOffSlotElem) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->solveCount++;
    }
    // CONTRACT: reactive-limit entries require OLF's forced-PV election keys. switchPvPq never lets
    // a network reach zero PV buses ("all PV buses should switch PQ, strongest one will stay PV"),
    // and rlForcedPvElectBatch is the only thing on the device that reproduces it. A context that
    // installed RL entries without the keys would pin every candidate and silently diverge from the
    // CPU (measured 1.9 kV on the forced-PV network). GpuBatchOuterLoopTables.install always sets
    // them, so reaching this is a table-installation bug — say so instead of routing the scenario
    // away and hiding it.
    if (ctx != nullptr && ctx->nrl > 0 && !ctx->hasRlElection) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "solveBatchContext: reactive-limit entries installed without the forced-PV "
                      "election keys (setBatchRlElection) — OLF's strongest-bus-stays-PV rule "
                      "cannot be reproduced on the device");
        return nullptr;
    }
    long long tMarshal0 = jniPhaseTime() ? nowNs() : 0;
    std::vector<int> r1OvSlotElemVec;
    std::vector<double> r1OvValVec;
    if (jR1OvSlotElem != nullptr) {
        jsize rn = env->GetArrayLength(jR1OvSlotElem);
        r1OvSlotElemVec.resize(rn);
        env->GetIntArrayRegion(jR1OvSlotElem, 0, rn, r1OvSlotElemVec.data());
        r1OvValVec.resize(static_cast<size_t>(rn) * 4);      // full 8-double cbIn row per (slot, elem) entry
        env->GetDoubleArrayRegion(jR1OvVal, 0, rn * 4, r1OvValVec.data());
    }
    std::vector<int> ziOffSlotElemVec;
    if (jZiOffSlotElem != nullptr) {
        jsize zn = env->GetArrayLength(jZiOffSlotElem);
        ziOffSlotElemVec.resize(zn);
        env->GetIntArrayRegion(jZiOffSlotElem, 0, zn, ziOffSlotElemVec.data());
    }
    std::vector<int> ziOnSlotElemVec;
    if (jZiOnSlotElem != nullptr) {
        jsize zn = env->GetArrayLength(jZiOnSlotElem);
        ziOnSlotElemVec.resize(zn);
        env->GetIntArrayRegion(jZiOnSlotElem, 0, zn, ziOnSlotElemVec.data());
    }
    std::vector<int> rlStateInVec;
    if (jRlStateIn != nullptr) {
        jsize rn = env->GetArrayLength(jRlStateIn);
        rlStateInVec.resize(rn);
        env->GetIntArrayRegion(jRlStateIn, 0, rn, rlStateInVec.data());
    }
    std::vector<int> rlStateOutVec;
    if (jRlStateOut != nullptr) {
        rlStateOutVec.resize(env->GetArrayLength(jRlStateOut));
    }
    std::vector<int> shOvSlotElemVec;
    std::vector<double> shOvGbVec;
    if (jShuntOvSlotElem != nullptr) {
        jsize sn = env->GetArrayLength(jShuntOvSlotElem);
        shOvSlotElemVec.resize(sn);
        env->GetIntArrayRegion(jShuntOvSlotElem, 0, sn, shOvSlotElemVec.data());
        shOvGbVec.resize(sn);
        env->GetDoubleArrayRegion(jShuntOvGb, 0, sn, shOvGbVec.data());
    }
    std::vector<double> x0FullVec;
    if (jx0Full != nullptr) {
        jsize fn = env->GetArrayLength(jx0Full);
        x0FullVec.resize(fn);
        env->GetDoubleArrayRegion(jx0Full, 0, fn, x0FullVec.data());
    }
    std::vector<int> varTypeVec;
    if (jvarType != nullptr) {
        jsize vn = env->GetArrayLength(jvarType);
        varTypeVec.resize(vn);
        env->GetIntArrayRegion(jvarType, 0, vn, varTypeVec.data());
    }

    std::string error;
    BatchResult res;
    try {
        if (ctx == nullptr) {
            throw std::runtime_error("invalid batch solver context");
        }
        int n = ctx->n;
        int nS = ctx->nScenarios;
        int nElem = ctx->nElements;
        if (env->GetArrayLength(jx0Base) != n) {
            throw std::runtime_error("base state length mismatch with batch context");
        }
        if (env->GetArrayLength(jdisabledMask) != nS) {
            throw std::runtime_error("scenario count mismatch with batch context");
        }

        if (env->GetArrayLength(jrowMode) != nS) {
            throw std::runtime_error("rowMode scenario count mismatch with batch context");
        }
        // x0/target are the BASE row (length n), identical for every scenario — read once and tiled
        // across nS on-device. rowMode is PER-SCENARIO ([nS][n]) — read all nS rows. Only the disabled
        // mask is also per-scenario.
        std::vector<double> x0(n);
        std::vector<double> target(n);
        env->GetDoubleArrayRegion(jx0Base, 0, n, x0.data());
        env->GetDoubleArrayRegion(jtargetBase, 0, n, target.data());
        // COMPACT row modes: a scenario whose row IS the base array (the Java side shares that one array for
        // every scenario it does not modify) is not copied; only the others travel, with their slots.
        std::vector<int> rowModeSlots;
        std::vector<int> rowMode;
        for (int s = 0; s < nS; s++) {
            auto mRow = (jintArray) env->GetObjectArrayElement(jrowMode, s);
            if (env->IsSameObject(mRow, jbaseRowMode)) {
                env->DeleteLocalRef(mRow);
                continue;
            }
            if (env->GetArrayLength(mRow) != n) {
                env->DeleteLocalRef(mRow);
                throw std::runtime_error("rowMode row length mismatch with batch context");
            }
            rowModeSlots.push_back(s);
            rowMode.resize(rowModeSlots.size() * (size_t) n);
            env->GetIntArrayRegion(mRow, 0, n, rowMode.data() + (rowModeSlots.size() - 1) * (size_t) n);
            env->DeleteLocalRef(mRow);
        }
        std::vector<int> baseRowMode(n);                     // contingency-free base row mode (length n)
        env->GetIntArrayRegion(jbaseRowMode, 0, n, baseRowMode.data());
        // One region copy per scenario: a jboolean is an unsigned byte holding 0 or 1, the mask's own encoding.
        std::unique_ptr<char[]> disabledBuf(new char[static_cast<size_t>(nElem) * nS]);
        char* disabledData = disabledBuf.get();
        for (int s = 0; s < nS; s++) {
            auto dRow = (jbooleanArray) env->GetObjectArrayElement(jdisabledMask, s);
            int dLen = env->GetArrayLength(dRow);
            int len = dLen < nElem ? dLen : nElem;
            char* dst = disabledData + (size_t) s * nElem;
            env->GetBooleanArrayRegion(dRow, 0, len, reinterpret_cast<jboolean*>(dst));
            if (len < nElem) {
                std::memset(dst + len, 0, nElem - len);
            }
            env->DeleteLocalRef(dRow);
        }

        int nOv = env->GetArrayLength(jtargetOverrideIndex);
        std::vector<int> ovIndex(nOv);
        std::vector<double> ovValue(nOv);
        if (nOv > 0) {
            env->GetIntArrayRegion(jtargetOverrideIndex, 0, nOv, ovIndex.data());
            env->GetDoubleArrayRegion(jtargetOverrideValue, 0, nOv, ovValue.data());
        }

        // Per-scenario distributed-slack reweight: present (length nS) only when a generator-loss scenario
        // sits on a DS-participant bus; empty array → no DS override (pass nullptr to the solver).
        int nDsOv = env->GetArrayLength(jdsOverrideBus);
        std::vector<int> dsOvBus(nDsOv);
        std::vector<double> dsOvFactorDelta(nDsOv), dsOvBaseDelta(nDsOv), dsOvMaxDelta(nDsOv), dsOvMinDelta(nDsOv);
        if (nDsOv > 0) {
            env->GetIntArrayRegion(jdsOverrideBus, 0, nDsOv, dsOvBus.data());
            env->GetDoubleArrayRegion(jdsOverrideFactorDelta, 0, nDsOv, dsOvFactorDelta.data());
            env->GetDoubleArrayRegion(jdsOverrideBaseDelta, 0, nDsOv, dsOvBaseDelta.data());
            env->GetDoubleArrayRegion(jdsOverrideMaxDelta, 0, nDsOv, dsOvMaxDelta.data());
            env->GetDoubleArrayRegion(jdsOverrideMinDelta, 0, nDsOv, dsOvMinDelta.data());
        }
        std::vector<double> dsAnchorVec;
        if (jdsAnchorShift != nullptr) {
            dsAnchorVec.resize(env->GetArrayLength(jdsAnchorShift));
            env->GetDoubleArrayRegion(jdsAnchorShift, 0, static_cast<jsize>(dsAnchorVec.size()), dsAnchorVec.data());
        }
        std::vector<double> dsSlackShiftVec;
        if (jdsSlackShift != nullptr) {
            dsSlackShiftVec.resize(env->GetArrayLength(jdsSlackShift));
            env->GetDoubleArrayRegion(jdsSlackShift, 0, static_cast<jsize>(dsSlackShiftVec.size()), dsSlackShiftVec.data());
        }
        std::vector<int> rlVtOvSlotIdxVec;
        std::vector<double> rlVtOvValVec;
        std::vector<int> rlQOvSlotEntryVec;
        std::vector<double> rlQOvMinMaxVec;
        if (jrlQOvSlotEntry != nullptr && jrlQOvMinMax != nullptr) {
            rlQOvSlotEntryVec.resize(env->GetArrayLength(jrlQOvSlotEntry));
            env->GetIntArrayRegion(jrlQOvSlotEntry, 0, static_cast<jsize>(rlQOvSlotEntryVec.size()), rlQOvSlotEntryVec.data());
            rlQOvMinMaxVec.resize(env->GetArrayLength(jrlQOvMinMax));
            env->GetDoubleArrayRegion(jrlQOvMinMax, 0, static_cast<jsize>(rlQOvMinMaxVec.size()), rlQOvMinMaxVec.data());
        }
        std::vector<int> cvOvSlotEntryVec;
        std::vector<double> cvOvStepScaleVec;
        if (jcvOvSlotEntry != nullptr && jcvOvStepScale != nullptr) {
            cvOvSlotEntryVec.resize(env->GetArrayLength(jcvOvSlotEntry));
            env->GetIntArrayRegion(jcvOvSlotEntry, 0, static_cast<jsize>(cvOvSlotEntryVec.size()), cvOvSlotEntryVec.data());
            cvOvStepScaleVec.resize(env->GetArrayLength(jcvOvStepScale));
            env->GetDoubleArrayRegion(jcvOvStepScale, 0, static_cast<jsize>(cvOvStepScaleVec.size()), cvOvStepScaleVec.data());
        }
        if (jrlVtOvSlotIdx != nullptr && jrlVtOvVal != nullptr) {
            rlVtOvSlotIdxVec.resize(env->GetArrayLength(jrlVtOvSlotIdx));
            env->GetIntArrayRegion(jrlVtOvSlotIdx, 0, static_cast<jsize>(rlVtOvSlotIdxVec.size()), rlVtOvSlotIdxVec.data());
            rlVtOvValVec.resize(env->GetArrayLength(jrlVtOvVal));
            env->GetDoubleArrayRegion(jrlVtOvVal, 0, static_cast<jsize>(rlVtOvValVec.size()), rlVtOvValVec.data());
        }

        if (jniPhaseTime()) {
            marshalNs += nowNs() - tMarshal0;
        }
        long long tSolve0 = jniPhaseTime() ? nowNs() : 0;
        static const int kNoModeSlots[1] = {0};              // non-null: COMPACT mode even when no scenario differs
        res = solveBatchContext(ctx, x0.data(), target.data(), rowMode.data(),
                                rowModeSlots.empty() ? kNoModeSlots : rowModeSlots.data(),
                                static_cast<int>(rowModeSlots.size()), baseRowMode.data(), disabledData,
                                nOv, ovIndex.data(), ovValue.data(),
                                nDsOv > 0 ? dsOvBus.data() : nullptr, dsOvFactorDelta.data(),
                                dsOvBaseDelta.data(), dsOvMaxDelta.data(), dsOvMinDelta.data(),
                                dsAnchorVec.empty() ? nullptr : dsAnchorVec.data(),
                                dsSlackShiftVec.empty() ? nullptr : dsSlackShiftVec.data(),
                                static_cast<int>(rlVtOvValVec.size()),
                                rlVtOvSlotIdxVec.empty() ? nullptr : rlVtOvSlotIdxVec.data(),
                                rlVtOvValVec.empty() ? nullptr : rlVtOvValVec.data(),
                                static_cast<int>(rlQOvMinMaxVec.size() / 5),
                                rlQOvSlotEntryVec.empty() ? nullptr : rlQOvSlotEntryVec.data(),
                                rlQOvMinMaxVec.empty() ? nullptr : rlQOvMinMaxVec.data(),
                                static_cast<int>(cvOvStepScaleVec.size() / 2),
                                cvOvSlotEntryVec.empty() ? nullptr : cvOvSlotEntryVec.data(),
                                cvOvStepScaleVec.empty() ? nullptr : cvOvStepScaleVec.data(), maxIter, tol,
                                reactiveLimitTol, slackDistTol, convEpsPerEq,
                                lsMode, maxDv, maxDphi, varTypeVec.empty() ? nullptr : varTypeVec.data(),
                                x0FullVec.empty() ? nullptr : x0FullVec.data(),
                                static_cast<int>(shOvSlotElemVec.size() / 2),
                                shOvSlotElemVec.empty() ? nullptr : shOvSlotElemVec.data(),
                                shOvGbVec.empty() ? nullptr : shOvGbVec.data(),
                                rlStateInVec.empty() ? nullptr : rlStateInVec.data(),
                                rlStateOutVec.empty() ? nullptr : rlStateOutVec.data(),
                                static_cast<int>(r1OvSlotElemVec.size() / 2),
                                r1OvSlotElemVec.empty() ? nullptr : r1OvSlotElemVec.data(),
                                r1OvValVec.empty() ? nullptr : r1OvValVec.data(),
                                maxOuterProductive, rlMaxSwitch, dsRejectResidual, skipRlPhase,
                                nziBase, static_cast<int>(ziOnSlotElemVec.size() / 2),
                                ziOnSlotElemVec.empty() ? nullptr : ziOnSlotElemVec.data(),
                                static_cast<int>(ziOffSlotElemVec.size() / 2),
                                ziOffSlotElemVec.empty() ? nullptr : ziOffSlotElemVec.data());
        if (jniPhaseTime()) {
            solveNs += nowNs() - tSolve0;
        }
        if (jRlStateOut != nullptr && !rlStateOutVec.empty()) {
            env->SetIntArrayRegion(jRlStateOut, 0, static_cast<jsize>(rlStateOutVec.size()), rlStateOutVec.data());
        }
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU batched Newton-Raphson failed: " + error).c_str());
        return nullptr;
    }

    // Pack per-scenario results: [state(n), iterations, finalMismatch, convergenceStatus].
    long long tPack0 = jniPhaseTime() ? nowNs() : 0;
    int n = ctx->n;
    int nS = ctx->nScenarios;
    jclass doubleArrayClass = env->FindClass("[D");
    // A null from either allocation means the Java HEAP is exhausted and an OutOfMemoryError is pending: return
    // and let it surface in Java. Writing into the null row instead was a SIGSEGV in JNIHandles::resolve that
    // killed the JVM - network.xiidm's full busbar scope, the robust retry packing its results on top of the
    // ~14000 precomputed states.
    jobjectArray out = env->NewObjectArray(nS, doubleArrayClass, nullptr);
    if (out == nullptr) {
        return nullptr;
    }
    for (int s = 0; s < nS; s++) {
        jdoubleArray row = env->NewDoubleArray(n + 3);
        if (row == nullptr) {
            return nullptr;
        }
        env->SetDoubleArrayRegion(row, 0, n, res.x.data() + (size_t) s * n);
        double tail[3] = {static_cast<double>(res.iters[s]), res.norm[s], static_cast<double>(res.status[s])};
        env->SetDoubleArrayRegion(row, n, 3, tail);
        env->SetObjectArrayElement(out, s, row);
        env->DeleteLocalRef(row);
    }
    if (jniPhaseTime()) {
        packNs += nowNs() - tPack0;
        calls++;
        fprintf(stderr, "JNI_PHASE_TIME calls=%lld marshal=%lld ms solve=%lld ms pack=%lld ms "
                "| outerPasses=%lld (solving=%lld) newtonIters=%lld activeIterSum=%lld widthIterSum=%lld compactIters=%lld"
                " | cuDSS analyses main=%lld compaction=%lld | irChecks=%ld irExtraSolves=%ld\n",
                calls, marshalNs / 1000000, solveNs / 1000000, packNs / 1000000,
                gOuterPasses, gSolvePasses, gNewtonIters, gActiveIterSum, gWidthIterSum, gCompactIters,
                gAnalysesMain, gAnalysesCompact, gIrChecks, gIrExtraSteps);
        double tot[4] = {0, 0, 0, 0};
        long long cnt[4] = {0, 0, 0, 0};
        for (const auto& kv : gAnalysisSites) {
            int ph = kv.first / 100000, line = kv.first % 100000;
            int k = ph == 'A' ? 0 : ph == 'F' ? 1 : ph == 'S' ? 2 : 3;
            tot[k] += kv.second.second;
            cnt[k] += kv.second.first;
            fprintf(stderr, "CUDSS_PHASE_SITE %c line=%d count=%lld ms=%.1f\n", ph, line, kv.second.first, kv.second.second);
        }
        for (int k = 1; k < 16; k++) {
            if (gOlHistCount[k] == 0) continue;
            fprintf(stderr, "GPU_OL_HIST passes=%d%s scenarios=%lld firstNrIters=%lld totalNrIters=%lld\n", k,
                    k == 15 ? "+" : "", gOlHistCount[k], gOlHistFirst[k], gOlHistTotal[k]);
        }
        fprintf(stderr, "GPU_OL_HIST notConvergedOnDevice=%lld iters=%lld\n", gOlNotConverged, gOlNotConvergedIters);
        fprintf(stderr, "GPU_OL_TIME first-solve passes=%lld ms=%.1f scenarioSolves=%lld laneIters=%lld widthIters=%lld\n",
                gOlFirstPasses, gOlFirstSolveMs, gOlFirstSolveScen, gOlFirstLaneIters, gOlFirstWidthIters);
        for (int k = 0; k < OL_KINDS; k++) {
            if (gOlReSolvePasses[k] > 0) {
                fprintf(stderr, "GPU_OL_TIME re-solve trigger=%s passes=%lld ms=%.1f scenarioSolves=%lld laneIters=%lld widthIters=%lld\n",
                        olKindName(k), gOlReSolvePasses[k], gOlReSolveMs[k], gOlReSolveScen[k], gOlReSolveLaneIters[k], gOlReSolveWidthIters[k]);
            }
        }
        for (int k = 0; k < OL_KINDS; k++) {
            if (gOlCheckPasses[k] > 0) {
                fprintf(stderr, "GPU_OL_TIME check loop=%s passes=%lld ms=%.1f\n", olKindName(k), gOlCheckPasses[k], gOlCheckMs[k]);
            }
        }
        if (rlReasonsOn()) {
            static const char* names[8] = {"group re-stamp only", "pin", "unpin", "pin+unpin", "limit refresh",
                                           "pin+refresh", "unpin+refresh", "pin+unpin+refresh"};
            fprintf(stderr, "GPU_RL_REASONS stable checks=%lld | entries: pins=%lld unpins=%lld refreshes=%lld\n",
                    gRlStableScen, gRlEntryCodes[1], gRlEntryCodes[2], gRlEntryCodes[3]);
            for (int m = 0; m < 8; m++) {
                if (gRlCheckScen[m] > 0) fprintf(stderr, "GPU_RL_REASONS unstable because %s: %lld\n", names[m], gRlCheckScen[m]);
            }
        }
        fprintf(stderr, "CUDSS_PHASE_TIME analysis=%lld x %.1f ms | factorization=%lld x %.1f ms | solve=%lld x %.1f ms"
                " | factorization+solve=%lld x %.1f ms\n", cnt[0], tot[0], cnt[1], tot[1], cnt[2], tot[2], cnt[3], tot[3]);
    }
    return out;
}

// Read back the batch context's per-scenario TARGET buffer (n * nScenarios doubles) after a solve —
// the device outer loops (distributed slack) mutate targets in place, and the strategy/pair phase needs
// the PARENT contingency's final distributed targets to re-anchor its own distribution (OLF's
// setGeneratorsInitialTargetPToTargetP semantics).
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    size_t nTot = static_cast<size_t>(ctx->n) * ctx->nScenarios;
    std::vector<double> host(nTot);
    cudaMemcpy(host.data(), ctx->dTarget.p, nTot * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(nTot));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(nTot), host.data());
    return out;
}

// The per-scenario ROW MODES as the device ENDED with them, the companion to readBatchTargets.
// A row's equation is (mode, target) together, and the outer-loop kernels rewrite BOTH: the host's
// rowModePerScenario is only what SETUP asked for, so reading a target without its final mode can
// say "the target is right" about a row that is no longer the equation it was set up to be. That is
// the whole failure shape of the device-target-writer class, where a wrong row converges cleanly and
// nothing flags it.
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchRowModes(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    size_t nTot = static_cast<size_t>(ctx->n) * ctx->nScenarios;
    std::vector<int> host(nTot);
    cudaMemcpy(host.data(), ctx->dMode.p, nTot * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(nTot));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(nTot), host.data());
    return out;
}

// Batched SENSITIVITY solve for the incremental outer-loop family (device port of OLF's
// SensitivityContext): J at the given per-scenario states (the same forward system the Newton
// solves — OLF's "solveTransposed" IS the forward direction in powsybl-math's transposed
// storage), factorized once, solved for a unit RHS at each controller's pinned-variable
// identity row (BRANCH_TARGET_RHO1 & co, storage row = rowMap[eqRow]); the requested variable
// columns of each solution are gathered per scenario. MUST be called on a context that has just
// completed solveBatchContext with the SAME per-scenario overrides/row modes — the fill reads
// them from the live device buffers. UBATCH mode (3) only.
std::vector<double> solveBatchSensitivities(BatchSolverContext* ctx, const double* states,
                                            int k, const int* unitEqRows, int m, const int* observeVarCols) {
    if (ctx->mode != 3) {
        throw std::runtime_error("solveBatchSensitivities: UBATCH mode (3) only");
    }
    int n = ctx->st->n;
    int nS = ctx->nScenarios;
    int nnz = ctx->st->nnz;

    CUDA_CHECK(cudaMemcpyAsync(ctx->dX.p, states, (size_t) n * nS * sizeof(double), cudaMemcpyHostToDevice, ctx->stream));
    std::vector<char> allActive(nS, 1);
    CUDA_CHECK(cudaMemcpyAsync(ctx->dActive.p, allActive.data(), nS * sizeof(char), cudaMemcpyHostToDevice, ctx->stream));

    // ---- J fill at the converged states (same launches as the solve loop's full fill) ----
    CUDA_CHECK(cudaMemsetAsync(ctx->dVals.p, 0, (size_t) nnz * nS * sizeof(double), ctx->stream));
    initFBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, ctx->dTarget.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->st->cb.ne) {
        fillClosedBranchBatch<<<grid(ctx->st->cb.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->cb.ne, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->cb.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->st->cb.altDest.as<int>(), ctx->st->cb.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->cb.dynReloc ? 1 : 0, ctx->st->cb.phaseCtl ? ctx->st->cb.pcDest.as<int>() : nullptr, ctx->st->cb.phaseCtl ? ctx->st->cb.pcSel.as<int>() : nullptr, withPilotSelf(relocOf(ctx->st->cb, ctx->st.get()), ctx));
    }
    if (ctx->st->ob.ne) {
        fillOpenBranchBatch<<<grid(ctx->st->ob.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->ob.ne, nS, n, nnz, ctx->nElements, ctx->obOffset, ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->st->ob.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->ob.altDest.as<int>(), ctx->st->ob.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ob.dynReloc ? 1 : 0, withPilotSelf(relocOf(ctx->st->ob, ctx->st.get()), ctx));
    }
    if (ctx->st->sh.ne) {
        fillShuntBatch<<<grid(ctx->st->sh.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->sh.ne, nS, n, nnz, ctx->nElements, ctx->shOffset, ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->sh.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), ctx->st->sh.altDest.as<int>(), ctx->st->sh.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->sh.dynReloc ? 1 : 0, withPilotSelf(relocOf(ctx->st->sh, ctx->st.get()), ctx));
    }
    if (ctx->st->nvdl) {
        fillVdlBatch<<<grid(ctx->st->nvdl * nS), BLOCK, 0, ctx->stream>>>(ctx->st->nvdl, nS, n, nnz, ctx->st->dVdlIn.as<double>(), ctx->st->dVdlIdx.as<int>(), ctx->st->dVdlDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->st->vdlDyn ? ctx->st->dVdlAltRow.as<int>() : nullptr, ctx->st->vdlDyn ? ctx->st->dVdlAltDest.as<int>() : nullptr, ctx->st->vdlDyn ? ctx->dMode.as<int>() : nullptr);
    }
    if (ctx->st->hv.ne) {
        fillHvdcBatch<<<grid(ctx->st->hv.ne * nS), BLOCK, 0, ctx->stream>>>(ctx->st->hv.ne, nS, n, nnz, ctx->nElements, ctx->hvOffset, ctx->st->hv.in.as<double>(), ctx->st->hv.ix.as<int>(), ctx->st->hv.dest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->st->hv.altDest.as<int>(), ctx->st->hv.selRow.as<int>(), ctx->dMode.as<int>(), ctx->st->hv.dynReloc ? 1 : 0);
    }
    int dqTraceRow = gpuDqTraceRow();
    if (ctx->hasPilotSelf && ctx->dPsSkip.p != nullptr && ctx->st->nQr > 0) {
        // Recompute the PILOT-SELF gate from the CURRENT pin states: such a pilot has a Q balance to
        // relocate only once a reactive limit has pinned it, and the RL pass moves pins between passes.
        pilotSelfGateBatch<<<grid(ctx->st->nQr * nS), BLOCK, 0, ctx->stream>>>(
                ctx->st->nQr, nS, ctx->nrl, ctx->dPsPilotEntry.as<int>(),
                ctx->dRlLimitType.as<char>(),
                ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr, ctx->dPsSkip.as<char>());
    }
    if (ctx->ndq) fillDistrQBatch<<<grid(ctx->ndq * nS), BLOCK, 0, ctx->stream>>>(ctx->ndq, nS, n, nnz, ctx->nElements, ctx->cbOffset, ctx->obOffset, ctx->st->dDqElem.as<int>(), ctx->st->dDqSide.as<int>(), ctx->st->dDqKind.as<int>(), ctx->st->dDqWeight.as<double>(), ctx->st->dDqDest.as<int>(), ctx->st->dDqAlt.as<int>(), ctx->st->dDqSel.as<int>(), ctx->st->dDqZiRow.as<int>(), ctx->st->cb.in.as<double>(), ctx->st->cb.ix.as<int>(), ctx->st->sh.in.as<double>(), ctx->st->sh.ix.as<int>(), ctx->st->ob.in.as<double>(), ctx->st->ob.ix.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dDisabled.as<char>(), ctx->dActive.as<char>(), ctx->nDqOv, ctx->nDqOv ? ctx->dDqOvIdx.as<int>() : nullptr, ctx->nDqOv ? ctx->dDqOvW.as<double>() : nullptr,
                ctx->nDqKeyOv, ctx->nDqKeyOv ? ctx->dDqKeyOvIdx.as<int>() : nullptr, ctx->nDqKeyOv ? ctx->dDqKeyOvKey.as<double>() : nullptr, ctx->dMode.as<int>(), ctx->nDqInact,
                ctx->nDqInact ? ctx->dDqInact.as<int>() : nullptr,
                ctx->nDqDyn ? ctx->dDqKMember.as<int>() : nullptr, ctx->nDqDyn || ctx->hasDqQrGroups ? ctx->dDqJMember.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberKey.as<double>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberRlEntry.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupStart.as<int>() : nullptr, ctx->nDqDyn ? ctx->dDqMemberGroupCount.as<int>() : nullptr, ctx->dRlLimitType.as<char>(), ctx->nrl, ctx->hasPilotSelf && ctx->dPsSkip.p != nullptr ? ctx->dPsSkip.as<char>() : nullptr, ctx->st->nQr, ctx->hasDqQrGroups ? ctx->dDqMemberQrGroup.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr, ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr, ctx->nR1Ov, ctx->dR1OvSlotElem.as<int>(), ctx->dR1OvVal.as<double>(), ctx->nShOv, ctx->dShOvSlotElem.as<int>(), ctx->dShOvGb.as<double>(), dqTraceRow,
                    ctx->hasDqSub ? ctx->dDqSub.as<char>() : nullptr, ctx->hasDqSub ? ctx->dQrMemberStart.as<int>() : nullptr, ctx->nDqMembers);
    if (ctx->ndr) fillDistrRhoBatch<<<grid(ctx->ndr * nS), BLOCK, 0, ctx->stream>>>(ctx->ndr, nS, n, nnz, ctx->st->dDrCol.as<int>(), ctx->st->dDrCoef.as<double>(), ctx->st->dDrDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nDrOv, ctx->nDrOv ? ctx->dDrOvIdx.as<int>() : nullptr, ctx->nDrOv ? ctx->dDrOvC.as<double>() : nullptr, ctx->st->drQr ? ctx->st->dDrQrGroup.as<int>() : nullptr, ctx->dMode.as<int>(), ctx->st->dQrPtr.as<int>(), ctx->st->dQrRows.as<int>());
    if (ctx->ndb) fillDistrShuntBBatch<<<grid(ctx->ndb * nS), BLOCK, 0, ctx->stream>>>(ctx->ndb, nS, n, nnz, ctx->st->dDbCol.as<int>(), ctx->st->dDbCoef.as<double>(), ctx->st->dDbDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>());
    if (ctx->nzi) fillZeroImpBatch<<<grid(ctx->nzi * nS), BLOCK, 0, ctx->stream>>>(ctx->nzi, nS, n, nnz, ctx->st->dZiIn.as<double>(), ctx->st->dZiIdx.as<int>(), ctx->st->dZiDest.as<int>(), ctx->dX.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->nziBase, ctx->nZiOn, ctx->dZiOnSlotElem.as<int>(), ctx->nZiOff, ctx->dZiOffSlotElem.as<int>(), ctx->st->dZiAltDest.as<int>(), ctx->st->dZiSelRow.as<int>(), ctx->dMode.as<int>(), ctx->st->ziDyn ? 1 : 0, withPilotSelf(ziRelocOf(ctx->st.get()), ctx), ctx->nZiTree, ctx->nZiTree ? ctx->dZiTreeSlotElemVal.as<int>() : nullptr, ctx->hasZiMasks ? ctx->dZiOffMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiOnMask.as<char>() : nullptr, ctx->hasZiMasks ? ctx->dZiTreeMask.as<char>() : nullptr);
    applyRowModeBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, nnz, ctx->dMode.as<int>(), ctx->st->dRowPtr.as<int>(), ctx->st->dDiagPos.as<int>(), ctx->dX.as<double>(), ctx->dTarget.as<double>(), ctx->dVals.as<double>(), ctx->dF.as<double>(), ctx->dActive.as<char>(), ctx->dRowSlope.as<double>(),
                    ctx->st->dColIdx.as<int>(), ctx->st->dRelocOwn.p != nullptr ? ctx->st->dRelocOwn.as<int>() : nullptr);
    hazRepairValues(ctx, nS, ctx->dVals.as<double>());

    if (!ctx->analyzed) {
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ctx->ds.handle, CUDSS_PHASE_ANALYSIS, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
        ctx->analyzed = true;
    }
    DeviceBuf dObs, dOut;
    dObs.upload(observeVarCols, m * sizeof(int));
    dOut.alloc((size_t) nS * k * m * sizeof(double));
    for (int ki = 0; ki < k; ki++) {
        int row = ctx->st->rowMap[unitEqRows[ki]];           // equation row -> CSR storage row
        if (row < 0) {
            throw std::runtime_error("solveBatchSensitivities: unit row not an active equation");
        }
        setUnitRhsBatch<<<grid(n * nS), BLOCK, 0, ctx->stream>>>(n, nS, row, ctx->dB.as<double>());
        hazRepairRightHandSide(ctx, nS, ctx->dB.as<double>());
        int phase = CUDSS_PHASE_SOLVE | (ki == 0 ? CUDSS_PHASE_FACTORIZATION : 0);
        { AnalysisTimer at_(__LINE__, ((phase) & (CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION)) ? 'G' : ((phase) & CUDSS_PHASE_ANALYSIS) ? 'A' : 'S'); CUDSS_CHECK(cudssExecute(ctx->ds.handle, phase, ctx->ds.config, ctx->ds.data, ctx->matUb, ctx->matUbX, ctx->matUbB)); }
        gatherSensiBatch<<<grid(nS * m), BLOCK, 0, ctx->stream>>>(n, nS, k, m, ki, dObs.as<int>(),
                ctx->dDx.as<double>(), dOut.as<double>());
    }
    std::vector<double> out((size_t) nS * k * m);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), dOut.p, out.size() * sizeof(double), cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveBatchSensitivities(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jstates, jintArray junitRows, jintArray jobserveCols) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return nullptr;
    }
    std::vector<double> states(env->GetArrayLength(jstates));
    env->GetDoubleArrayRegion(jstates, 0, (jsize) states.size(), states.data());
    std::vector<int> unitRows(env->GetArrayLength(junitRows));
    env->GetIntArrayRegion(junitRows, 0, (jsize) unitRows.size(), unitRows.data());
    std::vector<int> observeCols(env->GetArrayLength(jobserveCols));
    env->GetIntArrayRegion(jobserveCols, 0, (jsize) observeCols.size(), observeCols.data());
    std::string error;
    std::vector<double> out;
    try {
        if (states.size() != (size_t) ctx->st->n * ctx->nScenarios) {
            throw std::runtime_error("states size mismatch");
        }
        out = solveBatchSensitivities(ctx, states.data(),
                (int) unitRows.size(), unitRows.data(), (int) observeCols.size(), observeCols.data());
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU batched sensitivity solve failed: " + error).c_str());
        return nullptr;
    }
    jdoubleArray jout = env->NewDoubleArray((jsize) out.size());
    env->SetDoubleArrayRegion(jout, 0, (jsize) out.size(), out.data());
    return jout;
}

// Per-scenario slack-only-DS absorbed totals after a solve (see slackOnlyDsCheck) — the pair phase
// seeds its scenarios with the parent contingency's value (the device analogue of OLF's
// setGeneratorsInitialTargetPToTargetP before a strategy re-solve). Null when not slack-only.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchSlackAbsorbed(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dsSlackOnly) {
        return nullptr;
    }
    std::vector<double> host(ctx->nScenarios);
    cudaMemcpy(host.data(), ctx->dDsSlackAbsorbed.p, ctx->nScenarios * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(ctx->nScenarios));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(ctx->nScenarios), host.data());
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDsGenCum(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->ngen <= 0 || !ctx->dGenCum.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nScenarios) * ctx->ngen;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dGenCum.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(cnt));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

// Per scenario of the last solveBatchContext call (and its host-loop resumes): {solve passes, first pass's
// Newton iterations, total Newton iterations}, flattened s*3. The per-contingency counterpart of
// GPU_OL_HIST, for matching OLF's CPU_OL_STATS contingency by contingency.
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchOlStats(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->olPasses.size() != static_cast<size_t>(ctx->nScenarios)) {
        return nullptr;
    }
    std::vector<jint> host(static_cast<size_t>(ctx->nScenarios) * 3);
    for (int s = 0; s < ctx->nScenarios; s++) {
        host[s * 3] = ctx->olPasses[s];
        host[s * 3 + 1] = ctx->olFirst[s];
        host[s * 3 + 2] = ctx->olIters[s];
    }
    jintArray out = env->NewIntArray(static_cast<jsize>(host.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}

// How many solveBatchContext calls this context has run. The per-solve DS read-backs (retain, genCum,
// slack absorbed) are reset at every solve, so a caller summing them must know they span its solve.
extern "C" JNIEXPORT jlong JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchSolveCount(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    return ctx == nullptr ? -1 : static_cast<jlong>(ctx->solveCount);
}

// One scenario's state at its last converged inner solve of the last solve (its starting state if none converged).
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchLastConvergedState(
        JNIEnv* env, jclass, jlong handle, jint slot) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dXLastConv.p || slot < 0 || slot >= ctx->nScenarios) {
        return nullptr;
    }
    int n = ctx->st->n;
    std::vector<double> host(n);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(host.data(), ctx->dXLastConv.as<double>() + (size_t) slot * n, n * sizeof(double),
                          cudaMemcpyDeviceToHost));
    jdoubleArray out = env->NewDoubleArray(n);
    env->SetDoubleArrayRegion(out, 0, n, host.data());
    return out;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDsRetain(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || !ctx->dDsRetainApplied.p) {
        return nullptr;
    }
    std::vector<double> host(ctx->nScenarios);
    cudaMemcpy(host.data(), ctx->dDsRetainApplied.p, ctx->nScenarios * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(ctx->nScenarios));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(ctx->nScenarios), host.data());
    return out;
}

// LINE_SEARCH backtracking parameters for a batch context (OLF LineSearchStateVectorScaling's
// maxIteration and stepFold): consumed by solves called with lsMode == 1. Context-scoped like
// setBatchRowCriteria, so the solveBatchContext signature stays unchanged.
// Per-scenario starting values for the next solve given no full per-scenario state: flat index s*n+row -> value,
// written over the base state tiled on the device. Replaces sending nScenarios * n doubles when only a few rows
// differ (the frozen rows of islanded buses). Null or empty arrays clear them.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchX0Seeds(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jVal) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->x0SeedIdx.clear();
    ctx->x0SeedVal.clear();
    if (jIdx == nullptr || jVal == nullptr) {
        return;
    }
    jsize k = env->GetArrayLength(jIdx);
    ctx->x0SeedIdx.resize(k);
    ctx->x0SeedVal.resize(k);
    env->GetIntArrayRegion(jIdx, 0, k, ctx->x0SeedIdx.data());
    env->GetDoubleArrayRegion(jVal, 0, k, ctx->x0SeedVal.data());
}

// The linear-solver settings of the run (GpuLoadFlowParameters): read by every context built after this call and by
// the adaptive refinement of every solve. -1 reorder / pivotType = AUTO; NaN threshold = the library's.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setLinearSolverSettings(
        JNIEnv*, jclass, jint reorder, jint matching, jint pivotType, jdouble pivotThreshold,
        jint irBase, jint irMax, jdouble irTol) {
    gLinearSolver.reorder = reorder;
    gLinearSolver.matching = matching;
    gLinearSolver.pivotType = pivotType;
    gLinearSolver.pivotThreshold = pivotThreshold;
    gLinearSolver.irBase = irBase;
    gLinearSolver.irMax = irMax;
    gLinearSolver.irTol = irTol;
}

// The installed linear-solver settings, in setLinearSolverSettings's order (tests: the parameters reach the device).
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readLinearSolverSettings(JNIEnv* env, jclass) {
    const double v[7] = {static_cast<double>(gLinearSolver.reorder), static_cast<double>(gLinearSolver.matching),
                         static_cast<double>(gLinearSolver.pivotType), gLinearSolver.pivotThreshold,
                         static_cast<double>(gLinearSolver.irBase), static_cast<double>(gLinearSolver.irMax),
                         gLinearSolver.irTol};
    jdoubleArray out = env->NewDoubleArray(7);
    env->SetDoubleArrayRegion(out, 0, 7, v);
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchLineSearch(
        JNIEnv*, jclass, jlong handle, jint lsMaxIter, jdouble lsStepFold) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->lsMaxIter = lsMaxIter > 0 ? lsMaxIter : 10;
    ctx->lsStepFold = lsStepFold > 1.0 ? lsStepFold : 4.0 / 3.0;
}

// PER_EQUATION_TYPE stopping criterion: install (or clear, with nulls) the per-row threshold
// arrays on a batch context — thrIdent for identity/slope rows, thrPower for power rows, both
// length n. The convergence test then requires EVERY row's |F| under its threshold (OLF's
// PerEquationTypeStoppingCriteria), replacing the 2-norm/infnorm rule for this context.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRowCriteria(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jThrIdent, jdoubleArray jThrPower) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jThrIdent == nullptr || jThrPower == nullptr) {
        ctx->perRowCriteria = false;
        ctx->perRowAlt = false;
        return;
    }
    int n = ctx->st->n;
    jdouble* thrIdent = env->GetDoubleArrayElements(jThrIdent, nullptr);
    jdouble* thrPower = env->GetDoubleArrayElements(jThrPower, nullptr);
    ctx->dThrIdent.upload(thrIdent, static_cast<size_t>(n) * sizeof(double));
    ctx->dThrPower.upload(thrPower, static_cast<size_t>(n) * sizeof(double));
    env->ReleaseDoubleArrayElements(jThrIdent, thrIdent, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jThrPower, thrPower, JNI_ABORT);
    ctx->perRowCriteria = true;
}

// The per-row ALTERNATE threshold and its selector row (-1: none) of the PER_EQUATION_TYPE criterion, read by
// residualNormsBatch for a cross identity (mode 6) and for a relocated child's row once its coupler's dummy row
// is an identity. Nulls clear them. Install after setBatchRowCriteria, which clears them too.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRowCriteriaAlt(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jThrAlt, jintArray jThrSel) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jThrAlt == nullptr || jThrSel == nullptr) {
        ctx->perRowAlt = false;
        return;
    }
    int n = ctx->st->n;
    jdouble* thrAlt = env->GetDoubleArrayElements(jThrAlt, nullptr);
    jint* thrSel = env->GetIntArrayElements(jThrSel, nullptr);
    ctx->dThrAlt.upload(thrAlt, static_cast<size_t>(n) * sizeof(double));
    ctx->dThrSel.upload(thrSel, static_cast<size_t>(n) * sizeof(int));
    env->ReleaseDoubleArrayElements(jThrAlt, thrAlt, JNI_ABORT);
    env->ReleaseIntArrayElements(jThrSel, thrSel, JNI_ABORT);
    ctx->perRowAlt = true;
}

// Reactive-capability CURVES at runtime P: install (or clear, with nulls) the per-gen curve channel
// on a batch context — see rlCurveLimitsBatch. cvIdx stride 5 {rlIdx, pRow, ptsOff, npts, dsGenSlot}; cvIn
// stride 5 {basePpu, share, baseRowTargetP, baseEvalMinPu, baseEvalMaxPu}; cvPts stride 3
// {p_MW, minQ_MW, maxQ_MW} sorted by p; cvHv (null when no entry is an AC-emulation VSC) stride 2 {hvdc, side}.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlCurves(
        JNIEnv* env, jclass, jlong handle, jintArray jCvIdx, jdoubleArray jCvIn, jdoubleArray jCvPts,
        jintArray jCvHv, jint extrapolate) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->cvHasHvdc = false;
    if (jCvIdx == nullptr || jCvIn == nullptr || jCvPts == nullptr) {
        ctx->ncv = 0;
        return;
    }
    int ncv = env->GetArrayLength(jCvIdx) / 5;   // stride 5: {rlIdx, pRow, ptsOff, npts, dsGenSlot}
    if (ncv == 0) {
        ctx->ncv = 0;
        return;
    }
    if (jCvHv != nullptr) {                      // stride 2 per entry: {hvdc, side}, -1 for a plain generator
        jint* hv = env->GetIntArrayElements(jCvHv, nullptr);
        ctx->dCvHv.upload(hv, static_cast<size_t>(ncv) * 2 * sizeof(int));
        env->ReleaseIntArrayElements(jCvHv, hv, JNI_ABORT);
        ctx->cvHasHvdc = true;
    }
    jint* cvIdx = env->GetIntArrayElements(jCvIdx, nullptr);
    jdouble* cvIn = env->GetDoubleArrayElements(jCvIn, nullptr);
    jdouble* cvPts = env->GetDoubleArrayElements(jCvPts, nullptr);
    ctx->dCvIdx.upload(cvIdx, static_cast<size_t>(ncv) * 5 * sizeof(int));
    ctx->dCvIn.upload(cvIn, static_cast<size_t>(ncv) * 5 * sizeof(double));
    ctx->dCvPts.upload(cvPts, static_cast<size_t>(env->GetArrayLength(jCvPts)) * sizeof(double));
    // The curves of each reactive-limit entry, CSR (pack order kept within an entry): rlCurveLimitsBatch runs
    // one thread per (entry, scenario) and walked EVERY curve to find its own.
    {
        std::vector<int> ptr(static_cast<size_t>(ctx->nrl) + 1, 0);
        std::vector<int> list(ncv);
        for (int g = 0; g < ncv; g++) {
            int e = cvIdx[g * 5];
            if (e < 0 || e >= ctx->nrl) {
                env->ReleaseIntArrayElements(jCvIdx, cvIdx, JNI_ABORT);
                env->ReleaseDoubleArrayElements(jCvIn, cvIn, JNI_ABORT);
                env->ReleaseDoubleArrayElements(jCvPts, cvPts, JNI_ABORT);
                ctx->ncv = 0;
                env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                              "GPU batch: a curve's reactive-limit entry is outside the pack");
                return;
            }
            ptr[e + 1]++;
        }
        for (int e = 0; e < ctx->nrl; e++) ptr[e + 1] += ptr[e];
        std::vector<int> cursor(ptr.begin(), ptr.end() - 1);
        for (int g = 0; g < ncv; g++) list[cursor[cvIdx[g * 5]]++] = g;
        ctx->dCvPtr.upload(ptr.data(), ptr.size() * sizeof(int));
        ctx->dCvList.upload(list.data(), list.size() * sizeof(int));
    }
    ctx->dRlMinQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    ctx->dRlMaxQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    ctx->dRlLoadQDyn.alloc(static_cast<size_t>(ctx->nScenarios) * ctx->nrl * sizeof(double));
    env->ReleaseIntArrayElements(jCvIdx, cvIdx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCvIn, cvIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCvPts, cvPts, JNI_ABORT);
    ctx->ncv = ncv;
    ctx->cvExtrapolate = extrapolate;
}

// forceTargetQInReactiveLimits channel (GpuBatchForcedTargetQ): fqIdx stride 7 {qRow, dsGenSlot, ptsOff, npts,
// dsBusIdx, baseQRowMode, rlIdx}; fqIn stride 5 {basePpu, rawTargetQPu, baseClampedPu, share, baseRowTargetP}; fqPts
// stride 3 {p_MW, minQ_MW, maxQ_MW} sorted by p. Null clears the channel (a cached context reused by a run
// without it).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchForcedTargetQ(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jIn, jdoubleArray jPts, jint extrapolate) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->fqHasPre = false;
    if (jIdx == nullptr || jIn == nullptr || jPts == nullptr || env->GetArrayLength(jIdx) == 0) {
        ctx->nfq = 0;
        ctx->nfqLim = 0;
        return;
    }
    int nfq = env->GetArrayLength(jIdx) / 7;
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    int nfqLim = 0;
    for (int e = 0; e < nfq; e++) {
        nfqLim += idx[e * 7 + 6] >= 0 ? 1 : 0;
    }
    jdouble* in = env->GetDoubleArrayElements(jIn, nullptr);
    jdouble* pts = env->GetDoubleArrayElements(jPts, nullptr);
    ctx->dFqIdx.upload(idx, static_cast<size_t>(nfq) * 7 * sizeof(int));
    ctx->dFqIn.upload(in, static_cast<size_t>(nfq) * 5 * sizeof(double));
    ctx->dFqPts.upload(pts, static_cast<size_t>(env->GetArrayLength(jPts)) * sizeof(double));
    ctx->dFqApplied.alloc(static_cast<size_t>(ctx->nScenarios) * nfq * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jIn, in, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jPts, pts, JNI_ABORT);
    ctx->nfq = nfq;
    ctx->nfqLim = nfqLim;
    ctx->fqExtrapolate = extrapolate;
}

// Per-chunk loss pre-distribution moves of the forced generators ([nS * nfq], pu), or null when no scenario of the
// chunk pre-distributes. Set before each solve that carries contingencies.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchForcedTargetQPre(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jPre) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPre == nullptr || ctx->nfq == 0) {
        ctx->fqHasPre = false;
        return;
    }
    jdouble* pre = env->GetDoubleArrayElements(jPre, nullptr);
    ctx->dFqPre.upload(pre, static_cast<size_t>(env->GetArrayLength(jPre)) * sizeof(double));
    env->ReleaseDoubleArrayElements(jPre, pre, JNI_ABORT);
    ctx->fqHasPre = true;
}

// Remote-member RL entries (plan-B1): install (or clear) the per-entry remote kinds + compare rows
// on a batch context. The RL kernel SKIPS remote entries until the B3 transition machinery lands.
// Slack-only distribution: the slack generators' initial target per scenario (relative to the slack's target P)
// for the next solves - a strategy pair's parent total; null clears it (zero: a contingency's base).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchSlackOnlyAnchor(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jAnchor) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jAnchor == nullptr || !ctx->dsSlackOnly) {
        ctx->dsSlackHasAnchor = false;
        return;
    }
    jdouble* a = env->GetDoubleArrayElements(jAnchor, nullptr);
    ctx->dDsSlackAnchor.upload(a, static_cast<size_t>(env->GetArrayLength(jAnchor)) * sizeof(double));
    env->ReleaseDoubleArrayElements(jAnchor, a, JNI_ABORT);
    ctx->dsSlackHasAnchor = true;
}

// Per scenario and participating generator (s * ngen + g, pu): anchor = its move from its base target P to where the
// slack's water-fill starts it (a strategy pair: the parent's final target, or a GeneratorAction's value; null = the
// base), start = its move from that anchor at the solve's start (a contingency's loss pre-distribution; null = none).
// The curve and forced-target-Q channels evaluate a generator at anchor + (genCum once a pass ran, else start).
// seedAnchor (pairs): the anchor WITHOUT the strategy's GeneratorActions - where the continued pins are stamped.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchGenAnchor(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jAnchor, jdoubleArray jStart, jdoubleArray jSeedAnchor) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    size_t want = static_cast<size_t>(ctx->nScenarios) * std::max(ctx->ngen, 0);
    auto put = [&](jdoubleArray arr, DeviceBuf& buf, bool& has) {
        has = false;
        if (arr == nullptr || ctx->ngen <= 0) {
            return true;
        }
        if (static_cast<size_t>(env->GetArrayLength(arr)) != want) {
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "setBatchGenAnchor: expected nScenarios * ngen");
            return false;
        }
        jdouble* a = env->GetDoubleArrayElements(arr, nullptr);
        buf.upload(a, want * sizeof(double));
        env->ReleaseDoubleArrayElements(arr, a, JNI_ABORT);
        has = true;
        return true;
    };
    if (put(jAnchor, ctx->dDgAnchor, ctx->hasDgAnchor) && put(jStart, ctx->dDgStart, ctx->hasDgStart)) {
        put(jSeedAnchor, ctx->dDgSeedAnchor, ctx->hasDgSeedAnchor);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlElection(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jNomV, jintArray jPRow, jdoubleArray jStaticP, jintArray jIdRank) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hasRlElection = 0;
    if (ctx->nrl == 0 || jNomV == nullptr || jPRow == nullptr || jStaticP == nullptr || jIdRank == nullptr) {
        return;
    }
    if (env->GetArrayLength(jNomV) != ctx->nrl || env->GetArrayLength(jPRow) != ctx->nrl
            || env->GetArrayLength(jStaticP) != ctx->nrl || env->GetArrayLength(jIdRank) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRlElection: array length != nrl");
        return;
    }
    jdouble* nv = env->GetDoubleArrayElements(jNomV, nullptr);
    jint* pr = env->GetIntArrayElements(jPRow, nullptr);
    jdouble* sp = env->GetDoubleArrayElements(jStaticP, nullptr);
    jint* ir = env->GetIntArrayElements(jIdRank, nullptr);
    ctx->dRlElectNomV.upload(nv, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlElectPRow.upload(pr, static_cast<size_t>(ctx->nrl) * sizeof(int));
    ctx->dRlElectStaticP.upload(sp, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlElectIdRank.upload(ir, static_cast<size_t>(ctx->nrl) * sizeof(int));
    env->ReleaseDoubleArrayElements(jNomV, nv, JNI_ABORT);
    env->ReleaseIntArrayElements(jPRow, pr, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jStaticP, sp, JNI_ABORT);
    env->ReleaseIntArrayElements(jIdRank, ir, JNI_ABORT);
    ctx->hasRlElection = 1;
}

// voltageRemoteControlRobustMode (OLF's default): install the robust-mode channel on a batch context —
// the per-entry initial generation target Q that a realistic-V freeze pins at, and the two thresholds
// ReactiveLimitsOuterLoop tests the controller's OWN voltage against (minRealisticVoltage ·
// REALISTIC_VOLTAGE_MARGIN and maxRealisticVoltage / REALISTIC_VOLTAGE_MARGIN, margin 1.02). robust = 0
// clears the channel, so a cached context reused by a run with the mode off stops applying it.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlRobust(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jInitTgtQ, jdouble vLo, jdouble vHi, jint robust) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->rlRobust = 0;
    if (ctx->nrl == 0 || robust == 0 || jInitTgtQ == nullptr) {
        return;
    }
    if (env->GetArrayLength(jInitTgtQ) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRlRobust: array length != nrl");
        return;
    }
    jdouble* q0 = env->GetDoubleArrayElements(jInitTgtQ, nullptr);
    ctx->dRlInitTgtQ.upload(q0, static_cast<size_t>(ctx->nrl) * sizeof(double));
    env->ReleaseDoubleArrayElements(jInitTgtQ, q0, JNI_ABORT);
    ctx->rlVLoRobust = vLo;
    ctx->rlVHiRobust = vHi;
    ctx->rlRobust = 1;
}

// AcloadFlowEngine's realistic-voltage check: install (or clear) the per-row "this row is a
// checked bus voltage" mask and the two bounds. enabled = 0 clears it on a cached context.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRealisticCheck(
        JNIEnv* env, jclass, jlong handle, jbyteArray jRowChecked, jdouble minV, jdouble maxV, jint enabled,
        jint fixingLoop) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->realisticCheck = 0;
    if (enabled == 0 || jRowChecked == nullptr) {
        return;
    }
    if (env->GetArrayLength(jRowChecked) != ctx->n) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "setBatchRealisticCheck: array length != n");
        return;
    }
    jbyte* rc = env->GetByteArrayElements(jRowChecked, nullptr);
    ctx->dRealisticRow.upload(rc, static_cast<size_t>(ctx->n) * sizeof(char));
    env->ReleaseByteArrayElements(jRowChecked, rc, JNI_ABORT);
    ctx->realisticMinV = minV;
    ctx->realisticMaxV = maxV;
    ctx->realisticFixingLoop = fixingLoop;
    ctx->realisticCheck = 1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlSvcLimits(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jBmin, jdoubleArray jBmax, jdoubleArray jV2Base) {
    BatchSolverContext* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nrl == 0) {
        return;
    }
    if (env->GetArrayLength(jBmin) != ctx->nrl || env->GetArrayLength(jBmax) != ctx->nrl
            || env->GetArrayLength(jV2Base) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchRlSvcLimits: array length != nrl");
        return;
    }
    jdouble* bmin = env->GetDoubleArrayElements(jBmin, nullptr);
    jdouble* bmax = env->GetDoubleArrayElements(jBmax, nullptr);
    jdouble* v2b = env->GetDoubleArrayElements(jV2Base, nullptr);
    ctx->dRlSvcBmin.upload(bmin, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlSvcBmax.upload(bmax, static_cast<size_t>(ctx->nrl) * sizeof(double));
    ctx->dRlV2Base.upload(v2b, static_cast<size_t>(ctx->nrl) * sizeof(double));
    env->ReleaseDoubleArrayElements(jBmin, bmin, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jBmax, bmax, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jV2Base, v2b, JNI_ABORT);
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlBaseLimitType(
        JNIEnv* env, jclass, jlong handle, jbyteArray jKinds) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKinds == nullptr || env->GetArrayLength(jKinds) == 0) {
        ctx->hasRlBaseLt = 0;
        return;
    }
    if (env->GetArrayLength(jKinds) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchRlBaseLimitType: array length != nrl");
        return;
    }
    jbyte* kinds = env->GetByteArrayElements(jKinds, nullptr);
    ctx->dRlBaseLimitType.alloc(static_cast<size_t>(ctx->nrl));
    CUDA_CHECK(cudaMemcpy(ctx->dRlBaseLimitType.p, kinds, static_cast<size_t>(ctx->nrl), cudaMemcpyHostToDevice));
    env->ReleaseByteArrayElements(jKinds, kinds, JNI_ABORT);
    ctx->hasRlBaseLt = 1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlRemote(
        JNIEnv* env, jclass, jlong handle, jbyteArray jRemote, jintArray jCmpVrow,
        jintArray jRmqRow, jintArray jRmqElem, jintArray jRmqSide, jintArray jRmqKind, jdoubleArray jRmqWeight,
        jintArray jDqKMember, jintArray jDqJMember, jdoubleArray jMKey, jintArray jMRlEntry,
        jintArray jMGroupStart, jintArray jMGroupCount, jintArray jMRow, jbyteArray jMHasRow,
        jintArray jMQrGroup, jdoubleArray jMTargetQ, jintArray jMTgtPtr, jintArray jMTgtPeer) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jRemote == nullptr || jCmpVrow == nullptr || env->GetArrayLength(jRemote) == 0) {
        ctx->nRlRemote = 0;
        ctx->nRmq = 0;
        ctx->nDqDyn = 0;
        ctx->hasDqQrGroups = 0;
        return;
    }
    int nr = env->GetArrayLength(jRemote);
    if (nr != ctx->nrl || env->GetArrayLength(jCmpVrow) != ctx->nrl) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchRlRemote: array length != nrl");
        return;
    }
    jbyte* rem = env->GetByteArrayElements(jRemote, nullptr);
    jint* cmp = env->GetIntArrayElements(jCmpVrow, nullptr);
    int count = 0;
    int selfCount = 0;
    for (int i = 0; i < nr; i++) {
        if (rem[i]) {
            count++;
        }
        if (rem[i] == 2) {
            selfCount++;
        }
    }
    ctx->nRlPilotSelf = selfCount;
    ctx->dRlRemote.upload(rem, static_cast<size_t>(nr) * sizeof(char));
    ctx->dRlCmpVrow.upload(cmp, static_cast<size_t>(nr) * sizeof(int));
    env->ReleaseByteArrayElements(jRemote, rem, JNI_ABORT);
    env->ReleaseIntArrayElements(jCmpVrow, cmp, JNI_ABORT);
    ctx->nRlRemote = count;
    ctx->nRmq = 0;
    if (jRmqRow != nullptr && env->GetArrayLength(jRmqRow) > 0) {
        int nq = env->GetArrayLength(jRmqRow);
        jint* qr = env->GetIntArrayElements(jRmqRow, nullptr);
        jint* qe = env->GetIntArrayElements(jRmqElem, nullptr);
        jint* qsd = env->GetIntArrayElements(jRmqSide, nullptr);
        jint* qk = env->GetIntArrayElements(jRmqKind, nullptr);
        jdouble* qw = env->GetDoubleArrayElements(jRmqWeight, nullptr);
        ctx->dRmqRow.upload(qr, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqElem.upload(qe, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqSide.upload(qsd, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqKind.upload(qk, static_cast<size_t>(nq) * sizeof(int));
        ctx->dRmqWeight.upload(qw, static_cast<size_t>(nq) * sizeof(double));
        env->ReleaseIntArrayElements(jRmqRow, qr, JNI_ABORT);
        env->ReleaseIntArrayElements(jRmqElem, qe, JNI_ABORT);
        env->ReleaseIntArrayElements(jRmqSide, qsd, JNI_ABORT);
        env->ReleaseIntArrayElements(jRmqKind, qk, JNI_ABORT);
        env->ReleaseDoubleArrayElements(jRmqWeight, qw, JNI_ABORT);
        ctx->nRmq = nq;
    }
    ctx->nDqDyn = 0;
    ctx->hasDqQrGroups = 0;
    // Keyed on the MEMBER list, not on the DISTR_Q rows: a SINGLETON remote group has members (and a
    // pilot-Q relocation group, so its pilot's control-loss flip must be armed) but no DISTR_Q row at
    // all — gating on dqKMember left hasDqQrGroups at 0 for it, and the base load flow on the driver
    // kept the pilot's V equation after pinning its only member (2.4e-3 pu on the pilot's Q).
    if (jMKey != nullptr && env->GetArrayLength(jMKey) > 0) {
        int ndqc = jDqKMember != nullptr ? env->GetArrayLength(jDqKMember) : 0;
        int nm = env->GetArrayLength(jMKey);
        jint* km = ndqc > 0 ? env->GetIntArrayElements(jDqKMember, nullptr) : nullptr;
        jint* jm = ndqc > 0 ? env->GetIntArrayElements(jDqJMember, nullptr) : nullptr;
        jdouble* mk = env->GetDoubleArrayElements(jMKey, nullptr);
        jint* mre = env->GetIntArrayElements(jMRlEntry, nullptr);
        jint* mgs = env->GetIntArrayElements(jMGroupStart, nullptr);
        jint* mgc = env->GetIntArrayElements(jMGroupCount, nullptr);
        jint* mrw = env->GetIntArrayElements(jMRow, nullptr);
        jbyte* mhr = env->GetByteArrayElements(jMHasRow, nullptr);
        ctx->dDqMemberRow.upload(mrw, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberHasRow.upload(mhr, static_cast<size_t>(nm) * sizeof(char));
        env->ReleaseIntArrayElements(jMRow, mrw, JNI_ABORT);
        env->ReleaseByteArrayElements(jMHasRow, mhr, JNI_ABORT);
        if (jMTargetQ == nullptr || env->GetArrayLength(jMTargetQ) != nm) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchRlRemote: dqMemberTargetQ missing or length != members");
            return;
        }
        jdouble* mtq = env->GetDoubleArrayElements(jMTargetQ, nullptr);
        ctx->dDqMemberTargetQ.upload(mtq, static_cast<size_t>(nm) * sizeof(double));
        env->ReleaseDoubleArrayElements(jMTargetQ, mtq, JNI_ABORT);
        if (jMTgtPtr == nullptr || env->GetArrayLength(jMTgtPtr) != nm + 1 || jMTgtPeer == nullptr) {
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                          "setBatchRlRemote: dqMemberTgtPtr missing or length != members + 1");
            return;
        }
        jint* mtp = env->GetIntArrayElements(jMTgtPtr, nullptr);
        ctx->dDqMemberTgtPtr.upload(mtp, static_cast<size_t>(nm + 1) * sizeof(int));
        env->ReleaseIntArrayElements(jMTgtPtr, mtp, JNI_ABORT);
        int nPeer = env->GetArrayLength(jMTgtPeer);
        if (nPeer > 0) {
            jint* mpe = env->GetIntArrayElements(jMTgtPeer, nullptr);
            ctx->dDqMemberTgtPeer.upload(mpe, static_cast<size_t>(nPeer) * sizeof(int));
            env->ReleaseIntArrayElements(jMTgtPeer, mpe, JNI_ABORT);
        }
        ctx->nDqMembers = nm;
        if (ndqc > 0) {
            ctx->dDqKMember.upload(km, static_cast<size_t>(ndqc) * sizeof(int));
            ctx->dDqJMember.upload(jm, static_cast<size_t>(ndqc) * sizeof(int));
        }
        ctx->dDqMemberKey.upload(mk, static_cast<size_t>(nm) * sizeof(double));
        ctx->dDqMemberRlEntry.upload(mre, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberGroupStart.upload(mgs, static_cast<size_t>(nm) * sizeof(int));
        ctx->dDqMemberGroupCount.upload(mgc, static_cast<size_t>(nm) * sizeof(int));
        if (ndqc > 0) {
            env->ReleaseIntArrayElements(jDqKMember, km, JNI_ABORT);
            env->ReleaseIntArrayElements(jDqJMember, jm, JNI_ABORT);
        }
        env->ReleaseDoubleArrayElements(jMKey, mk, JNI_ABORT);
        env->ReleaseIntArrayElements(jMRlEntry, mre, JNI_ABORT);
        env->ReleaseIntArrayElements(jMGroupStart, mgs, JNI_ABORT);
        env->ReleaseIntArrayElements(jMGroupCount, mgc, JNI_ABORT);
        ctx->nDqDyn = ndqc;
        ctx->hasDqQrGroups = 0;
        if (jMQrGroup != nullptr && env->GetArrayLength(jMQrGroup) == nm) {
            jint* mqg = env->GetIntArrayElements(jMQrGroup, nullptr);
            std::vector<char> genVc(ctx->st->nQr > 0 ? ctx->st->nQr : 1, 0);
            for (int i = 0; i < nm; i++) {
                if (mqg[i] >= 0) {
                    ctx->hasDqQrGroups = 1;                  // plan-B4: relocation-linked group present
                    genVc[mqg[i]] = 1;                       // this qr group is a remote gen-VC group
                }
            }
            for (int g = 0; g < ctx->st->nQr; g++) {         // a DEPENDENT pilot's group is as generator-controlled
                if (g < static_cast<int>(ctx->st->hQrDepBase.size()) && ctx->st->hQrDepBase[g] >= 0) {   // as its base
                    genVc[g] = genVc[ctx->st->hQrDepBase[g]];
                }
            }
            ctx->dDqMemberQrGroup.upload(mqg, static_cast<size_t>(nm) * sizeof(int));
            if (ctx->hasDqQrGroups) {
                ctx->dQrGenVc.upload(genVc.data(), genVc.size() * sizeof(char));
            }
            env->ReleaseIntArrayElements(jMQrGroup, mqg, JNI_ABORT);
        }
  }
}

// plan-B4 (JAVA hybrid): install ONLY the carrier-row suppression tables — dqJMember (per DISTR_Q
// contribution: row-owner member) + dqMemberQrGroup (per member: pilot's relocation group) — so a
// pushed mode-4 free-member row gates the re-picked carrier's DISTR_Q row off in fillDistrQBatch.
// Does NOT touch nDqDyn: the hybrid's weights stay static-pack + dqOv (OLF renormalizes and the
// engine pushes live values). Nulls clear.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchZiTreeOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotElemVal) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    int len = jSlotElemVal == nullptr ? 0 : env->GetArrayLength(jSlotElemVal);
    ctx->nZiTree = len / 3;
    if (ctx->nZiTree > 0) {
        jint* p = env->GetIntArrayElements(jSlotElemVal, nullptr);
        ctx->dZiTreeSlotElemVal.upload(p, static_cast<size_t>(len) * sizeof(int));
        env->ReleaseIntArrayElements(jSlotElemVal, p, JNI_ABORT);
    }
    // Built whenever the masks are enabled, NOT only once the off/on setter has run: the two setters' order is
    // the caller's, and gating this one on hasZiMasks dropped every tree override of a chunk whose tree list
    // arrived first - the extraction-time tree then stood in for OLF's, and a coupler child like LLJ_2 on
    // network.xiidm lost its ZERO_V coupling in 2,308 scenarios (V 0.787, FAILED as unrealistic).
    if (ctx->nzi > 0) {                                      // the dense form (see dZiTreeMask)
        ctx->hasZiMasks = true;
        size_t mlen = static_cast<size_t>(ctx->nScenarios) * ctx->nzi;
        std::vector<char> tree(mlen, (char) -1);
        if (len > 0) {
            jint* p = env->GetIntArrayElements(jSlotElemVal, nullptr);
            for (int k = 0; k < len / 3; k++) {
                int sl = p[3 * k], e = p[3 * k + 1];
                if (sl >= 0 && sl < ctx->nScenarios && e >= 0 && e < ctx->nzi) tree[(size_t) sl * ctx->nzi + e] = (char) (p[3 * k + 2] != 0 ? 1 : 0);
            }
            env->ReleaseIntArrayElements(jSlotElemVal, p, JNI_ABORT);
        }
        ctx->dZiTreeMask.upload(tree.data(), mlen);
        if (std::getenv("OLF_GPU_PHASE_TIME") != nullptr) {
            std::fprintf(stderr, "ZI_LISTS nZiTree=%d\n", len / 3);
        }
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchLoopTraceSlot(
        JNIEnv*, jclass, jlong handle, jint slot) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->loopTraceSlot = slot;
    }
}

// DISTR_Q weight overrides: install (or clear, with nulls) the per-scenario qPct renormalization
// pairs {s, contrib} + new weights on a batch context (gen outage at a remote-VC controller bus).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqInactiveMembers(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotMember) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotMember == nullptr || env->GetArrayLength(jSlotMember) == 0) {
        ctx->nDqInact = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotMember);
    jint* sm = env->GetIntArrayElements(jSlotMember, nullptr);
    ctx->dDqInact.upload(sm, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotMember, sm, JNI_ABORT);
    ctx->nDqInact = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqSubgroups(
        JNIEnv* env, jclass, jlong handle, jbyteArray jDense, jintArray jQrMemberStart) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jDense == nullptr || env->GetArrayLength(jDense) == 0) {
        ctx->hasDqSub = false;
        return;
    }
    jbyte* d = env->GetByteArrayElements(jDense, nullptr);
    ctx->dDqSub.upload(d, static_cast<size_t>(env->GetArrayLength(jDense)));
    env->ReleaseByteArrayElements(jDense, d, JNI_ABORT);
    jint* q = env->GetIntArrayElements(jQrMemberStart, nullptr);
    ctx->dQrMemberStart.upload(q, static_cast<size_t>(env->GetArrayLength(jQrMemberStart)) * sizeof(int));
    env->ReleaseIntArrayElements(jQrMemberStart, q, JNI_ABORT);
    ctx->hasDqSub = true;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchFrozenRows(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotRow) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotRow == nullptr || env->GetArrayLength(jSlotRow) == 0) {
        ctx->nFrozenRow = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotRow);
    jint* sr = env->GetIntArrayElements(jSlotRow, nullptr);
    ctx->dFrozenRow.upload(sr, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotRow, sr, JNI_ABORT);
    ctx->nFrozenRow = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsLostGenerators(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotGen) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotGen == nullptr || env->GetArrayLength(jSlotGen) == 0) {
        ctx->nDsLostGen = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotGen);
    jint* sg = env->GetIntArrayElements(jSlotGen, nullptr);
    ctx->dDsLostGen.upload(sg, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotGen, sg, JNI_ABORT);
    ctx->nDsLostGen = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlPinTargets(
        JNIEnv* env, jclass, jlong handle, jdoubleArray jTargets) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jTargets == nullptr || env->GetArrayLength(jTargets) == 0) {
        ctx->hasRlPinTgtIn = false;
        return;
    }
    int len = env->GetArrayLength(jTargets);
    jdouble* t = env->GetDoubleArrayElements(jTargets, nullptr);
    ctx->dRlPinTgtIn.upload(t, static_cast<size_t>(len) * sizeof(double));
    env->ReleaseDoubleArrayElements(jTargets, t, JNI_ABORT);
    ctx->hasRlPinTgtIn = true;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchRlPinTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nrl <= 0) {
        return nullptr;
    }
    int nS = ctx->nScenarios;
    int n = ctx->st->n;
    DeviceBuf out;
    out.alloc(static_cast<size_t>(ctx->nrl) * nS * sizeof(double));
    exportRlPinTargets<<<grid(ctx->nrl * nS), BLOCK, 0, ctx->stream>>>(ctx->nrl, nS, n, ctx->dRlVrow.as<int>(),
            ctx->dRlQrow.p ? ctx->dRlQrow.as<int>() : nullptr, ctx->dRlLimitType.as<char>(), ctx->dMode.as<int>(),
            ctx->dTarget.as<double>(), ctx->nRlIslanded > 0 ? ctx->dRlIslanded.as<char>() : nullptr,
            ctx->hasDqQrGroups ? ctx->st->nQr : 0, psEntryPtr(ctx),
            ctx->hasDqQrGroups && ctx->st->nQr > 0 ? ctx->st->dQrPtr.as<int>() : nullptr,
            ctx->hasDqQrGroups && ctx->st->nQr > 0 ? ctx->st->dQrRows.as<int>() : nullptr,
            ctx->hasDqQrGroups && ctx->st->nQr > 0 ? ctx->dQrVeqOff.as<char>() : nullptr,
            ctx->st->dQrDepBase.p ? ctx->st->dQrDepBase.as<int>() : nullptr, out.as<double>());
    std::vector<double> host(static_cast<size_t>(ctx->nrl) * nS);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    CUDA_CHECK(cudaMemcpy(host.data(), out.p, host.size() * sizeof(double), cudaMemcpyDeviceToHost));
    jdoubleArray arr = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(arr, 0, static_cast<jsize>(host.size()), host.data());
    return arr;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsOutGenerators(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotGen) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotGen == nullptr || env->GetArrayLength(jSlotGen) == 0) {
        ctx->nDsOutGen = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotGen);
    jint* sg = env->GetIntArrayElements(jSlotGen, nullptr);
    ctx->dDsOutGen.upload(sg, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotGen, sg, JNI_ABORT);
    ctx->nDsOutGen = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqWeightOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jW) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jW == nullptr || env->GetArrayLength(jW) == 0) {
        ctx->nDqOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jW);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* w = env->GetDoubleArrayElements(jW, nullptr);
    ctx->dDqOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDqOvW.upload(w, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jW, w, JNI_ABORT);
    ctx->nDqOv = nOv;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqKeyOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jKey) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jKey == nullptr || env->GetArrayLength(jKey) == 0) {
        ctx->nDqKeyOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jKey);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* key = env->GetDoubleArrayElements(jKey, nullptr);
    ctx->dDqKeyOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDqKeyOvKey.upload(key, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jKey, key, JNI_ABORT);
    ctx->nDqKeyOv = nOv;
}

// Per-GENERATOR distributed-slack participants: the factors and limits OLF's water-fill actually
// saturates on, grouped by bus (a bus's row value is the sum over its contiguous slice).
// The slack bus's participating generators' total rooms: the bound of a slack-only distribution (slackOnlyDsCheck).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsSlackRooms(
        JNIEnv*, jclass, jlong handle, jdouble roomUp, jdouble roomDown) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->dsSlackRoomUp = roomUp;
        ctx->dsSlackRoomDown = roomDown;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsGenerators(
        JNIEnv* env, jclass, jlong handle, jintArray jBus, jdoubleArray jFactor,
        jdoubleArray jMaxDelta, jdoubleArray jMinDelta, jdoubleArray jMarginUp, jdoubleArray jMarginDown,
        jintArray jBusGenStart, jintArray jBusGenEnd) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jBus == nullptr || env->GetArrayLength(jBus) == 0) {
        ctx->ngen = 0;
        return;
    }
    int ngen = env->GetArrayLength(jBus);
    auto upI = [&](jintArray a, DeviceBuf& buf) {
        jint* p = env->GetIntArrayElements(a, nullptr);
        buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, p, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& buf) {
        if (a == nullptr || env->GetArrayLength(a) == 0) {
            return;
        }
        jdouble* p = env->GetDoubleArrayElements(a, nullptr);
        buf.upload(p, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, p, JNI_ABORT);
    };
    upI(jBus, ctx->dgBus);
    upD(jFactor, ctx->dgFactor);
    upD(jMaxDelta, ctx->dgMaxDelta);
    upD(jMinDelta, ctx->dgMinDelta);
    upD(jMarginUp, ctx->dgMarginUp);
    upD(jMarginDown, ctx->dgMarginDown);
    upI(jBusGenStart, ctx->dBusGenStart);
    upI(jBusGenEnd, ctx->dBusGenEnd);
    ctx->dGenCum.alloc((size_t) ctx->nScenarios * ngen * sizeof(double));
    ctx->dGenCumValid.alloc((size_t) ctx->nScenarios * sizeof(char));
    ctx->dGenFactorWork.alloc((size_t) ctx->nScenarios * ngen * sizeof(double));
    ctx->dGenRemoved.alloc((size_t) ctx->nScenarios * ngen * sizeof(char));
    ctx->ngen = ngen;
}

// The HVDC AC-emulation outer loops' tables + their per-scenario starting state (the statuses and
// applied converter-row deltas the host's pre-solve freeze produced). Install to run those loops as
// DRIVER PHASES in OLF's order; clear (null) to leave them to the host orchestrator.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchHvdcLoops(
        JNIEnv* env, jclass, jlong handle, jintArray jElem, jintArray jBus1Row, jintArray jBus2Row,
        jintArray jBus1Phi, jintArray jBus2Phi,
        jdoubleArray jIn, jbyteArray jCtgOut, jintArray jStatus,
        jdoubleArray jAppliedD1, jdoubleArray jAppliedD2, jint limitsActive) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jElem == nullptr || env->GetArrayLength(jElem) == 0) {
        ctx->nhv = 0;
        return;
    }
    int nh = env->GetArrayLength(jElem);
    auto upI = [&](jintArray a, DeviceBuf& b) {
        jint* q = env->GetIntArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, q, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& b) {
        jdouble* q = env->GetDoubleArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, q, JNI_ABORT);
    };
    upI(jElem, ctx->dHvElem);
    upI(jBus1Row, ctx->dHvBus1Row);
    upI(jBus2Row, ctx->dHvBus2Row);
    upI(jBus1Phi, ctx->dHvBus1Phi);
    upI(jBus2Phi, ctx->dHvBus2Phi);
    upD(jIn, ctx->dHvIn);
    upI(jStatus, ctx->dHvStatus);
    upD(jAppliedD1, ctx->dHvAppliedD1);
    upD(jAppliedD2, ctx->dHvAppliedD2);
    jbyte* co = env->GetByteArrayElements(jCtgOut, nullptr);
    ctx->dHvCtgOut.upload(co, (size_t) env->GetArrayLength(jCtgOut));
    env->ReleaseByteArrayElements(jCtgOut, co, JNI_ABORT);
    ctx->nhv = nh;
    ctx->hvLimitsActive = limitsActive;
}

/** Enable HOST-owned outer-loop slots via a bitmask.
 *  Bit 0 = SecondaryVoltageControl (OLF order 4, pre-RL),
 *  Bit 1 = PhaseControl (OLF order 7, post-RL),
 *  Bit 2 = TransformerVoltageControl (OLF order 8),
 *  Bit 3 = TransformerReactivePowerControl (OLF order 9),
 *  Bit 4 = ShuntVoltageControl (OLF order 10).
 *  0 disables all host loops; the driver behaves exactly as if the mechanism did not exist. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchHostLoop(
        JNIEnv*, jclass, jlong handle, jint mask) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->hostLoopMask = mask;
    }
}

/** Upload the incremental TVC controller data for the in-kernel outer loop.
 *  rhoRow/vRow/elemIndex: [nCtrl] int; targetV/halfDb: [nCtrl] double;
 *  tapPtr: [nCtrl+1] int (CSR into tapData); tapData: [totalTaps*8] double;
 *  baseTap: [nCtrl] int; groupPtr: [nGroups+1] int (CSR into groupCtrl);
 *  groupCtrl: [nCtrl] int (controller indices per group); maxTapShift: OLF's per-pass bound.
 *  Returns 1 when the kernel loop OWNS the controllers (shared controlled buses included — the
 *  one-tap-at-a-time round robin runs in tvcGroupDecisionBatch), 0 when there are none. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchTvcControllers(
        JNIEnv* env, jclass, jlong handle,
        jintArray jRhoRow, jintArray jVRow, jdoubleArray jTargetV, jdoubleArray jHalfDb,
        jintArray jElemIndex, jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap,
        jintArray jGroupPtr, jintArray jGroupCtrl, jint maxTapShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nTvc = 0;
    if (jRhoRow == nullptr || env->GetArrayLength(jRhoRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> groupPtr;
    readI(jGroupPtr, groupPtr);                              // shared controlled buses run in-kernel too
    std::vector<int> vRow, groupCtrl;
    std::vector<double> targetV, halfDb;
    readI(jRhoRow, ctx->hTvcRhoRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfDb, halfDb);
    readI(jElemIndex, ctx->hTvcElemIndex);
    readI(jTapPtr, ctx->hTvcTapPtr);
    readD(jTapData, ctx->hTvcTapData);
    readI(jBaseTap, ctx->hTvcBaseTap);
    readI(jGroupCtrl, groupCtrl);
    int nc = static_cast<int>(ctx->hTvcRhoRow.size());
    ctx->dTvcRhoRow.upload(ctx->hTvcRhoRow.data(), nc * sizeof(int));
    ctx->dTvcVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->dTvcTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->dTvcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dTvcElemIndex.upload(ctx->hTvcElemIndex.data(), nc * sizeof(int));
    ctx->dTvcTapPtr.upload(ctx->hTvcTapPtr.data(), ctx->hTvcTapPtr.size() * sizeof(int));
    ctx->dTvcTapData.upload(ctx->hTvcTapData.data(), ctx->hTvcTapData.size() * sizeof(double));
    ctx->dTvcBaseTap.upload(ctx->hTvcBaseTap.data(), nc * sizeof(int));
    ctx->nTvcGroups = static_cast<int>(groupPtr.size()) - 1;
    ctx->dTvcGroupPtr.upload(groupPtr.data(), groupPtr.size() * sizeof(int));
    ctx->dTvcGroupCtrl.upload(groupCtrl.data(), groupCtrl.size() * sizeof(int));
    ctx->tvcMaxTapShift = maxTapShift > 0 ? maxTapShift : 1;
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dTvcCurTap.ensure(nst * sizeof(int));
    ctx->dTvcPrevTap.ensure(nst * sizeof(int));
    ctx->dTvcRatchetDir.ensure(nst * sizeof(int));
    ctx->dTvcRatchetCount.ensure(nst * sizeof(int));
    ctx->dTvcRatchetAllowed.ensure(nst * sizeof(int));
    ctx->nTvc = nc;
    return 1;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchMonitoringTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nmon == 0 || !ctx->dMonActivated.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nmon) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dMonActivated.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(cnt));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

// The zone-regulated generator voltage control TARGETS the SVC phase settled on, per
// (scenario, controller) — what OLF's SecondaryVoltageControlOuterLoop leaves on the network. The
// batched SA restores per scenario and needs none of this; the base load flow has to give them back.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchSecondaryVcTargets(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvc <= 0 || !ctx->dSvcTargetV.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvc) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dSvcTargetV.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(cnt));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchTvcTaps(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvc <= 0 || !ctx->dTvcCurTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nTvc) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dTvcCurTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Upload the incremental SHUNT voltage-control tables for the in-kernel loop. Per controller:
 *  bRow (SHUNT_B identity row), vRow (controlled bus), targetV, halfDb, elemAbs (global element, -1),
 *  shOvElem (shunt-pack element for the static (dG, dB) override, -1); groups: groupPtr[nGroups+1] ->
 *  groupCtrl (controllers, |b|-descending); subs: subPtr[nCtrl+1] -> per sub secPtr[nSubs+1] into
 *  secB/secG, subMin, subBase. Returns 1 when the kernel loop owns them, 0 when there are none. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchShuntControllers(
        JNIEnv* env, jclass, jlong handle,
        jintArray jBRow, jintArray jVRow, jdoubleArray jTargetV, jdoubleArray jHalfDb, jintArray jElemAbs, jintArray jShOvElem,
        jintArray jGroupPtr, jintArray jGroupCtrl, jintArray jSubPtr, jintArray jSecPtr, jdoubleArray jSecB, jdoubleArray jSecG,
        jintArray jSubMin, jintArray jSubBase, jint maxSectionShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nShc = 0;
    if (jBRow == nullptr || env->GetArrayLength(jBRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> vRow, elemAbs, groupPtr, groupCtrl, subMin;
    std::vector<double> targetV, halfDb;
    readI(jBRow, ctx->hShcBRow);
    readI(jVRow, vRow);
    readD(jTargetV, targetV);
    readD(jHalfDb, halfDb);
    readI(jElemAbs, elemAbs);
    readI(jShOvElem, ctx->hShcShOvElem);
    readI(jGroupPtr, groupPtr);
    readI(jGroupCtrl, groupCtrl);
    readI(jSubPtr, ctx->hShcSubPtr);
    readI(jSecPtr, ctx->hShcSecPtr);
    readD(jSecB, ctx->hShcSecB);
    readD(jSecG, ctx->hShcSecG);
    readI(jSubMin, subMin);
    readI(jSubBase, ctx->hShcSubBase);
    int nc = static_cast<int>(ctx->hShcBRow.size());
    int nSubs = static_cast<int>(ctx->hShcSubBase.size());
    ctx->dShcBRow.upload(ctx->hShcBRow.data(), nc * sizeof(int));
    ctx->dShcVRow.upload(vRow.data(), nc * sizeof(int));
    ctx->dShcTargetV.upload(targetV.data(), nc * sizeof(double));
    ctx->dShcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dShcElemAbs.upload(elemAbs.data(), nc * sizeof(int));
    ctx->dShcShOvElem.upload(ctx->hShcShOvElem.data(), nc * sizeof(int));
    ctx->dShcGroupPtr.upload(groupPtr.data(), groupPtr.size() * sizeof(int));
    ctx->dShcGroupCtrl.upload(groupCtrl.data(), groupCtrl.size() * sizeof(int));
    ctx->dShcSubPtr.upload(ctx->hShcSubPtr.data(), ctx->hShcSubPtr.size() * sizeof(int));
    ctx->dShcSecPtr.upload(ctx->hShcSecPtr.data(), ctx->hShcSecPtr.size() * sizeof(int));
    ctx->dShcSecB.upload(ctx->hShcSecB.data(), ctx->hShcSecB.size() * sizeof(double));
    ctx->dShcSecG.upload(ctx->hShcSecG.data(), ctx->hShcSecG.size() * sizeof(double));
    ctx->dShcSubMin.upload(subMin.data(), nSubs * sizeof(int));
    ctx->dShcSubBase.upload(ctx->hShcSubBase.data(), nSubs * sizeof(int));
    ctx->nShcGroups = static_cast<int>(groupPtr.size()) - 1;
    ctx->nShcSubs = nSubs;
    ctx->shcMaxSectionShift = maxSectionShift >= 0 ? maxSectionShift : 3;
    size_t nsu = static_cast<size_t>(ctx->nScenarios) * nSubs;
    size_t nsc = static_cast<size_t>(ctx->nScenarios) * nc;
    size_t nsg = static_cast<size_t>(ctx->nScenarios) * ctx->nShcGroups;
    ctx->dShcPos.ensure(nsu * sizeof(int));
    ctx->dShcRatDir.ensure(nsu * sizeof(int));
    ctx->dShcRatCount.ensure(nsu * sizeof(int));
    ctx->dShcRatAllowed.ensure(nsu * sizeof(int));
    ctx->dShcShift.ensure(nsu * sizeof(int));
    ctx->dShcSensi.ensure(nsc * sizeof(double));
    ctx->dShcBaseB.ensure(nsc * sizeof(double));
    ctx->dShcAppliedDb.ensure(nsc * sizeof(double));
    ctx->dShcOutOfDb.ensure(nsg * sizeof(char));
    ctx->dShcDiffV.ensure(nsg * sizeof(double));
    ctx->nShc = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchShuntPositions(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nShc <= 0 || !ctx->dShcPos.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nShcSubs) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dShcPos.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Upload the FLOW-OBSERVED incremental controllers (RPC / incremental phase). Per controller: kind
 *  (0 RPC, 1 phase CONTROLLER, 2 phase LIMITER), pinRow (the rho/a1 identity row), pinCol (6 r1 / 7 a1),
 *  side (1/2) of the controlled flow, targetValue, halfDb, elem (controller's global element),
 *  ctlElem (controlled branch's global element — a closed branch), tapPtr/tapData ([y g1 g2 b1 b2 ksi r1
 *  a1] per tap), baseTap, maxShift. Returns 1 when the kernel loops own them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchFlowControllers(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jPinRow, jintArray jPinCol, jintArray jSide, jdoubleArray jTargetValue, jdoubleArray jHalfDb,
        jintArray jElem, jintArray jCtlElem, jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap, jintArray jMaxShift) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nFlc = 0;
    ctx->nFlcPhase = 0;
    ctx->nFlcRpc = 0;
    if (jKind == nullptr || env->GetArrayLength(jKind) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> pinCol, side, ctlElem, maxShift;
    std::vector<double> targetValue, halfDb;
    readI(jKind, ctx->hFlcKind);
    readI(jPinRow, ctx->hFlcPinRow);
    readI(jPinCol, pinCol);
    readI(jSide, side);
    readD(jTargetValue, targetValue);
    readD(jHalfDb, halfDb);
    readI(jElem, ctx->hFlcElem);
    readI(jCtlElem, ctlElem);
    readI(jTapPtr, ctx->hFlcTapPtr);
    readD(jTapData, ctx->hFlcTapData);
    readI(jBaseTap, ctx->hFlcBaseTap);
    readI(jMaxShift, maxShift);
    int nc = static_cast<int>(ctx->hFlcKind.size());
    std::vector<int> ctlPack(nc);
    for (int c = 0; c < nc; c++) {
        ctlPack[c] = ctlElem[c] - ctx->cbOffset;             // the controlled branch's index in the closed-branch pack
        if (ctlPack[c] < 0 || ctlPack[c] >= ctx->st->cb.ne) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
        if (ctx->hFlcKind[c] == 0) ctx->nFlcRpc++; else ctx->nFlcPhase++;
    }
    ctx->dFlcKind.upload(ctx->hFlcKind.data(), nc * sizeof(int));
    ctx->dFlcPinRow.upload(ctx->hFlcPinRow.data(), nc * sizeof(int));
    ctx->dFlcPinCol.upload(pinCol.data(), nc * sizeof(int));
    ctx->dFlcSide.upload(side.data(), nc * sizeof(int));
    ctx->dFlcTargetValue.upload(targetValue.data(), nc * sizeof(double));
    ctx->dFlcHalfDb.upload(halfDb.data(), nc * sizeof(double));
    ctx->dFlcElem.upload(ctx->hFlcElem.data(), nc * sizeof(int));
    ctx->dFlcCtlPack.upload(ctlPack.data(), nc * sizeof(int));
    ctx->dFlcTapPtr.upload(ctx->hFlcTapPtr.data(), ctx->hFlcTapPtr.size() * sizeof(int));
    ctx->dFlcTapData.upload(ctx->hFlcTapData.data(), ctx->hFlcTapData.size() * sizeof(double));
    ctx->dFlcBaseTap.upload(ctx->hFlcBaseTap.data(), nc * sizeof(int));
    ctx->dFlcMaxShift.upload(maxShift.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dFlcCurTap.ensure(nst * sizeof(int));
    ctx->dFlcPrevTap.ensure(nst * sizeof(int));
    ctx->dFlcRatDir.ensure(nst * sizeof(int));
    ctx->dFlcRatCount.ensure(nst * sizeof(int));
    ctx->dFlcRatAllowed.ensure(nst * sizeof(int));
    ctx->dFlcSensi.ensure(nst * sizeof(double));
    ctx->dFlcDiff.ensure(nst * sizeof(double));
    ctx->dFlcOutOfDb.ensure(nst * sizeof(char));
    ctx->nFlc = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchFlowTaps(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nFlc <= 0 || !ctx->dFlcCurTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nFlc) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dFlcCurTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Upload the CONTINUOUS phase controllers (kind 0 CONTROLLER / 1 LIMITER, a1 identity row, controlled
 *  side, current limit, global element, tap rows, base tap). Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchContinuousPhase(
        JNIEnv* env, jclass, jlong handle,
        jintArray jKind, jintArray jA1Row, jintArray jSide, jdoubleArray jLimit, jintArray jElem,
        jintArray jTapPtr, jdoubleArray jTapData, jintArray jBaseTap) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nCph = 0;
    if (jKind == nullptr || env->GetArrayLength(jKind) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> kind, a1Row, side;
    std::vector<double> limit;
    readI(jKind, kind);
    readI(jA1Row, a1Row);
    readI(jSide, side);
    readD(jLimit, limit);
    readI(jElem, ctx->hCphElem);
    readI(jTapPtr, ctx->hCphTapPtr);
    readD(jTapData, ctx->hCphTapData);
    readI(jBaseTap, ctx->hCphBaseTap);
    int nc = static_cast<int>(kind.size());
    for (int c = 0; c < nc; c++) {
        int e = ctx->hCphElem[c] - ctx->cbOffset;
        if (e < 0 || e >= ctx->st->cb.ne) {
            return 0;                                        // not a closed branch: the host orchestrator keeps it
        }
    }
    ctx->dCphKind.upload(kind.data(), nc * sizeof(int));
    ctx->dCphA1Row.upload(a1Row.data(), nc * sizeof(int));
    ctx->dCphSide.upload(side.data(), nc * sizeof(int));
    ctx->dCphLimit.upload(limit.data(), nc * sizeof(double));
    ctx->dCphElem.upload(ctx->hCphElem.data(), nc * sizeof(int));
    ctx->dCphTapPtr.upload(ctx->hCphTapPtr.data(), ctx->hCphTapPtr.size() * sizeof(int));
    ctx->dCphTapData.upload(ctx->hCphTapData.data(), ctx->hCphTapData.size() * sizeof(double));
    ctx->dCphBaseTap.upload(ctx->hCphBaseTap.data(), nc * sizeof(int));
    size_t nst = static_cast<size_t>(ctx->nScenarios) * nc;
    ctx->dCphTap.ensure(nst * sizeof(int));
    ctx->dCphPrevTap.ensure(nst * sizeof(int));
    ctx->dCphIter.ensure(static_cast<size_t>(ctx->nScenarios) * sizeof(int));
    ctx->nCph = nc;
    return 1;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchContinuousPhaseTaps(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nCph <= 0 || !ctx->dCphTap.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nCph) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dCphTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

/** Install the SecondaryVoltageControl zones for the in-kernel phase (OLF order 4). Controllers are
 *  addressed by their controlled bus's V row (the sensitivity's unit row and where a moved target lands)
 *  and their own bus V row (the reactive injection the K comes from); each carries a CSR slice of Q
 *  observables (kind 0/1 = a closed branch's side-1/side-2 reactive flow, 2 = a shunt) and the zones a
 *  CSR slice of controllers. Returns 1 when the kernel loop owns them — it refuses a zone set wider than
 *  the in-kernel dense system, or an observable on an element the batch does not pack. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchSecondaryVc(
        JNIEnv* env, jclass, jlong handle,
        jintArray jCtldVRow, jintArray jBusVRow, jintArray jRlIdx, jdoubleArray jBaseTargetV,
        jdoubleArray jMinQ, jdoubleArray jMaxQ, jdoubleArray jLoadQ,
        jintArray jQPtr, jintArray jQKind, jintArray jQElem, jintArray jQBRow,
        jintArray jZonePtr, jintArray jZoneCtl, jintArray jZonePilotRow, jdoubleArray jZoneTargetV,
        jdouble minPlausibleV, jdouble maxPlausibleV) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nSvc = 0;
    ctx->nSvcZones = 0;
    if (jCtldVRow == nullptr || env->GetArrayLength(jCtldVRow) == 0 || ctx->nrl <= 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> busVRow, rlIdx, qPtr, qKind, qElem, qBRow, zonePtr, zoneCtl, zonePilotRow;
    std::vector<double> minQ, maxQ, loadQ, zoneTargetV;
    readI(jCtldVRow, ctx->hSvcCtldVRow);
    readI(jBusVRow, busVRow);
    readI(jRlIdx, rlIdx);
    readD(jBaseTargetV, ctx->hSvcBaseTargetV);
    readD(jMinQ, minQ);
    readD(jMaxQ, maxQ);
    readD(jLoadQ, loadQ);
    readI(jQPtr, qPtr);
    readI(jQKind, qKind);
    readI(jQElem, qElem);
    readI(jQBRow, qBRow);
    readI(jZonePtr, zonePtr);
    readI(jZoneCtl, zoneCtl);
    readI(jZonePilotRow, zonePilotRow);
    readD(jZoneTargetV, zoneTargetV);
    int nC = static_cast<int>(ctx->hSvcCtldVRow.size());
    int nZ = static_cast<int>(zonePtr.size()) - 1;
    if (nZ <= 0) {
        return 0;
    }
    // OLF solves ONE system over every enabled controller of every zone, so the size is the total
    // controller count. The batched LU holds an N*N matrix per scenario, which is what actually bounds
    // this: refuse when that workspace would not fit rather than at an arbitrary controller count.
    double matBytes = static_cast<double>(ctx->nScenarios) * nC * nC * sizeof(double);
    if (matBytes > SVC_MAT_BUDGET_BYTES) {
        std::fprintf(stderr, "SVC in-kernel declined: %d controllers x %d scenarios needs %.1f MB of zone-system "
                     "workspace (budget %.0f MB)\n", nC, ctx->nScenarios, matBytes / 1048576.0,
                     SVC_MAT_BUDGET_BYTES / 1048576.0);
        return 0;
    }
    for (int e : qElem) {
        if (e < 0) {
            return 0;                                        // an observable the batch does not pack
        }
    }
    ctx->dSvcCtldVRow.upload(ctx->hSvcCtldVRow.data(), nC * sizeof(int));
    ctx->dSvcBusVRow.upload(busVRow.data(), nC * sizeof(int));
    ctx->dSvcRlIdx.upload(rlIdx.data(), nC * sizeof(int));
    ctx->dSvcBaseTargetV.upload(ctx->hSvcBaseTargetV.data(), nC * sizeof(double));
    ctx->dSvcMinQ.upload(minQ.data(), nC * sizeof(double));
    ctx->dSvcMaxQ.upload(maxQ.data(), nC * sizeof(double));
    ctx->dSvcLoadQ.upload(loadQ.data(), nC * sizeof(double));
    ctx->dSvcQPtr.upload(qPtr.data(), qPtr.size() * sizeof(int));
    ctx->dSvcQKind.upload(qKind.data(), std::max<size_t>(qKind.size(), 1) * sizeof(int));
    ctx->dSvcQElem.upload(qElem.data(), std::max<size_t>(qElem.size(), 1) * sizeof(int));
    ctx->dSvcQBRow.upload(qBRow.data(), std::max<size_t>(qBRow.size(), 1) * sizeof(int));
    ctx->dSvcZonePtr.upload(zonePtr.data(), zonePtr.size() * sizeof(int));
    ctx->dSvcZoneCtl.upload(zoneCtl.data(), zoneCtl.size() * sizeof(int));
    ctx->dSvcZonePilotRow.upload(zonePilotRow.data(), nZ * sizeof(int));
    ctx->dSvcZoneTargetV.upload(zoneTargetV.data(), nZ * sizeof(double));
    size_t nS = ctx->nScenarios;
    ctx->dSvcTargetV.ensure(nS * nC * sizeof(double));
    ctx->dSvcK0.ensure(nS * nC * sizeof(double));
    ctx->dSvcJk.ensure(nS * nC * nC * sizeof(double));
    ctx->dSvcJvpp.ensure(static_cast<size_t>(nS) * static_cast<size_t>(nZ) * nC * sizeof(double));   // per (zone pilot, controlled bus)
    ctx->dSvcEnabled.ensure(nS * nC * sizeof(char));
    ctx->dSvcNeed.ensure(nS * sizeof(char));
    ctx->dSvcReject.ensure(nS * sizeof(char));
    ctx->dSvcRlVt.ensure(nS * std::max(ctx->nrl, 1) * sizeof(double));
    ctx->dSvcMat.ensure(nS * static_cast<size_t>(nC) * nC * sizeof(double));
    ctx->dSvcRhs.ensure(nS * nC * sizeof(double));
    ctx->dSvcMatPtr.ensure(nS * sizeof(double*));
    ctx->dSvcRhsPtr.ensure(nS * sizeof(double*));
    ctx->dSvcPiv.ensure(nS * nC * sizeof(int));
    ctx->dSvcInfo.ensure(nS * sizeof(int));
    {
        std::vector<int> zoneOf(nC, 0);                      // controller -> its zone, for the assembly
        for (int z = 0; z < nZ; z++) {
            for (int k = zonePtr[z]; k < zonePtr[z + 1]; k++) {
                zoneOf[zoneCtl[k]] = z;
            }
        }
        ctx->dSvcCtlZone.upload(zoneOf.data(), zoneOf.size() * sizeof(int));
    }
    {
        // NaN = "this zone has not moved the controller's target", which is what rlTargetOf tests. The
        // reactive-limit SEEDING reads this buffer before the per-solve reset further down runs, so an
        // uninitialised one hands the switch-back a garbage comparison voltage and un-pins controllers
        // that should stay pinned (measured on ieee14's z1: pins 4 -> 2 instead of 4 -> 6, and 1.0e-10 kV).
        std::vector<double> nanv(nS * std::max(ctx->nrl, 1), std::nan(""));
        CUDA_CHECK(cudaMemcpy(ctx->dSvcRlVt.p, nanv.data(), nanv.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    ctx->svcMinPlausibleV = minPlausibleV;
    ctx->svcMaxPlausibleV = maxPlausibleV;
    ctx->nSvcZones = nZ;
    ctx->nSvc = nC;
    return 1;
}

/** Un-pin reactive-limit entries at a driver pause (SecondaryVoltageControl's re-enable step):
 *  {@code pairs} is (scenario, RL entry) flattened, {@code newTarget} the voltage target each row goes
 *  back to. No-op for an entry that is not pinned. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlUnpin(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jdoubleArray jTargets) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nrl <= 0 || jPairs == nullptr || jTargets == nullptr) {
        return;
    }
    int nPairs = env->GetArrayLength(jTargets);
    if (nPairs == 0 || env->GetArrayLength(jPairs) != 2 * nPairs) {
        return;
    }
    std::vector<int> pairs(2 * nPairs);
    std::vector<double> targets(nPairs);
    env->GetIntArrayRegion(jPairs, 0, 2 * nPairs, pairs.data());
    env->GetDoubleArrayRegion(jTargets, 0, nPairs, targets.data());
    DeviceBuf dPairs;
    DeviceBuf dTargets;
    dPairs.upload(pairs.data(), pairs.size() * sizeof(int));
    dTargets.upload(targets.data(), targets.size() * sizeof(double));
    rlUnpinBatch<<<grid(nPairs), BLOCK, 0, ctx->stream>>>(nPairs, ctx->nScenarios, ctx->n, ctx->nrl,
            dPairs.as<int>(), dTargets.as<double>(), ctx->dRlVrow.as<int>(), ctx->dRlLimitType.as<char>(),
            ctx->dRlSavedMode.as<char>(), ctx->dMode.as<int>(), ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

/** Did the last solve stop at a host-owned slot? Returns 0 if not paused, or the
 *  DeviceLoopKind value (LOOP_HOST_SVC=5, LOOP_HOST_PHASE=6, ...) of the paused slot. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDriverPaused(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    return (ctx == nullptr || !ctx->driverPaused) ? 0 : ctx->pausedLoopKind;
}

/** Resume the paused driver: {@code hostChanged[s] != 0} is the host loop's UNSTABLE verdict for
 *  that scenario, which drives the next pass exactly as a device loop's verdict would. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDriverResume(
        JNIEnv* env, jclass, jlong handle, jbyteArray jHostChanged) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hostChanged.assign(ctx->nScenarios, 0);
    if (jHostChanged != nullptr) {
        jsize len = std::min<jsize>(env->GetArrayLength(jHostChanged), ctx->nScenarios);
        env->GetByteArrayRegion(jHostChanged, 0, len, reinterpret_cast<jbyte*>(ctx->hostChanged.data()));
    }
    ctx->driverResume = 1;
}

/** Drop a paused driver sequence without resuming it: the host loop changed the equation layout or
 *  the row modes, so the next solveBatchContext must START OVER on the re-extracted system. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_cancelBatchDriverPause(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->driverPaused = 0;
    ctx->driverResume = 0;
    ctx->pausedLoopKind = 0;
}

/** The next FRESH solve resumes its sweep at this loop kind after its first solve (a restarted
 *  sequence: OLF re-checks the loop that went UNSTABLE before restarting the sweep). The context may
 *  have been rebuilt since the pause, so the Java side carries the kind across. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDriverStartLoop(
        JNIEnv*, jclass, jlong handle, jint kind) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->startLoopKind = kind;
}

/** Write absolute target values at (scenario, row) positions — how a host-owned loop applies its
 *  decision without disturbing the device-side targets the sequence has already accumulated. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchTargetDeltas(
        JNIEnv* env, jclass, jlong handle, jintArray jRows, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || jRows == nullptr) {
        return;
    }
    jsize cnt = env->GetArrayLength(jRows);
    if (cnt == 0) {
        return;
    }
    std::vector<int> rows(cnt);
    std::vector<double> vals(cnt);
    env->GetIntArrayRegion(jRows, 0, cnt, rows.data());
    env->GetDoubleArrayRegion(jVals, 0, cnt, vals.data());
    DeviceBuf dR;
    DeviceBuf dV;
    dR.upload(rows.data(), (size_t) cnt * sizeof(int));
    dV.upload(vals.data(), (size_t) cnt * sizeof(double));
    applyTargetOverrideBatch<<<grid(cnt), BLOCK, 0, ctx->stream>>>(cnt, dR.as<int>(), dV.as<double>(),
                                                                  ctx->dTarget.as<double>());
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

/** MonitoringVoltage controller table (OLF order 6). Empty array clears it. */
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchMonitoringLoops(
        JNIEnv* env, jclass, jlong handle, jintArray jVRow, jdoubleArray jBand,
        jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaMode, jdoubleArray jDeltaTarget) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jVRow == nullptr || env->GetArrayLength(jVRow) == 0) {
        ctx->nmon = 0;
        return;
    }
    auto upI = [&](jintArray a, DeviceBuf& b) {
        jint* q = env->GetIntArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(int));
        env->ReleaseIntArrayElements(a, q, JNI_ABORT);
    };
    auto upD = [&](jdoubleArray a, DeviceBuf& b) {
        jdouble* q = env->GetDoubleArrayElements(a, nullptr);
        b.upload(q, (size_t) env->GetArrayLength(a) * sizeof(double));
        env->ReleaseDoubleArrayElements(a, q, JNI_ABORT);
    };
    upI(jVRow, ctx->dMonVRow);
    upD(jBand, ctx->dMonBand);
    upI(jDeltaPtr, ctx->dMonDeltaPtr);
    upI(jDeltaRow, ctx->dMonDeltaRow);
    upI(jDeltaMode, ctx->dMonDeltaMode);
    upD(jDeltaTarget, ctx->dMonDeltaTarget);
    ctx->nmon = env->GetArrayLength(jVRow);
    ctx->dMonActivated.alloc(static_cast<size_t>(ctx->nmon) * ctx->nScenarios * sizeof(double));
}

/** How many (controller, scenario) monitoring automatons FIRED in the last solve. The host keeps
 *  the ACTIVATIONS counter meaningful now that the loop is a driver phase rather than an
 *  orchestrator — it is the coverage guard that a test case really exercises an activation. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchMonitoringActivations(
        JNIEnv*, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nmon == 0 || !ctx->dMonActivated.p) {
        return 0;
    }
    size_t cnt = static_cast<size_t>(ctx->nmon) * ctx->nScenarios;
    std::vector<double> host(cnt);
    cudaMemcpy(host.data(), ctx->dMonActivated.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    jint fired = 0;
    for (double v : host) {
        if (!std::isnan(v)) {
            fired++;
        }
    }
    return fired;
}

/** Read the HVDC statuses back after a solve, so the host keeps its own view in step. */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchHvdcStatus(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nhv == 0) {
        return nullptr;
    }
    size_t len = (size_t) ctx->nScenarios * ctx->nhv;
    std::vector<int> host(len);
    if (cudaMemcpy(host.data(), ctx->dHvStatus.p, len * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return nullptr;
    }
    jintArray out = env->NewIntArray((jsize) len);
    env->SetIntArrayRegion(out, 0, (jsize) len, host.data());
    return out;
}

// Reactive-limits controllers STRANDED in another synchronous component: install (or clear, with
// null) the per-scenario mask as {slot, controller} pairs.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlNoUnpin(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlNoUnpin = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<char> mask(len, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            mask[(size_t) s * ctx->nrl + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dRlNoUnpin.upload(mask.data(), len);
    std::vector<char> zero(ctx->nScenarios, 0);
    ctx->dRlUnpinBad.upload(zero.data(), ctx->nScenarios);
    ctx->nRlNoUnpin = nPairs;
}

// (scenario, entry) pairs whose bus lost EVERY generator this scenario: seeded FROZEN by initRlLimitTypeBatch.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlGenLost(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlGenLost = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<char> mask(len, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            mask[(size_t) s * ctx->nrl + i] = 1;
        }
    }
    if (std::getenv("OLF_GPU_GENLOST_TRACE") != nullptr) {
        for (int k = 0; k < nPairs; k++) {
            std::fprintf(stderr, "GENLOST slot=%d entry=%d\n", pairs[2 * k], pairs[2 * k + 1]);
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dRlGenLost.upload(mask.data(), len);
    ctx->nRlGenLost = nPairs;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlIslanded(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlIslanded = 0;
        return;
    }
    size_t maskLen = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<char> mask(maskLen, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            mask[(size_t) s * ctx->nrl + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dRlIslanded.upload(mask.data(), maskLen);
    ctx->nRlIslanded = nPairs;
}

// Re-homed voltage controls (GpuBatchQualification.reHomedControls): per (slot, entry) pair the entry is
// checked by the reactive-limits pass as a LOCAL controller at the given voltage target (pu) — the bus is
// a plain PV bus this scenario, whatever its base packing. Null clears the context's list.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlLocal(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jdoubleArray jVt) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || jVt == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlLocal = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<char> mask(len, 0);
    std::vector<double> vt(len, 0.0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    jdouble* vts = env->GetDoubleArrayElements(jVt, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            mask[(size_t) s * ctx->nrl + i] = 1;
            vt[(size_t) s * ctx->nrl + i] = vts[k];
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVt, vts, JNI_ABORT);
    ctx->dRlLocalOv.upload(mask.data(), len);
    ctx->dRlLocalVt.upload(vt.data(), len * sizeof(double));
    ctx->nRlLocal = nPairs;
}

// Remote controllers OLF re-merges onto a NEW pilot after a zero-impedance split: per (scenario, entry) the V row
// the reactive-limit switch-back compares against the (per-scenario) group target - the new pilot's, not the base
// pilot's whose row is frozen with its island. Null or empty clears the context's list.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchRlCmpOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs, jintArray jCmp) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || jCmp == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nrl == 0) {
        ctx->nRlCmp = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nrl;
    std::vector<int> cmp(len, -1);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    jint* rows = env->GetIntArrayElements(jCmp, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nrl) {
            cmp[(size_t) s * ctx->nrl + i] = rows[k];
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    env->ReleaseIntArrayElements(jCmp, rows, JNI_ABORT);
    ctx->dRlCmpOv.upload(cmp.data(), len * sizeof(int));
    ctx->nRlCmp = nPairs;
}

// Per-scenario member targetQ overrides for the DISTR_Q target kernel: {key, value} where key is
// slot * nDqMembers + memberIndex. Set when an outaged bus is a packed group member whose own targetQ
// moved (a reactive LOAD loss). Nulls clear.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDqMemberTargetQOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKeys == nullptr || env->GetArrayLength(jKeys) == 0) {
        ctx->nDqTqOv = 0;
        return;
    }
    int ne = env->GetArrayLength(jKeys);
    jint* keys = env->GetIntArrayElements(jKeys, nullptr);
    jdouble* vals = env->GetDoubleArrayElements(jVals, nullptr);
    ctx->dDqTqOvKey.upload(keys, static_cast<size_t>(ne) * sizeof(int));
    ctx->dDqTqOvVal.upload(vals, static_cast<size_t>(ne) * sizeof(double));
    ctx->nDqTqOv = ne;
    env->ReleaseIntArrayElements(jKeys, keys, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVals, vals, JNI_ABORT);
}

// Per-scenario moves of a pilot's captured reactive target (applyPilotQDeltasBatch): {key, delta} with key
// slot * nQr + group, one entry per key. Nulls clear.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchPilotQDeltas(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKeys == nullptr || env->GetArrayLength(jKeys) == 0) {
        ctx->nQrPqDelta = 0;
        return;
    }
    int ne = env->GetArrayLength(jKeys);
    jint* keys = env->GetIntArrayElements(jKeys, nullptr);
    jdouble* vals = env->GetDoubleArrayElements(jVals, nullptr);
    ctx->dQrPqDeltaKey.upload(keys, static_cast<size_t>(ne) * sizeof(int));
    ctx->dQrPqDeltaVal.upload(vals, static_cast<size_t>(ne) * sizeof(double));
    ctx->nQrPqDelta = ne;
    env->ReleaseIntArrayElements(jKeys, keys, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVals, vals, JNI_ABORT);
}

// Per slot, the row the distributed-slack kernels read the slack on (host row, -1 = the base slack row): a
// contingency that isolates the base slack makes OLF re-pick one, whose angle row becomes the reference. Null clears.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsSlackRows(
        JNIEnv* env, jclass, jlong handle, jintArray jRows) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jRows == nullptr || env->GetArrayLength(jRows) == 0) {
        ctx->hasDsSlackRowOv = 0;
        return;
    }
    int ne = env->GetArrayLength(jRows);
    std::vector<int> rows(ne);
    env->GetIntArrayRegion(jRows, 0, ne, rows.data());
    for (int& r : rows) {
        r = r >= 0 ? ctx->st->rowMap[r] : -1;
    }
    ctx->dDsSlackRowOv.upload(rows.data(), static_cast<size_t>(ne) * sizeof(int));
    ctx->hasDsSlackRowOv = 1;
}

// Pilot-row flip entries (see pilotRowAllPinnedBatch): one per (scenario, pilot) where the outage
// made the pilot stop being one of its own controllers. slots/rows/qTargets are per entry; rlCounts
// says how many surviving-controller RL entries each one owns and rlPool concatenates them in order.
// Nulls clear. Flip state (flipped/savedVTarget) is allocated zeroed here, so each solve starts with
// every pilot row unflipped.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchPilotAllPinnedFlip(
        JNIEnv* env, jclass, jlong handle, jintArray jSlots, jintArray jRows, jdoubleArray jQTargets,
        jintArray jRlCounts, jintArray jRlPool) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlots == nullptr || env->GetArrayLength(jSlots) == 0) {
        ctx->nPilotFlip = 0;
        return;
    }
    int ne = env->GetArrayLength(jSlots);
    jint* slots = env->GetIntArrayElements(jSlots, nullptr);
    jint* rows = env->GetIntArrayElements(jRows, nullptr);
    jdouble* qt = env->GetDoubleArrayElements(jQTargets, nullptr);
    jint* cnt = env->GetIntArrayElements(jRlCounts, nullptr);
    jint* pool = env->GetIntArrayElements(jRlPool, nullptr);
    int poolLen = env->GetArrayLength(jRlPool);
    std::vector<int> off(ne);
    int acc = 0;
    for (int i = 0; i < ne; i++) {
        off[i] = acc;
        acc += cnt[i];
    }
    ctx->dPfSlot.upload(slots, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfRow.upload(rows, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfQTarget.upload(qt, static_cast<size_t>(ne) * sizeof(double));
    ctx->dPfCount.upload(cnt, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfOff.upload(off.data(), static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfRlPool.upload(pool, static_cast<size_t>(poolLen > 0 ? poolLen : 1) * sizeof(int));
    std::vector<char> zeroC(ne, 0);
    std::vector<double> zeroD(ne, 0.0);
    ctx->dPfFlipped.upload(zeroC.data(), static_cast<size_t>(ne) * sizeof(char));
    ctx->dPfSavedVT.upload(zeroD.data(), static_cast<size_t>(ne) * sizeof(double));
    ctx->nPilotFlip = ne;
    env->ReleaseIntArrayElements(jSlots, slots, JNI_ABORT);
    env->ReleaseIntArrayElements(jRows, rows, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jQTargets, qt, JNI_ABORT);
    env->ReleaseIntArrayElements(jRlCounts, cnt, JNI_ABORT);
    env->ReleaseIntArrayElements(jRlPool, pool, JNI_ABORT);
}

// Distributed-slack participants STRANDED in another synchronous component: install (or clear, with
// null) the per-scenario mask as {slot, participant} pairs. A contingency that fragments the network
// leaves some participating buses outside the main component; OLF drops them from the participant
// set, so the device must renormalize over the survivors — see distributeSlackBatch.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsIslanded(
        JNIEnv* env, jclass, jlong handle, jintArray jPairs) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jPairs == nullptr || env->GetArrayLength(jPairs) == 0 || ctx->nds == 0) {
        ctx->nDsIslanded = 0;
        return;
    }
    // Expanded to a dense (scenario, participant) mask on the host: the kernel reads it once per
    // participant inside loops it already walks, so a scan over a pair list would be the only cost
    // the mask adds to scenarios that have no islanded participant at all.
    size_t maskLen = static_cast<size_t>(ctx->nScenarios) * ctx->nds;
    std::vector<char> mask(maskLen, 0);
    int nPairs = env->GetArrayLength(jPairs) / 2;
    jint* pairs = env->GetIntArrayElements(jPairs, nullptr);
    for (int k = 0; k < nPairs; k++) {
        int s = pairs[2 * k];
        int i = pairs[2 * k + 1];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nds) {
            mask[(size_t) s * ctx->nds + i] = 1;
        }
    }
    env->ReleaseIntArrayElements(jPairs, pairs, JNI_ABORT);
    ctx->dDsIslanded.upload(mask.data(), maskLen);
    ctx->nDsIslanded = nPairs;
}

// Per-scenario P-balance row of a DS participant this scenario MOVES: {slot, participant, row}
// triples, expanded to a dense (scenario, participant) row array (-1 = the structural dsPhiRow).
// The rows arrive in the same space dsPhiRow is given in and are taken through the structure's
// rowMap here, exactly as createBatchContext maps dsPhiRow itself — so the two are always in one
// space whatever rowMap is.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDsRowOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jTriples) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jTriples == nullptr || env->GetArrayLength(jTriples) == 0 || ctx->nds == 0) {
        ctx->nDsRowOv = 0;
        return;
    }
    size_t len = static_cast<size_t>(ctx->nScenarios) * ctx->nds;
    std::vector<int> rows(len, -1);
    int nTriples = env->GetArrayLength(jTriples) / 3;
    jint* t = env->GetIntArrayElements(jTriples, nullptr);
    const std::vector<int>& rowMap = ctx->st->rowMap;
    int nOk = 0;
    for (int k = 0; k < nTriples; k++) {
        int s = t[3 * k];
        int i = t[3 * k + 1];
        int row = t[3 * k + 2];
        if (s >= 0 && s < ctx->nScenarios && i >= 0 && i < ctx->nds
                && row >= 0 && row < static_cast<int>(rowMap.size()) && rowMap[row] >= 0) {
            rows[(size_t) s * ctx->nds + i] = rowMap[row];
            nOk++;
        }
    }
    env->ReleaseIntArrayElements(jTriples, t, JNI_ABORT);
    ctx->dDsRowOv.upload(rows.data(), len * sizeof(int));
    ctx->nDsRowOv = nOk;
}

// DISTR_RHO coefficient overrides: install (or clear, with nulls) the per-scenario 1/count'
// renormalization pairs {s, contrib} + new coefficients on a batch context (a shared-TVC group
// member outaged in that scenario — plan-group-member-dynamics A5).
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDrCoefOverrides(
        JNIEnv* env, jclass, jlong handle, jintArray jIdx, jdoubleArray jC) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jIdx == nullptr || jC == nullptr || env->GetArrayLength(jC) == 0) {
        ctx->nDrOv = 0;
        return;
    }
    int nOv = env->GetArrayLength(jC);
    jint* idx = env->GetIntArrayElements(jIdx, nullptr);
    jdouble* c = env->GetDoubleArrayElements(jC, nullptr);
    ctx->dDrOvIdx.upload(idx, static_cast<size_t>(nOv) * 2 * sizeof(int));
    ctx->dDrOvC.upload(c, static_cast<size_t>(nOv) * sizeof(double));
    env->ReleaseIntArrayElements(jIdx, idx, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jC, c, JNI_ABORT);
    ctx->nDrOv = nOv;
}

// Per-scenario ACTIVE equation counts for the L2 stopping threshold (see hNActive). Per-chunk state.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchActiveRowCounts(
        JNIEnv* env, jclass, jlong handle, jintArray jNActive) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    ctx->hNActive.clear();
    if (jNActive != nullptr) {
        int len = env->GetArrayLength(jNActive);
        ctx->hNActive.resize(len);
        env->GetIntArrayRegion(jNActive, 0, len, ctx->hNActive.data());
    }
}


/** Install the DISCRETE transformer voltage-control tables (OLF order 8): mode 0 = AFTER_GENERATOR
 *  (TransformerVoltageControlOuterLoop), 1 = WITH_GENERATOR (SimpleTransformerVoltageControlOuterLoop);
 *  entryOn = the uploaded activation already carries the control ON (a base load flow whose network
 *  OLF's initialize switched on). Groups = controlled buses (CSR over members), each with its toggle-
 *  diffed ON/OFF row deltas (CSR); members with rho row, element, base tap, tap table, r1 range and
 *  initial r1. Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteTvc(
        JNIEnv* env, jclass, jlong handle, jint mode, jint entryOn,
        jintArray jGrpPtr, jintArray jGrpVRow, jdoubleArray jGrpTargetV, jdoubleArray jGrpHalfDb, jintArray jGrpUseInit,
        jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaModeOn, jintArray jDeltaModeOff,
        jdoubleArray jDeltaTargetOn, jdoubleArray jDeltaTargetOff,
        jintArray jMemRhoRow, jintArray jMemElem, jintArray jMemBaseTap, jintArray jMemTapPtr,
        jdoubleArray jMemMin, jdoubleArray jMemMax, jdoubleArray jMemInit, jdoubleArray jTapData) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nTvd = 0;
    ctx->nTvdFz = 0;
    ctx->tvdFixTables = false;
    ctx->tvdFixWords = 0;
    if (jMemRhoRow == nullptr || env->GetArrayLength(jMemRhoRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> grpPtr, grpVRow, grpUseInit, deltaPtr, deltaRow, deltaModeOn, deltaModeOff, memRhoRow;
    std::vector<double> grpTargetV, grpHalfDb, deltaTargetOn, deltaTargetOff, memMin, memMax, memInit;
    readI(jGrpPtr, grpPtr);
    readI(jGrpVRow, grpVRow);
    readD(jGrpTargetV, grpTargetV);
    readD(jGrpHalfDb, grpHalfDb);
    readI(jGrpUseInit, grpUseInit);
    readI(jDeltaPtr, deltaPtr);
    readI(jDeltaRow, deltaRow);
    readI(jDeltaModeOn, deltaModeOn);
    readI(jDeltaModeOff, deltaModeOff);
    readD(jDeltaTargetOn, deltaTargetOn);
    readD(jDeltaTargetOff, deltaTargetOff);
    readI(jMemRhoRow, memRhoRow);
    readI(jMemElem, ctx->hTvdMemElem);
    readI(jMemBaseTap, ctx->hTvdMemBaseTap);
    readI(jMemTapPtr, ctx->hTvdMemTapPtr);
    readD(jMemMin, memMin);
    readD(jMemMax, memMax);
    readD(jMemInit, memInit);
    readD(jTapData, ctx->hTvdTapData);
    int nM = static_cast<int>(memRhoRow.size());
    int nG = static_cast<int>(grpPtr.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    for (int m = 0; m < nM; m++) {
        if (ctx->hTvdMemTapPtr[m + 1] - ctx->hTvdMemTapPtr[m] <= 0) {
            return 0;                                        // no tap table: the host orchestrator keeps the control
        }
    }
    ctx->dTvdGrpPtr.upload(grpPtr.data(), grpPtr.size() * sizeof(int));
    ctx->dTvdGrpVRow.upload(grpVRow.data(), nG * sizeof(int));
    ctx->dTvdGrpTargetV.upload(grpTargetV.data(), nG * sizeof(double));
    ctx->dTvdGrpHalfDb.upload(grpHalfDb.data(), nG * sizeof(double));
    ctx->dTvdGrpUseInit.upload(grpUseInit.data(), nG * sizeof(int));
    ctx->dTvdDeltaPtr.upload(deltaPtr.data(), deltaPtr.size() * sizeof(int));
    ctx->dTvdDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->dTvdDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->dTvdDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->dTvdMemRhoRow.upload(memRhoRow.data(), nM * sizeof(int));
    ctx->dTvdMemElem.upload(ctx->hTvdMemElem.data(), nM * sizeof(int));
    ctx->dTvdMemBaseTap.upload(ctx->hTvdMemBaseTap.data(), nM * sizeof(int));
    ctx->dTvdMemTapPtr.upload(ctx->hTvdMemTapPtr.data(), ctx->hTvdMemTapPtr.size() * sizeof(int));
    ctx->dTvdMemMin.upload(memMin.data(), nM * sizeof(double));
    ctx->dTvdMemMax.upload(memMax.data(), nM * sizeof(double));
    ctx->dTvdMemInit.upload(memInit.data(), nM * sizeof(double));
    ctx->dTvdTapData.upload(ctx->hTvdTapData.data(), ctx->hTvdTapData.size() * sizeof(double));
    size_t nS = ctx->nScenarios;
    ctx->dTvdScnStep.ensure(nS * sizeof(int));
    ctx->dTvdGrpStep.ensure(nS * nG * sizeof(int));
    ctx->dTvdGrpShMin.ensure(nS * nG * sizeof(double));
    ctx->dTvdGrpShMax.ensure(nS * nG * sizeof(double));
    ctx->dTvdGrpShInit.ensure(nS * nG * sizeof(double));
    ctx->dTvdMemState.ensure(nS * nM * sizeof(int));
    ctx->dTvdMemTap.ensure(nS * nM * sizeof(int));
    ctx->dTvdMemRoundFrom.ensure(nS * nM * sizeof(double));
    ctx->dTvdSaved.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->dTvdReject.ensure(nS * sizeof(char));
    ctx->nTvdGroups = nG;
    ctx->nTvdDelta = nD;
    ctx->tvdMode = mode;
    ctx->tvdEntryOn = entryOn;
    ctx->nTvd = nM;
    return 1;
}

/** GeneratorVoltageControlManager tables for the AFTER_GENERATOR phase installed by setBatchDiscreteTvc:
 *  the freeze candidates (controller bus V row, load Q) and fixTransformerVoltageControls' per-member
 *  PV-candidate row lists (CSR, [nM+1]; null = no connectivity fix). Returns 1 when installed. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteTvcFreeze(
        JNIEnv* env, jclass, jlong handle, jintArray jFzVRow, jdoubleArray jFzLoadQ, jintArray jFzRlEntry,
        jintArray jFixPvRow) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> vRow, fixRow;
    std::vector<double> loadQ;
    if (jFzVRow != nullptr) {
        readI(jFzVRow, vRow);
        loadQ.resize(env->GetArrayLength(jFzLoadQ));
        env->GetDoubleArrayRegion(jFzLoadQ, 0, static_cast<jsize>(loadQ.size()), loadQ.data());
    }
    int nFz = static_cast<int>(vRow.size());
    if (nFz > 0) {
        std::vector<int> rlEntry;
        if (jFzRlEntry != nullptr) {
            readI(jFzRlEntry, rlEntry);
        }
        if (static_cast<int>(rlEntry.size()) != nFz) {
            rlEntry.assign(nFz, -1);                         // no remote candidate
        }
        ctx->dTvdFzVRow.upload(vRow.data(), nFz * sizeof(int));
        ctx->dTvdFzLoadQ.upload(loadQ.data(), nFz * sizeof(double));
        ctx->dTvdFzRlEntry.upload(rlEntry.data(), nFz * sizeof(int));
        size_t nsc = static_cast<size_t>(ctx->nScenarios) * nFz;
        ctx->dTvdFzFrozen.ensure(nsc * sizeof(int));
        ctx->dTvdFzSavedMode.ensure(nsc * sizeof(int));
        ctx->dTvdFzSavedTarget.ensure(nsc * sizeof(double));
    }
    ctx->nTvdFz = nFz;
    ctx->tvdFixTables = false;                               // armed by setBatchDiscreteTvcFixMasks
    ctx->tvdFixWords = 0;
    if (jFixPvRow != nullptr) {
        readI(jFixPvRow, fixRow);
        if (!fixRow.empty()) {
            ctx->dTvdFixPvRow.upload(fixRow.data(), fixRow.size() * sizeof(int));
        }
    }
    return 1;
}

// The per-(scenario, member) connectivity masks of fixTransformerVoltageControls. Uploaded per CHUNK,
// unlike the tables above: OLF re-runs the check on the POST-CONTINGENCY network, so the component of a
// not-controlled side is a property of the scenario, not of the run.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteTvcFixMasks(
        JNIEnv* env, jclass, jlong handle, jintArray jMin, jintArray jMax, jint words) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0) {
        return;
    }
    ctx->tvdFixTables = false;
    ctx->tvdFixWords = 0;
    if (jMin == nullptr || jMax == nullptr || words <= 0) {
        return;
    }
    size_t want = static_cast<size_t>(ctx->nScenarios) * ctx->nTvd * words;
    if (static_cast<size_t>(env->GetArrayLength(jMin)) != want
            || static_cast<size_t>(env->GetArrayLength(jMax)) != want) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      "setBatchDiscreteTvcFixMasks: array length != nScenarios * nMembers * words");
        return;
    }
    std::vector<int> mn(want), mx(want);
    env->GetIntArrayRegion(jMin, 0, static_cast<jsize>(want), mn.data());
    env->GetIntArrayRegion(jMax, 0, static_cast<jsize>(want), mx.data());
    ctx->dTvdFixMinMask.upload(mn.data(), want * sizeof(int));
    ctx->dTvdFixMaxMask.upload(mx.data(), want * sizeof(int));
    ctx->tvdFixWords = words;
    ctx->tvdFixTables = true;
}

/** Per (scenario, freeze candidate): {frozen (0/1), the row target while frozen (gen − load)} — layout
 *  {@code (s * nFz + c) * 2}; null without candidates. */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteTvcFreeze(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0 || ctx->nTvdFz <= 0 || !ctx->dTvdFzFrozen.p) {
        return nullptr;
    }
    int n = ctx->n;
    size_t cnt = static_cast<size_t>(ctx->nTvdFz) * ctx->nScenarios;
    std::vector<int> fz(cnt), vrow(ctx->nTvdFz);
    std::vector<double> tgt(static_cast<size_t>(n) * ctx->nScenarios), out(cnt * 2);
    cudaMemcpy(fz.data(), ctx->dTvdFzFrozen.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vrow.data(), ctx->dTvdFzVRow.p, vrow.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tgt.data(), ctx->dTarget.p, tgt.size() * sizeof(double), cudaMemcpyDeviceToHost);
    for (int s = 0; s < ctx->nScenarios; s++) {
        for (int c = 0; c < ctx->nTvdFz; c++) {
            size_t i = static_cast<size_t>(s) * ctx->nTvdFz + c;
            out[i * 2] = fz[i];
            out[i * 2 + 1] = tgt[static_cast<size_t>(s) * n + vrow[c]];
        }
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Per (scenario, member): {state (0 off, 1 continuous, 2 fixed), tap index, the r1 fed to the
 *  rounding} — layout {@code (s * nM + m) * 3}; null without the table. */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteTvcState(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nTvd <= 0 || !ctx->dTvdMemState.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nTvd) * ctx->nScenarios;
    std::vector<int> st(cnt), tap(cnt);
    std::vector<double> rf(cnt), out(cnt * 3);
    cudaMemcpy(st.data(), ctx->dTvdMemState.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tap.data(), ctx->dTvdMemTap.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(rf.data(), ctx->dTvdMemRoundFrom.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < cnt; i++) {
        out[i * 3] = st[i];
        out[i * 3 + 1] = tap[i];
        out[i * 3 + 2] = rf[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Install the DISCRETE (WITH_GENERATOR) shunt voltage-control tables (OLF order 10): groups =
 *  controlled buses (CSR over member shunts) with their toggle-diffed ON/OFF row deltas; members with
 *  SHUNT_B row, element (-1 = unpacked, never outaged) and sub-controllers (CSR) whose section-B tables,
 *  min and base positions drive LfShuntImpl.dispatchB. Returns 1 when the kernel loop owns them. */
extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_setBatchDiscreteShunt(
        JNIEnv* env, jclass, jlong handle, jint entryOn,
        jintArray jGrpPtr, jintArray jDeltaPtr, jintArray jDeltaRow, jintArray jDeltaModeOn, jintArray jDeltaModeOff,
        jdoubleArray jDeltaTargetOn, jdoubleArray jDeltaTargetOff,
        jintArray jMemBRow, jintArray jMemElem, jintArray jMemSubPtr,
        jintArray jSecPtr, jdoubleArray jSecB, jintArray jSubMin, jintArray jSubBase) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->nSvd = 0;
    if (jMemBRow == nullptr || env->GetArrayLength(jMemBRow) == 0) {
        return 0;
    }
    auto readI = [&](jintArray a, std::vector<int>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetIntArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    auto readD = [&](jdoubleArray a, std::vector<double>& v) {
        v.resize(env->GetArrayLength(a));
        env->GetDoubleArrayRegion(a, 0, static_cast<jsize>(v.size()), v.data());
    };
    std::vector<int> grpPtr, deltaPtr, deltaRow, deltaModeOn, deltaModeOff, memBRow, memElem, memSubPtr, secPtr, subMin, subBase;
    std::vector<double> deltaTargetOn, deltaTargetOff, secB;
    readI(jGrpPtr, grpPtr);
    readI(jDeltaPtr, deltaPtr);
    readI(jDeltaRow, deltaRow);
    readI(jDeltaModeOn, deltaModeOn);
    readI(jDeltaModeOff, deltaModeOff);
    readD(jDeltaTargetOn, deltaTargetOn);
    readD(jDeltaTargetOff, deltaTargetOff);
    readI(jMemBRow, memBRow);
    readI(jMemElem, memElem);
    readI(jMemSubPtr, memSubPtr);
    readI(jSecPtr, secPtr);
    readD(jSecB, secB);
    readI(jSubMin, subMin);
    readI(jSubBase, subBase);
    int nM = static_cast<int>(memBRow.size());
    int nG = static_cast<int>(grpPtr.size()) - 1;
    int nD = static_cast<int>(deltaRow.size());
    int nSubs = static_cast<int>(subBase.size());
    ctx->dSvdGrpPtr.upload(grpPtr.data(), grpPtr.size() * sizeof(int));
    ctx->dSvdDeltaPtr.upload(deltaPtr.data(), deltaPtr.size() * sizeof(int));
    ctx->dSvdDeltaRow.upload(deltaRow.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaModeOn.upload(deltaModeOn.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaModeOff.upload(deltaModeOff.data(), std::max(nD, 1) * sizeof(int));
    ctx->dSvdDeltaTargetOn.upload(deltaTargetOn.data(), std::max(nD, 1) * sizeof(double));
    ctx->dSvdDeltaTargetOff.upload(deltaTargetOff.data(), std::max(nD, 1) * sizeof(double));
    ctx->dSvdMemBRow.upload(memBRow.data(), nM * sizeof(int));
    ctx->dSvdMemElem.upload(memElem.data(), nM * sizeof(int));
    ctx->dSvdMemSubPtr.upload(memSubPtr.data(), memSubPtr.size() * sizeof(int));
    ctx->dSvdSecPtr.upload(secPtr.data(), secPtr.size() * sizeof(int));
    ctx->dSvdSecB.upload(secB.data(), std::max<size_t>(secB.size(), 1) * sizeof(double));
    ctx->dSvdSubMin.upload(subMin.data(), std::max(nSubs, 1) * sizeof(int));
    ctx->dSvdSubBase.upload(subBase.data(), std::max(nSubs, 1) * sizeof(int));
    size_t nS = ctx->nScenarios;
    ctx->dSvdGrpStep.ensure(nS * nG * sizeof(int));
    ctx->dSvdMemState.ensure(nS * nM * sizeof(int));
    ctx->dSvdMemContB.ensure(nS * nM * sizeof(double));
    ctx->dSvdSubPos.ensure(nS * std::max(nSubs, 1) * sizeof(int));
    ctx->dSvdSaved.ensure(nS * std::max(nD, 1) * sizeof(double));
    ctx->nSvdGroups = nG;
    ctx->nSvdDelta = nD;
    ctx->nSvdSubs = nSubs;
    ctx->svdEntryOn = entryOn;
    ctx->nSvd = nM;
    return 1;
}

/** Per (scenario, member shunt): {state (0 off, 1 continuous, 2 dispatched), the continuous b the
 *  dispatch started from} — layout {@code (s * nM + m) * 2}; null without the table. */
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteShuntState(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvd <= 0 || !ctx->dSvdMemState.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvd) * ctx->nScenarios;
    std::vector<int> st(cnt);
    std::vector<double> cb(cnt), out(cnt * 2);
    cudaMemcpy(st.data(), ctx->dSvdMemState.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cb.data(), ctx->dSvdMemContB.p, cnt * sizeof(double), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < cnt; i++) {
        out[i * 2] = st[i];
        out[i * 2 + 1] = cb[i];
    }
    jdoubleArray res = env->NewDoubleArray(static_cast<jsize>(out.size()));
    env->SetDoubleArrayRegion(res, 0, static_cast<jsize>(out.size()), out.data());
    return res;
}

/** Per (scenario, sub-controller): the dispatched section — layout {@code s * nSubs + u}. */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_readBatchDiscreteShuntPositions(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nSvd <= 0 || ctx->nSvdSubs <= 0 || !ctx->dSvdSubPos.p) {
        return nullptr;
    }
    size_t cnt = static_cast<size_t>(ctx->nSvdSubs) * ctx->nScenarios;
    std::vector<int> host(cnt);
    cudaMemcpy(host.data(), ctx->dSvdSubPos.p, cnt * sizeof(int), cudaMemcpyDeviceToHost);
    jintArray out = env->NewIntArray(static_cast<jsize>(cnt));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(cnt), host.data());
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_destroyBatchContext(
        JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<BatchSolverContext*>(handle);
}

// VALUES refresh addressed by the SHARED STRUCTURE handle (no solver context needed): the
// sequential batch-of-1 path reconciles tap/shunt moves the same way the seq context did —
// one extraction has ONE value set, and every context sharing the structure sees the update.
extern "C" JNIEXPORT void JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_refreshStructureValues(
        JNIEnv* env, jclass, jlong structureHandle, jdoubleArray jcbIn, jdoubleArray jobIn,
        jdoubleArray jshIn, jdoubleArray jhvIn, jdoubleArray jziIn) {
    auto* stp = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    jdouble* cbIn = env->GetDoubleArrayElements(jcbIn, nullptr);
    jdouble* obIn = env->GetDoubleArrayElements(jobIn, nullptr);
    jdouble* shIn = env->GetDoubleArrayElements(jshIn, nullptr);
    jdouble* hvIn = env->GetDoubleArrayElements(jhvIn, nullptr);
    jdouble* ziIn = env->GetDoubleArrayElements(jziIn, nullptr);
    std::string error;
    try {
        if (stp == nullptr || !*stp) {
            throw std::runtime_error("invalid structure handle");
        }
        refreshStructureValues(**stp, env->GetArrayLength(jcbIn) / 8, cbIn, env->GetArrayLength(jobIn) / 7, obIn,
                               env->GetArrayLength(jshIn) / 2, shIn, env->GetArrayLength(jhvIn) / 5, hvIn,
                               env->GetArrayLength(jziIn), ziIn);
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }
    env->ReleaseDoubleArrayElements(jcbIn, cbIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jobIn, obIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jshIn, shIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jhvIn, hvIn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jziIn, ziIn, JNI_ABORT);
    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU structure values refresh failed: " + error).c_str());
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_recommendBatchSize(
        JNIEnv* env, jclass, jlong structureHandle, jint maxCap) {
    auto* stp = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    int rec = 1;
    try {
        if (stp == nullptr || !*stp) {
            throw std::runtime_error("invalid structure handle");
        }
        SharedStructure& st = **stp;
        // Per-scenario device footprint: the dominant term is cuDSS's LU numeric factors
        // (the exact fill from a probe analysis — cached on the structure — or the heuristic),
        // plus the value/state arrays (vals + x/target/F/B/dx). The LU index structure is
        // shared across the batch.
        constexpr double SAFETY = 0.7;                       // leave headroom for cuDSS workspace + fragmentation
        double luBytes = luBytesPerScenario(st);
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));  // after the probe freed its buffers
        // 7n doubles per scenario, not 5: x/target/F/B/dx PLUS dQbuf (the per-scenario reactive
        // accumulator) and dXprevLs (the line-search previous state, allocated under the RTE
        // parameters' MAX_VOLTAGE_CHANGE scaling).
        double perScenarioBytes = luBytes
                + (static_cast<double>(st.nnz) + 7.0 * st.n) * sizeof(double)
                + static_cast<double>(st.n) * sizeof(int)    // dMode
                + static_cast<double>(st.nElements);         // dDisabled (char)
        double budget = static_cast<double>(freeBytes) * SAFETY;
        long fit = perScenarioBytes > 0 ? static_cast<long>(budget / perScenarioBytes) : maxCap;
        rec = static_cast<int>(std::max<long>(1, std::min<long>(maxCap, fit)));
    } catch (...) {
        rec = 1;                                             // a query failure yields the safest chunk
    }
    return rec;
}

// ---------------------------------------------------------------------------------
// ADJOINT step 1a: solve Aᵀ x = b for an EXPLICIT CSR matrix, by building the transpose
// and factorizing it.
//
// Why this exists: cuDSS (through 0.8.0 — verified, its header has no transpose/OP_T
// symbol at all) offers only the non-transposed solve against its factors, so one GPU
// factorization yields ONE direction where KLU's klu_solve/klu_tsolve pair yields two.
// Our Jacobian is stored as TRUE CSR (row = equation, col = variable) so the plain solve
// serves the forward Newton; the adjoint (Jᵀλ = ...) is therefore the direction we cannot
// get from the forward's factors, and until cuDSS ships a transpose solve we buy it with a
// second factorization of an explicitly transposed matrix.
//
// This entry point is deliberately SELF-CONTAINED (its own handle/config/data, no reuse of
// the Newton context): it isolates exactly one question — is our CSR transpose + cuDSS
// factorize/solve of Aᵀ correct? — so a failure here cannot be confused with the adjoint's
// RHS assembly or contraction. It is a validation//measurement entry, not the production
// path; the production seam should reuse the forward's structure and swap to a native
// transpose solve the moment cuDSS provides one.
//
// CSR transpose is the standard counting sort: count entries per column of A (= rows of Aᵀ),
// prefix-sum into row pointers, then scatter. O(nnz + n), host-side, and for a FIXED pattern
// the resulting scatter is a permutation that can be computed once and reused per solve.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveTransposedExplicit(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx,
        jdoubleArray jVals, jdoubleArray jB) {
    try {
        const int nn = n;
        std::vector<int> rowPtr(nn + 1);
        env->GetIntArrayRegion(jRowPtr, 0, nn + 1, rowPtr.data());
        const int nnz = rowPtr[nn];
        std::vector<int> colIdx(nnz);
        std::vector<double> vals(nnz);
        std::vector<double> b(nn);
        env->GetIntArrayRegion(jColIdx, 0, nnz, colIdx.data());
        env->GetDoubleArrayRegion(jVals, 0, nnz, vals.data());
        env->GetDoubleArrayRegion(jB, 0, nn, b.data());

        // --- CSR(A) -> CSR(Aᵀ) by counting sort ---
        std::vector<int> tRowPtr(nn + 1, 0);
        std::vector<int> tColIdx(nnz);
        std::vector<double> tVals(nnz);
        for (int k = 0; k < nnz; k++) {
            tRowPtr[colIdx[k] + 1]++;                        // column of A becomes row of Aᵀ
        }
        for (int r = 0; r < nn; r++) {
            tRowPtr[r + 1] += tRowPtr[r];
        }
        std::vector<int> cursor(tRowPtr.begin(), tRowPtr.end() - 1);
        for (int r = 0; r < nn; r++) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
                const int dst = cursor[colIdx[k]]++;
                tColIdx[dst] = r;                            // row of A becomes column of Aᵀ
                tVals[dst] = vals[k];
            }
        }

        DeviceBuf dRowPtr;
        DeviceBuf dColIdx;
        DeviceBuf dVals;
        DeviceBuf dB;
        DeviceBuf dX;
        dRowPtr.upload(tRowPtr.data(), (nn + 1) * sizeof(int));
        dColIdx.upload(tColIdx.data(), nnz * sizeof(int));
        dVals.upload(tVals.data(), nnz * sizeof(double));
        dB.upload(b.data(), nn * sizeof(double));
        dX.alloc(nn * sizeof(double));

        CudssCtx ds;
        CUDSS_CHECK(cudssCreate(&ds.handle));
        CUDSS_CHECK(cudssConfigCreate(&ds.config));
        applyReorderAlg(ds.config, false);   // adjoint: its own matrix, matching enabled
        applyMatchingAlg(ds.config);
        CUDSS_CHECK(cudssDataCreate(ds.handle, &ds.data));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ds.matA, nn, nn, nnz, dRowPtr.as<int>(), nullptr,
                                         dColIdx.as<int>(), dVals.as<double>(), OLF_CUDSS_CSR_TYPES,
                                         CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matB, nn, 1, nn, dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matX, nn, 1, nn, dX.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ds.handle, CUDSS_PHASE_ANALYSIS, ds.config, ds.data, ds.matA, ds.matX, ds.matB)); }
        { AnalysisTimer at_(__LINE__, 'F'); CUDSS_CHECK(cudssExecute(ds.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                                 ds.config, ds.data, ds.matA, ds.matX, ds.matB)); }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> x(nn);
        CUDA_CHECK(cudaMemcpy(x.data(), dX.p, nn * sizeof(double), cudaMemcpyDeviceToHost));
        jdoubleArray out = env->NewDoubleArray(nn);
        env->SetDoubleArrayRegion(out, 0, nn, x.data());
        return out;
    } catch (const std::exception& e) {
        jclass ex = env->FindClass("java/lang/RuntimeException");
        if (ex) env->ThrowNew(ex, e.what());
        return nullptr;
    }
}
