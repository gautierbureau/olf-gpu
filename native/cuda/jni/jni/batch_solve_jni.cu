// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: The batched context and its solve.
#include "jni_common.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"

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

extern "C" JNIEXPORT jlong JNICALL
OLF_JNI(ac_solver_gpu, BatchSolveNative, createBatchContext)(
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

extern "C" JNIEXPORT jobjectArray JNICALL
OLF_JNI(ac_solver_gpu, BatchSolveNative, solveBatchContext)(
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
        for (int k = 0; k < GpuOuterLoopKind::COUNT; k++) {
            if (gOlReSolvePasses[k] > 0) {
                fprintf(stderr, "GPU_OL_TIME re-solve trigger=%s passes=%lld ms=%.1f scenarioSolves=%lld laneIters=%lld widthIters=%lld\n",
                        olKindName(k), gOlReSolvePasses[k], gOlReSolveMs[k], gOlReSolveScen[k], gOlReSolveLaneIters[k], gOlReSolveWidthIters[k]);
            }
        }
        for (int k = 0; k < GpuOuterLoopKind::COUNT; k++) {
            if (gOlCheckPasses[k] > 0) {
                fprintf(stderr, "GPU_OL_TIME check loop=%s passes=%lld ms=%.1f\n", olKindName(k), gOlCheckPasses[k], gOlCheckMs[k]);
            }
        }
        fprintf(stderr, "CUDSS_PHASE_TIME analysis=%lld x %.1f ms | factorization=%lld x %.1f ms | solve=%lld x %.1f ms"
                " | factorization+solve=%lld x %.1f ms\n", cnt[0], tot[0], cnt[1], tot[1], cnt[2], tot[2], cnt[3], tot[3]);
    }
    return out;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchSolveNative, destroyBatchContext)(
        JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<BatchSolverContext*>(handle);
}
