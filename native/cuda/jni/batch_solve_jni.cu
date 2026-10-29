// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: The batched context and its solve.
#include "jni_common.cuh"
#include "../common/codes/gpu_outer_loop_kind.cuh"
#include "../common/trace.cuh"

// OLF_GPU_PHASE_TIME: split one solveBatchContext JNI call into the three things it does —
// marshalling the Java arrays in, the solve itself, and packing the per-scenario result arrays out.
// The Java-side timer sees only the total, and on a 642-scenario chunk the total carries roughly a
// second of fixed cost that the Newton iterations do not explain.
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
        jintArray jReactiveLimitVoltageRow, jdoubleArray jReactiveLimitMinQ, jdoubleArray jReactiveLimitMaxQ, jdoubleArray jReactiveLimitLoadQ,
        jdoubleArray jReactiveLimitTargetV, jbyteArray jReactiveLimitFlags, jdoubleArray jrowSlope,
        jintArray jDistributedSlackBusPhiRow, jdoubleArray jDistributedSlackBusFactor,
        jint jSlackBusPhiRow, jdouble jSlackBusTargetP,
        jdoubleArray jDistributedSlackBusBaseTargetP, jdoubleArray jDistributedSlackBusMaxDelta, jdoubleArray jDistributedSlackBusMinDelta,
        jint jDistributedSlackFactorMode, jdoubleArray jDistributedSlackBusMarginUp, jdoubleArray jDistributedSlackBusMarginDown,
        jdouble jSlackBusMarginUp, jdouble jSlackBusMarginDown) {
    auto* structurePointer = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    jint* reactiveLimitVoltageRow = env->GetIntArrayElements(jReactiveLimitVoltageRow, nullptr);
    jdouble* reactiveLimitMinQ = env->GetDoubleArrayElements(jReactiveLimitMinQ, nullptr);
    jdouble* reactiveLimitMaxQ = env->GetDoubleArrayElements(jReactiveLimitMaxQ, nullptr);
    jdouble* reactiveLimitLoadQ = env->GetDoubleArrayElements(jReactiveLimitLoadQ, nullptr);
    jdouble* reactiveLimitTargetV = env->GetDoubleArrayElements(jReactiveLimitTargetV, nullptr);
    jbyte* reactiveLimitFlags = env->GetByteArrayElements(jReactiveLimitFlags, nullptr);
    jdouble* rowSlope = env->GetDoubleArrayElements(jrowSlope, nullptr);
    jint* distributedSlackBusPhiRow = env->GetIntArrayElements(jDistributedSlackBusPhiRow, nullptr);
    jdouble* distributedSlackBusFactor = env->GetDoubleArrayElements(jDistributedSlackBusFactor, nullptr);
    jdouble* distributedSlackBusBaseTargetP = env->GetDoubleArrayElements(jDistributedSlackBusBaseTargetP, nullptr);
    jdouble* distributedSlackBusMaxDelta = env->GetDoubleArrayElements(jDistributedSlackBusMaxDelta, nullptr);
    jdouble* distributedSlackBusMinDelta = env->GetDoubleArrayElements(jDistributedSlackBusMinDelta, nullptr);
    jdouble* distributedSlackBusMarginUp = env->GetDoubleArrayElements(jDistributedSlackBusMarginUp, nullptr);
    jdouble* distributedSlackBusMarginDown = env->GetDoubleArrayElements(jDistributedSlackBusMarginDown, nullptr);

    int nReactiveLimitEntries = env->GetArrayLength(jReactiveLimitVoltageRow);
    int nDistributedSlackBuses = env->GetArrayLength(jDistributedSlackBusPhiRow);

    std::string error;
    BatchSolverContext* ctx = nullptr;
    try {
        if (structurePointer == nullptr || !*structurePointer) {
            throw std::runtime_error("invalid structure handle");
        }
        ctx = createBatchContext(*structurePointer, nScenarios, mode,
                                 nReactiveLimitEntries, reactiveLimitVoltageRow, reactiveLimitMinQ, reactiveLimitMaxQ, reactiveLimitLoadQ, reactiveLimitTargetV, reactiveLimitFlags, rowSlope,
                                 nDistributedSlackBuses, distributedSlackBusPhiRow, distributedSlackBusFactor, jSlackBusPhiRow, jSlackBusTargetP,
                                 distributedSlackBusBaseTargetP, distributedSlackBusMaxDelta, distributedSlackBusMinDelta,
                                 jDistributedSlackFactorMode, distributedSlackBusMarginUp, distributedSlackBusMarginDown, jSlackBusMarginUp, jSlackBusMarginDown);
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    env->ReleaseIntArrayElements(jReactiveLimitVoltageRow, reactiveLimitVoltageRow, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jReactiveLimitMinQ, reactiveLimitMinQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jReactiveLimitMaxQ, reactiveLimitMaxQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jReactiveLimitLoadQ, reactiveLimitLoadQ, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jReactiveLimitTargetV, reactiveLimitTargetV, JNI_ABORT);
    env->ReleaseByteArrayElements(jReactiveLimitFlags, reactiveLimitFlags, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jrowSlope, rowSlope, JNI_ABORT);
    env->ReleaseIntArrayElements(jDistributedSlackBusPhiRow, distributedSlackBusPhiRow, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jDistributedSlackBusFactor, distributedSlackBusFactor, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jDistributedSlackBusBaseTargetP, distributedSlackBusBaseTargetP, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jDistributedSlackBusMaxDelta, distributedSlackBusMaxDelta, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jDistributedSlackBusMinDelta, distributedSlackBusMinDelta, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jDistributedSlackBusMarginUp, distributedSlackBusMarginUp, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jDistributedSlackBusMarginDown, distributedSlackBusMarginDown, JNI_ABORT);

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
        jintArray jDistributedSlackOverrideBus, jdoubleArray jDistributedSlackOverrideFactorDelta, jdoubleArray jDistributedSlackOverrideBaseTargetPDelta,
        jdoubleArray jDistributedSlackOverrideMaxDelta, jdoubleArray jDistributedSlackOverrideMinDelta, jdoubleArray jDistributedSlackBusAnchorShift,
        jdoubleArray jSlackBusAnchorShift, jintArray jReactiveLimitTargetVOverrideScenarioEntry, jdoubleArray jReactiveLimitTargetVOverrideValue,
        jintArray jReactiveLimitDeltaOverrideScenarioEntry, jdoubleArray jReactiveLimitDeltaOverrideValues,
        jintArray jReactiveCapabilityCurveOverrideScenarioCurve, jdoubleArray jReactiveCapabilityCurveOverrideStepScale, jint maxIter, jdouble tol,
        jdouble reactiveLimitTol, jdouble slackDistTol, jdouble convEpsPerEq,
        jint stateVectorScalingMode, jdouble maxDv, jdouble maxDphi, jintArray jvarType, jdoubleArray jx0Full,
        jintArray jShuntOvSlotElem, jdoubleArray jShuntOvGb,
        jintArray jReactiveLimitsSeedState, jintArray jReactiveLimitsFinalState,
        jintArray jR1OvSlotElem, jdoubleArray jR1OvVal,
        jint maxOuterProductive, jint maxPqPvSwitch, jint rejectSlackDistributionFailure, jint skipReactiveLimitsPhase,
        jint nBaseClosedZeroImpedanceBranches, jintArray jZeroImpedanceBranchCloseScenarioElement, jintArray jZeroImpedanceBranchOpenScenarioElement) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        ctx->solveCount++;
    }
    // CONTRACT: reactive-limit entries require OLF's forced-PV election keys. switchPvPq never lets
    // a network reach zero PV buses ("all PV buses should switch PQ, strongest one will stay PV"),
    // and reactiveLimitsForcedPvBatch is the only thing on the device that reproduces it. A context that
    // installed RL entries without the keys would switch every candidate to PQ and silently diverge from the
    // CPU (measured 1.9 kV on the forced-PV network). Every context with RL entries gets them when it
    // is created (DeviceLoopTables.installReactiveLimitStrongestPvKeys), so reaching this is a
    // table-installation bug — say so instead of routing the scenario away and hiding it.
    if (ctx != nullptr && ctx->nReactiveLimitEntries > 0 && !ctx->hasReactiveLimitStrongestPvKeys) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                      "solveBatchContext: reactive-limit entries installed without the forced-PV "
                      "election keys (setBatchReactiveLimitStrongestPvKeys) — OLF's strongest-bus-stays-PV rule "
                      "cannot be reproduced on the device");
        return nullptr;
    }
    long long tMarshal0 = traceOn(Trace::PHASE_TIME) ? nowNs() : 0;
    std::vector<int> closedBranchParameterOverrideScenarioElementVec;
    std::vector<double> closedBranchParameterOverrideValueVec;
    if (jR1OvSlotElem != nullptr) {
        jsize rn = env->GetArrayLength(jR1OvSlotElem);
        closedBranchParameterOverrideScenarioElementVec.resize(rn);
        env->GetIntArrayRegion(jR1OvSlotElem, 0, rn, closedBranchParameterOverrideScenarioElementVec.data());
        closedBranchParameterOverrideValueVec.resize(static_cast<size_t>(rn) * 4);      // full 8-double closedBranchParameters row per (slot, elem) entry
        env->GetDoubleArrayRegion(jR1OvVal, 0, rn * 4, closedBranchParameterOverrideValueVec.data());
    }
    std::vector<int> zeroImpedanceBranchOpenScenarioElementVec;
    if (jZeroImpedanceBranchOpenScenarioElement != nullptr) {
        jsize zn = env->GetArrayLength(jZeroImpedanceBranchOpenScenarioElement);
        zeroImpedanceBranchOpenScenarioElementVec.resize(zn);
        env->GetIntArrayRegion(jZeroImpedanceBranchOpenScenarioElement, 0, zn, zeroImpedanceBranchOpenScenarioElementVec.data());
    }
    std::vector<int> zeroImpedanceBranchCloseScenarioElementVec;
    if (jZeroImpedanceBranchCloseScenarioElement != nullptr) {
        jsize zn = env->GetArrayLength(jZeroImpedanceBranchCloseScenarioElement);
        zeroImpedanceBranchCloseScenarioElementVec.resize(zn);
        env->GetIntArrayRegion(jZeroImpedanceBranchCloseScenarioElement, 0, zn, zeroImpedanceBranchCloseScenarioElementVec.data());
    }
    std::vector<int> reactiveLimitsSeedStateVec;
    if (jReactiveLimitsSeedState != nullptr) {
        jsize rn = env->GetArrayLength(jReactiveLimitsSeedState);
        reactiveLimitsSeedStateVec.resize(rn);
        env->GetIntArrayRegion(jReactiveLimitsSeedState, 0, rn, reactiveLimitsSeedStateVec.data());
    }
    std::vector<int> reactiveLimitsFinalStateVec;
    if (jReactiveLimitsFinalState != nullptr) {
        reactiveLimitsFinalStateVec.resize(env->GetArrayLength(jReactiveLimitsFinalState));
    }
    std::vector<int> shuntAdmittanceOverrideScenarioElementVec;
    std::vector<double> shuntAdmittanceOverrideDeltaVec;
    if (jShuntOvSlotElem != nullptr) {
        jsize sn = env->GetArrayLength(jShuntOvSlotElem);
        shuntAdmittanceOverrideScenarioElementVec.resize(sn);
        env->GetIntArrayRegion(jShuntOvSlotElem, 0, sn, shuntAdmittanceOverrideScenarioElementVec.data());
        shuntAdmittanceOverrideDeltaVec.resize(sn);
        env->GetDoubleArrayRegion(jShuntOvGb, 0, sn, shuntAdmittanceOverrideDeltaVec.data());
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

        int nTargetOverrides = env->GetArrayLength(jtargetOverrideIndex);
        std::vector<int> targetOverrideIndex(nTargetOverrides);
        std::vector<double> targetOverrideValue(nTargetOverrides);
        if (nTargetOverrides > 0) {
            env->GetIntArrayRegion(jtargetOverrideIndex, 0, nTargetOverrides, targetOverrideIndex.data());
            env->GetDoubleArrayRegion(jtargetOverrideValue, 0, nTargetOverrides, targetOverrideValue.data());
        }

        // Per-scenario distributed-slack reweight: present (length nS) only when a generator-loss scenario
        // sits on a DS-participant bus; empty array → no DS override (pass nullptr to the solver).
        int nDistributedSlackOverrides = env->GetArrayLength(jDistributedSlackOverrideBus);
        std::vector<int> distributedSlackOverrideBus(nDistributedSlackOverrides);
        std::vector<double> distributedSlackOverrideFactorDelta(nDistributedSlackOverrides), distributedSlackOverrideBaseTargetPDelta(nDistributedSlackOverrides), distributedSlackOverrideMaxDelta(nDistributedSlackOverrides), distributedSlackOverrideMinDelta(nDistributedSlackOverrides);
        if (nDistributedSlackOverrides > 0) {
            env->GetIntArrayRegion(jDistributedSlackOverrideBus, 0, nDistributedSlackOverrides, distributedSlackOverrideBus.data());
            env->GetDoubleArrayRegion(jDistributedSlackOverrideFactorDelta, 0, nDistributedSlackOverrides, distributedSlackOverrideFactorDelta.data());
            env->GetDoubleArrayRegion(jDistributedSlackOverrideBaseTargetPDelta, 0, nDistributedSlackOverrides, distributedSlackOverrideBaseTargetPDelta.data());
            env->GetDoubleArrayRegion(jDistributedSlackOverrideMaxDelta, 0, nDistributedSlackOverrides, distributedSlackOverrideMaxDelta.data());
            env->GetDoubleArrayRegion(jDistributedSlackOverrideMinDelta, 0, nDistributedSlackOverrides, distributedSlackOverrideMinDelta.data());
        }
        std::vector<double> distributedSlackBusAnchorShiftVec;
        if (jDistributedSlackBusAnchorShift != nullptr) {
            distributedSlackBusAnchorShiftVec.resize(env->GetArrayLength(jDistributedSlackBusAnchorShift));
            env->GetDoubleArrayRegion(jDistributedSlackBusAnchorShift, 0, static_cast<jsize>(distributedSlackBusAnchorShiftVec.size()), distributedSlackBusAnchorShiftVec.data());
        }
        std::vector<double> slackBusAnchorShiftVec;
        if (jSlackBusAnchorShift != nullptr) {
            slackBusAnchorShiftVec.resize(env->GetArrayLength(jSlackBusAnchorShift));
            env->GetDoubleArrayRegion(jSlackBusAnchorShift, 0, static_cast<jsize>(slackBusAnchorShiftVec.size()), slackBusAnchorShiftVec.data());
        }
        std::vector<int> reactiveLimitTargetVOverrideScenarioEntryVec;
        std::vector<double> reactiveLimitTargetVOverrideValueVec;
        std::vector<int> reactiveLimitDeltaOverrideScenarioEntryVec;
        std::vector<double> reactiveLimitDeltaOverrideValuesVec;
        if (jReactiveLimitDeltaOverrideScenarioEntry != nullptr && jReactiveLimitDeltaOverrideValues != nullptr) {
            reactiveLimitDeltaOverrideScenarioEntryVec.resize(env->GetArrayLength(jReactiveLimitDeltaOverrideScenarioEntry));
            env->GetIntArrayRegion(jReactiveLimitDeltaOverrideScenarioEntry, 0, static_cast<jsize>(reactiveLimitDeltaOverrideScenarioEntryVec.size()), reactiveLimitDeltaOverrideScenarioEntryVec.data());
            reactiveLimitDeltaOverrideValuesVec.resize(env->GetArrayLength(jReactiveLimitDeltaOverrideValues));
            env->GetDoubleArrayRegion(jReactiveLimitDeltaOverrideValues, 0, static_cast<jsize>(reactiveLimitDeltaOverrideValuesVec.size()), reactiveLimitDeltaOverrideValuesVec.data());
        }
        std::vector<int> reactiveCapabilityCurveOverrideScenarioCurveVec;
        std::vector<double> reactiveCapabilityCurveOverrideStepScaleVec;
        if (jReactiveCapabilityCurveOverrideScenarioCurve != nullptr && jReactiveCapabilityCurveOverrideStepScale != nullptr) {
            reactiveCapabilityCurveOverrideScenarioCurveVec.resize(env->GetArrayLength(jReactiveCapabilityCurveOverrideScenarioCurve));
            env->GetIntArrayRegion(jReactiveCapabilityCurveOverrideScenarioCurve, 0, static_cast<jsize>(reactiveCapabilityCurveOverrideScenarioCurveVec.size()), reactiveCapabilityCurveOverrideScenarioCurveVec.data());
            reactiveCapabilityCurveOverrideStepScaleVec.resize(env->GetArrayLength(jReactiveCapabilityCurveOverrideStepScale));
            env->GetDoubleArrayRegion(jReactiveCapabilityCurveOverrideStepScale, 0, static_cast<jsize>(reactiveCapabilityCurveOverrideStepScaleVec.size()), reactiveCapabilityCurveOverrideStepScaleVec.data());
        }
        if (jReactiveLimitTargetVOverrideScenarioEntry != nullptr && jReactiveLimitTargetVOverrideValue != nullptr) {
            reactiveLimitTargetVOverrideScenarioEntryVec.resize(env->GetArrayLength(jReactiveLimitTargetVOverrideScenarioEntry));
            env->GetIntArrayRegion(jReactiveLimitTargetVOverrideScenarioEntry, 0, static_cast<jsize>(reactiveLimitTargetVOverrideScenarioEntryVec.size()), reactiveLimitTargetVOverrideScenarioEntryVec.data());
            reactiveLimitTargetVOverrideValueVec.resize(env->GetArrayLength(jReactiveLimitTargetVOverrideValue));
            env->GetDoubleArrayRegion(jReactiveLimitTargetVOverrideValue, 0, static_cast<jsize>(reactiveLimitTargetVOverrideValueVec.size()), reactiveLimitTargetVOverrideValueVec.data());
        }

        if (traceOn(Trace::PHASE_TIME)) {
            marshalNs += nowNs() - tMarshal0;
        }
        long long tSolve0 = traceOn(Trace::PHASE_TIME) ? nowNs() : 0;
        static const int kNoModeSlots[1] = {0};              // non-null: COMPACT mode even when no scenario differs
        res = solveBatchContext(ctx, x0.data(), target.data(), rowMode.data(),
                                rowModeSlots.empty() ? kNoModeSlots : rowModeSlots.data(),
                                static_cast<int>(rowModeSlots.size()), baseRowMode.data(), disabledData,
                                nTargetOverrides, targetOverrideIndex.data(), targetOverrideValue.data(),
                                nDistributedSlackOverrides > 0 ? distributedSlackOverrideBus.data() : nullptr, distributedSlackOverrideFactorDelta.data(),
                                distributedSlackOverrideBaseTargetPDelta.data(), distributedSlackOverrideMaxDelta.data(), distributedSlackOverrideMinDelta.data(),
                                distributedSlackBusAnchorShiftVec.empty() ? nullptr : distributedSlackBusAnchorShiftVec.data(),
                                slackBusAnchorShiftVec.empty() ? nullptr : slackBusAnchorShiftVec.data(),
                                static_cast<int>(reactiveLimitTargetVOverrideValueVec.size()),
                                reactiveLimitTargetVOverrideScenarioEntryVec.empty() ? nullptr : reactiveLimitTargetVOverrideScenarioEntryVec.data(),
                                reactiveLimitTargetVOverrideValueVec.empty() ? nullptr : reactiveLimitTargetVOverrideValueVec.data(),
                                static_cast<int>(reactiveLimitDeltaOverrideValuesVec.size() / 5),
                                reactiveLimitDeltaOverrideScenarioEntryVec.empty() ? nullptr : reactiveLimitDeltaOverrideScenarioEntryVec.data(),
                                reactiveLimitDeltaOverrideValuesVec.empty() ? nullptr : reactiveLimitDeltaOverrideValuesVec.data(),
                                static_cast<int>(reactiveCapabilityCurveOverrideStepScaleVec.size() / 2),
                                reactiveCapabilityCurveOverrideScenarioCurveVec.empty() ? nullptr : reactiveCapabilityCurveOverrideScenarioCurveVec.data(),
                                reactiveCapabilityCurveOverrideStepScaleVec.empty() ? nullptr : reactiveCapabilityCurveOverrideStepScaleVec.data(), maxIter, tol,
                                reactiveLimitTol, slackDistTol, convEpsPerEq,
                                stateVectorScalingMode, maxDv, maxDphi, varTypeVec.empty() ? nullptr : varTypeVec.data(),
                                x0FullVec.empty() ? nullptr : x0FullVec.data(),
                                static_cast<int>(shuntAdmittanceOverrideScenarioElementVec.size() / 2),
                                shuntAdmittanceOverrideScenarioElementVec.empty() ? nullptr : shuntAdmittanceOverrideScenarioElementVec.data(),
                                shuntAdmittanceOverrideDeltaVec.empty() ? nullptr : shuntAdmittanceOverrideDeltaVec.data(),
                                reactiveLimitsSeedStateVec.empty() ? nullptr : reactiveLimitsSeedStateVec.data(),
                                reactiveLimitsFinalStateVec.empty() ? nullptr : reactiveLimitsFinalStateVec.data(),
                                static_cast<int>(closedBranchParameterOverrideScenarioElementVec.size() / 2),
                                closedBranchParameterOverrideScenarioElementVec.empty() ? nullptr : closedBranchParameterOverrideScenarioElementVec.data(),
                                closedBranchParameterOverrideValueVec.empty() ? nullptr : closedBranchParameterOverrideValueVec.data(),
                                maxOuterProductive, maxPqPvSwitch, rejectSlackDistributionFailure, skipReactiveLimitsPhase,
                                nBaseClosedZeroImpedanceBranches, static_cast<int>(zeroImpedanceBranchCloseScenarioElementVec.size() / 2),
                                zeroImpedanceBranchCloseScenarioElementVec.empty() ? nullptr : zeroImpedanceBranchCloseScenarioElementVec.data(),
                                static_cast<int>(zeroImpedanceBranchOpenScenarioElementVec.size() / 2),
                                zeroImpedanceBranchOpenScenarioElementVec.empty() ? nullptr : zeroImpedanceBranchOpenScenarioElementVec.data());
        if (traceOn(Trace::PHASE_TIME)) {
            solveNs += nowNs() - tSolve0;
        }
        if (jReactiveLimitsFinalState != nullptr && !reactiveLimitsFinalStateVec.empty()) {
            env->SetIntArrayRegion(jReactiveLimitsFinalState, 0, static_cast<jsize>(reactiveLimitsFinalStateVec.size()), reactiveLimitsFinalStateVec.data());
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
    long long tPack0 = traceOn(Trace::PHASE_TIME) ? nowNs() : 0;
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
    if (traceOn(Trace::PHASE_TIME)) {
        packNs += nowNs() - tPack0;
        calls++;
        fprintf(stderr, "JNI_PHASE_TIME calls=%lld marshal=%lld ms solve=%lld ms pack=%lld ms "
                "| outerPasses=%lld (solving=%lld) newtonIters=%lld activeIterSum=%lld widthIterSum=%lld compactIters=%lld"
                " | cuDSS analyses main=%lld compaction=%lld | iterativeRefinementChecks=%ld iterativeRefinementExtraSolves=%ld"
                " | factorizations full=%lld compacted=%lld robust=%lld sensitivity=%lld sensitivitySolves=%lld refinement=%lld | checkRounds=%lld"
                " | batchedLu factorizations=%lld pivotedLanes=%lld pivotDelayOrders=%lld\n",
                calls, marshalNs / 1000000, solveNs / 1000000, packNs / 1000000,
                gOuterPasses, gSolvePasses, gNewtonIters, gActiveIterSum, gWidthIterSum, gCompactIters,
                gAnalysesMain, gAnalysesCompact, gIterativeRefinementChecks, gIterativeRefinementExtraSolves,
                gFullFactorizations, gCompactIters, gRobustFactorizations, gSensitivityFactorizations, gSensitivitySolves, gRefinementFactorizations, gCheckRounds,
                gBatchedLuFactorizations, gBatchedLuPivotedLanes, gBatchedLuPivotDelayOrders);
        double phaseMilliseconds[4] = {0, 0, 0, 0};
        long long phaseCount[4] = {0, 0, 0, 0};
        for (const auto& kv : gAnalysisSites) {
            int phase = kv.first / 100000, line = kv.first % 100000;
            int k = phase == 'A' ? 0 : phase == 'F' ? 1 : phase == 'S' ? 2 : 3;
            phaseMilliseconds[k] += kv.second.second;
            phaseCount[k] += kv.second.first;
            fprintf(stderr, "CUDSS_PHASE_SITE %c line=%d count=%lld ms=%.1f\n", phase, line, kv.second.first, kv.second.second);
        }
        for (int k = 1; k < 16; k++) {
            if (gOuterLoopPassHistogramCount[k] == 0) continue;
            fprintf(stderr, "GPU_OL_HIST passes=%d%s scenarios=%lld firstNrIters=%lld totalNrIters=%lld\n", k,
                    k == 15 ? "+" : "", gOuterLoopPassHistogramCount[k], gOuterLoopPassHistogramFirstSolveIterations[k], gOuterLoopPassHistogramTotalIterations[k]);
        }
        fprintf(stderr, "GPU_OL_HIST notConvergedOnDevice=%lld iters=%lld\n", gOuterLoopNotConverged, gOuterLoopNotConvergedIterations);
        fprintf(stderr, "GPU_OL_TIME first-solve passes=%lld ms=%.1f scenarioSolves=%lld laneIters=%lld widthIters=%lld\n",
                gOuterLoopFirstSolvePasses, gOuterLoopFirstSolveMilliseconds, gOuterLoopFirstSolveScenarios, gOuterLoopFirstSolveLaneIterations, gOuterLoopFirstSolveWidthIterations);
        for (int k = 0; k < GpuOuterLoopKind::COUNT; k++) {
            if (gOuterLoopReSolvePasses[k] > 0) {
                fprintf(stderr, "GPU_OL_TIME re-solve trigger=%s passes=%lld ms=%.1f scenarioSolves=%lld laneIters=%lld widthIters=%lld\n",
                        outerLoopKindName(k), gOuterLoopReSolvePasses[k], gOuterLoopReSolveMilliseconds[k], gOuterLoopReSolveScenarios[k], gOuterLoopReSolveLaneIterations[k], gOuterLoopReSolveWidthIterations[k]);
            }
        }
        for (int k = 0; k < GpuOuterLoopKind::COUNT; k++) {
            if (gOuterLoopCheckPasses[k] > 0) {
                fprintf(stderr, "GPU_OL_TIME check loop=%s passes=%lld ms=%.1f\n", outerLoopKindName(k), gOuterLoopCheckPasses[k], gOuterLoopCheckMilliseconds[k]);
            }
        }
        fprintf(stderr, "CUDSS_PHASE_TIME analysis=%lld x %.1f ms | factorization=%lld x %.1f ms | solve=%lld x %.1f ms"
                " | factorization+solve=%lld x %.1f ms\n", phaseCount[0], phaseMilliseconds[0], phaseCount[1], phaseMilliseconds[1], phaseCount[2], phaseMilliseconds[2], phaseCount[3],
                        phaseMilliseconds[3]);
    }
    return out;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_solver_gpu, BatchSolveNative, destroyBatchContext)(
        JNIEnv*, jclass, jlong handle) {
    delete reinterpret_cast<BatchSolverContext*>(handle);
}
