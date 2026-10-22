// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Remote generator voltage control (DISTR_Q) and its pilot: per-scenario members, keys and targets.
#include "jni_common.cuh"

// DISTR_Q weight overrides: install (or clear, with nulls) the per-scenario qPct renormalization
// pairs {s, contrib} + new weights on a batch context (gen outage at a remote-VC controller bus).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrQInactiveMembers)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotMember) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlotMember == nullptr || env->GetArrayLength(jSlotMember) == 0) {
        ctx->nDistrQInactiveMembers = 0;
        return;
    }
    int n2 = env->GetArrayLength(jSlotMember);
    jint* sm = env->GetIntArrayElements(jSlotMember, nullptr);
    ctx->dDistrQInactiveMembers.upload(sm, static_cast<size_t>(n2) * sizeof(int));
    env->ReleaseIntArrayElements(jSlotMember, sm, JNI_ABORT);
    ctx->nDistrQInactiveMembers = n2 / 2;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrQSubgroups)(
        JNIEnv* env, jclass, jlong handle, jbyteArray jDense, jintArray jReactiveRelocationDistrQMemberStart) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jDense == nullptr || env->GetArrayLength(jDense) == 0) {
        ctx->hasDistrQSubgroups = false;
        return;
    }
    jbyte* d = env->GetByteArrayElements(jDense, nullptr);
    ctx->dDistrQSubgroup.upload(d, static_cast<size_t>(env->GetArrayLength(jDense)));
    env->ReleaseByteArrayElements(jDense, d, JNI_ABORT);
    jint* q = env->GetIntArrayElements(jReactiveRelocationDistrQMemberStart, nullptr);
    ctx->dReactiveRelocationDistrQMemberStart.upload(q, static_cast<size_t>(env->GetArrayLength(jReactiveRelocationDistrQMemberStart)) * sizeof(int));
    env->ReleaseIntArrayElements(jReactiveRelocationDistrQMemberStart, q, JNI_ABORT);
    ctx->hasDistrQSubgroups = true;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrQWeightOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jScenarioTerm, jdoubleArray jWeight) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jScenarioTerm == nullptr || jWeight == nullptr || env->GetArrayLength(jWeight) == 0) {
        ctx->nDistrQWeightOverrides = 0;
        return;
    }
    int nDistrQWeightOverrides = env->GetArrayLength(jWeight);
    jint* scenarioTerm = env->GetIntArrayElements(jScenarioTerm, nullptr);
    jdouble* weight = env->GetDoubleArrayElements(jWeight, nullptr);
    ctx->dDistrQWeightOverrideScenarioTerm.upload(scenarioTerm, static_cast<size_t>(nDistrQWeightOverrides) * 2 * sizeof(int));
    ctx->dDistrQWeightOverrideWeight.upload(weight, static_cast<size_t>(nDistrQWeightOverrides) * sizeof(double));
    env->ReleaseIntArrayElements(jScenarioTerm, scenarioTerm, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jWeight, weight, JNI_ABORT);
    ctx->nDistrQWeightOverrides = nDistrQWeightOverrides;
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrQKeyOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jScenarioMember, jdoubleArray jKey) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jScenarioMember == nullptr || jKey == nullptr || env->GetArrayLength(jKey) == 0) {
        ctx->nDistrQKeyOverrides = 0;
        return;
    }
    int nDistrQKeyOverrides = env->GetArrayLength(jKey);
    jint* scenarioMember = env->GetIntArrayElements(jScenarioMember, nullptr);
    jdouble* key = env->GetDoubleArrayElements(jKey, nullptr);
    ctx->dDistrQKeyOverrideScenarioMember.upload(scenarioMember, static_cast<size_t>(nDistrQKeyOverrides) * 2 * sizeof(int));
    ctx->dDistrQKeyOverrideKey.upload(key, static_cast<size_t>(nDistrQKeyOverrides) * sizeof(double));
    env->ReleaseIntArrayElements(jScenarioMember, scenarioMember, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jKey, key, JNI_ABORT);
    ctx->nDistrQKeyOverrides = nDistrQKeyOverrides;
}

// Per-scenario member targetQ overrides for the DISTR_Q target kernel: {key, value} where key is
// slot * nDistrQMembers + memberIndex. Set when an outaged bus is a packed group member whose own targetQ
// moved (a reactive LOAD loss). Nulls clear.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrQMemberTargetQOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKeys == nullptr || env->GetArrayLength(jKeys) == 0) {
        ctx->nDistrQMemberTargetQOverrides = 0;
        return;
    }
    int ne = env->GetArrayLength(jKeys);
    jint* keys = env->GetIntArrayElements(jKeys, nullptr);
    jdouble* vals = env->GetDoubleArrayElements(jVals, nullptr);
    ctx->dDistrQMemberTargetQOverrideKey.upload(keys, static_cast<size_t>(ne) * sizeof(int));
    ctx->dDistrQMemberTargetQOverrideValue.upload(vals, static_cast<size_t>(ne) * sizeof(double));
    ctx->nDistrQMemberTargetQOverrides = ne;
    env->ReleaseIntArrayElements(jKeys, keys, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVals, vals, JNI_ABORT);
}

// Per-scenario moves of a pilot's captured reactive target (applyPilotQDeltasBatch): {key, delta} with key
// slot * nReactiveRelocationGroups + group, one entry per key. Nulls clear.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchPilotQDeltas)(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jKeys == nullptr || env->GetArrayLength(jKeys) == 0) {
        ctx->nReactiveRelocationPilotTargetQDeltas = 0;
        return;
    }
    int ne = env->GetArrayLength(jKeys);
    jint* keys = env->GetIntArrayElements(jKeys, nullptr);
    jdouble* vals = env->GetDoubleArrayElements(jVals, nullptr);
    ctx->dReactiveRelocationPilotTargetQDeltaKey.upload(keys, static_cast<size_t>(ne) * sizeof(int));
    ctx->dReactiveRelocationPilotTargetQDeltaValue.upload(vals, static_cast<size_t>(ne) * sizeof(double));
    ctx->nReactiveRelocationPilotTargetQDeltas = ne;
    env->ReleaseIntArrayElements(jKeys, keys, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVals, vals, JNI_ABORT);
}

// Pilot-row flip entries (see pilotRowAllPinnedBatch): one per (scenario, pilot) where the outage
// made the pilot stop being one of its own controllers. slots/rows/qTargets are per entry; survivorReactiveLimitEntryCounts
// says how many surviving-controller RL entries each one owns and survivorReactiveLimitEntries concatenates them in order.
// Nulls clear. Flip state (flipped/savedVTarget) is allocated zeroed here, so each solve starts with
// every pilot row unflipped.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchPilotAllPinnedFlip)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlots, jintArray jRows, jdoubleArray jQTargets,
        jintArray jSurvivorReactiveLimitEntryCounts, jintArray jSurvivorReactiveLimitEntries) {
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
    jint* survivorReactiveLimitEntryCounts = env->GetIntArrayElements(jSurvivorReactiveLimitEntryCounts, nullptr);
    jint* pool = env->GetIntArrayElements(jSurvivorReactiveLimitEntries, nullptr);
    int poolLen = env->GetArrayLength(jSurvivorReactiveLimitEntries);
    std::vector<int> off(ne);
    int acc = 0;
    for (int i = 0; i < ne; i++) {
        off[i] = acc;
        acc += survivorReactiveLimitEntryCounts[i];
    }
    ctx->dPfSlot.upload(slots, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfRow.upload(rows, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfQTarget.upload(qt, static_cast<size_t>(ne) * sizeof(double));
    ctx->dPfCount.upload(survivorReactiveLimitEntryCounts, static_cast<size_t>(ne) * sizeof(int));
    ctx->dPfOff.upload(off.data(), static_cast<size_t>(ne) * sizeof(int));
    ctx->dPilotFlipSurvivorReactiveLimitEntries.upload(pool, static_cast<size_t>(poolLen > 0 ? poolLen : 1) * sizeof(int));
    std::vector<char> zeroC(ne, 0);
    std::vector<double> zeroD(ne, 0.0);
    ctx->dPfFlipped.upload(zeroC.data(), static_cast<size_t>(ne) * sizeof(char));
    ctx->dPfSavedVT.upload(zeroD.data(), static_cast<size_t>(ne) * sizeof(double));
    ctx->nPilotFlip = ne;
    env->ReleaseIntArrayElements(jSlots, slots, JNI_ABORT);
    env->ReleaseIntArrayElements(jRows, rows, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jQTargets, qt, JNI_ABORT);
    env->ReleaseIntArrayElements(jSurvivorReactiveLimitEntryCounts, survivorReactiveLimitEntryCounts, JNI_ABORT);
    env->ReleaseIntArrayElements(jSurvivorReactiveLimitEntries, pool, JNI_ABORT);
}

// Re-merged controls whose elected main is one of its controllers (see updateReMergedVoltageTargetRowBatch): per control
// its slot, the main's reactive-limit entry and the control's voltage target; controllerCounts says how many of the
// cluster's other controllers each one has and controllerEntries concatenates their entries in OLF's merged order.
// Nulls clear. The holder state is per solve (launchReMergedVoltageTargetRowUpdate resets it).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchReMergedVoltageTargets)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlots, jintArray jMainEntries, jdoubleArray jVoltageTargets,
        jintArray jControllerCounts, jintArray jControllerEntries) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jSlots == nullptr || env->GetArrayLength(jSlots) == 0 || ctx->nReactiveLimitEntries == 0) {
        ctx->nReMergedVoltageTargets = 0;
        return;
    }
    int ne = env->GetArrayLength(jSlots);
    jint* slots = env->GetIntArrayElements(jSlots, nullptr);
    jint* mainEntries = env->GetIntArrayElements(jMainEntries, nullptr);
    jdouble* voltageTargets = env->GetDoubleArrayElements(jVoltageTargets, nullptr);
    jint* controllerCounts = env->GetIntArrayElements(jControllerCounts, nullptr);
    jint* controllerEntries = env->GetIntArrayElements(jControllerEntries, nullptr);
    int nControllerEntries = env->GetArrayLength(jControllerEntries);
    std::vector<int> start(ne);
    int acc = 0;
    for (int i = 0; i < ne; i++) {
        start[i] = acc;
        acc += controllerCounts[i];
    }
    ctx->dReMergedVoltageTargetSlot.upload(slots, static_cast<size_t>(ne) * sizeof(int));
    ctx->dReMergedVoltageTargetMainEntry.upload(mainEntries, static_cast<size_t>(ne) * sizeof(int));
    ctx->dReMergedVoltageTargetValue.upload(voltageTargets, static_cast<size_t>(ne) * sizeof(double));
    ctx->dReMergedVoltageTargetControllerStart.upload(start.data(), static_cast<size_t>(ne) * sizeof(int));
    ctx->dReMergedVoltageTargetControllerCount.upload(controllerCounts, static_cast<size_t>(ne) * sizeof(int));
    ctx->dReMergedVoltageTargetControllerEntries.upload(controllerEntries, static_cast<size_t>(nControllerEntries > 0 ? nControllerEntries : 1) * sizeof(int));
    std::vector<int> noHolder(ne, -1);
    ctx->dReMergedVoltageTargetHolder.upload(noHolder.data(), static_cast<size_t>(ne) * sizeof(int));
    size_t scenarioEntries = static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries;
    std::vector<char> holds(scenarioEntries, 0);
    std::vector<int> voltageRow(scenarioEntries, -1);
    ctx->dReactiveLimitHoldsReMergedVoltageTarget.upload(holds.data(), scenarioEntries);
    ctx->dReactiveLimitReMergedVoltageRow.upload(voltageRow.data(), scenarioEntries * sizeof(int));
    ctx->nReMergedVoltageTargets = ne;
    env->ReleaseIntArrayElements(jSlots, slots, JNI_ABORT);
    env->ReleaseIntArrayElements(jMainEntries, mainEntries, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jVoltageTargets, voltageTargets, JNI_ABORT);
    env->ReleaseIntArrayElements(jControllerCounts, controllerCounts, JNI_ABORT);
    env->ReleaseIntArrayElements(jControllerEntries, controllerEntries, JNI_ABORT);
}

// DISTR_RHO coefficient overrides: install (or clear, with nulls) the per-scenario 1/count'
// renormalization pairs {s, contrib} + new coefficients on a batch context (a shared-TVC group
// member outaged in that scenario — plan-group-member-dynamics A5).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrRhoCoefficientOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jScenarioTerm, jdoubleArray jCoefficient) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jScenarioTerm == nullptr || jCoefficient == nullptr || env->GetArrayLength(jCoefficient) == 0) {
        ctx->nDistrRhoCoefficientOverrides = 0;
        return;
    }
    int nDistrRhoCoefficientOverrides = env->GetArrayLength(jCoefficient);
    jint* scenarioTerm = env->GetIntArrayElements(jScenarioTerm, nullptr);
    jdouble* coefficient = env->GetDoubleArrayElements(jCoefficient, nullptr);
    ctx->dDistrRhoOverrideScenarioTerm.upload(scenarioTerm, static_cast<size_t>(nDistrRhoCoefficientOverrides) * 2 * sizeof(int));
    ctx->dDistrRhoOverrideCoefficient.upload(coefficient, static_cast<size_t>(nDistrRhoCoefficientOverrides) * sizeof(double));
    env->ReleaseIntArrayElements(jScenarioTerm, scenarioTerm, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jCoefficient, coefficient, JNI_ABORT);
    ctx->nDistrRhoCoefficientOverrides = nDistrRhoCoefficientOverrides;
}
