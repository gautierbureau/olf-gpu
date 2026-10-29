// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Remote generator voltage control (DISTR_Q) and its pilot: per-scenario members, keys and targets.
#include "jni_common.cuh"
#include "slot_table_install.cuh"

// {slot, member} pairs: the DISTR_Q members that stop controlling in that scenario (their bus lost its last voltage
// generator while the group's other members survive). Nulls empty the installed slots.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrQInactiveMembers)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotMember) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installSlotTable(env, ctx, ctx->distrQInactiveMembers, jSlotMember, nullptr);
    }
}

// Per (installed scenario, DISTR_Q member): the subgroup label of a group the scenario splits (0: none), and the
// relocation groups' first member, run-constant, sent along. Null: no installed scenario splits a group.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrQSubgroups)(
        JNIEnv* env, jclass, jlong handle, jbyteArray jDense, jintArray jReactiveRelocationDistrQMemberStart) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    if (jDense != nullptr && env->GetArrayLength(jDense) == 0) {
        jDense = nullptr;
    }
    installSlotRows(env, ctx, ctx->distrQSubgroup, ctx->nDistrQMembers, jDense, "setBatchDistrQSubgroups");
    if (jDense != nullptr && !env->ExceptionCheck()) {
        std::vector<int> memberStart = jniInts(env, jReactiveRelocationDistrQMemberStart);
        ctx->dReactiveRelocationDistrQMemberStart.upload(memberStart.data(), memberStart.size() * sizeof(int));
    }
}

// DISTR_Q weight overrides: the per-scenario qPct renormalization, {slot, contribution} pairs + the new weights (a gen
// outage at a remote-VC controller bus). Nulls empty the installed slots.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrQWeightOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jScenarioTerm, jdoubleArray jWeight) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installSlotTable(env, ctx, ctx->distrQWeightOverrides, jScenarioTerm, jWeight);
    }
}

// Per-scenario member reactive keys: {slot, member} pairs + OLF's key after the scenario's generator losses. Nulls
// empty the installed slots.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrQKeyOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jScenarioMember, jdoubleArray jKey) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installSlotTable(env, ctx, ctx->distrQKeyOverrides, jScenarioMember, jKey);
    }
}

// Per-scenario member targetQ overrides for the DISTR_Q target kernel: {key, value} where key is
// slot * nDistrQMembers + memberIndex. Set when an outaged bus is a packed group member whose own targetQ
// moved (a reactive LOAD loss). Nulls empty the installed slots.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrQMemberTargetQOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installKeyedSlotTable(env, ctx, ctx->distrQMemberTargetQOverrides, ctx->nDistrQMembers, jKeys, jVals);
    }
}

// Per-scenario moves of a pilot's captured reactive target (applyPilotQDeltasBatch): {key, delta} with key
// slot * nReactiveRelocationGroups + group, one entry per key. Nulls empty the installed slots.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchPilotQDeltas)(
        JNIEnv* env, jclass, jlong handle, jintArray jKeys, jdoubleArray jVals) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installKeyedSlotTable(env, ctx, ctx->reactiveRelocationPilotTargetQDeltas, ctx->structure->nReactiveRelocationGroups, jKeys, jVals);
    }
}

// Pilot-row flip entries (see pilotRowWithoutEnabledControllerBatch): one per (scenario, pilot) where the outage
// made the pilot stop being one of its own controllers. slots/rows/qTargets are per entry; survivorReactiveLimitEntryCounts
// says how many surviving-controller RL entries each one owns and survivorReactiveLimitEntries concatenates them in order.
// Nulls empty the installed slots, whose pilot rows start unflipped.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchPilotNoEnabledControllerFlip)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlots, jintArray jRows, jdoubleArray jQTargets,
        jintArray jSurvivorReactiveLimitEntryCounts, jintArray jSurvivorReactiveLimitEntries) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    std::vector<int> slots = jniInts(env, jSlots);
    std::vector<double> qTargets = jniDoubles(env, jQTargets);
    SlotRuns runs = slotRuns(slots, {jniInts(env, jRows)}, jniInts(env, jSurvivorReactiveLimitEntryCounts),
                             jniInts(env, jSurvivorReactiveLimitEntries));
    ctx->pilotFlips.install(ctx->nScenarios, ctx->installSlots, static_cast<int>(slots.size()), runs.entryInts.data(), qTargets.data());
    ctx->pilotFlipSurvivors.install(ctx->nScenarios, ctx->installSlots, static_cast<int>(runs.itemInts.size() / 2), runs.itemInts.data(), nullptr);
}

// Re-merged controls whose elected main is one of its controllers (see updateReMergedVoltageTargetRowBatch): per control
// its slot, the main's reactive-limit entry and the control's voltage target; controllerCounts says how many of the
// cluster's other controllers each one has and controllerEntries concatenates their entries in OLF's merged order.
// Nulls empty the installed slots. The holder state is per solve (launchReMergedVoltageTargetRowUpdate resets it).
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchReMergedVoltageTargets)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlots, jintArray jMainEntries, jdoubleArray jVoltageTargets,
        jintArray jControllerCounts, jintArray jControllerEntries) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    std::vector<int> slots = ctx->nReactiveLimitEntries > 0 ? jniInts(env, jSlots) : std::vector<int>();
    std::vector<double> voltageTargets = jniDoubles(env, jVoltageTargets);
    SlotRuns runs = slotRuns(slots, {jniInts(env, jMainEntries)}, jniInts(env, jControllerCounts), jniInts(env, jControllerEntries));
    ctx->reMergedVoltageTargets.install(ctx->nScenarios, ctx->installSlots, static_cast<int>(slots.size()), runs.entryInts.data(),
                                        voltageTargets.data());
    ctx->reMergedVoltageTargetControllers.install(ctx->nScenarios, ctx->installSlots, static_cast<int>(runs.itemInts.size() / 2),
                                                  runs.itemInts.data(), nullptr);
    size_t scenarioEntries = static_cast<size_t>(ctx->nScenarios) * ctx->nReactiveLimitEntries;
    if (ctx->reMergedVoltageTargets.entries() > 0 && ctx->dReactiveLimitReMergedVoltageRow.cap < scenarioEntries * sizeof(int)) {
        // the bank's first re-merge: nothing held anywhere (a fresh allocation is zeroed, the rows are -1); the solve's
        // start resets its installed slots' rows (launchReMergedVoltageTargetRowUpdate), never a slot still solving
        ctx->dReactiveLimitHoldsReMergedVoltageTarget.ensure(scenarioEntries);
        ctx->dReactiveLimitReMergedVoltageRow.ensure(scenarioEntries * sizeof(int));
        CUDA_CHECK(cudaMemset(ctx->dReactiveLimitReMergedVoltageRow.p, 0xFF, scenarioEntries * sizeof(int)));
    }
}

// DISTR_RHO coefficient overrides: the per-scenario 1/count' renormalization, {slot, contribution} pairs + the new
// coefficients (a shared-TVC group member outaged in that scenario — plan-group-member-dynamics A5). Nulls empty the
// installed slots.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, RemoteVoltageControlNative, setBatchDistrRhoCoefficientOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jScenarioTerm, jdoubleArray jCoefficient) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx != nullptr) {
        installSlotTable(env, ctx, ctx->distrRhoCoefficientOverrides, jScenarioTerm, jCoefficient);
    }
}
