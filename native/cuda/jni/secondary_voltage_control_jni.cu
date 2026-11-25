// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: SecondaryVoltageControlOuterLoop.
#include "jni_common.cuh"
#include "../solver/scenario_install.cuh"

/** Install the SecondaryVoltageControl zones for the in-kernel phase (OLF order 4). Controllers are
 *  addressed by their controlled bus's V row (the sensitivity's unit row and where a moved target lands)
 *  and their own bus V row (the reactive injection the K comes from); each carries a CSR slice of Q
 *  observables (a GpuReactiveTermKind: a closed branch's reactive flow on a GpuTwoSides side, or a shunt) and the zones a
 *  CSR slice of controllers. Returns 1 when the kernel loop owns them — it refuses a zone set wider than
 *  the in-kernel dense system, or an observable on an element the batch does not pack. */
extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_outerloop_gpu, SecondaryVoltageControlNative, setBatchSecondaryVoltageControl)(
        JNIEnv* env, jclass, jlong handle,
        jintArray jControlledBusVRow, jintArray jControllerBusVRow, jintArray jSecondaryVoltageControlReactiveLimitEntry, jdoubleArray jBaseTargetV,
        jdoubleArray jControlledBusNominalV,
        jdoubleArray jMinQ, jdoubleArray jMaxQ, jdoubleArray jLoadQ,
        jintArray jReactiveTermStart, jintArray jReactiveTermKind, jintArray jReactiveTermSide, jintArray jReactiveTermElement, jintArray jReactiveTermShuntBRow,
        jintArray jZoneStart, jintArray jZoneControllerBus, jintArray jZonePilotBusVRow, jdoubleArray jZoneTargetV,
        jdouble minPlausibleV, jdouble maxPlausibleV) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return 0;
    }
    ctx->secondaryVoltageControl.nSecondaryVoltageControlControllerBuses = 0;
    ctx->secondaryVoltageControl.nSecondaryVoltageControlZones = 0;
    // A controller's reactive-limit entry is -1 in a run without OLF's ReactiveLimitsOuterLoop: nothing switches it to PQ, and the
    // phase moves its target on the controlled bus's BUS_TARGET_V row alone.
    if (jControlledBusVRow == nullptr || env->GetArrayLength(jControlledBusVRow) == 0) {
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
    std::vector<int> busVRow, reactiveLimitEntry, qPtr, reactiveTermKind, reactiveTermSide, reactiveTermElement, reactiveTermShuntBRow, zoneStart, zoneControllerBus, zonePilotBusVRow;
    std::vector<double> minQ, maxQ, loadQ, zoneTargetV, controlledBusNominalV;
    readI(jControlledBusVRow, ctx->secondaryVoltageControl.hSecondaryVoltageControlControlledBusVRow);
    readI(jControllerBusVRow, busVRow);
    readI(jSecondaryVoltageControlReactiveLimitEntry, reactiveLimitEntry);
    readD(jBaseTargetV, ctx->secondaryVoltageControl.hSecondaryVoltageControlBaseTargetV);
    readD(jControlledBusNominalV, controlledBusNominalV);
    readD(jMinQ, minQ);
    readD(jMaxQ, maxQ);
    readD(jLoadQ, loadQ);
    readI(jReactiveTermStart, qPtr);
    readI(jReactiveTermKind, reactiveTermKind);
    readI(jReactiveTermSide, reactiveTermSide);
    readI(jReactiveTermElement, reactiveTermElement);
    readI(jReactiveTermShuntBRow, reactiveTermShuntBRow);
    readI(jZoneStart, zoneStart);
    readI(jZoneControllerBus, zoneControllerBus);
    readI(jZonePilotBusVRow, zonePilotBusVRow);
    readD(jZoneTargetV, zoneTargetV);
    int nC = static_cast<int>(ctx->secondaryVoltageControl.hSecondaryVoltageControlControlledBusVRow.size());
    int nZ = static_cast<int>(zoneStart.size()) - 1;
    if (nZ <= 0) {
        return 0;
    }
    // OLF solves ONE system over every enabled controller of every zone, so the size is the total
    // controller count. The batched LU holds an N*N matrix per scenario, which is what actually bounds
    // this: refuse when that workspace would not fit rather than at an arbitrary controller count.
    double matBytes = static_cast<double>(ctx->nScenarios) * nC * nC * sizeof(double);
    if (matBytes > SECONDARY_VOLTAGE_CONTROL_MATRIX_BUDGET_BYTES) {
        std::fprintf(stderr, "SVC in-kernel declined: %d controllers x %d scenarios needs %.1f MB of zone-system "
                     "workspace (budget %.0f MB)\n", nC, ctx->nScenarios, matBytes / 1048576.0,
                     SECONDARY_VOLTAGE_CONTROL_MATRIX_BUDGET_BYTES / 1048576.0);
        return 0;
    }
    for (int e : reactiveTermElement) {
        if (e < 0) {
            return 0;                                        // an observable the batch does not pack
        }
    }
    ctx->secondaryVoltageControl.dSecondaryVoltageControlControlledBusVRow.upload(ctx->secondaryVoltageControl.hSecondaryVoltageControlControlledBusVRow.data(), nC * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlControllerBusVRow.upload(busVRow.data(), nC * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveLimitEntry.upload(reactiveLimitEntry.data(), nC * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlBaseTargetV.upload(ctx->secondaryVoltageControl.hSecondaryVoltageControlBaseTargetV.data(), nC * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlControlledBusNominalV.upload(controlledBusNominalV.data(), nC * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlMinQ.upload(minQ.data(), nC * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlMaxQ.upload(maxQ.data(), nC * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlLoadTargetQ.upload(loadQ.data(), nC * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveTermStart.upload(qPtr.data(), qPtr.size() * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveTermKind.upload(reactiveTermKind.data(), std::max<size_t>(reactiveTermKind.size(), 1) * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveTermSide.upload(reactiveTermSide.data(), std::max<size_t>(reactiveTermSide.size(), 1) * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveTermElement.upload(reactiveTermElement.data(), std::max<size_t>(reactiveTermElement.size(), 1) * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveTermShuntBRow.upload(reactiveTermShuntBRow.data(), std::max<size_t>(reactiveTermShuntBRow.size(), 1) * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneStart.upload(zoneStart.data(), zoneStart.size() * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneControllerBus.upload(zoneControllerBus.data(), zoneControllerBus.size() * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZonePilotBusVRow.upload(zonePilotBusVRow.data(), nZ * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneTargetV.upload(zoneTargetV.data(), nZ * sizeof(double));
    size_t nS = ctx->nScenarios;
    ctx->secondaryVoltageControl.dSecondaryVoltageControlTargetV.ensure(nS * nC * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlK0.ensure(nS * nC * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlJk.ensure(nS * nC * nC * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlJvpp.ensure(static_cast<size_t>(nS) * static_cast<size_t>(nZ) * nC * sizeof(double));   // per (zone pilot, controlled bus)
    ctx->secondaryVoltageControl.dSecondaryVoltageControlControllerBusEnabled.ensure(nS * nC * sizeof(char));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlNeedsAdjustment.ensure(nS * sizeof(char));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlRejected.ensure(nS * sizeof(char));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneLastPilotDv.ensure(nS * nZ * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneLastDk.ensure(nS * nZ * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneNoProgress.ensure(nS * nZ * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneStalled.ensure(nS * nZ * sizeof(char));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveLimitTargetV.ensure(nS * std::max(ctx->reactiveLimits.nReactiveLimitEntries, 1) * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneMatrix.ensure(nS * static_cast<size_t>(nC) * nC * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneRightHandSide.ensure(nS * nC * sizeof(double));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneMatrixPointers.ensure(nS * sizeof(double*));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneRightHandSidePointers.ensure(nS * sizeof(double*));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZonePivots.ensure(nS * nC * sizeof(int));
    ctx->secondaryVoltageControl.dSecondaryVoltageControlZoneFactorizationInfo.ensure(nS * sizeof(int));
    {
        std::vector<int> zoneOf(nC, 0);                      // controller -> its zone, for the assembly
        for (int z = 0; z < nZ; z++) {
            for (int k = zoneStart[z]; k < zoneStart[z + 1]; k++) {
                zoneOf[zoneControllerBus[k]] = z;
            }
        }
        ctx->secondaryVoltageControl.dSecondaryVoltageControlControllerBusZone.upload(zoneOf.data(), zoneOf.size() * sizeof(int));
    }
    {
        // NaN = "this zone has not moved the controller's target", which is what reactiveLimitTargetVOf tests. The
        // reactive-limit SEEDING reads this buffer before the per-solve reset further down runs, so an
        // uninitialised one hands the switch-back a garbage comparison voltage and switches back to PV controllers
        // that should stay PQ (measured on ieee14's z1: controllers switched to PQ 4 -> 2 instead of 4 -> 6, and 1.0e-10 kV).
        std::vector<double> nanv(nS * std::max(ctx->reactiveLimits.nReactiveLimitEntries, 1), std::nan(""));
        CUDA_CHECK(cudaMemcpy(ctx->secondaryVoltageControl.dSecondaryVoltageControlReactiveLimitTargetV.p, nanv.data(), nanv.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    ctx->secondaryVoltageControl.secondaryVoltageControlMinPlausibleTargetV = minPlausibleV;
    ctx->secondaryVoltageControl.secondaryVoltageControlMaxPlausibleTargetV = maxPlausibleV;
    ctx->secondaryVoltageControl.nSecondaryVoltageControlZones = nZ;
    ctx->secondaryVoltageControl.nSecondaryVoltageControlControllerBuses = nC;
    return 1;
}

// The zone-regulated generator voltage control TARGETS the SVC phase settled on, per
// (scenario, controller) — what OLF's SecondaryVoltageControlOuterLoop leaves on the network. The
// batched SA restores per scenario and needs none of this; the base load flow has to give them back.
extern "C" JNIEXPORT jdoubleArray JNICALL
OLF_JNI(ac_outerloop_gpu, SecondaryVoltageControlNative, readBatchSecondaryVoltageControlTargets)(
        JNIEnv* env, jclass, jlong handle) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->secondaryVoltageControl.nSecondaryVoltageControlControllerBuses <= 0 || !ctx->secondaryVoltageControl.dSecondaryVoltageControlTargetV.p) {
        return nullptr;
    }
    std::vector<double> host;
    readInstalledRows(ctx, ctx->secondaryVoltageControl.dSecondaryVoltageControlTargetV.p, static_cast<size_t>(ctx->secondaryVoltageControl.nSecondaryVoltageControlControllerBuses), host);
    jdoubleArray out = env->NewDoubleArray(static_cast<jsize>(host.size()));
    env->SetDoubleArrayRegion(out, 0, static_cast<jsize>(host.size()), host.data());
    return out;
}
