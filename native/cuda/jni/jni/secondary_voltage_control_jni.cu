// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: SecondaryVoltageControlOuterLoop.
#include "jni_common.cuh"

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
