// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Zero-impedance couplers: per-scenario spanning-tree overrides.
#include "jni_common.cuh"

// plan-B4 (JAVA hybrid): install ONLY the carrier-row suppression tables — distrQInjectionMember (per DISTR_Q
// contribution: row-owner member) + distrQMemberRelocationGroup (per member: pilot's relocation group) — so a
// pushed mode-4 free-member row gates the re-picked carrier's DISTR_Q row off in fillDistrQBatch.
// Does NOT touch nDistrQDynamicKeys: the hybrid's weights stay static-pack + distrQWeightOverrideTable (OLF renormalizes and the
// engine pushes live values). Nulls clear.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, ZeroImpedanceNative, setBatchSpanningTreeOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotElemVal) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr) {
        return;
    }
    int len = jSlotElemVal == nullptr ? 0 : env->GetArrayLength(jSlotElemVal);
    ctx->nSpanningTreeOverrides = len / 3;
    if (ctx->nSpanningTreeOverrides > 0) {
        jint* p = env->GetIntArrayElements(jSlotElemVal, nullptr);
        ctx->dSpanningTreeOverrideScenarioElementValue.upload(p, static_cast<size_t>(len) * sizeof(int));
        env->ReleaseIntArrayElements(jSlotElemVal, p, JNI_ABORT);
    }
    // Built whenever the masks are enabled, NOT only once the off/on setter has run: the two setters' order is
    // the caller's, and gating this one on hasZeroImpedanceBranchMasks dropped every tree override of a chunk whose tree list
    // arrived first - the extraction-time tree then stood in for OLF's, and a coupler child like LLJ_2 on
    // network.xiidm lost its ZERO_V coupling in 2,308 scenarios (V 0.787, FAILED as unrealistic).
    if (ctx->nZeroImpedanceBranch > 0) {                                      // the dense form (see dSpanningTreeOverrideMask)
        ctx->hasZeroImpedanceBranchMasks = true;
        size_t mlen = static_cast<size_t>(ctx->nScenarios) * ctx->nZeroImpedanceBranch;
        std::vector<char> tree(mlen, (char) -1);
        if (len > 0) {
            jint* p = env->GetIntArrayElements(jSlotElemVal, nullptr);
            for (int k = 0; k < len / 3; k++) {
                int sl = p[3 * k], e = p[3 * k + 1];
                if (sl >= 0 && sl < ctx->nScenarios && e >= 0 && e < ctx->nZeroImpedanceBranch) tree[(size_t) sl * ctx->nZeroImpedanceBranch + e] = (char) (p[3 * k + 2] != 0 ? 1 : 0);
            }
            env->ReleaseIntArrayElements(jSlotElemVal, p, JNI_ABORT);
        }
        ctx->dSpanningTreeOverrideMask.upload(tree.data(), mlen);
        if (std::getenv("OLF_GPU_PHASE_TIME") != nullptr) {
            std::fprintf(stderr, "ZI_LISTS nSpanningTreeOverrides=%d\n", len / 3);
        }
    }
}
