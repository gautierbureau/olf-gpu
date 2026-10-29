// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: Zero-impedance couplers: per-scenario spanning-tree overrides.
#include "jni_common.cuh"
#include "slot_table_install.cuh"
#include "../common/trace.cuh"

// plan-B4 (JAVA hybrid): install ONLY the carrier-row suppression tables — distrQInjectionMember (per DISTR_Q
// contribution: row-owner member) + distrQMemberRelocationGroup (per member: pilot's relocation group) — so a
// pushed mode-4 free-member row gates the re-picked carrier's DISTR_Q row off in fillDistrQBatch.
// Does NOT touch nDistrQDynamicKeys: the hybrid's weights stay static-pack + distrQWeightOverrideTable (OLF renormalizes and the
// engine pushes live values). Nulls clear.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, ZeroImpedanceNative, setBatchSpanningTreeOverrides)(
        JNIEnv* env, jclass, jlong handle, jintArray jSlotElemVal) {
    auto* ctx = reinterpret_cast<BatchSolverContext*>(handle);
    if (ctx == nullptr || ctx->nZeroImpedanceBranch <= 0) {
        return;
    }
    // {slot, element, isTree} triples: OLF's spanning-tree flag of a coupler where it differs from extraction
    std::vector<int> triples = jniInts(env, jSlotElemVal);
    std::vector<int> pairs;
    std::vector<char> isTree;
    for (size_t k = 0; k + 2 < triples.size(); k += 3) {
        pairs.push_back(triples[k]);
        pairs.push_back(triples[k + 1]);
        isTree.push_back(triples[k + 2] != 0 ? 1 : 0);
    }
    ctx->spanningTreeOverrideMask.installEntries(ctx->nScenarios, ctx->installSlots, ctx->nZeroImpedanceBranch,
                                                 static_cast<int>(isTree.size()), pairs.data(), isTree.data());
    if (traceOn(Trace::PHASE_TIME)) {
        std::fprintf(stderr, "ZI_LISTS nSpanningTreeOverrides=%d\n", static_cast<int>(isTree.size()));
    }
}
