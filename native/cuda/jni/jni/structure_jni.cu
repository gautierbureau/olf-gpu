// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: GpuAcStructures: the shared structure built once per extraction, its values refresh and the chunk sizing.
#include "jni_common.cuh"

extern "C" JNIEXPORT jlong JNICALL
OLF_JNI(ac_equations_vector_gpu, StructureNative, createStructure)(
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
OLF_JNI(ac_equations_vector_gpu, StructureNative, destroyStructure)(
        JNIEnv*, jclass, jlong handle) {
    // Drops the Java-side owner reference only: contexts hold their own shared_ptr copies,
    // so the device-resident structure lives until the last context is destroyed.
    delete reinterpret_cast<SharedStructurePtr*>(handle);
}

// VALUES refresh addressed by the SHARED STRUCTURE handle (no solver context needed): the
// sequential batch-of-1 path reconciles tap/shunt moves the same way the seq context did —
// one extraction has ONE value set, and every context sharing the structure sees the update.
extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, StructureNative, refreshStructureValues)(
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
OLF_JNI(ac_equations_vector_gpu, StructureNative, recommendBatchSize)(
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
