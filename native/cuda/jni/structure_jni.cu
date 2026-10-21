// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: GpuAcStructures: the shared structure built once per extraction and the chunk sizing.
#include "jni_common.cuh"

extern "C" JNIEXPORT jlong JNICALL
OLF_JNI(ac_equations_vector_gpu, StructureNative, createStructure)(
        JNIEnv* env, jclass, jint n,
        jdoubleArray jClosedBranchParameters, jintArray jClosedBranchIndices, jdoubleArray jOpenBranchParameters, jintArray jOpenBranchIndices,
        jdoubleArray jShuntParameters, jintArray jShuntIndices, jdoubleArray jHvdcAcEmulationParameters, jintArray jHvdcAcEmulationIndices,
        jintArray jDistrQElement, jintArray jDistrQSide, jdoubleArray jDistrQWeight, jintArray jDistrQRow, jintArray jDistrQKind,
        jdoubleArray jZeroImpedanceBranchParameters, jintArray jZeroImpedanceBranchIndices,
        jintArray jDistrRhoRow, jintArray jDistrRhoColumn, jdoubleArray jDistrRhoCoefficient,
        jintArray jDistrShuntBRow, jintArray jDistrShuntBColumn, jdoubleArray jDistrShuntBCoefficient, jintArray jClosedBranchPhaseControlSide,
        jdoubleArray jLoadModelParameters, jintArray jLoadModelIndices, jintArray jReactiveRelocationGroupStart, jintArray jReactiveRelocationCandidateRows,
        jintArray jOpenBranchAlternateReactiveDestination, jintArray jZeroImpedanceBranchAlternateReactiveDestination, jintArray jReactiveRelocationDependentBaseGroup, jintArray jReactiveRelocationBaseTargetRow, jintArray jReactiveRelocationDependentGroupOfRow) {
    jdouble* closedBranchParameters = env->GetDoubleArrayElements(jClosedBranchParameters, nullptr);
    jint* closedBranchIndices = env->GetIntArrayElements(jClosedBranchIndices, nullptr);
    jdouble* openBranchParameters = env->GetDoubleArrayElements(jOpenBranchParameters, nullptr);
    jint* openBranchIndices = env->GetIntArrayElements(jOpenBranchIndices, nullptr);
    jdouble* shuntParameters = env->GetDoubleArrayElements(jShuntParameters, nullptr);
    jint* shuntIndices = env->GetIntArrayElements(jShuntIndices, nullptr);
    jdouble* hvdcAcEmulationParameters = env->GetDoubleArrayElements(jHvdcAcEmulationParameters, nullptr);
    jint* hvdcAcEmulationIndices = env->GetIntArrayElements(jHvdcAcEmulationIndices, nullptr);
    jint* distrQElement = env->GetIntArrayElements(jDistrQElement, nullptr);
    jint* distrQSide = env->GetIntArrayElements(jDistrQSide, nullptr);
    jdouble* distrQWeight = env->GetDoubleArrayElements(jDistrQWeight, nullptr);
    jint* distrQRow = env->GetIntArrayElements(jDistrQRow, nullptr);
    jint* distrQKind = env->GetIntArrayElements(jDistrQKind, nullptr);
    jdouble* zeroImpedanceBranchParameters = env->GetDoubleArrayElements(jZeroImpedanceBranchParameters, nullptr);
    jint* zeroImpedanceBranchIndices = env->GetIntArrayElements(jZeroImpedanceBranchIndices, nullptr);
    jint* distrRhoRow = env->GetIntArrayElements(jDistrRhoRow, nullptr);
    jint* distrRhoColumn = env->GetIntArrayElements(jDistrRhoColumn, nullptr);
    jdouble* distrRhoCoefficient = env->GetDoubleArrayElements(jDistrRhoCoefficient, nullptr);
    jint* distrShuntBRow = env->GetIntArrayElements(jDistrShuntBRow, nullptr);
    jint* distrShuntBColumn = env->GetIntArrayElements(jDistrShuntBColumn, nullptr);
    jdouble* distrShuntBCoefficient = env->GetDoubleArrayElements(jDistrShuntBCoefficient, nullptr);
    jint* closedBranchPhaseControlSide = env->GetIntArrayElements(jClosedBranchPhaseControlSide, nullptr);
    jint* reactiveRelocationGroupStart = env->GetIntArrayElements(jReactiveRelocationGroupStart, nullptr);
    jint* reactiveRelocationCandidateRows = env->GetIntArrayElements(jReactiveRelocationCandidateRows, nullptr);
    int nReactiveRelocationGroups = env->GetArrayLength(jReactiveRelocationGroupStart) - 1;               // groups; reactiveRelocationGroupStart = [0] when none
    int nReactiveRelocationCandidateRows = env->GetArrayLength(jReactiveRelocationCandidateRows);
    jdouble* loadModelParameters = env->GetDoubleArrayElements(jLoadModelParameters, nullptr);
    jint* loadModelIndices = env->GetIntArrayElements(jLoadModelIndices, nullptr);
    jint* openBranchAlternateReactiveDestination = env->GetIntArrayElements(jOpenBranchAlternateReactiveDestination, nullptr);
    jint* zeroImpedanceBranchAlternateReactiveDestination = env->GetIntArrayElements(jZeroImpedanceBranchAlternateReactiveDestination, nullptr);
    jint* reactiveRelocationDependentBaseGroup = env->GetIntArrayElements(jReactiveRelocationDependentBaseGroup, nullptr);
    jint* reactiveRelocationBaseTargetRow = env->GetIntArrayElements(jReactiveRelocationBaseTargetRow, nullptr);
    jint* reactiveRelocationDependentGroupOfRow = env->GetIntArrayElements(jReactiveRelocationDependentGroupOfRow, nullptr);

    int nClosedBranch = env->GetArrayLength(jClosedBranchIndices) / 12;             // closedBranchIndices stride 12 (…a1col r1col alternateQRow1 alternateQRow2)
    int nOpenBranch = env->GetArrayLength(jOpenBranchIndices) / 6;
    int nShunt = env->GetArrayLength(jShuntIndices) / 5;             // shuntIndices stride 5 (P Q v bcol alternateQRows)
    int nHvdcAcEmulation = env->GetArrayLength(jHvdcAcEmulationIndices) / 4;
    int nDistrQ = env->GetArrayLength(jDistrQElement);
    int nZeroImpedanceBranch = env->GetArrayLength(jZeroImpedanceBranchIndices) / 11;
    int nDistrRho = env->GetArrayLength(jDistrRhoRow);
    int nDistrShuntB = env->GetArrayLength(jDistrShuntBRow);
    int nLoadModelTerms = env->GetArrayLength(jLoadModelIndices) / 2;             // loadModelIndices stride 2 (row, vCol)

    std::string error;
    SharedStructurePtr* structurePointer = nullptr;
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
        checkRange("closedBranchIndices", closedBranchIndices, nClosedBranch, 12, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, -1, n);
        // alternateQRows slots ([10]/[11] cb, [4] sh) also carry the GROUP encoding -2-groupId (plan-A6)
        checkRange("closedBranchIndices.alternateQRows", closedBranchIndices, nClosedBranch, 12, {10, 11}, -1 - nReactiveRelocationGroups, n);
        checkRange("openBranchIndices", openBranchIndices, nOpenBranch, 6, {0, 1, 2, 3, 4, 5}, -1, n);
        checkRange("openBranchAlternateReactiveDestination", openBranchAlternateReactiveDestination, nOpenBranch, 2, {0, 1}, -1 - nReactiveRelocationGroups, n);
        checkRange("shuntIndices", shuntIndices, nShunt, 5, {0, 1, 2, 3}, -1, n);
        checkRange("shuntIndices.alternateQRows", shuntIndices, nShunt, 5, {4}, -1 - nReactiveRelocationGroups, n);
        checkRange("reactiveRelocationCandidateRows", reactiveRelocationCandidateRows, nReactiveRelocationCandidateRows, 1, {0}, 0, n);
        for (int g = 0; g < nReactiveRelocationGroups; g++) {
            if (reactiveRelocationGroupStart[g] < 0 || reactiveRelocationGroupStart[g + 1] <= reactiveRelocationGroupStart[g] + 1 || reactiveRelocationGroupStart[g + 1] > nReactiveRelocationCandidateRows) {
                throw std::runtime_error("bad reactiveRelocationGroupStart group " + std::to_string(g));
            }
        }
        checkRange("hvdcAcEmulationIndices", hvdcAcEmulationIndices, nHvdcAcEmulation, 4, {0, 1, 2, 3}, -1, n);
        checkRange("distrQRow", distrQRow, nDistrQ, 1, {0}, 0, n);
        checkRange("zeroImpedanceBranchIndices", zeroImpedanceBranchIndices, nZeroImpedanceBranch, 11, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, -1, n);
        // slot 10 is the FLAGS word, not a row: bit 0 spanning-tree, bits 1-2 the relocated child side + 1.
        // Checking it against n rejected a valid
        // structure on any network with fewer than 14 rows once the relocation went on by default
        // ("bad zeroImpedanceBranchIndices at elem 0 slot 10: 13 (allowed [-1, 10))", four fixtures).
        checkRange("zeroImpedanceBranchIndices.flags", zeroImpedanceBranchIndices, nZeroImpedanceBranch, 11, {10}, 0, 8);
        checkRange("zeroImpedanceBranchAlternateReactiveDestination", zeroImpedanceBranchAlternateReactiveDestination, nZeroImpedanceBranch, 2, {0, 1}, -1 - nReactiveRelocationGroups, n);
        checkRange("distrRhoRow", distrRhoRow, nDistrRho, 1, {0}, 0, n);
        checkRange("distrRhoColumn", distrRhoColumn, nDistrRho, 1, {0}, 0, n);
        checkRange("distrShuntBRow", distrShuntBRow, nDistrShuntB, 1, {0}, 0, n);
        checkRange("distrShuntBColumn", distrShuntBColumn, nDistrShuntB, 1, {0}, 0, n);
        checkRange("loadModelIndices", loadModelIndices, nLoadModelTerms, 2, {0, 1}, 0, n);
        for (int e = 0; e < nClosedBranch; e++) {                      // encoded (a1Col << 1) | side, or -1
            if (closedBranchPhaseControlSide[e] >= 0 && (closedBranchPhaseControlSide[e] >> 1) >= n) {
                throw std::runtime_error("bad closedBranchPhaseControlSide at elem " + std::to_string(e)
                        + ": a1Col " + std::to_string(closedBranchPhaseControlSide[e] >> 1) + " >= n " + std::to_string(n));
            }
        }
        structurePointer = new SharedStructurePtr(buildStructure(n, nClosedBranch, closedBranchParameters, closedBranchIndices, nOpenBranch, openBranchParameters, openBranchIndices,
                                                   nShunt, shuntParameters, shuntIndices, nHvdcAcEmulation, hvdcAcEmulationParameters, hvdcAcEmulationIndices,
                                                   nDistrQ, distrQElement, distrQSide, distrQWeight, distrQRow, distrQKind,
                                                   nZeroImpedanceBranch, zeroImpedanceBranchParameters, zeroImpedanceBranchIndices,
                                                   nDistrRho, distrRhoRow, distrRhoColumn, distrRhoCoefficient,
                                                   nDistrShuntB, distrShuntBRow, distrShuntBColumn, distrShuntBCoefficient, closedBranchPhaseControlSide,
                                                   nLoadModelTerms, loadModelParameters, loadModelIndices, nReactiveRelocationGroups, reactiveRelocationGroupStart, reactiveRelocationCandidateRows,
                                                   openBranchAlternateReactiveDestination, zeroImpedanceBranchAlternateReactiveDestination, reactiveRelocationDependentBaseGroup, reactiveRelocationBaseTargetRow, reactiveRelocationDependentGroupOfRow));
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }

    env->ReleaseDoubleArrayElements(jClosedBranchParameters, closedBranchParameters, JNI_ABORT);
    env->ReleaseIntArrayElements(jClosedBranchIndices, closedBranchIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(jClosedBranchPhaseControlSide, closedBranchPhaseControlSide, JNI_ABORT);
    env->ReleaseIntArrayElements(jReactiveRelocationGroupStart, reactiveRelocationGroupStart, JNI_ABORT);
    env->ReleaseIntArrayElements(jReactiveRelocationCandidateRows, reactiveRelocationCandidateRows, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jOpenBranchParameters, openBranchParameters, JNI_ABORT);
    env->ReleaseIntArrayElements(jOpenBranchIndices, openBranchIndices, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jShuntParameters, shuntParameters, JNI_ABORT);
    env->ReleaseIntArrayElements(jShuntIndices, shuntIndices, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jHvdcAcEmulationParameters, hvdcAcEmulationParameters, JNI_ABORT);
    env->ReleaseIntArrayElements(jHvdcAcEmulationIndices, hvdcAcEmulationIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(jDistrQElement, distrQElement, JNI_ABORT);
    env->ReleaseIntArrayElements(jDistrQSide, distrQSide, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jDistrQWeight, distrQWeight, JNI_ABORT);
    env->ReleaseIntArrayElements(jDistrQRow, distrQRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jDistrQKind, distrQKind, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jZeroImpedanceBranchParameters, zeroImpedanceBranchParameters, JNI_ABORT);
    env->ReleaseIntArrayElements(jZeroImpedanceBranchIndices, zeroImpedanceBranchIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(jDistrRhoRow, distrRhoRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jDistrRhoColumn, distrRhoColumn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jDistrRhoCoefficient, distrRhoCoefficient, JNI_ABORT);
    env->ReleaseIntArrayElements(jDistrShuntBRow, distrShuntBRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jDistrShuntBColumn, distrShuntBColumn, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jDistrShuntBCoefficient, distrShuntBCoefficient, JNI_ABORT);
    env->ReleaseDoubleArrayElements(jLoadModelParameters, loadModelParameters, JNI_ABORT);
    env->ReleaseIntArrayElements(jLoadModelIndices, loadModelIndices, JNI_ABORT);
    env->ReleaseIntArrayElements(jOpenBranchAlternateReactiveDestination, openBranchAlternateReactiveDestination, JNI_ABORT);
    env->ReleaseIntArrayElements(jZeroImpedanceBranchAlternateReactiveDestination, zeroImpedanceBranchAlternateReactiveDestination, JNI_ABORT);
    env->ReleaseIntArrayElements(jReactiveRelocationDependentBaseGroup, reactiveRelocationDependentBaseGroup, JNI_ABORT);
    env->ReleaseIntArrayElements(jReactiveRelocationBaseTargetRow, reactiveRelocationBaseTargetRow, JNI_ABORT);
    env->ReleaseIntArrayElements(jReactiveRelocationDependentGroupOfRow, reactiveRelocationDependentGroupOfRow, JNI_ABORT);

    if (!error.empty()) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                      ("GPU structure creation failed: " + error).c_str());
        return 0;
    }
    return reinterpret_cast<jlong>(structurePointer);
}

extern "C" JNIEXPORT void JNICALL
OLF_JNI(ac_equations_vector_gpu, StructureNative, destroyStructure)(
        JNIEnv*, jclass, jlong handle) {
    // Drops the Java-side owner reference only: contexts hold their own shared_ptr copies,
    // so the device-resident structure lives until the last context is destroyed.
    delete reinterpret_cast<SharedStructurePtr*>(handle);
}

extern "C" JNIEXPORT jint JNICALL
OLF_JNI(ac_equations_vector_gpu, StructureNative, recommendBatchSize)(
        JNIEnv* env, jclass, jlong structureHandle, jint maxCap) {
    auto* structurePointer = reinterpret_cast<SharedStructurePtr*>(structureHandle);
    int rec = 1;
    try {
        if (structurePointer == nullptr || !*structurePointer) {
            throw std::runtime_error("invalid structure handle");
        }
        SharedStructure& structure = **structurePointer;
        // Per-scenario device footprint: the dominant term is cuDSS's LU numeric factors
        // (the exact fill from a probe analysis — cached on the structure — or the heuristic),
        // plus the value/state arrays (vals + x/target/F/B/dx). The LU index structure is
        // shared across the batch.
        constexpr double SAFETY = 0.7;                       // leave headroom for cuDSS workspace + fragmentation
        double luBytes = luBytesPerScenario(structure);
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));  // after the probe freed its buffers
        // 7n doubles per scenario, not 5: x/target/F/B/dx PLUS dQbuf (the per-scenario reactive
        // accumulator) and dLineSearchPreviousX (the line-search previous state, allocated under the RTE
        // parameters' MAX_VOLTAGE_CHANGE scaling).
        double perScenarioBytes = luBytes
                + (static_cast<double>(structure.nnz) + 7.0 * structure.n) * sizeof(double)
                + static_cast<double>(structure.n) * sizeof(int)    // dMode
                + static_cast<double>(structure.nElements);         // dDisabled (char)
        double budget = static_cast<double>(freeBytes) * SAFETY;
        long fit = perScenarioBytes > 0 ? static_cast<long>(budget / perScenarioBytes) : maxCap;
        rec = static_cast<int>(std::max<long>(1, std::min<long>(maxCap, fit)));
    } catch (...) {
        rec = 1;                                             // a query failure yields the safest chunk
    }
    return rec;
}
