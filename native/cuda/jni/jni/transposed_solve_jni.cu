// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// JNI entry points of GpuAcNewtonSolver: The standalone transposed solve of the adjoint prototype.
#include "jni_common.cuh"

// ---------------------------------------------------------------------------------
// ADJOINT step 1a: solve Aᵀ x = b for an EXPLICIT CSR matrix, by building the transpose
// and factorizing it.
//
// Why this exists: cuDSS (through 0.8.0 — verified, its header has no transpose/OP_T
// symbol at all) offers only the non-transposed solve against its factors, so one GPU
// factorization yields ONE direction where KLU's klu_solve/klu_tsolve pair yields two.
// Our Jacobian is stored as TRUE CSR (row = equation, col = variable) so the plain solve
// serves the forward Newton; the adjoint (Jᵀλ = ...) is therefore the direction we cannot
// get from the forward's factors, and until cuDSS ships a transpose solve we buy it with a
// second factorization of an explicitly transposed matrix.
//
// This entry point is deliberately SELF-CONTAINED (its own handle/config/data, no reuse of
// the Newton context): it isolates exactly one question — is our CSR transpose + cuDSS
// factorize/solve of Aᵀ correct? — so a failure here cannot be confused with the adjoint's
// RHS assembly or contraction. It is a validation//measurement entry, not the production
// path; the production seam should reuse the forward's structure and swap to a native
// transpose solve the moment cuDSS provides one.
//
// CSR transpose is the standard counting sort: count entries per column of A (= rows of Aᵀ),
// prefix-sum into row pointers, then scatter. O(nnz + n), host-side, and for a FIXED pattern
// the resulting scatter is a permutation that can be computed once and reused per solve.
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_powsybl_openloadflow_ac_equations_vector_gpu_GpuAcNewtonSolver_solveTransposedExplicit(
        JNIEnv* env, jclass, jint n, jintArray jRowPtr, jintArray jColIdx,
        jdoubleArray jVals, jdoubleArray jB) {
    try {
        const int nn = n;
        std::vector<int> rowPtr(nn + 1);
        env->GetIntArrayRegion(jRowPtr, 0, nn + 1, rowPtr.data());
        const int nnz = rowPtr[nn];
        std::vector<int> colIdx(nnz);
        std::vector<double> vals(nnz);
        std::vector<double> b(nn);
        env->GetIntArrayRegion(jColIdx, 0, nnz, colIdx.data());
        env->GetDoubleArrayRegion(jVals, 0, nnz, vals.data());
        env->GetDoubleArrayRegion(jB, 0, nn, b.data());

        // --- CSR(A) -> CSR(Aᵀ) by counting sort ---
        std::vector<int> tRowPtr(nn + 1, 0);
        std::vector<int> tColIdx(nnz);
        std::vector<double> tVals(nnz);
        for (int k = 0; k < nnz; k++) {
            tRowPtr[colIdx[k] + 1]++;                        // column of A becomes row of Aᵀ
        }
        for (int r = 0; r < nn; r++) {
            tRowPtr[r + 1] += tRowPtr[r];
        }
        std::vector<int> cursor(tRowPtr.begin(), tRowPtr.end() - 1);
        for (int r = 0; r < nn; r++) {
            for (int k = rowPtr[r]; k < rowPtr[r + 1]; k++) {
                const int dst = cursor[colIdx[k]]++;
                tColIdx[dst] = r;                            // row of A becomes column of Aᵀ
                tVals[dst] = vals[k];
            }
        }

        DeviceBuf dRowPtr;
        DeviceBuf dColIdx;
        DeviceBuf dVals;
        DeviceBuf dB;
        DeviceBuf dX;
        dRowPtr.upload(tRowPtr.data(), (nn + 1) * sizeof(int));
        dColIdx.upload(tColIdx.data(), nnz * sizeof(int));
        dVals.upload(tVals.data(), nnz * sizeof(double));
        dB.upload(b.data(), nn * sizeof(double));
        dX.alloc(nn * sizeof(double));

        CudssCtx ds;
        CUDSS_CHECK(cudssCreate(&ds.handle));
        CUDSS_CHECK(cudssConfigCreate(&ds.config));
        applyReorderAlg(ds.config, false);   // adjoint: its own matrix, matching enabled
        applyMatchingAlg(ds.config);
        CUDSS_CHECK(cudssDataCreate(ds.handle, &ds.data));
        CUDSS_CHECK(cudssMatrixCreateCsr(&ds.matA, nn, nn, nnz, dRowPtr.as<int>(), nullptr,
                                         dColIdx.as<int>(), dVals.as<double>(), OLF_CUDSS_CSR_TYPES,
                                         CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matB, nn, 1, nn, dB.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        CUDSS_CHECK(cudssMatrixCreateDn(&ds.matX, nn, 1, nn, dX.as<double>(), OLF_CUDSS_DN_TYPE, CUDSS_LAYOUT_COL_MAJOR));
        gAnalysesMain++;
        { AnalysisTimer at_(__LINE__); CUDSS_CHECK(cudssExecute(ds.handle, CUDSS_PHASE_ANALYSIS, ds.config, ds.data, ds.matA, ds.matX, ds.matB)); }
        { AnalysisTimer at_(__LINE__, 'F'); CUDSS_CHECK(cudssExecute(ds.handle, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE,
                                 ds.config, ds.data, ds.matA, ds.matX, ds.matB)); }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> x(nn);
        CUDA_CHECK(cudaMemcpy(x.data(), dX.p, nn * sizeof(double), cudaMemcpyDeviceToHost));
        jdoubleArray out = env->NewDoubleArray(nn);
        env->SetDoubleArrayRegion(out, 0, nn, x.data());
        return out;
    } catch (const std::exception& e) {
        jclass ex = env->FindClass("java/lang/RuntimeException");
        if (ex) env->ThrowNew(ex, e.what());
        return nullptr;
    }
}
