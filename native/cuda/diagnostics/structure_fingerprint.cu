// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
#include "structure_fingerprint.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../common/check.cuh"
#include "../common/trace.cuh"

namespace olfgpu {

namespace {

// FNV-1a, 64 bits: order-sensitive, which is the point (a destination moved to another output is a change).
struct Hash {
    uint64_t value = 1469598103934665603ULL;
    void bytes(const void* data, size_t count) {
        const unsigned char* b = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < count; i++) {
            value = (value ^ b[i]) * 1099511628211ULL;
        }
    }
    void number(long long v) { bytes(&v, sizeof(v)); }
    template <typename T> void host(const std::vector<T>& v) {
        number(static_cast<long long>(v.size()));
        bytes(v.data(), v.size() * sizeof(T));
    }
    // A device buffer's whole allocation: every structure buffer is uploaded once at its exact size, and a fresh
    // allocation is zeroed (DeviceBuf::ensure), so the bytes are the structure's and nothing else.
    void device(const DeviceBuf& buffer) {
        number(static_cast<long long>(buffer.cap));
        if (buffer.cap == 0) {
            return;
        }
        std::vector<unsigned char> copy(buffer.cap);
        CUDA_CHECK(cudaMemcpy(copy.data(), buffer.p, buffer.cap, cudaMemcpyDeviceToHost));
        bytes(copy.data(), copy.size());
    }
};

uint64_t familyHash(const Family& f) {
    Hash h;
    h.number(f.elementCount);
    h.number(f.parameterStride);
    h.number(f.indexStride);
    h.number(f.outputCount);
    h.number(f.dynamicRelocation);
    h.number(f.relocatedCount);
    h.number(f.hasPhaseControl);
    for (const DeviceBuf* b : {&f.parameters, &f.indices, &f.destination, &f.alternateDestination, &f.selectorRow,
                               &f.relocationGroup, &f.relocationCandidateOffset, &f.relocationCandidateDestinations,
                               &f.relocationDependentPilot, &f.relocatedElement, &f.relocatedElements,
                               &f.phaseControlDestination, &f.phaseControlSelector}) {
        h.device(*b);
    }
    return h.value;
}

}  // namespace

void traceStructureFingerprint(const SharedStructure& s) {
    struct Part {
        const char* name;
        uint64_t hash;
    };
    std::vector<Part> parts;

    Hash pattern;
    pattern.number(s.n);
    pattern.number(s.nnz);
    pattern.host(s.rowMap);
    pattern.host(s.rowPtr);
    pattern.host(s.colIdx);
    pattern.device(s.dDiagPos);
    parts.push_back({"pattern", pattern.value});

    parts.push_back({"closedBranch", familyHash(s.closedBranch)});
    parts.push_back({"openBranch", familyHash(s.openBranch)});
    parts.push_back({"shunt", familyHash(s.shunt)});
    parts.push_back({"hvdcAcEmulation", familyHash(s.hvdcAcEmulation)});

    Hash distrQ;
    distrQ.number(s.nDistrQ);
    for (const DeviceBuf* b : {&s.dDistrQElement, &s.dDistrQSide, &s.dDistrQKind, &s.dDistrQWeight, &s.dDistrQDestination,
                               &s.dRelocatedChildOwnRow, &s.dDistrQDummyQRow, &s.dDistrQAlternateDestination,
                               &s.dDistrQSelectorRow}) {
        distrQ.device(*b);
    }
    parts.push_back({"distrQ", distrQ.value});

    Hash zeroImpedance;
    zeroImpedance.number(s.nZeroImpedanceBranch);
    zeroImpedance.number(s.zeroImpedanceDynamicRelocation);
    for (const DeviceBuf* b : {&s.dZeroImpedanceBranchParameters, &s.dZeroImpedanceBranchIndices,
                               &s.dZeroImpedanceBranchDestination, &s.dZeroImpedanceAlternateDestination,
                               &s.dZeroImpedanceSelectorRow, &s.dZeroImpedanceRelocationGroup,
                               &s.dZeroImpedanceRelocationCandidateOffset, &s.dZeroImpedanceRelocationCandidateDestinations,
                               &s.dZeroImpedanceRelocationDependentPilot}) {
        zeroImpedance.device(*b);
    }
    parts.push_back({"zeroImpedance", zeroImpedance.value});

    Hash distribution;
    distribution.number(s.nDistrRho);
    distribution.number(s.hasDistrRhoRelocationGroups);
    distribution.number(s.nDistrShuntB);
    for (const DeviceBuf* b : {&s.dDistrRhoColumn, &s.dDistrRhoCoefficient, &s.dDistrRhoDestination, &s.dDistrRhoRelocationGroup,
                               &s.dDistrShuntBColumn, &s.dDistrShuntBCoefficient, &s.dDistrShuntBDestination}) {
        distribution.device(*b);
    }
    parts.push_back({"distribution", distribution.value});

    Hash loadModel;
    loadModel.number(s.nLoadModelTerms);
    loadModel.number(s.loadModelRelocated);
    for (const DeviceBuf* b : {&s.dLoadModelParameters, &s.dLoadModelIndices, &s.dLoadModelDestination,
                               &s.dLoadModelAlternateRow, &s.dLoadModelAlternateDestination}) {
        loadModel.device(*b);
    }
    parts.push_back({"loadModel", loadModel.value});

    Hash reactiveRelocation;
    reactiveRelocation.number(s.nReactiveRelocationGroups);
    reactiveRelocation.host(s.hReactiveRelocationGroupStart);
    reactiveRelocation.host(s.hReactiveRelocationCandidateRows);
    reactiveRelocation.host(s.hReactiveRelocationDependentBaseGroup);
    reactiveRelocation.host(s.hReactiveRelocationBaseTargetRow);
    reactiveRelocation.host(s.hReactiveRelocationDependentGroupOfRow);
    for (const DeviceBuf* b : {&s.dReactiveRelocationGroupStart, &s.dReactiveRelocationCandidateRows,
                               &s.dReactiveRelocationDependentBaseGroup, &s.dReactiveRelocationBaseTargetRow}) {
        reactiveRelocation.device(*b);
    }
    parts.push_back({"reactiveRelocation", reactiveRelocation.value});

    Hash zeroDiagonal;
    zeroDiagonal.number(s.nZeroDiagonalRows);
    zeroDiagonal.number(s.nZeroDiagonalRowValues);
    zeroDiagonal.host(s.hZeroDiagonalRow);
    for (const DeviceBuf* b : {&s.dZeroDiagonalRow, &s.dZeroDiagonalRowValueStart, &s.dZeroDiagonalDonorStart,
                               &s.dZeroDiagonalDonor, &s.dZeroDiagonalDonorOwnIndex, &s.dZeroDiagonalRepairSign,
                               &s.dZeroDiagonalRowValuePosition}) {
        zeroDiagonal.device(*b);
    }
    parts.push_back({"zeroDiagonal", zeroDiagonal.value});

    Hash all;
    for (const Part& p : parts) {
        all.number(static_cast<long long>(p.hash));
    }
    // One line, written in one call: on stderr it would otherwise interleave with the test runner's output, which
    // a file avoids altogether (several structures of one run append to it).
    char line[1024];
    int length = std::snprintf(line, sizeof(line), "STRUCTURE_FINGERPRINT n=%d nnz=%d all=%016llx", s.n, s.nnz,
                               static_cast<unsigned long long>(all.value));
    for (const Part& p : parts) {
        length += std::snprintf(line + length, sizeof(line) - length, " %s=%016llx", p.name, static_cast<unsigned long long>(p.hash));
    }
    std::snprintf(line + length, sizeof(line) - length, "\n");
    const char* path = traceValue(Trace::STRUCTURE_FINGERPRINT);
    std::FILE* file = path != nullptr && std::strcmp(path, "1") != 0 ? std::fopen(path, "a") : nullptr;
    std::fputs(line, file != nullptr ? file : stderr);
    if (file != nullptr) {
        std::fclose(file);
    }
}

}  // namespace olfgpu
