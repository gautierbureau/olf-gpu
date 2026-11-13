// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// A fingerprint of a built shared structure (OLF_GPU_STRUCTURE_FINGERPRINT): one hash per part of what the structure
// holds, so that a change meant to leave the structure as it is can be checked against the same run before it, and a
// difference names the part it is in.
#ifndef OLF_GPU_DIAGNOSTICS_STRUCTURE_FINGERPRINT_CUH
#define OLF_GPU_DIAGNOSTICS_STRUCTURE_FINGERPRINT_CUH

#include "../structure/shared_structure.cuh"

namespace olfgpu {

// Writes one STRUCTURE_FINGERPRINT line: the sizes, a hash over every part, then each part's hash. Appended to the file
// OLF_GPU_STRUCTURE_FINGERPRINT names, or printed on stderr when it is 1.
void traceStructureFingerprint(const SharedStructure& structure);

}  // namespace olfgpu

#endif  // OLF_GPU_DIAGNOSTICS_STRUCTURE_FINGERPRINT_CUH
