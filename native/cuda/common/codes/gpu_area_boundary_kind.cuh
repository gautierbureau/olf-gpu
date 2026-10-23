// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// Which flow an area boundary reads (OLF's LfAreaImpl.BoundaryImpl.getP). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAreaBoundaryKind (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_AREA_BOUNDARY_KIND_CUH
#define OLF_GPU_COMMON_CODES_GPU_AREA_BOUNDARY_KIND_CUH

namespace GpuAreaBoundaryKind {
constexpr int CLOSED_BRANCH = 0;  // a branch connected at both sides: its p1 or p2
constexpr int OPEN_BRANCH = 1;    // a branch connected at one side: that side's p, 0 on the other
constexpr int TIE_LINE = 2;       // a tie line: the flow at the boundary point of the half on the boundary's side (SV.otherSideP)
}

#endif
