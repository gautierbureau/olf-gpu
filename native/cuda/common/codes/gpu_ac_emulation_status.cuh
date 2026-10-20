// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// OLF's LfHvdc.AcEmulationControl.AcEmulationStatus as the device holds it, per (scenario, hvdc). Mirrors
// com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcEmulationStatus (GpuCodesMirrorTest checks they agree).
#ifndef OLF_GPU_COMMON_CODES_GPU_AC_EMULATION_STATUS_CUH
#define OLF_GPU_COMMON_CODES_GPU_AC_EMULATION_STATUS_CUH

namespace GpuAcEmulationStatus {
constexpr int LINEAR_MODE = 0;
constexpr int SATURATION_MODE_FROM_CS1_TO_CS2 = 1;
constexpr int SATURATION_MODE_FROM_CS2_TO_CS1 = 2;
constexpr int FROZEN = 3;
}

#endif
