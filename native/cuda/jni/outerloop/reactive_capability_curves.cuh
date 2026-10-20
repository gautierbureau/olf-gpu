// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// ReactiveLimitsOuterLoop limits from a ReactiveCapabilityCurve: each entry's minimum and maximum Q read at the
// scenario's current active power (the generator's slack move, or an AC-emulation link's flow).
#ifndef OLF_GPU_OUTERLOOP_REACTIVE_CAPABILITY_CURVES_CUH
#define OLF_GPU_OUTERLOOP_REACTIVE_CAPABILITY_CURVES_CUH

#include <cstdlib>
#include <string>
#include <vector>

#include "../solver/driver_pass.cuh"

namespace olfgpu {

// The AC-emulation VSC entries of the curve channel: OLF evaluates such a station's curve at the live flow of its
// link in the linear mode (LfVscConverterStationImpl.getReactiveLimitsActivePower: minus the side's emulation term),
// and at its target P otherwise (saturated, frozen, disabled link), which the HVDC loop holds as baseD + appliedD.
// hv == nullptr: no such entry. nHvdcAcEmulation == 0: the HVDC loops are not on this context (then nothing was packed).
struct CvHvdc {
    const int* hv;            // per curve entry {hvdc index in the loop pack, side 1|2}, -1 for a plain generator
    int nHvdcAcEmulation;
    const int* status;        // per (scenario, hvdc)
    const double* d1;
    const double* d2;
    const double* in;         // per hvdc x9: p0 k pMax12 pMax21 loss1 loss2 r baseD1 baseD2
    const int* phi1;
    const int* phi2;
    const int* elem;
    const char* disabled;
    int nElements;
    const char* ctgOut;
    const double* x;
};

// The curve kernel's view of the AC-emulation loop state.
CvHvdc cvHvdcView(BatchSolverContext* ctx);

// Reactive limits of every curve entry at the scenario's current active power.
__global__ void rlCurveLimitsBatch(int nrl, int nScenarios, int ncv, int n, int cvExtrapolate,
                                   const int* cvIdx, const double* cvIn, const double* cvPts,
                                   const int* cvPtr, const int* cvList, CvHvdc cvHv,
                                   const double* rlMinQ, const double* rlMaxQ, const double* rlLoadQ,
                                   const double* target,
                                   int nQOv, const int* qOvSlotEntry, const double* qOvMinMax,
                                   const int* qOvPtr, const int* qOvList,
                                   const int* cvOvPtr, const int* cvOvEntry, const double* cvOvVal,
                                   const double* genCum, const char* genCumValid, const double* dgAnchor, const double* dgStart, int ngen,
                                   const int* dsPhiRow, const int* dsRowOv, int nds, const int* dgBus,
                                   double* minQDyn, double* maxQDyn, double* loadQDyn);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_REACTIVE_CAPABILITY_CURVES_CUH
