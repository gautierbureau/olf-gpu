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
struct CurveHvdcAcEmulationView {
    const int* hvdcStation;            // per curve entry {hvdc index in the loop pack, GpuTwoSides side}, -1 / NONE for a plain generator
    int nHvdcAcEmulation;
    const int* status;        // per (scenario, hvdc)
    const double* appliedDelta1;
    const double* appliedDelta2;
    const double* hvdcAcEmulationParameters;         // per hvdc x9: p0 k pMax12 pMax21 loss1 loss2 r baseD1 baseD2
    const int* bus1PhiColumn;
    const int* bus2PhiColumn;
    const int* hvdcElement;
    const char* disabled;
    int nElements;
    const char* contingencyOutage;
    const double* x;
};

// The curve kernel's view of the AC-emulation loop state.
CurveHvdcAcEmulationView curveHvdcAcEmulationView(BatchSolverContext* ctx);

// Reactive limits of every curve entry at the scenario's current active power.
__global__ void reactiveCapabilityCurveLimitsBatch(int nReactiveLimitEntries, int nScenarios, int nReactiveCapabilityCurves, int n, int reactiveCapabilityCurveExtrapolate,
                                   const int* reactiveCapabilityCurveIndices, const double* reactiveCapabilityCurveParameters, const double* reactiveCapabilityCurvePoints,
                                   const int* reactiveLimitEntryCurveStart, const int* reactiveLimitEntryCurves, CurveHvdcAcEmulationView reactiveCapabilityCurveHvdcStation,
                                   const double* reactiveLimitMinQ, const double* reactiveLimitMaxQ, const double* reactiveLimitLoadQ,
                                   const double* target,
                                   SlotEntries reactiveLimitDeltaOverrides, SlotEntries reactiveCapabilityCurveOverrides,
                                   const double* distributedSlackGeneratorMove, const char* distributedSlackGeneratorMoveValid, const double* distributedSlackGeneratorAnchor, const double* distributedSlackGeneratorInitialMove, int nDistributedSlackGenerators,
                                   const int* distributedSlackBusPhiRow, const int* distributedSlackBusRowOverride, int nDistributedSlackBuses, const int* distributedSlackGeneratorBus,
                                   double* minQDyn, double* maxQDyn, double* loadQDyn, const char* active);

}  // namespace olfgpu

#endif  // OLF_GPU_OUTERLOOP_REACTIVE_CAPABILITY_CURVES_CUH
