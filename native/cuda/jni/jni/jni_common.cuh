// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// What every JNI file includes: the JNI and CUDA headers, the generated formula headers and every solver header
// the entry points marshal into.
#ifndef OLF_GPU_JNI_JNI_COMMON_CUH
#define OLF_GPU_JNI_JNI_COMMON_CUH

#include <execinfo.h>
#include <jni.h>
#include "olf_jni.h"   // OLF_JNI
#include <cuda_runtime.h>
#include <cudss.h>
#include <cublas_v2.h>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <map>
#include <cstring>
#include <vector>
#include "closed_branch.cuh"
#include "closed_branch_scatter.h"
#include "open_branch.cuh"
#include "open_branch_scatter.h"
#include "shunt.cuh"
#include "shunt_scatter.h"
#include "hvdc.cuh"
#include "hvdc_scatter.h"
#include "vdl.cuh"
#include "../common/loop_kinds.cuh"
#include "../common/stats.cuh"
#include "../common/check.cuh"
#include "../common/device_buf.cuh"
#include "../common/launch.cuh"
#include "../diagnostics/driver_traces.cuh"
#include "../diagnostics/matrix_scan.cuh"
#include "../equations/branch_fill.cuh"
#include "../equations/bus_injection_fill.cuh"
#include "../equations/distribution_fill.cuh"
#include "../equations/fill_launch.cuh"
#include "../equations/scatter.cuh"
#include "../equations/scenario_overrides.cuh"
#include "../equations/zero_impedance_fill.cuh"
#include "../outerloop/distributed_slack.cuh"
#include "../outerloop/forced_target_q.cuh"
#include "../outerloop/hvdc_ac_emulation.cuh"
#include "../outerloop/incremental_flow_control.cuh"
#include "../outerloop/incremental_shunt_voltage_control.cuh"
#include "../outerloop/incremental_transformer_voltage_control.cuh"
#include "../outerloop/phase_control.cuh"
#include "../outerloop/pilot_voltage.cuh"
#include "../outerloop/reactive_capability_curve.cuh"
#include "../outerloop/reactive_capability_curves.cuh"
#include "../outerloop/reactive_limits.cuh"
#include "../outerloop/reactive_limits_seed.cuh"
#include "../outerloop/remote_voltage_control.cuh"
#include "../outerloop/secondary_voltage_control.cuh"
#include "../outerloop/sensitivity.cuh"
#include "../outerloop/shunt_voltage_control.cuh"
#include "../outerloop/slack_participants.cuh"
#include "../outerloop/transformer_voltage_control.cuh"
#include "../outerloop/voltage_monitoring.cuh"
#include "../solver/batch_context.cuh"
#include "../solver/compact_solve.cuh"
#include "../solver/convergence.cuh"
#include "../solver/driver_pass.cuh"
#include "../solver/driver_steps.cuh"
#include "../solver/iterative_refinement.cuh"
#include "../solver/linear_solver_settings.cuh"
#include "../solver/newton_iteration.cuh"
#include "../solver/realistic_voltage_check.cuh"
#include "../solver/solve_seed.cuh"
#include "../solver/state_vector_scaling.cuh"
#include "../solver/zero_diagonal_repair.cuh"
#include "../structure/csr_pattern.cuh"
#include "../structure/shared_structure.cuh"
#include "../solver/outer_loop_driver.cuh"

using namespace powsybl::openloadflow;
using namespace olfgpu;

#endif  // OLF_GPU_JNI_JNI_COMMON_CUH
