// Copyright (c) 2026, RTE (http://www.rte-france.com)
// SPDX-License-Identifier: MPL-2.0
//
// The diagnostic channels of the native side: each one an environment variable, read once on first use. A channel
// only prints or writes files; none changes a result. The Java side has its own list
// (com.powsybl.openloadflow.util.gpu.GpuTrace); a variable both sides read is listed on both.
#ifndef OLF_GPU_COMMON_TRACE_CUH
#define OLF_GPU_COMMON_TRACE_CUH

#include <array>
#include <cstdlib>

enum class Trace : int {
    BATCH_TRACE,          // OLF_GPU_BATCH_TRACE: per-iteration convergence, and what every outer loop decided
    DQ_TRACE,             // OLF_GPU_DQ_TRACE=<row>: every DISTR_Q residual contribution landing in that row
    DX_DUMP_ON_REJECT,    // OLF_GPU_DX_DUMP_ON_REJECT=<dir>: J, F, B, dx and modes of the slots the linear-solve guard rejects
    LOOP_TRACE,           // OLF_GPU_LOOP_TRACE=<slot>: that slot's outer-loop passes (or the OLF_GPU_LOOP_TRACE_CTG slot)
    LU_COMPARE,           // OLF_GPU_LU_COMPARE: every Newton step of our batched LU against cuDSS's uniform batch
    PHASE_TIME,           // OLF_GPU_PHASE_TIME: where the time goes (analysis, outer-loop passes, solves)
    REJECT_TRACE,         // OLF_GPU_REJECT_TRACE: the scenarios the device rejects, with the reason
    RL_AUDIT,             // OLF_GPU_RL_AUDIT[=<tol>]: the final state audited against the rule that ends OLF's reactive-limit loop
    RL_TRACE,             // OLF_GPU_RL_TRACE: the reactive-limit tables a context is built with
    SLOT_DUMP,            // OLF_GPU_SLOT_DUMP=<prefix>: the traced slot's row targets, state and modes at driver start and pass 0
    STATE_DUMP,           // OLF_GPU_STATE_DUMP=<n>: slot 0's rows (mode, target, state, residual) for iterations 0..n
    SW_LOG,               // OLF_GPU_SW_LOG=<slot> or all: every reactive-limit check of that slot, logged on the device
    SW_TRACE,             // OLF_GPU_SW_TRACE=<slot>: every controller that switches in that slot's reactive-limit passes
    TRACE,                // OLF_GPU_TRACE: cuDSS analysis timing and the zero-diagonal repair's tables
    UNREALISTIC_TRACE,    // OLF_GPU_UNREALISTIC_TRACE: the buses that make a post-contingency state unrealistic
    COUNT
};

// The variable's value, or nullptr when it is not set.
inline const char* traceValue(Trace channel) {
    static const std::array<const char*, static_cast<int>(Trace::COUNT)> values = [] {
        const char* variables[] = {"OLF_GPU_BATCH_TRACE", "OLF_GPU_DQ_TRACE", "OLF_GPU_DX_DUMP_ON_REJECT", "OLF_GPU_LOOP_TRACE",
                                   "OLF_GPU_LU_COMPARE", "OLF_GPU_PHASE_TIME", "OLF_GPU_REJECT_TRACE", "OLF_GPU_RL_AUDIT", "OLF_GPU_RL_TRACE",
                                   "OLF_GPU_SLOT_DUMP", "OLF_GPU_STATE_DUMP", "OLF_GPU_SW_LOG", "OLF_GPU_SW_TRACE", "OLF_GPU_TRACE",
                                   "OLF_GPU_UNREALISTIC_TRACE"};
        static_assert(sizeof(variables) / sizeof(variables[0]) == static_cast<int>(Trace::COUNT), "one variable per channel");
        std::array<const char*, static_cast<int>(Trace::COUNT)> read{};
        for (int c = 0; c < static_cast<int>(Trace::COUNT); c++) {
            read[c] = std::getenv(variables[c]);
        }
        return read;
    }();
    return values[static_cast<int>(channel)];
}

// True when the variable is set, to anything.
inline bool traceOn(Trace channel) {
    return traceValue(channel) != nullptr;
}

#endif  // OLF_GPU_COMMON_TRACE_CUH
