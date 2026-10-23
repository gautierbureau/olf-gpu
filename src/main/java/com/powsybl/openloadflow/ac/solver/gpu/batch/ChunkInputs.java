/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.outerloop.gpu.DevicePhases;
import com.powsybl.openloadflow.ac.outerloop.gpu.ForceTargetQInReactiveLimits;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuAreaInterchangeControlOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.GpuHvdcAcEmulationLimitsOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.gpu.ReactiveCapabilityCurves;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchQualification;
import com.powsybl.openloadflow.network.LfHvdc;

import java.util.List;
import java.util.Map;

/**
 * Everything the chunk solver reads, built once per security analysis run: the contingencies, the base case, the
 * qualification's scenario plan, the solve settings, the driver-phase inputs, the result sinks and the run's
 * configuration.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
record ChunkInputs(List<BatchContingency> contingencies, ChunkInputs.BaseCase base, GpuBatchQualification.ScenarioPlan plan,
                   ChunkInputs.SolveSettings settings, ChunkInputs.DevicePhaseInputs phases, ChunkInputs.ResultSinks sinks,
                   BatchConfig config) {

    /** The base case every scenario starts from: the extraction, its element numbering, the base state, targets
     *  and row modes, the state size, the element count, the variable types (line search mode 2 only, else
     *  null) and each HVDC's element index. */
    record BaseCase(GpuAcElementMapper elementMapper, GpuAcData gpuData, double[] baseState, double[] baseTargets,
                    int[] baseRowMode, int n, int nElements, int[] varType, Map<LfHvdc, Integer> hvdcElemIndex) {
    }

    /** The run's Newton-Raphson parameters, whether its outer loops hold OLF's DistributedSlackOuterLoop and
     *  ReactiveLimitsOuterLoop, its convergence thresholds and the per-row criteria (null under UNIFORM_CRITERIA). */
    record SolveSettings(NewtonRaphsonParameters parameters, boolean distributedSlackOuterLoop, boolean reactiveLimitsOuterLoop,
                         double tol, double reactiveLimitTol, double slackDistTol, double convEpsPerEq, double[][] rowCriteria) {
    }

    /** The run's driver phases, its AC-emulation HVDC controllers (null without the HVDC loops), the
     *  reactive-capability curve and forced-target-Q channels, and the AreaInterchangeControl tables with each
     *  participating generator's move from its initial target in the base (null without areas). */
    record DevicePhaseInputs(DevicePhases.Built phases, List<GpuHvdcAcEmulationLimitsOuterLoop.Controller> hvdcControllers,
                             ReactiveCapabilityCurves.Channel reactiveCapabilityCurves, ForceTargetQInReactiveLimits.Channel forcedTargetQ,
                             GpuAreaInterchangeControlOuterLoop.Tables areaInterchangeControl, double[] areaInterchangeControlBaseMoves) {

        /** The run balances areas on the device. */
        boolean areaInterchangeControlWithAreas() {
            return areaInterchangeControl != null && areaInterchangeControl.areaCount() > 0;
        }
    }

    /** Where the results go, per contingency index: the result row (null = route to the CPU), the replayed loss
     *  pre-distribution and the total distributed active power (pu, NaN = unknown). */
    record ResultSinks(double[][] resultsByIndex, double[] preDistributedByIndex, double[] distributedByIndex) {
    }

    /** The same inputs from another base state. */
    ChunkInputs withBaseState(double[] baseState) {
        return new ChunkInputs(contingencies, new BaseCase(base.elementMapper(), base.gpuData(), baseState,
                base.baseTargets(), base.baseRowMode(), base.n(), base.nElements(), base.varType(), base.hvdcElemIndex()),
                plan, settings, phases, sinks, config);
    }
}
