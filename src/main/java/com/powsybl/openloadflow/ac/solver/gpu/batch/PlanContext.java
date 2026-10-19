/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcElementMapper;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;

import java.util.List;
import java.util.Set;

/**
 * What every planner of {@link GpuBatchQualification#planScenarios} reads besides the contingency it plans: the
 * contingencies and their qualifications, the base extraction and its element numbering, the equation system, the
 * distributed-slack settings, the DISTR_RHO controllers and relocated couplers, and which control loops run.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
record PlanContext(List<BatchContingency> contingencies, List<GpuBatchContingencyValidator.Qualification> qualifications,
                   GpuAcElementMapper elementMapper, GpuAcData gpuData,
                   EquationSystem<AcVariableType, AcEquationType> es, double dsSum, boolean slackParticipates,
                   GpuDsParams dsParams, Set<LfBranch> distrRhoControllers, Set<LfBranch> relocatedCouplers,
                   boolean reactiveLimits, boolean incrementalTvc) {
}
