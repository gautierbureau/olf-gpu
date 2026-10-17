/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;

import java.util.Objects;

/**
 * OLF's Newton-Raphson parameters plus the GPU solver's own ({@link GpuLoadFlowParameters}), as
 * NewtonKrylovParameters carries Newton-Krylov's.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuNewtonRaphsonParameters extends NewtonRaphsonParameters {

    private GpuLoadFlowParameters gpuParameters = new GpuLoadFlowParameters();

    public GpuLoadFlowParameters getGpuParameters() {
        return gpuParameters;
    }

    public GpuNewtonRaphsonParameters setGpuParameters(GpuLoadFlowParameters gpuParameters) {
        this.gpuParameters = Objects.requireNonNull(gpuParameters);
        return this;
    }
}
