/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
/**
 * The GPU sensitivity-analysis provider: {@link com.powsybl.openloadflow.sensi.gpu.GpuSensitivityAnalysisProvider}
 * runs OLF's sensitivity analysis with its AC load flows on the GPU Newton-Raphson, the sensitivities computed by OLF
 * on the Jacobian at the state they reach.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
package com.powsybl.openloadflow.sensi.gpu;
