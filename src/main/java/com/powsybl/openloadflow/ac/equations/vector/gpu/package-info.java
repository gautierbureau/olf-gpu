/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
/**
 * GPU AC equation-system extraction and the JNI boundary: turns an OLF vectorized equation
 * system into the fixed-pattern device data ({@link com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData})
 * the native full Newton-Raphson consumes. The extractor ({@link com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor}
 * and its {@code GpuAcValidation} / {@code GpuAcFamilyPacker} /
 * per-pack {@code *Packer} helpers), the activation extractor, the per-extraction structure
 * cache, and the {@code GpuAcNewtonSolver} JNI binding (with {@code GpuNativeLoader}) live here.
 * The stable rows and the row placement they use (device-only) live in the {@code fixedmatrix} sub-package.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;
