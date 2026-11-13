/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

/**
 * The linear solver of the batched Newton-Raphson, and of the DC paths (load flow, Woodbury security analysis,
 * sensitivity analysis, PTDF). The robust single-matrix retry of the batch is cuDSS's either way.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public enum GpuLinearSolver {
    /** cuDSS's uniform batch: every slot of the bank is factorized, an idle one as an identity system. On the DC paths,
     *  cuDSS's factorization of the one DC matrix, with its multi-right-hand-side solve. */
    CUDSS,
    /** Our batched sparse LU, the default: only the scenarios still iterating are factorized, in cuDSS's elimination
     *  order. On the DC paths, the one DC matrix on one lane, ordered on its own element structure (each equation at
     *  its element's variable, then minimum degree), solved against many right-hand sides at a time; no cuDSS call. */
    BATCHED_LU
}
