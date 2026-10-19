/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
/**
 * The plan of a batched security analysis: what each contingency (or contingency and operator strategy pair)
 * changes, and whether the device can solve it. The contingencies as the device sees them ({@code BatchContingency}
 * and its lost loads, shunt shifts and tap overrides), their qualification ({@code GpuBatchContingencyValidator},
 * {@code GpuBatchQualification}), the branch and injection outage planners, the generator outages
 * ({@code GenOutage}), the voltage control overrides and re-homing, the islanded buses, the scheduled loss and its
 * pre-distribution, and the slack relocation. The solve side
 * ({@link com.powsybl.openloadflow.ac.solver.gpu.batch}) reads this package; this package never reads it.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;
