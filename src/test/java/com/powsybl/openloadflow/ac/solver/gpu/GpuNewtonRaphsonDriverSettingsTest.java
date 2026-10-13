/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.iidm.network.Network;
import com.powsybl.iidm.network.test.EurostagTutorialExample1Factory;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The bands the device driver reads off the run's own outer loops.
 */
class GpuNewtonRaphsonDriverSettingsTest {

    private static double rlBand(com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteriaType type) {
        Network network = EurostagTutorialExample1Factory.create();
        LoadFlowParameters lfp = new LoadFlowParameters().setUseReactiveLimits(true);
        OpenLoadFlowParameters olfp = OpenLoadFlowParameters.create(lfp)
                .setNewtonRaphsonStoppingCriteriaType(type)
                .setMaxReactivePowerMismatch(0.5)                 // MVar
                .setNewtonRaphsonConvEpsPerEq(1e-6);
        AcLoadFlowParameters acp = OpenLoadFlowParameters.createAcParameters(network, lfp, olfp,
                new SparseMatrixFactory(), new EvenShiloachGraphDecrementalConnectivityFactory<>(), false, false);
        return GpuNewtonRaphson.driverSettings(acp)[1];
    }

    /** OLF builds its reactive-limits loop with the band already in per unit (maxReactivePowerMismatch / SB under
     *  per-equation criteria); the driver must switch at that band, not at it divided by SB again. */
    @Test
    void perEquationTypeCriteriaBandIsOlfsPerUnitBand() {
        assertEquals(0.5 / 100, rlBand(com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteriaType.PER_EQUATION_TYPE_CRITERIA), 1e-15);
    }

    @Test
    void uniformCriteriaBandIsConvEpsPerEq() {
        assertEquals(1e-6, rlBand(com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteriaType.UNIFORM_CRITERIA), 1e-18);
    }
}
