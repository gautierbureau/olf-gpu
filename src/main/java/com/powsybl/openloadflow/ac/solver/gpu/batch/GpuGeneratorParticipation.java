/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.iidm.network.Generator;
import com.powsybl.iidm.network.extensions.ActivePowerControl;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.impl.AbstractLfGenerator;
import com.powsybl.openloadflow.util.PerUnit;

import java.util.function.DoublePredicate;

/**
 * Whether a generator takes part in the slack distribution at a given target P, as OLF decides it after a
 * GeneratorAction: {@code LfGeneratorAction.apply} sets the target and calls {@code reApplyActivePowerControlChecks},
 * which starts again from the generator's own participation flag (its ActivePowerControl extension) and runs
 * {@code checkActivePowerControl} at the new target (a zero target with generatorsWithZeroMwTargetAreNotStarted,
 * an implausible maxP, a target outside [minTargetP, maxTargetP], a flat range). The distribution step then also
 * needs a non-zero factor under the run's balance type, and the bus must be one the distribution covers.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuGeneratorParticipation {

    private GpuGeneratorParticipation() {
    }

    /** OLF's verdict for {@code lfGen} as a function of its target P (pu), or null when it is not a plain generator
     *  (OLF's re-check does nothing for the other kinds: their participation stays). */
    public static DoublePredicate atTargetP(LfGenerator lfGen, Generator generator, LfNetworkParameters parameters,
                                            LoadFlowParameters.BalanceType balanceType) {
        if (generator == null || !lfGen.getBus().isParticipating()) {
            return generator == null ? null : t -> false;
        }
        boolean initialParticipating = true;
        double minTargetP = generator.getMinP();
        double maxTargetP = generator.getMaxP();
        ActivePowerControl<Generator> apc = generator.getExtension(ActivePowerControl.class);
        if (apc != null) {
            initialParticipating = apc.isParticipate();
            if (apc.getMinTargetP().isPresent()) {
                minTargetP = apc.getMinTargetP().getAsDouble();
            }
            if (apc.getMaxTargetP().isPresent()) {
                maxTargetP = apc.getMaxTargetP().getAsDouble();
            }
        }
        boolean factor = switch (balanceType) {
            case PROPORTIONAL_TO_GENERATION_P_MAX -> lfGen.getDroop() != 0;
            case PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR -> lfGen.getParticipationFactor() > 0;
            case PROPORTIONAL_TO_GENERATION_P, PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN -> true;
            default -> false;                                // load-based: generators never participate
        };
        if (!initialParticipating || !factor) {
            return t -> false;
        }
        String id = generator.getId();
        double maxP = generator.getMaxP();
        double minT = minTargetP;
        double maxT = maxTargetP;
        return t -> AbstractLfGenerator.checkActivePowerControl(id, t * PerUnit.SB, maxP, minT, maxT, parameters, null);
    }
}
