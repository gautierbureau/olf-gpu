/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.network.LfGenerator;

import java.util.Objects;

/**
 * The distributed-slack parameterization the device DS pass models, mirroring OLF's
 * {@code GenerationActivePowerDistributionStep}: the participation factor per balance type and
 * whether active power limits bound the per-bus absorption.
 *
 * <p>Supported balance types are the STATIC-factor generation ones — {@code PROPORTIONAL_TO_GENERATION_P_MAX}
 * ({@code maxP/droop}), {@code PROPORTIONAL_TO_GENERATION_P} ({@code |targetP|}, evaluated at the anchor:
 * OLF's {@code resetToInitialState} runs before factor computation, so factors are anchor-static per run)
 * and {@code PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR}. {@code REMAINING_MARGIN} (mismatch-sign-
 * dependent factors) and the load-based types (a different participating-element set) are NOT modeled —
 * {@code GpuSaRunScope} routes such runs to the CPU.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public record GpuDsParams(LoadFlowParameters.BalanceType balanceType, boolean useActiveLimits) {

    /** OLF's defaults: PROPORTIONAL_TO_GENERATION_P_MAX with active limits. */
    public static final GpuDsParams DEFAULT =
            new GpuDsParams(LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX, true);

    public GpuDsParams {
        Objects.requireNonNull(balanceType);
    }

    /** True if the device DS pass models this balance type (static generation factors). */
    public static boolean supports(LoadFlowParameters.BalanceType balanceType) {
        return switch (balanceType) {
            case PROPORTIONAL_TO_GENERATION_P_MAX, PROPORTIONAL_TO_GENERATION_P,
                 PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR -> true;
            default -> false;
        };
    }

    /** The generator's raw participation factor under this balance type, exactly OLF's
     *  {@code getParticipationFactor} + {@code isParticipating} combined: 0 = does not participate. */
    public double factor(LfGenerator g) {
        if (!g.isParticipating()) {
            return 0;
        }
        return switch (balanceType) {
            case PROPORTIONAL_TO_GENERATION_P_MAX -> g.getDroop() != 0 ? g.getMaxP() / g.getDroop() : 0;
            case PROPORTIONAL_TO_GENERATION_P -> Math.abs(g.getTargetP());
            case PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR ->
                    g.getParticipationFactor() > 0 ? g.getParticipationFactor() : 0;
            default -> throw new IllegalStateException("unsupported device DS balance type: " + balanceType);
        };
    }
}
