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

    /** True if the device DS pass models this balance type: the static-factor generation types and the
     *  load-based types (per-bus factors = the variable load share, unbounded absorption — at BUS level the
     *  target update is identical to the generation case, so the same water-fill kernel serves both). */
    public static boolean supports(LoadFlowParameters.BalanceType balanceType) {
        return switch (balanceType) {
            case PROPORTIONAL_TO_GENERATION_P_MAX, PROPORTIONAL_TO_GENERATION_P,
                 PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR,
                 PROPORTIONAL_TO_LOAD, PROPORTIONAL_TO_CONFORM_LOAD,
                 PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN -> true;
        };
    }

    /** True for REMAINING_MARGIN: factors are the direction-dependent minP/maxP margins at the anchor,
     *  computed IN-KERNEL from the packed per-bus margins (see {@code busMargins}). */
    public boolean isRemainingMargin() {
        return balanceType == LoadFlowParameters.BalanceType.PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN;
    }

    /** REMAINING_MARGIN per-bus margins at the anchor, with OLF's no-sign-crossing rule on minP/maxP
     *  (NOT min/maxTargetP — those are the water-fill ROOMS, which may be plausibility-tightened):
     *  {@code [marginUp, marginDown]} = how much the bus's participating generation can rise / fall. */
    public double[] busMargins(com.powsybl.openloadflow.network.LfBus bus) {
        double up = 0;
        double down = 0;
        if (bus.isParticipating()) {
            for (LfGenerator g : bus.getGenerators()) {
                if (g.isParticipating()) {
                    double tp = g.getTargetP();
                    double minP = g.getMinP();
                    double maxP = g.getMaxP();
                    if (tp < 0) {
                        maxP = Math.min(maxP, 0);
                    } else {
                        minP = Math.max(minP, 0);
                    }
                    up += Math.max(0.0, maxP - tp);
                    down += Math.max(0.0, tp - minP);
                }
            }
        }
        return new double[] {up, down};
    }

    /** True for the load-based balance types (participants are LOADS, not generators). */
    public boolean isLoadBased() {
        return balanceType == LoadFlowParameters.BalanceType.PROPORTIONAL_TO_LOAD
                || balanceType == LoadFlowParameters.BalanceType.PROPORTIONAL_TO_CONFORM_LOAD;
    }

    /** The bus's raw participation factor: the sum of its participating generators' factors, or for the
     *  load-based types the sum of its loads' variable share ({@code absVariableTargetP} — the LfNetwork
     *  loader computed it for plain vs conform per the run's own balance type). Mirrors OLF's
     *  {@code getParticipatingElements} aggregated at bus level. */
    public double busFactor(com.powsybl.openloadflow.network.LfBus bus) {
        if (!bus.isParticipating()) {                        // countriesToBalance filter
            return 0;
        }
        double f = 0;
        if (isLoadBased()) {
            for (com.powsybl.openloadflow.network.LfLoad load : bus.getLoads()) {
                f += load.getAbsVariableTargetP();
            }
        } else {
            for (LfGenerator g : bus.getGenerators()) {
                f += factor(g);
            }
        }
        return f;
    }

    /** The generator's raw participation factor under this balance type, exactly OLF's
     *  {@code getParticipationFactor} + {@code isParticipating} combined: 0 = does not participate. */
    public double factor(LfGenerator g) {
        if (isLoadBased() || !g.isParticipating()) {
            return 0;                                        // load-based: generators never participate
        }
        return switch (balanceType) {
            case PROPORTIONAL_TO_GENERATION_P_MAX -> g.getDroop() != 0 ? g.getMaxP() / g.getDroop() : 0;
            case PROPORTIONAL_TO_GENERATION_P -> Math.abs(g.getTargetP());
            case PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR ->
                    g.getParticipationFactor() > 0 ? g.getParticipationFactor() : 0;
            // REMAINING_MARGIN factors are direction-dependent and live in the margins (busMargins /
            // in-kernel); here the value is only a PARTICIPATION INDICATOR (pack membership, slack-only
            // detection, lost-participation checks — the gen-outage reweight is CPU-gated under RM).
            case PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN -> 1;
            default -> throw new IllegalStateException("unsupported device DS balance type: " + balanceType);
        };
    }
}
