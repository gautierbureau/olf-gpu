/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.util.ActivePowerDistribution;
import com.powsybl.openloadflow.network.util.ParticipatingElement;

import java.util.*;

/**
 * OLF's generation slack distribution ({@code ActivePowerDistribution} +
 * {@code GenerationActivePowerDistributionStep}) replayed on ARRAYS, so a modified target column
 * can be built for a contingency without mutating the network and restoring it afterwards.
 *
 * <p>Semantics reproduced step for step, including the one that is easy to miss: OLF first RESETS
 * every participating generator (and the reference generator) to its INITIAL target P — the
 * original network's value, not the base-converged one — and adds what that undoes to the
 * mismatch. What gets water-filled is therefore the contingency mismatch plus everything the
 * base load flow had already distributed, from the original anchor, with the limits applied to
 * that total. Then: factors by participation type, normalized at each iteration; a generator
 * that would cross a limit (or zero) is clamped and dropped; until the residue is below
 * {@code P_RESIDUE_EPS} or nobody can move.
 *
 * <p>The model is built once at BASE (after the pre-contingency distribution) and evaluated per
 * contingency with the buses the contingency islands excluded — {@code filterParticipatingBuses}
 * over the remaining buses. Load-based balance types are not modelled (the caller routes them
 * to the stock path).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class DcSlackDistributionModel {

    private final LoadFlowParameters.BalanceType balanceType;
    private final boolean useActiveLimits;

    // one entry per generator of a participating bus, in OLF's iteration order (buses, then generators)
    private final int[] genBus;
    private final double[] genInitialTargetP;
    private final double[] genBaseTargetP;
    private final double[] genMinTargetP;
    private final double[] genMaxTargetP;
    private final double[] genMinP;
    private final double[] genMaxP;
    private final double[] genDroop;
    private final double[] genParticipationFactor;
    private final boolean[] genParticipating;
    private final int referenceGenerator;                    // index, or -1

    private DcSlackDistributionModel(LoadFlowParameters.BalanceType balanceType, boolean useActiveLimits, int[] genBus,
                                     double[] genInitialTargetP, double[] genBaseTargetP, double[] genMinTargetP, double[] genMaxTargetP,
                                     double[] genMinP, double[] genMaxP, double[] genDroop, double[] genParticipationFactor,
                                     boolean[] genParticipating, int referenceGenerator) {
        this.balanceType = balanceType;
        this.useActiveLimits = useActiveLimits;
        this.genBus = genBus;
        this.genInitialTargetP = genInitialTargetP;
        this.genBaseTargetP = genBaseTargetP;
        this.genMinTargetP = genMinTargetP;
        this.genMaxTargetP = genMaxTargetP;
        this.genMinP = genMinP;
        this.genMaxP = genMaxP;
        this.genDroop = genDroop;
        this.genParticipationFactor = genParticipationFactor;
        this.genParticipating = genParticipating;
        this.referenceGenerator = referenceGenerator;
    }

    static boolean supports(LoadFlowParameters.BalanceType balanceType) {
        return switch (balanceType) {
            case PROPORTIONAL_TO_GENERATION_P_MAX, PROPORTIONAL_TO_GENERATION_P,
                 PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR, PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN -> true;
            default -> false;
        };
    }

    /** Snapshot the base network. Call after the pre-contingency solve, before any contingency. */
    static DcSlackDistributionModel build(LfNetwork network, LoadFlowParameters.BalanceType balanceType, boolean useActiveLimits) {
        if (!supports(balanceType)) {
            throw new IllegalArgumentException("balance type not modelled: " + balanceType);
        }
        LfGenerator reference = network.getSynchronousNetworks().getFirst().getReferenceGenerator();
        List<LfGenerator> gens = new ArrayList<>();
        int referenceIndex = -1;
        for (LfBus bus : network.getBuses()) {
            if (!bus.isParticipating() || bus.isFictitious()) {
                continue;                                    // filterParticipatingBuses (disabled is per contingency)
            }
            for (LfGenerator g : bus.getGenerators()) {
                if (g == reference) {
                    referenceIndex = gens.size();
                }
                gens.add(g);
            }
        }
        if (reference != null && referenceIndex < 0) {
            // the reference generator sits on a non-participating bus: OLF still resets it
            referenceIndex = gens.size();
            gens.add(reference);
        }
        int m = gens.size();
        int[] genBus = new int[m];
        double[] initial = new double[m];
        double[] base = new double[m];
        double[] minT = new double[m];
        double[] maxT = new double[m];
        double[] minP = new double[m];
        double[] maxP = new double[m];
        double[] droop = new double[m];
        double[] pf = new double[m];
        boolean[] participating = new boolean[m];
        for (int i = 0; i < m; i++) {
            LfGenerator g = gens.get(i);
            genBus[i] = g.getBus().getNum();
            initial[i] = g.getInitialTargetP();
            base[i] = g.getTargetP();
            minT[i] = g.getMinTargetP();
            maxT[i] = g.getMaxTargetP();
            minP[i] = g.getMinP();
            maxP[i] = g.getMaxP();
            droop[i] = g.getDroop();
            pf[i] = g.getParticipationFactor();
            participating[i] = g.isParticipating() && g.getBus().isParticipating() && !g.getBus().isFictitious();
        }
        return new DcSlackDistributionModel(balanceType, useActiveLimits, genBus, initial, base, minT, maxT, minP, maxP, droop, pf,
                participating, referenceIndex);
    }

    private boolean stepParticipating(int i) {
        if (!genParticipating[i]) {
            return false;
        }
        return switch (balanceType) {
            case PROPORTIONAL_TO_GENERATION_P_MAX -> genDroop[i] != 0;
            case PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR -> genParticipationFactor[i] > 0;
            default -> true;
        };
    }

    private double factor(int i, double targetP, double mismatch) {
        return switch (balanceType) {
            case PROPORTIONAL_TO_GENERATION_P_MAX -> genMaxP[i] / genDroop[i];
            case PROPORTIONAL_TO_GENERATION_P -> Math.abs(targetP);
            case PROPORTIONAL_TO_GENERATION_PARTICIPATION_FACTOR -> genParticipationFactor[i];
            case PROPORTIONAL_TO_GENERATION_REMAINING_MARGIN -> {
                double minP = genMinP[i];
                double maxP = genMaxP[i];
                if (targetP < 0) {
                    maxP = Math.min(maxP, 0);
                } else {
                    minP = Math.max(minP, 0);
                }
                yield mismatch > 0. ? Math.max(0.0, maxP - targetP) : Math.max(0.0, targetP - minP);
            }
            default -> throw new IllegalStateException();
        };
    }

    /**
     * Distribute {@code mismatch} over the generators of the buses not in {@code excludedBuses},
     * exactly like {@code DcLoadFlowEngine.distributeSlack} on the base network with those buses
     * disabled, and return the resulting change of every bus's generation target P (per unit)
     * relative to the base — the delta the target column needs.
     */
    Map<Integer, Double> distribute(Set<Integer> excludedBuses, double mismatch) {
        int m = genBus.length;
        double[] target = genBaseTargetP.clone();

        // resetToInitialState: the reference generator first (the interface default), then every participating
        // generator of a participating (remaining) bus, each back to its initial target P — two separately
        // accumulated amounts added at the end, then added to the mismatch, as ActivePowerDistribution.run does
        double refPrevious = 0.;
        if (referenceGenerator >= 0) {
            refPrevious -= genInitialTargetP[referenceGenerator] - target[referenceGenerator];
            target[referenceGenerator] = genInitialTargetP[referenceGenerator];
        }
        double stepPrevious = 0;
        for (int i = 0; i < m; i++) {
            if (excludedBuses.contains(genBus[i]) || !genParticipating[i]) {
                continue;
            }
            stepPrevious -= genInitialTargetP[i] - target[i];
            target[i] = genInitialTargetP[i];
        }
        double remaining = mismatch + (stepPrevious + refPrevious);

        // getParticipatingElements, with factors evaluated on the RESET targets
        int[] elements = new int[m];
        double[] factors = new double[m];
        int count = 0;
        for (int i = 0; i < m; i++) {
            if (excludedBuses.contains(genBus[i]) || !stepParticipating(i)) {
                continue;
            }
            double f = factor(i, target[i], remaining);
            if (f != 0) {
                elements[count] = i;
                factors[count] = f;
                count++;
            }
        }

        // the water-filling loop
        int iteration = 0;
        while (count > 0 && Math.abs(remaining) > ActivePowerDistribution.P_RESIDUE_EPS) {
            double norm = 0;
            for (int k = 0; k < count; k++) {
                norm += factors[k];
            }
            if (!(norm > 0.0)) {
                break;
            }
            double done = 0;
            for (int k = 0; k < count;) {
                int i = elements[k];
                double f = factors[k] / norm;                 // normalizeParticipationFactors
                double minTargetP = useActiveLimits ? genMinTargetP[i] : -Double.MAX_VALUE;
                double maxTargetP = useActiveLimits ? genMaxTargetP[i] : Double.MAX_VALUE;
                double targetP = target[i];
                if (targetP < 0) {
                    maxTargetP = Math.min(maxTargetP, 0);
                } else {
                    minTargetP = Math.max(minTargetP, 0);
                }
                double newTargetP = targetP + remaining * f;
                boolean drop = false;
                if (remaining > 0 && newTargetP > maxTargetP) {
                    newTargetP = maxTargetP;
                    drop = true;
                } else if (remaining < 0 && newTargetP < minTargetP) {
                    newTargetP = minTargetP;
                    drop = true;
                }
                if (newTargetP != targetP) {
                    target[i] = newTargetP;
                    done += newTargetP - targetP;
                }
                if (drop) {
                    // OLF's iterator remove: the list keeps its order
                    System.arraycopy(elements, k + 1, elements, k, count - k - 1);
                    System.arraycopy(factors, k + 1, factors, k, count - k - 1);
                    count--;
                } else {
                    k++;
                }
            }
            remaining -= done;
            iteration++;
            if (iteration > 1000) {
                break;                                       // OLF has no cap; a runaway would spin there too
            }
        }

        Map<Integer, Double> busDelta = new HashMap<>();
        for (int i = 0; i < m; i++) {
            double d = target[i] - genBaseTargetP[i];
            if (d != 0) {
                busDelta.merge(genBus[i], d, Double::sum);
            }
        }
        return busDelta;
    }

    /** For tests: the participating elements OLF would build at base, to check the model's view. */
    static List<ParticipatingElement> olfParticipatingElements(LfNetwork network, LoadFlowParameters.BalanceType balanceType, boolean useActiveLimits) {
        ActivePowerDistribution.Step step = ActivePowerDistribution.getStep(balanceType, false, useActiveLimits);
        return step.getParticipatingElements(ActivePowerDistribution.filterParticipatingBuses(network.getBuses()), Double.NaN);
    }
}
