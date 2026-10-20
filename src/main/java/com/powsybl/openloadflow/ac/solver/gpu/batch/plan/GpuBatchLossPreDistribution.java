/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch.plan;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuDistributedSlackFactorMode;

/**
 * Host-side replay of OLF's {@code DefaultContingencyActivePowerLossDistribution}: with distributed
 * slack on, the CPU distributes the contingency's SCHEDULED active power loss (lost generation minus
 * lost load, {@code LfContingency.getActivePowerLoss()}) over the participating generators ONCE,
 * BEFORE the post-contingency solve — its DS outer loop then only sees the loss-nonlinearity
 * residual (typically far inside the {@code slackBusPMaxMismatch} band, so it never redistributes).
 * The device DS pass instead water-fills the full measured mismatch, which lands on a different
 * point INSIDE the band (measured 4.2e-4 kV on an ieee14 load loss).
 *
 * <p>{@link #distribute} ports the {@code distributeSlackBatch} kernel's water-fill verbatim
 * (same factor/base/headroom model, per-scenario gen-outage reweight, REMAINING_MARGIN margins,
 * saturation re-normalization rounds, {@code P_RESIDUE_EPS}); the result feeds the device as the
 * SAME three coupled pieces the strategy-pair re-anchoring uses — participant target overrides +
 * {@code distributedSlackBusAnchorShift} + the slack's own absorbed share in {@code slackBusAnchorShift} — so the device
 * solve starts at the CPU's post-pre-distribution operating point and its DS gate measures the
 * identical residual.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuBatchLossPreDistribution {

    /** OLF ActivePowerDistribution.P_RESIDUE_EPS (pu) — the water-fill's inner stop. */
    private static final double P_RESIDUE_EPS = 1e-5;

    private GpuBatchLossPreDistribution() {
    }

    /**
     * Distribute {@code mismatch} (pu, CPU sign: lost generation − lost load) exactly like the
     * device water-fill would, from the scenario's effective base anchors.
     *
     * <p>{@code islanded} marks the DS participants a network-FRAGMENTING contingency has put in
     * another synchronous component. OLF's own distribution runs on
     * {@code network.getSynchronousNetworks().getFirst()}, so those generators take no share and do
     * not count towards the factor sum — exactly what excluding them here reproduces. Before this
     * existed the whole pre-distribution was skipped for such a contingency and the device's DS loop
     * water-filled the mismatch during the solve instead, landing at a different point inside the
     * slack band: measured 1.07 MW of branch-flow difference against the CPU on rte6515, enough to
     * flip apparent-power violations.
     *
     * @param genMovesOut when not null ({@code double[nDistributedSlackGenerators]}), receives each generator's own move - what the
     *        forced-target-Q channel clamps at (ForceTargetQInReactiveLimits); left at zero when nothing moves
     * @return {@code double[nDistributedSlackBuses + 1]}: the participant shares (target/anchor deltas per DS row)
     *         followed by the slack bus's own absorbed share, or {@code null} when nothing can move
     *         (no participants, zero mismatch).
     */
    public static double[] distribute(GpuAcData gpuData, double mismatch,
                               int overriddenBus, double overrideFactorDelta, double overrideBaseTargetPDelta,
                               double overrideMaxDelta, double overrideMinDelta, boolean[] islanded,
                               int[] lostGenSlots, double[] genMovesOut) {
        int nDistributedSlackBuses = gpuData.distributedSlack().busPhiRow().length;
        if (nDistributedSlackBuses == 0 || mismatch == 0) {
            return null;
        }
        boolean dirUp = mismatch > 0;
        int nDistributedSlackGenerators = gpuData.distributedSlack().generatorBus() == null ? 0 : gpuData.distributedSlack().generatorBus().length;
        // The slack bus's own generators in the generator pack (distributedSlackGeneratorBus -1): they are water-filled with the others,
        // clamped at their own limits, so the whole mismatch goes into the list and the slack's share is theirs.
        boolean slackInPack = slackInPack(gpuData);
        // fs = the participants' collective share of the mismatch = 1 - rho, with rho the slack's
        // retained share. With every participant present that is just Σf. With some stranded in
        // another synchronous component it is NOT: OLF renormalizes over the survivors, so the slack's
        // share becomes rho/(Σ_active f + rho) and the survivors' collective share the complement.
        // Dropping the stranded factors WITHOUT renormalizing leaves their share sitting at the slack —
        // measured on LINE-3317-350 as 0.133 pu (13.3 MW) undistributed, which moved the whole
        // trajectory: the device then re-derived 13.481 where OLF re-derived 13.477, and the two
        // arms converged to different in-band points 5.7e-3 kV apart.
        double fs;
        if (gpuData.distributedSlack().factorMode() == GpuDistributedSlackFactorMode.REMAINING_MARGIN) {                        // REMAINING_MARGIN: direction-dependent margins
            double sumM = 0;
            for (int i = 0; i < nDistributedSlackBuses; i++) {
                if (islanded != null && islanded[i]) {
                    continue;                                // in another synchronous component now
                }
                sumM += dirUp ? gpuData.distributedSlack().busMarginUp()[i] : gpuData.distributedSlack().busMarginDown()[i];
            }
            double slackM = dirUp ? gpuData.distributedSlack().slackMarginUp() : gpuData.distributedSlack().slackMarginDown();
            // Already the renormalized share: dropping the stranded margins IS the renormalization.
            fs = sumM + slackM > 0 ? sumM / (sumM + slackM) : 0.0;
        } else {
            double fsAll = 0;
            double fsActive = 0;
            for (int i = 0; i < nDistributedSlackBuses; i++) {
                fsAll += gpuData.distributedSlack().busFactor()[i];
                if (islanded != null && islanded[i]) {
                    continue;                                // in another synchronous component now
                }
                // a lost participating generator leaves the participation like a stranded one
                fsActive += gpuData.distributedSlack().busFactor()[i] - (i == overriddenBus ? overrideFactorDelta : 0.0);
            }
            double rho = 1.0 - fsAll;                        // the slack's own share, over the FULL set
            fs = fsActive == fsAll || fsActive + rho <= 0
                    ? fsActive
                    : fsActive / (fsActive + rho);
            if (fs <= 0 && rho > 0 && !slackInPack) {
                // Only the slack bus's generators are left to participate: OLF puts the whole mismatch on
                // them. The device carries it in the slack's retained share; the result reports it.
                double[] out = new double[nDistributedSlackBuses + 1];
                out[nDistributedSlackBuses] = mismatch;
                return out;
            }
        }
        if (slackInPack) {
            fs = 1.0;
        }
        if (fs <= 0) {
            return null;
        }
        double rem = fs * mismatch;                          // the participants' collective share of the loss
        double[] cum = new double[nDistributedSlackBuses];
        double slackShare = 0;
        // PER-GENERATOR, exactly as the kernel and exactly as OLF: the seed and the device's own
        // water-fill have to be the same function, or the device re-derives from an anchor the seed
        // never produced. The bus-aggregate loop below stays for the cases the generator arrays do
        // not cover (a pack built without them).
        if (nDistributedSlackGenerators > 0) {
            double[] distributedSlackGeneratorMove = new double[nDistributedSlackGenerators];
            boolean[] removed = new boolean[nDistributedSlackGenerators];
            // normalizeParticipationFactors mutates the stored factor in place; mirror that, or the
            // seed takes a different floating-point path from the kernel it is meant to match.
            double[] fWork = new double[nDistributedSlackGenerators];
            for (int g = 0; g < nDistributedSlackGenerators; g++) {
                fWork[g] = genFactor(gpuData, dirUp, g);
            }
            // The bus that lost a generator is treated PER GENERATOR, exactly as the kernel does
            // since the outage carries the identity of what it removed: the lost generators are marked
            // out and the survivors take part like any others, with the outage's own move of the row
            // target carried by the anchor rather than by an aggregate participant. The aggregate
            // remains only for a bus whose loss has no identity to remove (a load-only outage).
            boolean overriddenBusPerGenerator = overriddenBus >= 0;
            for (int k = 0; lostGenSlots != null && k < lostGenSlots.length; k++) {
                removed[lostGenSlots[k]] = true;
                fWork[lostGenSlots[k]] = 0.0;
            }
            double overriddenBusMove = 0.0;
            for (int iter = 0; iter < nDistributedSlackGenerators && Math.abs(rem) > P_RESIDUE_EPS; iter++) {
                double fsumActive = 0.0;
                for (int g = 0; g < nDistributedSlackGenerators; g++) {
                    if (removed[g] || excluded(gpuData.distributedSlack().generatorBus()[g], overriddenBus, overriddenBusPerGenerator, islanded)) {
                        continue;
                    }
                    fsumActive += fWork[g];
                }
                double overriddenBusFactor = 0.0;
                double overriddenBusMaxDelta = 0.0;
                double overriddenBusMinDelta = 0.0;
                boolean overriddenBusActive = false;
                if (overriddenBus >= 0 && !overriddenBusPerGenerator && (islanded == null || !islanded[overriddenBus])) {
                    overriddenBusMaxDelta = gpuData.distributedSlack().busMaxDelta()[overriddenBus] - overrideMaxDelta;
                    overriddenBusMinDelta = gpuData.distributedSlack().busMinDelta()[overriddenBus] - overrideMinDelta;
                    // The kernel does NOT take the factor delta off a REMAINING_MARGIN factor; mirror
                    // it rather than "improve" on it (that shape is CPU-routed upstream anyway, and
                    // the seed must be the same function as the fill it seeds).
                    overriddenBusFactor = gpuData.distributedSlack().factorMode() == GpuDistributedSlackFactorMode.REMAINING_MARGIN
                            ? Math.max(0.0, dirUp ? gpuData.distributedSlack().busMarginUp()[overriddenBus] : gpuData.distributedSlack().busMarginDown()[overriddenBus])
                            : gpuData.distributedSlack().busFactor()[overriddenBus] - overrideFactorDelta;
                    overriddenBusActive = overriddenBusFactor > 0 && !(rem > 0 ? overriddenBusMove >= overriddenBusMaxDelta : overriddenBusMove <= overriddenBusMinDelta);
                    if (overriddenBusActive) {
                        fsumActive += overriddenBusFactor;
                    }
                }
                if (fsumActive <= 0.0) {
                    break;
                }
                double done = 0.0;
                for (int g = 0; g < nDistributedSlackGenerators; g++) {
                    if (removed[g] || excluded(gpuData.distributedSlack().generatorBus()[g], overriddenBus, overriddenBusPerGenerator, islanded)) {
                        continue;
                    }
                    fWork[g] /= fsumActive;
                    double newCum = distributedSlackGeneratorMove[g] + rem * fWork[g];
                    // OLF removes on a STRICT overshoot only.
                    if (rem > 0 && newCum > gpuData.distributedSlack().generatorMaxDelta()[g]) {
                        newCum = gpuData.distributedSlack().generatorMaxDelta()[g];
                        removed[g] = true;
                    } else if (rem < 0 && newCum < gpuData.distributedSlack().generatorMinDelta()[g]) {
                        newCum = gpuData.distributedSlack().generatorMinDelta()[g];
                        removed[g] = true;
                    }
                    if (newCum != distributedSlackGeneratorMove[g]) {
                        done += newCum - distributedSlackGeneratorMove[g];
                        distributedSlackGeneratorMove[g] = newCum;
                    }
                }
                if (overriddenBusActive) {
                    double newCum = Math.min(Math.max(overriddenBusMove + rem * (overriddenBusFactor / fsumActive), overriddenBusMinDelta), overriddenBusMaxDelta);
                    if (newCum != overriddenBusMove) {
                        done += newCum - overriddenBusMove;
                        overriddenBusMove = newCum;
                    }
                }
                rem -= done;
            }
            if (genMovesOut != null) {
                System.arraycopy(distributedSlackGeneratorMove, 0, genMovesOut, 0, nDistributedSlackGenerators);
            }
            for (int g = 0; g < nDistributedSlackGenerators; g++) {
                if (gpuData.distributedSlack().generatorBus()[g] < 0) {
                    slackShare += distributedSlackGeneratorMove[g];
                }
            }
            for (int i = 0; i < nDistributedSlackBuses; i++) {
                if (islanded != null && islanded[i]) {
                    continue;
                }
                if (i == overriddenBus && !overriddenBusPerGenerator) {
                    cum[i] = overriddenBusMove;
                    continue;
                }
                double sum = 0;
                for (int g = gpuData.distributedSlack().busGeneratorStart()[i]; g < gpuData.distributedSlack().busGeneratorEnd()[i]; g++) {
                    sum += distributedSlackGeneratorMove[g];
                }
                cum[i] = sum;
            }
        } else {
            for (int iter = 0; iter < nDistributedSlackBuses && Math.abs(rem) > P_RESIDUE_EPS; iter++) {
                double fsumActive = 0.0;                         // re-normalize over the still-unsaturated buses
                for (int i = 0; i < nDistributedSlackBuses; i++) {
                    double f = effFactor(gpuData, dirUp, i, overriddenBus, overrideFactorDelta, islanded);
                    double maxD = gpuData.distributedSlack().busMaxDelta()[i] - (overriddenBus == i ? overrideMaxDelta : 0.0);
                    double minD = gpuData.distributedSlack().busMinDelta()[i] - (overriddenBus == i ? overrideMinDelta : 0.0);
                    boolean sat = rem > 0 ? cum[i] >= maxD : cum[i] <= minD;
                    if (!sat) {
                        fsumActive += f;
                    }
                }
                if (fsumActive <= 0.0) {
                    break;                                       // every participant saturated — residual stays at the slack
                }
                double done = 0.0;
                for (int i = 0; i < nDistributedSlackBuses; i++) {
                    double f = effFactor(gpuData, dirUp, i, overriddenBus, overrideFactorDelta, islanded);
                    double maxD = gpuData.distributedSlack().busMaxDelta()[i] - (overriddenBus == i ? overrideMaxDelta : 0.0);
                    double minD = gpuData.distributedSlack().busMinDelta()[i] - (overriddenBus == i ? overrideMinDelta : 0.0);
                    boolean sat = rem > 0 ? cum[i] >= maxD : cum[i] <= minD;
                    if (sat) {
                        continue;
                    }
                    double newCum = Math.min(Math.max(cum[i] + rem * (f / fsumActive), minD), maxD);
                    if (newCum != cum[i]) {
                        done += newCum - cum[i];
                        cum[i] = newCum;
                    }
                }
                rem -= done;
            }
        }
        double absorbed = 0;
        for (double c : cum) {
            absorbed += c;
        }
        double[] out = new double[nDistributedSlackBuses + 1];
        System.arraycopy(cum, 0, out, 0, nDistributedSlackBuses);
        // the slack's own share: its generators' moves, or with them outside the list the pair-leg convention
        out[nDistributedSlackBuses] = slackInPack ? slackShare : (1.0 - fs) / fs * absorbed;
        return out;
    }

    /** Whether the slack bus's own participating generators are in the generator pack (DS row -1). */
    public static boolean slackInPack(GpuAcData gpuData) {
        int[] distributedSlackGeneratorBus = gpuData.distributedSlack().generatorBus();
        for (int g = 0; distributedSlackGeneratorBus != null && g < distributedSlackGeneratorBus.length; g++) {
            if (distributedSlackGeneratorBus[g] < 0) {
                return true;
            }
        }
        return false;
    }

    /** Whether generator pack entry on DS row {@code bus} sits out: the outaged bus's aggregate, or islanded. A slack
     *  generator (row -1) is neither. */
    private static boolean excluded(int bus, int overriddenBus, boolean overriddenBusPerGenerator, boolean[] islanded) {
        return bus >= 0 && (bus == overriddenBus && !overriddenBusPerGenerator || islanded != null && islanded[bus]);
    }

    /** A generator's factor under the run's balance type — the margins are direction-dependent. */
    private static double genFactor(GpuAcData gpuData, boolean dirUp, int g) {
        return gpuData.distributedSlack().factorMode() == GpuDistributedSlackFactorMode.REMAINING_MARGIN
                ? (dirUp ? gpuData.distributedSlack().generatorMarginUp()[g] : gpuData.distributedSlack().generatorMarginDown()[g])
                : gpuData.distributedSlack().generatorFactor()[g];
    }

    private static double effFactor(GpuAcData gpuData, boolean dirUp, int i, int overriddenBus, double overrideFactorDelta,
                                    boolean[] islanded) {
        if (islanded != null && islanded[i]) {
            return 0.0;                                      // excluded: it is not in this component any more
        }
        double f = gpuData.distributedSlack().factorMode() == GpuDistributedSlackFactorMode.REMAINING_MARGIN
                ? Math.max(0.0, dirUp ? gpuData.distributedSlack().busMarginUp()[i] : gpuData.distributedSlack().busMarginDown()[i])
                : gpuData.distributedSlack().busFactor()[i];
        if (overriddenBus == i) {
            f -= overrideFactorDelta;
        }
        return f;
    }
}
