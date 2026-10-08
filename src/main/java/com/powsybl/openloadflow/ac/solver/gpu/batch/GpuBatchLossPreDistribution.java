/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;

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
 * {@code dsAnchorShift} + the slack's own absorbed share in {@code dsSlackShift} — so the device
 * solve starts at the CPU's post-pre-distribution operating point and its DS gate measures the
 * identical residual.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuBatchLossPreDistribution {

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
     * @param genMovesOut when not null ({@code double[ngen]}), receives each generator's own move - what the
     *        forced-target-Q channel clamps at (GpuBatchForcedTargetQ); left at zero when nothing moves
     * @return {@code double[nds + 1]}: the participant shares (target/anchor deltas per DS row)
     *         followed by the slack bus's own absorbed share, or {@code null} when nothing can move
     *         (no participants, zero mismatch).
     */
    static double[] distribute(GpuAcData gpuData, double mismatch,
                               int ovBus, double ovFactorDelta, double ovBaseDelta,
                               double ovMaxDelta, double ovMinDelta, boolean[] islanded,
                               int[] lostGenSlots, double[] genMovesOut) {
        int nds = gpuData.dsPhiRow().length;
        if (nds == 0 || mismatch == 0) {
            return null;
        }
        boolean dirUp = mismatch > 0;
        // fs = the participants' collective share of the mismatch = 1 - rho, with rho the slack's
        // retained share. With every participant present that is just Σf. With some stranded in
        // another synchronous component it is NOT: OLF renormalizes over the survivors, so the slack's
        // share becomes rho/(Σ_active f + rho) and the survivors' collective share the complement.
        // Dropping the stranded factors WITHOUT renormalizing leaves their share sitting at the slack —
        // measured on LINE-3317-350 as 0.133 pu (13.3 MW) undistributed, which moved the whole
        // trajectory: the device then re-derived 13.481 where OLF re-derived 13.477, and the two
        // arms converged to different in-band points 5.7e-3 kV apart.
        double fs;
        if (gpuData.dsMode() == 1) {                        // REMAINING_MARGIN: direction-dependent margins
            double sumM = 0;
            for (int i = 0; i < nds; i++) {
                if (islanded != null && islanded[i]) {
                    continue;                                // in another synchronous component now
                }
                sumM += dirUp ? gpuData.dsMarginUp()[i] : gpuData.dsMarginDown()[i];
            }
            double slackM = dirUp ? gpuData.dsSlackMarginUp() : gpuData.dsSlackMarginDown();
            // Already the renormalized share: dropping the stranded margins IS the renormalization.
            fs = sumM + slackM > 0 ? sumM / (sumM + slackM) : 0.0;
        } else {
            double fsAll = 0;
            double fsActive = 0;
            for (int i = 0; i < nds; i++) {
                fsAll += gpuData.dsFactor()[i];
                if (islanded != null && islanded[i]) {
                    continue;                                // in another synchronous component now
                }
                fsActive += gpuData.dsFactor()[i];
            }
            double rho = 1.0 - fsAll;                        // the slack's own share, over the FULL set
            fs = fsActive == fsAll || fsActive + rho <= 0
                    ? fsActive
                    : fsActive / (fsActive + rho);
        }
        if (fs <= 0) {
            return null;
        }
        double rem = fs * mismatch;                          // the participants' collective share of the loss
        double[] cum = new double[nds];
        // PER-GENERATOR, exactly as the kernel and exactly as OLF: the seed and the device's own
        // water-fill have to be the same function, or the device re-derives from an anchor the seed
        // never produced. The bus-aggregate loop below stays for the cases the generator arrays do
        // not cover (a pack built without them).
        int ngen = gpuData.dgBus() == null ? 0 : gpuData.dgBus().length;
        if (ngen > 0) {
            double[] genCum = new double[ngen];
            boolean[] removed = new boolean[ngen];
            // normalizeParticipationFactors mutates the stored factor in place; mirror that, or the
            // seed takes a different floating-point path from the kernel it is meant to match.
            double[] fWork = new double[ngen];
            for (int g = 0; g < ngen; g++) {
                fWork[g] = genFactor(gpuData, dirUp, g);
            }
            // The bus that lost a generator is treated PER GENERATOR, exactly as the kernel does
            // since the outage carries the identity of what it removed: the lost generators are marked
            // out and the survivors take part like any others, with the outage's own move of the row
            // target carried by the anchor rather than by an aggregate participant. The aggregate
            // remains only for a bus whose loss has no identity to remove (a load-only outage).
            boolean ovPerGen = ovBus >= 0;
            for (int k = 0; lostGenSlots != null && k < lostGenSlots.length; k++) {
                removed[lostGenSlots[k]] = true;
                fWork[lostGenSlots[k]] = 0.0;
            }
            double ovCum = 0.0;
            for (int iter = 0; iter < ngen && Math.abs(rem) > P_RESIDUE_EPS; iter++) {
                double fsumActive = 0.0;
                for (int g = 0; g < ngen; g++) {
                    if (removed[g] || (gpuData.dgBus()[g] == ovBus && !ovPerGen)
                            || islanded != null && islanded[gpuData.dgBus()[g]]) {
                        continue;
                    }
                    fsumActive += fWork[g];
                }
                double ovF = 0.0;
                double ovMaxD = 0.0;
                double ovMinD = 0.0;
                boolean ovActive = false;
                if (ovBus >= 0 && !ovPerGen && (islanded == null || !islanded[ovBus])) {
                    ovMaxD = gpuData.dsMaxDelta()[ovBus] - ovMaxDelta;
                    ovMinD = gpuData.dsMinDelta()[ovBus] - ovMinDelta;
                    // The kernel does NOT take the factor delta off a REMAINING_MARGIN factor; mirror
                    // it rather than "improve" on it (that shape is CPU-routed upstream anyway, and
                    // the seed must be the same function as the fill it seeds).
                    ovF = gpuData.dsMode() == 1
                            ? Math.max(0.0, dirUp ? gpuData.dsMarginUp()[ovBus] : gpuData.dsMarginDown()[ovBus])
                            : gpuData.dsFactor()[ovBus] - ovFactorDelta;
                    ovActive = ovF > 0 && !(rem > 0 ? ovCum >= ovMaxD : ovCum <= ovMinD);
                    if (ovActive) {
                        fsumActive += ovF;
                    }
                }
                if (fsumActive <= 0.0) {
                    break;
                }
                double done = 0.0;
                for (int g = 0; g < ngen; g++) {
                    if (removed[g] || (gpuData.dgBus()[g] == ovBus && !ovPerGen)
                            || islanded != null && islanded[gpuData.dgBus()[g]]) {
                        continue;
                    }
                    fWork[g] /= fsumActive;
                    double newCum = genCum[g] + rem * fWork[g];
                    // OLF removes on a STRICT overshoot only.
                    if (rem > 0 && newCum > gpuData.dgMaxDelta()[g]) {
                        newCum = gpuData.dgMaxDelta()[g];
                        removed[g] = true;
                    } else if (rem < 0 && newCum < gpuData.dgMinDelta()[g]) {
                        newCum = gpuData.dgMinDelta()[g];
                        removed[g] = true;
                    }
                    if (newCum != genCum[g]) {
                        done += newCum - genCum[g];
                        genCum[g] = newCum;
                    }
                }
                if (ovActive) {
                    double newCum = Math.min(Math.max(ovCum + rem * (ovF / fsumActive), ovMinD), ovMaxD);
                    if (newCum != ovCum) {
                        done += newCum - ovCum;
                        ovCum = newCum;
                    }
                }
                if (System.getenv("OLF_GPU_SEED_TRACE") != null) {
                    // Pairs with OLF's OLF_DS_TRACE (DS_RUN / DS_SAT): same rounds, same saturations,
                    // same clamp values, or the seed is not the function it is meant to replay.
                    System.err.printf("SEED iter=%d fsumActive=%.12f rem=%.12f done=%.12f%n",
                            iter, fsumActive, rem, done);
                    for (int g = 0; g < ngen; g++) {
                        if (removed[g] && genCum[g] != 0) {
                            System.err.printf("  SEED_SAT gen=%d bus=%d cum=%.12f maxD=%.12f%n",
                                    g, gpuData.dgBus()[g], genCum[g], gpuData.dgMaxDelta()[g]);
                        }
                    }
                }
                rem -= done;
            }
            if (genMovesOut != null) {
                System.arraycopy(genCum, 0, genMovesOut, 0, ngen);
            }
            for (int i = 0; i < nds; i++) {
                if (islanded != null && islanded[i]) {
                    continue;
                }
                if (i == ovBus && !ovPerGen) {
                    cum[i] = ovCum;
                    continue;
                }
                double sum = 0;
                for (int g = gpuData.busGenStart()[i]; g < gpuData.busGenEnd()[i]; g++) {
                    sum += genCum[g];
                }
                cum[i] = sum;
            }
        } else {
        for (int iter = 0; iter < nds && Math.abs(rem) > P_RESIDUE_EPS; iter++) {
            double fsumActive = 0.0;                         // re-normalize over the still-unsaturated buses
            for (int i = 0; i < nds; i++) {
                double f = effFactor(gpuData, dirUp, i, ovBus, ovFactorDelta, islanded);
                double maxD = gpuData.dsMaxDelta()[i] - (ovBus == i ? ovMaxDelta : 0.0);
                double minD = gpuData.dsMinDelta()[i] - (ovBus == i ? ovMinDelta : 0.0);
                boolean sat = rem > 0 ? cum[i] >= maxD : cum[i] <= minD;
                if (!sat) {
                    fsumActive += f;
                }
            }
            if (fsumActive <= 0.0) {
                break;                                       // every participant saturated — residual stays at the slack
            }
            double done = 0.0;
            for (int i = 0; i < nds; i++) {
                double f = effFactor(gpuData, dirUp, i, ovBus, ovFactorDelta, islanded);
                double maxD = gpuData.dsMaxDelta()[i] - (ovBus == i ? ovMaxDelta : 0.0);
                double minD = gpuData.dsMinDelta()[i] - (ovBus == i ? ovMinDelta : 0.0);
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
        double[] out = new double[nds + 1];
        System.arraycopy(cum, 0, out, 0, nds);
        out[nds] = (1.0 - fs) / fs * absorbed;               // the slack's own share, pair-leg convention
        return out;
    }

    /** A generator's factor under the run's balance type — the margins are direction-dependent. */
    private static double genFactor(GpuAcData gpuData, boolean dirUp, int g) {
        return gpuData.dsMode() == 1
                ? (dirUp ? gpuData.dgMarginUp()[g] : gpuData.dgMarginDown()[g])
                : gpuData.dgFactor()[g];
    }

    private static double effFactor(GpuAcData gpuData, boolean dirUp, int i, int ovBus, double ovFactorDelta,
                                    boolean[] islanded) {
        if (islanded != null && islanded[i]) {
            return 0.0;                                      // excluded: it is not in this component any more
        }
        double f = gpuData.dsMode() == 1
                ? Math.max(0.0, dirUp ? gpuData.dsMarginUp()[i] : gpuData.dsMarginDown()[i])
                : gpuData.dsFactor()[i];
        if (ovBus == i) {
            f -= ovFactorDelta;
        }
        return f;
    }
}
