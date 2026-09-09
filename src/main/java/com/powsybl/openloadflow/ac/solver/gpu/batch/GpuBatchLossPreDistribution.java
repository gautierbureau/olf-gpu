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
     * @return {@code double[nds + 1]}: the participant shares (target/anchor deltas per DS row)
     *         followed by the slack bus's own absorbed share, or {@code null} when nothing can move
     *         (no participants, zero mismatch).
     */
    static double[] distribute(GpuAcData gpuData, double mismatch,
                               int ovBus, double ovFactorDelta, double ovBaseDelta,
                               double ovMaxDelta, double ovMinDelta, boolean[] islanded) {
        int nds = gpuData.dsPhiRow().length;
        if (nds == 0 || mismatch == 0) {
            return null;
        }
        boolean dirUp = mismatch > 0;
        // fs = the GLOBAL participant factor sum = (1 - rho) with rho the slack's retained share —
        // the same constant the retain kernel uses (a per-scenario participant reweight does not
        // change it; see the kernel's fsGlobal comment).
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
            fs = sumM + slackM > 0 ? sumM / (sumM + slackM) : 0.0;
        } else {
            fs = 0;
            for (int i = 0; i < nds; i++) {
                if (islanded != null && islanded[i]) {
                    continue;                                // in another synchronous component now
                }
                fs += gpuData.dsFactor()[i];
            }
        }
        if (fs <= 0) {
            return null;
        }
        double rem = fs * mismatch;                          // the participants' collective share of the loss
        double[] cum = new double[nds];
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
        double absorbed = 0;
        for (double c : cum) {
            absorbed += c;
        }
        double[] out = new double[nds + 1];
        System.arraycopy(cum, 0, out, 0, nds);
        out[nds] = (1.0 - fs) / fs * absorbed;               // the slack's own share, pair-leg convention
        return out;
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
