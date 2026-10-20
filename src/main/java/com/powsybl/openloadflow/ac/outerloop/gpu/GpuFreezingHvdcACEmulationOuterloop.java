/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.outerloop.gpu;

import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchOverrides;
import com.powsybl.openloadflow.ac.solver.gpu.codes.GpuAcEmulationStatus;

import java.util.List;

import static com.powsybl.openloadflow.ac.outerloop.gpu.GpuHvdcAcEmulationLimitsOuterLoop.HVDC_TRACE_ROW;

/**
 * DEVICE port of OLF's {@code FreezingHvdcACEmulationOuterloop}: initialize() FREEZES every AC-emulation HVDC at
 * its scenario's warm-start flows BEFORE the first solve (VSC injections pinned, coupling off); its first check,
 * which UNFREEZES them all, runs in the kernel (hvdc_ac_emulation.cu) on the statuses
 * {@link GpuHvdcAcEmulationLimitsOuterLoop} installs.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class GpuFreezingHvdcACEmulationOuterloop {

    private GpuFreezingHvdcACEmulationOuterloop() {
    }

    /** OLF switchToFrozenState at the given (warm-start) evaluation point: pMax-clamped when the
     *  linear flows overpass the limits, else pinned at the current flows. */
    private static double[] frozenDeltas(GpuHvdcAcEmulationLimitsOuterLoop.Controller c, double p1, double p2) {
        if (p1 > c.pMax12()) {
            return GpuHvdcAcEmulationLimitsOuterLoop.saturationDeltas(c, true);
        } else if (p2 > c.pMax21()) {
            return GpuHvdcAcEmulationLimitsOuterLoop.saturationDeltas(c, false);
        }
        return new double[] {-p1, -p2};
    }

    /**
     * Pre-solve freeze (OLF FreezingHvdcACEmulation.initialize): every live hvdc frozen at its scenario's WARM-START
     * flows — hv disabled bits set, converter-bus targets pinned, on the batch's own override list. Returns the
     * per-(scenario, hvdc) status matrix seeded GpuAcEmulationStatus.FROZEN (or GpuAcEmulationStatus.LINEAR_MODE where the freeze cannot apply), and records in
     * {@code appliedDelta1}/{@code appliedDelta2} the converter-row delta each freeze APPLIED (relative to the base-status
     * delta), which the in-kernel loops need to keep applying transitions incrementally.
     *
     * <p>A scenario's warm start is {@code x0PerScenario}'s slice when given; otherwise the base state with the
     * scenario's seeds written over it ({@code seedIdx} flat {@code s * n + row}, the start state the device builds
     * from the same seeds) — applied to one shared copy and put back after, so a chunk never materializes
     * {@code chunkSize * n} doubles for it.
     *
     * @param rowModePerScenario the per-scenario row modes (null = base layout everywhere): a converter bus whose
     *  parent coupler the scenario opens takes its P target on its OWN row (see GpuHvdcAcEmulationLimitsOuterLoop.Controller.bus1WriteRow).
     */
    public static int[][] applyFreeze(int chunkSize, int realCount, int n, List<GpuHvdcAcEmulationLimitsOuterLoop.Controller> controllers,
                               double[] baseState, double[] x0PerScenario, int[] seedIdx, double[] seedVal,
                               boolean[][] disabledMask, double[] baseTargets, GpuBatchOverrides overrides,
                               double[] appliedDelta1, double[] appliedDelta2, int[][] rowModePerScenario) {
        int[][] status = new int[chunkSize][controllers.size()];
        double[] warm = baseState.clone();
        int[][] seedsOf = x0PerScenario == null && seedIdx != null ? seedsByScenario(seedIdx, realCount, n) : null;
        for (int s = 0; s < realCount; s++) {
            int[] rm = rowModePerScenario == null ? null : rowModePerScenario[s];
            if (x0PerScenario != null) {
                System.arraycopy(x0PerScenario, s * n, warm, 0, n);
            } else if (seedsOf != null) {
                for (int k : seedsOf[s]) {
                    warm[seedIdx[k] - s * n] = seedVal[k];
                }
            }
            for (int h = 0; h < controllers.size(); h++) {
                GpuHvdcAcEmulationLimitsOuterLoop.Controller c = controllers.get(h);
                if (HVDC_TRACE_ROW != -1) {
                    System.err.printf("HVDC_CTL s=%d h=%d elem=%d baseStatus=%d baseD1=%.9f baseD2=%.9f disabled=%b p1warm=%.9f p2warm=%.9f pMax12=%.6f baseT1=%.9f%n",
                            s, h, c.hvdcElementIndex(), c.baseStatus(), c.baseD1(), c.baseD2(), disabledMask[s][c.hvdcElementIndex()],
                            c.p1At(warm), c.p2At(warm), c.pMax12(), c.bus1PRow() >= 0 ? baseTargets[c.bus1PRow()] : Double.NaN);
                }
                if (disabledMask[s][c.hvdcElementIndex()]) {
                    continue;                                // outaged hvdc — stays as the contingency left it
                }
                double[] d = frozenDeltas(c, c.p1At(warm), c.p2At(warm));
                if (HVDC_TRACE_ROW == -2 || HVDC_TRACE_ROW >= 0 && (c.bus1PhiRow() == HVDC_TRACE_ROW || c.bus2PhiRow() == HVDC_TRACE_ROW)) {
                    // OLF_GPU_HVDC_TRACE=<phiRow>: what this scenario's freeze fixed the hvdc at - the
                    // flows read at the warm-start state and the deltas that go into the two converter
                    // rows. OLF freezes each contingency at its own restored base angles; a scenario
                    // whose warm start is not the base freezes at different flows and nothing else
                    // in the residual will say so (the frozen P IS the equation).
                    System.err.printf("HVDC_FREEZE s=%d h=%d row1=%d row2=%d warmIsBase=%b p1=%.9f p2=%.9f d1=%.9f d2=%.9f"
                            + " baseD1=%.9f baseD2=%.9f pMax12=%.6f pMax21=%.6f%n", s, h, c.bus1PhiRow(), c.bus2PhiRow(),
                            x0PerScenario == null, c.p1At(warm), c.p2At(warm), d[0], d[1], c.baseD1(), c.baseD2(),
                            c.pMax12(), c.pMax21());
                }
                if (appliedDelta1 != null) {
                    appliedDelta1[s * controllers.size() + h] = d[0] - c.baseD1();
                    appliedDelta2[s * controllers.size() + h] = d[1] - c.baseD2();
                }
                status[s][h] = GpuAcEmulationStatus.FROZEN;
                disabledMask[s][c.hvdcElementIndex()] = true;
                // base frame: baseTargets (or the row's existing override — e.g. a lost load on the
                // converter bus) already carry the base injections
                if (c.bus1PActive()) {
                    overrides.merge(s * n + c.bus1WriteRow(rm), baseTargets[c.bus1WriteRow(rm)], c.baseD1(), d[0]);
                }
                if (c.bus2PActive()) {
                    overrides.merge(s * n + c.bus2WriteRow(rm), baseTargets[c.bus2WriteRow(rm)], c.baseD2(), d[1]);
                }
            }
            if (seedsOf != null) {
                for (int k : seedsOf[s]) {                   // back to the base state for the next scenario
                    int row = seedIdx[k] - s * n;
                    warm[row] = baseState[row];
                }
            }
        }
        return status;
    }

    /** The positions in {@code seedIdx} of each scenario's seeds. */
    private static int[][] seedsByScenario(int[] seedIdx, int realCount, int n) {
        int[] count = new int[realCount];
        for (int flat : seedIdx) {
            count[flat / n]++;
        }
        int[][] seedsOf = new int[realCount][];
        for (int s = 0; s < realCount; s++) {
            seedsOf[s] = new int[count[s]];
            count[s] = 0;
        }
        for (int k = 0; k < seedIdx.length; k++) {
            int s = seedIdx[k] / n;
            seedsOf[s][count[s]++] = k;
        }
        return seedsOf;
    }
}
