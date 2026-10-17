/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu.batch;

import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;

import java.util.ArrayList;
import java.util.List;

/**
 * Zero-impedance (coupler) clusters seen by a post-contingency solve: the coupler flows every solve starts from
 * (zero, as OLF's initStateVector), and which child of an opened coupler keeps the voltage target of its cluster
 * (OLF's re-election of the voltage-control main).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class ZeroImpedanceClusters {

    private ZeroImpedanceClusters() {
    }

    /** OLF starts EVERY solve with the zero-impedance branch flows at zero: AcSolverUtil.initStateVector
     *  has {@code case DUMMY_P, DUMMY_Q: x[v.getRow()] = 0}, and updateNetwork never writes them back
     *  (they have no network backing). So each post-contingency solve discards the coupler flows and
     *  rebuilds them in its first Newton step - on network.xiidm an entry residual of 14.988 pu, the same
     *  for every contingency, against 0.16 when they are carried. The device starts the same way, so both
     *  arms take the same first step from the same state. */
    static double[] zeroZiFlows(GpuAcData gpuData, double[] baseState) {
        int[] zi = gpuData.ziIdx();
        if (zi == null || zi.length == 0) {
            return baseState;
        }
        double[] x0 = baseState.clone();
        for (int k = 0; k + 10 < zi.length; k += 11) {
            if (zi[k + 4] >= 0) {
                x0[zi[k + 4]] = 0;                           // DUMMY_P column
            }
            if (zi[k + 5] >= 0) {
                x0[zi[k + 5]] = 0;                           // DUMMY_Q column
            }
        }
        return x0;
    }

    /**
     * Whether the CHILD of an opened coupler keeps a voltage target of its own. OLF re-elects the voltage-control
     * MAIN of every zero-impedance network a contingency splits (LfZeroImpedanceNetwork.updateVoltageControlMergeStatus:
     * the highest target, then the first controlled-bus id): that bus alone keeps BUS_TARGET_V, every other section
     * of the network follows it through the couplers' ZERO_V. A child left ALONE is its own main. A child the
     * contingency leaves coupled to other live sections - OLF's re-elected tree re-couples it through a coupler that
     * was out of the base tree - is a DEPENDENT unless it wins that election, and its row keeps the reactive
     * equation the opened coupler's DUMMY_Q row hands back.
     *
     * <p>The voltage identity used to go on every child. network.xiidm busbar HFU opens HFT_0's three tree couplers
     * (HHI, HHK, HHW) while HHO/HHQ/HHS, out of the base tree, re-couple HFT_2/3/4: identities on HFT_3 and HFT_4 on
     * top of the ZERO_V links over-determined the voltages and left the couplers' reactive flows FREE - 1e11 pu after
     * the first solve, the rounding of which moved the slack mismatch by +-0.058 MW from one run to the next
     * (1.041 MW distributed in one run where OLF distributes nothing).
     */
    static boolean holdsClusterVoltage(String contingencyId, LfBus child, List<LfBranch> outagedZi,
                                       java.util.Map<Integer, Integer> ziElemByNum, int[] ziIdx, int[] frozenRows) {
        java.util.Set<Integer> frozen = new java.util.HashSet<>();
        for (int r : frozenRows) {
            frozen.add(r);
        }
        java.util.Set<LfBranch> opened = new java.util.HashSet<>(outagedZi);
        List<LfBus> cluster = new ArrayList<>(List.of(child));
        java.util.Set<LfBus> seen = new java.util.HashSet<>(cluster);
        for (int k = 0; k < cluster.size(); k++) {
            LfBus b = cluster.get(k);
            for (LfBranch br : b.getBranches()) {
                Integer ze = ziElemByNum.get(br.getNum());
                if (ze == null || opened.contains(br) || br.isDisabled()) {
                    continue;
                }
                boolean fromBus1 = br.getBus1() == b;
                LfBus other = fromBus1 ? br.getBus2() : br.getBus1();
                int otherVRow = ziIdx[ze * 11 + (fromBus1 ? 7 : 6)];
                if (other == null || other.isDisabled() || frozen.contains(otherVRow) || !seen.add(other)) {
                    continue;
                }
                cluster.add(other);
            }
        }
        if (cluster.size() == 1) {
            return true;
        }
        if (System.getenv("OLF_GPU_ZITREE_TRACE") != null) {
            System.err.println("ZITREE_CLUSTER ctg=" + contingencyId + " child=" + child.getId() + " cluster=" + cluster.stream()
                    .map(bb -> bb.getId() + bb.getGeneratorVoltageControl().filter(vc -> vc.getControlledBus() == bb)
                            .map(vc -> "(vt=" + vc.getTargetValue() + "," + vc.getMergeStatus() + ",enabled=" + bb.isGeneratorVoltageControlEnabled() + ")")
                            .orElse(""))
                    .toList());
        }
        return cluster.stream()
                .map(bb -> bb.getGeneratorVoltageControl().filter(vc -> vc.getControlledBus() == bb).orElse(null))
                .filter(java.util.Objects::nonNull)
                .min(java.util.Comparator.<GeneratorVoltageControl>comparingDouble(vc -> -vc.getTargetValue())
                        .thenComparing(vc -> vc.getControlledBus().getId()))
                .map(vc -> vc.getControlledBus() == child)
                .orElse(true);
    }
}
