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
import com.powsybl.openloadflow.util.gpu.GpuTrace;

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
    static double[] zeroZeroImpedanceBranchFlows(GpuAcData gpuData, double[] baseState) {
        int[] zeroImpedanceBranchIndices = gpuData.zeroImpedanceBranch().indices();
        if (zeroImpedanceBranchIndices == null || zeroImpedanceBranchIndices.length == 0) {
            return baseState;
        }
        double[] x0 = baseState.clone();
        for (int k = 0; k + 10 < zeroImpedanceBranchIndices.length; k += 11) {
            if (zeroImpedanceBranchIndices[k + 4] >= 0) {
                x0[zeroImpedanceBranchIndices[k + 4]] = 0;                           // DUMMY_P column
            }
            if (zeroImpedanceBranchIndices[k + 5] >= 0) {
                x0[zeroImpedanceBranchIndices[k + 5]] = 0;                           // DUMMY_Q column
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
    static boolean holdsClusterVoltage(String contingencyId, LfBus child, List<LfBranch> outagedZeroImpedanceBranch,
                                       java.util.Map<Integer, Integer> zeroImpedanceElementByNum, int[] zeroImpedanceBranchIndices, int[] frozenRows) {
        java.util.Set<Integer> frozen = new java.util.HashSet<>();
        for (int r : frozenRows) {
            frozen.add(r);
        }
        java.util.Set<LfBranch> opened = new java.util.HashSet<>(outagedZeroImpedanceBranch);
        List<LfBus> cluster = new ArrayList<>(List.of(child));
        java.util.Set<LfBus> seen = new java.util.HashSet<>(cluster);
        for (int k = 0; k < cluster.size(); k++) {
            LfBus b = cluster.get(k);
            for (LfBranch br : b.getBranches()) {
                Integer ze = zeroImpedanceElementByNum.get(br.getNum());
                if (ze == null || opened.contains(br) || br.isDisabled()) {
                    continue;
                }
                boolean fromBus1 = br.getBus1() == b;
                LfBus other = fromBus1 ? br.getBus2() : br.getBus1();
                int otherVRow = zeroImpedanceBranchIndices[ze * 11 + (fromBus1 ? 7 : 6)];
                if (other == null || other.isDisabled() || frozen.contains(otherVRow) || !seen.add(other)) {
                    continue;
                }
                cluster.add(other);
            }
        }
        if (cluster.size() == 1) {
            return true;
        }
        if (GpuTrace.ZITREE_TRACE.isOn()) {
            System.err.println("ZITREE_CLUSTER ctg=" + contingencyId + " child=" + child.getId() + " cluster=" + cluster.stream()
                    .map(clusterBus -> clusterBus.getId() + clusterBus.getGeneratorVoltageControl().filter(voltageControl -> voltageControl.getControlledBus() == clusterBus)
                            .map(voltageControl -> "(vt=" + voltageControl.getTargetValue() + "," + voltageControl.getMergeStatus()
                                    + ",enabled=" + clusterBus.isGeneratorVoltageControlEnabled() + ")")
                            .orElse(""))
                    .toList());
        }
        return cluster.stream()
                .map(clusterBus -> clusterBus.getGeneratorVoltageControl().filter(voltageControl -> voltageControl.getControlledBus() == clusterBus).orElse(null))
                .filter(java.util.Objects::nonNull)
                .min(java.util.Comparator.<GeneratorVoltageControl>comparingDouble(voltageControl -> -voltageControl.getTargetValue())
                        .thenComparing(voltageControl -> voltageControl.getControlledBus().getId()))
                .map(voltageControl -> voltageControl.getControlledBus() == child)
                .orElse(true);
    }

    /** The zero-impedance elements OUT of the base spanning tree (zeroImpedanceBranchIndices slot 10 is 0), in element order. */
    static int[] baseNonTreeElements(GpuAcData gpuData) {
        int[] zIdx = gpuData.zeroImpedanceBranch().indices();
        int nz = gpuData.zeroImpedanceBranch().branchNum() == null ? 0 : gpuData.zeroImpedanceBranch().branchNum().length;
        int cnt = 0;
        for (int zb = 0; zb < nz; zb++) {
            if (zIdx[zb * 11 + 10] == 0) {
                cnt++;
            }
        }
        int[] baseNonSpanningTreeBranches = new int[cnt];
        cnt = 0;
        for (int zb = 0; zb < nz; zb++) {
            if (zIdx[zb * 11 + 10] == 0) {
                baseNonSpanningTreeBranches[cnt++] = zb;
            }
        }
        return baseNonSpanningTreeBranches;
    }
}
