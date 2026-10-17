/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfLoad;
import com.powsybl.openloadflow.network.LfNetwork;

import java.util.Map;

/**
 * The device's distributed-slack decisions given back to the network, as OLF's ActivePowerDistribution leaves
 * them: generator targets from their anchor plus their share, or the loads moved for a load-based balance type.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class DistributedSlackWriteBack {

    private DistributedSlackWriteBack() {
    }

    /** DistributedSlack, exactly OLF's ActivePowerDistribution: every participating generator ends at
     *  its ANCHOR target ({@code initialTargetP}) plus the cumulative share the water-fill gave it
     *  ({@code genCum}, per generator — each saturates at its own limits); the slack bus's own
     *  generators, which have no device row, split the slack's retained share by factor — the value
     *  captured at the last check that actually DISTRIBUTED, not the sub-tolerance one a STABLE check
     *  measures. A solve whose DistributedSlack phase never fired leaves the generators alone.
     *  Returns what it distributed (pu, see GpuNewtonRaphson#lastDistributedActivePower), NaN when nothing. */
    static double writeBack(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> equationSystem, long handle,
                            GpuDsParams dsParams,
                            Map<LfLoad, Double> loadAnchorTargetP,
                            double[] deviceTargets, double[] genCum, double[] retain, GpuAcData data) {
        double lastDistributedActivePower = Double.NaN;
        if (deviceTargets == null || data == null) {
            return lastDistributedActivePower;
        }
        LfBus slackBus = null;
        if (data.dsSlackPhiRow() >= 0) {
            for (LfBus bus : network.getBuses()) {
                var v = equationSystem.getVariableSet().getVariable(bus.getNum(), AcVariableType.BUS_PHI);
                if (v != null && v.getRow() == data.dsSlackPhiRow()) {
                    slackBus = bus;
                    break;
                }
            }
            if (slackBus == null) {
                // The COUPLER-TREE RELOCATION moved that bus's P balance onto its parent coupler's DUMMY_P row,
                // which no BUS_PHI variable owns — the pack ships the relocated row (dsSlackPhiRow = pRow). Left
                // unresolved, the slack's retained share was never written back and OLF measured the device's
                // converged state against stale targets: on rte6515's full busbar scope the device's own norm was
                // 9.45e-06 (OLF's own final 9.50e-06) while OLF re-evaluated 1.67 and called the base
                // MAX_ITERATION_REACHED, which took the whole security analysis with it.
                slackBus = relocatedChildOfPRow(network, data, data.dsSlackPhiRow());
            }
        }
        if (data.dsPhiRow().length == 0) {
            // slack-only participation: the whole distribution is the slack's absorbed total
            double[] absorbed = GpuAcNewtonSolver.readBatchSlackAbsorbed(handle);
            if (slackBus != null && absorbed != null && absorbed.length > 0 && absorbed[0] != 0) {
                setAnchoredShare(slackBus, absorbed[0], dsParams);
                lastDistributedActivePower = absorbed[0];
            }
            return lastDistributedActivePower;
        }
        boolean fired = retain != null && retain.length > 0 && !Double.isNaN(retain[0]);
        if (!fired) {
            return lastDistributedActivePower;
        }
        Map<String, LfGenerator> genById = new java.util.HashMap<>();
        for (LfBus bus : network.getBuses()) {
            for (LfGenerator g : bus.getGenerators()) {
                genById.put(g.getId(), g);
            }
        }
        double distributed = retain[0];                     // the slack's retained share + the participants' below
        boolean slackInPack = false;                        // the slack's own generators listed (DS row -1)
        if (genCum != null && data.dgGenId() != null && genCum.length >= data.dgGenId().length) {
            for (int g = 0; g < data.dgGenId().length; g++) {
                if (data.dgBus()[g] < 0) {
                    slackInPack = true;                     // its move is the retained share already counted
                } else {
                    distributed += genCum[g];
                }
                LfGenerator gen = genById.get(data.dgGenId()[g]);
                if (gen != null) {
                    gen.setTargetP(dsParams.anchorTargetP(gen) + genCum[g]);
                }
            }
        } else {
            for (int i = 0; i < data.dsPhiRow().length; i++) {        // bus-level fallback: split by factor
                int row = data.dsPhiRow()[i];
                if (row < 0 || row >= deviceTargets.length) {
                    continue;
                }
                LfBus bus = data.dsBusNum() != null && i < data.dsBusNum().length ? network.getBus(data.dsBusNum()[i]) : null;
                distributed += deviceTargets[row] - data.dsBaseTargetP()[i];
                if (bus != null) {
                    if (dsParams.isLoadBased()) {
                        moveLoads(bus, deviceTargets[row] - data.dsBaseTargetP()[i], loadAnchorTargetP);
                    } else {
                        setAnchoredShare(bus, deviceTargets[row] - data.dsBaseTargetP()[i], dsParams);
                    }
                }
            }
        }
        lastDistributedActivePower = distributed;
        if (slackBus != null && retain[0] != 0 && !slackInPack) {
            if (dsParams.isLoadBased()) {
                moveLoads(slackBus, retain[0], loadAnchorTargetP);
            } else {
                setAnchoredShare(slackBus, retain[0], dsParams);
            }
        }
    
        return lastDistributedActivePower;
    }

    /** Each load's target P now: the anchor a load-based distribution moves the loads from. */
    static Map<LfLoad, Double> captureLoadAnchors(LfNetwork network) {
        Map<LfLoad, Double> anchors = new java.util.IdentityHashMap<>();
        for (LfBus bus : network.getBuses()) {
            for (LfLoad load : bus.getLoads()) {
                anchors.put(load, load.getTargetP());
            }
        }
        return anchors;
    }

    /** A load-based distribution's bus-level amount on the bus's loads, as OLF's LoadActivePowerDistributionStep
     *  moves them: from where they stood when the distribution started (the step is incremental, it has no initial
     *  target to go back to), each in proportion to its variable active power. A bus injection rising by
     *  {@code delta} is its loads consuming {@code delta} less. Moving only generators, as for the generation-based
     *  types, left OLF's load targets where they were, and OLF measured the converged state against them: an ieee14
     *  base whose slack bus carries a participating load came back MAX_ITERATION_REACHED. */
    private static void moveLoads(LfBus bus, double delta,
                                  Map<LfLoad, Double> loadAnchorTargetP) {
        double factorSum = 0;
        for (LfLoad load : bus.getLoads()) {
            factorSum += load.getAbsVariableTargetP();
        }
        if (factorSum <= 0) {
            return;
        }
        for (LfLoad load : bus.getLoads()) {
            double f = load.getAbsVariableTargetP();
            if (f != 0) {
                load.setTargetP(loadAnchorTargetP.getOrDefault(load, load.getTargetP()) - delta * f / factorSum);
            }
        }
    }

    /** Give a bus's participating generators {@code anchor + delta × f / Σf} — OLF's split of a
     *  bus-level amount from the anchor, with the run's balance-type factors. */
    private static void setAnchoredShare(LfBus bus, double delta, GpuDsParams dsParams) {
        double factorSum = 0;
        for (LfGenerator g : bus.getGenerators()) {
            factorSum += dsParams.factor(g);
        }
        if (factorSum <= 0) {
            return;
        }
        for (LfGenerator g : bus.getGenerators()) {
            double f = dsParams.factor(g);
            if (f != 0) {
                g.setTargetP(dsParams.anchorTargetP(g) + delta * f / factorSum);
            }
        }
    }

    /** The bus whose P balance the relocation moved onto the coupler {@code DUMMY_P} row {@code pRow}. */
    private static LfBus relocatedChildOfPRow(LfNetwork network, GpuAcData data, int pRow) {
        int[] zi = data.ziIdx();
        int[] nums = data.ziBranchNums();
        if (zi == null || nums == null) {
            return null;
        }
        for (int e = 0; e < zi.length / 11 && e < nums.length; e++) {
            int o = e * 11;
            int childSide = (zi[o + 10] >> 1) & 3;
            if (childSide == 0 || zi[o + 4] != pRow || nums[e] < 0) {
                continue;
            }
            LfBranch br = network.getBranch(nums[e]);
            return childSide == 1 ? br.getBus1() : br.getBus2();
        }
        return null;
    }
}
