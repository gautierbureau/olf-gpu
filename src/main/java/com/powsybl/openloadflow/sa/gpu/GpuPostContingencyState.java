/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.ac.AcLoadFlowParameters;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;

import java.util.List;
import java.util.Map;

/**
 * The device's post-contingency state given to the network: bus voltages from the state vector, the network update
 * and OLF's realistic-voltage check on it.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuPostContingencyState {

    private GpuPostContingencyState() {
    }

    /** OLF_GPU_BUS_ARRAY_CHECK=1: after each array update, run the variable walk too and require every bus's V,
     *  angle and the realistic-voltage verdict to be bit-identical. */
    static final boolean BUS_ARRAY_CHECK = System.getenv("OLF_GPU_BUS_ARRAY_CHECK") != null;

    /**
     * Bus voltages and angles straight from the device state, and the realistic-voltage check, in one loop over the
     * buses. The device state is in the BASE row layout and every enabled bus keeps its base V and angle variables,
     * so {@code state[vRowOfBus[b]]} is exactly the value the injection put in the state vector for that bus; a
     * disabled bus has no variable in the contingency's index, and {@code AcSolverUtil.updateNetwork} never touched
     * it either. Replaces a walk over OLF's ~13k variable objects per contingency (1.9 s cold on rte6515). Falls
     * back to {@link #updateNetworkAndCheckRealistic} unless the base system is exactly one BUS_V and one BUS_PHI
     * per bus.
     */
    static boolean updateBusesFromState(double[] state, PrecomputedBatch precomputed, LfNetwork network,
                                                AcLoadFlowParameters parameters, GpuSaRunState rs) {
        Map<Variable<AcVariableType>, Integer> baseRows = precomputed.baseVariableRows();
        if (rs.busRowsFor != baseRows) {
            rs.busRowsFor = baseRows;
            int nb = network.getBuses().size();
            int[] vRow = new int[nb];
            int[] phiRow = new int[nb];
            java.util.Arrays.fill(vRow, -1);
            java.util.Arrays.fill(phiRow, -1);
            boolean usable = true;
            for (Map.Entry<Variable<AcVariableType>, Integer> e : baseRows.entrySet()) {
                Variable<AcVariableType> v = e.getKey();
                if (v.getType() == AcVariableType.BUS_V && v.getElementNum() < nb) {
                    vRow[v.getElementNum()] = e.getValue();
                } else if (v.getType() == AcVariableType.BUS_PHI && v.getElementNum() < nb) {
                    phiRow[v.getElementNum()] = e.getValue();
                } else if (v.getType() != AcVariableType.DUMMY_P && v.getType() != AcVariableType.DUMMY_Q) {
                    // another kind the network update WRITES (BRANCH_RHO1/ALPHA1, SHUNT_B, the asymmetrical
                    // ones): keep the exact walk. A zero-impedance coupler's DUMMY_P/DUMMY_Q is equation-only -
                    // AcSolverUtil.updateNetwork does nothing with it - so it must not send a network with
                    // couplers down the slow path: on network.xiidm every one of 24,379 contingencies walked
                    // all ~46k variable objects twice (updateNetwork + isStateUnrealistic, 9.7 s of a 207 s run).
                    usable = false;
                }
            }
            for (int b = 0; b < nb && usable; b++) {
                usable = vRow[b] >= 0 && phiRow[b] >= 0;
            }
            rs.vRowOfBus = vRow;
            rs.phiRowOfBus = phiRow;
            rs.busRowsUsable = usable;
        }
        if (!rs.busRowsUsable || state == null) {
            return updateNetworkAndCheckRealistic(precomputed.equationSystem(), network, parameters, rs);
        }
        if (rs.realisticBandApplies == null || rs.realisticBandNetwork != network) {
            double minNominalV = parameters.getMinNominalVoltageRealisticVoltageCheck();
            boolean[] applies = new boolean[network.getBuses().size()];
            for (LfBus b : network.getBuses()) {
                applies[b.getNum()] = !b.isFictitious() && b.getNominalV() >= minNominalV;
            }
            rs.realisticBandApplies = applies;
            rs.realisticBandNetwork = network;
        }
        boolean[] applies = rs.realisticBandApplies;
        double minV = parameters.getMinRealisticVoltage();
        double maxV = parameters.getMaxRealisticVoltage();
        boolean trace = System.getenv("OLF_GPU_UNREALISTIC_TRACE") != null;
        int[] vRow = rs.vRowOfBus;
        int[] phiRow = rs.phiRowOfBus;
        List<LfBus> buses = network.getBuses();
        boolean unrealistic = false;
        for (int num = 0; num < vRow.length; num++) {
            LfBus bus = buses.get(num);
            if (bus.isDisabled()) {
                continue;
            }
            double value = state[vRow[num]];
            bus.setV(value);
            bus.setAngle(state[phiRow[num]]);
            if ((!unrealistic || trace) && (value < minV || value > maxV) && applies[num]) {
                unrealistic = true;
                if (trace) {
                    System.err.printf("UNREALISTIC bus=%s v=%.9f nominalV=%.3f band=[%.3f, %.3f]%n",
                            bus.getId(), value, bus.getNominalV(), minV, maxV);
                }
            }
        }
        if (BUS_ARRAY_CHECK) {
            double[] v = new double[vRow.length];
            double[] a = new double[vRow.length];
            for (int num = 0; num < vRow.length; num++) {
                v[num] = buses.get(num).getV();
                a[num] = buses.get(num).getAngle();
            }
            boolean walked = updateNetworkAndCheckRealistic(precomputed.equationSystem(), network, parameters, rs);
            rs.busArrayChecked++;
            String diff = walked != unrealistic ? "realistic verdict " + unrealistic + " vs " + walked : null;
            for (int num = 0; num < vRow.length && diff == null; num++) {
                LfBus bus = buses.get(num);
                if (Double.doubleToLongBits(v[num]) != Double.doubleToLongBits(bus.getV())
                        || Double.doubleToLongBits(a[num]) != Double.doubleToLongBits(bus.getAngle())) {
                    diff = "bus " + bus.getId() + " v " + v[num] + " vs " + bus.getV() + " angle " + a[num] + " vs " + bus.getAngle();
                }
            }
            if (diff != null) {
                rs.busArrayDiffering++;
                if (rs.busArrayFirstDiff == null) {
                    rs.busArrayFirstDiff = diff;
                }
            }
        }
        return unrealistic;
    }

    /**
     * {@code AcSolverUtil.updateNetwork} and {@link #isStateUnrealistic} in ONE walk over the variables. Both
     * walked the ~13k variables of rte6515 once per contingency (2.9 s of the GPU arm, nearly all of it memory
     * traffic), the second one with two bus lookups per variable. Only BUS_V / BUS_PHI are handled here —
     * the only kinds the batch solves — so any other kind in the system falls back to the two original
     * calls. Which buses the realistic-voltage band applies to is a per-network constant, cached per run.
     */
    static boolean updateNetworkAndCheckRealistic(EquationSystem<AcVariableType, AcEquationType> es,
                                                          LfNetwork network, AcLoadFlowParameters parameters,
                                                          GpuSaRunState rs) {
        List<Variable<AcVariableType>> vars = es.getIndex().getSortedVariablesToFind();
        for (Variable<AcVariableType> v : vars) {
            if (v.getType() != AcVariableType.BUS_V && v.getType() != AcVariableType.BUS_PHI) {
                AcSolverUtil.updateNetwork(network, es);
                return isStateUnrealistic(es, network, parameters);
            }
        }
        if (rs.realisticBandApplies == null || rs.realisticBandNetwork != network) {
            double minNominalV = parameters.getMinNominalVoltageRealisticVoltageCheck();
            boolean[] applies = new boolean[network.getBuses().size()];
            for (LfBus b : network.getBuses()) {
                applies[b.getNum()] = !b.isFictitious() && b.getNominalV() >= minNominalV;
            }
            rs.realisticBandApplies = applies;
            rs.realisticBandNetwork = network;
        }
        boolean[] applies = rs.realisticBandApplies;
        double minV = parameters.getMinRealisticVoltage();
        double maxV = parameters.getMaxRealisticVoltage();
        boolean trace = System.getenv("OLF_GPU_UNREALISTIC_TRACE") != null;
        com.powsybl.openloadflow.equations.StateVector sv = es.getStateVector();
        boolean unrealistic = false;
        for (Variable<AcVariableType> v : vars) {
            double value = sv.get(v.getRow());
            LfBus bus = network.getBus(v.getElementNum());
            if (v.getType() == AcVariableType.BUS_PHI) {
                bus.setAngle(value);
                continue;
            }
            bus.setV(value);
            if (!unrealistic || trace) {
                if ((value < minV || value > maxV) && applies[v.getElementNum()]) {
                    unrealistic = true;
                    if (trace) {
                        System.err.printf("UNREALISTIC bus=%s v=%.9f nominalV=%.3f band=[%.3f, %.3f]%n",
                                bus.getId(), value, bus.getNominalV(), minV, maxV);
                    }
                }
            }
        }
        return unrealistic;
    }

    /**
     * OLF's realistic-voltage check, ported from {@code AcloadFlowEngine.isStateUnrealistic}: every
     * non-fictitious {@code BUS_V} variable whose value leaves
     * {@code [minRealisticVoltage, maxRealisticVoltage]} counts, but only on buses whose nominal
     * voltage reaches {@code minNominalVoltageRealisticVoltageCheck}. One such bus makes the whole
     * run {@code UNREALISTIC_STATE}, which the security analysis maps to {@code FAILED}.
     *
     * <p>Walks the variables and reads the STATE VECTOR, exactly as OLF does, rather than reading
     * {@code bus.getV()} — same numbers here, but the point of a port is not to paraphrase.
     *
     * <p>{@code OLF_GPU_UNREALISTIC_TRACE} names the offending buses.
     */
    static boolean isStateUnrealistic(EquationSystem<AcVariableType, AcEquationType> es,
                                              LfNetwork network, AcLoadFlowParameters parameters) {
        double minNominalV = parameters.getMinNominalVoltageRealisticVoltageCheck();
        boolean trace = System.getenv("OLF_GPU_UNREALISTIC_TRACE") != null;
        boolean unrealistic = false;
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() != AcVariableType.BUS_V || network.getBus(v.getElementNum()).isFictitious()) {
                continue;
            }
            double value = es.getStateVector().get(v.getRow());
            if ((value < parameters.getMinRealisticVoltage() || value > parameters.getMaxRealisticVoltage())
                    && network.getBus(v.getElementNum()).getNominalV() >= minNominalV) {
                unrealistic = true;
                if (!trace) {
                    return true;
                }
                System.err.printf("UNREALISTIC bus=%s v=%.9f nominalV=%.3f band=[%.3f, %.3f]%n",
                        network.getBus(v.getElementNum()).getId(), value,
                        network.getBus(v.getElementNum()).getNominalV(),
                        parameters.getMinRealisticVoltage(), parameters.getMaxRealisticVoltage());
            }
        }
        return unrealistic;
    }

    /** OLF's DISTR_Q target, reproduced from {@code AcTargetVector} (which keeps it private):
     *  {@code (qPct_i - 1) * targetQ_i + qPct_i * sum_{j != i} targetQ_j}. Diagnostic only. */
    static double distrQTarget(LfNetwork network, LfBus controllerBus) {
        double qPct = controllerBus.getRemoteControlReactivePercent();
        double target = (qPct - 1) * controllerBus.getTargetQ();
        var members = controllerBus.getGeneratorVoltageControl().isPresent()
                ? controllerBus.getGeneratorVoltageControl().orElseThrow().getMergedControllerElements()
                : controllerBus.getGeneratorReactivePowerControl().orElseThrow().getControllerBuses();
        for (LfBus other : members) {
            if (other != controllerBus) {
                target += qPct * other.getTargetQ();
            }
        }
        return target;
    }
}
