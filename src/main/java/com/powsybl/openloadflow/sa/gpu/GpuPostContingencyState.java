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
import com.powsybl.openloadflow.util.gpu.GpuTrace;

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
                                                AcLoadFlowParameters parameters, GpuSaRunState run) {
        Map<Variable<AcVariableType>, Integer> baseRows = precomputed.baseVariableRows();
        if (run.busRowsFor != baseRows) {
            run.busRowsFor = baseRows;
            int nBuses = network.getBuses().size();
            int[] vRow = new int[nBuses];
            int[] phiRow = new int[nBuses];
            java.util.Arrays.fill(vRow, -1);
            java.util.Arrays.fill(phiRow, -1);
            boolean usable = true;
            for (Map.Entry<Variable<AcVariableType>, Integer> e : baseRows.entrySet()) {
                Variable<AcVariableType> v = e.getKey();
                if (v.getType() == AcVariableType.BUS_V && v.getElementNum() < nBuses) {
                    vRow[v.getElementNum()] = e.getValue();
                } else if (v.getType() == AcVariableType.BUS_PHI && v.getElementNum() < nBuses) {
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
            for (int b = 0; b < nBuses && usable; b++) {
                usable = vRow[b] >= 0 && phiRow[b] >= 0;
            }
            run.vRowOfBus = vRow;
            run.phiRowOfBus = phiRow;
            run.busRowsUsable = usable;
        }
        if (!run.busRowsUsable || state == null) {
            return updateNetworkAndCheckRealistic(precomputed.equationSystem(), network, parameters, run);
        }
        boolean[] applies = realisticBandApplies(network, parameters, run);
        double minV = parameters.getMinRealisticVoltage();
        double maxV = parameters.getMaxRealisticVoltage();
        boolean trace = GpuTrace.UNREALISTIC_TRACE.isOn();
        int[] vRow = run.vRowOfBus;
        int[] phiRow = run.phiRowOfBus;
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
        return unrealistic;
    }

    /**
     * OLF's realistic-voltage check on a device state read through {@code view}, without writing the state back: the
     * decision {@link #updateBusesFromState} takes from the same array values, bus by bus. The buses get their voltage
     * and angle only when {@code write} (a state monitor asks for bus results). P1 of
     * {@code docs/device-final-results-plan.md}; the view must be {@link GpuStateView#usable() usable}.
     */
    static boolean checkBusesFromState(double[] state, GpuStateView view, LfNetwork network, AcLoadFlowParameters parameters,
                                       GpuSaRunState run, boolean write) {
        boolean[] applies = realisticBandApplies(network, parameters, run);
        double minV = parameters.getMinRealisticVoltage();
        double maxV = parameters.getMaxRealisticVoltage();
        boolean trace = GpuTrace.UNREALISTIC_TRACE.isOn();
        boolean unrealistic = false;
        for (LfBus bus : network.getBuses()) {
            if (bus.isDisabled()) {
                continue;
            }
            double value = view.stateV(state, bus);
            if (write) {
                bus.setV(value);
                bus.setAngle(view.busAngle(state, bus));
            }
            if ((!unrealistic || trace) && (value < minV || value > maxV) && applies[bus.getNum()]) {
                unrealistic = true;
                if (trace) {
                    System.err.printf("UNREALISTIC bus=%s v=%.9f nominalV=%.3f band=[%.3f, %.3f]%n",
                            bus.getId(), value, bus.getNominalV(), minV, maxV);
                }
            }
        }
        return unrealistic;
    }

    /** Which buses the realistic-voltage band applies to: a per-network constant, cached on the run state. */
    private static boolean[] realisticBandApplies(LfNetwork network, AcLoadFlowParameters parameters, GpuSaRunState run) {
        if (run.realisticBandApplies == null || run.realisticBandNetwork != network) {
            double minNominalV = parameters.getMinNominalVoltageRealisticVoltageCheck();
            boolean[] applies = new boolean[network.getBuses().size()];
            for (LfBus b : network.getBuses()) {
                applies[b.getNum()] = !b.isFictitious() && b.getNominalV() >= minNominalV;
            }
            run.realisticBandApplies = applies;
            run.realisticBandNetwork = network;
        }
        return run.realisticBandApplies;
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
                                                          GpuSaRunState run) {
        List<Variable<AcVariableType>> vars = es.getIndex().getSortedVariablesToFind();
        for (Variable<AcVariableType> v : vars) {
            if (v.getType() != AcVariableType.BUS_V && v.getType() != AcVariableType.BUS_PHI) {
                AcSolverUtil.updateNetwork(network, es);
                return isStateUnrealistic(es, network, parameters);
            }
        }
        boolean[] applies = realisticBandApplies(network, parameters, run);
        double minV = parameters.getMinRealisticVoltage();
        double maxV = parameters.getMaxRealisticVoltage();
        boolean trace = GpuTrace.UNREALISTIC_TRACE.isOn();
        com.powsybl.openloadflow.equations.StateVector stateVector = es.getStateVector();
        boolean unrealistic = false;
        for (Variable<AcVariableType> v : vars) {
            double value = stateVector.get(v.getRow());
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
        boolean trace = GpuTrace.UNREALISTIC_TRACE.isOn();
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
}
