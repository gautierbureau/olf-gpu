/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.iidm.network.TwoSides;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.EquationArray;
import com.powsybl.openloadflow.equations.EquationSystemIndexListener;
import com.powsybl.openloadflow.equations.EquationTermArray;
import com.powsybl.openloadflow.equations.SingleEquation;
import com.powsybl.openloadflow.equations.SingleEquationTerm;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfElement;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfLoad;
import com.powsybl.openloadflow.network.LfNetworkListener;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.LfZeroImpedanceNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.TransformerVoltageControl;

import java.util.List;

/**
 * The event-driven invalidation state for {@link GpuNewtonRaphson}, a SEPARATE listener
 * object (the Cleaner action references it for deregistration — referencing the solver
 * itself would keep it strongly reachable forever and the cleanup would never run).
 *
 * <p>Event classification. STRUCTURE = the extracted element packs / variable rows may
 * have changed -> full re-extract + validation. {@code run()} then splits STRUCTURE into two
 * levels by comparing the fresh data: a changed CSR pattern (index packs) rebuilds +
 * re-analyzes, while an unchanged pattern with only changed element PARAMETERS (a tap or
 * shunt-b move) takes the cheap VALUES re-upload (refreshStructureValues, analysis reused).
 * ACTIVATION = only which equations are active changed -> refresh rowMode/targetMap.
 * Target-value events need nothing: targets are re-read from the TargetVector (which
 * maintains itself through its own listeners) on every solve.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuSolverInvalidation
        implements EquationSystemIndexListener<AcVariableType, AcEquationType>, LfNetworkListener {

    /** Set by the CLEANER THREAD when the owning solver is collected. The listener then
     *  no-ops and is PURGED SYNCHRONOUSLY by the next solver's constructor — the cleaner
     *  itself must NEVER call removeListener: LfNetwork's listener list is a plain
     *  ArrayList iterated during event dispatch on the solve thread, and a cross-thread
     *  removal is a ConcurrentModificationException (measured 2026-07-17: NetworkState
     *  .restore -> notifyDisable iterating while a collected solver's cleaner removed). */
    volatile boolean dead = false;

    boolean structureDirty = true;

    boolean activationDirty = true;

    private void structureChanged() {
        structureDirty = true;
    }

    private void activationChanged() {
        activationDirty = true;
    }

    // --- EquationSystemIndexListener ---

    @Override
    public void onVariableChange(Variable<AcVariableType> variable, ChangeType changeType) {
        structureChanged();                                  // variable rows shift
    }

    @Override
    public void onEquationChange(SingleEquation<AcVariableType, AcEquationType> equation, ChangeType changeType) {
        activationChanged();                                 // single equation (de)activated
    }

    @Override
    public void onEquationTermChange(SingleEquationTerm<AcVariableType, AcEquationType> term) {
        if (term instanceof com.powsybl.openloadflow.equations.VariableEquationTerm) {
            activationChanged();                             // identity-term toggle = PV→PQ side effect
        } else {
            structureChanged();                              // any other term toggling (open/closed switching)
        }
    }

    @Override
    public void onEquationArrayChange(EquationArray<AcVariableType, AcEquationType> equationArray, ChangeType changeType) {
        activationChanged();                                 // array element (de)activated (PV→PQ)
    }

    @Override
    public void onEquationTermArrayChange(EquationTermArray<AcVariableType, AcEquationType> equationTermArray,
                                          int termNum, ChangeType changeType) {
        structureChanged();                                  // array term toggling
    }

    @Override
    public void onEquationIndexOrderChanged() {
        activationChanged();                                 // OLF equation columns renumbered:
    }                                                        // structure is column-free, targetMap is not

    // --- LfNetworkListener ---

    @Override
    public void onGeneratorVoltageControlChange(LfBus controllerBus, boolean newVoltageControllerEnabled) {
        if (controllerBus.getGeneratorVoltageControl().isPresent() && !controllerBus.isGeneratorVoltageControlled()) {
            structureChanged();                              // remote controller enable/disable reshapes qRow routing
        } else {
            activationChanged();                             // local PV↔PQ
        }
    }

    @Override
    public void onGeneratorVoltageControlTargetChange(GeneratorVoltageControl control, double newTargetVoltage) {
        // target values are re-read every solve
    }

    @Override
    public void onGeneratorReactivePowerControlChange(LfBus controllerBus, boolean newReactiveControllerEnabled) {
        structureChanged();
    }

    @Override
    public void onTransformerPhaseControlChange(LfBranch controllerBranch, boolean newPhaseControlEnabled) {
        structureChanged();
    }

    @Override
    public void onTransformerVoltageControlChange(LfBranch controllerBranch, boolean newVoltageControllerEnabled) {
        structureChanged();
    }

    @Override
    public void onTransformerVoltageControlTargetChange(TransformerVoltageControl transformerVoltageControl,
                                                        double newTargetVoltage) {
        // target values are re-read every solve
    }

    @Override
    public void onShuntVoltageControlChange(LfShunt controllerShunt, boolean newVoltageControllerEnabled) {
        structureChanged();
    }

    @Override
    public void onLoadActivePowerTargetChange(LfLoad load, double oldTargetP, double newTargetP) {
        structureChanged();                                  // DS packs embed bus/load active-power targets
    }

    @Override
    public void onLoadReactivePowerTargetChange(LfLoad load, double oldTargetQ, double newTargetQ) {
        // target values are re-read every solve
    }

    @Override
    public void onGenerationActivePowerTargetChange(LfGenerator generator, double oldGenerationTargetP,
                                                    double newGenerationTargetP) {
        structureChanged();                                  // DS packs embed generator targetP-derived headroom/base P
    }

    @Override
    public void onGenerationReactivePowerTargetChange(LfBus bus, double oldGenerationTargetQ,
                                                      double newGenerationTargetQ) {
        // target values are re-read every solve
    }

    @Override
    public void onDisableChange(LfElement element, boolean disabled) {
        structureChanged();
    }

    @Override
    public void onTapPositionChange(LfBranch branch, int oldPosition, int newPosition) {
        structureChanged();                                  // r1/a1 in closedBranchParameters; run() downgrades to VALUES
    }

    @Override
    public void onShuntSusceptanceChange(LfShunt shunt, double b) {
        structureChanged();                                  // b in shuntParameters; run() downgrades to VALUES
    }

    @Override
    public void onZeroImpedanceNetworkSpanningTreeChange(LfBranch branch, LoadFlowModel loadFlowModel,
                                                         boolean spanningTree) {
        structureChanged();
    }

    @Override
    public void onZeroImpedanceNetworkSplit(LfZeroImpedanceNetwork initialNetwork,
                                            List<LfZeroImpedanceNetwork> splitNetworks, LoadFlowModel loadFlowModel) {
        structureChanged();
    }

    @Override
    public void onZeroImpedanceNetworkMerge(LfZeroImpedanceNetwork network1, LfZeroImpedanceNetwork network2,
                                            LfZeroImpedanceNetwork mergedNetwork, LoadFlowModel loadFlowModel) {
        structureChanged();
    }

    @Override
    public void onBranchConnectionStatusChange(LfBranch branch, TwoSides side, boolean connected) {
        structureChanged();
    }

    @Override
    public void onSlackBusChange(LfBus bus, boolean slack) {
        activationChanged();                                 // PHI identity moves
    }

    @Override
    public void onReferenceBusChange(LfBus bus, boolean reference) {
        activationChanged();
    }

    @Override
    public void onHvdcAcEmulationStatusChange(LfHvdc hvdc, LfHvdc.AcEmulationControl.AcEmulationStatus acEmulationStatus) {
        structureChanged();
    }
}
