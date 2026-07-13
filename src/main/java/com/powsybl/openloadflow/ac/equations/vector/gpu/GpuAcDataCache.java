/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.TwoSides;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.EquationArray;
import com.powsybl.openloadflow.equations.EquationSystem;
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
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkListener;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.LfZeroImpedanceNetwork;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.TransformerVoltageControl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;

/**
 * Shared, invalidation-guarded {@link GpuAcDataExtractor} cache, so the two GPU consumers of ONE
 * equation system — the base-case {@link com.powsybl.openloadflow.ac.solver.GpuNewtonRaphson}
 * and the batched N-1 {@link com.powsybl.openloadflow.ac.solver.GpuBatchedSecurityAnalysisSolver}
 * — extract it once per state instead of once each. In a cold security-analysis pass the base
 * load flow's LAST extraction (post outer loops) is exactly the state the batch then needs, so
 * the batch's extraction becomes a cache hit (host extraction is a measurable slice of the pass
 * on large cases).
 *
 * <p>Invalidation is EXHAUSTIVELY CONSERVATIVE: every equation-index event and every
 * {@link LfNetworkListener} event clears the entry — including pure target/value changes that a
 * finer classification could keep (the DS/RL packs embed target values, so staleness there would
 * be a silent wrong answer; a spurious re-extract is merely the pre-cache behavior). The
 * base→batch hand-off fires no events in between, so the hit that matters is preserved. Entries
 * are keyed weakly by the equation system; the listener chain is reachable only through the
 * equation system / network themselves, so everything unroots together.
 */
public final class GpuAcDataCache {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuAcDataCache.class);

    /** The (distributedSlack, reactiveLimits) extraction flavor — different outer-loop packs. */
    private enum Flavor {
        DS_RL, DS_ONLY, RL_ONLY, NONE;

        static Flavor of(boolean distributedSlack, boolean reactiveLimits) {
            if (distributedSlack) {
                return reactiveLimits ? DS_RL : DS_ONLY;
            }
            return reactiveLimits ? RL_ONLY : NONE;
        }
    }

    private static final class Entry implements EquationSystemIndexListener<AcVariableType, AcEquationType>, LfNetworkListener {

        private final LfNetwork network;
        private final Map<Flavor, GpuAcData> byFlavor = new EnumMap<>(Flavor.class);

        private Entry(LfNetwork network) {
            this.network = network;
        }

        private synchronized void clear() {
            byFlavor.clear();
        }

        // --- EquationSystemIndexListener: every event clears ---

        @Override
        public void onVariableChange(Variable<AcVariableType> variable, ChangeType changeType) {
            clear();
        }

        @Override
        public void onEquationChange(SingleEquation<AcVariableType, AcEquationType> equation, ChangeType changeType) {
            clear();
        }

        @Override
        public void onEquationTermChange(SingleEquationTerm<AcVariableType, AcEquationType> term) {
            clear();
        }

        @Override
        public void onEquationArrayChange(EquationArray<AcVariableType, AcEquationType> equationArray, ChangeType changeType) {
            clear();
        }

        @Override
        public void onEquationTermArrayChange(EquationTermArray<AcVariableType, AcEquationType> equationTermArray,
                                              int termNum, ChangeType changeType) {
            clear();
        }

        @Override
        public void onEquationIndexOrderChanged() {
            clear();
        }

        // --- LfNetworkListener: every event clears (exhaustive on purpose — a new event type
        //     added upstream breaks this compile, which is the safe failure mode) ---

        @Override
        public void onGeneratorVoltageControlChange(LfBus controllerBus, boolean newVoltageControllerEnabled) {
            clear();
        }

        @Override
        public void onGeneratorVoltageControlTargetChange(GeneratorVoltageControl control, double newTargetVoltage) {
            clear();
        }

        @Override
        public void onGeneratorReactivePowerControlChange(LfBus controllerBus, boolean newReactiveControllerEnabled) {
            clear();
        }

        @Override
        public void onTransformerPhaseControlChange(LfBranch controllerBranch, boolean newPhaseControlEnabled) {
            clear();
        }

        @Override
        public void onTransformerVoltageControlTargetChange(TransformerVoltageControl transformerVoltageControl, double newTargetVoltage) {
            clear();
        }

        @Override
        public void onTransformerVoltageControlChange(LfBranch controllerBranch, boolean newVoltageControllerEnabled) {
            clear();
        }

        @Override
        public void onShuntVoltageControlChange(LfShunt controllerShunt, boolean newVoltageControllerEnabled) {
            clear();
        }

        @Override
        public void onLoadActivePowerTargetChange(LfLoad load, double oldTargetP, double newTargetP) {
            clear();
        }

        @Override
        public void onLoadReactivePowerTargetChange(LfLoad load, double oldTargetQ, double newTargetQ) {
            clear();
        }

        @Override
        public void onGenerationActivePowerTargetChange(LfGenerator generator, double oldGenerationTargetP, double newGenerationTargetP) {
            clear();
        }

        @Override
        public void onGenerationReactivePowerTargetChange(LfBus bus, double oldGenerationTargetQ, double newGenerationTargetQ) {
            clear();
        }

        @Override
        public void onDisableChange(LfElement element, boolean disabled) {
            clear();
        }

        @Override
        public void onTapPositionChange(LfBranch branch, int oldPosition, int newPosition) {
            clear();
        }

        @Override
        public void onShuntSusceptanceChange(LfShunt shunt, double b) {
            clear();
        }

        @Override
        public void onZeroImpedanceNetworkSpanningTreeChange(LfBranch branch, LoadFlowModel loadFlowModel, boolean spanningTree) {
            clear();
        }

        @Override
        public void onZeroImpedanceNetworkSplit(LfZeroImpedanceNetwork initialNetwork, List<LfZeroImpedanceNetwork> splitNetworks, LoadFlowModel loadFlowModel) {
            clear();
        }

        @Override
        public void onZeroImpedanceNetworkMerge(LfZeroImpedanceNetwork network1, LfZeroImpedanceNetwork network2, LfZeroImpedanceNetwork mergedNetwork, LoadFlowModel loadFlowModel) {
            clear();
        }

        @Override
        public void onBranchConnectionStatusChange(LfBranch branch, TwoSides side, boolean connected) {
            clear();
        }

        @Override
        public void onSlackBusChange(LfBus bus, boolean slack) {
            clear();
        }

        @Override
        public void onReferenceBusChange(LfBus bus, boolean reference) {
            clear();
        }

        @Override
        public void onHvdcAcEmulationStatusChange(LfHvdc hvdc, LfHvdc.AcEmulationControl.AcEmulationStatus acEmulationStatus) {
            clear();
        }
    }

    private static final Map<EquationSystem<AcVariableType, AcEquationType>, Entry> ENTRIES = new WeakHashMap<>();

    private GpuAcDataCache() {
    }

    /**
     * The extraction for (network, es, flags), reusing the cached result when no equation-index or
     * network event occurred since it was produced. Same contract as
     * {@link GpuAcDataExtractor#extract(LfNetwork, EquationSystem, boolean, boolean)} — including
     * throwing on an unsupported system.
     */
    public static synchronized GpuAcData extract(LfNetwork network,
                                                 EquationSystem<AcVariableType, AcEquationType> es,
                                                 boolean distributedSlack, boolean reactiveLimits) {
        Entry entry = ENTRIES.get(es);
        if (entry == null || entry.network != network) {
            entry = new Entry(network);
            es.getIndex().addListener(entry);
            network.addListener(entry);
            ENTRIES.put(es, entry);
        }
        Flavor flavor = Flavor.of(distributedSlack, reactiveLimits);
        synchronized (entry) {
            GpuAcData data = entry.byFlavor.get(flavor);
            if (data == null) {
                data = GpuAcDataExtractor.extract(network, es, distributedSlack, reactiveLimits);
                entry.byFlavor.put(flavor, data);
            } else {
                LOGGER.debug("GPU extraction cache hit ({})", flavor);
            }
            return data;
        }
    }
}
