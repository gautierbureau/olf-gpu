/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.TransformerVoltageControl;
import com.powsybl.openloadflow.network.VoltageControl;

/**
 * The STABLE row-indexing policy shared by the whole GPU extraction: variable/equation
 * row lookups and the diagonal-paired Q-balance row of each bus. Every equation row is
 * identified by its diagonal-paired VARIABLE row (a bus's P row is its BUS_PHI variable
 * row, its Q row its BUS_V row — or the controlling transformer's BRANCH_RHO1 row when
 * the bus is transformer-voltage-controlled), never by OLF's equation column, so the
 * extracted structure survives activation switching. {@link GpuAcDataExtractor},
 * {@link GpuAcActivationExtractor} and the family/outer-loop packers all lean on this.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuAcRowIndexing {

    private GpuAcRowIndexing() {
    }

    static int col(EquationSystem<AcVariableType, AcEquationType> es, int bus, AcVariableType t) {
        return es.getVariableSet().getVariable(bus, t).getRow();
    }

    static int row(EquationSystem<AcVariableType, AcEquationType> es, int bus, AcEquationType t) {
        return es.getEquation(bus, t).map(e -> e.getColumn()).orElse(-1);
    }

    /** Variable rows of every {@code type} variable, indexed by its element (branch) number; -1 when absent. */
    static int[] branchVarCols(EquationSystem<AcVariableType, AcEquationType> es, int nBranches,
                               AcVariableType type) {
        int[] cols = new int[nBranches];
        java.util.Arrays.fill(cols, -1);
        for (Variable<AcVariableType> v : es.getIndex().getSortedVariablesToFind()) {
            if (v.getType() == type) {
                cols[v.getElementNum()] = v.getRow();
            }
        }
        return cols;
    }

    /**
     * The stable Q-balance row of each bus (indexed by bus number). Normally a bus's reactive
     * balance sits on its own BUS_V variable row. When a bus is voltage-controlled by a transformer
     * — signalled by BOTH its BUS_TARGET_V and BUS_TARGET_Q being active (a generator PV bus has V
     * active but Q inactive, so it is excluded) — the BUS_V row instead carries the v−target identity,
     * and the reactive balance moves onto the controlling transformer's BRANCH_RHO1 column, which is
     * the extra degree of freedom. That rho column is then the strong diagonal of the Q row.
     *
     * <p>Only the simple local single-transformer case is supported; throws on shunt voltage control,
     * remote control, and ratio distribution across several transformers (DISTR_RHO).
     */
    static int[] computeQRow(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                             int[] vCol, int[] branchRhoCol) {
        int[] qRow = vCol.clone();
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            boolean vActive = row(es, bn, AcEquationType.BUS_TARGET_V) >= 0;
            boolean qActive = row(es, bn, AcEquationType.BUS_TARGET_Q) >= 0;
            // A remote generator voltage controller bus normally has its BUS_TARGET_Q deactivated — its reactive
            // injection feeds the DISTR_Q rows, never its own BUS_V row — so suppress its per-bus Q scatter.
            // If the controller is switched to PQ by reactive limits, BUS_TARGET_Q becomes active again and the
            // bus must recover its own BUS_V-backed Q row.
            if (bus.getGeneratorVoltageControl().isPresent() && !bus.isGeneratorVoltageControlled() && !qActive) {
                qRow[bn] = -1;
                continue;
            }
            if (!(vActive && qActive)) {
                continue;                                   // plain PQ / generator PV bus: Q stays on its V row
            }
            // A pilot bus controlled by REMOTE generators: its reactive balance moves onto the free
            // controller's BUS_V row (the controller whose DISTR_Q is inactive — the extra freedom).
            GeneratorVoltageControl gvc = bus.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> bus.isGeneratorVoltageControlled() && !c.isLocalControl())
                    .orElse(null);
            if (gvc != null) {
                LfBus freeController = gvc.getMergedControllerElements().stream()
                        .filter(c -> row(es, c.getNum(), AcEquationType.DISTR_Q) < 0)
                        .filter(c -> row(es, c.getNum(), AcEquationType.BUS_TARGET_Q) < 0)
                        .findFirst()
                        .orElseThrow(() -> new PowsyblException("GPU AC solver: remote generator voltage control "
                                + "of bus " + bus.getId() + " has no free enabled controller"));
                qRow[bn] = vCol[freeController.getNum()];
                continue;
            }
            // A pilot bus voltage-controlled by transformers: its reactive balance moves onto the FREE
            // controller's BRANCH_RHO1 row (the controller whose DISTR_RHO is inactive — the extra freedom).
            // With one local transformer this is item 4; with several it is ratio distribution (DISTR_RHO),
            // and the controller may even be remote (then the pilot-Q diagonal is zero → cuDSS pivoting).
            TransformerVoltageControl tvc = bus.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElseThrow(() -> new PowsyblException("GPU AC solver: bus " + bus.getId()
                            + " has active voltage and reactive targets but no supported voltage control "
                            + "(shunt voltage control is not supported)"));
            LfBranch freeController = tvc.getMergedControllerElements().stream()
                    .filter(c -> row(es, c.getNum(), AcEquationType.DISTR_RHO) < 0)
                    .findFirst()
                    .orElseThrow(() -> new PowsyblException("GPU AC solver: transformer voltage control of bus "
                            + bus.getId() + " has no free controller (all DISTR_RHO active)"));
            int rhoRow = branchRhoCol[freeController.getNum()];
            if (rhoRow < 0) {
                throw new PowsyblException("GPU AC solver: controlling transformer " + freeController.getId()
                        + " has no BRANCH_RHO1 variable");
            }
            qRow[bn] = rhoRow;
        }
        return qRow;
    }
}
