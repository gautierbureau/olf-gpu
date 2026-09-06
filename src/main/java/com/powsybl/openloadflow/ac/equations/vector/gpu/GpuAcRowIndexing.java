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
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.ShuntVoltageControl;
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
     * <p>Transformer AND shunt voltage control are supported (a shunt-controlled bus's reactive balance moves
     * onto the free controlling shunt's SHUNT_B column, the exact analogue of the transformer rho case).
     */
    static int[] computeQRow(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                             int[] vCol, int[] branchRhoCol, int[] shuntBCol) {
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
            // A pilot bus voltage-controlled by shunts: its reactive balance moves onto the FREE controller
            // shunt's SHUNT_B row (the controller whose DISTR_SHUNT_B is inactive — the extra freedom), the
            // exact analogue of the transformer rho case below.
            ShuntVoltageControl svc = bus.getShuntVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (svc != null) {
                LfShunt freeShunt = svc.getMergedControllerElements().stream()
                        .filter(c -> row(es, c.getNum(), AcEquationType.DISTR_SHUNT_B) < 0)
                        .findFirst()
                        .orElseThrow(() -> new PowsyblException("GPU AC solver: shunt voltage control of bus "
                                + bus.getId() + " has no free controller (all DISTR_SHUNT_B active)"));
                int bRow = shuntBCol[freeShunt.getNum()];
                if (bRow < 0) {
                    throw new PowsyblException("GPU AC solver: controlling shunt " + freeShunt.getId()
                            + " has no SHUNT_B variable");
                }
                qRow[bn] = bRow;
                continue;
            }
            // A pilot bus voltage-controlled by transformers: its reactive balance moves onto the FREE
            // controller's BRANCH_RHO1 row (the controller whose DISTR_RHO is inactive — the extra freedom).
            // With one local transformer this is item 4; with several it is ratio distribution (DISTR_RHO),
            // and the controller may even be remote (then the pilot-Q diagonal is zero → cuDSS pivoting).
            TransformerVoltageControl tvc = bus.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElseThrow(() -> new PowsyblException("GPU AC solver: bus " + bus.getId()
                            + " has active voltage and reactive targets but no supported voltage control"));
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

    /**
     * The ALTERNATE Q-balance-row candidate for each control-capable bus, for Dynamic Q-relocation
     * (see the production-design doc §11). A transformer/shunt voltage-controlled bus has exactly two
     * possible reactive-balance rows across the outer loop: its own BUS_V row (control OFF, the ratio /
     * susceptance held constant) and the free controller's BRANCH_RHO1 / SHUNT_B row (control ON, Q
     * relocated onto the control variable). {@link #computeQRow} bakes whichever one is active at
     * extraction; this returns the OTHER one so the fill kernel can follow a per-scenario control
     * toggle on the shared structure. {@code -1} for a bus that is not transformer/shunt voltage
     * controlled (no second candidate — its Q row is fixed). The control-var COLUMN persists whether
     * the control is on or off (pinned by BRANCH_TARGET_RHO1 / SHUNT_TARGET_B when off), so the row is
     * always resolvable. The free controller is selected identically to {@link #computeQRow}: the one
     * whose distribution equation is inactive (with all inactive, the first — stable across states).
     */
    /** True when {@code bus} is a member of a MAIN remote generator voltage-control group (its
     *  controlled bus is another bus). The FREE member (first controller-ENABLED, matching
     *  updateRemoteVoltageControlEquations) is the group's pilot-Q carrier and keeps the
     *  structure-frozen encoding; the others take the canonical suppressed-dest/mode-4 one. */
    static boolean isRemoteGenVcMember(LfBus bus) {
        return bus.getGeneratorVoltageControl()
                .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                .map(c -> c.getControlledBus() != bus)
                .orElse(false);
    }

    static boolean isFreeRemoteGenVcMember(LfBus bus) {
        return bus.getGeneratorVoltageControl()
                .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                .map(c -> c.getMergedControllerElements().stream()
                        .filter(c::isControllerEnabled).findFirst().map(m -> m == bus).orElse(false))
                .orElse(false);
    }

    static int[] computeQRowAlt(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                int[] vCol, int[] branchRhoCol, int[] shuntBCol, int[] qRow) {
        int[] alt = new int[vCol.length];
        java.util.Arrays.fill(alt, -1);
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            int controlVarRow = -1;
            TransformerVoltageControl tvc = bus.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            ShuntVoltageControl svc = bus.getShuntVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (tvc != null) {
                LfBranch ctl = tvc.getMergedControllerElements().stream()
                        .filter(c -> row(es, c.getNum(), AcEquationType.DISTR_RHO) < 0)
                        .findFirst().orElse(null);
                if (ctl != null) {
                    controlVarRow = branchRhoCol[ctl.getNum()];
                }
            } else if (svc != null) {
                LfShunt ctl = svc.getMergedControllerElements().stream()
                        .filter(c -> row(es, c.getNum(), AcEquationType.DISTR_SHUNT_B) < 0)
                        .findFirst().orElse(null);
                if (ctl != null) {
                    controlVarRow = shuntBCol[ctl.getNum()];
                }
            }
            if (controlVarRow >= 0 && vCol[bn] >= 0) {
                // Relocated at extraction (qRow == controlVarRow) -> alternate is the bus V row;
                // not relocated (qRow == vCol) -> alternate is the control-var row.
                alt[bn] = qRow[bn] == controlVarRow ? vCol[bn] : controlVarRow;
            }
        }
        return alt;
    }
}
