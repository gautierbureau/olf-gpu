/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu.fixedmatrix;

import com.powsybl.commons.PowsyblException;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.network.GeneratorVoltageControl;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.ShuntVoltageControl;
import com.powsybl.openloadflow.network.TransformerVoltageControl;
import com.powsybl.openloadflow.network.VoltageControl;
import com.powsybl.openloadflow.network.gpu.GeneratorVoltageControlRoles;

import java.util.List;

/**
 * docs/gpu-vs-olf.md section 2.3 (remote generator voltage control: the carrier; transformer and shunt voltage
 * control): the diagonal-paired Q-balance row of each bus, i.e. which row hosts its reactive balance — its own
 * BUS_V row, the free controlling transformer's BRANCH_RHO1 or shunt's SHUNT_B row, the free controller's BUS_V row
 * for a remote generator voltage control's pilot, or its parent coupler's DUMMY_Q row once the coupler-tree
 * relocation ({@link CouplerTreeRelocation}) has moved it.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class ReactiveRelocation {

    private ReactiveRelocation() {
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
    public static int[] computeQRow(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                             int[] vCol, int[] branchRhoCol, int[] shuntBCol) {
        int[] qRow = vCol.clone();
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            boolean vActive = StableRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_V) >= 0;
            boolean qActive = StableRowIndexing.row(es, bn, AcEquationType.BUS_TARGET_Q) >= 0;
            // A remote generator voltage controller bus normally has its BUS_TARGET_Q deactivated — its reactive
            // injection feeds the DISTR_Q rows, never its own BUS_V row — so suppress its per-bus Q scatter.
            // If the controller is switched to PQ by reactive limits, BUS_TARGET_Q becomes active again and the
            // bus must recover its own BUS_V-backed Q row.
            // CANONICAL encoding for a group with a candidate table (hasCandidateTable): the layout of the
            // all-controllers-enabled state, whatever the network's PV/PQ state at extraction time. The row
            // modes carry the state per scenario, and the batched security analysis extracts AFTER the base
            // load flow has switched buses to PQ: measured on rte6515's busbar window, the pilot-self pilot
            // VL-5997_0 was switched to PQ at SA extraction, so its q was frozen onto the member VL-5997_1's row
            // and the member lost its own alternate — the member's switch to PQ then assembled an EMPTY row
            // (slot 893, BUS-6088_BBS) while the base extraction (taken before any switch) had the layout the
            // kernels are built for.
            GeneratorVoltageControl mainVc = GeneratorVoltageControlRoles.mainGeneratorVoltageControl(bus);
            boolean canonical = mainVc != null && hasCandidateTable(mainVc, es);
            if (GeneratorVoltageControlRoles.isRemoteGeneratorVoltageControlControllerElement(bus) && (!qActive || canonical)) {
                qRow[bn] = -1;                               // includes a controller that is ALSO the pilot
                continue;
            }
            if (canonical && mainVc.getControlledBus() == bus) {
                // a pilot that is not a controller: its balance rides the canonical free member (the first),
                // re-picked per scenario by the candidate table's mode scan
                qRow[bn] = vCol[mainVc.getMergedControllerElements().get(0).getNum()];
                continue;
            }
            if (!(vActive && qActive)) {
                continue;                                   // plain PQ / generator PV bus: Q stays on its V row
            }
            // A pilot bus controlled by REMOTE generators: its reactive balance moves onto the free
            // controller's BUS_V row (the controller whose DISTR_Q is inactive — the extra freedom).
            GeneratorVoltageControl generatorVoltageControl = bus.getGeneratorVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .filter(c -> bus.isGeneratorVoltageControlled() && !c.isLocalControl())
                    .orElse(null);
            if (generatorVoltageControl != null) {
                LfBus freeController = generatorVoltageControl.getMergedControllerElements().stream()
                        .filter(c -> StableRowIndexing.row(es, c.getNum(), AcEquationType.DISTR_Q) < 0)
                        .filter(c -> StableRowIndexing.row(es, c.getNum(), AcEquationType.BUS_TARGET_Q) < 0)
                        .findFirst()
                        .orElseThrow(() -> new PowsyblException("GPU AC solver: remote generator voltage control "
                                + "of bus " + bus.getId() + " has no free enabled controller"));
                qRow[bn] = vCol[freeController.getNum()];
                continue;
            }
            // A pilot bus voltage-controlled by shunts: its reactive balance moves onto the FREE controller
            // shunt's SHUNT_B row (the controller whose DISTR_SHUNT_B is inactive — the extra freedom), the
            // exact analogue of the transformer rho case below.
            ShuntVoltageControl shuntVoltageControl = bus.getShuntVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (shuntVoltageControl != null) {
                LfShunt freeShunt = shuntVoltageControl.getMergedControllerElements().stream()
                        .filter(c -> !c.isDisabled())
                        .filter(c -> StableRowIndexing.row(es, c.getNum(), AcEquationType.DISTR_SHUNT_B) < 0)
                        .filter(c -> StableRowIndexing.row(es, c.getNum(), AcEquationType.SHUNT_TARGET_B) < 0)   // control ON
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
            TransformerVoltageControl transformerVoltageControl = bus.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElseThrow(() -> new PowsyblException("GPU AC solver: bus " + bus.getId()
                            + " has active voltage and reactive targets but no supported voltage control"));
            LfBranch freeController = transformerVoltageControl.getMergedControllerElements().stream()
                    .filter(c -> !c.isDisabled())            // an OUTAGED member also has DISTR_RHO
                    .filter(c -> StableRowIndexing.row(es, c.getNum(), AcEquationType.DISTR_RHO) < 0)   // inactive, but its rho
                    // a CLAMPED/ROUNDED member has DISTR_RHO inactive too, but its rho row already
                    // carries the BRANCH_TARGET_RHO1 identity — picking it as the free role would
                    // COLLIDE both equations on one row and leave the pilot's Q equation unmapped
                    // (the clamp-in-the-BASE sequential failure: device |f| 1.4e-5 vs CPU 7.4e-3)
                    .filter(c -> StableRowIndexing.row(es, c.getNum(), AcEquationType.BRANCH_TARGET_RHO1) < 0)
                    .findFirst()                             // = OLF's free role: first ENABLED member
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

    /** As below, following the coupler-tree relocation: a relocated CHILD's DISTR_Q moves onto its parent
     *  coupler's DUMMY_Q column, which its own reactive terms already carry (the kind-3 contribution), leaving
     *  its V row free for that coupler's ZERO_V. */
    public static int distrQRow(EquationSystem<AcVariableType, AcEquationType> es, LfBus controller,
                         int[] spanningTreeParentBranch, int[] dummyQCol) {
        int parent = spanningTreeParentBranch == null ? -1 : spanningTreeParentBranch[controller.getNum()];
        if (parent >= 0 && dummyQCol[parent] >= 0) {
            return dummyQCol[parent];
        }
        return distrQRow(es, controller);
    }

    /** The device row that carries a controller's DISTR_Q equation. Normally the controller's own
     *  BUS_V row — but when the controller IS the group's CONTROLLED bus, that row already carries
     *  the pilot's BUS_TARGET_V identity, so the DISTR_Q moves to the FREE controller's row: the
     *  row that hosts the pilot bus's reactive equation in every other form too (its Q balance when
     *  the pilot is a plain bus, or when the reactive limits switched this member to PQ). Without this the two equations
     *  collide on one row — one silently overwrites the other and the group's reactive sharing is
     *  left unsatisfied (measured on a richer-modelling RTE-like network: the device "converged"
     *  with a 0.45 pu DISTR_Q residual and OLF's own re-test then failed the load flow). */
    static int distrQRow(EquationSystem<AcVariableType, AcEquationType> es, LfBus controller) {
        var voltageControl = controller.getGeneratorVoltageControl()
                .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                .orElse(null);
        if (voltageControl != null && voltageControl.getControlledBus() == controller) {
            LfBus free = distrQHostMember(voltageControl, es);
            if (free == null) {
                throw new PowsyblException("GPU AC solver: remote generator voltage control of bus "
                        + controller.getId() + " has no controller");
            }
            return StableRowIndexing.col(es, free.getNum(), AcVariableType.BUS_V);
        }
        return StableRowIndexing.col(es, controller.getNum(), AcVariableType.BUS_V);
    }

    /** The member row that HOSTS the pilot's DISTR_Q when the pilot is one of its own controllers:
     *  OLF's reference controller (the first ENABLED merged controller), else - no enabled
     *  controller in the base state - the first merged controller. In that case the pilot's DISTR_Q is
     *  inactive and only needs a slot; the row modes carry the PQ state (mode 4 on every row).
     *  Measured on rte6515's full busbar scope: the base switches EVERY controller of VL-6003_0's
     *  group to PQ, "first enabled" found none and the whole precompute fell to the CPU.
     *  NOT the free-member predicate below: a member switched to PQ is never free (its row hosts its own
     *  Q balance, canonical encoding) - calling it free left a singleton switched to PQ at mode 1 with the
     *  structure-frozen encoding (deviceRemoteSingletonMemberSwitchToPq, deviceRemoteRobustRealisticVoltage). */
    static LfBus distrQHostMember(GeneratorVoltageControl voltageControl, EquationSystem<AcVariableType, AcEquationType> es) {
        List<LfBus> members = voltageControl.getMergedControllerElements();
        if (hasCandidateTable(voltageControl, es)) {
            return members.get(0);                           // canonical: the first member, whatever is switched to PQ
        }
        return members.stream().filter(voltageControl::isControllerEnabled).findFirst()
                .orElse(members.isEmpty() ? null : members.get(0));
    }

    /**
     * The row that hosts a bus's reactive balance once the COUPLER-TREE RELOCATION has had its say: its own
     * {@code BUS_V} row, or its parent coupler's {@code DUMMY_Q} row when the relocation moved its power
     * equations there. Every consumer of "where does this bus's q live" has to ask this — the reactive-limit
     * pass switches that row to PQ and reads the q there (native {@code reactiveLimitReactivePowerRow}), the candidate table enrols it, the
     * activation extractor targets it, the families write the bus's own q to it in mode 4, and the diagnostic
     * remote-member fill accumulates into it. A consumer left on the own row silently reads ZERO there.
     */
    public static int relocatedQRow(int busNum, int[] spanningTreeParentBranch, int[] dummyQCol, int[] vCol) {
        int parent = spanningTreeParentBranch == null ? -1 : spanningTreeParentBranch[busNum];
        return parent >= 0 && dummyQCol[parent] >= 0 ? dummyQCol[parent] : vCol[busNum];
    }

    /** The row that hosts a bus's ACTIVE Q balance in the CURRENT state when its group has a candidate table
     *  (see GpuAcActivationExtractor): for a member switched to PQ, its own V row; the pilot (a controller of its own
     *  group or not) rides the first ENABLED member's V row (the carrier) and falls back to its own V row when no
     *  controller is enabled — the row the in-kernel mode scan selects. */
    static int stateQRow(LfBus bus, int[] vCol) {
        return vCol[stateQCarrier(bus).getNum()];
    }

    /** {@link #stateQRow} with the COUPLER-TREE RELOCATION applied to the carrier, which is what the candidate
     *  table enrols and therefore what the in-kernel mode scan selects: a relocated member's reactive balance
     *  lives on its parent coupler's {@code DUMMY_Q} row, its own {@code BUS_V} row holding the coupler's
     *  {@code ZERO_V}. Naming the own row instead assigned two equations to it — measured on rte6515's
     *  [5000,6000) window with the relocation on: "device row 12562 assigned twice", the whole precompute lost. */
    public static int stateQRow(LfBus bus, int[] vCol, int[] spanningTreeParentBranch, int[] dummyQCol) {
        LfBus carrier = stateQCarrier(bus);
        int parent = spanningTreeParentBranch == null ? -1 : spanningTreeParentBranch[carrier.getNum()];
        return parent >= 0 && dummyQCol[parent] >= 0 ? dummyQCol[parent] : vCol[carrier.getNum()];
    }

    /** The bus whose row carries {@code bus}'s active Q balance: itself, or — for the pilot of a
     *  candidate-table group — the first ENABLED member (the carrier), falling back to itself when no
     *  controller is enabled. */
    private static LfBus stateQCarrier(LfBus bus) {
        GeneratorVoltageControl voltageControl = GeneratorVoltageControlRoles.mainGeneratorVoltageControl(bus);
        if (voltageControl != null && voltageControl.getControlledBus() == bus) {
            return voltageControl.getMergedControllerElements().stream()
                    .filter(voltageControl::isControllerEnabled).findFirst().orElse(bus);
        }
        return bus;
    }

    /**
     * True when {@code voltageControl} is a MAIN remote generator voltage control that ExtractionColumns gives a
     * candidate table (plan-B4 / pilot-self rotation): every member and the pilot have a BUS_V column, and a
     * pilot that is one of its own controllers rotates (property on, at least two controllers). Such a
     * group is encoded CANONICALLY — as if every controller were enabled — because the table's in-kernel
     * mode scan re-picks the carrier per scenario; the PV/PQ state at extraction time must not leak into the
     * structure (see computeQRow). Keep in step with the enrolment in ExtractionColumns.
     */
    public static boolean hasCandidateTable(GeneratorVoltageControl voltageControl, EquationSystem<AcVariableType, AcEquationType> es) {
        if (voltageControl == null || voltageControl.getMergeStatus() != VoltageControl.MergeStatus.MAIN || voltageControl.isLocalControl()) {
            return false;
        }
        LfBus pilot = voltageControl.getControlledBus();
        List<LfBus> members = voltageControl.getMergedControllerElements();
        if (members.isEmpty() || StableRowIndexing.col(es, pilot.getNum(), AcVariableType.BUS_V) < 0) {
            return false;
        }
        boolean pilotSelf = members.contains(pilot);
        if (pilotSelf && members.size() < 2) {
            return false;
        }
        return members.stream().allMatch(m -> StableRowIndexing.col(es, m.getNum(), AcVariableType.BUS_V) >= 0);
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
     * the control is on or off (fixed by BRANCH_TARGET_RHO1 / SHUNT_TARGET_B when off), so the row is
     * always resolvable. The free controller is selected identically to {@link #computeQRow}: the one
     * whose distribution equation is inactive (with all inactive, the first — stable across states).
     */
    public static int[] computeAlternateQRow(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                int[] vCol, int[] branchRhoCol, int[] shuntBCol, int[] qRow) {
        int[] alt = new int[vCol.length];
        java.util.Arrays.fill(alt, -1);
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            int controlVarRow = -1;
            TransformerVoltageControl transformerVoltageControl = bus.getTransformerVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            ShuntVoltageControl shuntVoltageControl = bus.getShuntVoltageControl()
                    .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                    .orElse(null);
            if (transformerVoltageControl != null) {
                // NO control-ON filter here (unlike computeQRow's live free pick): the ALTERNATE
                // is computed at a control-OFF base too, where every identity is active — the
                // stable "first with DISTR_RHO inactive" pick keeps the two-way relocation alive.
                LfBranch freeController = transformerVoltageControl.getMergedControllerElements().stream()
                        .filter(c -> !c.isDisabled())
                        .filter(c -> StableRowIndexing.row(es, c.getNum(), AcEquationType.DISTR_RHO) < 0)
                        .findFirst().orElse(null);
                if (freeController != null) {
                    controlVarRow = branchRhoCol[freeController.getNum()];
                }
            } else if (shuntVoltageControl != null) {
                LfShunt freeController = shuntVoltageControl.getMergedControllerElements().stream()
                        .filter(c -> StableRowIndexing.row(es, c.getNum(), AcEquationType.DISTR_SHUNT_B) < 0)
                        .findFirst().orElse(null);
                if (freeController != null) {
                    controlVarRow = shuntBCol[freeController.getNum()];
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
