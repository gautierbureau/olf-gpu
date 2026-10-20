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
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.ShuntVoltageControl;
import com.powsybl.openloadflow.network.TransformerVoltageControl;
import com.powsybl.openloadflow.network.VoltageControl;

import java.util.List;

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
            // CANONICAL encoding for a group with a candidate table (hasCandidateTable): the layout of the
            // all-controllers-enabled state, whatever the network's pin state at extraction time. The row
            // modes carry the state per scenario, and the batched security analysis extracts AFTER the base
            // load flow has pinned buses: measured on rte6515's busbar window, the pilot-self pilot VL-5997_0
            // was pinned at SA extraction, so its q was frozen onto the member VL-5997_1's row and the member
            // lost its own alternate — the member's pin then assembled an EMPTY row (slot 893, BUS-6088_BBS)
            // while the base extraction (taken before any pin) had the layout the kernels are built for.
            GeneratorVoltageControl mainVc = mainGenVc(bus);
            boolean canonical = mainVc != null && hasCandidateTable(mainVc, es);
            if (isRemoteGenVcControllerElement(bus) && (!qActive || canonical)) {
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
                        .filter(c -> !c.isDisabled())
                        .filter(c -> row(es, c.getNum(), AcEquationType.DISTR_SHUNT_B) < 0)
                        .filter(c -> row(es, c.getNum(), AcEquationType.SHUNT_TARGET_B) < 0)   // control ON
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
                    .filter(c -> !c.isDisabled())            // an OUTAGED member also has DISTR_RHO
                    .filter(c -> row(es, c.getNum(), AcEquationType.DISTR_RHO) < 0)   // inactive, but its rho
                    // a CLAMPED/ROUNDED member has DISTR_RHO inactive too, but its rho row already
                    // carries the BRANCH_TARGET_RHO1 identity — picking it as the free role would
                    // COLLIDE both equations on one row and leave the pilot's Q equation unmapped
                    // (the clamp-in-the-BASE sequential failure: device |f| 1.4e-5 vs CPU 7.4e-3)
                    .filter(c -> row(es, c.getNum(), AcEquationType.BRANCH_TARGET_RHO1) < 0)
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

    /** True when {@code bus} is a member of a MAIN remote generator voltage-control group (its
     *  controlled bus is another bus). The FREE member (first controller-ENABLED, matching
     *  updateRemoteVoltageControlEquations) is the group's pilot-Q carrier and keeps the
     *  structure-frozen encoding; the others take the canonical suppressed-dest/mode-4 one. */
    static boolean isRemoteGenVcMember(LfBus bus) {
        GeneratorVoltageControl vc = mainGenVc(bus);
        return vc != null && isRemoteGenVcControllerElement(bus) && vc.getControlledBus() != bus;
    }

    /**
     * The bus's MAIN generator voltage control, resolved the way {@link #isRemoteGenVcControllerElement}
     * already resolves it and for the same reason: a control MERGED into another is DEPENDENT, and on the
     * dependent object the controlled bus is the member ITSELF, so a {@code getMergeStatus() == MAIN} filter
     * answers "not a member" for every member of a merged group but one.
     *
     * <p>Both predicates here did filter that way, so {@link ExtractionColumns}'s own-row alternate loop
     * skipped those members and left them with {@code qRowAlt = -1}. A row then put in mode 4 by a
     * reactive-limit pin assembles EMPTY: measured on rte6515's base, where the pinned pass gave F = -target
     * exactly on both members of a pilot-self group, two identical rows and a dead DUMMY_Q column — a
     * singular matrix KLU refuses to factor.
     */
    static GeneratorVoltageControl mainGenVc(LfBus bus) {
        return bus.getGeneratorVoltageControl()
                .<GeneratorVoltageControl>map(VoltageControl::getMainVoltageControl)
                .orElse(null);
    }

    /** True when the bus is a CONTROLLER ELEMENT of a remote (non-local) generator voltage control
     *  — INCLUDING the case where it is also that control's CONTROLLED bus (a generator regulating
     *  its own bus, merged with remote controllers). Such a bus's reactive injection feeds the
     *  group's DISTR_Q rows, never a plain per-bus Q scatter.
     *
     *  <p>Resolved on the MAIN control via {@link VoltageControl#getMainVoltageControl()}, NOT by
     *  filtering {@code getMergeStatus() == MAIN}. A control that has been MERGED into another is
     *  DEPENDENT, and on the dependent object both tests below read the wrong thing: its
     *  {@code controllerElements} is just itself, so {@code isLocalControl()} is TRUE (it appears to
     *  be a generator regulating its own bus alone) and the merged element list is not reachable.
     *  Filtering DEPENDENT away therefore answered FALSE for a bus that really is a controller, and
     *  {@link #computeQRow} then left its Q balance on its own BUS_V row — a row that already hosts
     *  the bus's DISTR_Q equation. The zero-impedance family's direct +-DUMMY_Q coupling landed
     *  there alongside the DISTR_Q derivative and the two CANCELLED: measured on rte6515
     *  BUS-24_COUPLER_1 as kind-3 terms summing +1.0 against the zero-impedance coupling's -1.0, leaving the
     *  coupler's DUMMY_Q in NO equation at all — a STRUCTURALLY ZERO COLUMN, which is invisible in
     *  every residual (the free variable perturbs nothing, so |F| looks healthy while the whole
     *  Newton step is polluted). 45 such columns at 2000 busbar sections.
     *
     *  <p>Generator voltage controls merge exactly when their buses are joined by ZERO-IMPEDANCE
     *  branches, which is why the whole population is busbar couplers and why the defect only
     *  appears once enough busbar-section contingencies retain enough switches. */
    static boolean isRemoteGenVcControllerElement(LfBus bus) {
        return bus.getGeneratorVoltageControl()
                .<GeneratorVoltageControl>map(VoltageControl::getMainVoltageControl)
                .filter(c -> !c.isLocalControl())
                .map(c -> c.getMergedControllerElements().contains(bus))
                .orElse(false);
    }

    /** As below, following the coupler-tree relocation: a relocated CHILD's DISTR_Q moves onto its parent
     *  coupler's DUMMY_Q column, which its own reactive terms already carry (the kind-3 contribution), leaving
     *  its V row free for that coupler's ZERO_V. */
    static int distrQRow(EquationSystem<AcVariableType, AcEquationType> es, LfBus controller,
                         int[] ziParentEdge, int[] dummyQCol) {
        int parent = ziParentEdge == null ? -1 : ziParentEdge[controller.getNum()];
        if (parent >= 0 && dummyQCol[parent] >= 0) {
            return dummyQCol[parent];
        }
        return distrQRow(es, controller);
    }

    /** The device row that carries a controller's DISTR_Q equation. Normally the controller's own
     *  BUS_V row — but when the controller IS the group's CONTROLLED bus, that row already carries
     *  the pilot's BUS_TARGET_V identity, so the DISTR_Q moves to the FREE controller's row: the
     *  row that hosts the pilot bus's reactive equation in every other form too (its Q balance when
     *  the pilot is a plain bus, or when this member is RL-pinned). Without this the two equations
     *  collide on one row — one silently overwrites the other and the group's reactive sharing is
     *  left unsatisfied (measured on a richer-modelling RTE-like network: the device "converged"
     *  with a 0.45 pu DISTR_Q residual and OLF's own re-test then failed the load flow). */
    static int distrQRow(EquationSystem<AcVariableType, AcEquationType> es, LfBus controller) {
        var vc = controller.getGeneratorVoltageControl()
                .filter(c -> c.getMergeStatus() == VoltageControl.MergeStatus.MAIN)
                .orElse(null);
        if (vc != null && vc.getControlledBus() == controller) {
            LfBus free = distrQHostMember(vc, es);
            if (free == null) {
                throw new PowsyblException("GPU AC solver: remote generator voltage control of bus "
                        + controller.getId() + " has no controller");
            }
            return col(es, free.getNum(), AcVariableType.BUS_V);
        }
        return col(es, controller.getNum(), AcVariableType.BUS_V);
    }

    /** The member row that HOSTS the pilot's DISTR_Q when the pilot is one of its own controllers:
     *  OLF's reference controller (the first ENABLED merged controller), else - every controller
     *  pinned in the base state - the first merged controller. In that case the pilot's DISTR_Q is
     *  inactive and only needs a slot; the row modes carry the pinned state (mode 4 on every row).
     *  Measured on rte6515's full busbar scope: the base pins EVERY controller of VL-6003_0's
     *  group, "first enabled" found none and the whole precompute fell to the CPU.
     *  NOT the free-member predicate below: a pinned member is never free (its row hosts its own
     *  Q balance, canonical encoding) - calling it free left a pinned singleton at mode 1 with the
     *  structure-frozen encoding (deviceRemoteSingletonMemberPin, deviceRemoteRobustRealisticVoltage). */
    static LfBus distrQHostMember(GeneratorVoltageControl vc, EquationSystem<AcVariableType, AcEquationType> es) {
        List<LfBus> members = vc.getMergedControllerElements();
        if (hasCandidateTable(vc, es)) {
            return members.get(0);                           // canonical: the first member, whatever is pinned
        }
        return members.stream().filter(vc::isControllerEnabled).findFirst()
                .orElse(members.isEmpty() ? null : members.get(0));
    }

    /**
     * The row that hosts a bus's reactive balance once the COUPLER-TREE RELOCATION has had its say: its own
     * {@code BUS_V} row, or its parent coupler's {@code DUMMY_Q} row when the relocation moved its power
     * equations there. Every consumer of "where does this bus's q live" has to ask this — the reactive-limit
     * pass reads the pin and the q at that row (native {@code reactiveLimitReactivePowerRow}), the candidate table enrols it, the
     * activation extractor targets it, the families write the bus's own q to it in mode 4, and the diagnostic
     * remote-member fill accumulates into it. A consumer left on the own row silently reads ZERO there.
     */
    static int relocatedQRow(int busNum, int[] ziParentEdge, int[] dummyQCol, int[] vCol) {
        int parent = ziParentEdge == null ? -1 : ziParentEdge[busNum];
        return parent >= 0 && dummyQCol[parent] >= 0 ? dummyQCol[parent] : vCol[busNum];
    }

    /** The row that hosts a bus's ACTIVE Q balance in the CURRENT state when its group has a candidate table
     *  (see GpuAcActivationExtractor): a pinned member's own V row; the pilot (a controller of its own group
     *  or not) rides the first ENABLED member's V row (the carrier) and falls back to its own V row when every
     *  controller is pinned — the row the in-kernel mode scan selects. */
    static int stateQRow(LfBus bus, int[] vCol) {
        return vCol[stateQCarrier(bus).getNum()];
    }

    /** {@link #stateQRow} with the COUPLER-TREE RELOCATION applied to the carrier, which is what the candidate
     *  table enrols and therefore what the in-kernel mode scan selects: a relocated member's reactive balance
     *  lives on its parent coupler's {@code DUMMY_Q} row, its own {@code BUS_V} row holding the coupler's
     *  {@code ZERO_V}. Naming the own row instead assigned two equations to it — measured on rte6515's
     *  [5000,6000) window with the relocation on: "device row 12562 assigned twice", the whole precompute lost. */
    static int stateQRow(LfBus bus, int[] vCol, int[] ziParentEdge, int[] dummyQCol) {
        LfBus carrier = stateQCarrier(bus);
        int parent = ziParentEdge == null ? -1 : ziParentEdge[carrier.getNum()];
        return parent >= 0 && dummyQCol[parent] >= 0 ? dummyQCol[parent] : vCol[carrier.getNum()];
    }

    /** The bus whose row carries {@code bus}'s active Q balance: itself, or — for the pilot of a
     *  candidate-table group — the first ENABLED member (the carrier), falling back to itself when every
     *  controller is pinned. */
    private static LfBus stateQCarrier(LfBus bus) {
        GeneratorVoltageControl vc = mainGenVc(bus);
        if (vc != null && vc.getControlledBus() == bus) {
            return vc.getMergedControllerElements().stream()
                    .filter(vc::isControllerEnabled).findFirst().orElse(bus);
        }
        return bus;
    }

    /**
     * True when {@code vc} is a MAIN remote generator voltage control that ExtractionColumns gives a
     * candidate table (plan-B4 / pilot-self rotation): every member and the pilot have a BUS_V column, and a
     * pilot that is one of its own controllers rotates (property on, at least two controllers). Such a
     * group is encoded CANONICALLY — as if every controller were enabled — because the table's in-kernel
     * mode scan re-picks the carrier per scenario; the pin state at extraction time must not leak into the
     * structure (see computeQRow). Keep in step with the enrolment in ExtractionColumns.
     */
    static boolean hasCandidateTable(GeneratorVoltageControl vc, EquationSystem<AcVariableType, AcEquationType> es) {
        if (vc == null || vc.getMergeStatus() != VoltageControl.MergeStatus.MAIN || vc.isLocalControl()) {
            return false;
        }
        LfBus pilot = vc.getControlledBus();
        List<LfBus> members = vc.getMergedControllerElements();
        if (members.isEmpty() || col(es, pilot.getNum(), AcVariableType.BUS_V) < 0) {
            return false;
        }
        boolean pilotSelf = members.contains(pilot);
        if (pilotSelf && members.size() < 2) {
            return false;
        }
        return members.stream().allMatch(m -> col(es, m.getNum(), AcVariableType.BUS_V) >= 0);
    }

    /** The FREE member: the first ENABLED merged controller. None when every controller is pinned. */
    static boolean isFreeRemoteGenVcMember(LfBus bus) {
        GeneratorVoltageControl vc = mainGenVc(bus);
        return vc != null && isRemoteGenVcControllerElement(bus)
                && vc.getMergedControllerElements().stream()
                        .filter(vc::isControllerEnabled).findFirst().map(m -> m == bus).orElse(false);
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
                // NO control-ON filter here (unlike computeQRow's live free pick): the ALTERNATE
                // is computed at a control-OFF base too, where every identity is active — the
                // stable "first with DISTR_RHO inactive" pick keeps the two-way relocation alive.
                LfBranch ctl = tvc.getMergedControllerElements().stream()
                        .filter(c -> !c.isDisabled())
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
    /**
     * The COUPLER-TREE RELOCATION, the general form of the dead-end case below: per zero-impedance spanning tree,
     * every non-root bus's two power equations move onto its PARENT coupler's {@code DUMMY_P} / {@code DUMMY_Q}
     * columns, and every coupler's {@code ZERO_PHI} / {@code ZERO_V} move onto its CHILD bus's {@code PHI} / {@code V}.
     * Each of those four pairings has a structurally non-zero diagonal (+-1, or +-rho for ZERO_V), and the root keeps
     * its own rows, whose diagonal comes from its impedant branches.
     *
     * <p>Why: with the stock pairing a coupler's two equations sit on its own dummy rows, which neither depends on,
     * so a chain of couplers chains zero diagonals. The numeric row repair (hazRepairVals) rescues short chains but
     * not longer ones: measured against OLF's own Newton on an all-generator chain, the device's FIRST step is exact
     * at 3 couplers (2.8e-16) and wrong from 7 on (1.6e+04 at 7, 5.5e+04 at 15) - no zero diagonal, no tiny one and
     * no empty column at that point, just an unpivoted factorization losing the chain. rte6515's busbar scope (366
     * retained couplers) is the same defect at scale: the base load flow's factorization came back wrong
     * (||J.dx-B||inf / ||F||inf = 2.1e+08) and every scenario of a 1000-section window had to fall back to the CPU.
     *
     * @return {@code parentEdgeOfBus[busNum]} = the coupler whose dummy columns host that bus's power equations
     *         (-1 = a root, or not in a relocated tree), and {@code childBusOfEdge[branchNum]} = the bus whose
     *         {@code PHI} / {@code V} host that coupler's zero equations (-1 = not relocated).
     */

    static int[][] ziTreeRelocation(LfNetwork network, EquationSystem<AcVariableType, AcEquationType> es,
                                    int[] phiCol, int[] vCol, int[] qRow, int[] dummyPCol, int[] dummyQCol) {
        int nBuses = network.getBuses().size();
        int nBranches = network.getBranches().size();
        int[] parentEdgeOfBus = new int[nBuses];
        int[] childBusOfEdge = new int[nBranches];
        java.util.Arrays.fill(parentEdgeOfBus, -1);
        java.util.Arrays.fill(childBusOfEdge, -1);
        // Adjacency over the zero-impedance SPANNING-TREE edges only: a non-tree coupler carries no reactive flow
        // and no zero equations to relocate (OLF gives it no DUMMY variables either).
        java.util.Map<Integer, java.util.List<LfBranch>> treeEdges = new java.util.HashMap<>();
        for (LfBranch br : network.getBranches()) {
            if (!br.isZeroImpedance(LoadFlowModel.AC) || !br.isSpanningTreeEdge(LoadFlowModel.AC)
                    || br.getBus1() == null || br.getBus2() == null) {
                continue;
            }
            treeEdges.computeIfAbsent(br.getBus1().getNum(), k -> new java.util.ArrayList<>()).add(br);
            treeEdges.computeIfAbsent(br.getBus2().getNum(), k -> new java.util.ArrayList<>()).add(br);
        }
        // A child's own rows must be FREE for the pairing: its V row takes its coupler's ZERO_V once its reactive
        // equation moves to the parent's DUMMY_Q, so a bus whose V row hosts ANOTHER bus's reactive balance (a
        // group's free controller carries the pilot's) or whose reactive equation is relocated elsewhere (a
        // transformer- or shunt-controlled bus) keeps its rows, and the numeric repair covers it as before.
        boolean[] hostsAnotherReactiveRow = new boolean[nBuses];
        for (int other = 0; other < nBuses; other++) {
            if (qRow[other] < 0) {
                continue;
            }
            for (int bn = 0; bn < nBuses; bn++) {
                if (bn != other && qRow[other] == vCol[bn]) {
                    hostsAnotherReactiveRow[bn] = true;
                }
            }
        }
        // ROOT ORDER: a bus that cannot be a CHILD roots its tree when it can (an impedant branch gives its P row a
        // diagonal), the reference bus first: its PHI row holds BUS_TARGET_PHI, so a coupler's ZERO_PHI cannot take
        // it at all (made a child, it is a double row assignment and the whole extraction is refused). Then a bus
        // whose V row is an identity, hosts another bus's reactive balance, or does not hold its own. Rooted
        // anywhere else the tree is CUT at such a bus: the coupler between it and its parent keeps the stock
        // placement, two zero diagonals the numeric repair fills from a live balance row, and that borrowed pair is
        // an exactly singular 2x2 block an unpivoted factorization cannot eliminate in either order (network.xiidm
        // CKYD: the PV section CKYD_2 cut CLDK off a CKYD_0 root, and every inaccurate batched solve of a
        // 258-section window had its worst residual on CLDK's or its neighbours' rows).
        java.util.List<LfBus> rootOrder = new java.util.ArrayList<>();
        for (LfBus bus : network.getBuses()) {
            if (treeEdges.containsKey(bus.getNum()) && hasImpedantBranch(bus) && hostsPhiIdentity(bus, es)) {
                rootOrder.add(bus);
            }
        }
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            if (treeEdges.containsKey(bn) && hasImpedantBranch(bus) && !hostsPhiIdentity(bus, es)
                    && (hostsVoltageIdentity(bus, es) || hostsAnotherReactiveRow[bn]
                        || !(qRow[bn] == vCol[bn] || isRemoteGenVcControllerElement(bus)))) {
                rootOrder.add(bus);
            }
        }
        rootOrder.addAll(network.getBuses());
        boolean[] seen = new boolean[nBuses];
        for (LfBus bus : rootOrder) {
            int rootNum = bus.getNum();
            if (seen[rootNum] || !treeEdges.containsKey(rootNum) || !hasImpedantBranch(bus)) {
                continue;                                    // the root's own P row needs a real dP/dPHI diagonal
            }
            // Breadth-first from the root: the edge that first reaches a bus is its parent, and the bus is that
            // edge's child. A tree, so every bus is reached once and the pairing is a permutation.
            java.util.Deque<LfBus> queue = new java.util.ArrayDeque<>();
            java.util.List<Integer> childrenOfThisTree = new java.util.ArrayList<>();
            seen[rootNum] = true;
            queue.add(bus);
            while (!queue.isEmpty()) {
                LfBus b = queue.poll();
                for (LfBranch br : treeEdges.getOrDefault(b.getNum(), List.of())) {
                    LfBus other = br.getBus1() == b ? br.getBus2() : br.getBus1();
                    int otherNum = other.getNum();
                    if (seen[otherNum] || phiCol[otherNum] < 0 || vCol[otherNum] < 0
                            || hostsVoltageIdentity(other, es) || hostsPhiIdentity(other, es)) {
                        continue;                            // a pilot's V row is an identity: ZERO_V cannot take it;
                    }                                        // nor ZERO_PHI the reference bus's PHI row
                    seen[otherNum] = true;
                    parentEdgeOfBus[otherNum] = br.getNum();
                    childBusOfEdge[br.getNum()] = otherNum;
                    childrenOfThisTree.add(otherNum);
                    queue.add(other);
                }
            }
            if (System.getenv("OLF_GPU_ZITREE_TRACE") != null) {
                System.err.println("ZITREE root=" + bus.getId() + " couplers=" + childrenOfThisTree.size());
            }
        }
        for (LfBus bus : network.getBuses()) {
            int bn = bus.getNum();
            int parent = parentEdgeOfBus[bn];
            if (parent < 0) {
                continue;
            }
            // CANONICAL, i.e. pin-state independent (the lesson of the canonical extraction): a remote gen-VC
            // controller element's q is SUPPRESSED at extraction (qRow = -1) and its own V row is where its balance
            // comes back — its DISTR_Q while it shares, its own Q once a limit pins it — so its row is the home of
            // its reactive balance whatever the pin state. Reading the pin state here made the base and the
            // security analysis relocate DIFFERENTLY (measured: 44 of 47 couplers in the base extraction against
            // 41 in the SA's, the three differing ones exactly the pilot-self groups the base load flow pins), so
            // the two ran on different structures and every scenario failed.
            boolean ownReactiveOnItsVRow = qRow[bn] == vCol[bn] || isRemoteGenVcControllerElement(bus);
            if (dummyPCol[parent] < 0 || dummyQCol[parent] < 0 || hostsAnotherReactiveRow[bn]
                    || !ownReactiveOnItsVRow) {
                parentEdgeOfBus[bn] = -1;
                childBusOfEdge[parent] = -1;
            }
        }
        if (System.getenv("OLF_GPU_ZITREE_TRACE") != null) {
            int tree = 0;
            int relocated = 0;
            for (LfBranch br : network.getBranches()) {
                if (br.isZeroImpedance(LoadFlowModel.AC) && br.isSpanningTreeEdge(LoadFlowModel.AC)
                        && br.getBus1() != null && br.getBus2() != null) {
                    tree++;
                    if (childBusOfEdge[br.getNum()] >= 0) {
                        relocated++;
                    } else {                                 // why each stock coupler stayed stock: both ends' flags
                        System.err.println("ZITREE_STOCK coupler=" + br.getId()
                                + " dummyCols=" + (dummyPCol[br.getNum()] >= 0 && dummyQCol[br.getNum()] >= 0)
                                + " " + spanningTreeTraceBus(br.getBus1(), es, parentEdgeOfBus, hostsAnotherReactiveRow, qRow, vCol)
                                + " " + spanningTreeTraceBus(br.getBus2(), es, parentEdgeOfBus, hostsAnotherReactiveRow, qRow, vCol));
                    }
                }
            }
            System.err.println("ZITREE_SUMMARY treeCouplers=" + tree + " relocated=" + relocated
                    + " stock=" + (tree - relocated));
        }
        return new int[][] {parentEdgeOfBus, childBusOfEdge};
    }

    /** OLF_GPU_ZITREE_TRACE: one coupler end's placement flags, the inputs of {@link #ziTreeRelocation}'s decisions. */
    private static String spanningTreeTraceBus(LfBus bus, EquationSystem<AcVariableType, AcEquationType> es, int[] parentEdgeOfBus,
                                         boolean[] hostsAnotherReactiveRow, int[] qRow, int[] vCol) {
        int bn = bus.getNum();
        return bus.getId() + "{" + (parentEdgeOfBus[bn] >= 0 ? "child" : "root")
                + (hostsPhiIdentity(bus, es) ? ",phiId" : "")
                + (hostsVoltageIdentity(bus, es) ? ",vId" : "")
                + (hostsAnotherReactiveRow[bn] ? ",hostsOtherQ" : "")
                + (qRow[bn] == vCol[bn] ? ",ownQ" : isRemoteGenVcControllerElement(bus) ? ",remoteCtl" : ",qElsewhere")
                + (hasImpedantBranch(bus) ? ",imp" : ",noImp") + "}";
    }

    /**
     * True when this bus owns a {@code BUS_TARGET_V} identity row: it is where OLF puts the voltage equation of
     * a control it takes part in, so that row is not free for a coupler's {@code ZERO_V}.
     *
     * <p>Pin-state INDEPENDENT, which asking whether the equation is active right now is not — a pilot whose
     * controllers a reactive limit has pinned has no active identity, yet the row is still its own and comes
     * back the moment the limit releases. The owner is OLF's own answer, {@link
     * VoltageControl#findMainVisibleControlledBus()}: of the controls merged over a zero-impedance component
     * exactly ONE is main and visible, and its controlled bus carries the single identity the component gets.
     * Asking instead whether the bus has a control AT ALL excluded every merged dependent controlled bus and
     * cost most of the coverage (measured on rte6515's full busbar scope: 74 of 366 couplers relocated against
     * 329, 793 zero diagonals left and the base's first factorization off by a ratio of 1.3e+10).
     */
    private static boolean hostsVoltageIdentity(LfBus bus, EquationSystem<AcVariableType, AcEquationType> es) {
        return row(es, bus.getNum(), AcEquationType.BUS_TARGET_V) >= 0
                || identityOwner(bus.getGeneratorVoltageControl().orElse(null), bus)
                || identityOwner(bus.getTransformerVoltageControl().orElse(null), bus)
                || identityOwner(bus.getShuntVoltageControl().orElse(null), bus);
    }

    /** True when this bus's {@code PHI} row holds the reference angle ({@code BUS_TARGET_PHI}). */
    private static boolean hostsPhiIdentity(LfBus bus, EquationSystem<AcVariableType, AcEquationType> es) {
        return row(es, bus.getNum(), AcEquationType.BUS_TARGET_PHI) >= 0;
    }

    private static boolean identityOwner(VoltageControl<?> vc, LfBus bus) {
        return vc != null && vc.findMainVisibleControlledBus().map(b -> b == bus).orElse(false);
    }

    private static boolean hasImpedantBranch(LfBus bus) {
        for (LfBranch br : bus.getBranches()) {
            if (!br.isZeroImpedance(LoadFlowModel.AC)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Which side of a zero-impedance branch is a DEAD-END STUB that is SAFE to relocate: a bus whose
     * only branch is this one, with no shunt, no generator and no voltage-dependent load, and with a
     * plain P/Q balance of its own. 1 = side 1, 2 = side 2, 0 = neither (or both, which would be a
     * two-bus island and is left alone).
     *
     * <p>Why relocate at all: a zero-impedance branch puts ZERO_V (v1−rho·v2) on the DUMMY_P row and
     * ZERO_PHI (ph1−ph2) on the DUMMY_Q row, and neither depends on its own dummy — two structurally
     * zero diagonals, harmless while isolated. A dead-end stub adds two more, because its P and Q
     * balances then carry ONLY ∓DUMMY_P / ∓DUMMY_Q, and the four zero pivots CHAIN. cuDSS's AMD
     * ordering is symmetric and assumes a non-zero diagonal; on that chain it perturbs the pivot and
     * propagates the reciprocal (measured DUMMY_P = -1.220703125e+09 and phi = -1.220703125e+22
     * against exact answers of -0.05 and 0). Pairing the same four equations with different columns
     * — the stub's P balance on the DUMMY_P row, its Q balance on the DUMMY_Q row, ZERO_V on the
     * stub's v row and ZERO_PHI on its phi row — makes every diagonal non-zero and the factorization
     * well behaved, with AMD retained.
     *
     * <p>WHY THE CONDITIONS ARE THIS STRICT. Moving a bus's P balance off {@code phiCol} is only safe
     * when NOTHING ELSE is keyed on that row, and several packs are: {@code distributedSlackBusPhiRow} writes each
     * distributed-slack participant's target at {@code phiCol[bn]}, the reactive-capability curve
     * pack has its own P rows, the HVDC family scatters into both ends' P rows, and a load model
     * emits a load model term at the bus's P and Q rows. Excluding generators covers DS, the curves and
     * HVDC at once, because OLF models a converter station as an {@code LfGenerator}. A first,
     * looser version of this test (branch count + shunt only) fired on rte6515 buses that carry
     * those packs and moved the base error from 3.9e-08 to 5.0e-05.
     */
    static int ziDeadEndSide(EquationSystem<AcVariableType, AcEquationType> es, LfBranch br,
                             int[] phiCol, int[] vCol, int[] qRow) {
        if (!br.isSpanningTreeEdge(LoadFlowModel.AC)) {
            return 0;                                        // no ZERO_V/ZERO_PHI to relocate
        }
        boolean s1 = isZiDeadEnd(es, br.getBus1(), br, phiCol, vCol, qRow);
        boolean s2 = isZiDeadEnd(es, br.getBus2(), br, phiCol, vCol, qRow);
        if (s1 == s2) {
            return 0;                                        // neither, or a two-bus island
        }
        return s1 ? 1 : 2;
    }

    private static boolean isZiDeadEnd(EquationSystem<AcVariableType, AcEquationType> es, LfBus bus,
                                       LfBranch br, int[] phiCol, int[] vCol, int[] qRow) {
        if (bus == null) {
            return false;
        }
        int bn = bus.getNum();
        // Exactly one incident branch, counted regardless of state: a base-open branch still packs an
        // open-branch view that scatters into this bus's Q row, and a contingency can reclose one,
        // while the STRUCTURE is built once.
        if (bus.getBranches().size() != 1 || bus.getBranches().get(0) != br) {
            return false;
        }
        if (bus.getShunt().isPresent() || bus.getControllerShunt().isPresent() || bus.getSvcShunt().isPresent()) {
            return false;                                    // a shunt scatters into this bus's Q row
        }
        if (!bus.getGenerators().isEmpty()) {
            return false;                                    // DS rows, curve P rows, HVDC stations
        }
        for (com.powsybl.openloadflow.network.LfLoad load : bus.getLoads()) {
            if (load.getLoadModel().isPresent()) {
                return false;                                // load model term at this bus's P and Q rows
            }
        }
        return phiCol[bn] >= 0 && vCol[bn] >= 0 && qRow[bn] == vCol[bn]
                && row(es, bn, AcEquationType.BUS_TARGET_P) >= 0
                && row(es, bn, AcEquationType.BUS_TARGET_Q) >= 0;
    }

}
