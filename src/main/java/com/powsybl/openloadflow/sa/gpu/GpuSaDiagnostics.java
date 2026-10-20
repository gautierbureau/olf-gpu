/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.contingency.Contingency;
import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.equations.Variable;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfContingency;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.sa.LimitViolationManager;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.limitreduction.LimitReduction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * Diagnostics of the batched security analysis, each behind its OLF_GPU_* switch: the violation cross-check, OLF's
 * residual at the device state, and the device state and post-contingency equation dumps.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuSaDiagnostics {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuSaDiagnostics.class);

    /** How many contingencies the cross-check actually compared — printed at the end of the run, so
     *  "no failures" cannot be confused with "never ran". */
    static final java.util.concurrent.atomic.LongAdder CROSSCHECKED = new java.util.concurrent.atomic.LongAdder();

    private GpuSaDiagnostics() {
    }

    /** {@code OLF_GPU_VIOL_CROSSCHECK=1}: run the stock detection on the SAME state and demand the
     *  same violations, field for field and in the same order. A toy network cannot exercise the
     *  cases that make the flattening delicate — three-winding legs, tie lines, several operational
     *  limits groups, temporary limits — so the check runs on whatever network the caller is running,
     *  and fails loudly on the first contingency that disagrees. */
    static void crossCheck(LfNetwork network, LimitViolationManager manager, List<LimitReduction> limitReductions,
                                   LimitViolationManager reference, SecurityAnalysisParameters saParameters, String contingencyId) {
        CROSSCHECKED.increment();
        var stock = new LimitViolationManager(reference, limitReductions, saParameters.getIncreasedViolationsParameters());
        stock.detectViolations(network);
        List<com.powsybl.contingency.violations.LimitViolation> fromTable = manager.getLimitViolations();
        List<com.powsybl.contingency.violations.LimitViolation> fromStock = stock.getLimitViolations();
        if (fromTable.size() != fromStock.size()) {
            throw new PowsyblException("GPU violation table: " + contingencyId + " reported " + fromTable.size()
                    + " violations, the stock detection " + fromStock.size());
        }
        for (int i = 0; i < fromTable.size(); i++) {
            String a = describe(fromTable.get(i));
            String b = describe(fromStock.get(i));
            if (!a.equals(b)) {
                throw new PowsyblException("GPU violation table: " + contingencyId + " violation " + i
                        + " is " + a + ", the stock detection says " + b);
            }
        }
    }

    /** Every reported field of a violation, so the cross-check catches a wrong scale or a wrong
     *  temporary limit — not only a wrong verdict. */
    static String describe(com.powsybl.contingency.violations.LimitViolation v) {
        return v.getSubjectId() + "|" + v.getSide() + "|" + v.getLimitType() + "|" + v.getOperationalLimitsGroupId()
                + "|" + v.getLimitName() + "|" + v.getAcceptableDuration() + "|" + v.getLimit()
                + "|" + v.getLimitReduction() + "|" + v.getValue();
    }

    /** OLF's residual at the injected device state, worst rows first — see {@code OLF_GPU_SA_RESID}. */
    static void reportOlfResidual(AcLoadFlowContext context,
                                          com.powsybl.openloadflow.equations.EquationSystem<com.powsybl.openloadflow.ac.equations.AcVariableType,
                                                  com.powsybl.openloadflow.ac.equations.AcEquationType> es,
                                          String contingencyId, int topN,
                                          Map<com.powsybl.openloadflow.equations.Variable<com.powsybl.openloadflow.ac.equations.AcVariableType>,
                                                  Integer> baseVariableRows,
                                          int[] islandedBusNums) {
        var ev = context.getEquationVector();
        var tv = context.getTargetVector();
        ev.minus(tv);
        double[] fx = ev.getArray();
        System.err.printf("SA_RESID ctg=%s olfActiveEquations=%d%n", contingencyId, fx.length);
        Integer[] order = new Integer[fx.length];
        for (int i = 0; i < fx.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(Math.abs(fx[b]), Math.abs(fx[a])));
        for (int i = 0; i < Math.min(topN, order.length); i++) {
            int row = order[i];
            var eq = es.getIndex().getEquationAtColumn(row);
            var elem = context.getNetwork().getElement(eq.getType().getElementType(), eq.getElementNum());
            // OLF's own TARGET for that row, and the DEVICE row the same quantity lives on. A residual
            // is (state - target): printing OLF's target next to the device row lets the two be
            // compared directly, which separates "the device solved a different TARGET" from "the
            // device assembled a different EQUATION" —
            // indistinguishable from the residual alone.
            int devRow = -1;
            if (baseVariableRows != null && eq != null) {
                var vr = es.getVariableSet().getVariable(eq.getElementNum(),
                        eq.getType() == com.powsybl.openloadflow.ac.equations.AcEquationType.BUS_TARGET_P
                                ? com.powsybl.openloadflow.ac.equations.AcVariableType.BUS_PHI
                                : com.powsybl.openloadflow.ac.equations.AcVariableType.BUS_V);
                Integer r = vr == null ? null : baseVariableRows.get(vr);
                devRow = r == null ? -1 : r;
            }
            // SIGNED, not |F|. Two buses whose residuals are equal and OPPOSITE are the two ends of a
            // branch whose flow term is missing from both balances; two buses off the same way are not.
            // The magnitude alone cannot tell those apart.
            // Is this bus a NEIGHBOUR OF THE ISLAND? The island is frozen at a flat 1<0 while the live
            // network sits at its real voltages, so any branch still bridging the two would push power
            // into the live side. If the erring buses turn out NOT to touch the island, that whole
            // explanation is dead and the injection comes from somewhere else entirely.
            int toIsland = 0;
            int degree = 0;
            double islandShunt = Double.NaN;
            if (islandedBusNums != null && elem instanceof com.powsybl.openloadflow.network.LfBus b) {
                java.util.Set<Integer> isl = new java.util.HashSet<>();
                for (int bn : islandedBusNums) {
                    isl.add(bn);
                }
                for (var br : b.getBranches()) {
                    degree++;
                    var other = br.getBus1() == b ? br.getBus2() : br.getBus1();
                    if (other != null && isl.contains(other.getNum())) {
                        toIsland++;
                        // The QUANTITATIVE test of "the device dropped the dangling half": if that is
                        // what happened, the residual should be this branch's connected-side shunt
                        // term, b_side * v^2, which OLF keeps and the device deletes with the element.
                        var pi = br.getPiModel();
                        double bSide = br.getBus1() == b ? pi.getB1() : pi.getB2();
                        double v = b.getV();
                        islandShunt = bSide * v * v;
                    }
                }
            }
            // WHAT IS ATTACHED to the bus whose equation is violated. A residual that is the same
            // round number in every contingency is a fixed missing TERM, not a solve difference,
            // and only the bus's own inventory says which term: a shunt's b, the generators' and
            // loads' reactive, and whether OLF considers the bus disabled at all.
            if (elem instanceof LfBus rb) {
                double shB = rb.getShunt().map(sh -> sh.getB()).orElse(0.0)
                        + rb.getControllerShunt().map(sh -> sh.getB()).orElse(0.0)
                        + rb.getSvcShunt().map(sh -> sh.getB()).orElse(0.0);
                double genQ = 0;
                for (LfGenerator g : rb.getGenerators()) {
                    genQ += g.getTargetQ();
                }
                System.err.printf("SA_RESID_BUS ctg=%s id=%s disabled=%s v=%.9f shuntB=%.9f"
                        + " nGen=%d genTargetQ=%.9f loadTargetQ=%.9f vcEnabled=%s%n",
                        contingencyId, rb.getId(), rb.isDisabled(), rb.getV(), shB,
                        rb.getGenerators().size(), genQ, rb.getLoadTargetQ(),
                        rb.isGeneratorVoltageControlEnabled());
            }
            // EVERY branch of the network incident to this bus, disabled ones included. getBranches()
            // is the bus's LIVE list and came back empty while the equation still evaluated non-zero,
            // so one of the two is lying; this says which, and names the branch either way.
            if (elem instanceof LfBus ib) {
                for (LfBranch nb : context.getNetwork().getBranches()) {
                    if (nb.getBus1() == ib || nb.getBus2() == ib) {
                        System.err.printf("SA_RESID_INC ctg=%s bus=%s branch=%s num=%d disabled=%s"
                                + " zeroImp=%s side=%d%n", contingencyId, ib.getId(), nb.getId(),
                                nb.getNum(), nb.isDisabled(),
                                nb.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.AC),
                                nb.getBus1() == ib ? 1 : 2);
                    }
                }
            }
            // The TERMS of the violated equation, each with its element and its value at the
            // device's state. A bus with no branches and no shunt should have a Q balance with no
            // terms at all; when it evaluates to a round non-zero number something IS in there, and
            // only the term list says what.
            int shown = 0;
            for (var t : eq.<com.powsybl.openloadflow.equations.EquationTerm<AcVariableType,
                    AcEquationType>>getTerms()) {
                if (!t.isActive() || shown++ >= 12) {
                    continue;
                }
                double termVal;
                try {
                    termVal = t.eval();
                } catch (RuntimeException ex) {
                    termVal = Double.NaN;
                }
                // Name the term by its VARIABLES. The element number of an
                // EquationTermArrayElementImpl is the array wrapper's, not the branch's, so it
                // identifies nothing; the variable list distinguishes a branch flow (v/phi of
                // two buses) from a zero-impedance DUMMY_Q, which is the whole question when a
                // bus has no branches left.
                StringBuilder vars = new StringBuilder();
                try {
                    for (var vv : t.getVariables()) {
                        vars.append(' ').append(vv.getType()).append('#').append(vv.getElementNum());
                        if (vv.getType() == AcVariableType.BUS_V || vv.getType() == AcVariableType.BUS_PHI) {
                            LfBus vb = context.getNetwork().getBus(vv.getElementNum());
                            vars.append('(').append(vb == null ? "?" : vb.getId())
                                    .append(vb != null && vb.isDisabled() ? ",DISABLED" : "").append(')');
                        }
                    }
                } catch (RuntimeException ex) {
                    vars.append(" <unavailable>");
                }
                System.err.printf("SA_RESID_TERM ctg=%s row=%d term=%s elemType=%s value=%.9f vars:%s%n",
                        contingencyId, row, t.getClass().getSimpleName(),
                        t.getElementType(), termVal, vars);
            }
            // For a violated DISTR_Q, OLF's own q for EVERY member of the group, evaluated at the
            // DEVICE's state. The device believes its sharing equation is satisfied there; if OLF
            // disagrees, these are the numbers that say which member's q the two compute
            // differently - the equation residual alone cannot.
            if (eq.getType() == AcEquationType.DISTR_Q && elem instanceof LfBus db) {
                db.getGeneratorVoltageControl().ifPresent(vc -> {
                    for (LfBus m : vc.getMergedControllerElements()) {
                        System.err.printf("SA_RESID_Q ctg=%s member=%s qPercent=%.9f qAtDeviceState=%.9f"
                                + " loadQ=%.9f v=%.9f%n", contingencyId, m.getId(),
                                m.getRemoteControlReactivePercent(), m.getQ().eval(),
                                m.getLoadTargetQ(), m.getV());
                    }
                });
            }
            System.err.printf("SA_RESID ctg=%s row=%d F=%+.9e eq=%s elem=%d id=%s olfTarget=%.9f devRow=%d"
                            + " degree=%d toIsland=%d islandBranchShunt=%+.9f%n",
                    contingencyId, row, fx[row], eq.getType(), eq.getElementNum(),
                    elem == null ? "?" : elem.getId(), tv.getArray()[row], devRow, degree, toIsland,
                    islandShunt);
        }
    }

    /**
     * {@code OLF_GPU_SA_VAR_DUMP=<file>}: the DEVICE's converged state for one contingency, written as
     * {@code <variableType> <elementId> <value>} — the same shape and keys as OLF's own
     * {@code OLF_VAR_DUMP}, so the two files join directly with no row legend. Restricted to the
     * contingency named by {@code OLF_TRACE_CTG}, which is also what filters the OLF side.
     *
     * <p>Why this pairing exists: on rte6515 the last distributed-power disagreement
     * (BUS-4467_BBS / BUS-5250_BBS, tasks #66) shows a 2.012e-08 RELATIVE state difference over 5430
     * buses that does NOT shrink when the Newton criterion tightens 10^4x. That cannot be convergence
     * depth, so some variable differs by that much — and an aggregate norm can only say that one does.
     * {@code precomputed.baseVariableRows()} is the authoritative variable-to-row map the state
     * injection itself uses, so dumping through it cannot drift from what the device actually solved.
     */
    static void dumpDeviceStateVariables(double[] state, PrecomputedBatch precomputed, LfNetwork network,
                                          Contingency contingency) {
        String path = System.getenv("OLF_GPU_SA_VAR_DUMP");
        String want = System.getenv("OLF_TRACE_CTG");
        if (path == null || want != null && !want.equals(contingency.getId())) {
            return;
        }
        Map<Variable<AcVariableType>, Integer> rows = precomputed.baseVariableRows();
        if (rows == null) {
            return;
        }
        try (java.io.Writer w = new java.io.FileWriter(path, true)) {
            w.write("# VARS ctg=" + contingency.getId() + " device\n");
            for (var e : rows.entrySet()) {
                Variable<AcVariableType> v = e.getKey();
                int row = e.getValue();
                if (row < 0 || row >= state.length) {
                    continue;
                }
                String id;
                try {
                    id = switch (v.getType().getElementType()) {
                        case BUS -> network.getBus(v.getElementNum()).getId();
                        case BRANCH -> network.getBranch(v.getElementNum()).getId();
                        case SHUNT_COMPENSATOR -> network.getShunt(v.getElementNum()).getId();
                        default -> "#" + v.getElementNum();
                    };
                } catch (RuntimeException ex) {
                    id = "#" + v.getElementNum();
                }
                // the DEVICE ROW too: the row's per-scenario MODE is what says which equation governs
                // that bus here (1 = power/DISTR_Q, 4 = its own Q, 0 = identity), and that is the
                // question #66 turns on -- OLF has VL-2669_1..4 on DISTR_Q at convergence while the
                // device appears to leave them on their own BUS_TARGET_Q.
                w.write(v.getType().name() + " " + id + " " + state[row] + " row=" + row + "\n");
            }
        } catch (java.io.IOException | RuntimeException ex) {
            LOGGER.warn("OLF_GPU_SA_VAR_DUMP failed: {}", ex.toString());
        }
    }

    /**
     * {@code OLF_GPU_EQ_DUMP_CTG=<contingency id>}: writes OLF's OWN equation set for the POST-CONTINGENCY
     * state — {@code lfContingency.apply()} has already run when this is called, so the system here is the
     * one the CPU actually solves for that scenario, with inactive equations in {@code [ ]}.
     *
     * <p>This is the reference the device side has been missing. Reasoning about where a relocated row COULD
     * go produced two wrong fixes in a row (a DISTR_Q routed back to its bus's V row made the worst bus
     * difference 150x larger), because a coupler opening does not merely free a row: it changes the GROUP, and
     * OLF re-evaluates {@code updateRemoteVoltageControlEquations} over the live controllers, which can
     * DEACTIVATE an equation rather than move it. The only way to know is to read what OLF has.
     *
     * <p>Written to {@code OLF_GPU_EQ_DUMP_CTG_FILE}, or {@code <id>.olf.post} in the working directory.
     * Also lists the contingency's own elements, so the dump identifies which couplers opened.
     */
    static void dumpPostContingencyEquations(AcLoadFlowContext context, Contingency contingency,
                                              LfContingency lfContingency, LfNetwork network) {
        String want = System.getenv("OLF_GPU_EQ_DUMP_CTG");
        if (want == null || !want.equals(contingency.getId())) {
            return;
        }
        String path = System.getenv().getOrDefault("OLF_GPU_EQ_DUMP_CTG_FILE",
                contingency.getId().replaceAll("[^A-Za-z0-9_.-]", "_") + ".olf.post");
        try (java.io.Writer w = java.nio.file.Files.newBufferedWriter(java.nio.file.Path.of(path))) {
            w.write("# OLF's POST-CONTINGENCY equation system for " + contingency.getId()
                    + " (inactive equations in [ ])\n");
            w.write("# disabled buses: " + lfContingency.getDisabledNetwork().getBuses().stream()
                    .map(LfBus::getId).sorted().toList() + "\n");
            w.write("# disabled branches: " + lfContingency.getDisabledNetwork().getBranches().stream()
                    .map(LfBranch::getId).sorted().toList() + "\n");
            // OLF's own mismatch to distribute, the reference for the device's scheduledActivePowerLoss:
            // gen - load over the DISABLED buses plus the named lost loads/generators that are not on them.
            w.write(String.format("# OLF activePowerLoss=%.9f pu (disconnectedGeneration=%.9f"
                            + " disconnectedLoad=%.9f)%n", lfContingency.getActivePowerLoss(),
                    lfContingency.getDisconnectedGenerationActivePower(),
                    lfContingency.getDisconnectedLoadActivePower()));
            for (LfBus b : lfContingency.getDisabledNetwork().getBuses()) {
                w.write(String.format("#   disabled bus %s: generationTargetP=%.9f loadTargetP=%.9f%n",
                        b.getId(), b.getGenerationTargetP(), b.getLoadTargetP()));
            }
            // OLF writes equations as <symbol><elementNum>, so a legend is what makes the file readable
            // next to the device's row dump, which names elements by id.
            w.write("# bus legend: num = id\n");
            for (LfBus b : network.getBuses()) {
                w.write("#   bus " + b.getNum() + " = " + b.getId() + (b.isDisabled() ? "  [DISABLED]" : "") + "\n");
            }
            w.write("# branch legend: num = id\n");
            for (LfBranch br : network.getBranches()) {
                w.write("#   branch " + br.getNum() + " = " + br.getId()
                        + (br.isDisabled() ? "  [DISABLED]" : "") + "\n");
            }
            context.getEquationSystem().write(w, true);
            // ...and OLF's OWN f(x0) - target per ACTIVE equation at the post-contingency start (the residual
            // its first |f(x0)| line is the norm of): lined up against the device's first evaluation of that
            // slot, it names the equation the two arms evaluate differently before any step blurs it.
            // What this section IS, since a previous version of this comment got it wrong: the
            // POST-contingency equation system evaluated at the PRE-contingency state. The dump runs on
            // the first line of runPostContingencySimulation, before the solver touches x, but
            // lfContingency.apply() has already deactivated equations, which invalidates the vector, so
            // AbstractVector.getArray() RECREATES it against the current index -- verified on rte6515
            // BUS-4467_BBS: fx0.length is 14494, exactly the active-equation count of the
            // post-contingency system section, not the base's. So it is NOT stale.
            //
            // It is still NOT directly comparable with a device fill's norm. Both arms start from the
            // base state, but the device's first fill is taken after its own seeding (frozen islanded
            // rows held at a flat V=1/phi=0, the pre-loss target seeding) and after applyRowMode has
            // turned identity rows into x - target. Measured: OLF norm2=774.03 against the device's
            // 35.95, where OLF's is dominated almost entirely by TWO equations -- distr_q1952/1953 at
            // 519.00 and 518.98 -- so the norms are a statement about those two, not about the network.
            //
            // Hence f and target are written SEPARATELY below. With only the difference printed, a value
            // of 519 cannot be attributed: a large residual, a large target, or a variable the state
            // does not constrain yet all look identical. Those two equations are DISTR_Q rows built on
            // coupler DUMMY_Q variables (dummy_q9161/9162), whose base values are exactly what a
            // pre-contingency state leaves unconstrained -- which the split makes readable.
            try {
                double[] fx0 = context.getEquationVector().getArray();
                double[] tg0 = context.getTargetVector().getArray();
                double n2 = 0;
                w.write("# F0: column type elementNum id residual(f-target) f target\n");
                for (int c = 0; c < fx0.length; c++) {
                    double v = fx0[c] - (c < tg0.length ? tg0[c] : 0.0);
                    n2 += v * v;
                    var eq = context.getEquationSystem().getIndex().getEquationAtColumn(c);
                    String id;
                    try {
                        id = switch (eq.getType().getElementType()) {
                            case BUS -> network.getBus(eq.getElementNum()).getId();
                            case BRANCH -> network.getBranch(eq.getElementNum()).getId();
                            default -> "#" + eq.getElementNum();
                        };
                    } catch (RuntimeException e) {
                        id = "#" + eq.getElementNum();
                    }
                    w.write("F0 " + c + " " + eq.getType() + " " + eq.getElementNum() + " " + id + " " + v
                            + " " + fx0[c] + " " + (c < tg0.length ? tg0[c] : 0.0) + "\n");
                }
                w.write("# F0 norm2=" + Math.sqrt(n2) + "\n");
            } catch (RuntimeException e) {
                w.write("# F0 unavailable: " + e + "\n");
            }
        } catch (java.io.IOException | RuntimeException e) {
            LOGGER.warn("OLF_GPU_EQ_DUMP_CTG failed: {}", e.toString());
        }
        System.err.println("EQ_DUMP_CTG wrote " + path + " for " + contingency.getId());
    }
}
