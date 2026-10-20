/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.equations.vector.gpu;

import com.powsybl.iidm.network.Bus;
import com.powsybl.iidm.network.Network;
import com.powsybl.math.matrix.DenseMatrixFactory;
import com.powsybl.openloadflow.ac.AcTargetVector;
import com.powsybl.openloadflow.ac.equations.AcEquationSystemCreator;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.equations.StateVector;
import com.powsybl.openloadflow.network.AbstractLoadFlowNetworkFactory;
import com.powsybl.openloadflow.network.FirstSlackBusSelector;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkParameters;
import com.powsybl.openloadflow.network.LoadFlowModel;
import com.powsybl.openloadflow.network.impl.Networks;
import com.powsybl.openloadflow.network.util.UniformValueVoltageInitializer;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * The COUPLER-TREE RELOCATION (see
 * {@link GpuAcRowIndexing#ziTreeRelocation}) on the shape that needs it: a chain of zero-impedance couplers with a
 * generator on every bus, so OLF merges their voltage controls and the numeric row repair is the only thing holding
 * the factorization together.
 *
 * <p>With the relocation OFF (the default: dead-end stubs only) the device's FIRST Newton step, measured against
 * OLF's own dense Newton on the same equation system, is exact at 3 couplers and wrong from 7 on — 1.6e+04 at 7,
 * 5.5e+04 at 15 — with no zero diagonal, no tiny one and no empty column to point at, just an unpivoted
 * factorization losing a chain. Relocated, the same steps are exact.
 *
 * <p>KNOWN LIMITS, why the relocation is off by default: the error still grows with chain depth (1.6e-09 at 31
 * couplers, 2.1e+00 at 63), and on rte6515 it fixes the busbar window [5000,6000) (base on the device, violations
 * 0 differing against the CPU) while REGRESSING others — the base fails on the 500-section window and on
 * [0,1000), [2000,3000) reports 154 violations differently, and two security-analysis fixtures stop engaging the
 * batch (deviceMiddleSectionLoss*). The pairing has to agree with islanding, reactive-limit pins and the
 * per-scenario coupler changes a busbar outage makes, which it does not yet.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuZiCouplerTreeRelocationTest extends AbstractLoadFlowNetworkFactory {

    private static Network chain(int couplers) {
        Network network = Network.create("zi-chain-" + couplers, "code");
        Bus b0 = createBus(network, "s", "b0");
        createLoad(b0, "ld0", couplers, couplers / 2.0);
        Bus prev = createBus(network, "s", "b1");
        createGenerator(prev, "g1", 1, 1.0);
        createLine(network, b0, prev, "l01", 0.1);
        for (int i = 2; i <= couplers + 1; i++) {
            Bus b = createBus(network, "s", "b" + i);
            createGenerator(b, "g" + i, 1, 1.0);
            createLine(network, prev, b, "c" + i, 0.0, 0.0);
            prev = b;
        }
        return network;
    }

    /** Five sections chained by couplers, s0..s4 in that bus order, every one tied to the external bus by a line
     *  and only s2 regulating: s2's V row holds the tree's BUS_TARGET_V, so s2 cannot be a child. */
    private static Network chainRegulatedInTheMiddle() {
        Network network = Network.create("zi-chain-regulated-middle", "code");
        Bus ext = createBus(network, "x", "ext");
        createGenerator(ext, "gs", 5, 1.0);
        createLoad(ext, "lx", 2, 0.5);
        Bus prev = null;
        for (int k = 0; k < 5; k++) {
            Bus s = createBus(network, "s", "s" + k);
            if (k == 2) {
                createGenerator(s, "g2", 1, 1.02);
            } else {
                createLoad(s, "ld" + k, 0.5, 0.2);
            }
            createLine(network, ext, s, "l" + k, 0.1);
            if (prev != null) {
                createLine(network, prev, s, "c" + (k - 1) + k, 0.0, 0.0);
            }
            prev = s;
        }
        return network;
    }

    /** A two-section plant, the shape of 36 of rte6515's 37 stock couplers: a generator on each section, one merged
     *  control whose main visible controlled bus is s0 (the higher target), so s0's V row holds the identity, and
     *  s0 has NO impedant branch - it reaches the grid through the coupler alone - so its P row has no diagonal of
     *  its own. s0 can be neither a root (no dP/dphi) nor a child (the identity), so c01 keeps the STOCK placement
     *  and the numeric row repair, chained: s0's P row borrows ZERO_PHI, which borrows a P row. */
    private static Network twoSectionPlantGeneratorOnTheStub() {
        Network network = Network.create("zi-two-section-plant", "code");
        Bus ext = createBus(network, "x", "ext");
        createGenerator(ext, "gs", 5, 1.0);
        createLoad(ext, "lx", 3, 0.5);
        Bus s0 = createBus(network, "s", "s0");
        createGenerator(s0, "g0", 1, 1.03);
        Bus s1 = createBus(network, "s", "s1");
        createGenerator(s1, "g1", 1, 1.02);
        createLoad(s1, "ld1", 0.5, 0.2);
        createLine(network, ext, s1, "l1", 0.1);
        createLine(network, s0, s1, "c01", 0.0, 0.0);
        return network;
    }

    /** A zero-impedance phase shifter (x = 0, a fixed 5 degree tap) between two load sections that each reach the
     *  external bus by a line: its ZERO_PHI target is the phase shift, not 0, so it tells the equation apart from
     *  ZERO_V (target 0) in whatever rows the two sit on. */
    private static Network zeroImpedancePhaseShifter() {
        Network network = Network.create("zi-phase-shifter", "code");
        Bus ext = createBus(network, "s", "ext");
        createGenerator(ext, "gs", 5, 1.0);
        createLoad(ext, "lx", 2, 0.5);
        Bus s0 = createBus(network, "s", "s0");
        createLoad(s0, "ld0", 1, 0.3);
        Bus s1 = createBus(network, "s", "s1");
        createLoad(s1, "ld1", 1, 0.3);
        createLine(network, ext, s0, "l0", 0.1);
        createLine(network, ext, s1, "l1", 0.1);
        createTransformer(network, "s", s0, s1, "ps01", 0.0, 1.0).newPhaseTapChanger()
                .setTapPosition(0)
                .setRegulating(false)
                .setRegulationMode(com.powsybl.iidm.network.PhaseTapChanger.RegulationMode.CURRENT_LIMITER)
                .setRegulationValue(100)
                .setRegulationTerminal(network.getTwoWindingsTransformer("ps01").getTerminal2())
                .beginStep().setAlpha(5.0).setRho(1.0).setR(0).setX(0).setG(0).setB(0).endStep()
                .add();
        return network;
    }

    /** The device's step is OLF's with the ZERO_PHI target where the ZERO_PHI equation sits, relocated or stock:
     *  the target map (GpuAcActivationExtractor) and the native placement (ziZeroVRow / ziZeroPhiRow) name the same
     *  dummy row for each zero equation. With the two crossed, a phase-shifting coupler's step is wrong by the
     *  shift on both of its sections. */
    @Test
    void zeroImpedancePhaseShifterStepsWithOlf() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available");
        LfNetwork network = Networks.load(zeroImpedancePhaseShifter(),
                new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        LfBranch ps = network.getBranchById("ps01");
        assertTrue(ps.isZeroImpedance(LoadFlowModel.AC) && ps.isSpanningTreeEdge(LoadFlowModel.AC), "a zero-impedance tree edge");
        assertTrue(Math.abs(LfBranch.getA(ps)) > 1e-3, "with a phase shift: the ZERO_PHI target is not 0");
        assertFirstStepIsOlfs(zeroImpedancePhaseShifter(), "zero-impedance phase shifter, relocated");
    }

    /** The plant's coupler stays stock (the fixture is what it claims to be): s0 is a stub with the identity, so
     *  the relocation has no child for c01 and the step test below runs on the repaired stock placement. */
    @Test
    void twoSectionPlantKeepsItsCouplerStock() {
        LfNetwork network = Networks.load(twoSectionPlantGeneratorOnTheStub(),
                new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        ExtractionColumns columns = ExtractionColumns.of(network, es);
        LfBranch c01 = network.getBranchById("c01");
        LfBus s0 = network.getBusById("s0_vl_0");
        assertTrue(c01.isZeroImpedance(LoadFlowModel.AC) && c01.isSpanningTreeEdge(LoadFlowModel.AC));
        assertTrue(s0.getBranches().stream().allMatch(b -> b.isZeroImpedance(LoadFlowModel.AC)), "s0 is a stub");
        assertTrue(s0.isGeneratorVoltageControlled(), "s0 holds the plant's voltage identity");
        assertEquals(-1, columns.ziChildBus()[c01.getNum()], "c01 is stock: neither section can be its child");
    }

    /** The tree is rooted at the section that cannot be a child. Rooted at the first bus instead (s0), the walk
     *  stopped at s2 and left c12 in the stock placement: two zero diagonals the numeric repair fills from s1's
     *  live balance rows, an exactly singular 2x2 block for an unpivoted factorization (network.xiidm CKYD: every
     *  inaccurate batched solve of a 258-section window had its worst residual there). */
    @Test
    void couplerTreeIsRootedAtItsRegulatedSection() {
        LfNetwork network = Networks.load(chainRegulatedInTheMiddle(),
                new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        ExtractionColumns columns = ExtractionColumns.of(network, es);
        List<String> stock = network.getBranches().stream()
                .filter(b -> b.isZeroImpedance(LoadFlowModel.AC) && b.isSpanningTreeEdge(LoadFlowModel.AC))
                .filter(b -> columns.ziChildBus()[b.getNum()] < 0)
                .map(LfBranch::getId)
                .toList();
        assertEquals(List.of(), stock, "every coupler of the tree is relocated");
        LfBus regulated = network.getBuses().stream().filter(b -> b.getId().startsWith("s2")).findFirst().orElseThrow();
        assertEquals(-1, columns.ziParentEdge()[regulated.getNum()], "the regulated section is the root");
    }

    /** Relocated, a chain the repair cannot hold (7 and 15 couplers) steps exactly with OLF. */
    @Test
    void relocatedCouplerChainStepsWithOlf() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available");
        assertChainStepsMatchOlf();
    }

    /** The two-section plant's coupler is stock even with the relocation on (neither section can be its child),
     *  and the device's first step is OLF's all the same: its repaired blocks are not singular. */
    @Test
    void twoSectionPlantFirstStepIsOlfs() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available");
        assertFirstStepIsOlfs(twoSectionPlantGeneratorOnTheStub(), "two-section plant");
    }

    private static void assertChainStepsMatchOlf() {
        for (int couplers : new int[] {3, 7, 15}) {
            assertFirstStepIsOlfs(chain(couplers), "coupler chain of " + couplers);
        }
    }

    /** The device's FIRST Newton step from the flat start against OLF's own (KLU on the same system), at 1e-12. */
    private static void assertFirstStepIsOlfs(Network iidm, String what) {
        LfNetwork network = Networks.load(iidm,
                new LfNetworkParameters().setSlackBusSelector(new FirstSlackBusSelector())).get(0);
        EquationSystem<AcVariableType, AcEquationType> es = new AcEquationSystemCreator(network).create();
        AcSolverUtil.initStateVector(network, es, new UniformValueVoltageInitializer());
        AcTargetVector target = new AcTargetVector(network, es);
        EquationVector<AcVariableType, AcEquationType> eq = new EquationVector<>(es);
        int n = es.getIndex().getSortedVariablesToFind().size();
        double[] x0 = es.getStateVector().get().clone();

        GpuAcData d = GpuAcDataExtractor.extract(network, es);
        GpuAcActivation activation = GpuAcActivationExtractor.extractActivation(network, es, n);
        double[] stableTargets = activation.applyTargets(target.getArray());
        double[] gpu = GpuAcNewtonSolver.solve(x0, stableTargets, activation.rowMode(), 1, 0.0,
                d.cbIn(), d.cbIdx(), d.obIn(), d.obIdx(), d.shIn(), d.shIdx(), d.hvIn(), d.hvIdx(),
                d.dqElem(), d.dqSide(), d.dqWeight(), d.dqRow(), d.dqKind(),
                d.ziIn(), d.ziIdx(),
                d.distrRho().row(), d.distrRho().column(), d.distrRho().coefficient(), d.distrShuntB().row(), d.distrShuntB().column(), d.distrShuntB().coefficient(),
            // the group tables + ob/zi alternates travel with the packs: a pilot-self group enrolled
            // in the candidate table (rotation) encodes -2-g in the q alternates, refused without them
            d.reactiveRelocation().groupStart(), d.reactiveRelocation().candidateRows(), d.obQAlt(), d.ziQAlt());

        StateVector sv = es.getStateVector();
        sv.set(x0.clone());
        try (JacobianMatrix<AcVariableType, AcEquationType> j = new JacobianMatrix<>(es, new DenseMatrixFactory())) {
            eq.minus(target);
            if (System.getenv("OLF_GPU_F0_TRACE") != null) {
                // OLF's own f(x0) - target per equation, beside the device's first fill (OLF_GPU_VALS_DUMP_ANY):
                // the row the two arms evaluate differently, before any step blurs it (see GpuNewtonRaphson)
                double[] f0 = eq.getArray();
                for (int c = 0; c < f0.length; c++) {
                    var e = es.getIndex().getEquationAtColumn(c);
                    System.err.println("F0_TRACE " + what + " col=" + c + " " + e.getType() + " elem=" + e.getElementNum()
                            + " f0=" + f0[c]);
                }
                try (java.io.Writer w = java.nio.file.Files.newBufferedWriter(java.nio.file.Path.of(
                        System.getenv("OLF_GPU_F0_TRACE") + "." + what.replace(' ', '_') + ".eq"))) {
                    GpuEquationDump.write(w, network, es, d, activation.rowMode());
                } catch (java.io.IOException ex) {
                    throw new java.io.UncheckedIOException(ex);
                }
            }
            double[] dx = eq.getArray().clone();
            j.solveTransposed(dx);
            sv.minus(dx);
        }
        double[] cpu = sv.get().clone();
        double worst = 0;
        for (int i = 0; i < n; i++) {
            worst = Math.max(worst, Math.abs(gpu[i] - cpu[i]));
        }
        assertTrue(worst < 1e-12, what + ": the device's first step must be OLF's,"
                + " worst difference " + worst);
    }
}
