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
 * The COUPLER-TREE RELOCATION ({@code -Dolf.gpu.ziTreeRelocation=true}, see
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
        System.setProperty("olf.gpu.ziTreeRelocation", "true");
        try {
            assertChainStepsMatchOlf();
        } finally {
            System.clearProperty("olf.gpu.ziTreeRelocation");
        }
    }

    private static void assertChainStepsMatchOlf() {
        for (int couplers : new int[] {3, 7, 15}) {
            LfNetwork network = Networks.load(chain(couplers),
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
                    d.drRow(), d.drCol(), d.drCoef(), d.dbRow(), d.dbCol(), d.dbCoef(),
                // the group tables + ob/zi alternates travel with the packs: a pilot-self group enrolled
                // in the candidate table (rotation) encodes -2-g in the q alternates, refused without them
                d.qrPtr(), d.qrRows(), d.obQAlt(), d.ziQAlt());

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
                        System.err.println("F0_TRACE chain=" + couplers + " col=" + c + " " + e.getType() + " elem=" + e.getElementNum()
                                + " f0=" + f0[c]);
                    }
                    try (java.io.Writer w = java.nio.file.Files.newBufferedWriter(java.nio.file.Path.of(
                            System.getenv("OLF_GPU_F0_TRACE") + ".chain" + couplers + ".eq"))) {
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
            assertTrue(worst < 1e-12, "coupler chain of " + couplers + ": the device's first step must be OLF's,"
                    + " worst difference " + worst);
        }
    }
}
