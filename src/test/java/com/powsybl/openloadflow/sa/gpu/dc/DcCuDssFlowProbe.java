package com.powsybl.openloadflow.sa.gpu.dc;

import com.powsybl.commons.report.ReportNode;
import com.powsybl.contingency.BranchContingency;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.CuDssMatrixFactory;
import com.powsybl.math.matrix.DenseMatrix;
import com.powsybl.math.matrix.MatrixFactory;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.OpenLoadFlowParameters;
import com.powsybl.openloadflow.dc.DcLoadFlowContext;
import com.powsybl.openloadflow.dc.DcLoadFlowParameters;
import com.powsybl.openloadflow.dc.fastdc.ComputedContingencyElement;
import com.powsybl.openloadflow.dc.fastdc.ComputedElement;
import com.powsybl.openloadflow.dc.fastdc.WoodburyEngine;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.network.DisabledNetwork;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.impl.Networks;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Replay harness (kept for the next cuDSS surprise): replays the Woodbury SA's solve sequence on a cuDSS-backed
 * DcLoadFlowContext, step by step, to localise the SIGFPE. Knobs: OLF_K (element count),
 * OLF_SKIP_PRE=1 (no pre-contingency solve), OLF_KLU=1 (KLU instead of cuDSS).
 */
class DcCuDssFlowProbe {

    @Test
    void replay() throws Exception {
        String mode = System.getenv().getOrDefault("OLF_THREAD", "main");
        switch (mode) {
            case "fj" -> java.util.concurrent.CompletableFuture.runAsync(this::body).get();       // common ForkJoinPool, like the SA
            case "lcm" -> java.util.concurrent.CompletableFuture.runAsync(this::body,
                    com.powsybl.computation.local.LocalComputationManager.getDefault().getExecutor()).get();
            case "thread" -> {
                Thread t = new Thread(this::body, "probe");
                t.start();
                t.join();
            }
            default -> body();
        }
    }

    private void body() {
        org.junit.jupiter.api.Assumptions.assumeTrue(java.nio.file.Files.exists(Path.of("rte6515_full.xiidm.gz")), "rte6515_full.xiidm.gz not found");
        org.junit.jupiter.api.Assumptions.assumeTrue("1".equals(System.getenv("OLF_KLU")) || CuDssMatrixFactory.isAvailable(), "cuDSS not available");
        Network network = Network.read(Path.of("rte6515_full.xiidm.gz"));
        LoadFlowParameters lfp = JsonLoadFlowParameters.read(Path.of("rte-loadflow-default-parameters.json"));
        lfp.setDc(true);
        MatrixFactory mf = "1".equals(System.getenv("OLF_KLU")) ? new SparseMatrixFactory() : new CuDssMatrixFactory();
        DcLoadFlowParameters dcp = OpenLoadFlowParameters.createDcParameters(network, lfp, OpenLoadFlowParameters.get(lfp),
                mf, new EvenShiloachGraphDecrementalConnectivityFactory<>(), false);
        dcp.getNetworkParameters().setMinImpedance(true).setHvdcAcEmulation(false);
        dcp.getEquationSystemCreationParameters().setForcePhaseControlOffAndAddAngle1Var(true);
        int k = Integer.parseInt(System.getenv().getOrDefault("OLF_K", "4"));
        LfNetwork lf;
        com.powsybl.openloadflow.network.impl.LfNetworkList lfNetworks = null;
        if ("1".equals(System.getenv("OLF_SA_LIKE"))) {
            // exactly what AbstractSecurityAnalysis.runSync does: propagated contingencies -> topo config -> breakers
            List<com.powsybl.contingency.Contingency> contingencies = DcWoodburyBaselineProbe.mainComponentBranchContingencies(network);
            contingencies = contingencies.subList(0, Math.min(k, contingencies.size()));
            com.powsybl.openloadflow.network.LfTopoConfig topoConfig = new com.powsybl.openloadflow.network.LfTopoConfig();
            var creationParameters = new com.powsybl.openloadflow.network.impl.PropagatedContingencyCreationParameters()
                    .setContingencyPropagation(true)
                    .setShuntCompensatorVoltageControlOn(false)
                    .setSlackDistributionOnConformLoad(lfp.getBalanceType() == LoadFlowParameters.BalanceType.PROPORTIONAL_TO_CONFORM_LOAD)
                    .setHvdcAcEmulation(false);
            com.powsybl.openloadflow.network.impl.PropagatedContingency.createList(network, contingencies, topoConfig, creationParameters);
            dcp.getNetworkParameters().setBreakers(topoConfig.isBreaker());
            System.out.printf("FLOW-PROBE SA-like: breakers=%s%n", topoConfig.isBreaker());
            lfNetworks = Networks.loadWithReconnectableElements(network, topoConfig, dcp.getNetworkParameters(), ReportNode.NO_OP);
            lf = lfNetworks.getLargest().orElseThrow();
        } else {
            lf = Networks.load(network, dcp.getNetworkParameters()).stream()
                    .filter(n -> n.getValidity() == LfNetwork.Validity.VALID).findFirst().orElseThrow();
        }
        try (DcLoadFlowContext ctx = new DcLoadFlowContext(lf, dcp, false)) {
            System.out.printf("FLOW-PROBE es columns=%d variables=%d buses=%d branches=%d%n", ctx.getEquationSystem().getIndex().getColumnCount(),
                    ctx.getEquationSystem().getIndex().getRowCount(), lf.getBuses().size(), lf.getBranches().size());
            com.powsybl.math.matrix.SparseMatrix m = (com.powsybl.math.matrix.SparseMatrix) ctx.getJacobianMatrix().getMatrix();
            int[] ap = m.getColumnStart();
            int[] ai = m.getRowIndices();
            double[] ax = m.getValues();
            int nnz = ap[ap.length - 1];
            double minAbs = Double.MAX_VALUE;
            double maxAbs = 0;
            int emptyCols = 0;
            for (int j = 0; j + 1 < ap.length; j++) {
                if (ap[j] == ap[j + 1]) {
                    emptyCols++;
                }
                for (int kk = ap[j]; kk < ap[j + 1]; kk++) {
                    minAbs = Math.min(minAbs, Math.abs(ax[kk]));
                    maxAbs = Math.max(maxAbs, Math.abs(ax[kk]));
                }
            }
            System.out.printf("FLOW-PROBE matrix n=%d nnz=%d emptyColumns=%d min|v|=%.3e max|v|=%.3e%n", m.getRowCount(), nnz, emptyCols, minAbs, maxAbs);
            String dump = System.getenv("OLF_MATRIX_OUT");
            if (dump != null) {
                try (java.io.DataOutputStream o = new java.io.DataOutputStream(new java.io.BufferedOutputStream(new java.io.FileOutputStream(dump)))) {
                    o.writeInt(m.getRowCount());
                    o.writeInt(ap.length);
                    for (int v : ap) {
                        o.writeInt(v);
                    }
                    o.writeInt(nnz);
                    for (int kk = 0; kk < nnz; kk++) {
                        o.writeInt(ai[kk]);
                    }
                    for (int kk = 0; kk < nnz; kk++) {
                        o.writeDouble(ax[kk]);
                    }
                } catch (java.io.IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
                System.out.printf("FLOW-PROBE matrix dumped to %s%n", dump);
            }
            if (!"1".equals(System.getenv("OLF_SKIP_PRE"))) {
                double[] x = WoodburyEngine.runDcLoadFlowWithModifiedTargetVector(ctx, new DisabledNetwork(), ReportNode.NO_OP);
                System.out.printf("FLOW-PROBE pre-contingency solve ok, |x|max=%.4f%n", java.util.Arrays.stream(x).map(Math::abs).max().orElse(0));
            }
            List<ComputedContingencyElement> elements = new ArrayList<>();
            for (LfBranch b : k > 0 ? lf.getBranches() : List.<LfBranch>of()) {
                if (b.getBus1() != null && b.getBus2() != null && !b.isZeroImpedance(com.powsybl.openloadflow.network.LoadFlowModel.DC)) {
                    elements.add(new ComputedContingencyElement(new BranchContingency(b.getId()), lf, ctx.getEquationSystem(), ctx.getParameters().getEquationSystemCreationParameters()));
                    if (elements.size() == k) {
                        break;
                    }
                }
            }
            ComputedElement.setComputedElementIndexes(elements);
            System.out.printf("FLOW-PROBE %d elements -> calculateElementsStates%n", elements.size());
            DenseMatrix states = ComputedElement.calculateElementsStates(ctx, elements);
            double max = 0;
            for (int j = 0; j < states.getColumnCount(); j++) {
                for (int i = 0; i < states.getRowCount(); i++) {
                    max = Math.max(max, Math.abs(states.get(i, j)));
                }
            }
            System.out.printf("FLOW-PROBE states %dx%d ok, |S|max=%.4f%n", states.getRowCount(), states.getColumnCount(), max);
        } finally {
            if (lfNetworks != null) {
                lfNetworks.close();
            }
        }
    }
}
