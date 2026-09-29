/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.contingency.BranchContingency;
import com.powsybl.contingency.Contingency;
import com.powsybl.iidm.network.Branch;
import com.powsybl.iidm.network.Network;
import com.powsybl.loadflow.LoadFlowParameters;
import com.powsybl.loadflow.json.JsonLoadFlowParameters;
import com.powsybl.math.matrix.SparseMatrixFactory;
import com.powsybl.openloadflow.CommonTestConfig;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuTestPaths;
import com.powsybl.openloadflow.graph.EvenShiloachGraphDecrementalConnectivityFactory;
import com.powsybl.openloadflow.sa.AbstractOpenSecurityAnalysisTest;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisParameters;
import com.powsybl.openloadflow.sa.OpenSecurityAnalysisProvider;
import com.powsybl.security.PostContingencyComputationStatus;
import com.powsybl.security.SecurityAnalysisParameters;
import com.powsybl.security.SecurityAnalysisResult;
import com.powsybl.security.results.PostContingencyResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * GPU SA vs CPU SA on rte6515_full: ALL branch N-1 contingencies, RTE production parameters,
 * GPU single-thread vs CPU 8-thread. Asserts:
 * <ul>
 *   <li>0 violation count difference</li>
 *   <li>every contingency solved on GPU (0 CPU fallback)</li>
 *   <li>identical post-contingency status counts</li>
 * </ul>
 * Reports wall-clock timing for both arms.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
class GpuVsCpuRte6515SaTest extends AbstractOpenSecurityAnalysisTest {

    /** OLF_GPU_SA_CASE_FILE=<path>: the network to compare on (default rte6515_full.xiidm.gz). The
     *  same harness runs on network.xiidm at the repo root: every contingency kind except busbar
     *  sections, branches first, then the other kinds once parity holds. Real network data: never
     *  read its content, only run it. */
    private static final Path CASE_FILE = Path.of(System.getenv().getOrDefault("OLF_GPU_SA_CASE_FILE", "rte6515_full.xiidm.gz"));
    private static final Path PARAMS_FILE = Path.of("rte-loadflow-default-parameters.json");

    static {
        GpuTestPaths.init();
    }

    GpuVsCpuRte6515SaTest(CommonTestConfig commonTestConfig) {
        super(commonTestConfig);
    }

    @BeforeEach
    void checkGpu() {
        assumeTrue(GpuAcNewtonSolver.isAvailable(), "GPU native lib not available");
        assumeTrue(Files.exists(CASE_FILE), CASE_FILE + " not found at repo root");
        assumeTrue(Files.exists(PARAMS_FILE), PARAMS_FILE + " not found at repo root");
        securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(),
                new EvenShiloachGraphDecrementalConnectivityFactory<>());
    }

    /**
     * The contingency set for this run. {@code OLF_GPU_SA_CTG_KINDS} picks the kinds as a comma list of
     * {@code branch}, {@code generator} and {@code load} (default {@code branch}, the 9042-contingency
     * reference set — never quote a different set as the scope result).
     *
     * <p>{@code OLF_GPU_SA_GEN_MIN_KV} keeps generators at or above a nominal voltage (default 400) and
     * {@code OLF_GPU_SA_LOAD_TOP} the N biggest loads by P0 (default 50), so a first pass on the two new
     * kinds is a small, deliberately chosen slice rather than every injection in the network.
     */
    /** Is this terminal part of the network OLF actually SOLVES: connected, with a bus view bus, in
     *  the MAIN connected and synchronous component? An element that fails this is not in OLF's
     *  LfNetwork, so a contingency on it produces no result on EITHER arm - the harness used to
     *  create one anyway (one N-1 per IIDM branch), and every one cost a mapper rejection plus a
     *  pointless CPU replay. Measured on network.xiidm's full branch scope: 709 of 11249, made of
     *  384 half-open branches whose live side is outside the main component, 285 out of service at
     *  both ends, 38 wholly outside it and 2 outside the main synchronous component. */
    private static boolean inSolvedNetwork(com.powsybl.iidm.network.Terminal t) {
        if (t == null || !t.isConnected()) {
            return false;
        }
        com.powsybl.iidm.network.Bus bus = t.getBusView().getBus();
        return bus != null
                && bus.getConnectedComponent() != null
                && bus.getConnectedComponent().getNum() == com.powsybl.iidm.network.ComponentConstants.MAIN_NUM
                && bus.getSynchronousComponent() != null
                && bus.getSynchronousComponent().getNum() == com.powsybl.iidm.network.ComponentConstants.MAIN_NUM;
    }

    /** Count what the filter removed, so a scope report says it out loud instead of hiding it in a
     *  batching share. */
    private static <T> List<T> keepSolved(List<T> all, java.util.function.Predicate<T> live, String kind) {
        List<T> kept = all.stream().filter(live).toList();
        if (kept.size() != all.size()) {
            System.out.printf("  CTG_FILTER %s: %d of %d dropped (out of service, or outside the "
                    + "network OLF solves - no result on either arm)%n", kind, all.size() - kept.size(), all.size());
        }
        return kept;
    }

    private static List<Contingency> allBranchContingencies(Network network) {
        String kinds = System.getenv().getOrDefault("OLF_GPU_SA_CTG_KINDS", "branch");
        List<Contingency> all = new java.util.ArrayList<>();
        if (kinds.contains("branch")) {
            keepSolved(network.getBranchStream().toList(),
                    b -> inSolvedNetwork(b.getTerminal1()) || inSolvedNetwork(b.getTerminal2()), "branch")
                    .forEach(b -> all.add(new Contingency(b.getId(), new BranchContingency(b.getId()))));
        }
        if (kinds.contains("generator")) {
            double minKv = Double.parseDouble(System.getenv().getOrDefault("OLF_GPU_SA_GEN_MIN_KV", "400"));
            java.util.Map<Double, Long> byKv = new TreeMap<>();
            network.getGeneratorStream().forEach(g ->
                    byKv.merge(g.getTerminal().getVoltageLevel().getNominalV(), 1L, Long::sum));
            System.out.printf("  generators by nominal voltage: %s%n", byKv);
            List<String> gens = keepSolved(network.getGeneratorStream()
                    .filter(g -> g.getTerminal().getVoltageLevel().getNominalV() >= minKv).toList(),
                    g -> inSolvedNetwork(g.getTerminal()), "generator").stream()
                    .map(com.powsybl.iidm.network.Generator::getId)
                    .toList();
            gens.forEach(id -> all.add(new Contingency(id, new com.powsybl.contingency.GeneratorContingency(id))));
            System.out.printf("  + %d generator contingencies at or above %.0f kV%n", gens.size(), minKv);
        }
        if (kinds.contains("load")) {
            int top = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_LOAD_TOP", "50"));
            List<String> loads = keepSolved(network.getLoadStream()
                    .sorted(java.util.Comparator.comparingDouble(
                            (com.powsybl.iidm.network.Load l) -> l.getP0()).reversed()).toList(),
                    l -> inSolvedNetwork(l.getTerminal()), "load").stream()
                    .limit(top)
                    .map(com.powsybl.iidm.network.Load::getId)
                    .toList();
            loads.forEach(id -> all.add(new Contingency(id, new com.powsybl.contingency.LoadContingency(id))));
            System.out.printf("  + %d load contingencies (the biggest by P0)%n", loads.size());
        }
        // The non-injection equipment kinds. Each is a whole-element outage like a branch, but each
        // reaches different machinery: an HVDC line carries the AC-emulation freeze, an SVC is a
        // slope-controlled (mode-2) row, and a shunt moves the bus susceptance.
        if (kinds.contains("hvdc")) {
            List<String> ids = keepSolved(network.getHvdcLineStream().toList(),
                    h -> inSolvedNetwork(h.getConverterStation1().getTerminal())
                            || inSolvedNetwork(h.getConverterStation2().getTerminal()), "hvdc").stream()
                    .map(com.powsybl.iidm.network.HvdcLine::getId).toList();
            ids.forEach(id -> all.add(new Contingency(id, new com.powsybl.contingency.HvdcLineContingency(id))));
            System.out.printf("  + %d HVDC line contingencies%n", ids.size());
        }
        if (kinds.contains("svc")) {
            List<String> ids = keepSolved(network.getStaticVarCompensatorStream().toList(),
                    v -> inSolvedNetwork(v.getTerminal()), "svc").stream()
                    .map(com.powsybl.iidm.network.StaticVarCompensator::getId).toList();
            ids.forEach(id -> all.add(new Contingency(id,
                    new com.powsybl.contingency.StaticVarCompensatorContingency(id))));
            System.out.printf("  + %d static var compensator contingencies%n", ids.size());
        }
        if (kinds.contains("shunt")) {
            List<String> ids = keepSolved(network.getShuntCompensatorStream().toList(),
                    c -> inSolvedNetwork(c.getTerminal()), "shunt").stream()
                    .map(com.powsybl.iidm.network.ShuntCompensator::getId).toList();
            ids.forEach(id -> all.add(new Contingency(id,
                    new com.powsybl.contingency.ShuntCompensatorContingency(id))));
            System.out.printf("  + %d shunt compensator contingencies%n", ids.size());
        }
        if (kinds.contains("busbar")) {
            // A busbar section outage disconnects everything attached to it — the heaviest topological
            // change of any single-element contingency, and rte6515's biggest untested population.
            int top = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_BUSBAR_TOP", "500"));
            List<String> ids = network.getBusbarSectionStream()
                    .map(com.powsybl.iidm.network.BusbarSection::getId).sorted().limit(top).toList();
            // This list is the ONE that does not filter on inSolvedNetwork by default (branch, generator, SVC
            // and shunt all do), so it carries sections that are disconnected or outside the main
            // connected/synchronous component. OLF drops those itself — PropagatedContingency.toLfContingency
            // returns empty and logs "has no impact" — so NEITHER arm produces a post-contingency result for
            // them, while the GPU census still counts each as MAPPER:no-batchable-element. Measured 209 of 6876
            // at full scope, exactly the gap between 6876 and OLF's own 6667 results.
            //
            // MEASURED on rte6515: all 6876 sections are connected and in the main connected/synchronous
            // component (the count below is 0), so being "outside the solved network" is NOT why 209 of them
            // have no result. OLF drops those 209 for a topological reason instead —
            // PropagatedContingency.toLfContingency finds no branch to open, no bus to lose, no injection to
            // drop, logs "has no impact" and returns empty — so NEITHER arm simulates them and the GPU census
            // still counts each as MAPPER:no-batchable-element. Drop them with OLF_GPU_SA_CTG_EXCLUDE (below).
            long outside = ids.stream()
                    .filter(id -> !inSolvedNetwork(network.getBusbarSection(id).getTerminal()))
                    .count();
            ids.forEach(id -> all.add(new Contingency(id,
                    new com.powsybl.contingency.BusbarSectionContingency(id))));
            System.out.printf("  + %d busbar section contingencies (%d outside the solved network)%n",
                    ids.size(), outside);
        }
        if (kinds.contains("battery")) {
            List<String> ids = keepSolved(network.getBatteryStream().toList(),
                    b -> inSolvedNetwork(b.getTerminal()), "battery").stream()
                    .map(com.powsybl.iidm.network.Battery::getId).toList();
            ids.forEach(id -> all.add(new Contingency(id, new com.powsybl.contingency.BatteryContingency(id))));
            System.out.printf("  + %d battery contingencies%n", ids.size());
        }
        // No "vsc" kind: OLF itself refuses VOLTAGE_SOURCE_CONVERTER (and the DC_* types) in
        // PropagatedContingency — "Unsupported contingency element type" — so there is nothing for the
        // device to be behind on. rte6515 has 10 VSC stations; they are reachable only through their
        // HVDC line, which IS covered.
        // N-2. OLF_GPU_SA_N2_BRANCH / _N2_GEN = how many PAIRS to build. Pairs are drawn from a fixed
        // seed so a differing pair reproduces, and preferentially from elements sharing a voltage level
        // — the double-circuit shape that actually islands things, which is the point of running N-2.
        int n2Branch = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_N2_BRANCH", "0"));
        if (n2Branch > 0) {
            List<String> brIds = network.getBranchStream().map(Branch::getId).sorted().toList();
            java.util.Random rnd = new java.util.Random(20260917L);
            int added = 0;
            for (int t = 0; t < n2Branch * 50 && added < n2Branch; t++) {
                String aId = brIds.get(rnd.nextInt(brIds.size()));
                var a = network.getBranch(aId);
                String vl = a.getTerminal1().getVoltageLevel().getId();
                List<String> nb = brIds.stream().filter(id -> !id.equals(aId))
                        .filter(id -> {
                            var b = network.getBranch(id);
                            return b.getTerminal1().getVoltageLevel().getId().equals(vl)
                                    || b.getTerminal2().getVoltageLevel().getId().equals(vl);
                        }).toList();
                if (nb.isEmpty()) {
                    continue;
                }
                String bId = nb.get(rnd.nextInt(nb.size()));
                all.add(new Contingency(aId + "+" + bId,
                        List.of(new BranchContingency(aId), new BranchContingency(bId))));
                added++;
            }
            System.out.printf("  + %d N-2 branch pairs (same voltage level)%n", added);
        }
        int n2Gen = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_N2_GEN", "0"));
        if (n2Gen > 0) {
            List<String> gs = network.getGeneratorStream().map(com.powsybl.iidm.network.Generator::getId)
                    .sorted().toList();
            java.util.Random rnd = new java.util.Random(20260917L);
            for (int t = 0; t < n2Gen; t++) {
                String a = gs.get(rnd.nextInt(gs.size()));
                String b = gs.get(rnd.nextInt(gs.size()));
                if (a.equals(b)) {
                    continue;
                }
                all.add(new Contingency(a + "+" + b,
                        List.of(new com.powsybl.contingency.GeneratorContingency(a),
                                new com.powsybl.contingency.GeneratorContingency(b))));
            }
            System.out.printf("  + %d N-2 generator pairs%n", n2Gen);
        }
        // OLF_GPU_SA_INVENTORY=1: what the case actually CONTAINS, per contingency element type. The
        // answer to "which kinds are we not testing" is a property of the network, not of memory.
        if ("1".equals(System.getenv("OLF_GPU_SA_INVENTORY"))) {
            // A class histogram over getIdentifiables() rather than per-type accessors: it cannot go
            // stale against the iidm API and it cannot miss a type nobody thought to ask for.
            Map<String, Long> byType = new TreeMap<>();
            network.getIdentifiables().forEach(o -> {
                String t = java.util.Arrays.stream(o.getClass().getInterfaces())
                        .map(Class::getSimpleName).findFirst().orElse(o.getClass().getSimpleName());
                byType.merge(t, 1L, Long::sum);
            });
            byType.forEach((t, n) -> System.out.printf("  INVENTORY %-28s %d%n", t, n));
            System.out.printf("  INVENTORY %-28s %d%n", "(switches)", network.getSwitchStream().count());
        }
        // OLF_GPU_SA_CTG_RANGE=<from>:<to>: this window of the (sorted, kind-ordered) list, [from, to).
        // For kinds that do not change the network structure (branches, injections) a window is a
        // legitimate unit for the BUS diff, which does not fit the heap at full width on a large
        // network; statuses and violation parity still run full width.
        String range = System.getenv("OLF_GPU_SA_CTG_RANGE");
        if (range != null) {
            int colon = range.indexOf(':');
            int from = Integer.parseInt(range.substring(0, colon).trim());
            int to = Math.min(all.size(), Integer.parseInt(range.substring(colon + 1).trim()));
            List<Contingency> window = new java.util.ArrayList<>(all.subList(from, to));
            all.clear();
            all.addAll(window);
            System.out.printf("  window [%d, %d): %d contingencies%n", from, to, all.size());
        }
        // OLF_GPU_SA_CTG_ONLY=<id>[,<id>]: run only these — the instrument for a single differing line.
        String only = System.getenv("OLF_GPU_SA_CTG_ONLY");
        if (only != null) {
            java.util.Set<String> want = java.util.Set.of(only.split(","));
            all.removeIf(c -> !want.contains(c.getId()));
            System.out.printf("  DIAGNOSTIC: %d contingencies selected by id%n", all.size());
            all.forEach(c -> describeContingency(network, c.getId()));
            return all;
        }
        // OLF_GPU_SA_CTG_SHUFFLE=<seed>: shuffle the list before slicing, DETERMINISTICALLY. The list is
        // assembled kind by kind, so every chunk is homogeneous except at the two or three kind
        // boundaries — which makes "all kinds at once" a much weaker test than it looks: a batch of 500
        // branch outages exercises none of the interaction between an injection outage's per-scenario
        // target overrides and a branch outage's disabled-element mask in the SAME chunk. Shuffling puts
        // every kind in every chunk. Seeded so a failure is reproducible.
        String shuffle = System.getenv("OLF_GPU_SA_CTG_SHUFFLE");
        if (shuffle != null) {
            java.util.Collections.shuffle(all, new java.util.Random(Long.parseLong(shuffle)));
            System.out.printf("  SHUFFLED %d contingencies with seed %s%n", all.size(), shuffle);
        }
        // OLF_GPU_SA_MAX_CTG=<n>: run only the first n — a DIAGNOSTIC slice, so a parity question can be
        // iterated in seconds instead of the full 9042. Never quote a slice as the scope result.
        // OLF_GPU_SA_CTG_FROM=<i> slides that slice, so BUS-LEVEL diffing can cover a whole kind in
        // windows. It has to: the bus diff monitors every voltage level of both arms, and rte6515
        // holds ~29832 bus results per contingency — 1389 generator contingencies at once is tens of
        // GB and dies with an OutOfMemoryError before it reports anything. Windows of a few hundred
        // fit, and together they are the full set.
        // OLF_GPU_SA_CTG_EXCLUDE=<file>: drop the contingency ids listed there (one per line, '#' comments).
        // The mode that keeps ONLY THE ONES THAT MATTER: rte6515's busbar list carries 209 sections that OLF
        // itself does not simulate ("has no impact" — no branch opened, no bus lost, no injection dropped), and
        // they show up as MAPPER:no-batchable-element, diluting the census and the batching share while neither
        // arm produces a result for them. docs/rte6515-impactless-busbar-sections.txt is that list, produced
        // from a full-scope census run (grep 'reason=MAPPER:no-batchable-element' with OLF_GPU_REJECT_TRACE=1).
        // NOTE: the contingency list decides which switches the LfNetwork retains, so an excluded run has a
        // DIFFERENT base network — its numbers are comparable with other excluded runs, not with full ones.
        String exclude = System.getenv("OLF_GPU_SA_CTG_EXCLUDE");
        if (exclude != null) {
            try {
                java.util.Set<String> drop = java.nio.file.Files.readAllLines(java.nio.file.Path.of(exclude))
                        .stream().map(String::trim).filter(l -> !l.isEmpty() && !l.startsWith("#"))
                        .collect(Collectors.toSet());
                int before = all.size();
                all.removeIf(c -> drop.contains(c.getId()));
                System.out.printf("  CTG_EXCLUDE %s: %d of %d dropped, %d kept (a different base network than"
                        + " an unfiltered run — the retained switches follow the list)%n",
                        exclude, before - all.size(), before, all.size());
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
        String max = System.getenv("OLF_GPU_SA_MAX_CTG");
        int from = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_CTG_FROM", "0"));
        if (max != null || from > 0) {
            int n = max != null ? Integer.parseInt(max) : all.size();
            int lo = Math.min(Math.max(from, 0), all.size());
            int hi = n > 0 ? Math.min(lo + n, all.size()) : all.size();
            if (lo > 0 || hi < all.size()) {
                System.out.printf("  DIAGNOSTIC SLICE: contingencies [%d, %d) of %d%n", lo, hi, all.size());
                return all.subList(lo, hi);
            }
        }
        return all;
    }

    /** What kind of bus a generator contingency lands on — the facts that decide which reactive-limit
     *  path the two arms take: other generators at the bus, their limit KIND, and whether control is
     *  remote. */
    private static void describeContingency(Network network, String id) {
        var gen = network.getGenerator(id);
        if (gen == null) {
            return;
        }
        var bus = gen.getTerminal().getBusBreakerView().getBus();
        System.out.printf("  CTG %s: bus=%s nominalV=%.1f targetP=%.3f minQ=%.3f maxQ=%.3f limits=%s remote=%b%n",
                id, bus == null ? "?" : bus.getId(), gen.getTerminal().getVoltageLevel().getNominalV(),
                gen.getTargetP(), gen.getReactiveLimits().getMinQ(gen.getTargetP()),
                gen.getReactiveLimits().getMaxQ(gen.getTargetP()), gen.getReactiveLimits().getKind(),
                gen.getRegulatingTerminal() != null
                        && gen.getRegulatingTerminal().getBusBreakerView().getBus() != bus);
        if (bus != null) {
            bus.getVoltageLevel().getGeneratorStream()
                    .filter(g -> g.getTerminal().getBusBreakerView().getBus() == bus)
                    .forEach(g -> System.out.printf("      at the same bus: %s regulating=%b limits=%s targetP=%.3f%n",
                            g.getId(), g.isVoltageRegulatorOn(), g.getReactiveLimits().getKind(), g.getTargetP()));
        }
    }

    private static Map<String, Long> statusCounts(SecurityAnalysisResult r) {
        Map<String, Long> counts = new TreeMap<>();
        for (PostContingencyResult p : r.getPostContingencyResults()) {
            counts.merge(p.getStatus().name(), 1L, Long::sum);
        }
        return counts;
    }

    private static long violationCount(SecurityAnalysisResult r) {
        return r.getPostContingencyResults().stream()
                .mapToLong(p -> p.getLimitViolationsResult().getLimitViolations().size())
                .sum();
    }

    /** Post-contingency bus voltages STREAMED to a temp file the moment an arm returns, so the two arms'
     *  full results never sit in the heap together: 9042 branches (or 6876 busbar sections) x 6877
     *  BusResult objects x 2 arms overflowed the forked JVM's default heap. The arm's result comes back
     *  without its post-contingency bus objects (statuses, violations, branch results and the BASE bus
     *  results stay); the per-contingency comparison reads each arm's entry back on demand. */
    static final class BusVoltageSpill implements AutoCloseable {
        private final java.nio.file.Path path;
        private final java.io.RandomAccessFile raf;
        private final boolean temporary;
        private final Map<String, Long> offsetById = new java.util.HashMap<>();
        private long end;

        BusVoltageSpill(String tag) throws java.io.IOException {
            path = java.nio.file.Files.createTempFile("busv-" + tag + "-", ".bin");
            raf = new java.io.RandomAccessFile(path.toFile(), "rw");
            temporary = true;
        }

        private BusVoltageSpill(java.nio.file.Path persisted) throws java.io.IOException {
            path = persisted;
            raf = new java.io.RandomAccessFile(path.toFile(), "r");
            temporary = false;
        }

        SecurityAnalysisResult spill(SecurityAnalysisResult r) throws java.io.IOException {
            List<PostContingencyResult> slim = new java.util.ArrayList<>(r.getPostContingencyResults().size());
            for (PostContingencyResult p : r.getPostContingencyResults()) {
                java.io.ByteArrayOutputStream bo = new java.io.ByteArrayOutputStream(1 << 17);
                java.io.DataOutputStream d = new java.io.DataOutputStream(bo);
                List<com.powsybl.security.results.BusResult> buses = p.getNetworkResult().getBusResults();
                d.writeInt(buses.size());
                for (com.powsybl.security.results.BusResult b : buses) {
                    d.writeUTF(b.getBusId());
                    d.writeDouble(b.getV());
                }
                d.flush();
                raf.seek(end);
                raf.write(bo.toByteArray());
                offsetById.put(p.getContingency().getId(), end);
                end += bo.size();
                slim.add(withoutBusResults(p));
            }
            return new SecurityAnalysisResult(r.getPreContingencyResult(), slim, r.getOperatorStrategyResults());
        }

        /** The arm's bus voltages for one contingency in the order it reported them; null when absent. */
        java.util.LinkedHashMap<String, Double> load(String contingencyId) throws java.io.IOException {
            Long off = offsetById.get(contingencyId);
            if (off == null) {
                return null;
            }
            raf.seek(off);
            java.io.DataInputStream d = new java.io.DataInputStream(new java.io.BufferedInputStream(
                    new java.io.FileInputStream(raf.getFD()), 1 << 16));
            int n = d.readInt();
            java.util.LinkedHashMap<String, Double> v = new java.util.LinkedHashMap<>(n * 2);
            for (int i = 0; i < n; i++) {
                String id = d.readUTF();
                v.put(id, d.readDouble());
            }
            return v;
        }

        @Override
        public void close() throws java.io.IOException {
            raf.close();
            if (temporary) {
                java.nio.file.Files.deleteIfExists(path);
            }
        }

        /** Persist this spill next to a saved CPU arm: the voltages file and its contingency index. */
        void saveTo(java.nio.file.Path bin, java.nio.file.Path idx) throws java.io.IOException {
            raf.getChannel().force(true);
            java.nio.file.Files.copy(path, bin, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            try (java.io.BufferedWriter w = java.nio.file.Files.newBufferedWriter(idx)) {
                for (Map.Entry<String, Long> e : offsetById.entrySet()) {
                    w.write(e.getKey());
                    w.write('\t');
                    w.write(Long.toString(e.getValue()));
                    w.write('\n');
                }
            }
        }

        /** Re-open a persisted spill (read-only use: {@link #load}). */
        static BusVoltageSpill open(java.nio.file.Path bin, java.nio.file.Path idx) throws java.io.IOException {
            BusVoltageSpill sp = new BusVoltageSpill(bin);
            for (String line : java.nio.file.Files.readAllLines(idx)) {
                int t = line.indexOf('\t');
                if (t > 0) {
                    sp.offsetById.put(line.substring(0, t), Long.parseLong(line.substring(t + 1)));
                }
            }
            sp.end = sp.raf.length();
            return sp;
        }

        static PostContingencyResult withoutBusResults(PostContingencyResult p) {
            return new PostContingencyResult(p.getContingency(), p.getStatus(), p.getLimitViolationsResult(),
                    new com.powsybl.security.results.NetworkResult(p.getNetworkResult().getBranchResults(), List.of(),
                            p.getNetworkResult().getThreeWindingsTransformerResults()),
                    p.getConnectivityResult(), p.getDistributedActivePower());
        }

        /** Drop an arm's post-contingency bus objects without keeping them anywhere (the warm arm). */
        static SecurityAnalysisResult dropPostBusResults(SecurityAnalysisResult r) {
            return new SecurityAnalysisResult(r.getPreContingencyResult(),
                    r.getPostContingencyResults().stream().map(BusVoltageSpill::withoutBusResults).toList(),
                    r.getOperatorStrategyResults());
        }
    }

    /**
     * {@code getDistributedActivePower()} of the base and every post-contingency result, CPU against GPU —
     * the contingency's pre-distributed loss PLUS its load flow's own slack distribution. Neither the status
     * nor the violation checks look at it; the GPU path used to report the pre-distribution alone (up to
     * 124.5 MW off, assessment L10). {@code OLF_GPU_SA_DAP_DUMP=<file>} writes both values per contingency with
     * {@code Double.toString}, so a change on either arm diffs exactly. Returns the worst difference (MW).
     */
    private static double reportDistributedActivePower(SecurityAnalysisResult cpu, SecurityAnalysisResult gpu) throws java.io.IOException {
        double baseDiff = Math.abs(cpu.getPreContingencyResult().getDistributedActivePower()
                - gpu.getPreContingencyResult().getDistributedActivePower());
        System.out.printf("  distributedActivePower base: cpu=%.9f gpu=%.9f MW%n",
                cpu.getPreContingencyResult().getDistributedActivePower(), gpu.getPreContingencyResult().getDistributedActivePower());
        Map<String, Double> gpuDap = new TreeMap<>();
        gpu.getPostContingencyResults().forEach(p -> gpuDap.put(p.getContingency().getId(), p.getDistributedActivePower()));
        java.util.Set<String> convergedBoth = new java.util.HashSet<>();
        gpu.getPostContingencyResults().stream()
                .filter(p -> p.getStatus() == PostContingencyComputationStatus.CONVERGED)
                .forEach(p -> convergedBoth.add(p.getContingency().getId()));
        Map<String, double[]> pairs = new TreeMap<>();
        // A contingency that CONVERGED on neither arm has no comparable distributed power: what a diverging run
        // "distributed before giving up" is path-dependent — measured on rte6515 BUS-2289_BBS (MAX_ITERATION_REACHED
        // on both arms, replayed on the CPU inside the GPU arm) as 0.59 MW in one run and 0.20 MW in the next, from
        // the same binaries. Those are printed apart and never asserted.
        Map<String, double[]> unconverged = new TreeMap<>();
        cpu.getPostContingencyResults().forEach(p -> {
            Double g = gpuDap.get(p.getContingency().getId());
            if (g != null) {
                double[] pair = {p.getDistributedActivePower(), g};
                if (p.getStatus() == PostContingencyComputationStatus.CONVERGED
                        && convergedBoth.contains(p.getContingency().getId())) {
                    pairs.put(p.getContingency().getId(), pair);
                } else {
                    unconverged.put(p.getContingency().getId(), pair);
                }
            }
        });
        long differing = pairs.values().stream().filter(v -> Math.abs(v[0] - v[1]) > 1e-9).count();
        double worst = pairs.values().stream().mapToDouble(v -> Math.abs(v[0] - v[1])).max().orElse(0);
        System.out.printf("  distributedActivePower: %d of %d contingencies differ by > 1e-9 MW | worst %.6e MW%n",
                differing, pairs.size(), worst);
        pairs.entrySet().stream()
                .sorted(java.util.Comparator.comparingDouble((Map.Entry<String, double[]> e) ->
                        -Math.abs(e.getValue()[0] - e.getValue()[1])))
                .limit(5)
                .forEach(e -> System.out.printf("      %s cpu=%.9f gpu=%.9f MW%n", e.getKey(), e.getValue()[0], e.getValue()[1]));
        if (!unconverged.isEmpty()) {
            double worstUnconverged = unconverged.values().stream().mapToDouble(v -> Math.abs(v[0] - v[1])).max().orElse(0);
            System.out.printf("  distributedActivePower (not CONVERGED on both arms, NOT asserted): %d contingencies"
                    + " | worst %.6e MW%n", unconverged.size(), worstUnconverged);
        }
        String dump = System.getenv("OLF_GPU_SA_DAP_DUMP");
        if (dump != null) {
            try (java.io.PrintWriter w = new java.io.PrintWriter(Files.newBufferedWriter(Path.of(dump)))) {
                pairs.forEach((id, v) -> w.println(id + "\t" + Double.toString(v[0]) + "\t" + Double.toString(v[1])));
                unconverged.forEach((id, v) -> w.println(id + "\t" + Double.toString(v[0]) + "\t" + Double.toString(v[1])
                        + "\tnot-converged"));
            }
        }
        return Math.max(worst, baseDiff);
    }

    @Test
    void gpuSaMatchesCpuOnRte6515Full() throws Exception {
        Network network = Network.read(CASE_FILE);
        List<Contingency> contingencies = allBranchContingencies(network);
        System.out.printf("%s: %d branches → %d N-1 contingencies%n", CASE_FILE,
                network.getBranchCount(), contingencies.size());

        LoadFlowParameters lfp = JsonLoadFlowParameters.read(PARAMS_FILE);
        // OLF_GPU_DBG_CONV_EPS: tighten the inner-Newton criterion on BOTH arms. The only honest way to
        // tell a real state difference from convergence slack is to squeeze the slack and watch what
        // survives — see the standing rule against calling a difference a tolerance artefact unproven.
        // OLF_GPU_DBG_RL: TRACE the CPU arm's ReactiveLimitsOuterLoop, so every PV<->PQ switch it
        // makes (bus id, q, qLimit) can be diffed against the device's GPU_RL / GPU_RL_FINAL pin
        // list. A pin the device makes and OLF does not is invisible in any residual - the device
        // converges cleanly against the WRONG equation - so the two switch lists are the only
        // direct evidence. Ported from GpuVsCpuProductionSaBenchmarkTest; the harness has no
        // appender configured (core/classic version mismatch), so attach one.
        // OLF_GPU_DBG_NR: the CPU arm's own Newton trajectory — |f(x0)|, one line per iteration, and
        // every MAX_VOLTAGE_CHANGE step-size clamp it applies. Put beside the device's OLF_GPU_CONV_TRACE
        // this is the only direct way to see WHERE the two trajectories part at the DEFAULT criterion,
        // which is the question: the two arms must take the same iterates, not merely land in the same
        // tolerance ball. Tightening the tolerance hides this instead of answering it.
        if (System.getenv("OLF_GPU_DBG_NR") != null) {
            ch.qos.logback.classic.encoder.PatternLayoutEncoder nrEnc =
                    new ch.qos.logback.classic.encoder.PatternLayoutEncoder();
            ch.qos.logback.classic.Logger nrRoot = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
            nrEnc.setContext(nrRoot.getLoggerContext());
            nrEnc.setPattern("CPU_NR %logger{0} %msg%n");
            nrEnc.start();
            ch.qos.logback.core.ConsoleAppender<ch.qos.logback.classic.spi.ILoggingEvent> nrApp =
                    new ch.qos.logback.core.ConsoleAppender<>();
            nrApp.setContext(nrRoot.getLoggerContext());
            nrApp.setEncoder(nrEnc);
            nrApp.start();
            // AcloadFlowEngine logs "Start outer loop '<name>' iteration <n>", the CPU counterpart of the
            // device's OLF_GPU_BATCH_TRACE pass line: the two SEQUENCES are what say whether the device ran
            // the same loop, in the same order, the same number of times.
            for (String n : new String[] {"com.powsybl.openloadflow.ac.solver.NewtonRaphson",
                                          "com.powsybl.openloadflow.ac.solver.MaxVoltageChangeStateVectorScaling",
                                          "com.powsybl.openloadflow.ac.AcloadFlowEngine",
                                          // "Contingency '<id>' impact on network ..." (DEBUG): heads each
                                          // post-contingency block, so a block can be addressed by id inside
                                          // the full window (with OLF_GPU_SA_CPU_THREADS=1 they do not interleave)
                                          "com.powsybl.openloadflow.sa.AbstractSecurityAnalysis"}) {
                ch.qos.logback.classic.Logger lg = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger(n);
                lg.setLevel(ch.qos.logback.classic.Level.DEBUG);
                lg.setAdditive(false);
                lg.addAppender(nrApp);
            }
        }
        if (System.getenv("OLF_GPU_DBG_RL") != null) {
            ch.qos.logback.classic.Logger rlLog = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger("com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop");
            rlLog.setLevel(ch.qos.logback.classic.Level.TRACE);
            // The voltage-MONITORING loop too (standby SVC automata): its PQ->PV activations are the
            // other per-contingency DECISION the two arms can disagree on, and like a pin they leave
            // no residual - only the two decision logs tell them apart.
            ch.qos.logback.classic.Logger monLog = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger("com.powsybl.openloadflow.ac.outerloop.MonitoringVoltageOuterLoop");
            monLog.setLevel(ch.qos.logback.classic.Level.TRACE);
            // ...and the HVDC AC-emulation freeze: its per-contingency setpoint (Reports.reportFreezeHvdc,
            // INFO) is the third decision that leaves no residual - the frozen P IS the equation.
            ch.qos.logback.classic.Logger hvLog = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger("com.powsybl.openloadflow.ac.outerloop.FreezingHvdcACEmulationOuterloop");
            hvLog.setLevel(ch.qos.logback.classic.Level.TRACE);
            ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
            if (!root.iteratorForAppenders().hasNext()) {
                ch.qos.logback.classic.encoder.PatternLayoutEncoder enc =
                        new ch.qos.logback.classic.encoder.PatternLayoutEncoder();
                enc.setContext(root.getLoggerContext());
                enc.setPattern("CPU_RL %msg%n");
                enc.start();
                ch.qos.logback.core.ConsoleAppender<ch.qos.logback.classic.spi.ILoggingEvent> ca =
                        new ch.qos.logback.core.ConsoleAppender<>();
                ca.setContext(root.getLoggerContext());
                ca.setEncoder(enc);
                ca.start();
                rlLog.addAppender(ca);
                monLog.addAppender(ca);
                hvLog.addAppender(ca);
            }
        }
        // OLF_GPU_SA_NO_RL / OLF_GPU_SA_NO_DS: disable an outer loop on BOTH arms. Stage isolation -
        // inner Newton only, vs +DS, vs +DS+RL - to find which stage first introduces a divergence.
        // Decisive for a suspected PIN mismatch: if the arms agree with reactive limits OFF, the
        // underlying power flow (and every bus q) agrees and the fault is in the pin DECISION; if
        // they still disagree, q itself is wrong and the pin is only reporting it.
        if ("1".equals(System.getenv("OLF_GPU_SA_NO_RL"))) {
            lfp.setUseReactiveLimits(false);
            System.out.println("  reactive limits DISABLED on both arms (OLF_GPU_SA_NO_RL=1)");
        }
        if ("1".equals(System.getenv("OLF_GPU_SA_NO_DS"))) {
            lfp.setDistributedSlack(false);
            System.out.println("  distributed slack DISABLED on both arms (OLF_GPU_SA_NO_DS=1)");
        }
        String convEpsEnv = System.getenv("OLF_GPU_DBG_CONV_EPS");
        if (convEpsEnv != null) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp)
                    .setNewtonRaphsonConvEpsPerEq(Double.parseDouble(convEpsEnv));
        }
        // OLF_GPU_SA_SEQ_SOLVER=1: run the GPU arm's contingencies through the ORDINARY load flow on
        // the GPU (the sequential GpuNewtonRaphson) instead of OLF's CPU solver. Paired with
        // -Dolf.gpu.sa.disableBatch=true this is the manual-single-contingency debug path: it takes one
        // contingency out of the batch and solves it like any other load flow, which is where the
        // LF-level instruments live (OLF_GPU_RESIDUAL_TRACE above all). If a contingency is wrong
        // batched and right here, the defect is in the BATCH; if it is wrong both ways it reproduces
        // at load-flow level, which is a far cheaper thing to debug.
        // OLF_GPU_SVC=1: enable SecondaryVoltageControl — the RTE params default to off, but
        // the rte6515 case carries SVC zones. This is the path that differed between GPU and
        // CPU when SVC support was incomplete (the base LF didn't go through the chunk solver).
        if ("1".equals(System.getenv("OLF_GPU_SVC"))) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp).setSecondaryVoltageControl(true);
            System.out.println("  SVC enabled (OLF_GPU_SVC=1)");
            // OLF_GPU_SVC_PILOT_SHIFT=<kV>: move EVERY zone's pilot-point target so the loop actually
            // FIRES (with the case's own targets it is stable at the base and on every contingency).
            // rte6515 carries two zones — shifting only one leaves the other's controllers idle and the
            // run then exercises half the zone systems.
            String shift = System.getenv("OLF_GPU_SVC_PILOT_SHIFT");
            var svc = network.getExtension(com.powsybl.iidm.network.extensions.SecondaryVoltageControl.class);
            if (shift != null && svc != null) {
                for (var zone : svc.getControlZones()) {
                    var pilot = zone.getPilotPoint();
                    double before = pilot.getTargetV();
                    pilot.setTargetV(before + Double.parseDouble(shift));
                    System.out.printf("  SVC pilot '%s' target %.3f -> %.3f kV (%d control units)%n",
                            zone.getName(), before, pilot.getTargetV(), zone.getControlUnits().size());
                }
            }
        }
        // Non-zero increased-violations thresholds to avoid ulp-level flicker
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        sap.setIncreasedViolationsParameters(new SecurityAnalysisParameters.IncreasedViolationsParameters(
                0.5, 0.001, 0.5, 0.001, 0.001));

        // CPU arm: 8 threads (production-realistic) - EXCEPT for busbar sections, which compare
        // against ONE thread: OLF's partitioning splits the retained switches between threads, so a
        // multi-threaded CPU arm lands on a different bus decomposition than the device's and the
        // numbers are not comparable (the base alone differs by 8e-6 kV at 8 threads, bit-identical
        // at 1). OLF_GPU_SA_CPU_THREADS still overrides.
        String kindsEnv = System.getenv().getOrDefault("OLF_GPU_SA_CTG_KINDS", "");
        String defaultThreads = kindsEnv.contains("busbar") ? "1" : "8";
        int cpuThreads = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_CPU_THREADS", defaultThreads));
        sap.addExtension(OpenSecurityAnalysisParameters.class,
                new OpenSecurityAnalysisParameters().setThreadCount(cpuThreads));

        // ONE NETWORK PER ARM. The three runs used to share a single Network object, which is fine for
        // branch/generator/load outages but NOT for busbar-section ones: those open switches, and any
        // switch state the restore does not put back is inherited by whichever arm runs next. The GPU
        // arm runs second, so it started from a progressively more perturbed topology the more busbar
        // contingencies had already executed — which is exactly the shape of the "busbar 8.176e-06 kV
        // on a 0%-batched kind" that stood recorded as a GPU/CPU difference. Read before timing, so a
        // second parse never lands inside a measured region.
        Network networkGpuCold = Network.read(CASE_FILE);
        Network networkGpuWarm = Network.read(CASE_FILE);
        // OLF_GPU_SA_CPU_WARM=1: time the CPU arm TWICE, like the GPU arm. The first CPU run is the
        // JVM's first SA and pays JIT compilation the GPU arms never see, so the cold-vs-cold ratio
        // flatters the GPU. The ratios are then quoted against CPU warm.
        boolean cpuWarmRun = "1".equals(System.getenv("OLF_GPU_SA_CPU_WARM"));
        Network networkCpuWarm = cpuWarmRun ? Network.read(CASE_FILE) : null;

        armMark("cpu start");
        long cpuT0 = System.nanoTime();
        // With a single contingency under the lens, monitor every voltage level: whether the two arms
        // differ LOCALLY or network-wide is the first thing that splits the candidate causes.
        // Bus-level diffing is the check that actually holds: violation parity is a THRESHOLD test and
        // will report zero while a bus is 0.1 kV out (measured — rte6515 LOAD-1263). CTG_ONLY turns it
        // on for one contingency; OLF_GPU_SA_BUSDIFF=1 turns it on for the whole set, which is what a
        // new contingency KIND needs before anyone calls it clean.
        // OLF_GPU_SA_BUSDIFF_CTG=<id>[,<id>]: bus-diff those contingencies INSIDE the full scope. On a
        // node-breaker network the contingency LIST decides which switches are retained, so running a
        // differing contingency alone (CTG_ONLY) solves a different network - network.xiidm/LNER goes
        // from 13202 buses to 8859 and the difference vanishes. Monitoring the whole set instead is
        // what does not fit: bus results are produced BY THE MONITORS, so an all-contingency monitor
        // puts 10540 x 13202 BusResults on the heap before anything can spill them and the arm dies at
        // its 14g ceiling. Scoping the monitor to a few ids keeps the full list, the full network and
        // the full chunking, and costs nothing.
        java.util.Set<String> busDiffCtgs = System.getenv("OLF_GPU_SA_BUSDIFF_CTG") == null
                ? java.util.Set.of()
                : java.util.Arrays.stream(System.getenv("OLF_GPU_SA_BUSDIFF_CTG").split(","))
                        .map(String::trim).filter(x -> !x.isEmpty()).collect(Collectors.toSet());
        boolean busDiff = System.getenv("OLF_GPU_SA_CTG_ONLY") != null
                || !busDiffCtgs.isEmpty()
                || "1".equals(System.getenv("OLF_GPU_SA_BUSDIFF"));
        // OLF_GPU_SA_MON_BRANCH=<id>[,<id>]: monitor these branches and print BOTH arms' flows. A
        // one-sided violation says only that the other arm is on the other side of the limit — it does
        // NOT say by how much, and "the margin is small" is a statement about the CPU's distance to its
        // limit, not about the distance between the two arms.
        java.util.Set<String> monBranches = System.getenv("OLF_GPU_SA_MON_BRANCH") == null
                ? java.util.Set.of()
                : java.util.Arrays.stream(System.getenv("OLF_GPU_SA_MON_BRANCH").split(","))
                        .map(String::trim).collect(Collectors.toSet());
        java.util.Set<String> monVls = busDiff
                ? network.getVoltageLevelStream().map(com.powsybl.iidm.network.VoltageLevel::getId)
                        .collect(Collectors.toSet())
                : java.util.Set.of();
        // TWO monitors, not one. ContingencyContext.all() left the PRE-contingency network result
        // EMPTY, so "BASE BUSDIFF buses>1e-12kV=0 of 0 | uncompared: cpu-only=0 gpu-only=0" was not
        // a clean base - it was no base compared at all, in every run this test has ever done. A
        // post-contingency difference cannot be attributed to the contingency until the BASE states
        // are known to agree, so add an explicit N-state monitor.
        List<com.powsybl.security.monitor.StateMonitor> monitors;
        if (!busDiff && monBranches.isEmpty()) {
            monitors = Collections.emptyList();
        } else {
            List<com.powsybl.security.monitor.StateMonitor> ms = new ArrayList<>();
            if (busDiffCtgs.isEmpty()) {
                ms.add(new com.powsybl.security.monitor.StateMonitor(
                        com.powsybl.contingency.ContingencyContext.all(), monBranches,
                        monVls, java.util.Set.of()));
            } else {
                for (String id : busDiffCtgs) {           // one scoped monitor per id, so the heap holds
                    ms.add(new com.powsybl.security.monitor.StateMonitor(   // a handful of results, not 10540
                            com.powsybl.contingency.ContingencyContext.specificContingency(id),
                            monBranches, monVls, java.util.Set.of()));
                }
            }
            ms.add(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.none(), monBranches,
                    monVls, java.util.Set.of()));
            monitors = List.copyOf(ms);
        }
        System.out.printf("  JVM max heap: %d MB%n", Runtime.getRuntime().maxMemory() >> 20);
        // Each arm's post-contingency bus voltages are streamed out of the heap before the next arm
        // runs (BusVoltageSpill); the raw result lives in a block so nothing keeps it reachable.
        // OLF_GPU_SA_CPU_ARM_DIR=<dir> + OLF_GPU_SA_CPU_ARM_MODE=save|load: the two arms in SEPARATE
        // JVMs. On the full busbar scope (6876 sections x 6877 buses) each arm's result alone fills a
        // 14 GB heap; sharing a JVM ended in the OS killing the fork even with the spill. "save" runs
        // the CPU arm only and writes its slim result (statuses, violations, connectivity, the base
        // bus results - powsybl's JSON) plus the spilled bus voltages and their index; "load" rebuilds
        // the CPU arm from those files and runs only the GPU arms. Same contingency list either way
        // (sorted ids), so the two runs see the same network.
        String cpuArmDir = System.getenv("OLF_GPU_SA_CPU_ARM_DIR");
        String cpuArmMode = System.getenv().getOrDefault("OLF_GPU_SA_CPU_ARM_MODE", "");
        BusVoltageSpill cpuSpill = null;
        BusVoltageSpill gpuSpill = busDiff ? new BusVoltageSpill("gpu") : null;
        SecurityAnalysisResult cpu;
        long cpuMs;
        if ("load".equals(cpuArmMode)) {
            java.nio.file.Path dir = java.nio.file.Path.of(cpuArmDir);
            cpu = com.powsybl.security.json.SecurityAnalysisResultDeserializer.read(dir.resolve("cpu.json"));
            if (busDiff) {
                cpuSpill = BusVoltageSpill.open(dir.resolve("cpu-busv.bin"), dir.resolve("cpu-busv.idx"));
            }
            cpuMs = Long.parseLong(java.nio.file.Files.readString(dir.resolve("cpu-ms.txt")).trim());
            System.out.printf("  CPU arm loaded from %s (%d post-contingency results)%n", dir, cpu.getPostContingencyResults().size());
        } else {
            cpuSpill = busDiff ? new BusVoltageSpill("cpu") : null;
            // OLF_GPU_SA_GPU_ONLY=1: debug lever - skip the CPU arm (run it on an EMPTY contingency
            // list) so the GPU arm's own behaviour on a large scope can be traced in minutes.
            // Every comparison below is then meaningless; only the GPU arm's lines are.
            List<Contingency> cpuCtgs = "1".equals(System.getenv("OLF_GPU_SA_GPU_ONLY")) ? List.of() : contingencies;
            SecurityAnalysisResult raw = runSecurityAnalysis(network, cpuCtgs, monitors, sap);
            cpuMs = (System.nanoTime() - cpuT0) / 1_000_000;
            cpu = cpuSpill != null ? cpuSpill.spill(raw) : raw;
            raw = null;                                          // the block alone did not free it: 6876 x 6877
            System.gc();                                         // BusResult objects stayed reachable from the frame
            if ("save".equals(cpuArmMode)) {
                java.nio.file.Path dir = java.nio.file.Files.createDirectories(java.nio.file.Path.of(cpuArmDir));
                try (java.io.Writer w = java.nio.file.Files.newBufferedWriter(dir.resolve("cpu.json"))) {
                    com.powsybl.security.json.SecurityAnalysisResultSerializer.write(cpu, w);
                }
                if (cpuSpill != null) {
                    cpuSpill.saveTo(dir.resolve("cpu-busv.bin"), dir.resolve("cpu-busv.idx"));
                    cpuSpill.close();
                }
                java.nio.file.Files.writeString(dir.resolve("cpu-ms.txt"), Long.toString(cpuMs));
                System.out.printf("  CPU_ARM_SAVED %s | %d ms | statuses %s%n", dir, cpuMs, statusCounts(cpu));
                return;
            }
        }
        // OLF_GPU_SA_CPU_WARM=1: time the CPU arm TWICE, like the GPU arm. The first CPU run is the JVM's first SA
        // and pays JIT compilation the GPU arms never see, so the cold-vs-cold ratio flatters the GPU. The ratios
        // are quoted against CPU warm. Not available with a loaded (separate-JVM) CPU arm.
        armMark("cpu end");
        long cpuWarmMs = -1;
        if (cpuWarmRun && !"load".equals(cpuArmMode) && networkCpuWarm != null) {
            long cpuT1 = System.nanoTime();
            SecurityAnalysisResult cpuWarm = runSecurityAnalysis(networkCpuWarm, contingencies, monitors, sap);
            cpuWarmMs = (System.nanoTime() - cpuT1) / 1_000_000;
            System.out.printf("  CPU warm run: violations=%d statuses=%s%n", violationCount(cpuWarm), statusCounts(cpuWarm));
        }
        long cpuRefMs = cpuWarmMs > 0 ? cpuWarmMs : cpuMs;

        // GPU arm: single-thread, cold + warm runs
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(new SparseMatrixFactory(),
                new EvenShiloachGraphDecrementalConnectivityFactory<>());
        SecurityAnalysisResult gpuCold;
        SecurityAnalysisResult gpuWarm;
        long gpuColdMs;
        long gpuWarmMs;
        try {
            SecurityAnalysisParameters sapGpu = new SecurityAnalysisParameters();
            // OLF_GPU_SA_SEQ_SOLVER=1: run the GPU arm's contingencies through the ORDINARY load flow
            // on the GPU (the sequential GpuNewtonRaphson) instead of OLF's CPU solver. Paired with
            // -Dolf.gpu.sa.disableBatch=true this is the manual-single-contingency debug path: it takes a
            // contingency out of the batch and solves it like any other load flow, which is where the
            // LF-level instruments live (OLF_GPU_RESIDUAL_TRACE above all). A contingency wrong batched
            // and right here is a BATCH defect; wrong both ways reproduces at load-flow level.
            //
            // It MUST go on a COPY. The two arms share one LoadFlowParameters, so setting the solver on
            // `lfp` puts the GPU solver on the CPU arm too and the run compares the device against
            // itself — which agrees at 1e-11 and proves nothing, the same shape of trap as a contingency
            // that "agrees" because both arms fell back to the CPU.
            LoadFlowParameters lfpGpu = lfp;
            if ("1".equals(System.getenv("OLF_GPU_SA_SEQ_SOLVER"))) {
                lfpGpu = lfp.copy();
                com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfpGpu).setAcSolverType(
                        com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphsonFactory.NAME);
            }
            sapGpu.setLoadFlowParameters(lfpGpu);
            sapGpu.setIncreasedViolationsParameters(sap.getIncreasedViolationsParameters());
            // GPU runs host-single-threaded (threadCount > 1 → SIGSEGV in cuLaunchKernel)

            GpuBatchScopeCensus.reset();
            // OLF_GPU_SA_JFR=<file.jfr>: a Flight Recorder profile of the COLD GPU run only — network
            // reading, the CPU arm and the warm run stay out of the samples.
            String jfrOut = System.getenv("OLF_GPU_SA_JFR");
            jdk.jfr.Recording jfr = null;
            if (jfrOut != null) {
                jfr = new jdk.jfr.Recording(jdk.jfr.Configuration.getConfiguration("profile"));
                // OLF_GPU_SA_JFR_PERIOD_MS=<n>: sample Java AND native stacks every n ms ("profile" takes Java
                // stacks every 10-20 ms and native ones apart, too coarse to split a ~1 s difference by method)
                String period = System.getenv("OLF_GPU_SA_JFR_PERIOD_MS");
                if (period != null) {
                    java.time.Duration every = java.time.Duration.ofMillis(Long.parseLong(period));
                    jfr.enable("jdk.ExecutionSample").withPeriod(every);
                    jfr.enable("jdk.NativeMethodSample").withPeriod(every);
                }
                jfr.start();
            }
            armMark("gpu cold start");
            long g0 = System.nanoTime();
            {
                SecurityAnalysisResult raw = runSecurityAnalysis(networkGpuCold, contingencies, monitors, sapGpu);
                gpuColdMs = (System.nanoTime() - g0) / 1_000_000;
                gpuCold = gpuSpill != null ? gpuSpill.spill(raw) : raw;
                raw = null;
                System.gc();
            }
            armMark("gpu cold end");
            if (jfr != null) {
                jfr.stop();
                jfr.dump(Path.of(jfrOut));
                jfr.close();
                System.out.printf("  JFR profile of the cold GPU run written to %s%n", jfrOut);
            }

            // Check census: how many contingencies fell back to CPU?
            Map<String, Long> census = GpuBatchScopeCensus.snapshot();
            System.out.printf("  GPU CENSUS (cold): %s%n", census);
            int gpuPrecomputed = GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_COUNT.get();
            // Two denominators, because they differ: the LIST, and the contingencies OLF actually simulates.
            // A contingency OLF drops as having no impact has no result on either arm, so counting it against
            // the batching share reports a GPU limitation that does not exist (209 of rte6515's 6876 busbar
            // sections). The CPU arm's result count IS the simulated population.
            int simulated = cpu.getPostContingencyResults().size();
            if (simulated > 0) {                             // 0 under OLF_GPU_SA_GPU_ONLY: no CPU arm to count
                System.out.printf("  GPU precomputed on device: %d / %d contingencies (%d of %d simulated;"
                        + " %d simulated by neither arm)%n",
                        gpuPrecomputed, contingencies.size(), gpuPrecomputed, simulated,
                        contingencies.size() - simulated);
            } else {
                System.out.printf("  GPU precomputed on device: %d / %d contingencies%n",
                        gpuPrecomputed, contingencies.size());
            }

            // OLF_GPU_SA_COLD_ONLY=1: no warm run (a profile of the cold run alone; "warm" then repeats cold).
            if ("1".equals(System.getenv("OLF_GPU_SA_COLD_ONLY"))) {
                gpuWarm = gpuCold;
                gpuWarmMs = gpuColdMs;
            } else {
                GpuBatchScopeCensus.reset();
                long g1 = System.nanoTime();
                {
                    SecurityAnalysisResult raw = runSecurityAnalysis(networkGpuWarm, contingencies, monitors, sapGpu);
                    gpuWarmMs = (System.nanoTime() - g1) / 1_000_000;
                    gpuWarm = busDiff ? BusVoltageSpill.dropPostBusResults(raw) : raw;   // only its BASE is compared
                    raw = null;
                    System.gc();
                }
                Map<String, Long> censusWarm = GpuBatchScopeCensus.snapshot();
                System.out.printf("  GPU CENSUS (warm): %s%n", censusWarm);
            }
        } finally {
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(),
                    new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }

        // Report timing
        int n = contingencies.size();
        System.out.printf("%nRTE6515 SA BENCHMARK | %d contingencies | RTE production params%n", n);
        System.out.printf("  CPU  (%2d threads) : %6d ms | %.2f ms/ctg%s%n", cpuThreads, cpuMs, cpuMs / (double) n,
                cpuWarmMs > 0 ? " (COLD)" : "");
        if (cpuWarmMs > 0) {
            System.out.printf("  CPU  (%2d threads) : %6d ms | %.2f ms/ctg (warm) — the reference below%n",
                    cpuThreads, cpuWarmMs, cpuWarmMs / (double) n);
        }
        System.out.printf("  GPU  (COLD)       : %6d ms | %.2f ms/ctg | vs CPU x%.2f%n",
                gpuColdMs, gpuColdMs / (double) n, (double) cpuRefMs / gpuColdMs);
        System.out.printf("  GPU  (warm)       : %6d ms | %.2f ms/ctg | vs CPU x%.2f%n",
                gpuWarmMs, gpuWarmMs / (double) n, (double) cpuRefMs / gpuWarmMs);
        System.out.printf("  pre-ctg: CPU=%s GPU=%s%n",
                cpu.getPreContingencyResult().getStatus(), gpuCold.getPreContingencyResult().getStatus());
        // The BASE load flow, bus by bus. Every post-contingency state is computed FROM the base, so a
        // base that differs puts a floor under every contingency — and comparing only post-contingency
        // states cannot see it, because both arms' bases are their own. Relevant because the device
        // anchors its reactive-capability CURVE pack BEFORE the base load flow's own slack
        // distribution, which on rte6515 moves -28.26 pu, the same size as the contingency
        // redistributions that break the pack's proportional-share model.
        {
            Map<String, Double> gpuBaseV = new TreeMap<>();
            gpuCold.getPreContingencyResult().getNetworkResult().getBusResults()
                    .forEach(b -> gpuBaseV.put(b.getBusId(), b.getV()));
            var cpuBase = cpu.getPreContingencyResult().getNetworkResult().getBusResults().stream()
                    .filter(b -> gpuBaseV.containsKey(b.getBusId())).toList();
            long above = cpuBase.stream().filter(b -> Math.abs(b.getV() - gpuBaseV.get(b.getBusId())) > 1e-12).count();
            double worst = cpuBase.stream()
                    .mapToDouble(b -> Math.abs(b.getV() - gpuBaseV.get(b.getBusId()))).max().orElse(0);
            // The comparison can only cover buses BOTH arms reported. Say so out loud: a run whose two
            // arms name their buses differently (busbar-section outages re-split the bus view) used to
            // print "0 of 1" and read like a clean result, when it meant 29831 buses went uncompared.
            // OLF_GPU_SA_MON_BUS also applies to the BASE. A post-contingency value that equals the
            // base one to the last bit is a variable that never MOVED - which is what an
            // unconstrained column looks like, and is indistinguishable from a solved value unless
            // the base is printed next to it.
            String monBase = System.getenv("OLF_GPU_SA_MON_BUS");
            if (monBase != null) {
                java.util.Set<String> wantBase = java.util.Set.of(monBase.split(","));
                cpu.getPreContingencyResult().getNetworkResult().getBusResults().stream()
                        .filter(b -> wantBase.contains(b.getBusId()) && gpuBaseV.containsKey(b.getBusId()))
                        .forEach(b -> System.out.printf("  MONBUS-BASE bus=%s cpu=%.9f gpu=%.9f d=%.3e%n",
                                b.getBusId(), b.getV(), gpuBaseV.get(b.getBusId()),
                                b.getV() - gpuBaseV.get(b.getBusId())));
            }
            int cpuAll = cpu.getPreContingencyResult().getNetworkResult().getBusResults().size();
            System.out.printf("  BASE BUSDIFF buses>1e-12kV=%d of %d | worst abs=%.3e kV"
                            + " | uncompared: cpu-only=%d gpu-only=%d%n",
                    above, cpuBase.size(), worst, cpuAll - cpuBase.size(), gpuBaseV.size() - cpuBase.size());
            if (cpuAll != cpuBase.size() || gpuBaseV.size() != cpuBase.size()) {
                // NOT APPLES TO APPLES. OLF's multi-threaded SA builds a partitionTopoConfig PER
                // PARTITION and takes the pre-contingency result from one of them ("all pre contingency
                // are the same" — which does not hold when the partitions retain different switches).
                // The GPU arm is forced single-threaded (threadCount > 1 SIGSEGVs in cuLaunchKernel), so
                // with busbar-section contingencies the two arms can end up on DIFFERENT bus
                // decompositions entirely: measured at 60 busbar outages, CPU 29832 buses vs GPU 6877
                // with ONE in common, which the intersection-only print rendered as a clean 1.4e-13.
                // At matched thread counts the same cases agree BIT-IDENTICALLY (worst 0.0e+00).
                System.out.printf("  BASE BUSDIFF: *** NOT COMPARABLE *** the arms disagree on the bus "
                        + "decomposition (cpu=%d gpu=%d buses). Re-run with OLF_GPU_SA_CPU_THREADS=1 "
                        + "before reading any number above.%n", cpuAll, gpuBaseV.size());
            }
            cpuBase.stream()
                    .sorted(java.util.Comparator.comparingDouble(
                            (com.powsybl.security.results.BusResult b) -> -Math.abs(b.getV() - gpuBaseV.get(b.getBusId()))))
                    .limit(4)
                    .forEach(b -> System.out.printf("      %s %.9f vs %.9f (%.3e)%n", b.getBusId(), b.getV(),
                            gpuBaseV.get(b.getBusId()), b.getV() - gpuBaseV.get(b.getBusId())));

            // The SAME arm against ITSELF. The cold and warm GPU runs solve the identical problem on
            // identical fresh networks, so any difference between them is the device's own
            // reproducibility floor — the fill kernels scatter with atomicAdd, whose summation order
            // is not fixed, and on rte6515 with switches retained the Jacobian carries diagonals of
            // 3.3e+04 (a retained switch is x ~ 3e-05 pu), where one ulp is 4e-12 absolute. Print it
            // next to the CPU-vs-GPU number: a CPU-vs-GPU gap NO LARGER than the GPU-vs-GPU gap is
            // arithmetic noise and there is no defect underneath it to find. A gap that is larger is
            // a real disagreement and must be chased.
            Map<String, Double> warmBaseV = new TreeMap<>();
            gpuWarm.getPreContingencyResult().getNetworkResult().getBusResults()
                    .forEach(b -> warmBaseV.put(b.getBusId(), b.getV()));
            var selfPairs = gpuCold.getPreContingencyResult().getNetworkResult().getBusResults().stream()
                    .filter(b -> warmBaseV.containsKey(b.getBusId())).toList();
            double selfWorst = selfPairs.stream()
                    .mapToDouble(b -> Math.abs(b.getV() - warmBaseV.get(b.getBusId()))).max().orElse(0);
            long selfAbove = selfPairs.stream()
                    .filter(b -> Math.abs(b.getV() - warmBaseV.get(b.getBusId())) > 1e-12).count();
            System.out.printf("  BASE GPU-vs-GPU (cold vs warm, same arm) buses>1e-12kV=%d of %d"
                            + " | worst abs=%.3e kV%n", selfAbove, selfPairs.size(), selfWorst);
        }
        if ("1".equals(System.getenv("OLF_GPU_SA_BASE_ONLY"))) {
            // The BASE's own distributed active power, printed here because it is the cheapest reproducer of
            // the per-contingency spread: the base carries the SAME systematic difference (0.066 MW, the
            // device distributing less than OLF) that 462 of the window's contingencies show, and it needs one
            // load flow per arm instead of a thousand.
            double dapCpu = cpu.getPreContingencyResult().getDistributedActivePower();
            double dapCold = gpuCold.getPreContingencyResult().getDistributedActivePower();
            double dapWarm = gpuWarm.getPreContingencyResult().getDistributedActivePower();
            System.out.printf("  BASE distributedActivePower: cpu=%.9f gpu=%.9f MW | diff=%.6e MW%n",
                    dapCpu, dapCold, Math.abs(dapCpu - dapCold));
            // The SAME arm against ITSELF, as for the bus voltages above: the cold and warm GPU runs solve the
            // identical problem with the identical binary, so the gap between them is this quantity's own
            // reproducibility floor. It matters because the distributed amount is read off the slack row of a
            // state converged only to newtonRaphsonConvEpsPerEq, where that row's residual is ~0.2 MW at the
            // 1e-4 default — so the amount is FIRST-ORDER sensitive to which iterate the pass fires on
            // ([DS-CONV] measured 3 inner Newton iterations in one run and 2 in the other). A CPU-vs-GPU gap
            // no larger than this one is not evidence of a different system.
            System.out.printf("  BASE distributedActivePower GPU-vs-GPU (cold vs warm, same arm): "
                            + "cold=%.9f warm=%.9f MW | diff=%.6e MW%n",
                    dapCold, dapWarm, Math.abs(dapCold - dapWarm));
            System.out.println("  BASE_ONLY: stopping after the base comparison (no contingency simulated on the GPU arm)");
            return;
        }
        {
        }

        double worstDistributed = reportDistributedActivePower(cpu, gpuCold);

        long cpuViolations = violationCount(cpu);
        long gpuViolations = violationCount(gpuCold);
        System.out.printf("  statuses: CPU=%s GPU=%s%n", statusCounts(cpu), statusCounts(gpuCold));
        System.out.printf("  violations: CPU=%d GPU=%d diff=%d%n", cpuViolations, gpuViolations,
                Math.abs(cpuViolations - gpuViolations));

        // Find any status mismatches
        List<String> mismatched = cpu.getPostContingencyResults().stream()
                .filter(p -> {
                    PostContingencyResult g = gpuCold.getPostContingencyResults().stream()
                            .filter(q -> q.getContingency().getId().equals(p.getContingency().getId()))
                            .findFirst().orElse(null);
                    return g == null || g.getStatus() != p.getStatus();
                })
                .map(p -> p.getContingency().getId())
                .toList();
        if (!mismatched.isEmpty()) {
            System.out.printf("  STATUS MISMATCHES (%d): %s%n", mismatched.size(),
                    mismatched.subList(0, Math.min(20, mismatched.size())));
        }

        // Per-contingency violation diff: identify WHICH contingencies have differing violation counts
        Map<String, List<com.powsybl.contingency.violations.LimitViolation>> cpuViolByCtg = new TreeMap<>();
        Map<String, List<com.powsybl.contingency.violations.LimitViolation>> gpuViolByCtg = new TreeMap<>();
        cpu.getPostContingencyResults().forEach(p -> cpuViolByCtg.put(p.getContingency().getId(),
                p.getLimitViolationsResult().getLimitViolations()));
        gpuCold.getPostContingencyResults().forEach(p -> gpuViolByCtg.put(p.getContingency().getId(),
                p.getLimitViolationsResult().getLimitViolations()));
        if (!monBranches.isEmpty()) {
            Map<String, com.powsybl.security.results.BranchResult> gpuBr = new TreeMap<>();
            for (var gp : gpuWarm.getPostContingencyResults()) {
                gp.getNetworkResult().getBranchResults()
                        .forEach(b -> gpuBr.put(gp.getContingency().getId() + "|" + b.getBranchId(), b));
            }
            for (var cp : cpu.getPostContingencyResults()) {
                for (var cb : cp.getNetworkResult().getBranchResults()) {
                    var gb = gpuBr.get(cp.getContingency().getId() + "|" + cb.getBranchId());
                    if (gb == null) {
                        continue;
                    }
                    double i1d = Math.abs(cb.getI1() - gb.getI1());
                    System.out.printf("  BRDIFF ctg=%-14s %s I1 cpu=%.6f gpu=%.6f d=%.3e (%.3e rel) "
                            + "| P1 cpu=%.6f gpu=%.6f | Q1 cpu=%.6f gpu=%.6f%n",
                            cp.getContingency().getId(), cb.getBranchId(), cb.getI1(), gb.getI1(),
                            i1d, i1d / Math.max(1e-9, Math.abs(cb.getI1())),
                            cb.getP1(), gb.getP1(), cb.getQ1(), gb.getQ1());
                }
            }
        }
        if (busDiff) {
            for (var cp : cpu.getPostContingencyResults()) {
                var gp = gpuCold.getPostContingencyResults().stream()
                        .filter(x -> x.getContingency().getId().equals(cp.getContingency().getId()))
                        .findFirst().orElse(null);
                if (gp == null) {
                    continue;
                }
                Map<String, Double> gpuV = gpuSpill.load(gp.getContingency().getId());
                java.util.LinkedHashMap<String, Double> cpuVm = cpuSpill.load(cp.getContingency().getId());
                if (gpuV == null || cpuVm == null) {
                    continue;
                }
                List<com.powsybl.security.results.BusResult> cpuBuses = cpuVm.entrySet().stream()
                        .map(e -> new com.powsybl.security.results.BusResult("", e.getKey(), e.getValue(), 0))
                        .toList();
                List<String> worst = cpuBuses.stream()
                        .filter(b -> gpuV.containsKey(b.getBusId()))
                        .sorted(java.util.Comparator.comparingDouble(
                                (com.powsybl.security.results.BusResult b) -> -Math.abs(b.getV() - gpuV.get(b.getBusId()))))
                        .limit(8)
                        .map(b -> String.format("%s %.6f vs %.6f (%.3e)", b.getBusId(), b.getV(),
                                gpuV.get(b.getBusId()), b.getV() - gpuV.get(b.getBusId())))
                        .toList();
                // 1e-12 kV is the project's bar for "the two arms agree" — anything looser hides exactly
                // the class of difference this instrument exists to find.
                long above = cpuBuses.stream()
                        .filter(b -> gpuV.containsKey(b.getBusId()))
                        .filter(b -> Math.abs(b.getV() - gpuV.get(b.getBusId())) > 1e-12).count();
                double worstAbs = cpuBuses.stream()
                        .filter(b -> gpuV.containsKey(b.getBusId()))
                        .mapToDouble(b -> Math.abs(b.getV() - gpuV.get(b.getBusId()))).max().orElse(0);
                double worstRel = cpuBuses.stream()
                        .filter(b -> gpuV.containsKey(b.getBusId()))
                        .mapToDouble(b -> Math.abs(b.getV() - gpuV.get(b.getBusId())) / Math.max(1e-9, Math.abs(b.getV())))
                        .max().orElse(0);
                // Absolute AND relative: 1e-12 kV is below one ulp of a 400 kV bus (~5.7e-14), so an
                // absolute bar alone cannot tell a real divergence from accumulated round-off.
                // Same uncompared counters as the BASE diff. For a topology-changing outage the first
                // question is whether the two arms even agree on WHICH BUSES SURVIVE it — a bus one arm
                // disabled and the other kept is a different network, and every neighbouring bus's
                // balance shifts. The intersection-only view cannot show that.
                long cpuOnly = cpuBuses.stream()
                        .filter(b -> !gpuV.containsKey(b.getBusId())).count();
                long gpuOnly = gpuV.size() - (cpuBuses.size() - cpuOnly);
                System.out.printf("  BUSDIFF ctg=%s buses>1e-12kV=%d of %d | worst abs=%.3e kV rel=%.3e"
                                + " | uncompared: cpu-only=%d gpu-only=%d%n",
                        cp.getContingency().getId(), above, cpuBuses.size(),
                        worstAbs, worstRel, cpuOnly, gpuOnly);
                worst.forEach(w -> System.out.printf("      %s%n", w));
                // OLF_GPU_SA_MON_BUS=<id>[,<id>]: both arms' voltage at NAMED buses, whether or not they
                // are among the worst. A bus the two arms disagree about because one PINNED it and the
                // other did not shows up here as a target-V vs off-target pair, which the worst-8 list
                // will not surface when the pin's effect lands somewhere else in the network.
                String monBus = System.getenv("OLF_GPU_SA_MON_BUS");
                if (monBus != null) {
                    java.util.Set<String> want = java.util.Set.of(monBus.split(","));
                    cpuBuses.stream()
                            .filter(b -> want.contains(b.getBusId()) && gpuV.containsKey(b.getBusId()))
                            .forEach(b -> System.out.printf("      MONBUS ctg=%s bus=%s cpu=%.9f gpu=%.9f d=%.3e%n",
                                    cp.getContingency().getId(), b.getBusId(), b.getV(),
                                    gpuV.get(b.getBusId()), b.getV() - gpuV.get(b.getBusId())));
                }
            }
        }
        if (cpuSpill != null) {
            cpuSpill.close();
        }
        if (gpuSpill != null) {
            gpuSpill.close();
        }
        // The UNION of both key sets, not just the CPU's. The violation-diff metric is a TOTAL
        // (|cpuViolations - gpuViolations|), so a contingency present in ONE arm's results and not
        // the other contributes to it while an iteration over cpuViolByCtg alone never visits it —
        // measured at 500 busbar sections as "violation diff: 1" with no VIOL-DIFF line printed,
        // which reads as "nothing differs" and is the same intersection-only trap the bus diff's
        // uncompared counters exist to close.
        java.util.Set<String> allCtgIds = new TreeSet<>(cpuViolByCtg.keySet());
        allCtgIds.addAll(gpuViolByCtg.keySet());
        for (String ctgId : allCtgIds) {
            boolean cpuHas = cpuViolByCtg.containsKey(ctgId);
            boolean gpuHas = gpuViolByCtg.containsKey(ctgId);
            if (cpuHas != gpuHas) {
                System.out.printf("  VIOL-DIFF ctg=%s *** RESULT PRESENT IN ONLY ONE ARM *** cpu=%s gpu=%s"
                        + " (violations cpu=%d gpu=%d)%n", ctgId, cpuHas, gpuHas,
                        cpuViolByCtg.getOrDefault(ctgId, List.of()).size(),
                        gpuViolByCtg.getOrDefault(ctgId, List.of()).size());
            }
            int cpuCount = cpuViolByCtg.getOrDefault(ctgId, List.of()).size();
            int gpuCount = gpuViolByCtg.getOrDefault(ctgId, List.of()).size();
            if (cpuCount != gpuCount) {
                System.out.printf("  VIOL-DIFF ctg=%s cpu=%d gpu=%d delta=%d%n", ctgId, cpuCount, gpuCount, gpuCount - cpuCount);
                // Print the violations that differ (one-sided)
                var cpuSet = new java.util.HashSet<String>();
                cpuViolByCtg.getOrDefault(ctgId, List.of()).forEach(v ->
                        cpuSet.add(v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide()));
                var gpuSet = new java.util.HashSet<String>();
                gpuViolByCtg.getOrDefault(ctgId, List.of()).forEach(v ->
                        gpuSet.add(v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide()));
                for (var v : cpuViolByCtg.getOrDefault(ctgId, List.of())) {
                    String key = v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide();
                    if (!gpuSet.contains(key)) {
                        System.out.printf("    CPU-only: %s %s %s value=%.6f limit=%.6f margin=%.6e%n",
                                v.getSubjectId(), v.getLimitType(), v.getSide(), v.getValue(),
                                v.getLimit() * v.getLimitReduction(),
                                Math.abs(v.getValue() - v.getLimit() * v.getLimitReduction()));
                    }
                }
                for (var v : gpuViolByCtg.getOrDefault(ctgId, List.of())) {
                    String key = v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide();
                    if (!cpuSet.contains(key)) {
                        var subj = network.getIdentifiable(v.getSubjectId());   // the KIND of the element, not its data
                        System.out.printf("    GPU-only: %s %s %s value=%.6f limit=%.6f margin=%.6e kind=%s%s%n",
                                v.getSubjectId(), v.getLimitType(), v.getSide(), v.getValue(),
                                v.getLimit() * v.getLimitReduction(),
                                Math.abs(v.getValue() - v.getLimit() * v.getLimitReduction()),
                                subj == null ? "?" : subj.getType(),
                                subj instanceof com.powsybl.iidm.network.TwoWindingsTransformer t
                                        ? " rtc=" + t.hasRatioTapChanger() + " ptc=" + t.hasPhaseTapChanger()
                                          + " rtcReg=" + (t.hasRatioTapChanger() && t.getRatioTapChanger().isRegulating())
                                          + " ptcReg=" + (t.hasPhaseTapChanger() && t.getPhaseTapChanger().isRegulating())
                                        : "");
                    }
                }
            }
        }

        // ASSERTIONS
        // after every per-contingency diff line, so a failing run still says whether the STATES agree
        assertTrue(worstDistributed < 1e-3, "distributed active power must match the CPU: worst " + worstDistributed + " MW");
        assertEquals(statusCounts(cpu), statusCounts(gpuCold),
                "post-contingency status counts must match CPU");
        // Violation count: allow a small tolerance for border flips (ulp-level value near limit).
        // With powsybl's 0-default thresholds every solver-tolerance flicker re-reports; the
        // increased-violations thresholds above suppress most, but a few (measured: 2 out of 583k)
        // remain at the criterion's slack. Assert the diff is under 0.01% of the total.
        // |cpuViolations - gpuViolations| is a SIGNED NET over the whole run and lets real
        // disagreements CANCEL: measured at 500 busbar sections as "violation diff: 1 (0.0027%)",
        // which reads as clean, while the per-contingency lines showed TEN differing contingencies
        // including one at +6. Sum the ABSOLUTE per-contingency deltas instead, so nothing cancels,
        // and report the differing-contingency count next to it - the same lesson as the
        // intersection-only bus diff, one level up.
        long violDiff = 0;
        int differingCtg = 0;
        for (String ctgId : allCtgIds) {
            int d = gpuViolByCtg.getOrDefault(ctgId, List.of()).size()
                    - cpuViolByCtg.getOrDefault(ctgId, List.of()).size();
            if (d != 0) {
                violDiff += Math.abs(d);
                differingCtg++;
            }
        }
        long netDiff = Math.abs(cpuViolations - gpuViolations);
        double violDiffPct = cpuViolations > 0 ? 100.0 * violDiff / cpuViolations : 0;
        System.out.printf("  violation diff: %d over %d contingencies (%.4f%%) | signed net %d%n",
                violDiff, differingCtg, violDiffPct, netDiff);
        assertTrue(violDiffPct < 0.01,
                "violation diff " + violDiff + " over " + differingCtg + " contingencies ("
                        + violDiffPct + "%) exceeds 0.01% threshold");
    }

    /** OLF_GPU_SA_MEM=1: print the JVM uptime at an arm boundary, to cut an external memory trace (GC log with
     *  uptime decoration, RSS sampled against the process start) into per-arm windows, and the live heap there
     *  (after a full GC — never inside a timed region). */
    private static void armMark(String what) {
        if ("1".equals(System.getenv("OLF_GPU_SA_MEM"))) {
            long uptime = java.lang.management.ManagementFactory.getRuntimeMXBean().getUptime();
            System.gc();                                     // outside every timed region: the live heap at the mark
            Runtime rt = Runtime.getRuntime();
            System.out.printf("  ARM_MARK %s uptime=%d ms live heap=%d MB%n", what, uptime,
                    (rt.totalMemory() - rt.freeMemory()) >> 20);
        }
    }
}
