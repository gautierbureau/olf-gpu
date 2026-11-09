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
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.BatchContingency;
import com.powsybl.openloadflow.ac.solver.gpu.batch.plan.GpuBatchScopeCensus;
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

    /** Per contingency, |cpu-gpu| distributed active power of the last comparison (MW). */
    private static final Map<String, Double> DAP_BY_CTG = new TreeMap<>();

    /** Per contingency, the violation-count delta (gpu - cpu) of the last comparison. */
    private static final Map<String, Integer> VIOL_DELTA_BY_CTG = new TreeMap<>();

    /** Per contingency, the worst bus-voltage difference in kV — only for the ids that were bus-diffed. */
    private static final Map<String, Double> BUS_WORST_BY_CTG = new TreeMap<>();

    /**
     * The THREE signals together, per contingency, because no one of them settles anything on its own:
     * distributed active power is a single scalar and can match to 1e-12 MW while the operating point differs
     * by 0.45 kV (measured on rte6515 BUS-1869_BBS_1), a violation delta can be zero while the state drifts,
     * and a bus diff is only available for the ids a run monitors. Any contingency that differs in ANY of the
     * three is listed, with "-" where a signal was not measured, so a reader can see at a glance whether a
     * difference is one signal or all of them.
     */
    private static void reportTriple() {
        java.util.Set<String> ids = new java.util.TreeSet<>();
        DAP_BY_CTG.forEach((k, v) -> {
            if (v > 1e-6) {
                ids.add(k);
            }
        });
        VIOL_DELTA_BY_CTG.forEach((k, v) -> {
            if (v != 0) {
                ids.add(k);
            }
        });
        BUS_WORST_BY_CTG.forEach((k, v) -> {
            if (v > 1e-9) {
                ids.add(k);
            }
        });
        if (ids.isEmpty()) {
            System.out.println("  TRIPLE CHECK: no contingency differs in distributed power, violations or"
                    + " monitored bus voltage");
            return;
        }
        System.out.printf("  TRIPLE CHECK (%d contingencies differ in at least one of the three):%n", ids.size());
        System.out.printf("      %-20s %14s %8s %14s%n", "contingency", "dap |d| MW", "viol d", "bus worst kV");
        for (String id : ids) {
            Double dap = DAP_BY_CTG.get(id);
            Integer vd = VIOL_DELTA_BY_CTG.get(id);
            Double bw = BUS_WORST_BY_CTG.get(id);
            System.out.printf("      %-20s %14s %8s %14s%n", id,
                    dap == null ? "-" : String.format("%.3e", dap),
                    vd == null ? "-" : Integer.toString(vd),
                    bw == null ? "-" : String.format("%.3e", bw));
        }
    }

    /** Count what the filter removed, so a scope report says it out loud instead of hiding it in a
     *  batching share. */
    private static <T> List<T> keepSolved(List<T> all, java.util.function.Predicate<T> live, String kind) {
        List<T> kept = all.stream().filter(live).toList();
        // OLF_GPU_SA_ID_PROBE=<id>[,<id>]: for each named element, whether THIS scope kept it. A mapper
        // reject naming an element the harness never contingencied is not a harness bug: contingency
        // PROPAGATION runs on the IIDM network and pulls in neighbours, which can be elements our own
        // filter deliberately dropped. The probe is what tells those two apart.
        String probe = System.getenv("OLF_GPU_SA_ID_PROBE");
        if (probe != null) {
            java.util.Set<String> want = java.util.Arrays.stream(probe.split(",")).map(String::trim)
                    .collect(Collectors.toSet());
            for (T t : all) {
                if (t instanceof com.powsybl.iidm.network.Identifiable<?> i && want.contains(i.getId())) {
                    System.out.printf("  ID_PROBE %s kind=%s present=true kept=%b%n", i.getId(), kind, kept.contains(t));
                }
            }
        }
        if (kept.size() != all.size()) {
            System.out.printf("  CTG_FILTER %s: %d of %d dropped (out of service, or outside the "
                    + "network OLF solves - no result on either arm)%n", kind, all.size() - kept.size(), all.size());
        }
        return kept;
    }

    private static List<Contingency> allBranchContingencies(Network network) throws java.io.IOException {
        // OLF_GPU_SA_CTG_FILE=<file.json>: a FIXED list of contingencies, in powsybl's contingency JSON (a bare
        // array of {id, elements}, or a contingency list object) instead of the kinds built below. The list is
        // taken as is - no solved-network filter, OLF drops a contingency without impact itself - and the
        // CTG_FIRST / CTG_EXCLUDE / CTG_ONLY levers still apply to it. The operational list for network.xiidm
        // (14266 contingencies: lines, transformers, generators, busbar sections, shunts, SVC, HVDC, some
        // multi-element) is such a file.
        String ctgFile = System.getenv("OLF_GPU_SA_CTG_FILE");
        List<Contingency> fixedList = null;
        if (ctgFile != null && !ctgFile.isEmpty()) {
            com.fasterxml.jackson.databind.ObjectMapper om = com.powsybl.commons.json.JsonUtil.createObjectMapper()
                    .registerModule(new com.powsybl.contingency.json.ContingencyJsonModule());
            com.fasterxml.jackson.databind.JsonNode root = om.readTree(java.nio.file.Path.of(ctgFile).toFile());
            com.fasterxml.jackson.databind.JsonNode array = root.isArray() ? root : root.get("contingencies");
            List<Contingency> fixed = om.readerFor(new com.fasterxml.jackson.core.type.TypeReference<List<Contingency>>() { })
                    .readValue(array);
            java.util.Map<String, Long> byKind = new java.util.TreeMap<>();
            for (Contingency c : fixed) {
                for (var e : c.getElements()) {
                    byKind.merge(e.getType().name(), 1L, Long::sum);
                }
            }
            System.out.printf("  CTG_FILE %s: %d contingencies, elements by kind %s%n", ctgFile, fixed.size(), byKind);
            fixedList = new java.util.ArrayList<>(fixed);   // the RANGE / ONLY / EXCLUDE levers below still apply
        }
        // OLF_GPU_SA_ID_KIND=<id>[,<id>]: what CLASS an id actually is in the IIDM network. A mapper
        // reject naming an id the branch scope never built a contingency for is answered here: the
        // class says whether the harness could have covered it at all.
        String idKind = System.getenv("OLF_GPU_SA_ID_KIND");
        if (idKind != null) {
            for (String id : idKind.split(",")) {
                com.powsybl.iidm.network.Identifiable<?> i = network.getIdentifiable(id.trim());
                String extra = "";
                if (i instanceof com.powsybl.iidm.network.Switch switchElement) {
                    com.powsybl.iidm.network.VoltageLevel vl = switchElement.getVoltageLevel();
                    extra = " retained=" + switchElement.isRetained() + " open=" + switchElement.isOpen() + " kind=" + switchElement.getKind()
                            + " vl=" + vl.getId() + " vlBusesInMainCC="
                            + vl.getBusView().getBusStream().filter(b -> b.getConnectedComponent() != null
                                    && b.getConnectedComponent().getNum()
                                        == com.powsybl.iidm.network.ComponentConstants.MAIN_NUM).count()
                            + "/" + vl.getBusView().getBusStream().count();
                }
                System.out.printf("  ID_KIND %s -> %s%s%n", id.trim(),
                        i == null ? "ABSENT from the IIDM network" : i.getClass().getSimpleName() + " / " + i.getType(),
                        extra);
            }
        }
        String kinds = fixedList != null ? "" : System.getenv().getOrDefault("OLF_GPU_SA_CTG_KINDS", "branch");
        List<Contingency> all = fixedList != null ? fixedList : new java.util.ArrayList<>();
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
        // BOUNDARY LINES (formerly dangling lines) are their own population: BoundaryLine extends
        // Injection, not Branch, so the branch scope does not reach them, and they are neither
        // generators nor loads. OLF models one as an LfBoundaryLineGenerator — a generator-shaped
        // injection carrying the boundary's P/Q — so a boundary-line outage is an injection outage
        // whose bus may be anywhere. (TIE lines need no kind of their own: TieLine extends Branch, so
        // they are already in the branch scope.)
        if (kinds.contains("boundaryline")) {
            List<String> ids = keepSolved(network.getBoundaryLineStream().toList(),
                    b -> inSolvedNetwork(b.getTerminal()), "boundaryline").stream()
                    .map(com.powsybl.iidm.network.BoundaryLine::getId).toList();
            ids.forEach(id -> all.add(new Contingency(id,
                    new com.powsybl.contingency.BoundaryLineContingency(id))));
            System.out.printf("  + %d boundary line contingencies%n", ids.size());
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
        addN2Contingencies(network, all);
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
            // OLF_GPU_SA_CTG_ONLY_KEEP_TOPO=1: the dropped contingencies' elements go into ONE extra contingency,
            // so OLF retains the same switches and the selected ones solve the full scope's network.
            List<com.powsybl.contingency.ContingencyElement> dropped = new ArrayList<>();
            boolean keepTopo = "1".equals(System.getenv("OLF_GPU_SA_CTG_ONLY_KEEP_TOPO"));
            all.removeIf(c -> {
                if (want.contains(c.getId())) {
                    return false;
                }
                if (keepTopo) {
                    dropped.addAll(c.getElements());
                }
                return true;
            });
            if (!dropped.isEmpty()) {
                all.add(new Contingency("KEEP_TOPO", dropped));
            }
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

    /** N-2: OLF_GPU_SA_N2_BRANCH / _N2_GEN pairs, drawn from a fixed seed and preferentially from elements sharing
     *  a voltage level. */
    private static void addN2Contingencies(final Network network, final List<Contingency> all) {
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

    /** {@code OLF_GPU_SA_STATUS_LIST=1}: every NON-CONVERGED contingency of an arm, by id. The
     *  aggregate counts say the arms AGREE; only the ids say WHICH contingencies OLF itself fails, and
     *  that is what splits a device reject into "OLF fails it too, so the device may answer it
     *  directly" and "OLF converges it, so the device is losing one" (tasks #67). */
    private static void listNonConverged(String arm, SecurityAnalysisResult r) {
        if (!"1".equals(System.getenv("OLF_GPU_SA_STATUS_LIST"))) {
            return;
        }
        for (PostContingencyResult p : r.getPostContingencyResults()) {
            if (p.getStatus() != PostContingencyComputationStatus.CONVERGED) {
                System.out.println("  NONCONV " + arm + " " + p.getContingency().getId() + " " + p.getStatus().name());
            }
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
    /** The GPU arm against ITSELF after every contingency: the cold and the warm run solve the identical problem
     *  with the identical binary and the identical chunks, so whatever differs between them is the device's own
     *  run-to-run spread, and a CPU-vs-GPU gap of that size says nothing about the equations. Statuses and the
     *  distributed active power; OLF_GPU_SA_SELF_DUMP=<file> writes every contingency's cold and warm values. */
    private static void reportGpuSelfSpread(SecurityAnalysisResult cold, SecurityAnalysisResult warm) throws java.io.IOException {
        Map<String, PostContingencyResult> w = new TreeMap<>();
        warm.getPostContingencyResults().forEach(p -> w.put(p.getContingency().getId(), p));
        long identical = 0;
        long statusDiff = 0;
        List<Map.Entry<String, Double>> moved = new ArrayList<>();
        String dump = System.getenv("OLF_GPU_SA_SELF_DUMP");
        try (java.io.PrintWriter out = dump != null ? new java.io.PrintWriter(Files.newBufferedWriter(Path.of(dump))) : null) {
            for (PostContingencyResult c : cold.getPostContingencyResults()) {
                PostContingencyResult x = w.get(c.getContingency().getId());
                if (x == null) {
                    continue;
                }
                if (c.getStatus() != x.getStatus()) {
                    statusDiff++;
                }
                double d = Math.abs(c.getDistributedActivePower() - x.getDistributedActivePower());
                if (d == 0 && c.getStatus() == x.getStatus()) {
                    identical++;
                } else {
                    moved.add(Map.entry(c.getContingency().getId(), d));
                }
                if (out != null) {
                    out.println(c.getContingency().getId() + "\t" + c.getStatus() + "\t" + x.getStatus() + "\t"
                            + Double.toString(c.getDistributedActivePower()) + "\t" + Double.toString(x.getDistributedActivePower()));
                }
            }
        }
        moved.sort(Map.Entry.<String, Double>comparingByValue().reversed());
        System.out.printf("  POST GPU-vs-GPU (cold vs warm, same chunks): %d identical, %d moved, %d status changes | worst %.6e MW %s%n",
                identical, moved.size(), statusDiff, moved.isEmpty() ? 0.0 : moved.get(0).getValue(),
                moved.stream().limit(8).map(e -> e.getKey() + "=" + String.format("%.3e", e.getValue())).toList());
    }

    private static double reportDistributedActivePower(SecurityAnalysisResult cpu, SecurityAnalysisResult gpu) throws java.io.IOException {
        double baseDiff = Math.abs(cpu.getPreContingencyResult().getDistributedActivePower()
                - gpu.getPreContingencyResult().getDistributedActivePower());
        System.out.printf("  distributedActivePower base: cpu=%.9f gpu=%.9f MW%n",
                cpu.getPreContingencyResult().getDistributedActivePower(), gpu.getPreContingencyResult().getDistributedActivePower());
        Map<String, Double> gpuDap = new TreeMap<>();
        gpu.getPostContingencyResults().forEach(p -> gpuDap.put(p.getContingency().getId(),
                                      p.getDistributedActivePower()));
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
        DAP_BY_CTG.clear();                                  // kept for the joined verdict (see reportTriple)
        pairs.forEach((k, v) -> DAP_BY_CTG.put(k, Math.abs(v[0] - v[1])));
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
        // OLF_GPU_SA_CTG_FIRST=<id>[,<id>]: move these to the FRONT of the list, keeping the list
        // itself intact (so the retained switches, and therefore the network both arms solve, are
        // unchanged - which is what CTG_ONLY cannot do). OLF's zero-impedance spanning tree is
        // history-dependent within one security analysis: removing a ZI tree edge re-elects the tree
        // (LfZeroImpedanceNetwork.removeBranchAndTryToSplit) and re-adding the branch does NOT
        // restore it, so from the first ZI contingency onwards OLF solves on a tree the saved base
        // state's dummy flows no longer match. Running a contingency FIRST is how that history is
        // taken out of the measurement.
        String ctgFirst = System.getenv("OLF_GPU_SA_CTG_FIRST");
        if (ctgFirst != null && !ctgFirst.isEmpty()) {
            contingencies = moveToFront(contingencies, ctgFirst);
        }
        System.out.printf("%s: %d branches → %d N-1 contingencies%n", CASE_FILE,
                network.getBranchCount(), contingencies.size());
        // OLF_GPU_SA_CTG_DUMP=<file.json>: write THIS list (same order, same ids, same elements) as a powsybl
        // contingency-list JSON and stop - so another tool (the official pypowsybl, add_contingencies_from_json_file)
        // can run exactly the scope this harness compares.
        String ctgDump = System.getenv("OLF_GPU_SA_CTG_DUMP");
        if (ctgDump != null && !ctgDump.isEmpty()) {
            com.fasterxml.jackson.databind.ObjectMapper om = com.powsybl.commons.json.JsonUtil.createObjectMapper()
                    .registerModule(new com.powsybl.contingency.json.ContingencyJsonModule());
            om.writerWithDefaultPrettyPrinter().writeValue(Path.of(ctgDump).toFile(),
                    new com.powsybl.contingency.list.DefaultContingencyList("harness-scope", contingencies));
            System.out.printf("  CTG_DUMP %d contingencies -> %s%n", contingencies.size(), ctgDump);
            return;
        }

        LoadFlowParameters lfp = JsonLoadFlowParameters.read(PARAMS_FILE);
        // OLF_GPU_DBG_CONV_EPS: tighten the inner-Newton criterion on BOTH arms. The only honest way to
        // tell a real state difference from convergence slack is to squeeze the slack and watch what
        // survives — see the standing rule against calling a difference a tolerance artefact unproven.
        // OLF_GPU_DBG_RL: TRACE the CPU arm's ReactiveLimitsOuterLoop, so every PV<->PQ switch it
        // makes (bus id, q, qLimit) can be diffed against the device's GPU_RL / GPU_RL_FINAL switch
        // list. A PV -> PQ switch the device makes and OLF does not is invisible in any residual - the device
        // converges cleanly against the WRONG equation - so the two switch lists are the only
        // direct evidence. Ported from GpuVsCpuProductionSaBenchmarkTest; the harness has no
        // appender configured (core/classic version mismatch), so attach one.
        // OLF_GPU_DBG_NR: the CPU arm's own Newton trajectory — |f(x0)|, one line per iteration, and
        // every MAX_VOLTAGE_CHANGE step-size clamp it applies. Put beside the device's own trajectory
        // this is the only direct way to see WHERE the two trajectories part at the DEFAULT criterion,
        // which is the question: the two arms must take the same iterates, not merely land in the same
        // tolerance ball. Tightening the tolerance hides this instead of answering it.
        applyNewtonDebug();
        applyReactiveLimitsDebug();
        // OLF_GPU_SA_NO_RL / OLF_GPU_SA_NO_DS: disable an outer loop on BOTH arms. Stage isolation -
        // inner Newton only, vs +DS, vs +DS+RL - to find which stage first introduces a divergence.
        // Decisive for a suspected PV -> PQ SWITCH mismatch: if the arms agree with reactive limits OFF, the
        // underlying power flow (and every bus q) agrees and the fault is in the switch DECISION; if
        // they still disagree, q itself is wrong and the switch is only reporting it.
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
        // OLF_GPU_SVC=1: enable SecondaryVoltageControl — the RTE params default to off, but
        // the rte6515 case carries SVC zones. This is the path that differed between GPU and
        // CPU when SVC support was incomplete (the base LF didn't go through the chunk solver).
        applySvcScope(lfp, network);
        // Non-zero increased-violations thresholds to avoid ulp-level flicker
        SecurityAnalysisParameters sap = new SecurityAnalysisParameters();
        sap.setLoadFlowParameters(lfp);
        // (constructor order: lowV abs, lowV prop, highV abs, highV prop, FLOW prop - the flow threshold is 0.1 %).
        // OLF_GPU_SA_DEFAULT_INCREASED=1: powsybl's defaults instead (flow 10 %), as pypowsybl uses them.
        sap.setIncreasedViolationsParameters("1".equals(System.getenv("OLF_GPU_SA_DEFAULT_INCREASED"))
                ? new SecurityAnalysisParameters.IncreasedViolationsParameters()
                : new SecurityAnalysisParameters.IncreasedViolationsParameters(0.5, 0.001, 0.5, 0.001, 0.001));

        // CPU arm: 8 threads (production-realistic) on a BUS-BREAKER case, ONE thread on a
        // NODE-BREAKER one. OLF partitions the contingency list across threads and each partition
        // builds its OWN LfTopoConfig, so each lands on its own retained-switch set and its own bus
        // decomposition — while the GPU arm is single-threaded by construction and builds one network
        // for the whole list. The two arms then solve DIFFERENT networks and the numbers are not
        // comparable.
        //
        // This used to key on the KIND (busbar -> 1, everything else -> 8), which is the shape of the
        // measurement that found it, not the cause. The cause is BREAKER MODE, and what turns it on is
        // the contingency LIST, not the kind and not the topology: OLF is in breaker mode exactly when
        // some contingency's propagation opens a switch or loses a bus (LfTopoConfig#isBreaker), and
        // only then does a partition's own retained-switch set change its bus decomposition. Both these
        // cases are node-breaker, and they differ: rte6515's branch/load/generator lists open nothing,
        // so every partition builds the same network and 8 threads are safe (and 7x faster), while
        // network.xiidm's lists DO open switches — 3180 of its 7495 load contingencies have a branch to
        // open — so 8 threads split the retained switches and the arms stop being comparable.
        //
        // Measured on network.xiidm's full branch scope: the SAME GPU arm (3273 violations, 365 s)
        // scores 3273 against a 1-thread CPU arm and 3187 against an 8-thread one, and the
        // distributed-power worst goes from 1.4e-05 MW to 1.93 MW. That reads exactly like a GPU
        // regression and cost a full bisect to clear. Ask OLF instead of guessing.
        // OLF_GPU_SA_CPU_THREADS still overrides.
        int cpuThreads = cpuThreadCount(network, contingencies, lfp);
        sap.addExtension(OpenSecurityAnalysisParameters.class,
                new OpenSecurityAnalysisParameters().setThreadCount(cpuThreads));

        // ONE NETWORK PER ARM. The three runs used to share a single Network object, which is fine for
        // branch/generator/load outages but NOT for busbar-section ones: those open switches, and any
        // switch state the restore does not put back is inherited by whichever arm runs next. The GPU
        // arm runs second, so it started from a progressively more perturbed topology the more busbar
        // contingencies had already executed — which is exactly the shape of the "busbar 8.176e-06 kV
        // on a 0%-batched kind" that stood recorded as a GPU/CPU difference. Read before timing, so a
        // second parse never lands inside a measured region.
        long readCold0 = System.nanoTime();
        Network networkGpuCold = Network.read(CASE_FILE);
        long readWarm0 = System.nanoTime();
        Network networkGpuWarm = Network.read(CASE_FILE);
        System.out.printf("  NETWORK_READ gpu-cold arm %d ms | gpu-warm arm %d ms%n",
                (readWarm0 - readCold0) / 1_000_000, (System.nanoTime() - readWarm0) / 1_000_000);
        // OLF_GPU_SA_CPU_WARM=1: time the CPU arm TWICE, like the GPU arm. The first CPU run is the
        // JVM's first SA and pays JIT compilation the GPU arms never see, so the cold-vs-cold ratio
        // flatters the GPU. The ratios are then quoted against CPU warm.
        boolean cpuWarmRun = "1".equals(System.getenv("OLF_GPU_SA_CPU_WARM"));
        Network networkCpuWarm = cpuWarmRun ? Network.read(CASE_FILE) : null;

        // OLF_GPU_SA_STRATEGIES=1: operator strategies on every k-th contingency (OLF_GPU_SA_STRATEGY_EVERY, default 20),
        // one per action kind the device composes, built from the network's own elements (syntheticStrategies)
        if ("1".equals(System.getenv("OLF_GPU_SA_STRATEGIES"))) {
            syntheticStrategies(network, contingencies,
                    Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_STRATEGY_EVERY", "20")));
        }
        // OLF_GPU_SA_STRATEGY_FILE=<json>: the actions and operator strategies of a case file in pypowsybl's names
        // ("actions" SWITCH / PHASE_TAP_CHANGER_POSITION, "operator_strategies")
        String strategyFile = System.getenv("OLF_GPU_SA_STRATEGY_FILE");
        if (strategyFile != null) {
            strategiesFromFile(Path.of(strategyFile));
        }
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
        java.util.Set<String> busDiffCtgs = new java.util.TreeSet<>(System.getenv("OLF_GPU_SA_BUSDIFF_CTG") == null
                ? java.util.Set.of()
                : java.util.Arrays.stream(System.getenv("OLF_GPU_SA_BUSDIFF_CTG").split(","))
                        .map(String::trim).filter(x -> !x.isEmpty()).collect(Collectors.toSet()));
        // OLF_GPU_SA_BUSDIFF_STRATEGIES=1: the same for every contingency a strategy starts from, whose monitor gives
        // its strategies their bus results too (distributed power matched to 1e-9 MW on network.xiidm LLFP while a
        // strategy's buses were 8.5e-4 kV apart)
        if ("1".equals(System.getenv("OLF_GPU_SA_BUSDIFF_STRATEGIES"))) {
            strategies.forEach(strategy -> busDiffCtgs.add(strategy.getContingencyContext().getContingencyId()));
        }
        // OLF_GPU_SA_BUSDIFF=0 turns it off even with CTG_ONLY: a timing run on an id-selected list must not
        // pay one BusResult per bus per contingency on both arms (1000 rte6515 branches, cold GPU arm: 34.1 s with it, 9.8 s without).
        boolean busDiff = !"0".equals(System.getenv("OLF_GPU_SA_BUSDIFF"))
                && (System.getenv("OLF_GPU_SA_CTG_ONLY") != null
                || !busDiffCtgs.isEmpty()
                || "1".equals(System.getenv("OLF_GPU_SA_BUSDIFF")));
        // OLF_GPU_SA_MON_BRANCH=<id>[,<id>]: monitor these branches and print BOTH arms' flows. A
        // one-sided violation says only that the other arm is on the other side of the limit — it does
        // NOT say by how much, and "the margin is small" is a statement about the CPU's distance to its
        // limit, not about the distance between the two arms.
        java.util.Set<String> monBranches = System.getenv("OLF_GPU_SA_MON_BRANCH") == null
                ? java.util.Set.of()
                : java.util.Arrays.stream(System.getenv("OLF_GPU_SA_MON_BRANCH").split(","))
                        .map(String::trim).collect(Collectors.toSet());
        List<com.powsybl.security.monitor.StateMonitor> monitors = buildMonitors(network, busDiff, busDiffCtgs, monBranches);
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
        // OLF_GPU_SA_NO_SPILL=1: keep every bus result in the result objects (a small list, read back afterwards)
        boolean spill = busDiff && !"1".equals(System.getenv("OLF_GPU_SA_NO_SPILL"));
        BusVoltageSpill gpuSpill = spill ? new BusVoltageSpill("gpu") : null;
        CpuArm cpuArm = runCpuArm(cpuArmMode, cpuArmDir, busDiff, spill, network, contingencies, monitors, sap, cpuT0);
        if (cpuArm == null) {
            return;
        }
        BusVoltageSpill cpuSpill = cpuArm.cpuSpill();
        SecurityAnalysisResult cpu = cpuArm.cpu();
        long cpuMs = cpuArm.cpuMs();
        // OLF_GPU_SA_CPU_WARM=1: time the CPU arm TWICE, like the GPU arm. The first CPU run is the JVM's first SA
        // and pays JIT compilation the GPU arms never see, so the cold-vs-cold ratio flatters the GPU. The ratios
        // are quoted against CPU warm. Not available with a loaded (separate-JVM) CPU arm.
        armMark("cpu end");
        long cpuWarmMs = -1;
        if (cpuWarmRun && !"load".equals(cpuArmMode) && networkCpuWarm != null) {
            long cpuT1 = System.nanoTime();
            SecurityAnalysisResult cpuWarm = runSa(networkCpuWarm, contingencies, monitors, sap);
            cpuWarmMs = (System.nanoTime() - cpuT1) / 1_000_000;
            System.out.printf("  CPU warm run: violations=%d statuses=%s%n", violationCount(cpuWarm), statusCounts(cpuWarm));
        }
        long cpuRefMs = cpuWarmMs > 0 ? cpuWarmMs : cpuMs;

        // GPU arm: single-thread, cold + warm runs
        securityAnalysisProvider = new GpuSecurityAnalysisProvider(new SparseMatrixFactory(),
                new EvenShiloachGraphDecrementalConnectivityFactory<>());
        GpuArms runGpuArmsOut = runGpuArms(lfp, sap, networkGpuCold, contingencies, monitors, gpuSpill, cpu,
                                           networkGpuWarm, busDiff);
        SecurityAnalysisResult gpuCold = runGpuArmsOut.gpuCold();
        SecurityAnalysisResult gpuWarm = runGpuArmsOut.gpuWarm();
        long gpuColdMs = runGpuArmsOut.gpuColdMs();
        long gpuWarmMs = runGpuArmsOut.gpuWarmMs();

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
        reportBaseVoltages(gpuCold, cpu, gpuWarm);

        // OLF_GPU_SA_GPU_RESULT_DUMP=<file.json>: the cold GPU arm's result, in the same powsybl JSON as a saved CPU
        // arm's cpu.json, so the two can be compared value by value (the "violations" line below compares counts).
        String gpuDump = System.getenv("OLF_GPU_SA_GPU_RESULT_DUMP");
        if (gpuDump != null && !gpuDump.isEmpty()) {
            try (java.io.Writer w = java.nio.file.Files.newBufferedWriter(Path.of(gpuDump))) {
                com.powsybl.security.json.SecurityAnalysisResultSerializer.write(gpuCold, w);
            }
            System.out.printf("  GPU_RESULT_DUMP -> %s%n", gpuDump);
        }
        double worstDistributed = reportDistributedActivePower(cpu, gpuCold);

        long cpuViolations = violationCount(cpu);
        long gpuViolations = violationCount(gpuCold);
        System.out.printf("  statuses: CPU=%s GPU=%s%n", statusCounts(cpu), statusCounts(gpuCold));
        listNonConverged("CPU", cpu);
        listNonConverged("GPU", gpuCold);
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
        // A status comparison is only evidence for a contingency the DEVICE solved: for one it declined, the
        // GPU arm's status comes from the CPU fallback, so comparing it with the CPU arm compares OLF with OLF
        // and validates nothing. Split the two populations and name the one that IS a device failure: a
        // contingency the device declined and OLF converged.
        java.util.Set<String> onDevice = GpuBatchedAcSecurityAnalysis.LAST_GPU_PRECOMPUTED_IDS;
        reportOnDeviceScope(onDevice, cpu);
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
        reportMonitoredBranches(monBranches, gpuWarm, cpu);
        reportBusVoltageDiffs(busDiff, spill, cpu, gpuCold, gpuSpill, cpuSpill);
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
        reportViolationDiffs(allCtgIds, cpuViolByCtg, gpuViolByCtg, cpu, network);

        List<String> strategyMismatches = compareStrategies(cpu, gpuCold);

        // ASSERTIONS
        // after every per-contingency diff line, so a failing run still says whether the STATES agree
        reportTriple();
        // before any assertion: a failing run is exactly when the joined table is wanted
        assertTrue(strategyMismatches.isEmpty(), "operator strategy results must match the CPU: " + strategyMismatches.size()
                + " differ " + strategyMismatches.subList(0, Math.min(20, strategyMismatches.size())));
        assertTrue(worstDistributed < 1e-3, "distributed active power must match the CPU: worst " + worstDistributed + " MW");
        assertEquals(statusCounts(cpu), statusCounts(gpuCold),
                "post-contingency status counts must match CPU");
        // per contingency, not only the counts: a device convergence where OLF fails and a device failure where
        // OLF converges cancel in the counts, and either one alone is a bug
        assertTrue(mismatched.isEmpty(), "post-contingency status must match CPU per contingency: " + mismatched.size()
                + " differ " + mismatched.subList(0, Math.min(20, mismatched.size())));
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

    /** OLF_GPU_DBG_NR: the Newton-Raphson debug parameters. */
    private void applyNewtonDebug() {
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
    }

    /** OLF_GPU_DBG_RL: the reactive-limit loop's debug parameters. */
    private void applyReactiveLimitsDebug() {
        if (System.getenv("OLF_GPU_DBG_RL") != null) {
            ch.qos.logback.classic.Logger reactiveLimitsLogger = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger("com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop");
            reactiveLimitsLogger.setLevel(ch.qos.logback.classic.Level.TRACE);
            // The voltage-MONITORING loop too (standby SVC automata): its PQ->PV activations are the
            // other per-contingency DECISION the two arms can disagree on, and like a PV -> PQ switch they leave
            // no residual - only the two decision logs tell them apart.
            ch.qos.logback.classic.Logger monLog = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger("com.powsybl.openloadflow.ac.outerloop.MonitoringVoltageOuterLoop");
            monLog.setLevel(ch.qos.logback.classic.Level.TRACE);
            // ...and the HVDC AC-emulation freeze: its per-contingency setpoint (Reports.reportFreezeHvdc,
            // INFO) is the third decision that leaves no residual - the frozen P IS the equation.
            ch.qos.logback.classic.Logger hvdcLogger = (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory
                    .getLogger("com.powsybl.openloadflow.ac.outerloop.FreezingHvdcACEmulationOuterloop");
            hvdcLogger.setLevel(ch.qos.logback.classic.Level.TRACE);
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
                reactiveLimitsLogger.addAppender(ca);
                monLog.addAppender(ca);
                hvdcLogger.addAppender(ca);
            }
        }
    }

    /** OLF_GPU_SVC=1: the secondary voltage control scope (the parameters and the network settings it needs). */
    private void applySvcScope(final LoadFlowParameters lfp, final Network network) {
        if ("1".equals(System.getenv("OLF_GPU_SVC"))) {
            com.powsybl.openloadflow.OpenLoadFlowParameters.get(lfp).setSecondaryVoltageControl(true);
            System.out.println("  SVC enabled (OLF_GPU_SVC=1)");
            // OLF_GPU_SVC_PILOT_SHIFT=<kV>: move EVERY zone's pilot-point target so the loop actually
            // FIRES (with the case's own targets it is stable at the base and on every contingency).
            // rte6515 carries two zones — shifting only one leaves the other's controllers idle and the
            // run then exercises half the zone systems.
            String shift = System.getenv("OLF_GPU_SVC_PILOT_SHIFT");
            var secondaryVoltageControl = network.getExtension(com.powsybl.iidm.network.extensions.SecondaryVoltageControl.class);
            if (shift != null && secondaryVoltageControl != null) {
                for (var zone : secondaryVoltageControl.getControlZones()) {
                    var pilot = zone.getPilotPoint();
                    double before = pilot.getTargetV();
                    pilot.setTargetV(before + Double.parseDouble(shift));
                    System.out.printf("  SVC pilot '%s' target %.3f -> %.3f kV (%d control units)%n",
                            zone.getName(), before, pilot.getTargetV(), zone.getControlUnits().size());
                }
            }
        }
    }

    /** The BASE load flow, bus by bus: every post-contingency state is computed from the base, so a base that
     *  differs puts a floor under every contingency, which post-contingency comparisons cannot see. */
    private void reportBaseVoltages(final SecurityAnalysisResult gpuCold, final SecurityAnalysisResult cpu,
                                    final SecurityAnalysisResult gpuWarm) {
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

    /** The contingencies the device answered: how many, and which of the mismatched ones were among them. */
    private void reportOnDeviceScope(final java.util.Set<String> onDevice, final SecurityAnalysisResult cpu) {
        if (!onDevice.isEmpty()) {
            List<String> declinedButOlfConverged = cpu.getPostContingencyResults().stream()
                    .filter(p -> !onDevice.contains(p.getContingency().getId()))
                    .filter(p -> p.getStatus() == PostContingencyComputationStatus.CONVERGED)
                    .map(p -> p.getContingency().getId())
                    .toList();
            long declined = cpu.getPostContingencyResults().stream()
                    .filter(p -> !onDevice.contains(p.getContingency().getId())).count();
            // Split the declined ones, or the number misleads: a scenario REFUSED AT PLAN TIME was never
            // attempted on the device (an unmodeled shape — OLF converges those perfectly well, they are a
            // scope gap), while one the device TRIED and threw away is a device failure whenever OLF converges
            // it. By ID (GpuBatchScopeCensus.planRefusedIds): the census COUNTS are per attempt — cold + warm
            // runs and the robust retry each count a refusal again — and subtracting them from a
            // per-contingency list printed "-20 of them tried and failed" and called 12 plan refusals
            // (re-merged mains switched to PQ in the base / RELOC_VROW) device failures (2026-10-04).
            java.util.Set<String> planRefusedIds = GpuBatchScopeCensus.planRefusedIds();
            List<String> triedAndFailedOlfConverged = declinedButOlfConverged.stream()
                    .filter(id -> !planRefusedIds.contains(id)).toList();
            List<String> refusedOlfConverged = declinedButOlfConverged.stream()
                    .filter(planRefusedIds::contains).toList();
            long declinedRefused = cpu.getPostContingencyResults().stream()
                    .map(p -> p.getContingency().getId())
                    .filter(id -> !onDevice.contains(id) && planRefusedIds.contains(id)).count();
            System.out.printf("  DEVICE VERDICT: %d of %d simulated solved on device; %d declined = %d refused at"
                    + " plan time (unmodeled shape) + %d tried on the device and failed%n",
                    onDevice.size(), cpu.getPostContingencyResults().size(), declined, declinedRefused,
                    declined - declinedRefused);
            System.out.printf("  DEVICE FAILURES on a system OLF converges: %d (tried and failed — each one a device"
                    + " bug): %s%n", triedAndFailedOlfConverged.size(), triedAndFailedOlfConverged);
            System.out.printf("  PLAN REFUSALS on a system OLF converges: %d (scope gaps, CPU answers): %s%n",
                    refusedOlfConverged.size(),
                    refusedOlfConverged.stream().map(id -> id + "(" + GpuBatchScopeCensus.planRefusalTag(id) + ")").toList());
            System.out.printf("  STATUS COMPARISON covers the %d device-solved contingencies; for the rest both"
                    + " arms report the SAME CPU computation%n", onDevice.size());
        }
    }

    /** The monitored branches' flows in both arms (OLF_GPU_SA_MON_BRANCH). */
    private void reportMonitoredBranches(final java.util.Set<String> monBranches,
                                         final SecurityAnalysisResult gpuWarm, final SecurityAnalysisResult cpu) {
        if (!monBranches.isEmpty()) {
            Map<String, com.powsybl.security.results.BranchResult> gpuBr = new TreeMap<>();
            for (var gpuResult : gpuWarm.getPostContingencyResults()) {
                gpuResult.getNetworkResult().getBranchResults()
                        .forEach(b -> gpuBr.put(gpuResult.getContingency().getId() + "|" + b.getBranchId(), b));
            }
            for (var cpuResult : cpu.getPostContingencyResults()) {
                for (var cpuBranch : cpuResult.getNetworkResult().getBranchResults()) {
                    var gpuBranch = gpuBr.get(cpuResult.getContingency().getId() + "|" + cpuBranch.getBranchId());
                    if (gpuBranch == null) {
                        continue;
                    }
                    double i1d = Math.abs(cpuBranch.getI1() - gpuBranch.getI1());
                    // BOTH sides: a one-sided violation is usually on side TWO, and side ONE matching says
                    // nothing about it (rte6515 BUS-125_BBS_1: LINE-2533-1468's I1 agreed to 1e-13 while only
                    // the CPU reported a CURRENT TWO violation 21% over its limit).
                    double i2d = Math.abs(cpuBranch.getI2() - gpuBranch.getI2());
                    System.out.printf("  BRDIFF ctg=%-14s %s I1 cpu=%.6f gpu=%.6f d=%.3e (%.3e rel) "
                            + "| I2 cpu=%.6f gpu=%.6f d=%.3e (%.3e rel)"
                            + "| P1 cpu=%.6f gpu=%.6f | Q1 cpu=%.6f gpu=%.6f%n",
                            cpuResult.getContingency().getId(), cpuBranch.getBranchId(), cpuBranch.getI1(), gpuBranch.getI1(),
                            i1d, i1d / Math.max(1e-9, Math.abs(cpuBranch.getI1())),
                            cpuBranch.getI2(), gpuBranch.getI2(), i2d, i2d / Math.max(1e-9, Math.abs(cpuBranch.getI2())),
                            cpuBranch.getP1(), gpuBranch.getP1(), cpuBranch.getQ1(), gpuBranch.getQ1());
                }
            }
        }
    }

    /** The post-contingency bus voltages of both arms, compared contingency by contingency from the spilled
     *  results. */
    private void reportBusVoltageDiffs(final boolean busDiff, final boolean spill, final SecurityAnalysisResult cpu,
                                       final SecurityAnalysisResult gpuCold, final BusVoltageSpill gpuSpill,
                                       final BusVoltageSpill cpuSpill) throws java.io.IOException {
        if (busDiff && spill) {
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
                BUS_WORST_BY_CTG.put(cp.getContingency().getId(), worstAbs);
                System.out.printf("  BUSDIFF ctg=%s buses>1e-12kV=%d of %d | worst abs=%.3e kV rel=%.3e"
                                + " | uncompared: cpu-only=%d gpu-only=%d%n",
                        cp.getContingency().getId(), above, cpuBuses.size(),
                        worstAbs, worstRel, cpuOnly, gpuOnly);
                worst.forEach(w -> System.out.printf("      %s%n", w));
                // OLF_GPU_SA_MON_BUS=<id>[,<id>]: both arms' voltage at NAMED buses, whether or not they
                // are among the worst. A bus the two arms disagree about because one SWITCHED it to PQ and the
                // other did not shows up here as a target-V vs off-target pair, which the worst-8 list
                // will not surface when the switch's effect lands somewhere else in the network.
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
    }

    /** Every contingency whose violations differ between the arms, over the union of both arms' ids (a contingency
     *  present in one arm only is listed too). */
    private void reportViolationDiffs(final java.util.Set<String> allCtgIds,
                                      final Map<String, List<com.powsybl.contingency.violations.LimitViolation>> cpuViolByCtg,
                                      final Map<String, List<com.powsybl.contingency.violations.LimitViolation>> gpuViolByCtg,
                                      final SecurityAnalysisResult cpu, final Network network) {
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
            VIOL_DELTA_BY_CTG.put(ctgId, gpuCount - cpuCount);
            if (cpuCount != gpuCount) {
                System.out.printf("  VIOL-DIFF ctg=%s cpu=%d gpu=%d delta=%d%n", ctgId, cpuCount, gpuCount, gpuCount - cpuCount);
                // Whether the one-sided entry ALREADY violated in the pre-contingency state decides what it is.
                // A NEW violation missing on one arm is a defect. A PRE-EXISTING one is filtered by powsybl's
                // increased-violations thresholds (this harness sets 0.5 proportional on flows), so it is
                // re-reported only past `base + min(abs, prop*base)` — a boundary a 1e-5-relative state
                // difference flips either way, which is what `docs/increased-violations-thresholds.md` is about.
                // Print the violations that differ (one-sided)
                var cpuSet = new java.util.HashSet<String>();
                cpuViolByCtg.getOrDefault(ctgId, List.of()).forEach(v ->
                        cpuSet.add(v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide()));
                var gpuSet = new java.util.HashSet<String>();
                gpuViolByCtg.getOrDefault(ctgId, List.of()).forEach(v ->
                        gpuSet.add(v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide()));
                var preKeys = new java.util.HashMap<String, Double>();
                for (var v : cpu.getPreContingencyResult().getLimitViolationsResult().getLimitViolations()) {
                    preKeys.put(v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide(), v.getValue());
                }
                for (var v : cpuViolByCtg.getOrDefault(ctgId, List.of())) {
                    String key = v.getSubjectId() + "|" + v.getLimitType() + "|" + v.getSide();
                    if (!gpuSet.contains(key)) {
                        Double base = preKeys.get(key);
                        System.out.printf("    CPU-only: %s %s %s value=%.6f limit=%.6f margin=%.6e pre=%s%n",
                                v.getSubjectId(), v.getLimitType(), v.getSide(), v.getValue(),
                                v.getLimit() * v.getLimitReduction(),
                                Math.abs(v.getValue() - v.getLimit() * v.getLimitReduction()),
                                base == null ? "NEW" : String.format("pre-existing base=%.6f", base));
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
    }

    private List<com.powsybl.contingency.strategy.OperatorStrategy> strategies = List.of();
    private List<com.powsybl.action.Action> actions = List.of();

    private SecurityAnalysisResult runSa(Network n, List<Contingency> ctgs, List<com.powsybl.security.monitor.StateMonitor> monitors,
                                         SecurityAnalysisParameters sap) {
        // an empty list (the CPU arm under OLF_GPU_SA_GPU_ONLY) carries no strategy: OLF rejects a strategy whose
        // contingency is not in the list
        return strategies.isEmpty() || ctgs.isEmpty() ? runSecurityAnalysis(n, ctgs, monitors, sap)
                : runSecurityAnalysis(n, ctgs, monitors, sap, strategies, actions,
                                      com.powsybl.commons.report.ReportNode.NO_OP);
    }

    /**
     * Operator strategies on every {@code every}-th contingency, one per action kind the device composes, on elements
     * picked deterministically from the network (by id order): a generator moved relative and absolute, a load moved,
     * a line opened, the contingency's own branch reclosed, a shunt section added, a phase tap moved, an AC-emulation
     * HVDC switched to set points, and a two-stage strategy (always + any violation). Prints which elements it took.
     */
    private void syntheticStrategies(Network network, List<Contingency> contingencies, int every) {
        List<com.powsybl.action.Action> acts = new ArrayList<>();
        List<String> common = new ArrayList<>();
        java.util.function.Predicate<com.powsybl.iidm.network.Terminal> live = t -> t.isConnected()
                && t.getBusView().getBus() != null && t.getBusView().getBus().isInMainSynchronousComponent();
        network.getGeneratorStream()
                .filter(g -> live.test(g.getTerminal()) && g.getMaxP() < 5000 && g.getTargetP() > 50
                        && g.getTargetP() < g.getMaxP() - 50 && g.getTargetP() > g.getMinP() + 50)
                .max(java.util.Comparator.comparingDouble(com.powsybl.iidm.network.Generator::getMaxP)
                        .thenComparing(com.powsybl.iidm.network.Generator::getId))
                .ifPresent(g -> {
                    acts.add(new com.powsybl.action.GeneratorActionBuilder().withId("genUp").withGeneratorId(g.getId())
                            .withActivePowerRelativeValue(true).withActivePowerValue(20).build());
                    acts.add(new com.powsybl.action.GeneratorActionBuilder().withId("genSet").withGeneratorId(g.getId())
                            .withActivePowerRelativeValue(false).withActivePowerValue(g.getTargetP() - 20).build());
                    common.add("genUp");
                    common.add("genSet");
                    if ("1".equals(System.getenv("OLF_GPU_SA_STRATEGY_NOOP"))) {   // the pair re-solves its parent
                        acts.add(new com.powsybl.action.GeneratorActionBuilder().withId("genZero").withGeneratorId(g.getId())
                                .withActivePowerRelativeValue(true).withActivePowerValue(0).build());
                        common.add("genZero");
                    }
                    System.out.printf("  STRATEGY generator %s%n", g.getId());
                });
        network.getLoadStream().filter(l -> live.test(l.getTerminal()) && l.getP0() > 50)
                .max(java.util.Comparator.comparingDouble(com.powsybl.iidm.network.Load::getP0)
                        .thenComparing(com.powsybl.iidm.network.Load::getId))
                .ifPresent(l -> {
                    acts.add(new com.powsybl.action.LoadActionBuilder().withId("loadDown").withLoadId(l.getId())
                            .withRelativeValue(true).withActivePowerValue(-20).build());
                    common.add("loadDown");
                    System.out.printf("  STRATEGY load %s%n", l.getId());
                });
        List<com.powsybl.iidm.network.Line> lines = network.getLineStream()
                .filter(l -> live.test(l.getTerminal1()) && live.test(l.getTerminal2()))
                .sorted(java.util.Comparator.comparing(com.powsybl.iidm.network.Line::getId)).toList();
        if (!lines.isEmpty()) {
            String lineId = lines.get(lines.size() / 2).getId();
            acts.add(new com.powsybl.action.TerminalsConnectionActionBuilder().withId("openLine")
                    .withNetworkElementId(lineId).withOpen(true).build());
            common.add("openLine");
            System.out.printf("  STRATEGY line %s%n", lineId);
        }
        network.getShuntCompensatorStream()
                .filter(sc -> live.test(sc.getTerminal()) && sc.getModelType() == com.powsybl.iidm.network.ShuntCompensatorModelType.LINEAR
                        && sc.getSectionCount() < sc.getMaximumSectionCount())
                .min(java.util.Comparator.comparing(com.powsybl.iidm.network.ShuntCompensator::getId))
                .ifPresent(sc -> {
                    acts.add(new com.powsybl.action.ShuntCompensatorPositionActionBuilder().withId("shuntUp")
                            .withShuntCompensatorId(sc.getId()).withSectionCount(sc.getSectionCount() + 1).build());
                    common.add("shuntUp");
                    System.out.printf("  STRATEGY shunt %s%n", sc.getId());
                });
        network.getTwoWindingsTransformerStream()
                .filter(t -> live.test(t.getTerminal1()) && live.test(t.getTerminal2()) && t.hasPhaseTapChanger()
                        && t.getPhaseTapChanger().getTapPosition() < t.getPhaseTapChanger().getHighTapPosition())
                .min(java.util.Comparator.comparing(com.powsybl.iidm.network.TwoWindingsTransformer::getId))
                .ifPresent(t -> {
                    acts.add(new com.powsybl.action.PhaseTapChangerTapPositionActionBuilder().withId("pstUp")
                            .withTransformerId(t.getId()).withRelativeValue(true).withTapPosition(1).build());
                    common.add("pstUp");
                    System.out.printf("  STRATEGY phase shifter %s%n", t.getId());
                });
        network.getHvdcLineStream()
                .filter(h -> live.test(h.getConverterStation1().getTerminal()) && live.test(h.getConverterStation2().getTerminal())
                        && h.getExtension(com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControl.class) != null
                        && h.getExtension(com.powsybl.iidm.network.extensions.HvdcAngleDroopActivePowerControl.class).isEnabled())
                .min(java.util.Comparator.comparing(com.powsybl.iidm.network.HvdcLine::getId))
                .ifPresent(h -> {
                    acts.add(new com.powsybl.action.HvdcActionBuilder().withId("hvdcOff").withHvdcId(h.getId())
                            .withAcEmulationEnabled(false).build());
                    common.add("hvdcOff");
                    System.out.printf("  STRATEGY AC-emulation HVDC %s%n", h.getId());
                });
        // OLF_GPU_SA_STRATEGY_KINDS=<kind>[,<kind>]: only these kinds (genUp, genSet, genZero, loadDown, openLine, shuntUp,
        // pstUp, hvdcOff, close, twoStage)
        String kindsEnv = System.getenv("OLF_GPU_SA_STRATEGY_KINDS");
        java.util.Set<String> kinds = kindsEnv == null ? null : java.util.Set.of(kindsEnv.split(","));
        if (kinds != null) {
            common.retainAll(kinds);
        }
        List<com.powsybl.contingency.strategy.OperatorStrategy> strats = new ArrayList<>();
        int chosen = 0;
        // OLF_GPU_SA_STRATEGY_ONLY=<id>[,<id>]: strategies on these contingencies only (a bisection lever)
        String onlyEnv = System.getenv("OLF_GPU_SA_STRATEGY_ONLY");
        java.util.Set<String> onlyParents = onlyEnv == null ? null : java.util.Set.of(onlyEnv.split(","));
        for (int i = 0; i < contingencies.size(); i += onlyParents != null ? 1 : every) {
            Contingency c = contingencies.get(i);
            if (c.getId().startsWith("KEEP_TOPO") || onlyParents != null && !onlyParents.contains(c.getId())) {
                continue;                                    // the harness's topology placeholder, not a real outage
            }
            chosen++;
            var ctx = com.powsybl.contingency.ContingencyContext.specificContingency(c.getId());
            for (String a : common) {
                strats.add(new com.powsybl.contingency.strategy.OperatorStrategy("s-" + a + "-" + c.getId(), ctx,
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of(a)));
            }
            if ((kinds == null || kinds.contains("close")) && c.getElements().size() == 1 && c.getElements().get(0) instanceof BranchContingency bc
                    && network.getLine(bc.getId()) != null) {
                String closeId = "close-" + bc.getId();
                acts.add(new com.powsybl.action.TerminalsConnectionActionBuilder().withId(closeId)
                        .withNetworkElementId(bc.getId()).withOpen(false).build());
                strats.add(new com.powsybl.contingency.strategy.OperatorStrategy("s-close-" + c.getId(), ctx,
                        new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of(closeId)));
            }
            if ((kinds == null || kinds.contains("twoStage")) && acts.stream().anyMatch(a -> a.getId().equals("genUp"))
                    && acts.stream().anyMatch(a -> a.getId().equals("openLine"))) {
                strats.add(new com.powsybl.contingency.strategy.OperatorStrategy("s-twoStage-" + c.getId(), ctx, List.of(
                        new com.powsybl.contingency.strategy.ConditionalActions("always",
                                new com.powsybl.contingency.strategy.condition.TrueCondition(), List.of("genUp")),
                        new com.powsybl.contingency.strategy.ConditionalActions("onViolation",
                                new com.powsybl.contingency.strategy.condition.AnyViolationCondition(), List.of("openLine")))));
            }
        }
        this.actions = List.copyOf(acts);
        this.strategies = List.copyOf(strats);
        String parentsDump = System.getenv("OLF_GPU_SA_STRATEGY_PARENTS_DUMP");   // the contingencies given strategies, in order
        if (parentsDump != null) {
            try {
                Files.write(Path.of(parentsDump), strats.stream()
                        .map(strategy -> strategy.getContingencyContext().getContingencyId()).distinct().toList());
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
        System.out.printf("  STRATEGIES: %d on %d contingencies (every %d-th), %d actions%n", strats.size(), chosen, every, acts.size());
    }

    /** Actions and operator strategies from a case file (see OLF_GPU_SA_STRATEGY_FILE): SWITCH (switch_id, open) and
     *  PHASE_TAP_CHANGER_POSITION (transformer_id, tap_position, relative) actions; strategies on one contingency with
     *  TRUE_CONDITION (default) or AT_LEAST_ONE_VIOLATION_CONDITION. Counts only are printed (network data rule). */
    private void strategiesFromFile(Path file) {
        com.fasterxml.jackson.databind.JsonNode root;
        try {
            root = com.powsybl.commons.json.JsonUtil.createObjectMapper().readTree(file.toFile());
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
        List<com.powsybl.action.Action> acts = new ArrayList<>();
        for (com.fasterxml.jackson.databind.JsonNode a : root.get("actions")) {
            String id = a.get("id").asText();
            String type = a.get("type").asText();
            if ("SWITCH".equals(type)) {
                acts.add(new com.powsybl.action.SwitchActionBuilder().withId(id).withNetworkElementId(a.get("switch_id").asText())
                        .withOpen(a.get("open").asBoolean()).build());
            } else if ("PHASE_TAP_CHANGER_POSITION".equals(type)) {
                acts.add(new com.powsybl.action.PhaseTapChangerTapPositionActionBuilder().withId(id)
                        .withTransformerId(a.get("transformer_id").asText()).withTapPosition(a.get("tap_position").asInt())
                        .withRelativeValue(a.path("relative").asBoolean(false)).build());
            } else {
                throw new IllegalArgumentException("action type " + type + " not supported");
            }
        }
        List<com.powsybl.contingency.strategy.OperatorStrategy> strats = new ArrayList<>();
        for (com.fasterxml.jackson.databind.JsonNode s : root.get("operator_strategies")) {
            String condition = s.path("condition_type").asText("TRUE_CONDITION");
            com.powsybl.contingency.strategy.condition.Condition c = switch (condition) {
                case "TRUE_CONDITION" -> new com.powsybl.contingency.strategy.condition.TrueCondition();
                case "AT_LEAST_ONE_VIOLATION_CONDITION" -> new com.powsybl.contingency.strategy.condition.AnyViolationCondition();
                default -> throw new IllegalArgumentException("condition " + condition + " not supported");
            };
            List<String> ids = new ArrayList<>();
            s.get("action_ids").forEach(x -> ids.add(x.asText()));
            strats.add(new com.powsybl.contingency.strategy.OperatorStrategy(s.get("id").asText(),
                    com.powsybl.contingency.ContingencyContext.specificContingency(s.get("contingency_id").asText()), c, ids));
        }
        this.actions = List.copyOf(acts);
        this.strategies = List.copyOf(strats);
        System.out.printf("  STRATEGIES from file: %d on %d contingencies, %d actions%n", strats.size(),
                strats.stream().map(x -> x.getContingencyContext().getContingencyId()).distinct().count(), acts.size());
    }

    /** One violation's identity across the arms. The LOCATION is part of it: a voltage level split into several buses
     *  reports a LOW_VOLTAGE per bus under one subject, and comparing their values crosswise reads two buses' distance
     *  as an arm gap (rte6515 VL-4551 after BUS-595_BBS: 397.0019 and 397.0774 kV, a phantom 7.55e-2 kV). */
    private static String violationKey(com.powsybl.contingency.violations.LimitViolation v) {
        return v.getSubjectId() + "|" + v.getSide() + "|" + v.getLimitType() + "|" + v.getLimitName() + "|" + v.getAcceptableDuration()
                + "|" + v.getViolationLocation().map(Object::toString).orElse("");
    }

    /** Strategy by strategy: status, distributed active power, violation count. Returns the differing ones. */
    private static List<String> compareStrategies(SecurityAnalysisResult cpu, SecurityAnalysisResult gpu) {
        // (the parent's bus voltages are read from the same results: the GPU's post-contingency ones may be dropped)
        if (cpu.getOperatorStrategyResults().isEmpty() && gpu.getOperatorStrategyResults().isEmpty()) {
            return List.of();
        }
        Map<String, com.powsybl.security.results.OperatorStrategyResult> g = new TreeMap<>();
        gpu.getOperatorStrategyResults().forEach(r -> g.put(r.getOperatorStrategy().getId(), r));
        List<String> mismatches = new ArrayList<>();
        Map<String, Long> statuses = new TreeMap<>();
        double worstDap = 0;
        String worstId = "-";
        int differingViolations = 0;
        int aboveFloor = 0;
        List<Map.Entry<String, Double>> ranked = new ArrayList<>();   // strategies above the floor, for the worst list
        List<Map.Entry<String, Double>> busRanked = new ArrayList<>();   // every bus-diffed strategy: its worst bus
        List<Map.Entry<String, Double>> violRanked = new ArrayList<>();  // the worst value gap over matching violations
        for (var c : cpu.getOperatorStrategyResults()) {
            String id = c.getOperatorStrategy().getId();
            var r = g.remove(id);
            var cs = c.getConditionalActionsResults().get(0);
            statuses.merge(cs.getStatus().name(), 1L, Long::sum);
            if (r == null) {
                mismatches.add(id + " (no GPU result)");
                continue;
            }
            var gs = r.getConditionalActionsResults().get(0);
            if (cs.getStatus() != gs.getStatus()) {
                mismatches.add(id + " status " + cs.getStatus() + " vs " + gs.getStatus());
                continue;
            }
            double d = Math.abs(cs.getDistributedActivePower() - gs.getDistributedActivePower());
            if ("1".equals(System.getenv("OLF_GPU_SA_STRATEGY_DAP_PRINT"))) {   // every strategy's two totals
                System.out.printf("    STRATEGY_DAP %s cpu=%.12f gpu=%.12f d=%.3e%n", id, cs.getDistributedActivePower(),
                        gs.getDistributedActivePower(), d);
            }
            if (Double.isFinite(d) && d > worstDap) {
                worstDap = d;
                worstId = id;
            }
            if (d > 1e-3) {
                mismatches.add(id + " distributed power " + cs.getDistributedActivePower() + " vs " + gs.getDistributedActivePower());
            }
            if (Double.isFinite(d) && d > 1e-9) {
                aboveFloor++;
                ranked.add(java.util.Map.entry(id, d));
            }
            if (cs.getLimitViolationsResult().getLimitViolations().size() != gs.getLimitViolationsResult().getLimitViolations().size()) {
                differingViolations++;
                System.out.printf("    STRATEGY_VIOLATIONS %s cpu=%d gpu=%d%n", id, cs.getLimitViolationsResult().getLimitViolations().size(),
                        gs.getLimitViolationsResult().getLimitViolations().size());
                // the violations one side has and the other does not, with value and limit (a knife edge shows as a
                // value on its limit)
                java.util.function.Function<com.powsybl.contingency.violations.LimitViolation, String> key = GpuVsCpuRte6515SaTest::violationKey;
                Map<String, com.powsybl.contingency.violations.LimitViolation> gByKey = new TreeMap<>();
                gs.getLimitViolationsResult().getLimitViolations().forEach(v -> gByKey.put(key.apply(v), v));
                Map<String, com.powsybl.contingency.violations.LimitViolation> cByKey = new TreeMap<>();
                cs.getLimitViolationsResult().getLimitViolations().forEach(v -> cByKey.put(key.apply(v), v));
                cByKey.forEach((k, v) -> {
                    if (!gByKey.containsKey(k)) {
                        System.out.printf("      CPU_ONLY %s value=%.12f limit=%.12f reduction=%.6f%n", k, v.getValue(), v.getLimit(),
                                v.getLimitReduction());
                    }
                });
                gByKey.forEach((k, v) -> {
                    if (!cByKey.containsKey(k)) {
                        System.out.printf("      GPU_ONLY %s value=%.12f limit=%.12f reduction=%.6f%n", k, v.getValue(), v.getLimit(),
                                v.getLimitReduction());
                    }
                });
            }
            // the values of the violations both arms report (a flow or a voltage, in its own unit): what the strategy
            // tells without a monitor
            Map<String, Double> gViol = new java.util.HashMap<>();
            gs.getLimitViolationsResult().getLimitViolations().forEach(v -> gViol.put(violationKey(v), v.getValue()));
            double violWorst = -1;
            String violSubject = "";
            for (var v : cs.getLimitViolationsResult().getLimitViolations()) {
                Double gvv = gViol.get(violationKey(v));
                if (gvv != null && Math.abs(gvv - v.getValue()) > violWorst) {
                    violWorst = Math.abs(gvv - v.getValue());
                    violSubject = v.getSubjectId() + " " + v.getLimitType();
                }
            }
            if (violWorst >= 0) {
                violRanked.add(java.util.Map.entry(id + "(" + violSubject + ")", violWorst));
            }
            // bus voltages, when the run monitors them: the worst bus, and the few worst for the worst strategy
            Map<String, Double> gv = new TreeMap<>();
            gs.getNetworkResult().getBusResults().forEach(b -> gv.put(b.getBusId(), b.getV()));
            List<double[]> diffs = new ArrayList<>();
            List<String> busIds = new ArrayList<>();
            Map<String, String> vlOfBus = new java.util.HashMap<>();   // a bus id that does not name its level
            for (var b : cs.getNetworkResult().getBusResults()) {
                Double v = gv.get(b.getBusId());
                if (v != null && Double.isFinite(v) && Double.isFinite(b.getV())) {
                    diffs.add(new double[] {Math.abs(v - b.getV()), busIds.size()});
                    busIds.add(b.getBusId());
                    vlOfBus.put(b.getBusId(), b.getVoltageLevelId());
                }
            }
            if (!diffs.isEmpty()) {
                diffs.sort((x, y) -> Double.compare(y[0], x[0]));
                StringBuilder top = new StringBuilder();
                for (int k = 0; k < Math.min(5, diffs.size()); k++) {
                    String bid = busIds.get((int) diffs.get(k)[1]);
                    String vl = vlOfBus.get(bid);
                    top.append(' ').append(bid).append(vl == null || bid.startsWith(vl) ? "" : "@" + vl)
                            .append('=').append(String.format("%.2e", diffs.get(k)[0]));
                }
                String wb = busIds.get((int) diffs.get(0)[1]);
                busRanked.add(java.util.Map.entry(id + "(" + wb + (vlOfBus.get(wb) == null || wb.startsWith(vlOfBus.get(wb))
                        ? "" : "@" + vlOfBus.get(wb)) + ")", diffs.get(0)[0]));
                if (!"1".equals(System.getenv("OLF_GPU_SA_STRATEGY_DAP_PRINT")) && diffs.get(0)[0] <= 1e-6) {
                    continue;                                // the tail line below counts it; only the far ones print
                }
                System.out.printf("    STRATEGY_BUS %s worst %.3e kV over %d buses:%s%n", id, diffs.get(0)[0], diffs.size(), top);
                String ctgId = c.getOperatorStrategy().getContingencyContext().getContingencyId();
                java.util.function.BiFunction<SecurityAnalysisResult, Boolean, Double> parentV = (res, x) -> res.getPostContingencyResults().stream()
                        .filter(postContingencyResult -> postContingencyResult.getContingency().getId().equals(ctgId)).findFirst()
                        .flatMap(postContingencyResult -> postContingencyResult.getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals(wb)).findFirst())
                        .map(b -> b.getV()).orElse(Double.NaN);
                double cpuS = cs.getNetworkResult().getBusResults().stream().filter(b -> b.getBusId().equals(wb)).findFirst().get().getV();
                System.out.printf("    STRATEGY_BUS_V %s %s: strategy cpu=%.9f gpu=%.9f | parent cpu=%.9f gpu=%.9f%n", id, wb,
                        cpuS, gv.get(wb), parentV.apply(cpu, true), parentV.apply(gpu, true));
            }
        }
        g.keySet().forEach(id -> mismatches.add(id + " (no CPU result)"));
        // a strategy OLF converges whose pair the device SOLVED and did not converge: a device failure (the CPU replay
        // then hides it in the statuses)
        java.util.Set<String> failed = GpuBatchScopeCensus.pairDeviceFailedIds();
        List<String> deviceFailures = cpu.getOperatorStrategyResults().stream()
                .filter(r -> r.getConditionalActionsResults().get(0).getStatus() == PostContingencyComputationStatus.CONVERGED)
                .map(r -> r.getOperatorStrategy().getContingencyContext().getContingencyId()
                        + BatchContingency.STRATEGY_KEY_SEP + r.getOperatorStrategy().getId())
                .filter(id -> failed.stream().anyMatch(f -> f.equals(id)
                        || f.startsWith(id + BatchContingency.STRATEGY_KEY_SEP)))
                .map(id -> id.replace(BatchContingency.STRATEGY_KEY_SEP, '|'))
                .toList();
        System.out.printf("  STRATEGY DEVICE FAILURES on a strategy OLF converges: %d %s%n", deviceFailures.size(),
                deviceFailures.subList(0, Math.min(20, deviceFailures.size())));
        deviceFailures.forEach(id -> mismatches.add(id + " (device failure, OLF converges)"));
        System.out.printf("  STRATEGY RESULTS: %d CPU / %d GPU | statuses %s | worst distributed power %.3e MW (%s), %d above 1e-9 MW"
                        + " | %d with a different violation count | %d mismatches%n",
                cpu.getOperatorStrategyResults().size(), gpu.getOperatorStrategyResults().size(), statuses,
                worstDap, worstId, aboveFloor, differingViolations, mismatches.size());
        mismatches.stream().limit(30).forEach(m -> System.out.println("    STRATEGY_DIFF " + m));
        // the worst strategies under the mismatch gate: a tail far above the parents' is a modelling gap, not slack
        ranked.sort((x, y) -> Double.compare(y.getValue(), x.getValue()));
        long above1e6 = ranked.stream().filter(e -> e.getValue() > 1e-6).count();
        long above1e4 = ranked.stream().filter(e -> e.getValue() > 1e-4).count();
        System.out.printf("  STRATEGY DAP TAIL: %d above 1e-6 MW, %d above 1e-4 MW; worst:%s%n", above1e6, above1e4,
                ranked.stream().limit(12).map(e -> String.format(" %s=%.2e", e.getKey(), e.getValue()))
                        .collect(java.util.stream.Collectors.joining()));
        for (var tail : List.of(java.util.Map.entry("BUS", busRanked), java.util.Map.entry("VIOLATION VALUE", violRanked))) {
            List<Map.Entry<String, Double>> l = new ArrayList<>(tail.getValue());
            if (l.isEmpty()) {
                continue;
            }
            l.sort((x, y) -> Double.compare(y.getValue(), x.getValue()));
            System.out.printf("  STRATEGY %s TAIL: %d strategies compared, %d above 1e-9, %d above 1e-6, %d above 1e-4 (kV for buses,"
                            + " the violation's unit otherwise); worst:%s%n", tail.getKey(), l.size(),
                    l.stream().filter(e -> e.getValue() > 1e-9).count(), l.stream().filter(e -> e.getValue() > 1e-6).count(),
                    l.stream().filter(e -> e.getValue() > 1e-4).count(),
                    l.stream().limit(12).map(e -> String.format(" %s=%.2e", e.getKey(), e.getValue()))
                            .collect(java.util.stream.Collectors.joining()));
        }
        return mismatches;
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

    /** {@code contingencies} with the listed ids moved to the front, in the listed order; the rest keep theirs. */
    private static List<Contingency> moveToFront(List<Contingency> contingencies, String ctgFirst) {
        List<String> wanted = java.util.Arrays.stream(ctgFirst.split(",")).map(String::trim).toList();
        List<Contingency> head = new ArrayList<>();
        for (String w : wanted) {
            contingencies.stream().filter(c -> c.getId().equals(w)).findFirst().ifPresent(head::add);
        }
        List<Contingency> reordered = new ArrayList<>(head);
        for (Contingency c : contingencies) {
            if (head.stream().noneMatch(h -> h.getId().equals(c.getId()))) {
                reordered.add(c);
            }
        }
        System.out.printf("  CTG_FIRST: %s moved to the front (list otherwise unchanged)%n", head.stream().map(Contingency::getId).toList());
        return reordered;
    }

    /** The CPU arm's thread count: one in breaker mode (see the call site), eight otherwise; OLF_GPU_SA_CPU_THREADS overrides. */
    private static int cpuThreadCount(Network network, List<Contingency> contingencies, LoadFlowParameters lfp) {
        com.powsybl.openloadflow.network.LfTopoConfig probeTopoConfig =
                new com.powsybl.openloadflow.network.LfTopoConfig();
        com.powsybl.openloadflow.network.impl.PropagatedContingency.createList(network, contingencies, probeTopoConfig,
                new com.powsybl.openloadflow.network.impl.PropagatedContingencyCreationParameters()
                        .setContingencyPropagation(true)
                        .setShuntCompensatorVoltageControlOn(lfp.isShuntCompensatorVoltageControlOn())
                        .setSlackDistributionOnConformLoad(lfp.getBalanceType()
                                == com.powsybl.loadflow.LoadFlowParameters.BalanceType.PROPORTIONAL_TO_CONFORM_LOAD)
                        .setHvdcAcEmulation(lfp.isHvdcAcEmulation()));
        String defaultThreads = probeTopoConfig.isBreaker() ? "1" : "8";
        return Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_CPU_THREADS", defaultThreads));
    }

    /** The state monitors: bus voltages under OLF_GPU_SA_BUSDIFF*, the OLF_GPU_SA_MON_BRANCH flows, and the N state. */
    private static List<com.powsybl.security.monitor.StateMonitor> buildMonitors(Network network, boolean busDiff,
                                                                              java.util.Set<String> busDiffCtgs,
                                                                              java.util.Set<String> monBranches) {
        if (!busDiff && monBranches.isEmpty()) {
            return Collections.emptyList();
        }
        // OLF_GPU_SA_BUSDIFF_VL_STRIDE=<k>: every k-th voltage level in id order, so that thousands of monitored
        // results fit the heap (network.xiidm: 4104 strategies x 13502 buses on each arm)
        int vlStride = Integer.parseInt(System.getenv().getOrDefault("OLF_GPU_SA_BUSDIFF_VL_STRIDE", "1"));
        List<String> vlIds = network.getVoltageLevelStream().map(com.powsybl.iidm.network.VoltageLevel::getId).sorted().toList();
        java.util.Set<String> monVls = busDiff
                ? java.util.stream.IntStream.range(0, vlIds.size()).filter(i -> i % vlStride == 0).mapToObj(vlIds::get)
                        .collect(Collectors.toSet())
                : java.util.Set.of();
        // TWO monitors, not one. ContingencyContext.all() left the PRE-contingency network result
        // EMPTY, so "BASE BUSDIFF buses>1e-12kV=0 of 0 | uncompared: cpu-only=0 gpu-only=0" was not
        // a clean base - it was no base compared at all, in every run this test has ever done. A
        // post-contingency difference cannot be attributed to the contingency until the BASE states
        // are known to agree, so add an explicit N-state monitor.
        List<com.powsybl.security.monitor.StateMonitor> ms = new ArrayList<>();
        if (busDiffCtgs.isEmpty()) {
            ms.add(new com.powsybl.security.monitor.StateMonitor(
                    com.powsybl.contingency.ContingencyContext.all(), monBranches,
                    monVls, java.util.Set.of()));
        } else {
            // one scoped monitor per id, so the heap holds a handful of results, not 10540
            for (String id : busDiffCtgs) {
                ms.add(new com.powsybl.security.monitor.StateMonitor(
                        com.powsybl.contingency.ContingencyContext.specificContingency(id),
                        monBranches, monVls, java.util.Set.of()));
            }
        }
        ms.add(new com.powsybl.security.monitor.StateMonitor(
                com.powsybl.contingency.ContingencyContext.none(), monBranches,
                monVls, java.util.Set.of()));
        return List.copyOf(ms);
    }

    /**
     * The CPU arm: run now, or loaded from a saved arm (OLF_GPU_SA_CPU_ARM_DIR / _MODE). Null once the arm was
     * saved, which ends the test.
     */
    private CpuArm runCpuArm(String cpuArmMode, String cpuArmDir, boolean busDiff, boolean spill, Network network,
                             List<Contingency> contingencies, List<com.powsybl.security.monitor.StateMonitor> monitors,
                             SecurityAnalysisParameters sap, long cpuT0) throws java.io.IOException {
        BusVoltageSpill cpuSpill = null;
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
            cpuSpill = spill ? new BusVoltageSpill("cpu") : null;
            // OLF_GPU_SA_GPU_ONLY=1: debug lever - skip the CPU arm (run it on an EMPTY contingency
            // list) so the GPU arm's own behaviour on a large scope can be traced in minutes.
            // Every comparison below is then meaningless; only the GPU arm's lines are.
            List<Contingency> cpuCtgs = "1".equals(System.getenv("OLF_GPU_SA_GPU_ONLY")) ? List.of() : contingencies;
            SecurityAnalysisResult raw = runSa(network, cpuCtgs, monitors, sap);
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
                return null;
            }
        }
        return new CpuArm(cpuSpill, cpu, cpuMs);
    }

    private record CpuArm(BusVoltageSpill cpuSpill, SecurityAnalysisResult cpu, long cpuMs) {
    }

    /** The two GPU arms, cold then warm (the warm one only when asked), each timed, with the scope census between
     *  them. */
    private GpuArms runGpuArms(final LoadFlowParameters lfp, final SecurityAnalysisParameters sap,
                               final Network networkGpuCold, final List<Contingency> contingencies,
                               final List<com.powsybl.security.monitor.StateMonitor> monitors, final BusVoltageSpill gpuSpill,
                               final SecurityAnalysisResult cpu, final Network networkGpuWarm, final boolean busDiff)
                               throws java.io.IOException, java.text.ParseException {
        SecurityAnalysisResult gpuCold;
        SecurityAnalysisResult gpuWarm;
        long gpuColdMs;
        long gpuWarmMs;
        // OLF_GPU_SA_BANK=<slots>: cap the GPU arms' bank at that many slots instead of the device-memory sizing (the bank
        // is balanced over the call's scenarios, then refilled from the queue), to measure what its width costs
        String bank = System.getenv("OLF_GPU_SA_BANK");
        if (bank != null && !bank.isEmpty()) {
            com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache.forceChunkSizeForTests(Integer.parseInt(bank));
        }
        try {
            SecurityAnalysisParameters sapGpu = new SecurityAnalysisParameters();
            sapGpu.setLoadFlowParameters(lfp);
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
            SecurityAnalysisResult rawCold = runSa(networkGpuCold, contingencies, monitors, sapGpu);
            gpuColdMs = (System.nanoTime() - g0) / 1_000_000;
            gpuCold = gpuSpill != null ? gpuSpill.spill(rawCold) : rawCold;
            rawCold = null;
            System.gc();
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
                SecurityAnalysisResult rawWarm = runSa(networkGpuWarm, contingencies, monitors, sapGpu);
                gpuWarmMs = (System.nanoTime() - g1) / 1_000_000;
                gpuWarm = busDiff ? BusVoltageSpill.dropPostBusResults(rawWarm) : rawWarm;   // only its BASE is compared
                rawWarm = null;
                System.gc();
                Map<String, Long> censusWarm = GpuBatchScopeCensus.snapshot();
                System.out.printf("  GPU CENSUS (warm): %s%n", censusWarm);
                reportGpuSelfSpread(gpuCold, gpuWarm);
            }
        } finally {
            com.powsybl.openloadflow.ac.solver.gpu.GpuBatchContextCache.forceChunkSizeForTests(null);
            securityAnalysisProvider = new OpenSecurityAnalysisProvider(new SparseMatrixFactory(),
                    new EvenShiloachGraphDecrementalConnectivityFactory<>());
        }
        return new GpuArms(gpuCold, gpuWarm, gpuColdMs, gpuWarmMs);
    }

    private record GpuArms(SecurityAnalysisResult gpuCold, SecurityAnalysisResult gpuWarm, long gpuColdMs,
                           long gpuWarmMs) {
    }
}
