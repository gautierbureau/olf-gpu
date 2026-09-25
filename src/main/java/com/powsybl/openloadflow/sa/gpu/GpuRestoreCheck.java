/*
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.network.LfArea;
import com.powsybl.openloadflow.network.LfBranch;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfLoad;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfShunt;
import com.powsybl.openloadflow.network.LfSynchronousNetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * {@code OLF_GPU_RESTORE_CHECK=1}: after every base-state restore of a security analysis, compare a fingerprint
 * of everything a {@code NetworkState} restores — per bus, generator, load, shunt, branch, HVDC, area and
 * synchronous network — with the fingerprint a FULL restore produced, and report the first field that differs.
 *
 * <p>The reference is taken after the first restore, which is always a full one: there is no hook between the
 * base state being saved and the first contingency being applied, and "identical to what a full restore gives"
 * is exactly what a cheaper restore must guarantee. A restore that misses a mutation leaves the next
 * contingency starting from a different network, which violation counts need not reveal.
 *
 * <p>Diagnostic only: it walks the whole network twice per contingency.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
final class GpuRestoreCheck {

    static final boolean ENABLED = System.getenv("OLF_GPU_RESTORE_CHECK") != null;

    private double[] reference;
    private LfNetwork referenceNetwork;
    private long checked;
    private long mismatching;
    private final List<String> samples = new ArrayList<>();

    /** Record the fingerprint of a network a full restore just produced. */
    void setReference(LfNetwork network) {
        reference = fingerprint(network, null);
        referenceNetwork = network;
    }

    boolean hasReference(LfNetwork network) {
        return reference != null && referenceNetwork == network;
    }

    /** Compare the network, just restored after {@code contingencyId}, with the reference. */
    void check(LfNetwork network, String contingencyId) {
        double[] now = fingerprint(network, null);
        checked++;
        int first = -1;
        int count = 0;
        int n = Math.min(now.length, reference.length);
        for (int i = 0; i < n; i++) {
            if (Double.doubleToLongBits(now[i]) != Double.doubleToLongBits(reference[i])) {
                count++;
                if (first < 0) {
                    first = i;
                }
            }
        }
        if (now.length != reference.length) {
            count++;
            first = first < 0 ? n : first;
        }
        if (count > 0) {
            mismatching++;
            if (samples.size() < 20) {
                List<String> labels = new ArrayList<>();
                fingerprint(network, labels);
                String label = first < labels.size() ? labels.get(first) : "(length " + now.length + " vs " + reference.length + ")";
                samples.add(String.format("after %s: %d field(s) differ, first %s: restored %s, full restore %s",
                        contingencyId, count, label,
                        first < now.length ? Double.toString(now[first]) : "-",
                        first < reference.length ? Double.toString(reference[first]) : "-"));
            }
        }
    }

    void print() {
        System.err.printf("RESTORE_CHECK restores compared=%d identical=%d differing=%d%n",
                checked, checked - mismatching, mismatching);
        samples.forEach(s -> System.err.println("RESTORE_CHECK   " + s));
    }

    /** Every restorable field, in a fixed order; with {@code labels} non-null, also names each one. */
    private static double[] fingerprint(LfNetwork network, List<String> labels) {
        Fp fp = new Fp(labels);
        for (LfBus b : network.getBuses()) {
            String id = "bus " + b.getId();
            fp.add(id + " disabled", b.isDisabled());
            fp.add(id + " v", b.getV());
            fp.add(id + " angle", b.getAngle());
            fp.add(id + " generatorVoltageControlEnabled", b.isGeneratorVoltageControlEnabled());
            fp.add(id + " generatorReactivePowerControlEnabled", b.isGeneratorReactivePowerControlEnabled());
            fp.add(id + " generationTargetQ", b.getGenerationTargetQ());
            fp.add(id + " generationTargetQFrozen", b.isGenerationTargetQFrozen());
            fp.add(id + " qLimitType", b.getQLimitType().map(Enum::ordinal).orElse(-1));
            shunt(fp, id + " shunt", b.getShunt());
            shunt(fp, id + " controllerShunt", b.getControllerShunt());
            shunt(fp, id + " svcShunt", b.getSvcShunt());
            b.getControllerShunt().ifPresent(s -> fp.add(id + " controllerShunt voltageControlEnabled", s.isVoltageControlEnabled()));
            for (LfGenerator g : b.getGenerators()) {
                String gid = "generator " + g.getId();
                fp.add(gid + " targetP", g.getTargetP());
                fp.add(gid + " initialTargetP", g.getInitialTargetP());
                fp.add(gid + " targetQ", g.getTargetQ());
                fp.add(gid + " participating", g.isParticipating());
                fp.add(gid + " disabled", g.isDisabled());
                fp.add(gid + " controlType", g.getGeneratorControlType().ordinal());
            }
            for (LfLoad l : b.getLoads()) {
                String lid = "load " + l.getId();
                fp.add(lid + " targetP", l.getTargetP());
                fp.add(lid + " targetQ", l.getTargetQ());
                fp.add(lid + " absVariableTargetP", l.getAbsVariableTargetP());
                for (Map.Entry<String, Boolean> e : new java.util.TreeMap<>(l.getOriginalLoadsDisablingStatus()).entrySet()) {
                    fp.add(lid + " original " + e.getKey() + " disabled", e.getValue());
                }
            }
        }
        for (LfBranch br : network.getBranches()) {
            String id = "branch " + br.getId();
            fp.add(id + " disabled", br.isDisabled());
            fp.add(id + " connectedSide1", br.isConnectedSide1());
            fp.add(id + " connectedSide2", br.isConnectedSide2());
            fp.add(id + " a1", br.getPiModel().getA1());
            fp.add(id + " r1", br.getPiModel().getR1());
            if (br.getPiModel() instanceof com.powsybl.openloadflow.network.PiModelArray) {   // as BranchState saves it
                fp.add(id + " tapPosition", br.getPiModel().getTapPosition());
            }
            fp.add(id + " phaseControlEnabled", br.isPhaseControlEnabled());
            fp.add(id + " voltageControlEnabled", br.isVoltageControlEnabled());
        }
        for (LfHvdc h : network.getHvdcs()) {
            String id = "hvdc " + h.getId();
            fp.add(id + " disabled", h.isDisabled());
            fp.add(id + " acEmulation", h.isAcEmulation());
            fp.add(id + " acEmulationStatus", h.getAcEmulationControl() == null || h.getAcEmulationControl().getAcEmulationStatus() == null
                    ? -1 : h.getAcEmulationControl().getAcEmulationStatus().ordinal());
            fp.add(id + " converter1 targetP", h.getConverterStation1().getTargetP());
            fp.add(id + " converter2 targetP", h.getConverterStation2().getTargetP());
        }
        for (LfArea a : network.getAreas()) {
            fp.add("area " + a.getId() + " interchangeTarget", a.getInterchangeTarget());
        }
        for (LfSynchronousNetwork sc : network.getSynchronousNetworks()) {
            String id = "synchronous network " + sc.getNumSC();
            fp.add(id + " excludedSlackBuses", sc.getExcludedSlackBuses().size());
            sc.getExcludedSlackBuses().stream().mapToInt(LfBus::getNum).sorted()
                    .forEach(num -> fp.add(id + " excludedSlackBus", num));
        }
        return fp.values();
    }

    private static void shunt(Fp fp, String id, Optional<LfShunt> shunt) {
        shunt.ifPresent(s -> {
            fp.add(id + " b", s.getB());
            fp.add(id + " g", s.getG());
        });
    }

    /** An append-only double list, with the optional parallel label list. */
    private static final class Fp {
        private double[] v = new double[1024];
        private int size;
        private final List<String> labels;

        private Fp(List<String> labels) {
            this.labels = labels;
        }

        void add(String label, double x) {
            if (size == v.length) {
                v = java.util.Arrays.copyOf(v, 2 * size);
            }
            v[size++] = x;
            if (labels != null) {
                labels.add(label);
            }
        }

        void add(String label, boolean x) {
            add(label, x ? 1.0 : 0.0);
        }

        double[] values() {
            return java.util.Arrays.copyOf(v, size);
        }
    }

}
