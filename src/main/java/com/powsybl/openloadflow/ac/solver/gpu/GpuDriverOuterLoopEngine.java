/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcOuterLoopContext;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.outerloop.AcActivePowerDistributionOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop;
import com.powsybl.openloadflow.util.gpu.GpuTrace;
import org.apache.commons.lang3.tuple.Pair;

import java.util.List;

/**
 * The engine of a GPU load flow — a plain load flow with {@link GpuNewtonRaphsonFactory} and the base
 * load flow of the GPU security analysis alike: the device driver solves it as a batch of one and owns
 * the outer loops, in-kernel at their OLF positions. This engine suppresses those loops from OLF's
 * standard outer-loop cycle, or they would run twice.
 *
 * <p>Dropping them HERE rather than from {@code parameters.getOuterLoops()} is deliberate: the GPU
 * security analysis derives its own batch configuration and qualification gates from that shared
 * list, and editing it stopped the run batching at all (share 100% -> 0%).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuDriverOuterLoopEngine extends AcloadFlowEngine {

    /** The solver of the current engine run (set by {@link #createSolver}): the loops defer to the
     *  device only while it is the GPU driver AND its last solve ran on the device. */
    private com.powsybl.openloadflow.ac.solver.AcSolver currentSolver;

    public GpuDriverOuterLoopEngine(AcLoadFlowContext context) {
        super(context);
    }

    @Override
    protected com.powsybl.openloadflow.ac.solver.AcSolver createSolver(
            com.powsybl.openloadflow.ac.solver.AcSolverFactory solverFactory, AcLoadFlowContext context) {
        currentSolver = super.createSolver(solverFactory, context);
        if (currentSolver instanceof com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson gpu) {
            gpu.setDriverEngineActive(true);                 // this engine owns the loops it proxies
        }
        return currentSolver;
    }

    /** The device drove the last solve of this run: the proxied loops' decisions are already made. */
    boolean deviceDroveLastSolve() {
        return currentSolver instanceof com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson gpu && gpu.lastRunOnDevice();
    }

    /** What the device's distributed-slack passes placed in the last solve (pu), NaN when nothing / unknown. */
    double lastDeviceDistributedActivePower() {
        return currentSolver instanceof com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson gpu
                ? gpu.lastDistributedActivePower() : Double.NaN;
    }

    /** True for loops the device driver runs itself, in-kernel. These must not also run in OLF's engine, or they
     *  apply twice. A loop the device cannot own fails the device solve, which goes to OLF's CPU Newton-Raphson. */
    static boolean drivenOnDevice(AcOuterLoop outerLoop) {
        if (outerLoop instanceof DistributedSlackOuterLoop || outerLoop instanceof ReactiveLimitsOuterLoop) {
            return true;
        }
        return switch (outerLoop.getName()) {
            case "FreezingHvdcACEmulation", "AcHvdcAcEmulationLimits", "VoltageMonitoring", "SecondaryVoltageControl",
                 "PhaseControl", "IncrementalPhaseControl",
                 "TransformerVoltageControl", "SimpleTransformerVoltageControl", "IncrementalTransformerVoltageControl",
                 "IncrementalTransformerReactivePowerControl",
                 "ShuntVoltageControl", "IncrementalShuntVoltageControl" -> true;
            default -> false;
        };
    }

    /**
     * A loop the device driver owns. While the solver runs on the device the driver has already
     * made this loop's decisions inside the solve, so the check is STABLE. The moment a solve FALLS
     * BACK to the CPU Newton-Raphson (a device-broken shape, a rejected extraction) the driver made
     * no decision at all, and OLF's real loop must run exactly as it does without the GPU. Dropping
     * the loop instead left the fallback base without any DistributedSlack/ReactiveLimits (1.3e-4 kV
     * on the zero-impedance member network).
     *
     * <p>OLF initialized the real loop when the engine created it (createOuterLoopsAndContexts), before
     * it was wrapped here — the driver starts from that state too (FreezingHvdcACEmulation's frozen
     * statuses, PhaseControl's enabled controls). It is not initialized again: a second initialize reset
     * its state and re-froze the emulations the run had unfrozen.
     */
    static class DeviceDrivenLoop implements AcOuterLoop {
        protected final GpuDriverOuterLoopEngine engine;
        private final AcOuterLoop real;

        DeviceDrivenLoop(GpuDriverOuterLoopEngine engine, AcOuterLoop real) {
            this.engine = engine;
            this.real = real;
        }

        @Override
        public String getName() {
            return real.getName();
        }

        @Override
        public String getType() {
            return real.getType();
        }

        @Override
        public boolean isNeeded(AcLoadFlowContext context) {
            return real.isNeeded(context);
        }

        @Override
        public void initialize(AcOuterLoopContext context) {
            real.initialize(context);
        }

        @Override
        public com.powsybl.openloadflow.lf.outerloop.OuterLoopResult check(AcOuterLoopContext context,
                                                                          com.powsybl.commons.report.ReportNode reportNode) {
            if (engine.deviceDroveLastSolve()) {
                return new com.powsybl.openloadflow.lf.outerloop.OuterLoopResult(this,
                        com.powsybl.openloadflow.lf.outerloop.OuterLoopStatus.STABLE);
            }
            return real.check(context, reportNode);
        }

        @Override
        public void cleanup(AcOuterLoopContext context) {
            real.cleanup(context);
        }

        @Override
        public boolean canFixUnrealisticState() {
            return real.canFixUnrealisticState();
        }
    }

    /**
     * A device-driven DISTRIBUTED-SLACK loop. OLF's engine reads the active power a run distributed from the
     * loops that are {@link AcActivePowerDistributionOuterLoop}s — a plain {@link DeviceDrivenLoop} is not one,
     * so every GPU-driven load flow reported 0 (the base case of a GPU security analysis: -60.29 MW on the
     * CPU, 0.0 on the GPU, IEEE 14). While the device drives, the amount is what its passes placed: each
     * device solve re-derives the whole distribution from the run's anchor, so the last solve that
     * distributed carries it, and one that distributed nothing leaves it standing.
     *
     * <p>The device's amount is read whoever drove the last solve, and the two accounts ADD. A solve can
     * place its distribution in-kernel and only then fall back (a later pass meets a shape the device cannot
     * solve), so {@code deviceDroveLastSolve()} is false while the device's passes are ALREADY in the network
     * state; OLF's real loop, taking over on that balanced state, correctly books nothing, because the
     * mismatch it now sees is far inside the {@code slackBusPMaxMismatch} deadband. Reading the device's
     * amount only while it was still driving therefore reported rte6515's busbar-window base as
     * gpu=0.000000000 MW against cpu=14.359916553 — the largest single disagreement in that comparison —
     * with the device holding 14.294 MW at the very check that reported zero (OLF_GPU_DS_ACCOUNT=1). Where
     * the real loop does shift power on top, it books only its own increment, so the sum is the run's total
     * and a pure-CPU run adds nothing (the device's amount is NaN until a pass fires).
     */
    static final class DeviceDrivenDistributionLoop extends DeviceDrivenLoop implements AcActivePowerDistributionOuterLoop {
        private final AcActivePowerDistributionOuterLoop realDistribution;
        private double deviceDistributed;
        private boolean fellBack;

        DeviceDrivenDistributionLoop(GpuDriverOuterLoopEngine engine, AcOuterLoop real,
                                     AcActivePowerDistributionOuterLoop realDistribution) {
            super(engine, real);
            this.realDistribution = realDistribution;
        }

        @Override
        public com.powsybl.openloadflow.lf.outerloop.OuterLoopResult check(AcOuterLoopContext context,
                                                                          com.powsybl.commons.report.ReportNode reportNode) {
            // Take the device's amount whoever drove the last solve: what its passes placed is in the state
            // whether or not the solve that placed it went on to fall back.
            double d = engine.lastDeviceDistributedActivePower();
            if (!Double.isNaN(d)) {
                deviceDistributed = d;
            }
            if (!engine.deviceDroveLastSolve()) {
                fellBack = true;
                if (GpuTrace.DS_ACCOUNT.isOn()) {
                    // The mismatch OLF's own loop is about to gate on: a loop that books nothing either saw
                    // nothing to distribute (the state was already balanced before this run) or was asked on
                    // the wrong context. The two look identical from the account alone.
                    System.err.printf("GPU_DS_ACCOUNT check FELL BACK run=%08x mismatch=%.9f booked=%.9f deviceHas=%s%n",
                            System.identityHashCode(this), realDistribution.getSlackBusActivePowerMismatch(context),
                            realDistribution.getDistributedActivePower(context),
                            Double.toString(engine.lastDeviceDistributedActivePower()));
                }
            }
            return super.check(context, reportNode);
        }

        @Override
        public double getDistributedActivePower(AcOuterLoopContext context, int numSC) {
            // the device distributes over the main synchronous component only
            int mainSc = context.getNetwork().getSynchronousNetworks().getFirst().getNumSC();
            double onDevice = numSC == mainSc ? deviceDistributed : 0.0;
            double onCpu = fellBack ? realDistribution.getDistributedActivePower(context, numSC) : 0.0;
            if (GpuTrace.DS_ACCOUNT.isOn()) {
                System.err.printf("GPU_DS_ACCOUNT ask numSC=%d mainSC=%d fellBack=%b device=%.9f cpu=%.9f%n",
                        numSC, mainSc, fellBack, onDevice, onCpu);
            }
            return onDevice + onCpu;
        }

        @Override
        public double getDistributedActivePower(AcOuterLoopContext context) {
            double onCpu = fellBack ? realDistribution.getDistributedActivePower(context) : 0.0;
            if (GpuTrace.DS_ACCOUNT.isOn()) {
                // Which of the two accounts holds the run's distribution, asked of the loop itself: a base
                // reported as 0.0 is either a device that recorded nothing or a fallback whose real loop was
                // never asked. Guessing between them cost a wrong fix once.
                System.err.printf("GPU_DS_ACCOUNT fellBack=%b device=%.9f cpu=%.9f loop=%s%n",
                        fellBack, deviceDistributed, onCpu, getName());
            }
            return deviceDistributed + onCpu;
        }

        @Override
        public double getSlackBusActivePowerMismatch(AcOuterLoopContext context) {
            return realDistribution.getSlackBusActivePowerMismatch(context);
        }
    }

    @Override
    protected List<Pair<AcOuterLoop, AcOuterLoopContext>> engineOuterLoopsAndContexts(AcLoadFlowContext context) {
        var all = super.engineOuterLoopsAndContexts(context);
        List<Pair<AcOuterLoop, AcOuterLoopContext>> out = new java.util.ArrayList<>();
        for (var p : all) {
            AcOuterLoop loop = p.getLeft();
            if (!drivenOnDevice(loop)) {
                out.add(p);
            } else if (loop instanceof AcActivePowerDistributionOuterLoop distribution) {
                out.add(Pair.of(new DeviceDrivenDistributionLoop(this, loop, distribution), p.getRight()));
            } else {
                out.add(Pair.of(new DeviceDrivenLoop(this, loop), p.getRight()));
            }
        }
        if (GpuTrace.BATCH_TRACE.isOn()) {
            System.err.println("GPU_DRIVER_ENGINE loops=" + all.stream().map(p -> p.getLeft().getName()).toList()
                    + " keptOnHost=" + all.stream().filter(p -> !drivenOnDevice(p.getLeft())).map(p -> p.getLeft().getName()).toList());
        }
        return out;
    }
}
