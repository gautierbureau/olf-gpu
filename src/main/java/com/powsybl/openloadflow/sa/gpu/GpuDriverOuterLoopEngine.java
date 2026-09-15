/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

import com.powsybl.openloadflow.ac.AcLoadFlowContext;
import com.powsybl.openloadflow.ac.AcloadFlowEngine;
import com.powsybl.openloadflow.ac.outerloop.AcOuterLoop;
import com.powsybl.openloadflow.ac.AcOuterLoopContext;
import com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop;
import com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop;
import org.apache.commons.lang3.tuple.Pair;

import java.util.List;

/**
 * The base load flow's engine when the DEVICE driver owns the outer loops.
 *
 * <p>The base LF goes through the batch code as a batch of one. With
 * {@code OLF_GPU_DRIVER_OUTER_LOOPS=1} the driver runs ALL outer loops: DS and RL as in-kernel
 * phases, and the remaining loops (SVC, TVC, Phase, TRPC, Shunt) as host-owned pause points
 * in their correct OLF positions. This engine suppresses those loops from OLF's standard
 * outer-loop cycle, or they would run twice.
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
        // Make the engine's AcLoadFlowContext available to the solver's host-loop pause cycle,
        // so outer-loop contexts share the engine's equation system and Jacobian (not duplicates).
        com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.GPU_ENGINE_CONTEXT.set(context);
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

    /** True for loops the device driver runs itself — in-kernel (DS, RL) or host-paused (SVC,
     *  TVC, Phase, TRPC, Shunt). These must not also run in OLF's engine, or they apply twice. */
    static boolean drivenOnDevice(AcOuterLoop outerLoop) {
        if (outerLoop instanceof DistributedSlackOuterLoop || outerLoop instanceof ReactiveLimitsOuterLoop) {
            return true;  // in-kernel driver phases
        }
        switch (outerLoop.getName()) {                       // in-kernel driver phases with their own tables
            case "FreezingHvdcACEmulation", "AcHvdcAcEmulationLimits", "VoltageMonitoring" -> {
                return true;
            }
            default -> { }
        }
        // Host-paused loops: anything with a LOOP_HOST_* kind
        return com.powsybl.openloadflow.ac.solver.gpu.GpuNewtonRaphson.loopKindFor(outerLoop.getName())
                >= com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver.HOST_LOOP_SVC;
    }

    /**
     * A loop the device driver owns. While the solver runs on the device the driver has already
     * made this loop's decisions inside the solve, so the check is STABLE and OLF's loop is never
     * initialized (FreezingHvdcACEmulation.initialize would freeze every emulation on its own). The
     * moment a solve FALLS BACK to the CPU Newton-Raphson (a device-broken shape, a forced-PV
     * election the kernel does not run, a rejected extraction) the driver made no decision at all,
     * and OLF's real loop must run exactly as it does without the GPU — initialized late, on the
     * state the fallback left. Dropping the loop instead left the fallback base without any
     * DistributedSlack/ReactiveLimits (1.3e-4 kV on the zero-impedance member network).
     */
    static final class DeviceDrivenLoop implements AcOuterLoop {
        private final GpuDriverOuterLoopEngine engine;
        private final AcOuterLoop real;
        private boolean initialized;

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
            // OLF's initialize is part of the loop's contract and the device relies on it having run
            // (PhaseControl.initialize ENABLES the continuous controls the first solve must see as
            // row-mode-3 rows). The one exception is FreezingHvdcACEmulation: its initialize freezes
            // every emulation on the network, which the driver replays itself from the warm state.
            if (!"FreezingHvdcACEmulation".equals(real.getName())) {
                real.initialize(context);
                initialized = true;
            }
        }

        @Override
        public com.powsybl.openloadflow.lf.outerloop.OuterLoopResult check(AcOuterLoopContext context,
                                                                          com.powsybl.commons.report.ReportNode reportNode) {
            if (engine.deviceDroveLastSolve()) {
                return new com.powsybl.openloadflow.lf.outerloop.OuterLoopResult(this,
                        com.powsybl.openloadflow.lf.outerloop.OuterLoopStatus.STABLE);
            }
            if (!initialized) {
                real.initialize(context);
                initialized = true;
            }
            return real.check(context, reportNode);
        }

        @Override
        public void cleanup(AcOuterLoopContext context) {
            if (initialized) {
                real.cleanup(context);
            }
        }

        @Override
        public boolean canFixUnrealisticState() {
            return real.canFixUnrealisticState();
        }
    }

    @Override
    protected List<Pair<AcOuterLoop, AcOuterLoopContext>> engineOuterLoopsAndContexts(AcLoadFlowContext context) {
        var all = super.engineOuterLoopsAndContexts(context);
        List<Pair<AcOuterLoop, AcOuterLoopContext>> out = new java.util.ArrayList<>();
        for (var p : all) {
            out.add(drivenOnDevice(p.getLeft()) ? Pair.of(new DeviceDrivenLoop(this, p.getLeft()), p.getRight()) : p);
        }
        if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
            System.err.println("GPU_DRIVER_ENGINE loops=" + all.stream().map(p -> p.getLeft().getName()).toList()
                    + " keptOnHost=" + all.stream().filter(p -> !drivenOnDevice(p.getLeft())).map(p -> p.getLeft().getName()).toList());
        }
        return out;
    }
}
