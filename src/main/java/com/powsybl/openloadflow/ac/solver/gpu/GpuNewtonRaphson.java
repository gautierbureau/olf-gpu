/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac.solver.gpu;

import com.powsybl.commons.PowsyblException;
import com.powsybl.commons.report.ReportNode;
import com.powsybl.openloadflow.ac.equations.AcEquationType;
import com.powsybl.openloadflow.ac.equations.AcVariableType;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivation;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcActivationExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcData;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataCache;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchHvdcAcEmulation;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchIncrementalTvc;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchOuterLoopTables;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchVoltageMonitoring;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchRlCurves;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcDataExtractor;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcNewtonSolver;
import com.powsybl.openloadflow.ac.equations.vector.gpu.GpuAcStructures;
import com.powsybl.openloadflow.ac.solver.AbstractAcSolver;
import com.powsybl.openloadflow.ac.solver.AcSolverResult;
import com.powsybl.openloadflow.ac.solver.AcSolverStatus;
import com.powsybl.openloadflow.ac.solver.AcSolverUtil;
import com.powsybl.openloadflow.ac.solver.DefaultNewtonRaphsonStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.NewtonRaphson;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonParameters;
import com.powsybl.openloadflow.ac.solver.NewtonRaphsonStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.PerEquationTypeStoppingCriteria;
import com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchContextCache;
import com.powsybl.openloadflow.equations.EquationSystem;
import com.powsybl.openloadflow.equations.EquationSystemIndex;
import com.powsybl.openloadflow.equations.EquationSystemIndexListener;
import com.powsybl.openloadflow.equations.EquationVector;
import com.powsybl.openloadflow.equations.JacobianMatrix;
import com.powsybl.openloadflow.equations.TargetVector;
import com.powsybl.openloadflow.network.LfBus;
import com.powsybl.openloadflow.network.LfGenerator;
import com.powsybl.openloadflow.network.LfHvdc;
import com.powsybl.openloadflow.network.LfNetwork;
import com.powsybl.openloadflow.network.LfNetworkListener;
import com.powsybl.openloadflow.network.LfSynchronousNetwork;
import com.powsybl.openloadflow.network.util.VoltageInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Newton-Raphson run ENTIRELY on the GPU behind one JNI call ({@link GpuAcNewtonSolver}):
 * the fixed superset-pattern CSR Jacobian and the mismatch are filled on the device by
 * the generated fused kernels (closed-branch / open-branch / shunt / HVDC AC emulation),
 * cuDSS analyzes once and refactorizes/solves per iteration, the state update is a
 * kernel.
 *
 * <p>Invalidation is EVENT-DRIVEN, mirroring {@link JacobianMatrix}'s pattern (the same
 * {@link EquationSystemIndexListener} / {@link LfNetworkListener} interfaces, with this
 * consumer's own granularity): structural events (topology, branch parameters) force a
 * full re-extract + device context rebuild; activation events (PV→PQ switching, slack
 * change) only refresh the row-mode mask and target mapping; with no events, a rerun
 * goes straight to the device solve — targets and state are re-read every solve. Both
 * interfaces are implemented DIRECTLY (not via the no-op adapter) so any event added to
 * OLF later forces a compile-time classification here, keeping invalidation conservative.
 *
 * <p>Scope: plain AC power flow. {@link GpuAcDataExtractor} validates the equation
 * system on every structural rebuild and throws on anything not covered; an unsupported
 * equation type ACTIVATED between rebuilds fails loudly in the activation walk. The
 * device loop iterates to an infinity-norm mismatch of {@value GPU_TOLERANCE} (stricter
 * than the usual stopping criteria); the final CONVERGED/MAX_ITERATION_REACHED status is
 * then decided by the CONFIGURED stopping criteria, evaluated on the CPU from the
 * written-back state, exactly as for the CPU Newton-Raphson. State-vector scaling
 * options are not applied (no line search on the device).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuNewtonRaphson extends AbstractAcSolver {

    private static final Logger LOGGER = LoggerFactory.getLogger(GpuNewtonRaphson.class);

    private static final Cleaner CLEANER = Cleaner.create();

    public static final double GPU_TOLERANCE = 1e-8;

    /** The engine's AcLoadFlowContext, set by GpuDriverOuterLoopEngine before calling solver.run()
     *  so the solver's host-loop pause cycle can create outer-loop contexts that share the engine's
     *  equation system and Jacobian. Thread-local because the engine and solver run on the same thread. */
    public static final ThreadLocal<com.powsybl.openloadflow.ac.AcLoadFlowContext> GPU_ENGINE_CONTEXT = new ThreadLocal<>();

    /** Did this thread's LAST solve run on the device driver (true) or fall back to the CPU
     *  Newton-Raphson (false)? {@code GpuDriverOuterLoopEngine} reads it to decide whether OLF's own
     *  loop must run a check the driver would otherwise have owned. */
    private static final ThreadLocal<Boolean> LAST_RUN_ON_DEVICE = ThreadLocal.withInitial(() -> false);

    public static boolean lastRunOnDevice() {
        return LAST_RUN_ON_DEVICE.get();
    }

    // Diagnostics for the invalidation taxonomy: how the cached native context is reconciled across
    // solves — STRUCTURE rebuilds (pattern + cuDSS re-analysis) vs cheap VALUES re-uploads (a transformer
    // tap / shunt-susceptance move keeps the pattern, only the element packs change). Process-wide totals.
    private static final java.util.concurrent.atomic.AtomicLong CONTEXT_REBUILDS = new java.util.concurrent.atomic.AtomicLong();
    private static final java.util.concurrent.atomic.AtomicLong VALUE_REFRESHES = new java.util.concurrent.atomic.AtomicLong();

    public static long contextRebuildCount() {
        return CONTEXT_REBUILDS.get();
    }

    public static long valueRefreshCount() {
        return VALUE_REFRESHES.get();
    }

    protected final NewtonRaphsonParameters parameters;

    /**
     * The run's outer-loop settings, so the batch-of-one can drive the SAME device loops the batched
     * SA drives. A batch of one is still a batch: nothing about nS == 1 justifies restricting it.
     *
     * <p>{@code driverOuterLoops} off keeps the historical behaviour — the solve neutralises the
     * driver ({@code slackDistTol} 1e300, {@code skipRlPhase} 1) and OLF's Java loops run around it
     * on the CPU.
     */
    private final boolean driverOuterLoops;
    private final double driverSlackDistTol;
    private final double driverReactiveLimitTol;
    private final int driverMaxOuterIterations;
    private final int driverMaxPqPvSwitch;

    /** The OLF outer loops that the driver will pause for — empty when driverOuterLoops is off. */
    private final List<com.powsybl.openloadflow.ac.outerloop.AcOuterLoop> hostOuterLoops;
    private final com.powsybl.openloadflow.ac.AcLoadFlowParameters acLoadFlowParameters;

    /** Map an OLF outer loop name to the native {@code LOOP_HOST_*} kind, or -1 if it is an
     *  in-kernel driver phase (DS, RL, HVDC, Monitoring) or unknown. */
    public static int loopKindFor(String name) {
        return switch (name) {
            case "SecondaryVoltageControl" -> GpuAcNewtonSolver.HOST_LOOP_SVC;
            case "PhaseControl", "IncrementalPhaseControl" -> GpuAcNewtonSolver.HOST_LOOP_PHASE;
            case "TransformerVoltageControl", "SimpleTransformerVoltageControl",
                 "IncrementalTransformerVoltageControl" -> GpuAcNewtonSolver.HOST_LOOP_TVC;
            case "IncrementalTransformerReactivePowerControl" -> GpuAcNewtonSolver.HOST_LOOP_TRPC;
            case "ShuntVoltageControl", "IncrementalShuntVoltageControl" -> GpuAcNewtonSolver.HOST_LOOP_SHUNT;
            default -> -1;
        };
    }

    /** Build the bitmask for {@link GpuAcNewtonSolver#setBatchHostLoop} from the configured outer loops. */
    private int buildHostLoopMask() {
        int mask = 0;
        for (var loop : hostOuterLoops) {
            int kind = loopKindFor(loop.getName());
            if (kind >= GpuAcNewtonSolver.HOST_LOOP_SVC) {
                mask |= (1 << (kind - GpuAcNewtonSolver.HOST_LOOP_SVC));
            }
        }
        return mask;
    }

    /** The driver's per-controller reactive-limit state after the last solve ({@code (limitType & 3) |
     *  (switchCount << 2)}), read back so the PV↔PQ decisions can be given to the network. */
    private int[] rlStateOut;

    /**
     * Give the driver's outer-loop decisions back to the network — what OLF's own loops do as a side
     * effect of running: without it OLF's world still holds the pre-loop targets and controls, the
     * converged state is measured against them and the run is declared MAX_ITERATION_REACHED
     * (measured 170 BUS_TARGET_P rows, 14 BUS_TARGET_V rows and 2 BUS_TARGET_Q rows over 1e-4 on
     * rte6515 where the same solve with the driver off leaves none), and everything downstream —
     * the batched SA's base targets, the CPU-routed contingencies — starts from the wrong network.
     */
    private void writeBackDriverMutations(GpuAcData data) {
        long handle = context.handle.get();
        double[] deviceTargets = GpuAcNewtonSolver.readBatchTargets(handle);
        if (hvdcCtls != null && !hvdcCtls.isEmpty()) {
            GpuBatchHvdcAcEmulation.writeBack(network, hvdcCtls, GpuAcNewtonSolver.readBatchHvdcStatus(handle),
                    data.getTotalElementCount());
        }
        if (monCtls != null && !monCtls.isEmpty()) {
            GpuBatchVoltageMonitoring.writeBack(network, monCtls, GpuAcNewtonSolver.readBatchMonitoringTargets(handle),
                    data.rlBusNum());
        }
        writeBackDistributedSlack(deviceTargets,
                GpuAcNewtonSolver.readBatchDsGenCum(handle), GpuAcNewtonSolver.readBatchDsRetain(handle), data);
        writeBackReactiveLimits(data, deviceTargets);
        if (tvcOwned && tvcCtls != null) {
            GpuBatchIncrementalTvc.writeBackTaps(network, tvcCtls, GpuAcNewtonSolver.readBatchTvcTaps(handle));
        }
    }

    /** DistributedSlack: OLF's ActivePowerDistribution sets every participating generator's targetP —
     *  per GENERATOR (each saturates at its own limits), the slack bus's own included. The device's
     *  water-fill keeps the same per-generator cumulative deltas ({@code genCum}) for the buses that
     *  have a row, and the slack's share as the retained mismatch ({@code dsRetain}); the slack bus has
     *  no row, so its generators split the retained share by factor, which is exactly OLF's own split
     *  while none of them saturates. */
    private void writeBackDistributedSlack(double[] deviceTargets, double[] genCum, double[] retain, GpuAcData data) {
        if (deviceTargets == null || data == null || data.dsPhiRow().length == 0) {
            return;
        }
        java.util.Map<String, LfGenerator> genById = new java.util.HashMap<>();
        for (LfBus bus : network.getBuses()) {
            for (LfGenerator g : bus.getGenerators()) {
                genById.put(g.getId(), g);
            }
        }
        if (genCum != null && data.dgGenId() != null && genCum.length >= data.dgGenId().length) {
            for (int g = 0; g < data.dgGenId().length; g++) {
                if (genCum[g] != 0) {
                    LfGenerator gen = genById.get(data.dgGenId()[g]);
                    if (gen != null) {
                        gen.setTargetP(gen.getTargetP() + genCum[g]);
                    }
                }
            }
        } else {
            for (int i = 0; i < data.dsPhiRow().length; i++) {        // bus-level fallback: split by factor
                int row = data.dsPhiRow()[i];
                if (row < 0 || row >= deviceTargets.length) {
                    continue;
                }
                double delta = deviceTargets[row] - data.dsBaseTargetP()[i];
                LfBus bus = data.dsBusNum() != null && i < data.dsBusNum().length ? network.getBus(data.dsBusNum()[i]) : null;
                if (bus != null && delta != 0) {
                    splitByFactor(bus, delta);
                }
            }
        }
        if (retain != null && retain.length > 0 && retain[0] != 0 && data.dsSlackPhiRow() >= 0) {
            for (LfBus bus : network.getBuses()) {
                var v = equationSystem.getVariableSet().getVariable(bus.getNum(), AcVariableType.BUS_PHI);
                if (v != null && v.getRow() == data.dsSlackPhiRow()) {
                    splitByFactor(bus, retain[0]);
                    break;
                }
            }
        }
    }

    private static void splitByFactor(LfBus bus, double delta) {
        double factorSum = bus.getGenerators().stream()
                .filter(LfGenerator::isParticipating)
                .mapToDouble(LfGenerator::getParticipationFactor)
                .sum();
        if (factorSum <= 0) {
            return;
        }
        for (LfGenerator g : bus.getGenerators()) {
            if (g.isParticipating()) {
                g.setTargetP(g.getTargetP() + delta * g.getParticipationFactor() / factorSum);
            }
        }
    }

    /** ReactiveLimits: the device pins a controller at its limit (row mode V→Q, target the limit) where
     *  OLF's loop freezes the bus's generation Q at that limit and disables its voltage control — the
     *  same two calls, so OLF's equation system swaps BUS_TARGET_V for BUS_TARGET_Q on its own and the
     *  target vector picks the limit up. A pin the device released (a PQ→PV switch-back) is undone the
     *  way OLF undoes it. Realistic-V limits are not modeled on the device and never appear here. */
    private void writeBackReactiveLimits(GpuAcData data, double[] deviceTargets) {
        if (rlStateOut == null || data == null || data.rlBusNum() == null) {
            return;
        }
        int nrl = Math.min(rlStateOut.length, data.rlBusNum().length);
        for (int i = 0; i < nrl; i++) {
            int limitType = rlStateOut[i] & 3;
            LfBus bus = network.getBus(data.rlBusNum()[i]);
            if (bus == null) {
                continue;
            }
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null && (limitType != 0 || (rlStateOut[i] >> 2) != 0)) {
                System.err.println("WB_RL bus=" + bus.getId() + " limitType=" + limitType + " switchCount=" + (rlStateOut[i] >> 2)
                        + " vcEnabled=" + bus.isGeneratorVoltageControlEnabled() + " qLimitType=" + bus.getQLimitType().map(Object::toString).orElse("-")
                        + " remote=" + data.rlRemote()[i]);
            }
            if (limitType != 0 && bus.isGeneratorVoltageControlEnabled()) {
                // The limit the device pinned at — its row target is limit − loadQ, evaluated at the LIVE
                // voltage for an SVC (OLF freezes at bus.getMinQ()/getMaxQ() read at the check, the same
                // live value); the packed base-V aggregate is 1.3e-4 pu off on rte6515's VL-6226_0.
                int vRow = data.rlVrow()[i];
                double qLimit = deviceTargets != null && vRow >= 0 && vRow < deviceTargets.length
                        ? deviceTargets[vRow] + data.rlLoadQ()[i]
                        : (limitType == 1 ? data.rlMinQ()[i] : data.rlMaxQ()[i]);
                bus.freezeGenerationTargetQAndDisableGeneratorVoltageControl(qLimit);
                bus.setQLimitType(limitType == 1 ? LfBus.QLimitType.MIN_Q : LfBus.QLimitType.MAX_Q);
            } else if (limitType == 0 && !bus.isGeneratorVoltageControlEnabled() && bus.getQLimitType().isPresent()
                    && bus.getQLimitType().get() != LfBus.QLimitType.MIN_REALISTIC_V
                    && bus.getQLimitType().get() != LfBus.QLimitType.MAX_REALISTIC_V) {
                bus.setGeneratorVoltageControlEnabledAndRecomputeTargetQ(true);
                bus.setQLimitType(null);
            }
        }
    }

    /** Read the bands out of the run's OWN configured loops, so the device decides at OLF's threshold. */
    static double[] driverSettings(com.powsybl.openloadflow.ac.AcLoadFlowParameters p) {
        double slackTol = Double.MAX_VALUE;
        double rlTol = 0;
        int maxSwitch = 0;
        for (var ol : p.getOuterLoops()) {
            if (ol instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop ds) {
                slackTol = ds.getSlackBusPMaxMismatch() / com.powsybl.openloadflow.util.PerUnit.SB;
            } else if (ol instanceof com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop rl) {
                // Same derivation GpuSaRunScope uses for the batched SA: UNIFORM_CRITERIA (the default)
                // ties the RL band to convEpsPerEq, PER_EQUATION_TYPE_CRITERIA to
                // maxReactivePowerMismatch/SB. Taking maxReactivePowerMismatch unconditionally gives the
                // device a different switch threshold from OLF's and it switches far too eagerly.
                rlTol = p.getAcSolverParameters() instanceof NewtonRaphsonParameters nrp
                        && nrp.getStoppingCriteria() instanceof DefaultNewtonRaphsonStoppingCriteria dc
                        ? dc.getConvEpsPerEq()
                        : rl.getMaxReactivePowerMismatch() / com.powsybl.openloadflow.util.PerUnit.SB;
                maxSwitch = rl.getMaxPqPvSwitch();
            }
        }
        return new double[] {slackTol, rlTol, p.getMaxOuterLoopIterations(), maxSwitch};
    }

    private static double deviceConvEpsPerEq(NewtonRaphsonParameters parameters) {
        if (parameters.getStoppingCriteria() instanceof DefaultNewtonRaphsonStoppingCriteria criteria) {
            return criteria.getConvEpsPerEq();
        }
        return 0.0;
    }

    /** System property: fall back to the CPU Newton-Raphson when the GPU path rejects the
     *  network or fails at runtime (default true). Set to false to fail loudly instead. */
    public static final String CPU_FALLBACK_PROPERTY = "olf.gpu.cpuFallback";

    /**
     * Number of solves that bailed out of the GPU path since JVM start. Tests assert on it because a
     * silent fallback is INVISIBLE otherwise: results stay CPU-identical (that is the whole point of the
     * contract), the warning is logged through a logger the test config does not attach an appender to,
     * and {@link #CPU_FALLBACK_PROPERTY}=false only covers throws that reach the catch below. A GPU test
     * that does not check this counter can pass while never touching the GPU at all — that is exactly how
     * the BRANCH_ALPHA1 packing looked "green but uncovered" until the counter existed.
     */
    public static final java.util.concurrent.atomic.AtomicLong CPU_FALLBACK_COUNT =
            new java.util.concurrent.atomic.AtomicLong();

    private final GpuNativeContext context = new GpuNativeContext();

    private final GpuSolverInvalidation invalidation = new GpuSolverInvalidation();

    private GpuAcData cachedData;

    private GpuAcActivation activation;

    // per-row variable kind (0=BUS_PHI, 1=BUS_V, 2=other) for the MAX_VOLTAGE_CHANGE device step
    // clamp; structural, recomputed whenever the activation/structure is rebuilt
    private int[] varType;

    // PER_EQUATION_TYPE row thresholds need (re)installing on the context: after a context rebuild
    // and after any activation change (the thresholds follow the variables at each row)
    private boolean criteriaDirty = true;

    /** The shared outer-loop tables ({@link GpuBatchOuterLoopTables}) follow the extraction. */
    private boolean tablesDirty = true;

    // The driver phases with their own per-run tables (HVDC AC emulation, MonitoringVoltage, in-kernel
    // incremental TVC): built at every solve from the network as it stands, written back after it.
    private List<GpuBatchHvdcAcEmulation.Ctl> hvdcCtls;
    private List<GpuBatchVoltageMonitoring.Ctl> monCtls;
    private List<GpuBatchIncrementalTvc.Controller> tvcCtls;
    private boolean tvcOwned;

    private com.powsybl.openloadflow.ac.outerloop.AcOuterLoop driverLoop(String name) {
        if (acLoadFlowParameters == null) {
            return null;
        }
        for (var ol : acLoadFlowParameters.getOuterLoops()) {
            if (name.equals(ol.getName())) {
                return ol;
            }
        }
        return null;
    }

    private com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop driverDsLoop() {
        if (acLoadFlowParameters == null) {
            return null;
        }
        for (var ol : acLoadFlowParameters.getOuterLoops()) {
            if (ol instanceof com.powsybl.openloadflow.ac.outerloop.DistributedSlackOuterLoop ds) {
                return ds;
            }
        }
        return null;
    }

    private com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop driverRlLoop() {
        if (acLoadFlowParameters == null) {
            return null;
        }
        for (var ol : acLoadFlowParameters.getOuterLoops()) {
            if (ol instanceof com.powsybl.openloadflow.ac.outerloop.ReactiveLimitsOuterLoop rl) {
                return rl;
            }
        }
        return null;
    }

    /** The run's own DistributedSlack settings (balance type, active limits) — never defaults. */
    private com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams driverDsParams() {
        var ds = driverDsLoop();
        return ds == null
                ? com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams.DEFAULT
                : new com.powsybl.openloadflow.ac.equations.vector.gpu.GpuDsParams(
                        ds.getActivePowerDistribution().getBalanceType(), ds.getActivePowerDistribution().isUseActiveLimits());
    }

    /** Install (or clear) the PER_EQUATION_TYPE per-row stopping thresholds on the batch context,
     *  from the run's actual criteria object — the device then stops on the SAME decision as OLF's
     *  PerEquationTypeStoppingCriteria (every row's |F| under its type threshold). */
    private void installRowCriteria(int n) {
        if (parameters.getStoppingCriteria() instanceof PerEquationTypeStoppingCriteria c) {
            double[][] thr = GpuAcActivationExtractor.extractRowCriteria(equationSystem, n,
                    c.getMaxActivePowerMismatch(), c.getMaxReactivePowerMismatch(), c.getMaxVoltageMismatch(),
                    c.getMaxDefaultAngleMismatch(), c.getMaxDefaultRatioMismatch(),
                    c.getMaxDefaultSusceptanceMismatch(), c.getConvEpsPerEq());
            GpuAcNewtonSolver.setBatchRowCriteria(context.handle.get(), thr[0], thr[1]);
        } else {
            GpuAcNewtonSolver.setBatchRowCriteria(context.handle.get(), null, null);
        }
    }

    public GpuNewtonRaphson(LfNetwork network, NewtonRaphsonParameters parameters,
                            EquationSystem<AcVariableType, AcEquationType> equationSystem,
                            JacobianMatrix<AcVariableType, AcEquationType> j,
                            TargetVector<AcVariableType, AcEquationType> targetVector,
                            EquationVector<AcVariableType, AcEquationType> equationVector,
                            boolean detailedReport) {
        this(network, parameters, equationSystem, j, targetVector, equationVector, detailedReport,
                false, Double.MAX_VALUE, 0, 0, 0, List.of(), null);
    }

    @SuppressWarnings("java:S107")
    public GpuNewtonRaphson(LfNetwork network, NewtonRaphsonParameters parameters,
                            EquationSystem<AcVariableType, AcEquationType> equationSystem,
                            JacobianMatrix<AcVariableType, AcEquationType> j,
                            TargetVector<AcVariableType, AcEquationType> targetVector,
                            EquationVector<AcVariableType, AcEquationType> equationVector,
                            boolean detailedReport, boolean driverOuterLoops,
                            double driverSlackDistTol, double driverReactiveLimitTol,
                            int driverMaxOuterIterations, int driverMaxPqPvSwitch,
                            List<com.powsybl.openloadflow.ac.outerloop.AcOuterLoop> hostOuterLoops,
                            com.powsybl.openloadflow.ac.AcLoadFlowParameters acLoadFlowParameters) {
        super(network, equationSystem, j, targetVector, equationVector, detailedReport);
        this.parameters = Objects.requireNonNull(parameters);
        this.driverOuterLoops = driverOuterLoops;
        this.driverSlackDistTol = driverSlackDistTol;
        this.driverReactiveLimitTol = driverReactiveLimitTol;
        this.driverMaxOuterIterations = driverMaxOuterIterations;
        this.driverMaxPqPvSwitch = driverMaxPqPvSwitch;
        this.hostOuterLoops = Objects.requireNonNull(hostOuterLoops);
        this.acLoadFlowParameters = acLoadFlowParameters;
        EquationSystemIndex<AcVariableType, AcEquationType> index = equationSystem.getIndex();
        // Purge listeners of ALREADY-COLLECTED solvers synchronously (we are on the solve thread,
        // no dispatch in flight) — this caps accumulation on a long-lived LfNetwork at ~1 while
        // keeping the cleaner thread away from the (non-thread-safe) listener lists.
        for (LfNetworkListener l : List.copyOf(network.getListeners())) {
            if (l instanceof GpuSolverInvalidation inv && inv.dead) {
                network.removeListener(l);
            }
        }
        index.addListener(invalidation);
        network.addListener(invalidation);
        // the cleaning action must not reference the solver itself (it would never become
        // phantom-reachable); the Invalidation listener object is referenced instead. It only
        // dead-flags: the index-side listener stays registered (no-op) for the es lifetime.
        GpuSolverInvalidation listener = invalidation;
        context.removeListeners = () -> listener.dead = true;
        CLEANER.register(this, context);
    }

    /** The native batch context handle, for the engine's pause/resume protocol with the driver.
     *  Zero until the first {@link #run} builds the context. */
    public long getContextHandle() {
        return context.handle.get();
    }

    @Override
    public String getName() {
        return "GPU Newton-Raphson";
    }

    @Override
    public AcSolverResult run(VoltageInitializer voltageInitializer, ReportNode reportNode) {
        if (!GpuAcNewtonSolver.isAvailable()) {
            throw new PowsyblException("GPU Newton-Raphson requested but the native library is not available "
                    + "(build with native/build-gpu.sh and provide cuDSS — see GpuAcNewtonSolver)");
        }
        // RESUME: the engine's pause/resume cycle already called setBatchDriverResume and
        // setBatchTargetDeltas — just re-enter solveBatchContext (the native side reads
        // driverResume=1 and continues from its paused position). Skip initStateVector and
        // the context rebuild — the state is on the device where we left it.
        boolean resuming = GpuAcNewtonSolver.readBatchDriverPaused(context.handle.get()) != 0;
        if (resuming) {
            return runResume(reportNode);
        }
        AcSolverUtil.initStateVector(network, equationSystem, voltageInitializer);

        AcSolverStatus status;
        int iterations = 0;
        int n = equationSystem.getIndex().getSortedVariablesToFind().size();
        try {
            long t0 = System.nanoTime();
            boolean rebuilt = false;
            if (invalidation.structureDirty) {
                // validate + build, through the shared cache (the batched SA reuses this extraction)
                // The RUN's outer-loop set and its DistributedSlack settings — the device must not
                // distribute with a default balance type, nor run a loop the run does not configure.
                GpuAcData data = GpuAcDataCache.extract(network, equationSystem,
                        !driverOuterLoops || driverDsLoop() != null, !driverOuterLoops || driverRlLoop() != null,
                        driverDsParams());
                // Three-level reconciliation, decided by the DATA not the event: a changed CSR pattern
                // (index packs) is STRUCTURE -> rebuild + cuDSS re-analysis; an unchanged pattern with
                // only changed element PARAMETERS (tap r1/a1, shunt b) is VALUES -> a cheap in-place
                // pack re-upload reusing the analysis; an unchanged data is a no-op (context reused).
                if (!data.sameStructureAs(cachedData)) {
                    GpuAcNewtonSolver.destroyBatchContext(context.handle.getAndSet(0));
                    // BATCH-OF-1 (2026-09-08): the sequential solve runs on the BATCH context in
                    // pure-NR mode (skipRlPhase=1, slackDistTol=inf neutralize the in-kernel outer
                    // phases; OLF's own outer loops stay on the CPU around this solver). One kernel
                    // set instead of two: per-row PER_EQUATION_TYPE criteria, slope-SVC (mode 2),
                    // continuous phase (mode 3) and the vdl/ZIP fills come from the batch pipeline
                    // instead of being ported twice. The shared structure comes from the
                    // per-extraction cache — the batched SA building its context from the SAME
                    // extraction reuses it device-side.
                    context.handle.set(GpuAcNewtonSolver.createBatchContext(GpuAcStructures.handle(n, data), 1,
                            GpuBatchContextCache.resolveBatchMode(),
                            data.rlVrow(), data.rlMinQ(), data.rlMaxQ(), data.rlLoadQ(), data.rlVtarget(),
                            data.rlMonitored(), data.rowSlope(),
                            data.dsPhiRow(), data.dsFactor(), data.dsSlackPhiRow(), data.dsSlackTargetP(),
                            data.dsBaseTargetP(), data.dsMaxDelta(), data.dsMinDelta(),
                            data.dsMode(), data.dsMarginUp(), data.dsMarginDown(),
                            data.dsSlackMarginUp(), data.dsSlackMarginDown()));
                    cachedData = data;
                    rebuilt = true;
                    criteriaDirty = true;
                    tablesDirty = true;
                    CONTEXT_REBUILDS.incrementAndGet();
                } else if (!data.sameAs(cachedData)) {
                    GpuAcNewtonSolver.refreshStructureValues(GpuAcStructures.handle(n, cachedData),
                            data.cbIn(), data.obIn(), data.shIn(), data.hvIn(), data.ziIn());
                    // the SHARED structure now holds the new values — re-key it to the new
                    // extraction so content matching never aliases it under the old one
                    GpuAcStructures.rebind(cachedData, data);
                    cachedData = data;
                    tablesDirty = true;
                    VALUE_REFRESHES.incrementAndGet();
                }
                invalidation.structureDirty = false;
                invalidation.activationDirty = true;
            }
            if (invalidation.activationDirty) {
                activation = GpuAcActivationExtractor.extractActivation(network, equationSystem, n);
                varType = GpuAcActivationExtractor.extractVarType(equationSystem, n);
                invalidation.activationDirty = false;
                criteriaDirty = true;                        // per-row thresholds follow the variables
            }
            if (criteriaDirty) {
                installRowCriteria(n);
                criteriaDirty = false;
            }
            // Since the batch-of-1 route (2026-09-08) modes 2 (slope-SVC) and 3 (continuous phase)
            // solve on-device — the batch kernels carry them. Mode 4 (a PINNED remote gen-VC member,
            // a base reactive-limit switch during this engine run) stays DELIBERATELY CPU-routed:
            // the pack's DISTR_Q weights are STATIC extraction-time qPct, and OLF's pin renormalizes
            // them (updateReactiveKeys) with NO equation-system event — solving on-device would use
            // stale weights. Every post-pin solve of the run runs on the CPU (correct, just
            // un-accelerated).
            for (int mode : activation.rowMode()) {
                if (mode >= 4) {
                    throw new UnsupportedOperationException(
                            "row mode " + mode + " (pinned remote gen-VC member) uses stale static qPct on-device");
                }
            }
            double[] target = activation.applyTargets(targetVector.getArray());
            double[] x0 = equationSystem.getStateVector().get().clone();
            rlStateOut = driverOuterLoops && cachedData.rlVrow().length > 0 ? new int[cachedData.rlVrow().length] : null;
            // Device-side state-vector scaling: NONE = undamped Newton; LINE_SEARCH backtracks on the
            // device (mirrors LineSearchStateVectorScaling); MAX_VOLTAGE_CHANGE clamps each step so the
            // largest |Δv|/|Δφ| stays within maxDv/maxDphi (mirrors MaxVoltageChangeStateVectorScaling).
            int lsMode = switch (parameters.getStateVectorScalingMode()) {
                case NONE -> 0;
                case LINE_SEARCH -> 1;
                case MAX_VOLTAGE_CHANGE -> 2;
            };
            long t1 = System.nanoTime();
            double convEpsPerEq = deviceConvEpsPerEq(parameters);
            if (driverOuterLoops && tablesDirty) {
                // The SAME run-constant outer-loop tables the batched SA installs on its chunk
                // contexts — a batch of one drives the same loops as a batch of five hundred.
                var dsLoop = driverDsLoop();
                GpuBatchRlCurves.Channel rlCurves = driverRlLoop() != null && dsLoop != null
                        ? GpuBatchRlCurves.build(network, equationSystem, cachedData, driverDsParams(), true, target,
                                acLoadFlowParameters.getNetworkParameters().isExtrapolateReactiveLimits())
                        : null;
                GpuBatchOuterLoopTables.install(context.handle.get(), cachedData, rlCurves);
                tablesDirty = false;
            }
            if (lsMode == 1) {                               // batch loop LINE_SEARCH (ported 2026-09-08)
                GpuAcNewtonSolver.setBatchLineSearch(context.handle.get(),
                        parameters.getLineSearchStateVectorScalingMaxIteration(),
                        parameters.getLineSearchStateVectorScalingStepFold());
            }
            // Pure-NR batch-of-1: skipRlPhase=1 disables the in-kernel reactive-limit phase and
            // the huge slackDistTol makes the distributed-slack phase a no-op (fabs(rem) <= tol
            // -> return) — OLF's outer loops run on the CPU around this solver, exactly as before.
            // AC-emulation HVDC statuses: OLF's freezing/limits loops DEACTIVATE the hvdc coupling
            // terms for any non-LINEAR status (the injections ride the target vector as VSC station
            // targetP, re-extracted every solve) — mirror it with the hv element's disabled bit,
            // which fillHvdcBatch honors. Without this the fill emitted the LINEAR coupling on top
            // of the frozen/saturated injections (the former frozen-HVDC cpu-base-only condition).
            boolean[][] elementMask = new boolean[1][cachedData.getTotalElementCount()];
            List<LfHvdc> hvPacked = network.getHvdcs().stream()
                    .filter(h -> h.getBus1() != null && h.getBus2() != null && h.isAcEmulation())
                    .toList();                               // the packer's exact enumeration order
            int hvOffset = cachedData.getTotalElementCount() - hvPacked.size();
            for (int h = 0; h < hvPacked.size(); h++) {
                if (hvPacked.get(h).getAcEmulationControl().getAcEmulationStatus()
                        != LfHvdc.AcEmulationControl.AcEmulationStatus.LINEAR_MODE) {
                    elementMask[0][hvOffset + h] = true;
                }
            }
            int[] ovIdx = new int[0];
            double[] ovVal = new double[0];
            hvdcCtls = null;
            monCtls = null;
            tvcCtls = null;
            tvcOwned = false;
            if (driverOuterLoops) {
                long handle = context.handle.get();
                int nElements = cachedData.getTotalElementCount();
                // HVDC AC emulation (freeze at OLF order 4, limits at 5) as driver phases. The freeze is
                // what OLF's FreezingHvdcACEmulation.initialize does before the first solve — every
                // emulation frozen at the warm state's flow — replayed host-side into target overrides,
                // then unfrozen by the driver's own phase.
                boolean hvdcLimits = driverLoop("AcHvdcAcEmulationLimits") != null;
                boolean hvdcFreeze = driverLoop("FreezingHvdcACEmulation") != null;
                if (hvdcLimits || hvdcFreeze) {
                    hvdcCtls = GpuBatchHvdcAcEmulation.buildControllers(network, equationSystem, nElements);
                }
                if (hvdcCtls != null && !hvdcCtls.isEmpty()) {
                    int nh = hvdcCtls.size();
                    int[][] hvStatus = new int[1][nh];
                    double[] appliedD1 = new double[nh];
                    double[] appliedD2 = new double[nh];
                    if (hvdcFreeze) {
                        List<Integer> ovI = new java.util.ArrayList<>();
                        List<Double> ovV = new java.util.ArrayList<>();
                        hvStatus = GpuBatchHvdcAcEmulation.applyFreeze(1, 1, n, hvdcCtls, x0, null, elementMask, target,
                                ovI, ovV, appliedD1, appliedD2);
                        ovIdx = ovI.stream().mapToInt(Integer::intValue).toArray();
                        ovVal = ovV.stream().mapToDouble(Double::doubleValue).toArray();
                    }
                    GpuBatchHvdcAcEmulation.seedBaseStatuses(1, 1, hvdcCtls, hvStatus, elementMask);
                    GpuBatchHvdcAcEmulation.installDriverLoops(handle, hvdcCtls, 1, hvStatus, elementMask,
                            appliedD1, appliedD2, hvdcLimits);
                } else {
                    GpuBatchHvdcAcEmulation.installDriverLoops(handle, null, 1, null, null, null, null, false);
                }
                // MonitoringVoltage (OLF order 6) as a driver phase. The builder probes each standby bus
                // with a VC-on/off round trip that fires network events — keep our dirty flags as they were.
                if (driverLoop("VoltageMonitoring") != null) {
                    boolean sd = invalidation.structureDirty;
                    boolean ad = invalidation.activationDirty;
                    monCtls = GpuBatchVoltageMonitoring.buildControllers(network, equationSystem, targetVector,
                            activation, target, cachedData.rlVrow());
                    invalidation.structureDirty = sd;
                    invalidation.activationDirty = ad;
                }
                GpuBatchVoltageMonitoring.installDriverLoop(handle, monCtls);
                // IncrementalTransformerVoltageControl (OLF order 8) IN-KERNEL when every controlled bus
                // has a single controller; otherwise the driver pauses at the slot for OLF's own loop.
                int hostMask = buildHostLoopMask();
                if (driverLoop("IncrementalTransformerVoltageControl")
                        instanceof com.powsybl.openloadflow.ac.outerloop.IncrementalTransformerVoltageControlOuterLoop itvc) {
                    tvcCtls = GpuBatchIncrementalTvc.buildControllers(network, equationSystem,
                            new com.powsybl.openloadflow.ac.solver.gpu.batch.GpuBatchElementMapper(network));
                    tvcOwned = GpuBatchIncrementalTvc.uploadToDevice(handle, tvcCtls, itvc.getMaxTapShift());
                    if (tvcOwned) {
                        hostMask &= ~(1 << (GpuAcNewtonSolver.HOST_LOOP_TVC - GpuAcNewtonSolver.HOST_LOOP_SVC));
                    }
                } else {
                    GpuBatchIncrementalTvc.uploadToDevice(handle, null, 1);
                }
                // Host-owned outer-loop slots: the driver pauses at their OLF positions and the
                // pause/resume cycle after solveBatchContext runs OLF's loop there. 0 clears a cached mask.
                GpuAcNewtonSolver.setBatchHostLoop(handle, hostMask);
            }
            double[] out = GpuAcNewtonSolver.solveBatchContext(context.handle.get(),
                    x0, target, new int[][] {activation.rowMode()}, activation.rowMode(),
                    elementMask,
                    ovIdx, ovVal,
                    new int[0], new double[0], new double[0], new double[0], new double[0], null, null,
                    null, null,
                    parameters.getMaxIterations(), GPU_TOLERANCE,
                    driverOuterLoops ? driverReactiveLimitTol : 0.0,
                    driverOuterLoops ? driverSlackDistTol : 1e300, convEpsPerEq,
                    lsMode,
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDv(),
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDphi(),
                    varType, null,
                    null, null, null, rlStateOut, null, null,
                    driverOuterLoops ? driverMaxOuterIterations : 0,
                    driverOuterLoops ? driverMaxPqPvSwitch : 0, 0,
                    driverOuterLoops ? 0 : 1,                // skipRlPhase: the driver owns RL now
                    0, null, null)[0];
            long t2 = System.nanoTime();
            if (driverOuterLoops && out.length > n + 2 && (int) out[n + 2] == -2) {
                // The driver's own detector rejected the scenario (every pinnable PV pinned: OLF's
                // GLOBAL "strongest stays PV" election, which the kernel does not run). The chunk
                // solver routes such a scenario to the CPU; the batch-of-one does the same — the
                // CPU Newton-Raphson reruns it and OLF's loops (deferred by the engine) decide.
                throw new PowsyblException("GPU driver rejected the base solve (all-pinned / forced-PV election)");
            }
            LOGGER.debug("GPU Newton-Raphson timing: prepare {} ms (context {}), solve {} ms",
                    (t1 - t0) / 1_000_000, rebuilt ? "rebuilt" : "reused", (t2 - t1) / 1_000_000);
            iterations = (int) out[n];
            // PAUSE/RESUME CYCLE: if the driver paused at a LOOP_HOST_* slot, run the corresponding
            // OLF outer loop on the host and resume. Repeat until the driver completes without pausing.
            if (driverOuterLoops && !hostOuterLoops.isEmpty()) {
                out = handleDriverPauses(out, n, target, elementMask, lsMode, convEpsPerEq);
                iterations = (int) out[n];
            }
            if (driverOuterLoops) {
                // The driver ran OLF's outer loops, which MUTATE TARGETS on the device. OLF's world
                // still holds the pre-loop ones, so without this the converged state is measured
                // against stale targets and the run is declared MAX_ITERATION_REACHED — measured as
                // 170 BUS_TARGET_P rows and 14 BUS_TARGET_V rows over 1e-4, where the same solve with
                // the driver off leaves none. Give the distribution back to the network, exactly as
                // OLF's own DistributedSlack does by setting generator targetP.
                writeBackDriverMutations(cachedData);
            }
            // write the device solution back; the equation vector re-evaluates from it
            equationSystem.getStateVector().set(Arrays.copyOf(out, n));
            equationVector.minus(targetVector);
            NewtonRaphsonStoppingCriteria.TestResult result =
                    parameters.getStoppingCriteria().test(equationVector.getArray(), equationSystem);
            // OLF_GPU_RESIDUAL_TRACE: the device solved to ITS residual, this is OLF's own residual at
            // the state it returned — a large value here means the device's equations disagree with
            // OLF's (a packing/row-mapping bug), not that the solver failed to converge. The
            // by-equation-type breakdown names the family to look at. (Test logback is broken —
            // stderr lever.)
            if (System.getenv("OLF_GPU_RESIDUAL_TRACE") != null) {
                double[] fx = equationVector.getArray();
                double n2 = 0;
                int worstRow = -1;
                double worst = 0;
                for (int i2 = 0; i2 < fx.length; i2++) {
                    n2 += fx[i2] * fx[i2];
                    if (Math.abs(fx[i2]) > worst) {
                        worst = Math.abs(fx[i2]);
                        worstRow = i2;
                    }
                }
                System.err.println("RESID_TRACE rows=" + fx.length + " cpuNorm2=" + Math.sqrt(n2)
                        + " cpuInfNorm=" + worst + " worstRow=" + worstRow
                        + " deviceIters=" + iterations + " stop=" + result.isStop());
                var worstEq = equationSystem.getIndex().getEquationAtColumn(worstRow);
                System.err.println("RESID_TRACE worst equation type=" + worstEq.getType()
                        + " elementNum=" + worstEq.getElementNum());
                java.util.Map<String, Integer> byType = new java.util.TreeMap<>();
                java.util.Map<String, Double> maxByType = new java.util.TreeMap<>();
                for (int i2 = 0; i2 < fx.length; i2++) {
                    if (Math.abs(fx[i2]) > 1e-4) {
                        String t = equationSystem.getIndex().getEquationAtColumn(i2).getType().toString();
                        byType.merge(t, 1, Integer::sum);
                        maxByType.merge(t, Math.abs(fx[i2]), Math::max);
                    }
                }
                System.err.println("RESID_TRACE rows over 1e-4 by type=" + byType + " max=" + maxByType);
                double[] devT = GpuAcNewtonSolver.readBatchTargets(context.handle.get());
                double[] olfT = targetVector.getArray();
                for (int i2 = 0; i2 < fx.length; i2++) {
                    if (Math.abs(fx[i2]) <= 1e-4) {
                        continue;
                    }
                    var eq = equationSystem.getIndex().getEquationAtColumn(i2);
                    StringBuilder sb = new StringBuilder("RESID_ROW row=" + i2 + " type=" + eq.getType()
                            + " f=" + fx[i2] + " olfTarget=" + olfT[i2]
                            + " devTarget=" + (devT != null && i2 < devT.length ? devT[i2] : Double.NaN)
                            + " devMode=" + activation.rowMode()[i2] + " x=" + out[i2]);
                    LfBus bus = eq.getType().toString().startsWith("BUS_") ? network.getBus(eq.getElementNum()) : null;
                    if (bus != null) {
                        sb.append(" bus=").append(bus.getId()).append(" vcEnabled=").append(bus.isGeneratorVoltageControlEnabled())
                          .append(" qLimitType=").append(bus.getQLimitType().map(Object::toString).orElse("-"))
                          .append(" genTargetQ=").append(bus.getGenerationTargetQ()).append(" loadTargetQ=").append(bus.getLoadTargetQ())
                          .append(" minQ=").append(bus.getMinQ()).append(" maxQ=").append(bus.getMaxQ())
                          .append(" hasVc=").append(bus.getGeneratorVoltageControl().isPresent())
                          .append(" isVcController=").append(bus.isGeneratorVoltageControlled());
                        if (cachedData.rlBusNum() != null && rlStateOut != null) {
                            for (int k = 0; k < cachedData.rlBusNum().length; k++) {
                                if (cachedData.rlBusNum()[k] == bus.getNum()) {
                                    sb.append(" rl[").append(k).append("]: state=").append(rlStateOut[k])
                                      .append(" vRow=").append(cachedData.rlVrow()[k]).append(" remote=").append(cachedData.rlRemote()[k])
                                      .append(" rlMinQ=").append(cachedData.rlMinQ()[k]).append(" rlMaxQ=").append(cachedData.rlMaxQ()[k])
                                      .append(" rlLoadQ=").append(cachedData.rlLoadQ()[k])
                                      .append(" devTargetAtVrow=").append(devT != null ? devT[cachedData.rlVrow()[k]] : Double.NaN)
                                      .append(" devModeAtVrow=").append(activation.rowMode()[cachedData.rlVrow()[k]]);
                                }
                            }
                        }
                    }
                    System.err.println(sb);
                }
            }
            status = result.isStop() ? AcSolverStatus.CONVERGED : AcSolverStatus.MAX_ITERATION_REACHED;
            LAST_RUN_ON_DEVICE.set(true);
            // DEVICE-BROKEN result safety net: a non-finite or absurd residual is a device-side
            // numerical failure (e.g. the open zi-with-loaded-far-bus-behind-a-remote-member shape,
            // where the sequential solve returns phi ~1e22 while the SAME system solves on the CPU
            // and in the batch context — see GpuSeqBaseFallbackTest). Returning it would poison
            // OLF's outer loops and fail the whole run (an SA base would come back FAILED where the
            // stock CPU provider succeeds); throw into the CPU-fallback path instead, so the solve
            // reruns on the CPU Newton-Raphson: never wrong, only un-accelerated. Ordinary
            // non-convergence (a plausible norm) is NOT retried — the CPU would reproduce it.
            if (status == AcSolverStatus.MAX_ITERATION_REACHED
                    && (!Double.isFinite(result.getNorm()) || result.getNorm() > 1e2)) {
                throw new PowsyblException("GPU Newton-Raphson returned a device-broken state (|f(x)|="
                        + result.getNorm() + ")");
            }
            LOGGER.debug("GPU Newton-Raphson: {} iterations, |f(x)|={}, {}", iterations, result.getNorm(), status);
            if (System.getenv("OLF_GPU_SEQ_TRACE") != null) {   // test logback is broken — stderr trace
                System.err.println("SEQ_TRACE solve iters=" + iterations + " norm=" + result.getNorm()
                        + " devNorm=" + out[n + 1] + " status=" + status);
                double[] fv = equationVector.getArray();
                int[] tm = activation.targetMap();
                var vars = equationSystem.getIndex().getSortedVariablesToFind();
                for (int c = 0; c < fv.length; c++) {
                    if (Math.abs(fv[c]) > 1e-4) {
                        int row = -1;
                        for (int r = 0; r < tm.length; r++) {
                            if (tm[r] == c) {
                                row = r;
                            }
                        }
                        final int rr = row;
                        var vv = vars.stream().filter(x -> x.getRow() == rr).findFirst().orElse(null);
                        System.err.println("SEQ_TRACE   |f|=" + Math.abs(fv[c]) + " col=" + c + " row=" + row
                                + " mode=" + (row >= 0 ? activation.rowMode()[row] : -9)
                                + " var=" + (vv != null ? vv.getType() + "#" + vv.getElementNum() : "?"));
                    }
                }
            }
        } catch (RuntimeException e) {
            CPU_FALLBACK_COUNT.incrementAndGet();
            LAST_RUN_ON_DEVICE.set(false);
            // the native context may be mid-iteration inconsistent — rebuild next run
            GpuAcNewtonSolver.destroyBatchContext(context.handle.getAndSet(0));
            cachedData = null;
            invalidation.structureDirty = true;
            if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                System.err.println("GPU_BASE_FALLBACK " + e);   // test logback is broken — stderr lever
                e.printStackTrace();
            }
            if (Boolean.parseBoolean(System.getProperty(CPU_FALLBACK_PROPERTY, "true"))) {
                // Graceful degradation: anything the GPU path cannot handle — an extractor
                // validation rejection (unsupported variable/equation/term, remote control,
                // zero-impedance shapes) or a native/device error — is re-run on the standard
                // CPU Newton-Raphson with the SAME equation system and parameters. This makes
                // the GPU solver safe as a DEFAULT (e.g. the SA base load flow): never wrong,
                // only un-accelerated.
                LOGGER.warn("GPU Newton-Raphson unavailable for this run ({}); falling back to the CPU Newton-Raphson",
                        e.toString());
                return new NewtonRaphson(network, parameters, equationSystem, j, targetVector, equationVector,
                        detailedReport).run(voltageInitializer, reportNode);
            }
            LOGGER.error("GPU Newton-Raphson failed: {}", e.toString());
            status = AcSolverStatus.SOLVER_FAILED;
        }

        if (status == AcSolverStatus.CONVERGED || parameters.isAlwaysUpdateNetwork()) {
            AcSolverUtil.updateNetwork(network, equationSystem);
        }

        Map<Integer, Double> slackBusActivePowerMismatch = new TreeMap<>();
        for (LfSynchronousNetwork lfScNetwork : network.getSynchronousNetworks()) {
            slackBusActivePowerMismatch.put(lfScNetwork.getNumSC(), lfScNetwork.getSlackBuses().stream().mapToDouble(LfBus::getMismatchP).sum());
        }
        return new AcSolverResult(status, iterations, slackBusActivePowerMismatch);
    }

    /** Build a map from HOST_LOOP_* kind to the OLF outer loop instance. */
    private java.util.Map<Integer, com.powsybl.openloadflow.ac.outerloop.AcOuterLoop> buildHostLoopMap() {
        var map = new java.util.HashMap<Integer, com.powsybl.openloadflow.ac.outerloop.AcOuterLoop>();
        for (var loop : hostOuterLoops) {
            int kind = loopKindFor(loop.getName());
            if (kind >= GpuAcNewtonSolver.HOST_LOOP_SVC) {
                map.put(kind, loop);
            }
        }
        return map;
    }

    /**
     * Handle the driver pause/resume cycle: while the native driver is paused at a LOOP_HOST_*
     * slot, load the device state into the network, run the corresponding OLF outer loop, write
     * any target changes back to the device, and resume. Returns the final device output (the
     * converged state when no more loops fire).
     */
    @SuppressWarnings("java:S3776")
    private double[] handleDriverPauses(double[] out, int n, double[] target,
                                         boolean[][] elementMask, int lsMode, double convEpsPerEq) {
        long handle = context.handle.get();
        int pausedKind = GpuAcNewtonSolver.readBatchDriverPaused(handle);
        if (pausedKind == 0) {
            return out;   // driver completed without pausing
        }
        var loopMap = buildHostLoopMap();
        // Per-loop persistent context data (iteration count, per-loop state). Outer loops store
        // their own state via context.setData(); we keep one context per loop across rounds.
        // Build (outer loop, context) pairs for the host-paused loops. Use the engine's own
        // AcLoadFlowContext if available (set by GpuDriverOuterLoopEngine before the solve), so the
        // outer loops share the solver's equation system and Jacobian. When unavailable (standalone
        // LoadFlow.run, no engine override) the loops run without a loadFlowContext — simple loops
        // (PhaseControl, SimpleTransformerVoltageControl) work, sensitivity-dependent ones will need
        // the context wired (Step 5).
        var ctxMap = new java.util.HashMap<Integer, org.apache.commons.lang3.tuple.Pair<
                com.powsybl.openloadflow.ac.outerloop.AcOuterLoop,
                com.powsybl.openloadflow.ac.AcOuterLoopContext>>();
        var engineContext = GPU_ENGINE_CONTEXT.get();
        if (engineContext != null) {
            var pairs = com.powsybl.openloadflow.ac.AcloadFlowEngine.createOuterLoopsAndContexts(engineContext, false);
            for (var pair : pairs) {
                int kind = loopKindFor(pair.getLeft().getName());
                if (kind >= GpuAcNewtonSolver.HOST_LOOP_SVC && loopMap.containsKey(kind)) {
                    ctxMap.put(kind, pair);
                    pair.getLeft().initialize(pair.getRight());
                }
            }
        }
        while (pausedKind != 0) {
            var pair = ctxMap.get(pausedKind);
            if (pair == null) {
                LOGGER.warn("Driver paused at host loop kind {} with no registered handler — resuming as STABLE", pausedKind);
                GpuAcNewtonSolver.setBatchDriverResume(handle, new byte[]{0});
            } else {
                var loop = pair.getLeft();
                var loopCtx = pair.getRight();
                // Load device state into the equation system and network
                equationSystem.getStateVector().set(Arrays.copyOf(out, n));
                AcSolverUtil.updateNetwork(network, equationSystem);
                // Run OLF's own outer loop check
                var result = loop.check(loopCtx, com.powsybl.commons.report.ReportNode.NO_OP);
                if (System.getenv("OLF_GPU_BATCH_TRACE") != null) {
                    System.err.printf("HOST_LOOP kind=%d name=%s iter=%d status=%s%n",
                            pausedKind, loop.getName(), loopCtx.getIteration(), result.status());
                }
                if (result.status() == com.powsybl.openloadflow.lf.outerloop.OuterLoopStatus.UNSTABLE) {
                    // The loop mutated the network. Re-extract targets (the target vector evaluates
                    // against the updated network state, so getArray() reflects the loop's changes).
                    double[] newTargets = activation.applyTargets(targetVector.getArray());
                    // Write per-row target deltas to the device (only rows that changed)
                    List<Integer> deltaRows = new java.util.ArrayList<>();
                    List<Double> deltaVals = new java.util.ArrayList<>();
                    for (int r = 0; r < n; r++) {
                        if (newTargets[r] != target[r]) {
                            deltaRows.add(r);          // flat (scenario=0 * n + row) for batch-of-1
                            deltaVals.add(newTargets[r]);
                        }
                    }
                    if (!deltaRows.isEmpty()) {
                        GpuAcNewtonSolver.setBatchTargetDeltas(handle,
                                deltaRows.stream().mapToInt(Integer::intValue).toArray(),
                                deltaVals.stream().mapToDouble(Double::doubleValue).toArray());
                    }
                    // Update our local target copy for the next round's diff
                    System.arraycopy(newTargets, 0, target, 0, n);
                    GpuAcNewtonSolver.setBatchDriverResume(handle, new byte[]{1});
                    loopCtx.setIteration(loopCtx.getIteration() + 1);
                } else {
                    GpuAcNewtonSolver.setBatchDriverResume(handle, new byte[]{0});
                }
            }
            // Resume the driver — solveBatchContext continues from its paused position
            out = GpuAcNewtonSolver.solveBatchContext(context.handle.get(),
                    equationSystem.getStateVector().get().clone(), target,
                    new int[][] {activation.rowMode()}, activation.rowMode(),
                    elementMask,
                    new int[0], new double[0],
                    new int[0], new double[0], new double[0], new double[0], new double[0], null, null,
                    null, null,
                    parameters.getMaxIterations(), GPU_TOLERANCE,
                    driverReactiveLimitTol, driverSlackDistTol, convEpsPerEq,
                    lsMode,
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDv(),
                    parameters.getMaxVoltageChangeStateVectorScalingMaxDphi(),
                    varType, null,
                    null, null, null, rlStateOut, null, null,
                    driverMaxOuterIterations, driverMaxPqPvSwitch, 0, 0,
                    0, null, null)[0];
            pausedKind = GpuAcNewtonSolver.readBatchDriverPaused(handle);
        }
        // Cleanup: finalize outer loops (same as OLF engine's reverse-order cleanup)
        for (var entry : ctxMap.entrySet()) {
            entry.getValue().getLeft().cleanup(entry.getValue().getRight());
        }
        return out;
    }

    /**
     * Resume a paused driver. The engine's pause/resume cycle already called
     * {@link GpuAcNewtonSolver#setBatchDriverResume} and (if the host loop was UNSTABLE)
     * {@link GpuAcNewtonSolver#setBatchTargetDeltas} — this just re-enters
     * {@code solveBatchContext} which the native side resumes from its paused position.
     * The activation, context, and element mask from the first call are still valid (the
     * host loop mutated the NETWORK but the device-side structures are unchanged — only
     * targets are patched via the target-delta channel).
     */
    private AcSolverResult runResume(ReportNode reportNode) {
        int n = equationSystem.getIndex().getSortedVariablesToFind().size();
        // Re-use the activation and element mask from the paused call. The native side
        // ignores x0/target/rowMode/overrides on a resume (driverResume=1 skips all uploads),
        // but the JNI wrapper needs non-null arrays at the right sizes.
        double[] target = activation.applyTargets(targetVector.getArray());
        double[] x0 = equationSystem.getStateVector().get().clone();
        boolean[][] elementMask = new boolean[1][cachedData.getTotalElementCount()];
        double convEpsPerEq = deviceConvEpsPerEq(parameters);
        int lsMode = switch (parameters.getStateVectorScalingMode()) {
            case NONE -> 0;
            case LINE_SEARCH -> 1;
            case MAX_VOLTAGE_CHANGE -> 2;
        };
        double[] out = GpuAcNewtonSolver.solveBatchContext(context.handle.get(),
                x0, target, new int[][] {activation.rowMode()}, activation.rowMode(),
                elementMask,
                new int[0], new double[0],
                new int[0], new double[0], new double[0], new double[0], new double[0], null, null,
                null, null,
                parameters.getMaxIterations(), GPU_TOLERANCE,
                driverOuterLoops ? driverReactiveLimitTol : 0.0,
                driverOuterLoops ? driverSlackDistTol : 1e300, convEpsPerEq,
                lsMode,
                parameters.getMaxVoltageChangeStateVectorScalingMaxDv(),
                parameters.getMaxVoltageChangeStateVectorScalingMaxDphi(),
                varType, null,
                null, null, null, rlStateOut, null, null,
                driverOuterLoops ? driverMaxOuterIterations : 0,
                driverOuterLoops ? driverMaxPqPvSwitch : 0, 0,
                driverOuterLoops ? 0 : 1,
                0, null, null)[0];
        int iterations = (int) out[n];
        // Write the (possibly intermediate) state back to the equation system so the engine
        // can run updateNetwork + the host loop's check() against it.
        equationSystem.getStateVector().set(Arrays.copyOf(out, n));
        equationVector.minus(targetVector);
        NewtonRaphsonStoppingCriteria.TestResult result =
                parameters.getStoppingCriteria().test(equationVector.getArray(), equationSystem);
        if (driverOuterLoops && GpuAcNewtonSolver.readBatchDriverPaused(context.handle.get()) == 0) {
            writeBackDriverMutations(cachedData);
            equationVector.minus(targetVector);               // re-measure against the written-back targets
            result = parameters.getStoppingCriteria().test(equationVector.getArray(), equationSystem);
        }
        AcSolverStatus status = result.isStop() ? AcSolverStatus.CONVERGED : AcSolverStatus.MAX_ITERATION_REACHED;
        LOGGER.debug("GPU Newton-Raphson resume: {} iterations, |f(x)|={}, {}", iterations, result.getNorm(), status);
        Map<Integer, Double> slackBusActivePowerMismatch = new TreeMap<>();
        for (LfSynchronousNetwork lfScNetwork : network.getSynchronousNetworks()) {
            slackBusActivePowerMismatch.put(lfScNetwork.getNumSC(), lfScNetwork.getSlackBuses().stream().mapToDouble(LfBus::getMismatchP).sum());
        }
        return new AcSolverResult(status, iterations, slackBusActivePowerMismatch);
    }
}
