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
 * <p>The base LF already goes through the batch code as a batch of one. With
 * {@code OLF_GPU_DRIVER_OUTER_LOOPS=1} its driver also runs OLF's outer loops, exactly as it does
 * for a 600-scenario chunk — a batch of one is still a batch. This engine then has to STOP driving
 * the same loops in Java, or they are applied twice: the device distributes into its own targets
 * while the network still holds the pre-distribution ones, so OLF's DistributedSlack measures the
 * amount just distributed as a fresh mismatch and distributes it again.
 *
 * <p>Dropping them HERE rather than from {@code parameters.getOuterLoops()} is deliberate: the GPU
 * security analysis derives its own batch configuration and qualification gates from that shared
 * list, and editing it stopped the run batching at all (share 100% -> 0%).
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public class GpuDriverOuterLoopEngine extends AcloadFlowEngine {

    public GpuDriverOuterLoopEngine(AcLoadFlowContext context) {
        super(context);
    }

    /** The loops the device driver runs itself, and which must therefore not run here as well. */
    static boolean drivenOnDevice(AcOuterLoop outerLoop) {
        return outerLoop instanceof DistributedSlackOuterLoop
                || outerLoop instanceof ReactiveLimitsOuterLoop;
    }

    @Override
    protected List<Pair<AcOuterLoop, AcOuterLoopContext>> engineOuterLoopsAndContexts(AcLoadFlowContext context) {
        return super.engineOuterLoopsAndContexts(context).stream()
                .filter(p -> !drivenOnDevice(p.getLeft()))
                .toList();
    }
}
