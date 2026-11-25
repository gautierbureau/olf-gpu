/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.ac;

/**
 * An outer loop context for tests, in OLF's package for its package-private constructor.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
public final class AcOuterLoopContexts {

    private AcOuterLoopContexts() {
    }

    /** A context on the load flow context's network, wired to it, its loop not initialized: what OLF's engine builds
     *  for a loop before initializing it. */
    public static AcOuterLoopContext create(AcLoadFlowContext context) {
        AcOuterLoopContext outerLoopContext = new AcOuterLoopContext(context.getNetwork());
        outerLoopContext.setLoadFlowContext(context);
        return outerLoopContext;
    }
}
