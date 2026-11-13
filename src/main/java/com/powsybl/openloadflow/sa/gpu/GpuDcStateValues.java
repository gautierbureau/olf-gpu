/**
 * Copyright (c) 2026, RTE (http://www.rte-france.com)
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * SPDX-License-Identifier: MPL-2.0
 */
package com.powsybl.openloadflow.sa.gpu;

/**
 * What {@link GpuDcStateReader} reads a contingency's values from: its whole state array, or the few values the device
 * gathered for it (the closed side-1 flows of the branches read, the state rows of the buses whose angles are read),
 * without the state ever leaving the device.
 *
 * @author Gautier Bureau {@literal <gautier.bureau at rte-france.com>}
 */
interface GpuDcStateValues {

    /** A state row: an angle or a shift. */
    double state(int row);

    /** Whether the device computed this branch's closed side-1 flow ({@link #flow1}). */
    boolean hasFlow(int branchNum);

    /** The branch's closed side-1 flow as the device computed it: OLF's -power * deltaPhase on the base model. */
    double flow1(int branchNum);

    /** The whole state. */
    static GpuDcStateValues of(double[] state) {
        return new GpuDcStateValues() {
            @Override
            public double state(int row) {
                return state[row];
            }

            @Override
            public boolean hasFlow(int branchNum) {
                return false;
            }

            @Override
            public double flow1(int branchNum) {
                throw new IllegalStateException("no device flow");
            }
        };
    }

    /**
     * The values gathered for the contingencies of a chunk, one contingency at a time ({@link #select}): lookups by
     * branch and by row through stamped arrays, so selecting a contingency costs only its own values.
     */
    final class Gathered implements GpuDcStateValues {

        private final int[] flowStamp;
        private final double[] flow;
        private final int[] rowStamp;
        private final double[] row;
        private int stamp;

        // per contingency of the chunk: its pairs in the gathered arrays
        private int[] flowPtr = new int[1];
        private int[] flowBranch = new int[0];
        private double[] flowValue = new double[0];
        private int[] rowPtr = new int[1];
        private int[] rowIndex = new int[0];
        private double[] rowValue = new double[0];

        Gathered(int branchCount, int n) {
            flowStamp = new int[branchCount];
            flow = new double[branchCount];
            rowStamp = new int[n];
            row = new double[n];
        }

        /** The chunk's gathered values: contingency c's flows at [flowPtr[c], flowPtr[c + 1]), its rows likewise. */
        void setChunk(int[] flowPtr, int[] flowBranch, double[] flowValue, int[] rowPtr, int[] rowIndex, double[] rowValue) {
            this.flowPtr = flowPtr;
            this.flowBranch = flowBranch;
            this.flowValue = flowValue;
            this.rowPtr = rowPtr;
            this.rowIndex = rowIndex;
            this.rowValue = rowValue;
        }

        /** Read contingency c of the chunk from now on. */
        Gathered select(int c) {
            stamp++;
            for (int k = flowPtr[c]; k < flowPtr[c + 1]; k++) {
                flowStamp[flowBranch[k]] = stamp;
                flow[flowBranch[k]] = flowValue[k];
            }
            for (int k = rowPtr[c]; k < rowPtr[c + 1]; k++) {
                rowStamp[rowIndex[k]] = stamp;
                row[rowIndex[k]] = rowValue[k];
            }
            return this;
        }

        @Override
        public double state(int r) {
            if (rowStamp[r] != stamp) {
                throw new IllegalStateException("state row " + r + " not gathered");
            }
            return row[r];
        }

        @Override
        public boolean hasFlow(int branchNum) {
            return flowStamp[branchNum] == stamp;
        }

        @Override
        public double flow1(int branchNum) {
            return flow[branchNum];
        }
    }
}
