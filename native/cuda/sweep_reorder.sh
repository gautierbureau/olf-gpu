#!/usr/bin/env bash
# cuDSS fill-reducing-ordering SWEEP on the batched N-1 bench: factorize (272 ms/iter, the
# largest per-iteration term) is LU-fill-bound, and the fill is decided by the ordering cuDSS
# picks at ANALYSIS. This driver runs the standard bench once per OLF_GPU_REORDER_ALG value and
# reports, per algorithm: the reported LU_NNZ (the deterministic quality signal — smaller is
# sparser factors), the warm K=4 chunk line (the timing verdict) and the convergence count.
#
# Usage:
#   native/cuda/sweep_reorder.sh                    # build once, sweep default btf_colamd colamd amd nd
#   native/cuda/sweep_reorder.sh --no-build         # lib already built vs the pinned cuDSS
#   OLF_GPU_REORDER_ALGS="default amd" native/cuda/sweep_reorder.sh   # explicit subset
#   OLF_GPU_BASE_ON_GPU=1 native/cuda/sweep_reorder.sh                # base-on-GPU E2E variant
#
# Notes:
# - "none" (natural order) is EXCLUDED by default: expect pathological fill on grid patterns —
#   opt in via OLF_GPU_REORDER_ALGS and a small OLF_GPU_MAXBATCH or it may exhaust device memory.
# - Read the verdict on BOTH columns: a sparser LU that converges worse (a different elimination
#   order changes last-ulp rounding) is not a win — require 499/500 @ 1e-6 like the baseline.
# - Before adopting a non-default winner, run the FULL suite with OLF_GPU_REORDER_ALG set — the
#   known knife-edge (GpuLineSearchTest plateau) flips on exactly this kind of numerics change.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

ALGS="${OLF_GPU_REORDER_ALGS:-default btf_colamd colamd amd nd}"
BUILD_FLAG=""
if [[ "${1:-}" == "--no-build" ]]; then
    BUILD_FLAG="--no-build"
else
    echo ">> building libolfgpu once vs the pinned cuDSS"
    native/build-gpu.sh >/dev/null 2>&1 || { echo "native build FAILED — re-run native/build-gpu.sh to see the error"; exit 1; }
fi

SUMMARY="$(mktemp)"
for alg in $ALGS; do
    echo ""
    echo "########## OLF_GPU_REORDER_ALG=$alg ##########"
    LOG="$(mktemp)"
    if OLF_GPU_REORDER_ALG="$alg" native/cuda/run_bench.sh --no-build >"$LOG" 2>&1; then
        grep -aE "OLF_GPU_CUDSS|E2E K=|K=[0-9]+ +:" "$LOG" || true
        {
            echo "=== $alg ==="
            grep -a "OLF_GPU_CUDSS" "$LOG" | sort -u || true
            grep -aE "E2E K=|K=[0-9]+ +:" "$LOG" || true
        } >>"$SUMMARY"
    else
        echo "RUN FAILED for $alg (device memory? unsupported ordering?) — log: $LOG"
        echo "=== $alg === RUN FAILED (log: $LOG)" >>"$SUMMARY"
        continue
    fi
    rm -f "$LOG"
done

echo ""
echo "=================== SWEEP SUMMARY ==================="
cat "$SUMMARY"
echo "====================================================="
echo "Verdict rule: smallest LU_NNZ / fastest warm chunk WITH unchanged convergence (499/500 @ 1e-6)."
echo "Before adopting a winner: full suite with OLF_GPU_REORDER_ALG=<winner> (knife-edge: GpuLineSearchTest)."
rm -f "$SUMMARY"
