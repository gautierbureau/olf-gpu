#!/usr/bin/env bash
# Copyright (c) 2026, RTE (http://www.rte-france.com) - SPDX-License-Identifier: MPL-2.0
#
# What the cuDSS fill-reducing ordering does to the GPU solver: runs the fixed-iteration bench (default) or the
# full test suite once per ordering, each under its own platform configuration
#     open-loadflow-gpu-default-parameters:
#       reorderingAlgorithm: <ORDERING>
# (the GpuLoadFlowParameters every run without its own takes, the bench and the suite included), and prints one
# block per ordering.
#
# Usage:
#   native/sweep_ordering.sh [bench|suite] [ORDERING...]
#   ORDERING: a GpuReorderingAlgorithm. Default: AUTO LIBRARY_DEFAULT BTF_COLAMD COLAMD AMD NESTED_DISSECTION.
#   NONE (natural order) only when named: its fill can exhaust device memory.
#
# Read the bench blocks on all their lines: LU_NNZ (OLF_GPU_CUDSS, the fill the analysis produced: smaller is
# sparser), the warm K line (time) and the convergence count. A sparser or faster ordering that converges worse is
# not a win. The suite blocks give the counts and the failing tests.
# The ordering also applies to the robust retry (matching and pivoting), which cuDSS 0.8 refuses under the COLAMD
# orderings: the suite mode is what shows the effect there.
# The native library is not rebuilt: build it first (native/build-gpu.sh).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -f "$REPO/native/local-env.sh" ] && . "$REPO/native/local-env.sh"

MODE="${1:-bench}"
[[ "$MODE" == bench || "$MODE" == suite ]] || { echo "mode must be bench or suite, not $MODE"; exit 2; }
shift || true
ORDERINGS=("$@")
if [ ${#ORDERINGS[@]} -eq 0 ]; then
    ORDERINGS=(AUTO LIBRARY_DEFAULT BTF_COLAMD COLAMD AMD NESTED_DISSECTION)
fi

WORK="$(mktemp -d)"
SUMMARY="$WORK/summary.txt"
for ordering in "${ORDERINGS[@]}"; do
    dir="$WORK/$ordering"
    mkdir -p "$dir"
    printf 'open-loadflow-gpu-default-parameters:\n  reorderingAlgorithm: %s\n' "$ordering" > "$dir/config.yml"
    log="$WORK/$ordering.log"
    echo ">> $MODE with reorderingAlgorithm=$ordering (log: $log)"
    if [[ "$MODE" == bench ]]; then
        OLF_GPU_TRACE=1 native/run_bench.sh --no-build -Dpowsybl.config.dirs="$dir" > "$log" 2>&1 || true
        {
            echo "=== $ordering"
            grep -aE "OLF_GPU_CUDSS" "$log" | sort | uniq -c || true
            grep -aE "BARE-FIXED-ITER|E2E[^ ]* K=|K=[0-9]+ +:|Tests run: [0-9]+, Fail" "$log" || echo "   no result line"
        } >> "$SUMMARY"
    else
        mvn -s "$HOME/Projects/powsybl/tmp/settings.xml" -o -Dcheckstyle.skip test -Dpowsybl.config.dirs="$dir" \
            < /dev/null > "$log" 2>&1 || true
        {
            echo "=== $ordering"
            grep -aE "Tests run: [0-9]+, Failures: [0-9]+, Errors: [0-9]+, Skipped: [0-9]+$" "$log" | tail -1 \
                || echo "   no result line"
            grep -aE "<<< (FAILURE|ERROR)!" "$log" | sed 's/ -- Time elapsed.*//' | sort -u || true
        } >> "$SUMMARY"
    fi
done

echo
echo "=================== ORDERING SWEEP ($MODE) ==================="
cat "$SUMMARY"
echo "=============================================================="
echo "(logs and configurations: $WORK)"
