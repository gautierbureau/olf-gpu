#!/usr/bin/env bash
# Global number tracker: runs the production SA bench + the fixed-iter bench and appends
# one row to docs/numbers.md. Run after every change touching the batched GPU path.
# Usage: tools/track-numbers.sh ["note for the row"]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -f native/local-env.sh ] && . native/local-env.sh
export OLF_GPU_CONT_FILE="${OLF_GPU_CONT_FILE:-$HOME/Projects/powsybl/cuda/matched_contingencies/matched_test6_ids.txt}"
NOTE="${1:-}"
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet && git diff --cached --quiet || echo '+dirty')"
LOG_P="$(mktemp)"; LOG_F="$(mktemp)"

echo ">> production bench (RTE params, CPU-4t vs GPU batch)"
mvn -s "$HOME/Projects/powsybl/tmp/settings.xml" test -Dcheckstyle.skip \
    -Dtest='GpuVsCpuProductionSaBenchmarkTest' -DfailIfNoTests=false >"$LOG_P" 2>&1 \
    || { echo "production bench FAILED — see $LOG_P"; exit 1; }
echo ">> fixed-iter bench (K=4, UBATCH 500)"
bash tools/run_bench.sh --no-build >"$LOG_F" 2>&1 \
    || { echo "fixed-iter bench FAILED — see $LOG_F"; exit 1; }

num() { grep -aoE "$1" "$2" | head -1 | grep -oE '[0-9]+[.][0-9]+' | head -1; }
PROD_COLD=$(num 'COLD, run 1\)[^|]*\|[ ]*[0-9.]+ ms/ctg' "$LOG_P")
PROD_WARM=$(num 'warm, run 2\)[^|]*\|[ ]*[0-9.]+ ms/ctg' "$LOG_P")
CPU_MT=$(num 'AcSecurityAnalysis \( 4 threads\)[^|]*\|[ ]*[0-9.]+ ms/ctg' "$LOG_P")
RATIO=$(awk -v c="$CPU_MT" -v g="$PROD_WARM" 'BEGIN{ if (g>0) printf "x%.2f", c/g }')
DIFF=$(grep -aoE 'onlyCPU=[0-9]+ onlyGPU=[0-9]+' "$LOG_P" | head -1 | grep -oE '[0-9]+' | paste -sd/ -)
CTG=$(grep -aoE 'contingencies affected=[0-9]+' "$LOG_P" | head -1 | grep -oE '[0-9]+')
FX_COLD=$(num 'E2E K=4\][^|]*\|[^|]*\|[ ]*[0-9.]+ ms/ctg' "$LOG_F")
FX_WARM=$(num '[0-9.]+ ms/scenario' "$LOG_F")
FX_CONV=$(grep -aoE '[0-9]+/500 reached' "$LOG_F" | head -1 | cut -d' ' -f1)
CAPS=$(python3 codegen/capability_count.py | grep -oE 'GPU-covered.*: [0-9]+/[0-9]+ \([0-9]+%\)' | grep -oE '[0-9]+/[0-9]+ \([0-9]+%\)')

ROW="| $(date +%F) | ${COMMIT} | ${PROD_COLD:-—} | ${PROD_WARM:-—} | ${CPU_MT:-—} | ${RATIO:-—} | ${DIFF:-—}/${CTG:-—} | ${FX_COLD:-—} | ${FX_WARM:-—} | ${FX_CONV:-—} | ${CAPS:-—} | ${NOTE} |"
echo "$ROW" >> docs/numbers.md
echo "appended to docs/numbers.md:"
echo "$ROW"
echo "(full logs: $LOG_P / $LOG_F)"
