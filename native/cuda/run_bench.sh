#!/usr/bin/env bash
# GPU batched N-1 throughput regression harness: fixed-Newton-iteration + UBATCH on
# case9241pegase, 500 non-islanding contingencies, K=4 by default.
# MUST be run (and stay at the recorded baseline) after every change touching the batched GPU path.
#
# Usage:
#   native/cuda/run_bench.sh                 # K=4 regression check (builds vs the pinned cuDSS first)
#   native/cuda/run_bench.sh --no-build      # skip the native rebuild (lib already vs the pinned cuDSS)
#   OLF_GPU_FIXED_ITERS="4,6,10" native/cuda/run_bench.sh   # K-sweep instead of just K=4
#
# Recorded baseline (RTX A1000, cuDSS 0.8.0, UBATCH, K=4, maxBatch=500): warm ~4.9 ms/scn (205 scn/s);
# 499/500 reached 1e-6, med ||F||inf ~3.5e-12. Old 0.7.1/maxBatch=32 baseline was ~8.2 ms/scn.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
[ -f "$REPO/native/local-env.sh" ] && . "$REPO/native/local-env.sh"   # machine paths (CUDSS_ROOT, OLF_GPU_CASE_DIR, ...)

export CUDA_HOME="${CUDA_HOME:-$HOME/Projects/Software/cuda-toolkit}"
export CUDSS_ROOT="${CUDSS_ROOT:-$HOME/Projects/powsybl/cuda/cudss}"   # 0.8.0 = the pinned cuDSS version (UBATCH scales with batch size there)
export OLF_GPU_CONT_FILE="${OLF_GPU_CONT_FILE:-$HOME/Projects/powsybl/cuda/matched_contingencies/matched_test6_ids.txt}"
export OLF_GPU_BATCHMODE="${OLF_GPU_BATCHMODE:-3}"   # 3 = UBATCH (single shared-block analysis)
export OLF_GPU_MAXBATCH="${OLF_GPU_MAXBATCH:-500}"    # 0.8.0 UBATCH amortizes per-launch cost with batch size
export OLF_GPU_E2E="${OLF_GPU_E2E:-1}"               # also print cold end-to-end (CPU base LF + cold GPU SA)
export OLF_GPU_E2E_K="${OLF_GPU_E2E_K:-4}"
export OLF_GPU_FIXED_ITERS="${OLF_GPU_FIXED_ITERS:-4}"

if [[ "${1:-}" != "--no-build" ]]; then
    echo ">> building libolfgpu against cuDSS at $CUDSS_ROOT"
    native/build-gpu.sh >/dev/null 2>&1 || { echo "native build FAILED — re-run native/build-gpu.sh to see the error"; exit 1; }
fi

LOG="$(mktemp)"
echo ">> running GpuBatchedN1FixedIterTest#pegase9241FixedIterations (UBATCH, K=$OLF_GPU_FIXED_ITERS, 500 matched ctg)"
MVN_STATUS=0
mvn -s "$HOME/Projects/powsybl/tmp/settings.xml" test -Dcheckstyle.skip -Dtest='GpuBatchedN1FixedIterTest#pegase9241FixedIterations' -DfailIfNoTests=false >"$LOG" 2>&1 || MVN_STATUS=$?

echo "=================== RESULT ==================="
grep -aE "OLF_GPU_CUDSS|BARE-FIXED-ITER|E2E[^ ]* K=|K=[0-9]+ +:|Tests run: [0-9]" "$LOG" || { echo "no result lines — full log:"; tail -40 "$LOG"; }
echo "=============================================="
echo "(full log: $LOG)"
exit "$MVN_STATUS"
