#!/usr/bin/env bash
# Copyright (c) 2026, RTE (http://www.rte-france.com) - SPDX-License-Identifier: MPL-2.0
#
# The host check of our batched LU's transposed solve (native/host/lu_transposed_check.cpp): no GPU, no CUDA toolkit,
# g++ only. Compiles the LU plans and the transposed solve's bodies with it, runs it on its built-in systems and on the
# CSR systems given as arguments ("n nnz", then the n + 1 row pointers, the nnz columns, the nnz values), and exits
# non-zero on a failed check.
#
#   CXX  default g++
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
[ -f "$ROOT/native/local-env.sh" ] && . "$ROOT/native/local-env.sh"   # optional local machine paths (gitignored)
LU="$ROOT/native/cuda/lu"
OUT="$ROOT/target/lu-host-check"
mkdir -p "$OUT"
"${CXX:-g++}" -O2 -std=c++17 -Wall -Wextra -DLU_TRANSPOSED_FN=inline -I"$LU" -o "$OUT/lu_transposed_check" \
    -x c++ "$LU/scalar_lu_plan.cu" "$LU/supernodal_lu_plan.cu" "$LU/minimum_degree.cu" "$LU/pivot_delays.cu" \
    -x none "$ROOT/native/host/lu_transposed_check.cpp"
"$OUT/lu_transposed_check" "$@"
