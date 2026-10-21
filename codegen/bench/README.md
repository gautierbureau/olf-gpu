# GPU batched N-1 benchmark harness

Measures the throughput of the test6 GPU batched security-analysis solver
(`GpuBatchedN1FixedIterTest`) on a single-branch N-1 workload. Reported numbers live
in `docs/archive/gpu_benchmark.md`.

1. **Generate a reproducible contingency set** (non-islanding, unique-pair branches
   straight from the IIDM network — needs `networkx`):

   ```
   python gen_contingencies.py <case9241pegase.xiidm.gz> <out_dir> [N]
   ```

   emits `<out_dir>/contingencies.txt` (one IIDM branch id per line). Optional — with
   no `OLF_GPU_CONT_FILE` the test just takes the first `MAX_CONTINGENCIES` branches.

2. **Run the benchmark:**

   ```
   OLF_GPU_CONT_FILE=<out_dir>/contingencies.txt \
   OLF_GPU_MAXBATCH=32 \
   mvn test -Dtest=GpuBatchedN1FixedIterTest
   ```

   Environment knobs (inherited by the surefire fork):
   - `OLF_GPU_CONT_FILE` — explicit contingency id list (unset → first N branches).
   - `OLF_GPU_MAXBATCH` — chunk/batch size (default 32; a conservative default keeps
     the test green on a 6 GB card).
   - `OLF_GPU_BATCHMODE` — `1` = uniform/lockstep batched cuDSS (single shared block,
     `CUDSS_CONFIG_UBATCH_SIZE`; cuDSS ≥ 0.8 required), `0` = per-scenario fallback
     (default). Both modes are compiled in and chosen at runtime via `-Dolf.gpu.batchMode`.
   - `OLF_GPU_ITERS` — fixed iteration count `K` for the bare-Newton throughput mode
     (device-side fixed-iter, `tol = 0`); unset uses the per-scenario retirement loop.

The test prints `scenarios/s` and `ms/scenario`.
