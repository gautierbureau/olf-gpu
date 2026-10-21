# GPU Jacobian assembly — design note

How the codegen feeds the GPU AC power flow: the CPU/GPU split and the one place the
scatter accumulates. The production implementation is
`native/cuda/jni/olf_gpu_nr_jni.cu` (→ `libolfgpu`); companion docs:
`codegen/gpu_roadmap.md`, `codegen/gpu_rte_full_support.md`. Strategy context (keep OLF's
branch formulation, borrow the structural machinery) is on the
`claude/codegen-experimental` branch (`codegen/experimental/gpu_fit.md`).

## The split: structure once on CPU, values on GPU per iteration

| Phase | Where | When | What |
|---|---|---|---|
| build `J` structure | **CPU** | **once / topology** | fixed CSR pattern + per-branch scatter map (each of a branch's 28 outputs → flat slot in the `J` value array / `F` row) |
| cuDSS analysis | CPU-driven | **once** | symbolic factorization of that fixed pattern |
| fill `J`/`F` values | **GPU** | **every NR iter** | one thread per branch: run the fused `updateClosedBranch` (28 values), scatter into the prebuilt slots |
| cuDSS refactor + solve | GPU | every iter | numeric re-LU on the analyzed pattern, then `J·Δx = -g` |

The pattern is invariant across iterations **and** across the N-1 batch (a branch trip is
a structure-preserving downdate, `supp(J^(c)) ⊆ supp(J)`), so structure + maps + cuDSS
analysis are paid once. The CPU computes **no values** per iteration.

This is the standard structure-once / fill-per-iter GPU pattern: a host routine (once)
builds the CSR pattern + per-entry scatter maps; a device fill kernel (per iter) scatters
values via those maps; cuDSS analyses once.

## A property of the scatter: it accumulates

A Ybus-keyed scatter keys off **Ybus nonzeros**, and Ybus has already aggregated the
per-branch stamps, so each `k → J-slot` is **unique** → the fill is an atomic-free write
(duplicate-index scatter is UB).

We key off **branches**, and OLF does not pre-aggregate into a Ybus, so a bus-diagonal
entry (e.g. `∂P_i/∂V_i`) is a **sum over incident branches**. On the GPU that `+=` must be
a real accumulation. The kernel uses **`atomicAdd` into `J_values`** (one thread per
branch; double atomics are cheap on modern GPUs, and the per-branch scatter map is built
once on the CPU) — the least-code option that matches the sum-over-incident-branches
directly. The alternative (a per-`J`-slot gather / segment-reduce, atomic-free but needing
a per-slot contribution list) is only worth it if diagonal contention shows up in
profiling; it has not. Either way the *structure + maps* are built once on the CPU; only
the write discipline differs.

## Status — implemented

The full device-resident Newton-Raphson lives in `native/cuda/jni/olf_gpu_nr_jni.cu`
(→ `libolfgpu`): the host builds the fixed-pattern CSR + the per-output scatter *dest*
arrays once, the four fused kernels (closed/open branch, shunt, HVDC) fill `J`/`F` on
device via `atomicAdd`, and cuDSS refactorizes + solves each iteration. OLF stores its
matrix as `M[variable.getRow()][equation.getColumn()]` (rows = variables, cols =
equations); the host-built dest arrays fold that layout in so the kernels never index it.
The batched N-1 path adds a per-scenario disabled mask + rowMode over the one shared
pattern (`GpuBatchedSecurityAnalysisSolver`).

Validated against OLF's own `JacobianMatrix` Newton by `GpuFullNewtonTest` and end to end
by `GpuLoadFlowTest` (run via `native/run_gpu_tests.sh`). Earlier CPU-reference and
single-system GPU spike prototypes were removed once the device path subsumed them; see
`codegen/gpu_roadmap.md` for the rung history.
