# structure-gap: how far apart are the CPU and GPU structures?

Throwaway. Phase 0 of `docs/design/fixed-structure-equation-system.tex`: before designing one fixed-structure
description for both solvers, measure the distance between

- OLF's **legacy** pattern (equations activated and deactivated),
- OLF's **alternative-equations** union pattern (the fork's `experiment/alt-eq-contingency-fallback`),
- the **GPU superset** pattern of this repository.

Delete this directory, the `OLF_GPU_STRUCTURE_DUMP` channel and the fork's `StructureGapDump` once the numbers
have served.

## How it works

Every pattern is reduced to a set of `(slot, column)` pairs in which both are variables written
`TYPE/elementId` (for example `BUS_V/VL-1001_0`), so two dumps compare by network element and not by matrix
index. The two sides can therefore run from different OLF builds.

- On the GPU a row already is a variable's slot.
- On the CPU a matrix column is an equation. `compare.py` gives each equation the variable whose slot it
  would occupy (P and the angle reference on `BUS_PHI`; Q, the voltage target and `DISTR_Q` on `BUS_V`; and
  so on). Equations that collide on a slot, or have no natural one, are placed by a matching restricted to the
  variables they depend on, and the tool reports how many were displaced or could not be placed.

## Running it

### CPU side (anywhere, no GPU)

In the OLF fork, on the branch that carries `StructureGapDump` (built on the alternative-equations branch):

```
./mvnw -q test-compile exec:java -Dexec.classpathScope=test \
    -Dexec.mainClass=com.powsybl.openloadflow.sa.StructureGapDump \
    -Dexec.args="case9241pegase target/structure-gap [parameters.json]"
```

The network is `ieee14`, `ieee57`, `ieee118`, `ieee300`, a MATPOWER Pegase case name (downloaded once) or a
path to a file powsybl can read. It writes three files per network (`legacy`, `alternative`, `alternative_sa`),
each with the pattern before and after a load flow, the eligibility census, the number of structure builds and
a factorisation timing.

```
tools/structure-gap/summarize.py <dump dir> [--markdown]
tools/structure-gap/compare.py <dump dir>/case9241pegase.legacy.json <dump dir>/case9241pegase.alternative_sa.json
```

### GPU side (the GPU machine)

```
OLF_GPU_STRUCTURE_DUMP=/tmp/gap/pegase tools/run_bench.sh      # or any run on the network of interest
```

writes, per structure size `n`, `/tmp/gap/pegase-n<n>.pattern` (native: `n nnz`, `rowMap`, `rowPtr`, `colIdx`)
and `/tmp/gap/pegase-n<n>.rows` (Java: the variable owning each row). A later structure of the same size
overwrites the file. Then:

```
tools/structure-gap/compare.py <dump dir>/case9241pegase.alternative_sa.json /tmp/gap/pegase-n18482 --out /tmp/gap/diff
```

**Not yet compiled or run.** The channel was written without a CUDA toolchain or the fork dependencies at
hand. The loader for this format was exercised on a synthetic dump only. One assumption to confirm on the
first real run: device row `r` is the slot of the variable at column `r` (the structure build forces a
diagonal on every row); `compare.py` reads `rowMap` but does not use it.

## First results (CPU side only, 2026-11-10)

Default load flow parameters, sparse matrix, one run in a shared container: the timings are indicative.



- `structure builds`: how many times the load flow built the matrix structure. Legacy rebuilds on each PV to PQ
  wave (4 on case9241pegase); alternative equations build once.
- `LU us`: median of 20 full LU decompositions (symbolic and numeric) of the solved pattern.

What the comparison shows on these networks:

- Legacy is a strict subset of the alternative-equations pattern, and the load flow only ever adds entries to
  the legacy one.
- Everything the union adds is on `BUS_V` rows: the reactive balance of buses still in voltage control. On
  case9241pegase that is 10 709 entries over the initial legacy pattern (7.1% of the union), of which the
  load flow's own switches to PQ would have added 2 189 anyway.
- The security-analysis variant (every bus can be disabled) adds no entry: the disabled alternatives sit on
  diagonals that already exist.
- The union costs a few percent more per factorisation on the large cases (+4% to +11% from case1354 up),
  against three structure rebuilds saved per load flow on case9241pegase. The small IEEE cases are too small
  to time.

What these networks cannot show: they have no remote or shared voltage control, no zero-impedance branch, no
transformer or shunt voltage control, so the eligibility census is empty and none of the fallbacks fire. The
numbers that decide the plan need a real network (rte6515 with the production parameters), on both sides.

## Still to measure

1. The GPU superset against the alternative-equations union, on pegase9241 and rte6515 (needs the GPU machine).
2. The census and the legacy fallbacks on rte6515: how many buses and controls are not eligible, and why.
3. Security analysis on the CPU with alternative equations: structure builds and fallbacks per contingency
   (`AlternativeEquationsBenchmark` in the fork already counts them).
4. The KLU cost of the GPU superset pattern itself, once dump 1 exists: feed it to a CPU factorisation.
