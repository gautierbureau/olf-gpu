# Multi-machine OLF security analysis with pypowsybl + Ray

`ray_sa.py` (stock OLF on the CPU — not the GPU extension) splits one AC security analysis (OpenLoadFlow) over the machines of a Ray cluster and merges the
results into the dataframes a single `run_ac` gives.

How it works:

1. The driver solves the base load flow once and sends the solved network (BIIDM bytes) through Ray's object store,
   so no shared filesystem is needed.
2. One long-lived actor per node loads the network once. Each actor runs OLF with `threadCount` = the node's CPUs and
   starts every SA from the solved base (`PREVIOUS_VALUES`).
3. Contingencies are shuffled and cut into chunks (`--chunk-size`). Actors pull chunks as they free up, so slow
   contingencies spread across the nodes.
4. Each operator strategy is sent with its contingency's chunk (SPECIFIC context only). Actions go to every chunk.
5. The coordinator merges the results. It checks that every chunk gives the same pre-contingency result and keeps
   one copy of the pre-contingency rows.

## Run

```
pip install pypowsybl ray pandas            # same versions on every node
ray start --head --num-cpus=32              # head node
ray start --address=<head>:6379 --num-cpus=32   # each other node
python tools/ray-sa/ray_sa.py --address auto --network case.xiidm \
    [--contingencies ctg.json] [--actions a.json --strategies s.json] \
    [--lf-params lf.json] [--olf-param key=value ...] [--monitor-all] \
    --voltage-threshold 0.5 0.001 --out results/ [--check]
```

`--check` runs the same list in one SA on the driver and compares the results. Statuses and row sets must match
exactly; floats must agree within `--rtol` (1e-9).

## Two things the check found

- **Set `--voltage-threshold`.** powsybl's increased-violation voltage thresholds default to 0. A voltage
  violation that already exists in the base case is then re-reported after a contingency as soon as the post
  value is 1 ulp worse. On ieee300 with tight limits, one violation (VL9022 after L224-225-1) came and went between
  two identical single-thread runs. Set both legs: the effective threshold is `min(abs, value*prop)`.
- **OLF is not bit-reproducible from run to run.** Even single-threaded in one process, repeated runs differ by up to
  4e-10 A on currents and 1e-12 on angles. That is why the check uses a tolerance.

Tested on a 2-node Ray cluster (both nodes on one 4-core VM): ieee300, 411 contingencies, 43 operator strategies,
monitor-all. 3/3 passes with thresholds set. Not timed at scale.

With Ray 2.60 (the version tested) token authentication, start every node and the driver with the same `RAY_AUTH_MODE`
(e.g. `export RAY_AUTH_MODE=token`) and the cluster's token.
