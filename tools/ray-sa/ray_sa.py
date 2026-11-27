#!/usr/bin/env python3
"""Multi-machine AC security analysis: pypowsybl + OpenLoadFlow, distributed with Ray.

The driver solves the base load flow ONCE, ships the solved network (BIIDM bytes) through Ray's object
store, and hands chunks of contingencies to long-lived worker actors (one per machine by default, each
running OLF with threadCount = its CPUs). Workers start every chunk from the solved base
(voltage_init_mode=PREVIOUS_VALUES) so every pre-contingency result agrees, and chunks are pulled
dynamically (ActorPool) from a shuffled list so slow contingencies spread over the machines.

No shared filesystem is needed: the network, contingencies, actions and operator strategies all travel
through Ray; only the driver reads input files and writes outputs.

Cluster:   head:   ray start --head --num-cpus=<cores>
           others: ray start --address=<head>:6379 --num-cpus=<cores>
Run:       python ray_sa.py --address auto --network case.xiidm --out results/
Local try: python ray_sa.py --network case.xiidm --workers 2 --threads 2 --check
"""
import argparse
import io
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict

import pandas as pd
import pypowsybl as pp
import ray
from ray.util import ActorPool


# ---------------------------------------------------------------------------------------------- inputs

def contingencies_from_args(args, network):
    """The contingency list as powsybl contingency-list JSON dicts (one per contingency)."""
    if args.contingencies:
        with open(args.contingencies) as f:
            return json.load(f)["contingencies"]
    branches = network.get_branches()
    if args.n1 == "lines":
        branches = branches[branches["type"] == "LINE"]
    return [{"id": b, "elements": [{"id": b, "type": "BRANCH"}]} for b in branches.index]


def load_flow_parameters(args):
    if args.lf_params:
        with open(args.lf_params) as f:
            params = pp.loadflow.Parameters.from_json(f.read())
    else:
        params = pp.loadflow.Parameters()
    extra = dict(kv.split("=", 1) for kv in args.olf_param)
    if extra:
        params.provider_parameters = {**(params.provider_parameters or {}), **extra}
    return params


def strategies_by_contingency(path):
    """operator strategies grouped by the contingency they apply to (each goes with its contingency's chunk)."""
    if not path:
        return None, {}
    with open(path) as f:
        doc = json.load(f)
    grouped = defaultdict(list)
    for s in doc["operatorStrategies"]:
        if s.get("contingencyContextType", "SPECIFIC") != "SPECIFIC":
            sys.exit(f"operator strategy {s['id']}: only SPECIFIC contingency context is supported when splitting")
        grouped[s["contingencyId"]].append(s)
    return doc.get("version", "1.1"), grouped


# ---------------------------------------------------------------------------------------------- one run

def run_security_analysis(network, ctgs, strategies, strat_version, actions_json, sa_params, monitor_all, tmpdir):
    """One pypowsybl SA on `ctgs`; returns plain pandas/dicts (pypowsybl result objects do not pickle)."""
    sa = pp.security.create_analysis()
    ctg_file = os.path.join(tmpdir, "ctg.json")
    with open(ctg_file, "w") as f:
        json.dump({"type": "default", "version": "1.0", "name": "chunk", "contingencies": ctgs}, f)
    sa.add_contingencies_from_json_file(ctg_file)
    if strategies:
        act_file = os.path.join(tmpdir, "actions.json")
        with open(act_file, "w") as f:
            f.write(actions_json)
        sa.add_actions_from_json_file(act_file)
        strat_file = os.path.join(tmpdir, "strategies.json")
        with open(strat_file, "w") as f:
            json.dump({"version": strat_version, "operatorStrategies": strategies}, f)
        sa.add_operator_strategies_from_json_file(strat_file)
    if monitor_all:
        sa.add_monitored_elements(branch_ids=list(network.get_branches().index),
                                  voltage_level_ids=list(network.get_voltage_levels().index))

    r = sa.run_ac(network, parameters=sa_params, provider="OpenLoadFlow")

    statuses = [{"contingency_id": cid, "operator_strategy_id": "", "status": res.status.name}
                for cid, res in r.post_contingency_results.items()]
    strat_rows = []
    for sid, res in r.operator_strategy_results.items():
        statuses.append({"contingency_id": "", "operator_strategy_id": sid, "status": res.status.name})
        for v in res.limit_violations:
            strat_rows.append({"operator_strategy_id": sid, "subject_id": v.subject_id, "subject_name": v.subject_name,
                               "limit_type": v.limit_type.name, "limit_name": v.limit_name, "limit": v.limit,
                               "acceptable_duration": v.acceptable_duration, "limit_reduction": v.limit_reduction,
                               "value": v.value, "side": v.side.name})
    viol = r.limit_violations.reset_index()
    return {
        "pre_status": r.pre_contingency_result.status.name,
        "pre_violations": viol[viol["contingency_id"] == ""].reset_index(drop=True),
        "violations": viol[viol["contingency_id"] != ""],
        "statuses": pd.DataFrame(statuses, columns=["contingency_id", "operator_strategy_id", "status"]),
        "strategy_violations": pd.DataFrame(strat_rows),
        "branch_results": r.branch_results.reset_index(),
        "bus_results": r.bus_results.reset_index(),
    }


def security_parameters(lf_json, threads, thresholds):
    lf = pp.loadflow.Parameters.from_json(lf_json)
    lf.voltage_init_mode = pp.loadflow.VoltageInitMode.PREVIOUS_VALUES  # start from the solved base
    flow, kv, prop = thresholds
    inc = pp.security.IncreasedViolationsParameters(flow_proportional_threshold=flow,
                                                    low_voltage_proportional_threshold=prop,
                                                    low_voltage_absolute_threshold=kv,
                                                    high_voltage_proportional_threshold=prop,
                                                    high_voltage_absolute_threshold=kv)
    return pp.security.Parameters(load_flow_parameters=lf, increased_violations_parameters=inc,
                                  provider_parameters={"threadCount": str(threads)})


@ray.remote
class SaWorker:
    """Long-lived: loads the solved network once, then runs any number of chunks on it."""

    def __init__(self, network_bytes, lf_json, threads, thresholds, actions_json, strat_version, monitor_all):
        self.network = pp.network.load_from_binary_buffer(io.BytesIO(network_bytes))
        self.sa_params = security_parameters(lf_json, threads, thresholds)
        self.actions_json = actions_json
        self.strat_version = strat_version
        self.monitor_all = monitor_all
        self.host = os.uname().nodename

    def run(self, chunk_id, ctgs, strategies):
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            out = run_security_analysis(self.network, ctgs, strategies, self.strat_version, self.actions_json,
                                        self.sa_params, self.monitor_all, tmp)
        out.update(chunk_id=chunk_id, host=self.host, seconds=time.time() - t0, n=len(ctgs))
        return out


# ---------------------------------------------------------------------------------------------- merge / check

def merge(parts, ctg_order):
    pre = {p["pre_status"] for p in parts}
    if len(pre) != 1:
        raise RuntimeError(f"workers disagree on the pre-contingency status: {pre}")
    ref = parts[0]["pre_violations"]
    for p in parts[1:]:
        # same violations; values may differ in the last digits (OLF is not bit-reproducible run to run)
        pd.testing.assert_frame_equal(ref, p["pre_violations"], check_exact=False, rtol=1e-9,
                                      obj=f"pre-contingency violations of chunk {p['chunk_id']}")
    rank = {c: i for i, c in enumerate(ctg_order)}

    def cat(key):
        frames = [p[key] for p in parts if len(p[key])]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if "contingency_id" in df:
            df = df.assign(_r=df["contingency_id"].map(rank).fillna(-1)).sort_values(
                ["_r"] + [c for c in ("operator_strategy_id",) if c in df], kind="stable").drop(columns="_r")
        return df.reset_index(drop=True)

    pre_branch = [p["branch_results"][p["branch_results"]["contingency_id"] == ""] for p in parts[:1]]
    pre_bus = [p["bus_results"][p["bus_results"]["contingency_id"] == ""] for p in parts[:1]]
    for p in parts:  # keep one copy of the base rows
        for k in ("branch_results", "bus_results"):
            p[k] = p[k][p[k]["contingency_id"] != ""]
    return {
        "pre_status": pre.pop(),
        "violations": pd.concat([ref, cat("violations")], ignore_index=True),
        "statuses": cat("statuses"),
        "strategy_violations": cat("strategy_violations"),
        "branch_results": pd.concat(pre_branch + [cat("branch_results")], ignore_index=True),
        "bus_results": pd.concat(pre_bus + [cat("bus_results")], ignore_index=True),
    }


def compare(name, a, b, keys, rtol):
    """Same rows (by key) and same non-float columns exactly; floats within rtol (OLF is not bit-reproducible
    run to run, even single-threaded: last-ulp drift)."""
    if a.empty and b.empty:
        print(f"  {name}: both empty")
        return True
    if "value" in a.columns:
        # a voltage violation is keyed by its voltage level (no bus id in the dataframe): number the
        # duplicates of a key in value order so several buses of one level pair up
        def numbered(df):
            df = df.sort_values(keys + ["value"], kind="stable")
            return df.assign(_n=df.groupby(keys, dropna=False).cumcount())
        a, b, keys = numbered(a), numbered(b), keys + ["_n"]
    m = a.merge(b, on=keys, how="outer", indicator=True, suffixes=("_dist", "_single"))
    only = m[m["_merge"] != "both"]
    if len(only):
        print(f"  {name}: {len(only)} row(s) in one run only\n{only[keys + ['_merge']].head(20).to_string()}")
        return False
    ok, worst, worst_col = True, 0.0, None
    for c in (c for c in a.columns if c not in keys):
        x, y = m[f"{c}_dist"], m[f"{c}_single"]
        if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
            both_nan = x.isna() & y.isna()
            d = (x - y).abs()[~both_nan]
            if d.isna().any():
                ok = False
                print(f"  {name}.{c}: NaN on one side only")
            elif len(d) and d.max() > worst:
                worst, worst_col = d.max(), c
            tol = rtol * y.abs()[~both_nan].clip(lower=1.0)
            if (d > tol).any():
                ok = False
                print(f"  {name}.{c}: {(d > tol).sum()} value(s) beyond rtol {rtol}, max diff {d.max():.3e}")
        elif not (x.astype(str) == y.astype(str)).all():
            ok = False
            print(f"  {name}.{c}: differs")
    if ok:
        print(f"  {name}: same {len(a)} rows, max float diff {worst:.1e}" + (f" ({worst_col})" if worst_col else ""))
    return ok


# ---------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", required=True, help="any format pypowsybl loads")
    ap.add_argument("--contingencies", help="powsybl contingency-list JSON (default: N-1 on branches)")
    ap.add_argument("--n1", choices=["branches", "lines"], default="branches")
    ap.add_argument("--actions", help="actions JSON (needed with --strategies)")
    ap.add_argument("--strategies", help="operator strategies JSON (SPECIFIC contingency context only)")
    ap.add_argument("--lf-params", help="LoadFlowParameters JSON (OLF extension included)")
    ap.add_argument("--olf-param", action="append", default=[], metavar="KEY=VALUE", help="extra OLF provider parameter")
    ap.add_argument("--voltage-threshold", type=float, nargs=2, metavar=("KV", "PROP"),
                    help="increased-violation thresholds for low AND high voltage (absolute kV, proportional); powsybl "
                         "defaults both to 0, which re-reports a pre-existing voltage violation on ulp-level drift. "
                         "Set both legs: the effective threshold is min(abs, value*prop)")
    ap.add_argument("--flow-threshold", type=float, default=0.1, help="increased-violation flow proportional threshold")
    ap.add_argument("--monitor-all", action="store_true", help="branch + bus results for every branch/voltage level")
    ap.add_argument("--address", help="Ray cluster address ('auto' on a node of a running cluster); default: local Ray")
    ap.add_argument("--workers", type=int, help="actors (default: one per Ray node)")
    ap.add_argument("--threads", type=int, help="OLF threadCount per actor (default: CPUs of the smallest node / actors per node)")
    ap.add_argument("--chunk-size", type=int, default=500, help="contingencies per task")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed (-1: keep the input order)")
    ap.add_argument("--out", help="directory for the merged CSVs")
    ap.add_argument("--check", action="store_true", help="also run the whole list in one local SA and compare")
    ap.add_argument("--rtol", type=float, default=1e-9, help="--check float tolerance (relative, absolute below 1)")
    args = ap.parse_args()

    ray.init(address=args.address, runtime_env={"env_vars": {"PYTHONWARNINGS": "ignore"}}, log_to_driver=False)
    nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU", 0) > 0]
    workers = args.workers or len(nodes)
    per_node = max(1, -(-workers // len(nodes)))
    threads = args.threads or max(1, int(min(n["Resources"]["CPU"] for n in nodes)) // per_node)
    print(f"Ray: {len(nodes)} node(s), {workers} worker(s) x {threads} OLF threads")

    t0 = time.time()
    network = pp.network.load(args.network)
    lf_params = load_flow_parameters(args)
    base = pp.loadflow.run_ac(network, lf_params, provider="OpenLoadFlow")
    if base[0].status != pp.loadflow.ComponentStatus.CONVERGED:
        sys.exit(f"base load flow did not converge: {base[0].status}")
    print(f"base load flow: {base[0].status.name} in {base[0].iteration_count} iterations ({time.time() - t0:.1f} s)")
    network_bytes = network.save_to_binary_buffer("BIIDM").getvalue()

    ctgs = contingencies_from_args(args, network)
    strat_version, strategies = strategies_by_contingency(args.strategies)
    if strategies and not args.actions:
        sys.exit("--strategies needs --actions")
    actions_json = open(args.actions).read() if args.actions else None
    order = [c["id"] for c in ctgs]
    shuffled = list(ctgs)
    if args.seed >= 0:
        random.Random(args.seed).shuffle(shuffled)  # spread the expensive contingencies over the chunks
    chunks = [shuffled[i:i + args.chunk_size] for i in range(0, len(shuffled), args.chunk_size)]
    print(f"{len(ctgs)} contingencies in {len(chunks)} chunk(s) of <= {args.chunk_size}")

    if not args.voltage_threshold:
        print("WARNING: no --voltage-threshold: a voltage violation already present in the base case is re-reported "
              "after a contingency whenever the post value is even 1 ulp worse, so violation counts can flicker "
              "from run to run (OLF is not bit-reproducible). Example: --voltage-threshold 0.5 0.001")
    kv, prop = args.voltage_threshold or (0.0, 0.0)
    thresholds = (args.flow_threshold, kv, prop)
    net_ref = ray.put(network_bytes)
    actors = [SaWorker.options(num_cpus=threads, scheduling_strategy="SPREAD").remote(
        net_ref, lf_params.to_json(), threads, thresholds, actions_json, strat_version, args.monitor_all) for _ in range(workers)]
    pool = ActorPool(actors)
    t1 = time.time()
    parts = []
    tasks = [(i, c, [s for x in c for s in strategies.get(x["id"], [])]) for i, c in enumerate(chunks)]
    for p in pool.map_unordered(lambda a, t: a.run.remote(*t), tasks):
        parts.append(p)
        print(f"  chunk {p['chunk_id']:4d} on {p['host']}: {p['n']} ctg in {p['seconds']:.1f} s "
              f"({len(parts)}/{len(chunks)})")
    merged = merge(parts, order)
    t_dist = time.time() - t1
    counts = merged["statuses"]["status"].value_counts().to_dict()
    print(f"distributed SA: {t_dist:.1f} s (+ {t1 - t0:.1f} s setup), pre-contingency {merged['pre_status']}, "
          f"statuses {counts}, {len(merged['violations'])} violations")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for k, v in merged.items():
            if isinstance(v, pd.DataFrame) and not v.empty:
                v.to_csv(os.path.join(args.out, f"{k}.csv"), index=False)
        print(f"written to {args.out}/")

    if args.check:
        print("check: the same list in ONE SecurityAnalysis on the driver ...")
        t2 = time.time()
        sa_params = security_parameters(lf_params.to_json(), os.cpu_count(), thresholds)
        ref_net = pp.network.load_from_binary_buffer(io.BytesIO(network_bytes))
        with tempfile.TemporaryDirectory() as tmp:
            single = run_security_analysis(ref_net, ctgs, [s for v in strategies.values() for s in v], strat_version,
                                           actions_json, sa_params, args.monitor_all, tmp)
        print(f"  single run: {time.time() - t2:.1f} s")
        single = merge([dict(single, chunk_id=0)], order)
        ok = single["pre_status"] == merged["pre_status"]
        ok &= compare("statuses", merged["statuses"], single["statuses"], ["contingency_id", "operator_strategy_id"], args.rtol)
        ok &= compare("violations", merged["violations"], single["violations"],
                      ["contingency_id", "subject_id", "limit_type", "limit_name", "acceptable_duration", "side"], args.rtol)
        ok &= compare("strategy_violations", merged["strategy_violations"], single["strategy_violations"],
                      ["operator_strategy_id", "subject_id", "limit_type", "limit_name", "acceptable_duration", "side"], args.rtol)
        ok &= compare("branch_results", merged["branch_results"], single["branch_results"],
                      ["contingency_id", "operator_strategy_id", "branch_id"], args.rtol)
        ok &= compare("bus_results", merged["bus_results"], single["bus_results"],
                      ["contingency_id", "operator_strategy_id", "voltage_level_id", "bus_id"], args.rtol)
        print("CHECK PASSED" if ok else "CHECK FAILED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
