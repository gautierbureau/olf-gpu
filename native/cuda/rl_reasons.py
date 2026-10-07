#!/usr/bin/env python3
"""Why the ReactiveLimits outer loop costs what it does.
CPU (log with OLF_OL_STATS=1): every reactive-limit check's causes (CPU_RL_CHECK): a PV->PQ switch, a PQ->PV switch,
a limit update of a pinned bus, a reactive controller pinned; checks per contingency.
GPU (log with OLF_GPU_PHASE_TIME=1 OLF_GPU_RL_REASONS=1): the same causes per checked scenario (GPU_RL_REASONS) and
every reactive-limit re-solve pass (GPU_RL_PASS): active scenarios against time and batched iterations.
usage: rl_reasons.py cpu <log> | gpu <log>"""
import re, sys, collections
arm, log = sys.argv[1], sys.argv[2]
if arm == 'cpu':
    causes = collections.Counter(); per_ctg = collections.Counter(); unstable_per_ctg = collections.Counter(); n = 0
    sizes = collections.Counter()
    for l in open(log, errors='replace'):
        m = re.match(r'CPU_RL_CHECK ctg=(\S+) iter=(\d+) pvToPq=(\d+)(\*?) pqToPv=(\d+)(\*?) qLimitUpdates=(\d+) reactiveCtrlToPq=(\d+)(\*?) status=(\S+)', l)
        if not m: continue
        n += 1; c = m.group(1); per_ctg[c] += 1
        parts = [name for name, on in (('PV->PQ', m.group(4)), ('PQ->PV', m.group(6)), ('limit update', m.group(7) != '0'),
                                       ('reactive ctrl->PQ', m.group(9))) if on]
        if m.group(10) == 'UNSTABLE':
            unstable_per_ctg[c] += 1
            causes[' + '.join(parts) if parts else '(unstable, no cause?)'] += 1
            if m.group(4): sizes['PV->PQ buses per switching check: ' + ('1' if m.group(3) == '1' else '2-3' if int(m.group(3)) <= 3 else '4+')] += 1
        else:
            causes['stable'] += 1
    print(f"CPU: {n} reactive-limit checks over {len(per_ctg)} contingencies "
          f"({sum(unstable_per_ctg.values())} unstable = re-solves)")
    for k, v in causes.most_common(): print(f"  {v:7d}  {k}")
    for k, v in sorted(sizes.items()): print(f"  {v:7d}  {k}")
    h = collections.Counter(unstable_per_ctg[c] for c in per_ctg)
    print("  unstable checks per contingency:", dict(sorted(h.items())))
else:
    reasons = [l.strip() for l in open(log, errors='replace') if l.startswith('GPU_RL_REASONS')]
    k = max((i for i, l in enumerate(reasons) if 'stable checks=' in l), default=None)
    print("GPU reactive-limit check causes (cumulative, last print):")
    if k is not None:
        for l in reasons[k:]:
            if 'stable checks=' in l and l is not reasons[k]: break
            print("  " + l.replace('GPU_RL_REASONS ', ''))
    passes = []
    for l in open(log, errors='replace'):
        m = re.match(r'GPU_RL_PASS outer=(\d+) width=(\d+) active=(\d+) ms=(\S+) batchIters=(\d+) laneIters=(\d+) widthIters=(\d+)', l)
        if m: passes.append(tuple(float(x) for x in m.groups()))
    buckets = collections.defaultdict(list)
    for p in passes:
        frac = p[2] / p[1]
        b = '<1%' if frac < 0.01 else '1-10%' if frac < 0.1 else '10-50%' if frac < 0.5 else '>=50%'
        buckets[b].append(p)
    print(f"GPU reactive-limit re-solve passes: {len(passes)}, {sum(p[3] for p in passes) / 1000:.1f} s")
    print("  active share of the chunk | passes | time | avg active | avg ms/pass | avg batched iterations/pass")
    for b in ['>=50%', '10-50%', '1-10%', '<1%']:
        ps = buckets.get(b, [])
        if not ps: continue
        print(f"  {b:>6} | {len(ps):4d} | {sum(p[3] for p in ps) / 1000:6.1f} s | {sum(p[2] for p in ps) / len(ps):7.0f} | "
              f"{sum(p[3] for p in ps) / len(ps):7.0f} | {sum(p[4] for p in ps) / len(ps):5.1f}")
