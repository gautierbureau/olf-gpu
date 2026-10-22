#!/usr/bin/env python3
"""Why the ReactiveLimits outer loop costs what it does.
CPU (log with OLF_OL_STATS=1): every reactive-limit check's causes (CPU_RL_CHECK): a PV->PQ switch, a PQ->PV switch,
a limit update of a bus switched to PQ, a reactive power controller switched to PQ; checks per contingency.
usage: rl_reasons.py cpu <log>"""
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
    sys.exit(__doc__)
