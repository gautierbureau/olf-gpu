"""pypowsybl run (pp_sa.py output) vs a harness CPU arm (cpu.json), same contingencies: statuses, violation counts,
violation keys as multisets, and values matched as SORTED lists per key (a voltage level with several buses carries
several violations under one key)."""
import json, sys, collections
def norm(x): return '' if x in (None, 'None', '', 'nan') else x
pp = json.load(open(sys.argv[1]))['post']; j = json.load(open(sys.argv[2]))
java = {}
for p in j['postContingencyResults']:
    d = collections.defaultdict(list)
    for v in p['limitViolationsResult']['limitViolations']:
        d[(v['subjectId'], v['limitType'], norm(str(v.get('side'))), norm(str(v.get('limitName'))))].append(v['value'])
    java[p['contingency']['id']] = (p['status'], d)
common = [c for c in java if c in pp]
st = sum(1 for c in common if pp[c][0] != java[c][0])
keyd = 0; bins = collections.Counter(); worst = (0.0, None); nj = npp = 0
for c in common:
    d = collections.defaultdict(list)
    for v in pp[c][1]:
        d[(v[0], v[1], norm(v[2]), norm(v[3]))].append(v[4])
    for k in set(d) | set(java[c][1]):
        a, b = sorted(d.get(k, [])), sorted(java[c][1].get(k, []))
        npp += len(a); nj += len(b)
        if len(a) != len(b):
            keyd += abs(len(a) - len(b)); continue
        for x, y in zip(a, b):
            dd = abs(x - y)
            bins['>1e-3' if dd > 1e-3 else '>1e-6' if dd > 1e-6 else '>1e-9' if dd > 1e-9 else '<=1e-9'] += 1
            if dd > worst[0]: worst = (dd, c, k[0], k[1])
print(f"common {len(common)}, status diffs {st}, violations pypowsybl {npp} fork {nj}, key-count diffs {keyd}")
print('value diffs', dict(bins), 'worst', worst)
