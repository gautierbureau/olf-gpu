"""Two powsybl SA result JSONs, per contingency: violation keys (multisets) and values matched as sorted lists per
key; absolute and relative value differences."""
import json, sys, collections
def norm(x): return '' if x in (None, 'None', '', 'nan') else x
def load(f):
    out = {}
    for p in json.load(open(f))['postContingencyResults']:
        d = collections.defaultdict(list)
        for v in p['limitViolationsResult']['limitViolations']:
            d[(v['subjectId'], v['limitType'], norm(str(v.get('side'))), norm(str(v.get('limitName'))))].append(v['value'])
        out[p['contingency']['id']] = (p['status'], d)
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
common = [c for c in a if c in b]
keyd = 0; n = 0; same = 0; worst_abs = (0.0, None); worst_rel = (0.0, None); bins = collections.Counter()
for c in common:
    for k in set(a[c][1]) | set(b[c][1]):
        x, y = sorted(a[c][1].get(k, [])), sorted(b[c][1].get(k, []))
        if len(x) != len(y): keyd += abs(len(x) - len(y)); continue
        for u, w in zip(x, y):
            n += 1; d = abs(u - w); r = d / max(abs(u), abs(w), 1e-300)
            same += d == 0
            bins['0' if d == 0 else '<1e-12 rel' if r < 1e-12 else '<1e-9 rel' if r < 1e-9 else '<1e-6 rel' if r < 1e-6 else '>=1e-6 rel'] += 1
            if d > worst_abs[0]: worst_abs = (d, c, k)
            if r > worst_rel[0]: worst_rel = (r, c, k)
print(f"violations compared {n}, key-count differences {keyd}, bit-identical {same}")
print("relative difference bins", dict(bins))
print(f"worst absolute {worst_abs[0]:.3e} at {worst_abs[1]} {worst_abs[2]}")
print(f"worst relative {worst_rel[0]:.3e} at {worst_rel[1]} {worst_rel[2]}")
