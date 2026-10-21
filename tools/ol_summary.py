#!/usr/bin/env python3
"""Outer-loop summary of one SA run, in the shape: reportedContingencies, contingenciesTriggeringOuterLoops,
totalOuterLoopIterations, outerLoopIterationsByType, outerLoopIterationHistogram, plus the Newton split (first inner
solve vs outer-loop re-solves). CPU from OLF's CPU_OL_STATS lines (OLF_OL_STATS=1); GPU from GPU_OL_STATS lines
(OLF_OL_STATS=1) and, for the types, the GPU_OL_TIME re-solve lines (OLF_GPU_PHASE_TIME=1).
usage: ol_summary.py cpu <log> | gpu <log>"""
import json, re, sys, collections
arm, log = sys.argv[1], sys.argv[2]
hist = collections.Counter(); by_type = collections.Counter(); n = trig = tot = first = total = 0
if arm == 'cpu':
    for l in open(log, errors='replace'):
        m = re.match(r'CPU_OL_STATS ctg=(\S+) firstNr=(\d+) totalNr=(\d+) outer=(\d+) status=CONVERGED loops=(\S*)', l)
        if not m: continue
        n += 1; o = int(m.group(4)); hist[o] += 1; tot += o; trig += o > 0
        first += int(m.group(2)); total += int(m.group(3))
        for t in m.group(5).split(','):
            if t: name, v = t.split(':'); by_type[name] += int(v.split('/')[0])
else:
    last = {}
    for l in open(log, errors='replace'):
        m = re.match(r'GPU_OL_STATS ctg=(\S+) passes=(\d+) firstNr=(\d+) totalNr=(\d+) status=(\S+)', l)
        if m: last[m.group(1)] = (int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5))
        m = re.match(r'GPU_OL_TIME re-solve trigger=(\S+) passes=\d+ ms=\S+ scenarioSolves=(\d+)', l)
        if m: by_type[m.group(1)] = int(m.group(2))   # cumulative print: the last one wins
    for c, (p, f, t, status) in last.items():
        if status != '0': continue
        n += 1; o = p - 1; hist[o] += 1; tot += o; trig += o > 0; first += f; total += t
print(json.dumps({"arm": arm, "reportedContingencies": n, "contingenciesTriggeringOuterLoops": trig,
                  "totalOuterLoopIterations": tot, "outerLoopIterationsByType": dict(by_type.most_common()),
                  "outerLoopIterationHistogram": {str(k): hist[k] for k in sorted(hist)},
                  "newtonIterations": {"firstInnerSolve": first, "outerLoopReSolves": total - first, "total": total}},
                 indent=2))
