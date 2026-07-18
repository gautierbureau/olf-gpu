#!/usr/bin/env python3
"""Capability coverage counter over docs/gpu-vs-cpu-capabilities.md.

Counts every capability table row by its status mark (first cell that starts with one).
"GPU now" = full (v) + partial (~ / warn); total excludes N/A rows. Prints a per-section
breakdown and a one-line summary used by native/track-numbers.sh for docs/numbers.md.
"""
import re
import sys
import pathlib

DOC = pathlib.Path(__file__).resolve().parent.parent / 'docs' / 'gpu-vs-cpu-capabilities.md'
FULL, PARTIAL, DELTA, FALLBACK, REJECT, NA = '✅', '\U0001f536', '⚠️', '⏩', '❌', '—'
MARKS = [FULL, PARTIAL, DELTA, FALLBACK, REJECT, NA]

section = 'preamble'
skip = False
counts = {}
for line in DOC.read_text().splitlines():
    if line.startswith('#'):
        title = line.lstrip('#').strip()
        skip = 'Legend' in title
        if line.startswith('## ') or line.startswith('# '):
            section = title
        continue
    if skip or not line.strip().startswith('|') or re.match(r'^\s*\|[-\s|]+\|\s*$', line):
        continue
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    mark = next((m for m in MARKS for c in cells if c.startswith(m)), None)
    if mark is None:
        continue
    counts.setdefault(section, {m: 0 for m in MARKS})[mark] += 1

tot = {m: sum(s[m] for s in counts.values()) for m in MARKS}
gpu = tot[FULL] + tot[PARTIAL] + tot[DELTA]
total = sum(tot.values()) - tot[NA]
print(f'{"Section":58s} {"full":>4} {"part":>4} {"delta":>5} {"cpu":>4} {"rej":>4} {"n/a":>4}')
for sec, s in counts.items():
    print(f'{sec[:58]:58s} {s[FULL]:4d} {s[PARTIAL]:4d} {s[DELTA]:5d} {s[FALLBACK]:4d} {s[REJECT]:4d} {s[NA]:4d}')
print(f'{"TOTAL":58s} {tot[FULL]:4d} {tot[PARTIAL]:4d} {tot[DELTA]:5d} {tot[FALLBACK]:4d} {tot[REJECT]:4d} {tot[NA]:4d}')
print(f'GPU-covered (full+partial+delta): {gpu}/{total} ({100*gpu/total:.0f}%)')
if '--summary' in sys.argv:
    pass
