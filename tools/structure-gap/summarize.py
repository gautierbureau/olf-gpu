#!/usr/bin/env python3
"""THROWAWAY: one table from the OLF fork's StructureGapDump files.

Usage: summarize.py <dump directory> [--markdown]

Per network and modeling: system size, pattern entries before and after the load flow, how many times the load
flow rebuilt the matrix structure, and the median time of a full LU decomposition on the solved pattern. Then
the eligibility census of the alternative equations, for the networks where something is not eligible.
"""
import glob
import json
import os
import sys

ORDER = {'LEGACY': 0, 'ALTERNATIVE': 1, 'ALTERNATIVE_SA': 2}


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    markdown = '--markdown' in sys.argv
    dumps = []
    for path in glob.glob(os.path.join(sys.argv[1], '*.json')):
        with open(path) as f:
            dumps.append(json.load(f))
    dumps.sort(key=lambda d: (d['initial']['n'], d['network'], ORDER[d['modeling']]))

    header = ['network', 'modeling', 'n', 'nnz initial', 'nnz solved', 'vs legacy solved', 'structure builds',
              'pattern stable', 'LU us (solved)', 'vs legacy']
    rows = []
    legacy = {d['network']: d for d in dumps if d['modeling'] == 'LEGACY'}
    for d in dumps:
        i, s = d['initial'], d['solved']
        ref = legacy.get(d['network'])
        nnz_ratio = lu_ratio = ''
        if ref is not None and d is not ref:
            nnz_ratio = '%+.1f%%' % (100.0 * (s['nnz'] - ref['solved']['nnz']) / ref['solved']['nnz'])
            lu_ratio = '%+.0f%%' % (100.0 * (s['factorizationMicroseconds'] - ref['solved']['factorizationMicroseconds'])
                                    / max(1, ref['solved']['factorizationMicroseconds']))
        modeling = d['modeling']
        if d['alternativeEquationsRequested'] and not d['alternativeEquationsEffective']:
            modeling += ' (fell back to legacy)'
        rows.append([d['network'], modeling, i['n'], i['nnz'], s['nnz'], nnz_ratio, d['structureBuildCount'],
                     'yes' if d['patternUnchangedByLoadFlow'] else 'no', s['factorizationMicroseconds'], lu_ratio])

    if markdown:
        print('| ' + ' | '.join(header) + ' |')
        print('|' + '---|' * len(header))
        for r in rows:
            print('| ' + ' | '.join(str(x) for x in r) + ' |')
    else:
        widths = [max(len(str(x)) for x in [h] + [r[k] for r in rows]) for k, h in enumerate(header)]
        for r in [header] + rows:
            print('  '.join(str(x).ljust(w) for x, w in zip(r, widths)))

    print()
    for d in dumps:
        if d['modeling'] != 'ALTERNATIVE_SA':
            continue
        c = d['census']
        print('%s: %d of %d buses not eligible %s; %d of %d generator voltage controls not eligible %s '
              '(%d remote, %d shared); %d zero-impedance branches'
              % (d['network'], c['busesNotEligible'], c['buses'], c['busesNotEligibleByReason'],
                 c['generatorVoltageControlsNotEligible'], c['generatorVoltageControls'],
                 c['generatorVoltageControlsNotEligibleByReason'], c['generatorVoltageControlsRemote'],
                 c['generatorVoltageControlsShared'], c['zeroImpedanceBranches']))


if __name__ == '__main__':
    main()
