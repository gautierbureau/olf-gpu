#!/usr/bin/env python3
"""THROWAWAY: compare two Jacobian sparsity patterns named by network element.

Phase 0 of docs/design/fixed-structure-equation-system.tex: how far apart are OLF's legacy pattern, OLF's
alternative-equations union pattern and the GPU superset pattern?

A pattern source is one of
  <file>.json[:initial|:solved]   a dump of the OLF fork's StructureGapDump (default: solved)
  <prefix>-n<n>                   a GPU dump: <prefix>-n<n>.pattern (native) + <prefix>-n<n>.rows (java),
                                  written when OLF_GPU_STRUCTURE_DUMP=<prefix> is set

Both are reduced to the same thing: a set of (slot, column) pairs where a slot and a column are variables,
written TYPE/elementId. On the GPU a row IS a variable's slot. On the CPU a matrix column is an equation, so
each equation is first given the variable whose slot it would occupy (P and the angle reference on BUS_PHI,
Q, the voltage target and DISTR_Q on BUS_V, ...); equations that collide on a slot, or have no natural one,
are placed by a matching restricted to the variables they actually depend on.

Usage: compare.py A B [--out DIR] [--examples N]
"""
import argparse
import collections
import json
import os
import sys

# the variable whose slot an OLF equation naturally occupies, by primary equation type
NATURAL_SLOT = {
    'BUS_TARGET_P': 'BUS_PHI', 'BUS_TARGET_PHI': 'BUS_PHI', 'BUS_DISTR_SLACK_P': 'BUS_PHI',
    'BUS_TARGET_Q': 'BUS_V', 'BUS_TARGET_V': 'BUS_V', 'DISTR_Q': 'BUS_V',
    'BRANCH_TARGET_RHO1': 'BRANCH_RHO1', 'DISTR_RHO': 'BRANCH_RHO1',
    'BRANCH_TARGET_ALPHA1': 'BRANCH_ALPHA1',
    'SHUNT_TARGET_B': 'SHUNT_B', 'DISTR_SHUNT_B': 'SHUNT_B',
    'ZERO_PHI': 'DUMMY_P', 'DUMMY_TARGET_P': 'DUMMY_P',
    'ZERO_V': 'DUMMY_Q', 'DUMMY_TARGET_Q': 'DUMMY_Q',
}
# when two equations want the same slot, the one listed first keeps it
PRIORITY = ['BUS_TARGET_P', 'BUS_TARGET_Q', 'BUS_TARGET_PHI', 'BUS_TARGET_V', 'DISTR_Q']


class Pattern:
    def __init__(self, label):
        self.label = label
        self.variables = set()          # TYPE/id
        self.entries = set()            # (slot variable, column variable)
        self.slot_content = {}          # slot variable -> what sits there (CPU: active equation; GPU: '')
        self.unplaced = []              # CPU equations the matching could not place
        self.displaced = 0              # CPU equations not on their natural slot


def load_cpu(path, snapshot):
    with open(path) as f:
        d = json.load(f)
    s = d[snapshot]
    p = Pattern('%s %s (%s)' % (d['network'], d['modeling'], snapshot))
    var_of_row = {}
    for row, vtype, vid in s['variables']:
        var_of_row[row] = '%s/%s' % (vtype, vid)
    p.variables = set(var_of_row.values())
    n = s['n']
    column_start, row_indices = s['columnStart'], s['rowIndices']
    rows_of = [row_indices[column_start[c]:column_start[c + 1]] for c in range(n)]
    equations = {c: (etype, eid, atype, aid) for c, etype, eid, atype, aid in s['equations']}

    # natural placement, by priority, then a matching for the rest
    slot_of = {}                        # column -> variable
    taken = {}                          # variable -> column
    order = sorted(range(n), key=lambda c: PRIORITY.index(equations[c][0]) if equations[c][0] in PRIORITY else len(PRIORITY))
    leftover = []
    for c in order:
        etype, eid = equations[c][0], equations[c][1]
        natural = '%s/%s' % (NATURAL_SLOT[etype], eid) if etype in NATURAL_SLOT else None
        if natural in p.variables and natural not in taken:
            slot_of[c] = natural
            taken[natural] = c
        else:
            leftover.append(c)

    def augment(c, seen):
        for r in rows_of[c]:
            v = var_of_row[r]
            if v in seen:
                continue
            seen.add(v)
            if v not in taken or augment(taken[v], seen):
                taken[v] = c
                slot_of[c] = v
                return True
        return False

    sys.setrecursionlimit(max(10000, 4 * n))
    for c in leftover:
        if not augment(c, set()):
            p.unplaced.append(equations[c])
    for c, v in slot_of.items():
        etype, eid, atype, aid = equations[c]
        natural = '%s/%s' % (NATURAL_SLOT.get(etype, '?'), eid)
        if v != natural:
            p.displaced += 1
        p.slot_content[v] = '%s/%s' % (atype, aid)
        for r in rows_of[c]:
            p.entries.add((v, var_of_row[r]))
    return p, d


def load_gpu(prefix):
    var_of_row = {}
    with open(prefix + '.rows') as f:
        for line in f:
            row, vtype, vid = line.rstrip('\n').split('\t')
            var_of_row[int(row)] = '%s/%s' % (vtype, vid)
    with open(prefix + '.pattern') as f:
        n, nnz = (int(x) for x in f.readline().split())
        f.readline()                    # rowMap: OLF row -> device row; device row r is paired with column r
        row_ptr = [int(x) for x in f.readline().split()]
        col_idx = [int(x) for x in f.readline().split()]
    if len(var_of_row) != n or len(row_ptr) != n + 1 or len(col_idx) != nnz:
        raise SystemExit('%s: .rows and .pattern disagree (n=%d, %d row owners, nnz=%d, %d columns)'
                         % (prefix, n, len(var_of_row), nnz, len(col_idx)))
    p = Pattern('GPU superset %s' % os.path.basename(prefix))
    p.variables = set(var_of_row.values())
    for r in range(n):
        for i in range(row_ptr[r], row_ptr[r + 1]):
            p.entries.add((var_of_row[r], var_of_row[col_idx[i]]))
    return p, None


def load(source):
    if '.json' in source:
        path, _, snapshot = source.partition(':')
        return load_cpu(path, snapshot or 'solved')
    return load_gpu(source)


def vtype(v):
    return v.split('/', 1)[0]


def breakdown(entries, slot_content=None):
    counts = collections.Counter()
    for slot, col in entries:
        key = '%s row, %s column' % (vtype(slot), vtype(col))
        if slot_content is not None:
            key += '   [slot holds %s]' % vtype(slot_content.get(slot, '?'))
        counts[key] += 1
    return counts.most_common()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--out', help='directory for the full difference lists (TSV)')
    ap.add_argument('--examples', type=int, default=5)
    args = ap.parse_args()

    (a, _), (b, _) = load(args.a), load(args.b)
    print('A: %s' % a.label)
    print('B: %s' % b.label)
    print()
    print('%-34s %10s %10s' % ('', 'A', 'B'))
    print('%-34s %10d %10d' % ('variables (= slots)', len(a.variables), len(b.variables)))
    print('%-34s %10d %10d' % ('pattern entries', len(a.entries), len(b.entries)))
    for p, name in ((a, 'A'), (b, 'B')):
        if p.displaced or p.unplaced:
            print('%s: %d equations placed off their natural slot, %d not placed at all' % (name, p.displaced, len(p.unplaced)))
            for e in p.unplaced[:args.examples]:
                print('     unplaced: %s/%s (active %s/%s)' % e)

    only_a_vars, only_b_vars = a.variables - b.variables, b.variables - a.variables
    if only_a_vars or only_b_vars:
        print()
        print('variables only in A: %d %s' % (len(only_a_vars), dict(collections.Counter(vtype(v) for v in only_a_vars))))
        print('variables only in B: %d %s' % (len(only_b_vars), dict(collections.Counter(vtype(v) for v in only_b_vars))))

    common = a.entries & b.entries
    only_a, only_b = a.entries - b.entries, b.entries - a.entries
    print()
    print('entries in both      %10d' % len(common))
    print('entries only in A    %10d   (%.1f%% of A)' % (len(only_a), 100.0 * len(only_a) / max(1, len(a.entries))))
    print('entries only in B    %10d   (%.1f%% of B)' % (len(only_b), 100.0 * len(only_b) / max(1, len(b.entries))))
    print('A is a subset of B: %s      B is a subset of A: %s' % (not only_a, not only_b))

    for name, only, p in (('A', only_a, a), ('B', only_b, b)):
        if not only:
            continue
        print()
        print('only in %s, by kind:' % name)
        for key, count in breakdown(only, p.slot_content if p.slot_content else None):
            print('  %8d  %s' % (count, key))
        for slot, col in sorted(only)[:args.examples]:
            print('     e.g. row %s, column %s' % (slot, col))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for name, only, p in (('only_in_a', only_a, a), ('only_in_b', only_b, b)):
            with open(os.path.join(args.out, name + '.tsv'), 'w') as f:
                f.write('slot\tcolumn\tslot_holds\n')
                for slot, col in sorted(only):
                    f.write('%s\t%s\t%s\n' % (slot, col, p.slot_content.get(slot, '')))
        print()
        print('full lists written to %s' % args.out)


if __name__ == '__main__':
    main()
