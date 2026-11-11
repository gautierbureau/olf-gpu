#!/usr/bin/env python3
"""Report repeated code blocks: the numbers behind docs/design/duplication-cleanup-plan.md.

Usage: clone_scan.py <root> <extensions> [--window N] [--shape] [--blocks FILE...]

  clone_scan.py native/cuda .cu,.cuh,.h
  clone_scan.py src/main/java .java
  clone_scan.py native/cuda .cu,.cuh,.h --blocks native/cuda/lu/scalar_lu.cu

Comments, blank lines, includes and imports are dropped, whitespace is collapsed, and a line counts as repeated
when it lies in a window of N consecutive code lines (default 6) that occurs more than once under the root.
Directories named "generated" are skipped. --shape also replaces identifiers and numbers, which finds code of
the same shape under different names; read that figure as an upper bound, it also counts look-alike boilerplate.
--blocks lists, for the given files, their longest repeated runs and where the other copy is.
"""
import argparse
import collections
import hashlib
import os
import re


def code_lines(path, shape):
    raw = open(path, encoding='utf-8', errors='replace').read()
    raw = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), raw, flags=re.S)
    out = []
    for number, line in enumerate(raw.split('\n'), 1):
        line = re.sub(r'//.*', '', line).strip()
        if shape:
            line = re.sub(r'\d+', 'N', re.sub(r'[A-Za-z_]\w*', 'I', line))
        line = re.sub(r'\s+', ' ', line)
        if len(line) > 3 and not line.startswith(('#include', 'import ', 'package ')):
            out.append((number, line))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('extensions')
    ap.add_argument('--window', type=int, default=6)
    ap.add_argument('--shape', action='store_true')
    ap.add_argument('--blocks', nargs='*', default=[])
    args = ap.parse_args()
    extensions = tuple(args.extensions.split(','))
    w = args.window

    files = {}
    for directory, _, names in os.walk(args.root):
        if 'generated' in directory.split(os.sep):
            continue
        for name in names:
            if name.endswith(extensions):
                path = os.path.join(directory, name)
                files[path] = code_lines(path, args.shape)

    windows = collections.defaultdict(list)
    for path, lines in files.items():
        for i in range(len(lines) - w + 1):
            key = hashlib.md5('\n'.join(text for _, text in lines[i:i + w]).encode()).digest()
            windows[key].append((path, i))

    repeated = collections.defaultdict(dict)        # path -> line index -> (other path, other line number)
    pairs = collections.Counter()
    for occurrences in windows.values():
        if len(occurrences) < 2:
            continue
        for path, i in occurrences:
            other = next((q, files[q][j][0]) for q, j in occurrences if (q, j) != (path, i))
            for k in range(w):
                repeated[path].setdefault(i + k, other if k == 0 else None)
        involved = sorted({p for p, _ in occurrences})
        for a in involved:
            for b in involved:
                if a <= b:
                    pairs[(a, b)] += 1

    total = sum(len(v) for v in files.values())
    dup = sum(len(v) for v in repeated.values())
    rel = lambda p: os.path.relpath(p, args.root)
    print('%s: %d files, %d code lines, %d in a repeated %d-line window (%.1f%%)%s'
          % (args.root, len(files), total, dup, w, 100.0 * dup / max(1, total), ' [shape]' if args.shape else ''))

    by_dir, dir_total = collections.Counter(), collections.Counter()
    for path, lines in files.items():
        d = os.path.dirname(rel(path)) or '.'
        dir_total[d] += len(lines)
        by_dir[d] += len(repeated.get(path, ()))
    print('by directory:')
    for d, count in by_dir.most_common(12):
        if count:
            print('  %6d / %6d  %5.1f%%  %s' % (count, dir_total[d], 100.0 * count / dir_total[d], d))
    print('files with the most repeated lines:')
    for path, lines in sorted(repeated.items(), key=lambda x: -len(x[1]))[:12]:
        print('  %5d / %5d  %s' % (len(lines), len(files[path]), rel(path)))
    print('pairs sharing the most windows:')
    for (a, b), count in pairs.most_common(10):
        print('  %5d  %s  <->  %s' % (count, rel(a), rel(b) if a != b else '(itself)'))

    for path in args.blocks:
        marks = repeated.get(path, {})
        runs = []
        for i in sorted(marks):
            if runs and i == runs[-1][1] + 1:
                runs[-1][1] = i
            else:
                runs.append([i, i])
        print('longest repeated runs of %s:' % rel(path))
        for a, b in sorted(runs, key=lambda r: r[0] - r[1])[:8]:
            other = marks[a]
            print('  lines %d-%d (%d code lines), also at %s:%d'
                  % (files[path][a][0], files[path][b][0], b - a + 1, rel(other[0]), other[1]))


if __name__ == '__main__':
    main()
