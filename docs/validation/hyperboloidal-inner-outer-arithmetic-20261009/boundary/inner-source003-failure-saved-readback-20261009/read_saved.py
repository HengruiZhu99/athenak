#!/usr/bin/env python3
"""Stdlib-only association of existing source002/003 oracle failures.

No oracle import, target evaluation, multiplication replay, or source query.
The original report did not record labels. Native bit patterns plus its exact
record/update ordering give a monotone candidate association; ambiguity is kept.
"""
import argparse
from bisect import bisect_right, bisect_left
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def guard(pins):
    for p in pins:
        if Path(p['path']).stat().st_size != p['bytes'] or digest(p['path']) != p['sha256']:
            raise RuntimeError('Input drift: ' + p['path'])


def primal(x):
    return x[0] if isinstance(x, list) else x


RADII = [0, .025, .05, .1, .3, .45, .5, .65, .84, .85, .9, .95, .98, .995]
FAMILIES = ['reference', 'mild-lapse', 'mild-chi', 'shift', 'physical-P', 'Lambda',
            'chi-gradient', 'SPD-tensor', 'off-constraint-Theta', 'mixed', 'collapsed',
            'small-alpha-large-chi', 'large-alpha-small-chi', 'chi-gradient-contrast']
PARTS = ['regular.alpha', 'regular.beta_x', 'regular.beta_y', 'regular.beta_z',
         'pole.alpha', 'pole.beta_x', 'pole.beta_y', 'pole.beta_z']
RHS = ['alpha', 'beta_x', 'beta_y', 'beta_z']


def context(row, mode, line):
    out = dict(mode=mode, line=line, kind=row['kind'], family=row.get('family'),
               a=row.get('curvature'), radius=primal(row.get('radius', [None])),
               direction=row.get('direction'), G0=row.get('G0'), W=row.get('W'),
               alpha=primal(row['input']['alpha']), chi=primal(row['input']['chi']))
    if mode == 'sources':
        i = line - 1
        fi, i = i % 14, i // 14
        gi, i = i % 2, i // 2
        di, i = i % 3, i // 3
        ri, ai = i % 14, i // 14
        assert row['family'] == FAMILIES[fi]
        assert row['G0'] == [.375, .75][gi] and row['direction'] == di
        assert row['curvature'] == [.5, 1, 2, 4][ai]
        out['fixed_radius'] = RADII[ri]
    return out


def inspect(case):
    recipe = load(case['recipe'])
    report = load(case['report'])
    receipt = load(case['receipt'])
    assert report['passed'] is False and receipt['passed'] is False
    assert receipt['returncode'] == 1 and receipt['source_inputs_unchanged'] is True
    assert report['counts'] == recipe['expected_record_counts']
    failures = report['failures']
    assert set(f['check'] for f in failures) <= {'source-parts', 'source-rhs'}
    needed = set()
    for f in failures:
        x = float(f['got'])
        assert Decimal.from_float(x) == Decimal(f['got'])
        needed.add((f['check'], x.hex()))
    candidates = defaultdict(list)
    event_info = {}
    row_info = {}
    counts, audit = Counter(), Counter()
    ordinal = 0
    for mode in recipe['modes']:
        with open(Path(case['attempt']) / (mode + '.jsonl')) as stream:
            for line, text in enumerate(stream, 1):
                row = json.loads(text)
                counts[row['kind']] += 1
                if mode == 'sources':
                    c = context(row, mode, line)
                    row_info[line] = c
                    audit.update(row.get('arithmetic', {}))
                if row['kind'] not in ('reference', 'source', 'principal', 'dual'):
                    continue
                c = context(row, mode, line)
                for field, check, labels in [('parts', 'source-parts', PARTS), ('rhs', 'source-rhs', RHS)]:
                    for component, atom in enumerate(row[field]):
                        value = primal(atom)
                        key = (check, float(value).hex())
                        if key in needed:
                            candidates[key].append(ordinal)
                            event_info[ordinal] = {**c, 'component': component, 'field': labels[component],
                                                   'native_hex': key[1], 'event': ordinal}
                        ordinal += 1
    assert dict(counts) == recipe['expected_record_counts']
    # All candidate update positions that can occur in an ordered embedding of
    # the recorded failure list. A native value may also occur in a passing row;
    # the procedure never fabricates a label absent from the original report.
    lists = [candidates[(f['check'], float(f['got']).hex())] for f in failures]
    earliest = []
    previous = -1
    for positions in lists:
        j = bisect_right(positions, previous)
        if j == len(positions):
            raise RuntimeError('Failure sequence has no monotone native association')
        previous = positions[j]
        earliest.append(previous)
    latest = [None] * len(lists)
    following = ordinal
    for i in range(len(lists) - 1, -1, -1):
        positions = lists[i]
        j = bisect_left(positions, following) - 1
        if j < 0:
            raise RuntimeError('Failure sequence has no reverse association')
        following = positions[j]
        latest[i] = following
    rows, ambiguous = [], 0
    for i, (f, positions) in enumerate(zip(failures, lists)):
        lower = earliest[i-1] if i else -1
        upper = latest[i+1] if i+1 < len(lists) else ordinal
        viable = positions[bisect_right(positions, lower):bisect_left(positions, upper)]
        assert viable
        matches = [event_info[p] for p in viable]
        ambiguous += len(matches) != 1
        rows.append({'failure_index': i, **f, 'matches': matches})
    def classify(fields):
        counter = Counter()
        ambiguous_class = 0
        for r in rows:
            keys = {tuple(m.get(k) for k in fields) for m in r['matches']}
            if len(keys) == 1:
                counter[next(iter(keys))] += 1
            else:
                ambiguous_class += 1
        return {'rows': [{'class': dict(zip(fields, key)), 'failures': n}
                         for key, n in sorted(counter.items(), key=lambda item: repr(item[0]))],
                'ambiguous_failures': ambiguous_class}
    groups = {}
    for name, predicate in [('all', lambda r: True), ('source-parts', lambda r: r['check']=='source-parts'),
                            ('source-rhs', lambda r: r['check']=='source-rhs')]:
        subset = [r for r in rows if predicate(r)]
        worst = max(subset, key=lambda r: Decimal(r['error']))
        largest = max(subset, key=lambda r: abs(Decimal(r['got'])))
        groups[name] = {'first': subset[0], 'worst_scaled': worst, 'largest_native_magnitude': largest}
    signatures = set()
    signatures_complete = True
    for r in rows:
        options = {(m['mode'], m['line'], m['component'], r['check']) for m in r['matches']}
        if len(options) == 1:
            signatures.add(next(iter(options)))
        else:
            signatures_complete = False
    return {'source': case['name'], 'original_passed': False, 'original_returncode': receipt['returncode'],
            'counts': dict(counts), 'failure_count': len(rows),
            'by_check': dict(Counter(r['check'] for r in rows)), 'maxima': report['maxima'],
            'association': {'method': 'native exact binary64 value + original monotone update ordering',
                            'target_oracle_recomputed': False, 'ambiguous_failure_count': ambiguous,
                            'total_matched_failures': len(rows), 'exact_signatures_complete': signatures_complete},
            'by_family': classify(['family']), 'by_kind_family': classify(['kind','family']),
            'by_field': classify(['field']), 'by_a_radius_family': classify(['a','fixed_radius','family']),
            'first_worst': groups, 'source_mode_branch_audit_sums': dict(audit)}, rows, signatures


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', required=True)
    p.add_argument('--attempt', required=True)
    args = p.parse_args()
    if sys.flags.optimize or not sys.dont_write_bytecode:
        raise RuntimeError('Use unoptimized bytecode-off Python')
    recipe = load(args.recipe)
    pins = load(recipe['input_pins'])
    guard(pins)
    attempt = Path(args.attempt)
    attempt.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = {'passed': False, 'completed': False, 'inputs_unchanged': False,
               'scope': 'saved stdlib scalar association only; no oracle/target/query/recompute'}
    try:
        results, all_rows, signatures = [], {}, []
        for case in recipe['cases']:
            result, rows, sig = inspect(case)
            results.append(result)
            all_rows[case['name']] = rows
            signatures.append(sig)
        comparison = {'failure_count_delta_003_minus_002': results[1]['failure_count']-results[0]['failure_count'],
                      'exact_signature_sets_complete': all(r['association']['exact_signatures_complete'] for r in results)}
        if comparison['exact_signature_sets_complete']:
            comparison.update(common_failures=len(signatures[0]&signatures[1]),
                              source002_only=len(signatures[0]-signatures[1]), source003_only=len(signatures[1]-signatures[0]))
        output = {'passed_saved_readback': True, 'original_source002_and003_still_failed': True,
                  'supplement_eligible': False, 'cases': results, 'comparison': comparison,
                  'limitation': 'Original oracle failure entries lack row labels. All viable monotone matches are retained; any count by family/field is qualified by its recorded ambiguity count. No numerical target was recalculated.'}
        write(attempt/'summary.json', output)
        # This compact metadata stores associations and original saved scalars,
        # never any complete field/RHS array or oracle-generated target payload.
        compact = {}
        for name, rows in all_rows.items():
            compact[name] = [{'failure_index': r['failure_index'], 'check': r['check'],
                              'got_hex': float(r['got']).hex(), 'error': r['error'],
                              'matches': r['matches']} for r in rows]
        write(attempt/'associations.json', compact)
        guard(pins)
        receipt.update(passed=True, completed=True, inputs_unchanged=True, returncode=0,
                       elapsed_seconds=time.monotonic()-started,
                       outputs=[{'path': str(attempt/name), 'sha256': digest(attempt/name),
                                 'bytes': (attempt/name).stat().st_size}
                                for name in ('summary.json','associations.json')])
        write(attempt/'receipt.json', receipt)
        print(json.dumps({'passed':True, 'counts':[r['failure_count'] for r in results],
                          'ambiguous':[r['association']['ambiguous_failure_count'] for r in results],
                          'comparison':comparison, 'summary_sha256':digest(attempt/'summary.json')}))
    except Exception as e:
        receipt.update(returncode=1, error=repr(e), elapsed_seconds=time.monotonic()-started)
        try:
            guard(pins)
            receipt['inputs_unchanged'] = True
        except Exception as drift:
            receipt['pin_error'] = repr(drift)
        write(attempt/'receipt.json', receipt)
        raise


if __name__ == '__main__':
    main()
