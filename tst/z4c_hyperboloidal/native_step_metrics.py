#!/usr/bin/env python3
"""Compare complete native stage observations with historical per-step metrics.

No evolution or scheduler access. A passing prefix is explicitly not completion
of the3551-step trajectory. The scalar gates were declared before long native
data: short1e-12+5e-10|reference|; long1e-10+1e-8|reference|.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import struct

from n12_checkpoint import DT, expected_time, require, sha256
from native_step_observer_validate import stages

TARGET = 3551
GATES = {'short': (1e-12, 5e-10), 'long': (1e-10, 1e-8)}


def historical_steps(path, count, expected_sha256=None, allow_tail=False):
    path = Path(path).resolve()
    digest = sha256(path)
    require(expected_sha256 is None or digest == expected_sha256,
            'historical step CSV hash differs')
    with path.open() as stream:
        reader = csv.DictReader(stream)
        require(set(('step', 't', 'dt', 'CFL', 'stage_clock')) <= set(reader.fieldnames or []),
                'historical step metric columns missing')
        rows = list(reader)
    require((len(rows) >= count if allow_tail else len(rows) == count) and
            [int(row['step']) for row in rows] == list(range(1, len(rows)+1)),
            'historical steps missing, repeated, reordered or extra')
    for step, row in enumerate(rows, 1):
        require(float(row['t']) == expected_time(step) and float(row['dt']) == DT,
                'historical accepted time or timestep differs')
        require(all(math.isfinite(float(row[key])) and float(row[key]) >= 0
                    for key in ('CFL', 'stage_clock')), 'historical clock/CFL invalid')
    return rows, digest


def compare_metric(actual, reference, gate):
    absolute, relative = GATES[gate]
    delta = abs(actual-reference)
    tolerance = absolute+relative*abs(reference)
    return dict(actual=actual, reference=reference, absolute_difference=delta,
                scaled_difference=delta/(1+abs(reference)), tolerance=tolerance,
                tolerance_ratio=delta/tolerance,
                bitwise_equal=struct.pack('<d', actual) == struct.pack('<d', reference),
                passed=delta <= tolerance)


def compare_step_metrics(runs, historical_csv, last_step, gate='long', ranks=24,
                         historical_sha256=None):
    """Validate a contiguous completed prefix using all four actual RK stages.

    `runs` contains ordinary observed-main output directories. Each stage must
    occur exactly once; no duplicate or favorable row is selected. `ranks` is
    the actual common decomposition. The short reference has8 steps; the long
    reference must contain all3551 steps even when assessing an earlier prefix.
    A longer historical file is retained and validated in full, never trimmed.
    """
    require(gate in GATES and 1 <= last_step <= (8 if gate == 'short' else TARGET),
            'unsupported metric gate or requested endpoint')
    require(ranks in (24, 96), 'scientific metrics require actual24/96 GPU observations')
    runs = [Path(path).resolve() for path in runs]
    require(runs and len(set(runs)) == len(runs), 'missing or repeated observer directories')
    reference, reference_sha = historical_steps(historical_csv,
        8 if gate == 'short' else TARGET, historical_sha256, allow_tail=gate == 'long')
    hashes = {str(Path(historical_csv).resolve()): reference_sha}
    by_stage = {}; completions = []
    for folder in runs:
        meta_path = folder/'native-step-observer.json'
        metadata = json.loads(meta_path.read_text())
        require(metadata['backend'] == 'SYCL', 'CPU observations are not GPU scientific evidence')
        first, last = metadata['initial_cycle'], metadata['final_cycle']
        require(0 <= first < last <= last_step, 'observer segment is outside the requested prefix')
        rows, checked = stages(folder, first, last, ranks)
        completions.append(dict(path=str(folder), **checked))
        hashes[str(meta_path)] = sha256(meta_path)
        hashes[str(folder/'native-stages.csv')] = sha256(folder/'native-stages.csv')
        for row in rows:
            key = (int(row['step']), int(row['stage']))
            require(key not in by_stage, 'duplicate observed RK stage')
            by_stage[key] = row
    expected = {(step, stage) for step in range(1, last_step+1) for stage in range(1, 5)}
    require(by_stage.keys() == expected, 'missing or extra observed RK stage')
    report = dict(schema='native-step-metrics-v1', passed=True, through_step=last_step,
        full3551_complete=False, stage_rows=len(by_stage), ranks=ranks, gate=gate,
        gate_formula=('1e-12+5e-10*abs(reference)' if gate == 'short' else
                      '1e-10+1e-8*abs(reference)'),
        gate_scope='predeclared scalar engineering application of the adopted arithmetic gate; not a derived stability or propagation bound',
        exact_step_stage_time_dt=True, compatibility_bound=1e-7,
        maximum_compatibility=max(row['compatibility'] for row in by_stage.values()),
        historical_steps_available=len(reference), completed_steps=list(range(1, last_step+1)),
        historical_uncompared_tail=dict(count=len(reference)-last_step,
            first_step=last_step+1 if len(reference)>last_step else None,
            last_step=len(reference) if len(reference)>last_step else None,
            beyond3551=max(0, len(reference)-TARGET)),
        missing_future_steps=list(range(last_step+1, TARGET+1)),
        observer_segments=completions, input_sha256=hashes, steps=[], maxima={}, failed_values=0,
        scope='observed stage values only; successful executable completion, full fields/records and history/horizon admission remain separate')
    for step in range(1, last_step+1):
        rows = [by_stage[(step, stage)] for stage in range(1, 5)]
        row = dict(step=step, time=expected_time(step), dt=DT,
                   compatibility=max(x['compatibility'] for x in rows))
        for field, old_column in [('clock', 'stage_clock'), ('CFL', 'CFL')]:
            comparison = compare_metric(max(x[field] for x in rows),
                                        float(reference[step-1][old_column]), gate)
            row[field] = comparison
            report['failed_values'] += not comparison['passed']
        report['steps'].append(row)
    for field in ('clock', 'CFL'):
        result = {}
        for metric in ('absolute_difference', 'scaled_difference', 'tolerance_ratio'):
            worst = max(report['steps'], key=lambda row: row[field][metric])
            result[metric] = dict(step=worst['step'], **worst[field])
        report['maxima'][field] = result
    report['passed'] = report['failed_values'] == 0
    report['full3551_complete'] = report['passed'] and last_step == TARGET
    report['source_sha256'] = {str(Path(__file__).with_name(name).resolve()):
        sha256(Path(__file__).with_name(name)) for name in
        ('native_step_metrics.py', 'native_step_observer_validate.py', 'n12_checkpoint.py')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--historical-csv', type=Path, required=True)
    parser.add_argument('--historical-sha256')
    parser.add_argument('--through-step', type=int, required=True)
    parser.add_argument('--gate', choices=GATES, default='long')
    parser.add_argument('--ranks', type=int, choices=(24, 96), default=24)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = compare_step_metrics(args.run, args.historical_csv, args.through_step,
                                      args.gate, args.ranks, args.historical_sha256)
    except Exception as error:
        result = dict(schema='native-step-metrics-v1', passed=False,
                      through_step=args.through_step, full3551_complete=False,
                      validation_error=f'{type(error).__name__}: {error}')
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('passed', 'through_step', 'full3551_complete',
                     'stage_rows', 'failed_values', 'validation_error') if key in result}, indent=2))
    raise SystemExit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
