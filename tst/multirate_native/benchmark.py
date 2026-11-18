#!/usr/bin/env python3
"""Prepare/run matched-interval multirate comparisons inside an existing allocation.

Pass --execute inside an existing allocation to measure the configured cases.
Supply an optional gate report to record the qualification status with the timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', required=True, help='quoted launcher with {ranks}, including GPU binding')
    parser.add_argument('--allocation-nodes', type=int, choices=(1, 2), required=True)
    parser.add_argument('--gpus-per-node', type=int, required=True)
    parser.add_argument('--active-ranks', type=int, nargs='+', required=True)
    parser.add_argument('--base-dt', type=float, required=True)
    parser.add_argument('--finest-level', type=int, required=True)
    parser.add_argument('--intervals', type=int, default=256)
    parser.add_argument('--coordinate-time-per-M0', type=float, required=True)
    parser.add_argument('--gate-report', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if (args.gpus_per_node < 1 or not 0 <= args.finest_level <= 30 or args.base_dt <= 0 or
            args.intervals < 1 or args.coordinate_time_per_M0 <= 0):
        parser.error('invalid positive time, interval, level or GPU parameters')
    capacity = args.allocation_nodes*args.gpus_per_node
    if any(rank < 1 or rank > capacity for rank in args.active_ranks):
        parser.error('active ranks must fit the explicitly supplied one/two-node allocation')
    if '{ranks}' not in args.launcher:
        parser.error('launcher must explicitly substitute {ranks}')
    gates = None
    if args.gate_report is not None:
        gates = json.loads(args.gate_report.read_text())
    binary, fixture, output = args.binary.resolve(), args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    factor = 2**args.finest_level
    cases = []
    for ranks in args.active_ranks:
        for mode in ('uniform', 'scalar', 'metis'):
            directory = output/f'{mode}-{ranks}'
            dt = args.base_dt/factor if mode == 'uniform' else args.base_dt
            cycles = args.intervals*factor if mode == 'uniform' else args.intervals
            directory.mkdir()
            case_input = directory/'benchmark.athinput'
            # Athena requires CLI overrides to exist in the original input. Its
            # native reader merges repeated blocks, with later settings taking precedence.
            settings = (f'\n<time>\nintegrator = mrgark4\ndt = {dt:.17g}\nnlim = {cycles}\n'
                        f'tlim = 1e100\nndiag = 32\n<multirate>\nuniform_reference = {str(mode=="uniform").lower()}\n'
                        f'balance = {"none" if mode=="uniform" else mode}\npartition_log = {directory}/partition.log\n')
            case_input.write_text(fixture.read_text() + settings)
            command = [part.format(ranks=ranks) for part in shlex.split(args.launcher)]
            command += [str(binary), '-i', str(case_input)]
            cases.append(dict(mode=mode, ranks=ranks, cwd=str(directory), command=command,
                              coordinate_interval=args.base_dt*args.intervals))
    manifest = dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    input=str(fixture), input_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    allocation_nodes=args.allocation_nodes, gpus_per_node=args.gpus_per_node,
                    coordinate_time_per_M0=args.coordinate_time_per_M0,
                    gate_report=gates, executed=args.execute, cases=cases)
    (output/'commands.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if not args.execute:
        print(f'Prepared {len(cases)} commands: {output}/commands.json')
        return
    results = []
    for case in cases:
        directory = Path(case['cwd'])
        started = time.monotonic()
        with (directory/'run.log').open('w') as stream:
            status = subprocess.run(case['command'], cwd=directory, stdout=stream,
                                    stderr=subprocess.STDOUT, env=os.environ).returncode
        wall = time.monotonic()-started
        text = (directory/'run.log').read_text()
        if status:
            raise RuntimeError(f'benchmark failed ({status}); inspect {directory}/run.log')
        intervals = re.findall(r'MULTIRATE_INTERVAL[^\n]*', text)
        stage_evaluations = [int(value) for value in re.findall(r'stage_evaluations=(\d+)', '\n'.join(intervals))]
        receipts = [line for line in text.splitlines() if line.startswith('MULTIRATE_')]
        result = dict(mode=case['mode'], ranks=case['ranks'], wall_seconds=wall,
                      seconds_per_M0=wall*args.coordinate_time_per_M0/case['coordinate_interval'],
                      reported_stage_evaluations=sum(stage_evaluations),
                      interval_receipts=len(intervals), runtime_receipts=receipts,
                      log=str(directory/'run.log'))
        results.append(result)
        (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print(f'Completed measured comparisons: {output}/results.json')


if __name__ == '__main__':
    main()
