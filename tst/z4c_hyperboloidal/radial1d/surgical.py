#!/usr/bin/env python3
"""Reproduce the bounded radial damping/endpoint experiment; no MPI required."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time


def cases(phase):
    # name, equivalent outer-shell N, damping, endpoint donors, duration,
    # wide pulse, half timestep. Each run uses one CPU thread.
    if phase == 'refine':
        return ([('control-damping-N128', 128, .5, 7, .8, False, False)]
                + [(f'control-moderate-N{n}', n, .375, 7, .8, False, False)
                   for n in [12, 16, 20, 32, 64]])
    if phase == 'screen':
        return [(f'control-{tag}-N{n}', n, k, ep, .8, False, False)
                for tag, k, ep, ns in [('base', .25, 7, [12, 16, 20]),
                                       ('damping', .5, 7, [12, 16, 20]),
                                       ('endpoint9', .25, 9, [12, 16])]
                for n in ns]
    return ([(f'long-{tag}-N{n}', n, k, 7, 1.6, False, False)
             for tag, k in [('base', .25), ('damping', .5)]
             for n in [12, 20, 32, 64]]
            + [(f'wide-{tag}-N{n}', n, k, 7, .8, True, False)
               for tag, k in [('base', .25), ('damping', .5)]
               for n in [16, 32, 64]]
            + [('control-damping-halfdt-N20', 20, .5, 7, .8, False, True),
               ('wide-damping-halfdt-N16', 16, .5, 7, .8, True, True)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--phase', choices=['screen', 'validate', 'refine'], required=True)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    binary, output = a.binary.resolve(), a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()

    def run(case):
        name, n, k, ep, end, wide, half = case
        dest = output/name
        dest.mkdir()  # Never silently overwrite an earlier experiment.
        times = '0,.24,.48,.8,1.2,1.6' if a.phase == 'validate' else '0,.24,.48,.8'
        command = [str(binary), '--h', str(.06/n), '--rin', '.5',
                   '--tend', str(end), '--kappa', str(k),
                   '--endpoint-points', str(ep), '--threads', '1',
                   '--stride', '3', '--profiles', times, '--out', str(dest)]
        if wide:
            command += ['--amp', '.001', '--center', '.86', '--width', '.08',
                        '--profile', '1']
        if half:
            command += ['--dtfac', str(.0938884074712349/2)]
        start = time.monotonic()
        with (dest/'stdout.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        receipt = dict(case=name, command=command, returncode=result.returncode,
                       seconds=time.monotonic()-start, executable_sha256=digest)
        (dest/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt), flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        receipts = list(pool.map(run, cases(a.phase)))
    (output/(a.phase+'-receipts.json')).write_text(json.dumps(receipts, indent=2)+'\n')
    if any(r['returncode'] != 0 for r in receipts):
        raise SystemExit('Some runs failed; inspect their preserved receipts and logs.')


if __name__ == '__main__':
    main()
