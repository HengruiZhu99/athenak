#!/usr/bin/env python3
"""Review current single-hole timing, complete checkpoint payloads and diagnostics.

The old and regular chi policies need not agree bitwise. Each policy's timed
and ordinary runs must agree exactly. No evolution or submission occurs here.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source/'tst/z4c_hyperboloidal'))
    from n12_checkpoint import parse_native, DT, expected_time, sha256
    root = args.run
    require((root/'completed.txt').is_file(), 'campaign did not finish')
    binding = collections.defaultdict(set)
    for line in (root/'rank-environment.txt').read_text().splitlines():
        match = re.fullmatch(r'(\S+) rank=(\d+) tile=(\d+)\.(\d+) implicit=0 cache=specialized size=limited entries=128', line)
        require(match is not None, 'unexpected rank/cache binding: '+line)
        host, rank, gpu, tile = match.groups()
        rank = int(rank)
        require((int(gpu), int(tile)) == (rank//2, rank % 2), 'incorrect GPU tile')
        require(rank not in binding[host], 'duplicate rank')
        binding[host].add(rank)
    require(len(binding) == 2 and all(v == set(range(12)) for v in binding.values()), 'not 24 unique tile ranks on two nodes')
    report = dict(schema='current-single-hole-performance-v1', passed=False,
                  job=(root/'job-id.txt').read_text().strip(), nodes=2, ranks=24,
                  historical_seconds_per_step=0.10611714095, cases={})
    states = {}
    for name in ['saved-a','origin-a','regular-a','regular-b','origin-b','saved-b',
                 'origin-normal','regular-normal']:
        folder = root/name
        require((folder/'exit-code.txt').read_text().strip() == '0', name+' failed')
        log = (folder/'run.log').read_text()
        final_cycle = 2556 if name.startswith('saved') else 70
        require('Terminating on cycle limit' in log and
                re.search(r'^time=\S+ cycle='+str(final_cycle)+r'$', log, re.M), 'incomplete '+name)
        checkpoints = sorted((folder/'rst').glob('*.rst'))
        require(bool(checkpoints), 'missing restart '+name)
        info = parse_native(checkpoints[-1])
        require(info['cycle'] == final_cycle and info['dt'] == DT and
                info['time'] == expected_time(final_cycle), 'wrong checkpoint clock '+name)
        regular = name.startswith('regular')
        require((b'residual-weight=regular-c6-v1' in info['identity']) == regular, 'wrong chi policy '+name)
        if regular:
            require(float(info['parameters']['z4c_hyperboloidal/weight_inner']) == .3 and
                    float(info['parameters']['z4c_hyperboloidal/weight_outer']) == .55, 'wrong regular weight radii')
        data = np.memmap(info['path'], mode='r', dtype='<f8', offset=info['offset'], shape=(106, 5976, 39))
        require(np.isfinite(data).all(), 'nonfinite checkpoint '+name)
        active = np.any(data[:, :, :31] != 0, axis=2)
        require(int(active.sum()) == 229392+864, 'wrong active point count '+name)
        require((data[:, :, 0][active] > 0).all(), 'nonpositive chi '+name)
        state = np.array(data[active, :31])
        if name in ['origin-a', 'regular-a']:
            states[name] = state
        digest = hashlib.sha256(memoryview(data).cast('B')).hexdigest()
        hst = np.loadtxt(folder/'native_n12_5M.z4c_hyperboloidal.hst', ndmin=2)
        horizon = np.loadtxt(folder/'native_n12_5M.horizon_summary_0.txt', ndmin=2)
        shape = np.loadtxt(folder/'native_n12_5M.horizon_shape_0.txt', ndmin=2)
        require(np.isfinite(hst).all() and np.isfinite(horizon).all() and np.isfinite(shape).all(), 'nonfinite diagnostics '+name)
        require(int(horizon[-1, 0]) == final_cycle and horizon[-1, 2] > 0 and
                horizon[-1, 8] < 1e-3 and horizon[-1, 11] > 0, 'horizon failed '+name)
        row = dict(final_cycle=final_cycle, final_time=float(info['time']),
                   final_time_over_M0=float(info['time']*3), chi_policy='regular_c6' if regular else 'origin_r2',
                   active_points=int(active.sum()), minimum_chi=float(state[:, 0].min()),
                   checkpoint_payload_sha256=digest, checkpoint_sha256=sha256(checkpoints[-1]),
                   final_history=hst[-1].tolist(), final_horizon=horizon[-1].tolist())
        if not name.endswith('normal'):
            timing = json.loads((folder/'native-cycle-timing.json').read_text())
            require(timing['passed'] and timing['ranks'] == 24 and timing['blocks'] == 106 and
                    timing['bulk_points'] == 229392 and timing['scri_points'] == 864 and
                    timing['complete_steps'] == 70 and timing['warmup_steps'] == 10 and
                    timing['steps_per_window'] == 20 and timing['window_count'] == 3 and
                    timing['timed_output_count'] == timing['timed_console_count'] == 0 and
                    not timing['profiling_library_loaded'], 'invalid timing scope '+name)
            windows = [w['seconds_per_complete_step'] for w in timing['windows']]
            require(len(windows) == 3 and all(np.isfinite(w) and w > 0 for w in windows), 'invalid windows')
            row.update(seconds_per_complete_step=windows, median=statistics.median(windows),
                       startup_seconds=timing['startup_max_seconds'], finalize_seconds=timing['finalize_max_seconds'])
        report['cases'][name] = row
        del data
    for prefix in ['origin','regular']:
        a = report['cases'][prefix+'-a']['checkpoint_payload_sha256']
        for suffix in ['b','normal']:
            require(report['cases'][prefix+'-'+suffix]['checkpoint_payload_sha256'] == a,
                    'instrumentation/repeat changed full 31+8 payload '+prefix)
            for extension in ['z4c_hyperboloidal.hst','horizon_summary_0.txt','horizon_shape_0.txt']:
                require((root/(prefix+'-a')/('native_n12_5M.'+extension)).read_bytes() ==
                        (root/(prefix+'-'+suffix)/('native_n12_5M.'+extension)).read_bytes(),
                        'instrumentation/repeat changed diagnostics '+prefix)
    require(report['cases']['saved-a']['checkpoint_payload_sha256'] == report['cases']['saved-b']['checkpoint_payload_sha256'], 'saved repeat changed state')
    report['saved_payload_matches_historical'] = report['cases']['saved-a']['checkpoint_payload_sha256'] == 'fa76b178240a0c8c3fb41979168230bc5fd389e6e782df50fb631dae47d8201b'
    report['timed_vs_normal_complete_payload_and_diagnostics_exact'] = True
    report['medians'] = {}
    for prefix in ['saved','origin','regular']:
        windows = sum([report['cases'][prefix+'-'+suffix]['seconds_per_complete_step'] for suffix in ['a','b']], [])
        report['medians'][prefix] = dict(seconds_per_step=statistics.median(windows), minimum=min(windows), maximum=max(windows), windows=windows)
    difference = states['regular-a']-states['origin-a']
    report['regular_minus_origin'] = dict(max_absolute_by_field=np.max(np.abs(difference), axis=0).tolist(),
                                        rms_by_field=np.sqrt(np.mean(difference*difference, axis=0)).tolist())
    report['passed'] = True
    (root/'review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['passed','job','saved_payload_matches_historical','medians']}, indent=2))


if __name__ == '__main__':
    main()
