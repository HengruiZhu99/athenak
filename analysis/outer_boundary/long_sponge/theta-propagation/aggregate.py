#!/usr/bin/env python3
"""Validate full eight-rank Athena 1.1 Theta dumps and retain compact profiles.

No histories, restart files, or running-job settings are modified. The supported
geometry is the uniform eight-block 64^3 Minkowski test, [-2048,2048]^3.
Weights are coordinate volume, not proper metric volume. Cached complete
cohorts are reused only while all file sizes and nanosecond mtimes match.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import time

import numpy as np

CASES = {'theta_primary': 'theta_loweta_coremask',
         'theta_lapse01': 'theta_loweta_lapse01_coremask'}
REGIONS = ['core_r_le_512', 'ramp_512_to_1792', 'plateau_r_ge_1792',
           'face_distance_le_256', 'whole_domain']


def parse(path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed while reading'
    f = io.BytesIO(raw)
    assert f.readline() == b'Athena binary output version=1.1\n'
    head = {}
    for unused in range(6):
        key, value = f.readline().decode('ascii').strip().split('=', 1)
        head[key] = value
    assert head['size of preheader'] == '5'
    names = f.readline().decode('ascii').strip().split(':', 1)[1].split()
    assert names == ['z4c_Theta'] and int(head['number of variables']) == 1
    offset = int(f.readline().decode('ascii').split('=', 1)[1])
    input_raw = f.read(offset)
    assert len(input_raw) == offset
    params = {}
    for line in input_raw.decode('ascii').splitlines():
        line = line.split('#')[0].strip()
        if line.startswith('<'):
            section = line[1:-1]
            params[section] = {}
        elif '=' in line:
            key, value = line.split('=', 1)
            params[section][key.strip()] = value.strip()
    assert float(params['problem']['bh_mass']) == 0
    assert params['mesh_refinement']['refinement'] == 'none'
    for axis in (1, 2, 3):
        assert int(params['mesh']['nx%d' % axis]) == 64
        assert float(params['mesh']['x%dmin' % axis]) == -2048
        assert float(params['mesh']['x%dmax' % axis]) == 2048
        assert int(params['meshblock']['nx%d' % axis]) == 32
    assert int(head['size of location']) == 8 and int(head['size of variable']) == 4
    blocks = []
    while f.tell() < len(raw):
        data = f.read(40)
        assert len(data) == 40, 'Incomplete block header'
        indices = struct.unpack('<6i', data[:24])
        loc = struct.unpack('<4i', data[24:])
        assert indices == (4, 35, 4, 35, 4, 35), 'Expected full active block, no ghosts/slice'
        assert all(x in (0, 1) for x in loc[:3]) and loc[3] == 0
        data = f.read(48)
        assert len(data) == 48
        limits = struct.unpack('<6d', data)
        expected = tuple(v for i in loc[:3] for v in (-2048+2048*i, 2048*i))
        assert limits == expected, 'Logical/physical coordinate mismatch'
        data = f.read(32**3*4)
        assert len(data) == 32**3*4, 'Incomplete payload'
        theta = np.frombuffer(data, dtype='<f4').astype(np.float64).reshape(32, 32, 32)
        assert np.isfinite(theta).all(), 'Nonfinite Theta'
        blocks.append((loc, theta))
    assert len(blocks) == 1, 'Expected one full block per rank'
    return head, hashlib.sha256(input_raw).hexdigest(), blocks[0], hashlib.sha256(raw).hexdigest()


def geometry(loc, edges):
    axes = [-2048+2048*j+(np.arange(32)+.5)*64 for j in loc[:3]]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
    radius = np.sqrt(x*x+y*y+z*z)
    face_dist = 2048-np.maximum.reduce([abs(x), abs(y), abs(z)])
    masks = [radius <= 512, (radius > 512) & (radius < 1792),
             radius >= 1792, face_dist <= 256, np.ones(radius.shape, bool)]
    return np.searchsorted(edges, radius.ravel(), side='right')-1, masks, (x, y, z), radius


def aggregate(paths, edges):
    size = len(edges)-1
    count, sum1, sum2, maxabs = [np.zeros(size) for unused in range(4)]
    reg_count, reg_sum1, reg_sum2, reg_max = [np.zeros(len(REGIONS)) for unused in range(4)]
    seen, hashes = set(), []
    reference = None
    max_theta, peak = -1., None
    for rank, path in enumerate(paths):
        header, phash, (loc, theta), digest = parse(path)
        match = (header, phash)
        if reference is None:
            reference = match
        assert match == reference, 'Rank time/cycle/parameter/header mismatch'
        assert loc not in seen, 'Duplicate block coverage'
        seen.add(loc)
        ix, masks, xyz, radius = geometry(loc, edges)
        count += np.bincount(ix, minlength=size)
        sum1 += np.bincount(ix, weights=theta.ravel(), minlength=size)
        sum2 += np.bincount(ix, weights=theta.ravel()**2, minlength=size)
        np.maximum.at(maxabs, ix, abs(theta).ravel())
        for n, mask in enumerate(masks):
            reg_count[n] += mask.sum()
            reg_sum1[n] += theta[mask].sum()
            reg_sum2[n] += (theta[mask]**2).sum()
            reg_max[n] = max(reg_max[n], float(abs(theta[mask]).max()) if mask.any() else 0.)
        k, j, i = np.unravel_index(np.argmax(abs(theta)), theta.shape)
        if abs(theta[k,j,i]) > max_theta:
            max_theta = float(abs(theta[k,j,i]))
            peak = {'signed_value': float(theta[k,j,i]), 'rank': rank, 'logical_location': list(loc),
                    'xyz': [float(v[k,j,i]) for v in xyz], 'radius': float(radius[k,j,i])}
        hashes.append(digest)
    assert len(seen) == 8 and count.sum() == 64**3, 'Missing domain coverage'
    mean = np.divide(sum1, count, out=np.zeros(size), where=count > 0)
    rms = np.sqrt(np.divide(sum2, count, out=np.zeros(size), where=count > 0))
    result = dict(mean=mean, rms=rms, maxabs=maxabs, count=count,
                  regional_mean=reg_sum1/reg_count, regional_rms=np.sqrt(reg_sum2/reg_count),
                  regional_maxabs=reg_max, regional_integral2=reg_sum2*64**3,
                  regional_coordinate_volume=reg_count*64**3)
    meta = dict(time=float(reference[0]['time']), cycle=int(reference[0]['cycle']),
                rank_sha256=hashes, peak=peak, all_finite=True, matching_headers=True,
                matching_payloads=True, unique_blocks=8, active_cells=64**3)
    return result, meta


def run_case(run, out, case, prefix):
    folder = out/case
    cache = folder/'cache'
    cache.mkdir(parents=True, exist_ok=True)
    edges = np.arange(0., 3648., 64.)
    old = json.loads((folder/'manifest.json').read_text()) if (folder/'manifest.json').exists() else {}
    old = {item['filename']: item for item in old.get('included', [])}
    files = sorted((run/case/'bin/rank_00000000').glob(prefix+'.z4c_Theta.*.bin'))
    included, excluded, profiles = [], [], []
    for first in files:
        paths = [run/case/'bin'/('rank_%08d' % rank)/first.name for rank in range(8)]
        try:
            stats = [[p.stat().st_size, p.stat().st_mtime_ns] for p in paths]
            cached = old.get(first.name)
            target = cache/(first.stem+'.npz')
            if cached and cached['size_mtime_ns'] == stats and target.exists():
                result = dict(np.load(str(target)))
                meta = cached
            else:
                result, meta = aggregate(paths, edges)
                assert stats == [[p.stat().st_size, p.stat().st_mtime_ns] for p in paths], 'Cohort changed during validation'
                meta.update(filename=first.name, size_mtime_ns=stats)
                np.savez_compressed(str(target), **result)
            included.append(meta)
            profiles.append(result)
        except (AssertionError, ValueError, OSError, KeyError, struct.error) as exc:
            excluded.append({'filename': first.name, 'reason': str(exc) or type(exc).__name__})
    ordering = sorted(range(len(included)), key=lambda i: (included[i]['time'], included[i]['cycle']))
    included = [included[i] for i in ordering]
    profiles = [profiles[i] for i in ordering]
    assert len({item['time'] for item in included}) == len(included), 'Duplicate simulation times'
    assert all(b['cycle'] > a['cycle'] for a, b in zip(included, included[1:])), 'Non-increasing cycles'
    if profiles:
        data = {key: np.array([p[key] for p in profiles]) for key in profiles[0]}
        data.update(time=np.array([p['time'] for p in included]), cycle=np.array([p['cycle'] for p in included]), edges=edges)
        np.savez_compressed(str(folder/'profiles.npz'), **data)
    manifest = {'case': case, 'source_run': str(run/case), 'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope': 'float32 active Theta; coordinate-volume weighting; no metric, proper volume, H/M/Q, or checkpoint validity inferred',
                'regions': REGIONS, 'included': included, 'excluded': excluded,
                'snapshots': len(included), 'latest_time': included[-1]['time'] if included else None}
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return {k: manifest[k] for k in ('case', 'snapshots', 'latest_time', 'excluded')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--case', action='append', metavar='NAME=PREFIX',
                        help='Restrict/override case subdirectory and file prefix; repeatable')
    args = parser.parse_args()
    cases = dict(item.split('=', 1) for item in args.case) if args.case else CASES
    args.output.mkdir(parents=True, exist_ok=True)
    status = [run_case(args.run, args.output, case, prefix) for case, prefix in cases.items()]
    (args.output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status))
