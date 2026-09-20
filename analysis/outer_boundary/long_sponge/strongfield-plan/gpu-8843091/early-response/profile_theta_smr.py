#!/usr/bin/env python3
"""Read-only, fail-closed profiles of completed 24-rank static SMR Theta dumps.

Float32 active cells only; coordinate volume, not proper metric volume.
Interface bands mean <= two local cell widths from a nested cubic interface.
They overlap radial regions. This does not inspect ghosts or establish validity
of the other metric fields, nor locate the first RK-stage injection.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import struct
import numpy as np


def parse(path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed while reading'
    f = io.BytesIO(raw)
    assert f.readline() == b'Athena binary output version=1.1\n'
    head = dict(f.readline().decode().strip().split('=', 1) for unused in range(6))
    assert head['size of preheader'] == '5'
    names = f.readline().decode().strip().split(':', 1)[1].split()
    assert names == ['z4c_Theta'] and int(head['number of variables']) == 1
    assert head['size of location'] == '8' and head['size of variable'] == '4'
    nheader = int(f.readline().decode().split('=', 1)[1])
    inp = f.read(nheader)
    assert len(inp) == nheader
    blocks = []
    while f.tell() < len(raw):
        block_head = f.read(40)
        assert len(block_head) == 40, 'Incomplete block header'
        indices = struct.unpack('<6i', block_head[:24])
        loc = struct.unpack('<4i', block_head[24:])
        assert indices == (4, 19, 4, 19, 4, 19), 'Expected complete active 16^3 block'
        bb = f.read(48)
        assert len(bb) == 48
        limits = struct.unpack('<6d', bb)
        bb = f.read(16**3*4)
        assert len(bb) == 16**3*4, 'Incomplete block payload'
        theta = np.frombuffer(bb, dtype='<f4').astype(np.float64).reshape(16,16,16)
        assert np.isfinite(theta).all(), 'Nonfinite Theta'
        blocks.append((loc, limits, theta))
    return head, hashlib.sha256(inp).hexdigest(), blocks, hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('audit', type=Path)
    ap.add_argument('--max-index', type=int, default=10)
    args = ap.parse_args()
    audit = json.loads(args.audit.read_text())['cases']['mesh16_r1']
    blocks = audit['blocks_geometry']
    lookup = {tuple(b['logical_location'])+(b['logical_level']-2,): b for b in blocks}
    assert len(lookup) == 232
    nper = [9]*8+[10]*16
    expected_ranks = {gid: rank for rank in range(24) for gid in range(sum(nper[:rank]),sum(nper[:rank+1]))}
    edges = np.r_[np.arange(0,8,.25),np.arange(8,56.01,.5)]
    included, excluded = [], []
    candidates = sorted((args.run/'bin/rank_00000000').glob('*.z4c_Theta.*.bin'))
    for first in candidates:
        if int(first.name.split('.')[-2]) > args.max_index:
            continue
        try:
            paths = [args.run/'bin'/('rank_%08d'%r)/first.name for r in range(24)]
            assert set((args.run/'bin').glob('rank_*/'+first.name)) == set(paths), 'Missing/extra ranks'
            stats = [(p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
            regions = {}
            rad_v, rad_i2, rad_max = [np.zeros(len(edges)-1) for i in range(3)]
            reference, seen, hashes = None, set(), []
            for rank,path in enumerate(paths):
                head, ihash, entries, digest = parse(path)
                signature = (head,ihash)
                if reference is None:
                    reference = signature
                assert signature == reference, 'Cohort time/cycle/header mismatch'
                assert len(entries) == nper[rank], 'Incorrect rank block count'
                hashes.append(digest)
                for loc,limits,theta in entries:
                    assert loc in lookup and loc not in seen, 'Unknown/duplicate logical block'
                    seen.add(loc)
                    b = lookup[loc]
                    assert expected_ranks[b['gid']] == rank, 'Wrong gid ownership'
                    expect = tuple(v for lo,hi in zip(b['min'],b['max']) for v in (lo,hi))
                    assert limits == expect, 'Geometry disagrees with mesh audit'
                    dx = float(b['dx_M'])
                    axes = [b['min'][i]+(np.arange(16)+.5)*dx for i in range(3)]
                    z,y,x = np.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
                    xyz = (x,y,z)
                    radius = np.sqrt(x*x+y*y+z*z)
                    cube = np.maximum.reduce([abs(x),abs(y),abs(z)])
                    masks = {'whole':np.ones(theta.shape,bool), 'inside_horizon_r_le_1':radius<=1,
                             'near_horizon_1_lt_r_lt_2':(radius>1)&(radius<2),
                             'protected_r_le_8':radius<=8,'ramp_8_lt_r_lt_28':(radius>8)&(radius<28),
                             'full_sponge_r_ge_28':radius>=28,'outer_face_distance_le_2':cube>=30,
                             'relative_level_%d'%loc[3]:np.ones(theta.shape,bool)}
                    for q in (4,8,16):
                        masks['cubic_interface_%d_band_2dx'%q] = abs(cube-q)<=2*dx
                    for name,mask in masks.items():
                        r = regions.setdefault(name,dict(cells=0,coordinate_volume=0.,integral_theta2=0.,maxabs=-1.,peak=None))
                        r['cells'] += int(mask.sum()); r['coordinate_volume'] += float(mask.sum())*dx**3
                        r['integral_theta2'] += float(np.sum(theta[mask]**2))*dx**3
                        if not mask.any():
                            continue
                        index = np.unravel_index(np.argmax(np.where(mask,abs(theta),-1)),theta.shape)
                        value = float(theta[index])
                        if abs(value)>r['maxabs']:
                            r['maxabs'] = abs(value)
                            r['peak'] = dict(value=value,rank=rank,gid=b['gid'],relative_level=loc[3],logical_level=b['logical_level'],xyz=[float(a[index]) for a in xyz],radius=float(radius[index]),dx=dx)
                    bins = np.searchsorted(edges,radius.ravel(),side='right')-1
                    assert np.all((bins>=0)&(bins<len(rad_v)))
                    rad_v += np.bincount(bins,weights=np.full(theta.size,dx**3),minlength=len(rad_v))
                    rad_i2 += np.bincount(bins,weights=theta.ravel()**2*dx**3,minlength=len(rad_v))
                    np.maximum.at(rad_max,bins,abs(theta).ravel())
            assert len(seen)==232 and regions['whole']['cells']==950272
            assert regions['whole']['coordinate_volume']==64**3
            assert stats == [(p.stat().st_size,p.stat().st_mtime_ns) for p in paths], 'Cohort changed while reading'
            for region in regions.values():
                region['coordinate_rms'] = (region['integral_theta2']/region['coordinate_volume'])**.5 if region['coordinate_volume'] else None
            included.append(dict(filename=first.name,time_M=float(reference[0]['time']),cycle=int(reference[0]['cycle']),ranks=24,blocks=232,all_finite=True,matching_headers=True,rank_sha256=hashes,regions=regions,radial_coordinate_volume=rad_v.tolist(),radial_integral_theta2=rad_i2.tolist(),radial_maxabs=rad_max.tolist()))
        except (AssertionError,ValueError,OSError,KeyError,struct.error) as e:
            excluded.append(dict(filename=first.name,reason=str(e) or type(e).__name__))
    print(json.dumps(dict(collected_utc=datetime.datetime.utcnow().isoformat()+'Z',source=str(args.run),scope=__doc__,radial_edges=edges.tolist(),included=included,excluded=excluded),allow_nan=False))


if __name__ == '__main__':
    main()
