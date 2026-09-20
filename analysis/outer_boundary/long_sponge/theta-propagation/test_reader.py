#!/usr/bin/env python3
"""Synthetic format/coverage regression, not a numerical-evolution test."""
import struct
from pathlib import Path
import tempfile
import numpy as np
from aggregate import aggregate

params = '''<problem>
bh_mass=0
<mesh_refinement>
refinement=none
<mesh>
nx1=64
nx2=64
nx3=64
x1min=-2048
x2min=-2048
x3min=-2048
x1max=2048
x2max=2048
x3max=2048
<meshblock>
nx1=32
nx2=32
nx3=32
'''.encode()


def make(loc, cycle=1, value=2.):
    header = ('Athena binary output version=1.1\n  size of preheader=5\n'
              '  time=3.2\n  cycle=%d\n  size of location=8\n  size of variable=4\n'
              '  number of variables=1\n  variables: z4c_Theta\n  header offset=%d\n' % (cycle, len(params))).encode()
    limits = tuple(v for j in loc for v in (-2048+2048*j, 2048*j))
    block = struct.pack('<10i6d', 4,35,4,35,4,35,*loc,0,*limits)
    return header+params+block+np.full(32**3, value, dtype='<f4').tobytes()


with tempfile.TemporaryDirectory() as tmp:
    paths, originals = [], []
    for rank in range(8):
        p = Path(tmp)/('%d.bin' % rank)
        raw = make((rank%2, (rank//2)%2, rank//4))
        p.write_bytes(raw)
        paths.append(p)
        originals.append(raw)
    result, meta = aggregate(paths, np.arange(0., 3648., 64.))
    assert meta['active_cells'] == 64**3 and meta['unique_blocks'] == 8
    assert np.all(result['mean'][result['count'] > 0] == 2.)
    assert np.all(result['rms'][result['count'] > 0] == 2.)
    assert result['regional_integral2'][-1] == 4*4096**3
    bad = [originals[-1][:-1], make((1,1,1), cycle=2), make((0,0,0)), make((1,1,1), value=np.nan)]
    for raw in bad:
        paths[-1].write_bytes(raw)
        try:
            aggregate(paths, np.arange(0.,3648.,64.))
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid cohort accepted')
    print('PASS: full constant-field coverage; reject truncated, mismatched-cycle, duplicate-block, nonfinite cohorts')
