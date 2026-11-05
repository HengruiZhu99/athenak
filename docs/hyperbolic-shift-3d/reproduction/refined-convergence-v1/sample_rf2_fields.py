"""Read-only common-coordinate sampling for admitted RF2 checkpoint comparisons.

This samples stored Cartesian components, not constraints or physical-volume
norms. Six/eight-point tensor interpolation is analysis only. Compare both
orders before interpreting field self-convergence; never sample the puncture
and call it an exterior error. Admission and exact SHA/identity are mandatory.
"""
from pathlib import Path
import hashlib
import math
import struct
import numpy as np

FIELDS = ('chi gxx-1 gxy gxz gyy-1 gyz gzz-1 Axx Axy Axz Ayy Ayz Azz P+3 '
          'Theta Lambda_x Lambda_y Lambda_z alpha-(1+r2)/2 B_x B_y B_z '
          'beta_x+x beta_y+y beta_z+z D_x D_y D_z J_x J_y J_z').split()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8388608), b''):
            h.update(b)
    return h.hexdigest()

class Checkpoint:
    def __init__(self, path, *, sha256, identity):
        self.path = Path(path).resolve()
        assert digest(self.path) == sha256, 'Checkpoint differs from admitted hash'
        with self.path.open('rb') as f:
            def get(fmt):
                n = struct.calcsize('<' + fmt)
                return struct.unpack('<' + fmt, f.read(n))
            def cstring():
                b = bytearray()
                while len(b) < 65536:
                    c = f.read(1)
                    assert c, 'Truncated string'
                    if c == b'\0':
                        return b.decode()
                    b.extend(c)
                raise ValueError('Oversized checkpoint string')
            self.tag = cstring()
            assert self.tag.startswith('HYP3D_AMR_RF2;')
            size, = get('Q'); assert size < 65536
            self.identity = f.read(size).decode()
            assert self.identity == identity, 'Different scientific configuration'
            assert [cstring() for _ in range(31)] == FIELDS
            self.n, self.kind, self.subdivisions, self.steps, self.regrids = get('iiiQQ')
            self.time, self.dt, self.seed = get('ddd')
            self.tracker = get('3d'); self.floors = get('8d'); self.signs = get('8d')
            count, = get('Q'); assert 8 <= count <= 1024
            self.leaves = np.frombuffer(f.read(count*16), dtype='<i4').reshape(count, 4).copy()
            self.ages = np.frombuffer(f.read(count*4), dtype='<i4').copy()
            self.arrays = []
            for _ in range(8):
                count, = get('Q')
                assert count <= 31*1024*(self.n+8)**3
                offset = f.tell(); assert offset + count*8 <= self.path.stat().st_size
                self.arrays.append(np.memmap(self.path, mode='r', dtype='<f8',
                                             offset=offset, shape=(count,)) if count else np.array([]))
                f.seek(count*8, 1)
            assert get('Q')[0] == 0x414d5231434f4d50
            self.fnv, = get('Q')
            assert f.tell() == self.path.stat().st_size
        # SHA binds the same bytes whose FNV/SPD/finite checks passed admission.
        # This reader does not replace that native admission or modify the file.

class Geometry:
    def __init__(self, leaves, subdivisions, config):
        values = {}
        for line in Path(config).read_text().splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                k, v = line.split('=', 1); values[k.strip()] = v.strip()
        self.half = float(values['core_half_width'])
        self.inner = float(values['shell_inner_radius'])
        self.radii = np.array([float(x) for x in values['shell_radii'].split(',')])
        self.factors = [int(x) for x in values.get('shell_angular_subdivisions',
                         ','.join('1' for _ in self.radii)).split(',')]
        assert len(self.factors) == len(self.radii) and set(self.factors) <= {1, 2}
        root_level = int(math.log2(subdivisions)); assert 2**root_level == subdivisions
        self.blocks = []
        for lx, ly, lz, level in leaves:
            refinement = int(level)-root_level
            assert refinement >= 0
            width = 2*self.half/(subdivisions*2**refinement)
            loc = np.array([lx, ly, lz])
            lo = -self.half+loc*width; hi = -self.half+(loc+1)*width
            self.blocks.append((-1, -1, refinement, lo, hi))
        self.core_count = len(self.blocks)
        for layer, factor in enumerate(self.factors):
            for face in range(6):
                for v in range(factor):
                    for u in range(factor):
                        lo = np.array([0., -1+2*u/factor, -1+2*v/factor])
                        hi = np.array([1., -1+2*(u+1)/factor, -1+2*(v+1)/factor])
                        self.blocks.append((face, layer, 0, lo, hi))

    def inverse(self, block, x):
        face, layer, _, lo, hi = self.blocks[block]
        x = np.asarray(x, dtype=float)
        if face < 0:
            q = x
        else:
            a = face//2; scale = (-1 if face % 2 else 1)*x[a]
            assert scale > 0
            lower = self.inner if layer == 0 else self.radii[layer-1]
            q = np.array([(math.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])-lower)/(self.radii[layer]-lower),
                          4*math.atan(x[(a+1) % 3]/scale)/math.pi,
                          4*math.atan(x[(a+2) % 3]/scale)/math.pi])
        return (q-lo)/(hi-lo)

    def owner(self, x):
        x = np.asarray(x, dtype=float); r = math.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
        assert 0 < r <= self.radii[-1]*(1+2e-14)
        if np.max(np.abs(x)) <= self.half*(1+2e-14):
            face = layer = -1
            candidates = range(self.core_count)
        else:
            a = int(np.argmax(np.abs(x))); face = 2*a+int(x[a] < 0)
            layer = int(np.searchsorted(self.radii, min(r, self.radii[-1])))
            candidates = range(self.core_count, len(self.blocks))
        selected = None
        for m in candidates:
            f, l, level, _, _ = self.blocks[m]
            if (f, l) != (face, layer):
                continue
            q = self.inverse(m, x)
            if np.all(q >= -2e-13) and np.all(q <= 1+2e-13):
                if selected is None or level > self.blocks[selected][2]:
                    selected = m
        assert selected is not None, 'No native nominal owner'
        return selected, self.inverse(selected, x)

def weights(q, n, stencil):
    assert stencil in (6, 8) and n >= stencil
    x = q*n-.5
    start = min(max(math.floor(x)-(stencil//2-1), 0), n-stencil)
    indices = np.arange(start, start+stencil)
    w = np.ones(stencil)
    for j in range(stencil):
        for k in range(stencil):
            if j != k:
                w[j] *= (x-indices[k])/(indices[j]-indices[k])
    return indices, w

def sample(cp, geometry, points, stencil=8):
    nominal = len(geometry.blocks)*cp.n**3
    assert len(cp.arrays[0]) >= 31*nominal
    fields = cp.arrays[0][:31*nominal].reshape(len(geometry.blocks), cp.n, cp.n, cp.n, 31)
    result, metadata = [], []
    for x in points:
        m, q = geometry.owner(x)
        ix, wx = weights(q[0], cp.n, stencil)
        iy, wy = weights(q[1], cp.n, stencil)
        iz, wz = weights(q[2], cp.n, stencil)
        values = fields[m][np.ix_(iz, iy, ix)]
        result.append(np.einsum('i,j,k,kjif->f', wx, wy, wz, values))
        metadata.append(dict(block=m, face=geometry.blocks[m][0],
                             layer=geometry.blocks[m][1], q=q.tolist(),
                             weight_l1=float(abs(wx).sum()*abs(wy).sum()*abs(wz).sum())))
    result = np.array(result)
    assert np.isfinite(result).all()
    return result, metadata
