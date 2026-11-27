#!/usr/bin/env python3
"""Version-1 AthenaK native scri archive reader (Python + NumPy only).

No Bondi/inertial frame, radiation extraction, or time derivative is implied.
"""
from pathlib import Path
import argparse
import json
import struct
import numpy as np

INDEPENDENT = [f for f in range(28) if f not in (6, 12)]


def read(path):
    """Read a complete typed archive; reject truncation and duplicate keys."""
    result = {}
    with Path(path).open('rb') as stream:
        def take(n):
            data = stream.read(n)
            if len(data) != n:
                raise ValueError('truncated scri archive')
            return data
        if take(8) != b'SCRI0001':
            raise ValueError('unsupported scri schema')
        marker = take(4)
        endian = {b'\x04\x03\x02\x01': '<', b'\x01\x02\x03\x04': '>'}.get(marker)
        if endian is None:
            raise ValueError('invalid endian marker')
        def integer(code):
            return struct.unpack(endian + code, take(struct.calcsize(code)))[0]
        while True:
            length = integer('I')
            if length == 0:
                break
            if length > 4096:
                raise ValueError('invalid array name length')
            kind, ndim = integer('I'), integer('I')
            if kind not in (1, 2, 3) or ndim > 8:
                raise ValueError('invalid type or dimensionality')
            shape = tuple(integer('Q') for _ in range(ndim))
            name = take(length).decode('utf8')
            if name in result:
                raise ValueError('duplicate array name')
            count = 1
            for extent in shape:
                count *= extent
            size = count * (1 if kind == 3 else 8)
            if size > Path(path).stat().st_size - stream.tell():
                raise ValueError('invalid array extent')
            data = take(size)
            result[name] = (data.decode('utf8') if kind == 3 else
                            np.frombuffer(data, dtype=endian + ('f8' if kind == 1 else 'i8')).reshape(shape))
        if stream.read(1):
            raise ValueError('trailing archive data')
    return result


def load(path):
    """Return (accepted_state, topology_epoch), resolving portable relative links."""
    path = Path(path)
    state = read(path)
    if state.get('kind') != 'native scri accepted state':
        raise ValueError('expected accepted-state archive')
    epoch = Path(state['epoch'])
    if not epoch.is_absolute():
        epoch = path.parent / epoch
    return state, read(epoch)


def load_record(paths):
    """Load one distributed record and reject missing/mixed/duplicate ranks."""
    parts = [load(path) for path in paths]
    if not parts:
        raise ValueError('empty record')
    first = parts[0][0]
    ranks = int(first['record'][4])
    if len(parts) != ranks or sorted(int(s['record'][3]) for s, _ in parts) != list(range(ranks)):
        raise ValueError('missing or duplicate ranks')
    for state, epoch in parts:
        if (not np.array_equal(state['record'][[0, 1, 2, 4, 5, 6, 7]], first['record'][[0, 1, 2, 4, 5, 6, 7]])
                or not np.array_equal(state['time'], first['time'])
                or epoch['fields'] != parts[0][1]['fields']
                or epoch['initial_data_identity'] != parts[0][1]['initial_data_identity']):
            raise ValueError('mixed distributed record')
    return sorted(parts, key=lambda part: int(part[0]['record'][3]))


def residual_support(state, epoch, name):
    values = np.zeros_like(state[name + '_state'])
    values[:, INDEPENDENT] = (state[name + '_state'] - epoch[name + '_reference'][:, :, 0])[:, INDEPENDENT]
    values[:, 0] /= epoch[name + '_residual_weight']
    return values


def reconstruct_sampled_jets(state, epoch):
    """Apply the actual native endpoint rows, including seam/refinement donors."""
    bulk, edge = (residual_support(state, epoch, n) for n in ('bulk', 'edge'))
    ids = epoch['term_ids'].astype(np.int64)
    weights = epoch['term_weights']
    out = np.zeros((len(epoch['rows']), 31, 13))
    for point, (begin, end, _) in enumerate(epoch['rows']):
        begin, end = int(begin), int(end)
        boundary, ordinal = ids[begin]
        anchor = (edge if boundary else bulk)[ordinal]
        out[point, :, 0] = anchor
        # Match the native anchor-difference and summation ordering.
        for t in range(begin, end):
            boundary, ordinal = ids[t]
            delta = (edge if boundary else bulk)[ordinal] - anchor
            out[point] += delta[:, None] * weights[t]
    return out


def constant(value):
    value = np.asarray(value)
    out = np.zeros(value.shape + (13,))
    out[..., 0] = value
    return out


def multiply(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.zeros_like(a)
    out[..., 0] = a[..., 0] * b[..., 0]
    out[..., 1:4] = a[..., 1:4]*b[..., :1] + a[..., :1]*b[..., 1:4]
    h = (a[..., 4:].reshape(a.shape[:-1]+(3, 3))*b[..., 0, None, None]
         + b[..., 4:].reshape(b.shape[:-1]+(3, 3))*a[..., 0, None, None]
         + a[..., 1:4, None]*b[..., None, 1:4]
         + b[..., 1:4, None]*a[..., None, 1:4])
    out[..., 4:] = h.reshape(out.shape[:-1]+(9,))
    return out


def power(a, p):
    out = np.zeros_like(a)
    out[..., 0] = a[..., 0]**p
    d = p*a[..., 0]**(p-1)
    dd = p*(p-1)*a[..., 0]**(p-2)
    out[..., 1:4] = d[..., None]*a[..., 1:4]
    out[..., 4:] = (d[..., None, None]*a[..., 4:].reshape(a.shape[:-1]+(3, 3))
                   + dd[..., None, None]*a[..., 1:4, None]*a[..., None, 1:4]).reshape(a.shape[:-1]+(9,))
    return out


def expand(jets):
    """Restore determinant-one gzz and trace-free Azz, including second jets."""
    v = jets.copy()
    a, b, c, d, e = v[..., 1, :]+constant(1.), v[..., 2, :], v[..., 3, :], v[..., 4, :]+constant(1.), v[..., 5, :]
    mul = multiply
    inv = power(mul(a, d)-mul(b, b), -1)
    f = mul(constant(1.)-2*mul(mul(b, c), e)+mul(a, mul(e, e))+mul(d, mul(c, c)), inv)
    v[..., 6, :] = f-constant(1.)
    v[..., 12, :] = -mul(mul(mul(d, f)-mul(e, e), v[..., 7, :])
        + 2*mul(mul(c, e)-mul(b, f), v[..., 8, :])
        + 2*mul(mul(b, e)-mul(c, d), v[..., 9, :])
        + mul(mul(a, f)-mul(c, c), v[..., 10, :])
        + 2*mul(mul(b, c)-mul(a, e), v[..., 11, :]), inv)
    return v


def close_dependent(jets, sampled):
    """Restore gzz and Azz of endpoint jets.

    By default the solver leaves the sampled residual jets of slots 6 and 12 at zero and
    both components follow from det(gamma)=1 and gamma^ij A_ij=0. A record written with
    evolve_all_components carries their differenced jets, which are then used as stored.
    """
    if np.any(sampled[..., [6, 12], :] != 0):
        return jets
    return expand(jets)


def normal_lifts(state, epoch):
    """Rebuild the six-by-six angular interpolant and seventh-degree normal lift."""
    ids = epoch['surface_ids'].astype(int)
    n = int(epoch['configuration'][0])
    corrections = state['normal_correction']
    nc = corrections.shape[1]
    out = np.zeros((len(ids), nc, 13))
    for row, (_, face, k, j, _) in enumerate(ids):
        geometry = epoch['surface_geometry'][row]
        x, J, H = geometry[:3], geometry[3:12].reshape(3, 3), geometry[12:].reshape(3, 3, 3)
        first = [min(max(j-2, 0), n-6), min(max(k-2, 0), n-6)]
        bases = []
        for axis, index in enumerate((j, k)):
            q = constant((index+.5)/n);q[axis+2] = 1
            weights = []
            for i in range(6):
                b = constant(1.)
                for l in range(6):
                    if l != i:
                        b = multiply(b, q-constant((first[axis]+l+.5)/n)) * (n/(i-l))
                weights.append(b)
            bases.append(weights)
        logical = np.zeros((nc, 13))
        for v in range(6):
            for u in range(6):
                donor = (face*n+first[1]+v)*n+first[0]+u
                logical += corrections[donor, :, None]*multiply(bases[0][u], bases[1][v])
        cart = logical.copy()
        cart[:, 1:4] = logical[:, 1:4] @ J
        cart[:, 4:] = (np.einsum('cA,Aij->cij', logical[:, 1:4], H)
             + np.einsum('Ai,Bj,cAB->cij', J, J, logical[:, 4:].reshape(nc, 3, 3))).reshape(nc, 9)
        r2 = constant(0.)
        for a in range(3):
            coord = constant(x[a]);coord[1+a] = 1
            r2 += multiply(coord, coord)
        r = power(r2, .5)
        dr, gap = epoch['faces'][face, 1:3]
        z = (r-constant(1.))/dr
        h = z.copy();normalizer = 1.
        for i in range(6):
            h = multiply(h, z+constant(i+gap));normalizer *= i+gap
        out[row] = multiply(h*(dr/normalizer), cart)
    return out


def restored_endpoint_jets(state, epoch):
    """Native chart jets with reference/dependent restoration and actual normal lift."""
    sampled = reconstruct_sampled_jets(state, epoch)
    ref = epoch['surface_reference']
    x = epoch['surface_geometry'][:, :3]
    r2 = constant(np.sum(x*x, axis=1));r2[:, 1:4] = 2*x
    r2[:, [4, 8, 12]] = 2
    jets = sampled + ref
    # This exactly follows SurfaceExchange::Solve's default Restore convention.
    jets[:, 0] = multiply(sampled[:, 0]+multiply(ref[:, 0], power(r2, -1)), r2)
    jets = close_dependent(jets, sampled)
    lifts = normal_lifts(state, epoch)
    jets[:, 22:25] += lifts[:, :3]
    if lifts.shape[1] == 4:
        jets[:, 18] += lifts[:, 3]
    return jets


def physical_values(native, x):
    """Undo storage offsets only; these remain conformal, native-frame fields."""
    out = expand(constant(native))[..., 0]
    out[..., [1, 4, 6]] += 1
    out[..., 13] -= 3  # P = K - 2 Theta
    out[..., 18] += (1+np.sum(x*x, axis=-1))/2
    out[..., 22:25] -= x
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    state, epoch = load(args.path)
    report = dict(cycle=int(state['record'][0]), time=float(state['time'][0]),
                  rank=int(state['record'][3]), points=len(epoch['surface_ids']),
                  fields=epoch['fields'].splitlines(), bytes=args.path.stat().st_size)
    if args.verify:
        recovered = reconstruct_sampled_jets(state, epoch)
        error = np.max(np.abs(recovered-state['sampled_residual_jets']), initial=0.)
        if not np.allclose(recovered, state['sampled_residual_jets'], rtol=2e-12, atol=2e-12):
            raise ValueError(f'endpoint reconstruction mismatch: {error}')
        report['endpoint_jet_max_error'] = float(error)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
