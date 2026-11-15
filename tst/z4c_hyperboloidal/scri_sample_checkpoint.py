#!/usr/bin/env python3
"""Read-only common-point samples of outer fields from the static A/B campaign.

Subtract each grid's initial reference before interpolating, including the native
chi/r^2 residual representation. Return physical field *differences from that
reference*, whose cross-resolution differences equal state differences. Sample
all six faces away from seams at fixed radii, including the separate scri state.
Two polynomial widths expose postprocessing sensitivity. This intentionally
rejects outer AMR or a different atlas instead of guessing its layout.
"""
import argparse, hashlib, json, re, struct
from pathlib import Path
import numpy as np
from restart_roundtrip import parse, payload

# Offset the common sampling lattice from every evolved radial grid. Otherwise
# coincident N12 nodes can misleadingly make its interpolation look exact while
# the finer grids are interpolated. Keep the evolved endpoint separately.
RADII = np.r_[np.linspace(.9431, .9983, 16), 1.]
ANGLES = np.linspace(.05, .95, 9)  # local logical angles; equiangular map

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def weights(nodes, targets, width):
    assert width <= len(nodes)
    w = np.zeros((len(targets), len(nodes)))
    for row, x in enumerate(targets):
        # Fixed local polynomial, biased only where there are fewer neighbors.
        start = min(max(int(np.searchsorted(nodes, x))-width//2, 0), len(nodes)-width)
        indices = np.arange(start, start+width)
        for i in indices:
            value = 1.
            for j in indices:
                if i != j:
                    value *= (x-nodes[j])/(nodes[i]-nodes[j])
            w[row, i] = value
    assert np.max(abs(w.sum(axis=1)-1)) < 1e-12
    return w

def outer_fields(info):
    identity = info['identity'].decode()
    match = re.search(r';n=(\d+),(\d+),(\d+);overlap=(\d+);', identity)
    n1,n2,n3,o = map(int, match.groups())
    assert n1 == n2 == n3 and o == 3
    n = n1
    atlas = dict(s.split('=',1) for s in info['atlas'].decode().split(';')[1:])
    assert atlas['angular'] == '1,1,' and atlas['outer_endpoint'] == '1'
    assert atlas['equiangular'] == '1' and atlas['overlap'] == '3'
    radii = [float.fromhex(x) for x in atlas['radii'].split(',') if x]
    assert radii[-2:] == [.94, 1.]
    with Path(info['path']).open('rb') as stream:
        stream.seek(info['start']+4); root, = struct.unpack('=i',stream.read(4))
    locations = np.frombuffer(info['topology'], dtype='=i4').reshape(-1,5)
    assert info['record_blocks'] == info['blocks']  # global native checkpoint
    data = payload(info)
    bulk_slots = (n+2*o)**3
    result = []
    for face in range(6):
        selected = np.flatnonzero((locations[:,4] == face+1) &
                                  (locations[:,0] == len(radii)-1))
        assert len(selected) == 1
        gid = selected[0]
        assert np.array_equal(locations[gid], [len(radii)-1,0,0,root,face+1])
        bulk = data[gid,:bulk_slots,:31].reshape(n+2*o,n+2*o,n+2*o,31)
        surface = data[gid,bulk_slots:,:31].reshape(n,n,31)
        fields = np.concatenate((bulk[o:o+n,o:o+n,o:o+n,:],surface[:,:,None,:]),axis=2)
        assert np.isfinite(fields).all() and np.all(fields[:,:,:,0]>0)
        result.append(fields)
    return np.array(result), n

def sample(path, initial):
    info, ref = parse(Path(path)), parse(Path(initial))
    assert ref['time'] == 0 and ref['cycle'] == 0
    for key in ('identity','atlas','topology','stride'):
        assert info[key] == ref[key], key
    u,n = outer_fields(info); zero,n0 = outer_fields(ref)
    assert n == n0
    delta = u-zero
    radial = .94+np.arange(n+1)*(.06/n)
    radial[-1] = 1.
    angular = (np.arange(n)+.5)/n
    delta[:,:,:,:,0] /= radial[None,None,None,:]**2
    outputs = {}
    for width in (6,8):
        wr = weights(radial, RADII, width+1)
        wa = weights(angular, ANGLES, width)
        values = np.einsum('ar,bk,cj,fkjrv->fabcv',wr,wa,wa,delta,optimize=True)
        values[:,:,:,:,0] *= RADII[None,:,None,None]**2
        outputs['fields_w'+str(width)] = values
    return info, n, outputs

def coordinates():
    out = np.zeros((6,len(RADII),len(ANGLES),len(ANGLES),3))
    for face in range(6):
        a,b,c = face//2,(face//2+1)%3,(face//2+2)%3
        for k,qk in enumerate(ANGLES):
            for j,qj in enumerate(ANGLES):
                y,z = np.tan(np.pi*(2*qj-1)/4),np.tan(np.pi*(2*qk-1)/4)
                scale = RADII/np.sqrt(1+y*y+z*z)
                out[face,:,k,j,a] = (-1 if face%2 else 1)*scale
                out[face,:,k,j,b] = y*scale;out[face,:,k,j,c] = z*scale
    return out

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    info,n,outputs=sample(args.checkpoint,args.initial)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,coordinates=coordinates(),radii=RADII,angles=ANGLES,**outputs)
    identity=info['identity'].decode()
    common_identity=re.sub(r';n=\d+,\d+,\d+|;p.requested_dt=[^;]*','',identity)
    meta=dict(checkpoint=str(args.checkpoint),initial=str(args.initial),n=n,
              time_M0=3*info['time'],cycle=info['cycle'],
              physics_identity_without_resolution_and_dt=common_identity,
              checkpoint_sha256=digest(args.checkpoint),initial_sha256=digest(args.initial),
              sampled_field='stored 31 Cartesian components minus their initial reference',
              shape=list(outputs['fields_w6'].shape),
              max_width_sensitivity=float(np.max(abs(outputs['fields_w6']-outputs['fields_w8']))))
    args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
