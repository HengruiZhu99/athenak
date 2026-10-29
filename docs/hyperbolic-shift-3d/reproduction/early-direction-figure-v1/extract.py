"""Extract a small plotting table from verified V4 diagnostic captures."""
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

root = Path(__file__).resolve().parents[4]
source = Path(__file__).with_name('comparison-input.json')
comparison = json.loads(source.read_text())
data = dict(comparison_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            comparison_source=str(source.relative_to(root)), snapshots=[],
            interpretation='Diagnostic captures of an unstable scheme; no eigenmode or stability admission.')
for metadata, pattern in zip(comparison['inputs'], comparison['angular_patterns']):
    path = Path(metadata['path'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == metadata['sha256']
    worker = json.loads((path.parent.parent/'worker-status.json').read_text())
    binding = [v for c in worker['cases'] for k,v in c['artifacts'].items()
               if k == metadata['bound_artifact']]
    assert len(binding) == 1 and binding[0]['sha256'] == digest
    assert binding[0]['bytes'] == path.stat().st_size
    with path.open('rb') as stream:
        magic,n,nf,kind,admitted,cells,ns,steps,t,dt = struct.unpack('<16s4i3Q2d',stream.read(72))
        epsilon,degree,reserved = struct.unpack('<dii',stream.read(16))
        assert magic.rstrip(b'\0') == b'HYP3D_MPI_V4'
        assert (n,nf,kind,epsilon,degree,reserved) == (16,31,0,2,7,0)
        assert cells == 19*n**3 and ns == 31*6*n*n
        volume = np.fromfile(stream,dtype='<f8',count=31*cells).reshape(19,n,n,n,31)
        stream.seek(8*31*cells,1)
        surface = np.fromfile(stream,dtype='<f8',count=ns).reshape(6,n,n,31)
    # Exact reference solid-angle weights, identical to the native Q plan.
    edges = np.linspace(-1,1,n+1)
    u,v = np.meshgrid(edges,edges)
    primitive = np.arctan2(u*v,np.sqrt(1+u*u+v*v))
    area = np.diff(np.diff(primitive,axis=0),axis=1)
    assert np.all(area > 0) and abs(6*np.sum(area)-4*np.pi) < 2e-14
    frobenius = np.array([1,2,2,1,2,1])
    anchor = surface[:,:,:,7:13]
    weight = area[None,:,:,None]*frobenius
    denominator = np.sum(weight*anchor*anchor)
    coefficients = [float(np.sum(weight*anchor*volume[13:19,:,:,i,7:13])/denominator)
                    for i in range(n)]
    data['snapshots'].append(dict(time=t,steps=steps,checkpoint_sha256=digest,
        worker_sha256=metadata['worker_sha256'],
        raw_volume_max=metadata['raw_volume_maximum_deviation'],
        raw_surface_max=metadata['raw_surface_maximum_deviation'],
        surface_Q_energy_quotient=pattern['surface']['Q2_damping_rayleigh'],
        surface_B_relative=pattern['surface']['B_relative_norm'],
        radial_centers=(.75+(np.arange(n)+.5)*.25/n).tolist(),
        A_signed_projection_on_surface=coefficients,
        native_admitted=bool(admitted),scientific_restart_admitted=False))
data['radial_measure']='Sum over identical angular nodes with exact reference solid-angle weights and tensor Frobenius weights. Coefficient=<A_surface,A_plane>/<A_surface,A_surface>; surface=1. This is a signed component correlation, not physical energy.'
data['growth_fits']=[dict(first=p['first'],second=p['second'],**p['metrics']['all'])
                     for p in comparison['shape_comparisons']]
Path(__file__).with_name('data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
