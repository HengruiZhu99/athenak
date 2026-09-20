"""Bounded responses of validated sigma=.3 directions; no sigma=1 eigensolve."""
from pathlib import Path
import sys, json
import numpy as np

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parents[1]/'mode-analysis'
sys.path.insert(0,str(OLD))
from mode_operator import Operator,SHAPE,ACTIVE,discrepancy

op=Operator(str(ROOT/'sigma1_directional_responses'),
            binary=ROOT.parent/'athena-covariant-modehook',
            input_file=ROOT/'input.athinput',dt=.0375,
            overrides=('z4c/ccz4_covariant_sources=true',
                       'z4c/damp_lapse_scaled=true','z4c/damp_kappa1=1'))
probe=Operator(str(ROOT/'sigma1_physical_probes'),
               binary=OLD/'athena-constraint-probe',
               input_file=OLD/'input-constraints.athinput')

def constraints(v,label):
    parts=[]
    for sign in (1,-1):
        lab=f'{label}_{sign}'
        probe.advance(sign*1e-3*v,0,label=lab)
        parts.append(np.fromfile(probe.directory/lab/'output.bin.constraints.bin').reshape(7,24,24,24))
    return (parts[0]-parts[1])/2e-3

def dx(v,axis):
    out=np.zeros((16,16,16))
    for off,c in [(-3,-1/60),(3,1/60),(-2,3/20),(2,-3/20),(-1,-3/4),(1,3/4)]:
        sl=[slice(4,20)]*3
        sl[2-axis]=slice(4+off,20+off)
        out+=c*v[tuple(sl)]/.25
    return out

def Q(v):
    g=[[1,2,3],[2,4,5],[3,5,6]]
    tr=v[1]+v[4]+v[6]
    return np.stack([v[14+i,4:20,4:20,4:20]
                     -sum(dx(v[g[i][j]],j) for j in range(3))
                     +.5*dx(tr,i) for i in range(3)])

def gains(a,b):
    projection=float(np.vdot(a,b).real/np.vdot(a,a).real)
    return {'norm_gain':float(np.linalg.norm(b)/np.linalg.norm(a)),
            'projected_gain':projection,
            'relative_shape_change':float(np.linalg.norm(b-projection*a)/np.linalg.norm(b))}

zero=op.advance(steps=20,label='zero20')
assert np.all(zero.view(np.uint64)==0)
records=[]
validated=json.loads((ROOT/'candidate-validated-modes.json').read_text())
assert len(validated)==2
for n,m in enumerate(json.loads((ROOT/'covariant_constant03-arnoldi32-results.json').read_text())['modes'][:2]):
    assert m['imag']==0
    assert validated[n]['eigen_residual']['relative_l2']<1e-4
    v=np.fromfile(m['state_real_file']).reshape(SHAPE)
    r=op.response(v,80,1e-3,label=f'mode{n}_eps0.001')
    low=op.response(v,80,3e-4,label=f'mode{n}_eps0.0003')
    ref=np.load(ROOT/f'covariant03-mode{n}-physical-constraints.npz')
    cr=constraints(r,f'mode{n}_advanced')
    rec={'sigma':1,'input_direction_sigma':.3,'mode_index':n,
         'interval_M':3,'zero20_all_bits_zero':True,
         'interpretation':'Transient response of a sigma=.3 eigenvector; not a sigma=1 eigenvalue.',
         'amplitude_convergence':discrepancy(r,low),
         'full_state':gains(v,r),'active_state':gains(v[ACTIVE],r[ACTIVE]),
         'physical_constraints':{}}
    for name,a,b in [('H',ref['H'].real,cr[1,4:20,4:20,4:20]),
                     ('M_cov',ref['M'].real,cr[4:7,4:20,4:20,4:20]),
                     ('Q_contrav',ref['Q'].real,Q(r))]:
        rec['physical_constraints'][name]=gains(a,b)
    r.tofile(ROOT/f'sigma1-covariant03-mode{n}-response.bin')
    np.savez_compressed(ROOT/f'sigma1-covariant03-mode{n}-physical-constraints.npz',
                        H=cr[1,4:20,4:20,4:20],M=cr[4:7,4:20,4:20,4:20],Q=Q(r))
    records.append(rec)
    (ROOT/'sigma1-directional-responses.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(rec,indent=2),flush=True)
