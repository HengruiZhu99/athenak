"""Read-only SMR coverage audit; reference drift is not an evolved trajectory."""
from pathlib import Path
import json, hashlib, struct, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists())
DOC=Path(__file__).resolve().parent
CP=ROOT/'output/refined-convergence/reference-v2-stationary-n16-mpi2.checkpoint'
OUT=DOC/'refinement-coverage';OUT.mkdir(exist_ok=True)
expected=json.loads((DOC/'stationary-reference-v2-checks/verification.json').read_text())
b=CP.read_bytes();assert hashlib.sha256(b).hexdigest()==expected['checkpoint_sha256']
at=b.index(b'\0')+1
assert b[:at-1].startswith(b'HYP3D_AMR_RF2;')
def get(fmt):
 global at
 values=struct.unpack_from('<'+fmt,b,at);at+=struct.calcsize('<'+fmt)
 return values[0] if len(values)==1 else values
size=get('Q');identity=b[at:at+size].decode();at+=size
assert identity==expected['identity']
for _ in range(31):at=b.index(b'\0',at)+1
n,kind,subdivisions,steps,regrids=get('iiiQQ');time,dt,seed=get('ddd');tracker=get('3d');floors=get('8d');diagnostics=get('8d')
count=get('Q');leaves=np.frombuffer(b,dtype='<i4',count=4*count,offset=at).reshape(count,4).copy()
assert n==16 and subdivisions==4 and count==232 and steps==2
width=.4;root_level=2;finest=int(leaves[:,3].max())
h=2*width/(2.**leaves[:,3]);lo=-width+leaves[:,:3]*h[:,None];hi=lo+h[:,None]
assert abs(np.sum(h**3)-(2*width)**3)<1e-14
# Explicitly check disjoint leaf interiors, not just matching total volume.
for i in range(count):
 overlaps=np.minimum(hi[i],hi[i+1:])-np.maximum(lo[i],lo[i+1:])
 assert not np.any(np.all(overlaps>1e-14,axis=1))
fine=leaves[:,3]==finest;fl=lo[fine].min(axis=0);fh=hi[fine].max(axis=0)
assert abs(np.sum(h[fine]**3)-np.prod(fh-fl))<1e-14
assert np.allclose(fl,-.05,rtol=0,atol=1e-14) and np.allclose(fh,.05,rtol=0,atol=1e-14)
# Within the affine lab chart x=k X, so the reference Lorentz trajectory has
# dx/dt=k v. Live gauge uses xdot=-beta and need not follow this trajectory.
k=float(identity.split(';k=')[1].split(';')[0]);M=1/3;v=.1
spacing=float(h[fine][0]/n);buffer=4*spacing;drift10=k*v*10*M
safe_half=float(fh[0]-buffer);exit_M=safe_half/(k*v*M)
history=Path(str(CP).replace('.checkpoint','.history.jsonl'));initial=json.loads(history.read_text().splitlines()[0]);assert initial['step']==0 and initial['horizon_complete']
horizon_proxy=initial['horizon_max_radius'];envelope_exit_M=(safe_half-horizon_proxy)/(k*v*M)
result=dict(checkpoint=str(CP.relative_to(ROOT)),checkpoint_sha256=expected['checkpoint_sha256'],identity=identity,n=n,topology_leaves=int(count),level_counts={str(int(l-root_level)):int(c) for l,c in sorted(collections.Counter(leaves[:,3]).items())},finest_cube_lower=fl.tolist(),finest_cube_upper=fh.tolist(),finest_spacing=spacing,finest_spacing_in_inner_isotropic_M=spacing/k/M,four_cell_buffer=buffer,reference_boost=v,reference_x_at_10M=drift10,buffered_finest_exit_reference_M=exit_M,finest_cube_exit_reference_M=float(fh[0]/(k*v*M)),cartesian_cube_exit_reference_M=width/(k*v*M),reference_margin_at_10M=safe_half-drift10,stationary_initial_horizon_proxy=horizon_proxy,initial_history_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),reference_plus_stationary_horizon_and_buffer_exit_M=envelope_exit_M,reference_envelope_margin_at_10M=safe_half-drift10-horizon_proxy,live_tracker_from_stationary_two_step=tracker,qualification='Topology read from admitted local stationary two-step checkpoint; no evolution or new trajectory measurement. Reference drift k*v is the Lorentz-chart estimate, not live-gauge xdot=-beta. Dynamic policy in current driver is a fixture explicitly rejected in production. A boosted production run must monitor actual tracker, finest-leaf distance and stencil support; enlarge static support or qualify a production moving criterion if needed. The saved stationary initial horizon radius is only a coverage proxy, not a boosted/evolved horizon prediction; its translated envelope plus the same buffer exits earlier than the puncture alone. This does not prove gradient resolution or stability.')
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
xs=np.linspace(-.4+1e-8,.4-1e-8,3001);dx=[]
for x in xs:
 point=np.array([x,1e-8,1e-8]);hit=np.all(point>=lo,axis=1)&np.all(point<hi,axis=1);assert np.sum(hit)==1
 dx.append(float(h[hit][0]/n/k/M))
fig,axes=plt.subplots(1,2,figsize=(10.8,3.9),layout='constrained')
axes[0].step(xs,dx,where='mid',color='navy');axes[0].axvspan(-.05,.05,color='skyblue',alpha=.3,label='Finest cube along x')
axes[0].set(xlabel='Compact Cartesian x (y,z near zero)',ylabel='Inner isotropic cell width / M',title='Actual stationary SMR topology',yscale='log');axes[0].legend(fontsize=8);axes[0].grid(alpha=.25)
t=np.linspace(0,15,200);axes[1].plot(t,k*v*t*M,label='Reference chart: x = k v t');axes[1].plot(t,k*v*t*M+horizon_proxy,ls='-.',color='purple',label='+ stationary initial horizon radius (proxy)');axes[1].axhline(.05,color='black',label='Finest cube edge');axes[1].axhline(safe_half,color='darkorange',ls='--',label='Four-cell support margin');axes[1].axvline(10,color='gray',ls=':');axes[1].set(xlabel='Reference t/M',ylabel='Compact Cartesian x',title='Coverage estimate only; not live trajectory');axes[1].legend(fontsize=8);axes[1].grid(alpha=.25)
fig.savefig(OUT/'coverage.png',dpi=170);fig.savefig(OUT/'coverage.pdf');plt.close(fig)
print(json.dumps({k:result[k] for k in ('level_counts','finest_spacing_in_inner_isotropic_M','reference_x_at_10M','buffered_finest_exit_reference_M','reference_margin_at_10M','reference_plus_stationary_horizon_and_buffer_exit_M','reference_envelope_margin_at_10M')},indent=2))
