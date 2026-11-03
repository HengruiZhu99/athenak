"""Read-only reconstruction of BG1's rejected metric stage; never a restart.

Run from any directory. Requires the locally preserved production artifacts.
Counterfactuals act on the same last-accepted state, not on finer-grid solutions.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'core-instability-audit'
OUT.mkdir(exist_ok=True)
RAW = ROOT / 'output/boosted-free-gauge/aurora/campaigns/n16-boosted-free-gauge-v1-b16-evolution-segment01/output'
EXPECTED = {'puncture.checkpoint': 'bef2e6c5f4a503215ccc4343c3e7309f8fba814cf3f18553d4dd2b7c797254be',
            'puncture.failed-stage.bin': '6230b38fae55f21d046c3118460024d36a690c0044f8787f1736ef97ec6211ae'}
for name, digest in EXPECTED.items():
    assert hashlib.sha256((RAW/name).read_bytes()).hexdigest() == digest
u = np.fromfile(RAW/'puncture.checkpoint', dtype='<f8', offset=208, count=232928*31).reshape(-1,31)
failure = np.fromfile(RAW/'puncture.failed-stage.bin', dtype='<f8').reshape(-1,31)
v = np.empty((32,32,32,31))
for b in range(8):
    v[(b//4)*16:(b//4+1)*16, ((b//2)%2)*16:((b//2)%2+1)*16, (b%2)*16:(b%2+1)*16] = u[b*4096:(b+1)*4096].reshape(16,16,16,31)
h = .18/16

def tensor(a):
    return np.array([[a[0],a[1],a[2]],[a[1],a[3],a[4]],[a[2],a[4],a[5]]])
def metric(q):
    return tensor(q[1:7])+np.eye(3)
def locate(i):
    b,q = divmod(i,4096)
    return np.array([b//4*16+q//256, (b//2)%2*16+(q//16)%16, b%2*16+q%16])

rows = []
for cell in [4094,12046,16638,24590]:
    idx = locate(cell)
    x = -.18+(idx[::-1]+.5)*h
    q = v[tuple(idx)]
    g = metric(q)
    beta = q[22:25]-x
    alpha = q[18]+(1+x@x)/2
    first, adv = [], np.zeros(31)
    touched = []
    for a in range(3):
        off = np.eye(3,dtype=int)[2-a]
        sample = lambda s: v[tuple(idx+s*off)]
        first.append((sample(-2)-8*sample(-1)+8*sample(1)-sample(2))/(12*h))
        sg = 1 if beta[a] >= 0 else -1
        adv += beta[a]*sg*(-3*(sample(-sg)-q)+18*(sample(sg)-q)-6*(sample(2*sg)-q)+(sample(3*sg)-q))/(12*h)
        touched.extend(x+j*h*np.eye(3)[a] for j in range(-4,5))
    db = np.array(first)[:,22:25]-np.eye(3)
    parts = {'curvature': -2*alpha*tensor(q[7:13]), 'advection': tensor(adv[1:7]),
             'Lie stretching': db@g+g@db.T-2/3*g*np.trace(db)}
    ko = np.zeros(31)
    c = np.array([1,-4,6,-4,1])
    for a in range(3):
        off = np.eye(3,dtype=int)[2-a]
        for t in range(5):
            line = np.array([v[tuple(idx+(j-t)*off)] for j in range(5)])
            xx = np.array([x+(j-t)*h*np.eye(3)[a] for j in range(5)])
            taper = np.maximum(0,1-(xx*xx).sum(axis=1)/.64)**6
            ko -= .5/(256*h)*c[t]*np.mean(taper)*np.sum(c[:,None]*(line-line[0]),axis=0)
    parts['KO8'] = tensor(ko[1:7])
    rate = sum(parts.values())
    eig,evec = np.linalg.eigh(g)
    z = evec[:,0]
    predicted = g+.00025*rate
    error = float(abs(predicted-metric(failure[cell])).max())
    assert error < 2e-13
    alternatives = {}
    for name,offsets,weights in [
        ('upwind1',[0,1],[-1,1]),
        ('biased6_main',[-2,-1,0,1,2,3,4],[1/30,-2/5,-7/12,4/3,-1/2,2/15,-1/60]),
        ('centered4',[-2,-1,1,2],[1/12,-2/3,2/3,-1/12])]:
        a31 = np.zeros(31)
        for a in range(3):
            off = np.eye(3,dtype=int)[2-a]
            sg = 1 if beta[a] >= 0 else -1
            a31 += beta[a]*sg/h*sum(w*(v[tuple(idx+j*sg*off)]-q) for j,w in zip(offsets,weights))
        alt = tensor(a31[1:7])
        alternatives[name] = {'advection_rayleigh_rate':float(z@alt@z),
                             'half_stage_min_eigenvalue':float(np.linalg.eigvalsh(g+.00025*(rate-parts['advection']+alt))[0])}
    # Each donor metric is positive definite. A high-order derivative need not
    # preserve this convex cone in an explicit stage.
    donor_mins = []
    for a in range(3):
        off = np.eye(3,dtype=int)[2-a]
        for j in range(-4,5):
            donor_mins.append(np.linalg.eigvalsh(metric(v[tuple(idx+j*off)]))[0])
    rows.append(dict(cell=cell,xyz=x.tolist(),last_accepted_metric_eigenvalues=eig.tolist(),
                     beta=beta.tolist(),chi=float(q[0]),baralpha=float(alpha),D=q[25:28].tolist(),
                     rate_projections={k:float(z@r@z) for k,r in parts.items()},
                     total_rate_projection=float(z@rate@z),
                     reconstructed_half_stage_eigenvalues=np.linalg.eigvalsh(predicted).tolist(),
                     metric_capture_max_abs_difference=error,
                     half_full_timestep_first_stage_min_eigenvalue=float(np.linalg.eigvalsh(g+.000125*rate)[0]),
                     transport_only_counterfactuals=alternatives,
                     stencil_max_radius=float(np.linalg.norm(touched,axis=1).max()),
                     stencil_min_metric_eigenvalue=float(min(donor_mins))))
    if cell == 4094:
        g0, rate0, parts0, eigvec0, idx0 = g, rate, parts, z, idx

# Immutable main-branch comparison source identities, not a new main-branch run.
main = 'c8d5c33e39089f005e416bf839652f6796fa6e82'
source_files = ['src/z4c/z4c_calcrhs.cpp','src/z4c/z4c_tasks.cpp','src/utils/finite_diff.hpp',
                'src/pgen/tests/z4c_boosted_puncture.cpp','inputs/z4c/z4c_boosted_puncture.athinput']
main_sources = {p:hashlib.sha256(subprocess.check_output(['git','show',f'{main}:{p}'],cwd=ROOT)).hexdigest() for p in source_files}
k = .306142283363079
result = dict(input_sha256=EXPECTED,analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              last_accepted_t=1.4099999999999002,full_dt=.0005,
              inferred_rejected_stage_t=1.4099999999999002+.00025,
              stage_inference='Metric-only reconstruction matches all four failed cells. RK stage number was not recorded in the report.',
              points=rows,inner_gauge_end_of_plateau=.2,restoring_collar_begin=.6,
              inner_coordinate_map='x=k X, t=T; constant k on the inner plateau',k=k,
              grid=[dict(n=n,compact_dx=.18/n,physical_lab_dx_over_M=(.18/n)/k*3,
                         initial_isotropic_horizon_radius_in_cells=k/6/(.18/n)) for n in [16,20,24]],
              main_commit=main,main_source_sha256=main_sources,
              qualification='Read-only failed-run analysis. No restart, admitted endpoint, finer-grid solution, full reduced-dt evolution or causal gauge/resolution A/B test. Alternative advection is substituted only at this fixed state; all other terms held fixed.')
(OUT/'metric-stage.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

fig,axes = plt.subplots(1,3,figsize=(14,4.4))
# Unsmooth samples across the actual puncture-adjacent feature.
xx = -.18+(np.arange(32)+.5)*h
samples = v[idx0[0],idx0[1],:]
axes[0].plot(xx,[np.linalg.eigvalsh(metric(a))[0] for a in samples],'.-',label='Pre-step minimum eigenvalue')
axes[0].plot(xx,[eigvec0@metric(a)@eigvec0 for a in samples],'.-',label='Projection along failing mode')
axes[0].axvline(-.02318276315853688,ls='--',c='gray',label='Tracker')
axes[0].axvline(rows[0]['xyz'][0],ls=':',c='r',label='Failed cell')
axes[0].set(xlim=(-.08,.06),xlabel='Compact x, y=z=-0.005625',ylabel='Metric value',title='Sharp inner metric feature')
axes[0].legend(fontsize=7)
labels=list(parts0)
axes[1].barh(labels,[eigvec0@parts0[a]@eigvec0 for a in labels],color=['gray','tab:red','tab:orange','tab:blue'])
axes[1].axvline(0,c='k',lw=.6)
axes[1].set(xlabel='Weakest-mode projected rate',title='Advection drives the negative stage')
dts=np.linspace(0,.00055,201)
axes[2].plot(dts,[np.linalg.eigvalsh(g0+.5*d*rate0)[0] for d in dts])
for dt,label in [(.0005,'Actual dt'),(.00025,'Half dt')]:
    axes[2].plot(dt,np.linalg.eigvalsh(g0+.5*dt*rate0)[0],'o',label=label)
axes[2].axhline(0,c='k',lw=.6)
axes[2].set(xlabel='Hypothetical full timestep',ylabel='First half-stage minimum eigenvalue',title='One-stage sensitivity only')
axes[2].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('BG1 N16: exact reconstruction of the rejected metric stage; no failed-state evolution',fontsize=12)
fig.tight_layout()
fig.savefig(OUT/'metric-stage.png',dpi=170)
plt.close(fig)
print(json.dumps({'points':rows,'grid':result['grid']},indent=2))
