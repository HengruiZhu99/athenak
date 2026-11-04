"""Read-only local metric-stage and gradient audit; never evolves/restarts a state."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=HERE/'n20-core-audit';OUT.mkdir(exist_ok=True)
RAW=ROOT/'output/boosted-refined-core-resolution/aurora/campaigns/n20-boosted-refined-core-resolution-v1-c20-evolution-segment01/output'
verification=json.loads((HERE/'n20-failure/local-verification.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
hashes={}
for name in ['puncture.checkpoint','puncture.failed-stage.bin']:
    meta=next(v for k,v in verification['artifacts'].items() if k.endswith('/'+name))
    hashes[name]=sha(RAW/name)
    assert hashes[name]==meta['sha256'] and (RAW/name).stat().st_size==meta['bytes']
n=20;h=.09/n;dt=.0002;cell=328016
u=np.memmap(RAW/'puncture.checkpoint',dtype='<f8',mode='r',offset=208,shape=(969344,31))
failed=np.memmap(RAW/'puncture.failed-stage.bin',dtype='<f8',mode='r',shape=(969344+2400,31))
v=np.empty((4*n,4*n,4*n,31))
for b in range(64):
    bx,by,bz=b%4,(b//4)%4,b//16
    v[bz*n:(bz+1)*n,by*n:(by+1)*n,bx*n:(bx+1)*n]=u[b*n**3:(b+1)*n**3].reshape(n,n,n,31)
b,c=divmod(cell,n**3)
idx=np.array([b//16*n+c//n**2,(b//4)%4*n+(c//n)%n,b%4*n+c%n])
x=-.18+(idx[::-1]+.5)*h;q=v[tuple(idx)]
def tensor(a):return np.array([[a[0],a[1],a[2]],[a[1],a[3],a[4]],[a[2],a[4],a[5]]])
def metric(q):return tensor(q[1:7])+np.eye(3)
g=metric(q);eig,zs=np.linalg.eigh(g);z=zs[:,0]
beta=q[22:25]-x;alpha=q[18]+(1+x@x)/2
first=[];adv=np.zeros(31)
for a in range(3):
    off=np.eye(3,dtype=int)[2-a];sg=1 if beta[a]>=0 else -1
    sample=lambda j:v[tuple(idx+j*off)]
    first.append((sample(-2)-8*sample(-1)+8*sample(1)-sample(2))/(12*h))
    adv+=beta[a]*sg*(-3*(sample(-sg)-q)+18*(sample(sg)-q)-6*(sample(2*sg)-q)+(sample(3*sg)-q))/(12*h)
db=np.array(first)[:,22:25]-np.eye(3)
parts={'Curvature':-2*alpha*tensor(q[7:13]),'Biased advection':tensor(adv[1:7]),
       'Lie stretching':db@g+g@db.T-2/3*g*np.trace(db)}

def ko_rate(block_local=True):
    ko=np.zeros(31);c=np.array([1,-4,6,-4,1]);counts=[]
    for a in range(3):
        off=np.eye(3,dtype=int)[2-a];count=0
        for t in range(5):
            logical=idx[2-a]%n
            if block_local and not(0<=logical-t and logical-t+4<n):continue
            line=np.array([v[tuple(idx+(j-t)*off)] for j in range(5)])
            xx=np.array([x+(j-t)*h*np.eye(3)[a] for j in range(5)])
            taper=np.maximum(0,1-(xx*xx).sum(axis=1)/.64)**6
            ko-=.5/(256*h)*c[t]*np.mean(taper)*np.sum(c[:,None]*(line-line[0]),axis=0)
            count+=1
        counts.append(count)
    return tensor(ko[1:7]),counts
parts['Block-local KO8'],counts=ko_rate()
rate=sum(parts.values());capture=metric(failed[cell]);pred=g+dt/2*rate
err=float(abs(pred-capture).max())
alternatives={}
for name,offsets,weights in [
    ('first_order_transport',[0,1],[-1,1]),
    ('telegrapher_biased6_transport',[-2,-1,0,1,2,3,4],[1/30,-2/5,-7/12,4/3,-1/2,2/15,-1/60]),
    ('centered4_transport',[-2,-1,1,2],[1/12,-2/3,2/3,-1/12])]:
    aa=np.zeros(31)
    for a in range(3):
        off=np.eye(3,dtype=int)[2-a];sg=1 if beta[a]>=0 else -1
        aa+=beta[a]*sg/h*sum(w*(v[tuple(idx+j*sg*off)]-q) for j,w in zip(offsets,weights))
    rr=rate-parts['Biased advection']+tensor(aa[1:7])
    alternatives[name]=float(np.linalg.eigvalsh(g+dt/2*rr)[0])
alternatives['half_timestep']=float(np.linalg.eigvalsh(g+dt/4*rate)[0])
alternatives['cross_seam_KO8_same_coefficient']=float(np.linalg.eigvalsh(g+dt/2*(rate-parts['Block-local KO8']+ko_rate(False)[0]))[0])
donors=[];line_rows=[]
tracker=np.array(json.loads((HERE/'n20-failure/puncture.report.json').read_text())['puncture_position'])
for a in range(3):
    off=np.eye(3,dtype=int)[2-a]
    for j in range(-8,9):
        qq=v[tuple(idx+j*off)];xx=x+j*h*np.eye(3)[a];gg=metric(qq)
        if abs(j)<=4:donors.append(dict(xyz=xx.tolist(),chi=float(qq[0]),eigenmin=float(np.linalg.eigvalsh(gg)[0])))
        line_rows.append([a,j,*xx,float(qq[0]),float(qq[18]+(1+xx@xx)/2),*list(qq[22:25]-xx),
                          float(np.linalg.eigvalsh(gg)[0]),float(z@gg@z),float(np.linalg.norm(qq[15:18])),float(np.linalg.norm(qq[25:28]))])
np.savetxt(OUT/'axis-samples.csv',line_rows,delimiter=',',header='axis,offset_cells,x,y,z,chi,baralpha,beta_x,beta_y,beta_z,metric_eigenmin,weak_mode_projection,Lambda_norm,D_norm',comments='')
main='b99e27ffd47c2a0c13e126e0c2c7560e3fc9944a'
files=['src/z4c/z4c_calcrhs.cpp','src/z4c/z4c.cpp','src/z4c/z4c_tasks.cpp','src/utils/finite_diff.hpp',
       'experiments/telegrapher_aurora_20260929/inputs_v4/g1.5_bhahaha_L5.athinput']
source_hashes={f:hashlib.sha256(subprocess.check_output(['git','show',f'{main}:{f}'],cwd=ROOT)).hexdigest() for f in files}
frozen='cb0f47b520afe4f88c98e11569f3ba88e632df9a'
frozen_files=['src/z4c/hyperboloidal/hyperbolic_composite_device.hpp',
              'src/z4c/hyperboloidal/hyperbolic_shift.hpp','src/z4c/hyperboloidal/conformal_rhs.hpp',
              'tst/hyperbolic_shift_3d/coupled_mpi.cpp']
frozen_hashes={}
for f in frozen_files:
    data=subprocess.check_output(['git','show',f'{frozen}:{f}'],cwd=ROOT)
    assert data==(ROOT/f).read_bytes(), 'Audit source differs from frozen science'
    frozen_hashes[f]=hashlib.sha256(data).hexdigest()
k=.306142283363079
coord=-.18+(np.arange(4*n)+.5)*h
zz,yy,xx=np.meshgrid(coord,coord,coord,indexing='ij');omega=(1-xx**2-yy**2-zz**2)/2
lab_chi=omega**2*v[:,:,:,0]/k**2
near=(xx-tracker[0])**2+(yy-tracker[1])**2+(zz-tracker[2])**2 < (.04)**2
near_points=int(near.sum());guard_points=int(np.count_nonzero(near&(lab_chi<1e-5)))
frozen_parameters=dict(source=frozen,source_hashes=frozen_hashes,ko_coefficient=.5,
    inner_spatial_scale=k,matched_lab_gamma_coefficient=k*k,
    current_gamma_coefficient=1.,gamma_coefficient_ratio=1/(k*k),
    guard_reference='AF chi=Omega^2 chi_native/k^2 on the constant inner map',
    near_tracker_radius=.04,near_tracker_points=near_points,near_tracker_AF_guard_active_points=guard_points)
result=dict(input_hashes=hashes,analysis_sha256=sha(Path(__file__)),comparison_commit=main,comparison_source_hashes=source_hashes,
    frozen_parameters=frozen_parameters,
    cell=cell,xyz=x.tolist(),logical_in_block=(idx[::-1]%n).tolist(),core_dx=h,dt=dt,
    tracker=tracker.tolist(),distance_to_tracker_cells=float(np.linalg.norm(x-tracker)/h),
    pre_step_eigenvalues=eig.tolist(),failed_eigenvalues=np.linalg.eigvalsh(capture).tolist(),
    pre_step_metric_condition_number=float(eig[-1]/eig[0]),
    stencil_max_radius=float(max(np.linalg.norm(p['xyz']) for p in donors)),
    predicted_first_half_stage_eigenvalues=np.linalg.eigvalsh(pred).tolist(),metric_reconstruction_max_error=err,
    first_half_stage_matches=err<2e-12,pre_step_chi=float(q[0]),baralpha=float(alpha),beta=beta.tolist(),
    rate_projections={k:float(z@r@z) for k,r in parts.items()},ko_complete_difference_lines_per_axis=counts,
    counterfactual_first_half_stage_min_eigenvalue=alternatives,donors=donors,
    nominal_core_chi_min=float(v[:,:,:,0].min()),
    physical_lab_dx_over_M=h/.306142283363079*3,
    qualification='Read-only failed-run forensic analysis, not an admitted endpoint or restart. Counterfactuals are local algebraic stage estimates, not evolution tests. Metric-stage inference only valid if reconstruction matches.')
(OUT/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
fig,axes=plt.subplots(2,3,figsize=(13,7.5))
rows=np.array(line_rows)
for a in range(3):
    ar=rows[rows[:,0]==a];ax=axes[0,a]
    ax.plot(ar[:,1],ar[:,10],'.-',label='Minimum metric eigenvalue')
    ax.plot(ar[:,1],ar[:,11],'.-',label='Fixed weak-mode projection')
    ax.axvline(0,c='red',ls=':',label='Failing cell');ax.axhline(0,c='gray',lw=.7)
    ax.set(title='Pre-step '+['x','y','z'][a]+' line',xlabel='Offset from failing cell / grid spacing',ylabel='Metric value')
    ax.grid(alpha=.2)
    bx=axes[1,a]
    bx.semilogy(ar[:,1],ar[:,5],'.-',label=r'$\chi$')
    bx.semilogy(ar[:,1],ar[:,6],'.-',label=r'$\bar\alpha$')
    bx.semilogy(ar[:,1],ar[:,12],'.-',label=r'$|\Lambda|$')
    bx.axvline(0,c='red',ls=':');bx.grid(alpha=.2)
    bx.set(xlabel='Offset from failing cell / grid spacing',ylabel='Field magnitude')
axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
fig.suptitle('N20 BC2 last accepted state at 6.2082M: unsmoothed puncture-adjacent samples\nUnadmitted failed-run checkpoint used only for forensics; spacing = 0.0045')
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'core-gradients.png',dpi=170);plt.close(fig)
print(json.dumps({k:v for k,v in result.items() if k not in ['donors','comparison_source_hashes']},indent=2))
