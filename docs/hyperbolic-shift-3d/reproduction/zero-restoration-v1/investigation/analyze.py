from pathlib import Path
import json,hashlib,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[5];d=Path(__file__).resolve().parent;o=r/'output/zero-restoration/investigation'
p=r/'output/zero-restoration/aurora/campaigns/n16-zero-restoration-v1-z16-evolution-segment01/output'
xyz=np.genfromtxt(r/'output/hyperbolic-core-shell/inner-telegrapher-v1/coordinates-n16.csv',delimiter=',',names=True)
x=np.column_stack([xyz[k] for k in ['x','y','z']]);rad=xyz['r'];nv=159200;ns=1536
probe=np.genfromtxt(o/'last-valid-e8.csv',delimiter=',',names=True);sx=np.column_stack([probe[k] for k in ['x','y','z']])
def cp(f):
 a=np.fromfile(f,dtype='<f8',offset=168);return a[:31*nv].reshape(nv,31),a[2*31*nv:2*31*nv+31*ns].reshape(ns,31)
def metrics(v,n):
 g=np.zeros((len(v),3,3))
 for k,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):g[:,i,j]=g[:,j,i]=v[:,k+1]+(i==j)
 eig,ev=np.linalg.eigh(g)
 return {'grr':np.einsum('ni,nij,nj->n',n,g,n),'lambda_min':eig[:,0],'radial_alignment':np.abs(np.sum(ev[:,:,0]*n,axis=1)), 'chi':v[:,0], 'v_radial':np.sum(v[:,22:25]*n,axis=1),'D_radial':np.sum(v[:,25:28]*n,axis=1)}
accepted=cp(p/'puncture.checkpoint');a=np.fromfile(p/'puncture.failed-stage.bin',dtype='<f8').reshape(-1,31);failed=(a[:nv],a[nv:]);a_cp=r/'output/native-shared-transition/aurora/campaigns/n16-native-transition-v2-a-evolution-segment01/output/puncture.checkpoint'
sets={'zero_last_valid':accepted,'zero_rejected':failed,'A_10M':cp(a_cp)}
results={};profiles={}
for label,(v,s) in sets.items():
 sm=metrics(s,sx);vm=metrics(v,x/rad[:,None]);results[label]={k:{'min':float(q.min()),'max':float(q.max()),'mean':float(q.mean()),'std':float(q.std())} for k,q in sm.items()};rows=[]
 for rr in np.unique(np.round(rad[rad>.75],12)):
  m=np.abs(rad-rr)<1e-11;rows.append(dict(r=float(rr),**{k:float(q[m].mean()) for k,q in vm.items()}))
 rows.append(dict(r=1.,**{k:float(q.mean()) for k,q in sm.items()}));profiles[label]=rows
summary={'surface':results,'radial_profiles':profiles,'probe':{}}
for f in o.glob('*.csv'):
 q=np.genfromtxt(f,delimiter=',',names=True);summary['probe'][f.stem]={k:{'mean':float(np.mean(q[k])),'maxabs':float(np.max(np.abs(q[k])))} for k in q.dtype.names if k not in ['row','x','y','z']}
q=np.genfromtxt(o/'last-valid-e9.csv',delimiter=',',names=True);summary['directional_derivative_halving_relative_max']=float(np.max(abs(probe['D_trace_dt_radial']-q['D_trace_dt_radial']))/np.max(abs(probe['D_trace_dt_radial'])))
summary['snapshot_hashes']={str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'puncture.checkpoint',p/'puncture.failed-stage.bin',a_cp]}
summary['limitations']=['Postmortem reads do not admit the failed checkpoint or restart it.','A10M profile is a healthy reference at a different time, not a matched-time comparison.','D compatibility uses directional derivatives of the discrete algebraic trace and a sampled continuum D PDE; it is not a truncation-order proof.','RHS direction includes native bulk KO; surface D has no independently filtered RHS.','No temporal/spatial refinement of evolving failure has been performed.']
rows=profiles['zero_last_valid'][-7:];xx=np.array([v['r'] for v in rows]);dx=xx-1
rhs=np.zeros(7);rhs[1]=1;w=np.linalg.solve(np.array([dx**k for k in range(7)]),rhs)
g=np.array([v['grr'] for v in rows]);vr=np.array([v['v_radial'] for v in rows])
summary['radial_stencil_analysis']={'r':xx.tolist(),'endpoint_first_derivative_weights':w.tolist(),'grr_values':g.tolist(),'grr_native7_derivative':float(w@g),'grr_last_interval_secant':float((g[-1]-g[-2])/(xx[-1]-xx[-2])),'v_native7_derivative':float(w@vr),'v_last_interval_secant':float((vr[-1]-vr[-2])/(xx[-1]-xx[-2])),'diagnostic_only':'Angular-mean radial interpolation; no operator changed. Final-state stencil sensitivity does not identify initial cause.'}
(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axs=plt.subplots(2,3,figsize=(13,7))
for ax,k,title in zip(axs.flat,['grr','lambda_min','chi','v_radial','D_radial','radial_alignment'],['Radial conformal metric','Smallest metric eigenvalue','Conformal chi','Radial shift deviation beta+x','Radial auxiliary D','Smallest-eigenvector radial alignment']):
 for label,rows in profiles.items():ax.plot([v['r'] for v in rows],[v[k] for v in rows],'.-',label=label)
 ax.set_title(title);ax.set_xlabel('compact radius r');ax.grid(alpha=.3)
 if k=='D_radial':ax.set_yscale('symlog',linthresh=10)
axs[0,0].legend(fontsize=8);fig.suptitle('N16 postmortem: broad radial scri collapse, not a localized angular seam\nA at10M is a healthy reference at a different time',fontsize=12);fig.tight_layout();fig.savefig(d/'radial-postmortem.png',dpi=150)
print(json.dumps({k:summary[k] for k in ['directional_derivative_halving_relative_max','surface']},indent=2))
