from pathlib import Path
import sys,json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2];OLD=REPO/'review/stability-modes-20260919/mode-analysis'
sys.path.insert(0,str(OLD))
from mode_operator import Operator,SHAPE,ACTIVE,FIELDS,profile
op=Operator(str(ROOT/'validation_calls'),binary=ROOT.parent/'bin/athena-radiation-v2-hook',input_file=ROOT/'input.athinput',dt=.0375)
def apply(v,steps,eps,label):
 r=op.response(v.real,steps,eps,label=label+'_real')
 if np.max(abs(v.imag))>1e-10:r=r+1j*op.response(v.imag,steps,eps,label=label+'_imag')
 return r

def metrics(a,b):
 return {'relative_l2':float(np.linalg.norm(a-b)/np.linalg.norm(a)), 'active_relative_l2':float(np.linalg.norm((a-b)[ACTIVE])/np.linalg.norm(a[ACTIVE]))}

def polarization(v):
 a=v[ACTIVE];zz,yy,xx=np.meshgrid(np.arange(16),np.arange(16),np.arange(16),indexing='ij');bdry=(xx==0)|(xx==15)|(yy==0)|(yy==15)|(zz==0)|(zz==15)
 side=np.stack(((xx==15).astype(float)-(xx==0), (yy==15).astype(float)-(yy==0), (zz==15).astype(float)-(zz==0)))
 side/=np.maximum(np.linalg.norm(side,axis=0),1)
 g=np.array([[a[1],a[2],a[3]],[a[2],a[4],a[5]],[a[3],a[5],a[6]]]);A=np.array([[a[8],a[9],a[10]],[a[9],a[11],a[12]],[a[10],a[12],a[13]]])
 n=side
 res={}
 for name,T in [('g',g),('A',A)]:
  Tn=np.einsum('ab...,b...->a...',T,n);Tnn=np.einsum('a...,a...->...',Tn,n);tr=np.trace(T,axis1=0,axis2=1);nv=Tn-n*Tnn
  P=np.eye(3)[:,:,None,None,None]-np.einsum('a...,b...->ab...',n,n)
  scalar=np.einsum('a...,b...,...->ab...',n,n,Tnn)+.5*P*(tr-Tnn)
  vector=np.einsum('a...,b...->ab...',n,nv)+np.einsum('a...,b...->ab...',nv,n)
  tensor=T-scalar-vector
  for label,val in [('scalar',scalar),('vector',vector),('tensor',tensor)]:res[name+'_'+label+'_boundary_energy']=float(np.sum(abs(val[:,:,bdry])**2))
 for name,val in [('Gamma',a[14:17]),('beta',a[19:22])]:
  scalar=np.einsum('a...,a...->...',n,val);vec=val-n*scalar
  res[name+'_normal_boundary_energy']=float(np.sum(abs(scalar[bdry])**2));res[name+'_tangent_boundary_energy']=float(np.sum(abs(vec[:,bdry])**2))
 return res

results=[];fig,axes=plt.subplots(2,3,figsize=(10.6,6.1),constrained_layout=True)
for n,m in enumerate(json.loads((ROOT/'results.json').read_text())['modes']):
 v=np.fromfile(m['state_real']).reshape(SHAPE)+1j*np.fromfile(m['state_imag']).reshape(SHAPE);mu=complex(m['mu_real'],m['mu_imag'])
 r=apply(v,8,1e-4,f'mode{n}_8');rlo=apply(v,8,3e-5,f'mode{n}_8low');r1=apply(v,1,1e-4,f'mode{n}_1')
 ray=np.vdot(v[ACTIVE],r1[ACTIVE])/np.vdot(v[ACTIVE],v[ACTIVE]);a=v[ACTIVE]
 zz,yy,xx=np.meshgrid(np.arange(16),np.arange(16),np.arange(16),indexing='ij');codim=((xx==0)|(xx==15)).astype(int)+((yy==0)|(yy==15)).astype(int)+((zz==0)|(zz==15)).astype(int)
 e=np.sum(abs(a)**2,axis=0);te=abs(a[17])**2
 loc={key:float(e[codim==k].sum()/e.sum()) for k,key in enumerate(['interior_energy_fraction','face_energy_fraction','edge_energy_fraction','corner_energy_fraction'])}
 loc['Theta_by_codimension']=[float(te[codim==k].sum()/te.sum()) for k in range(4)]
 idx=np.unravel_index(np.argmax(abs(a)),a.shape);loc['all_field_peak']={'field':FIELDS[idx[0]],'xyz':[(q+.5)*.25-2 for q in idx[-3:][::-1]]}
 fields=[{'field':name,'active_energy_fraction':float(np.sum(abs(a[k])**2)/e.sum()),'active_peak':float(abs(a[k]).max()),'peak_xyz':[(q+.5)*.25-2 for q in np.unravel_index(np.argmax(abs(a[k])),a[k].shape)[::-1]]} for k,name in enumerate(FIELDS)]
 fields.sort(key=lambda q:-q['active_energy_fraction'])
 rec={'index':n,'projected_mu':[mu.real,mu.imag],'projected_gamma':m['gamma'],'direct_eigen_residual':metrics(mu*v,r),'amplitude_linearity':metrics(r,rlo),'one_step_eigen_residual':metrics(mu**(1/8)*v,r1),'one_step_active_rayleigh_gamma':float(np.log(abs(ray))/.0375),'one_step_active_rayleigh_omega':float(np.angle(ray)/.0375),'active_norm_gain_8steps':float(np.linalg.norm(r[ACTIVE])/np.linalg.norm(a)),'location':loc,'field_energy':fields,'polarization':polarization(v),'scope':'Unconverged Ritz candidate: gain/Rayleigh data are directional measurements, not validated eigenvalues.'}
 results.append(rec)
 for row,field in enumerate([17,7]):
  maxaxis=np.argmax(np.sum(abs(a[field])**2,axis=(1,2)));sl=abs(a[field,maxaxis])
  im=axes[row,n].imshow(sl,extent=(-2,2,-2,2),origin='lower',cmap='magma');fig.colorbar(im,ax=axes[row,n],shrink=.9);axes[row,n].set_title(f'Candidate {n}, {FIELDS[field]}, z={(maxaxis+.5)*.25-2:g}M');axes[row,n].set_xlabel('x/M');axes[row,n].set_ylabel('y/M')
 (ROOT/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps({k:v for k,v in rec.items() if k not in ['field_energy','polarization']},indent=2),flush=True)
fig.savefig(ROOT/'candidate-localization.png',dpi=180);fig.savefig(ROOT/'candidate-localization.pdf')
