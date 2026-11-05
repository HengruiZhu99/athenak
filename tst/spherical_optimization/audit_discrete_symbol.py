#!/usr/bin/env python3
"""Frozen local Fourier symbols from saved full Jacobians and shared C++ jets.

The actual discrete symbol sums the saved physical-variable Jacobian row over
Fourier nodal perturbations. Continuum and stencil substitutions use independent
q/q'/q'' derivatives of the shared interior RHS at the same evolved jets.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import make_evolution,weights

def jet_jacobians(k,r,q,d,dd,source,independent,epsilon):
 base=[q[independent][None].copy(),d[independent][None].copy(),dd[independent][None].copy()];out=[]
 def rhs(jets):
  Q,D,DD=k.expand_chart(*jets);R,S,_,_=k.parts(np.array([r]),Q,D,DD,source[None]);_,_,om=k.initial(np.array([r]));return (R+S/om[:,:1])[0,independent]
 for order in range(3):
  mat=np.zeros((10,10))
  for field in range(10):
   h=epsilon*max(1.,abs(base[order][0,field]));plus=[x.copy() for x in base];minus=[x.copy() for x in base];plus[order][0,field]+=h;minus[order][0,field]-=h
   mat[:,field]=(rhs(plus)-rhs(minus))/(2*h)
  out.append(mat)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--spectra',nargs='+',type=Path,required=True);ap.add_argument('--radii',nargs='+',type=float,default=[.78,.8,.82,.84,.86,.9]);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[];arrays={};inputs={}
 for folder in a.spectra:
  saved=np.load(folder/'linearization.npz');audit=json.loads((folder/'audit.json').read_text());cfg=audit['run_configuration'];k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);e.rhs(saved['state']);q,d,dd=[e.last[x].copy() for x in ('q','d','dd')];r=saved['r'];h=r[1]-r[0];assert np.allclose(np.diff(r),h)
  active=saved['active'];scales=saved['scales'];indices=np.full(active.shape,-1);indices[active]=np.arange(active.sum());ci,cf=np.where(active);J=saved['J'];theta=np.linspace(.01,np.pi,161);inputs[str(folder/'linearization.npz')]=hashlib.sha256((folder/'linearization.npz').read_bytes()).hexdigest()
  for requested in a.radii:
   i=int(np.argmin(abs(r-requested)));assert 5<=i<len(r)-6
   physical_rows=J[indices[i]]*scales[i,:,None]/scales[active][None,:]
   coarse=jet_jacobians(k,r[i],q[i],d[i],dd[i],e.source[i],e.independent,1e-5);fine=jet_jacobians(k,r[i],q[i],d[i],dd[i],e.source[i],e.independent,3e-6);A,B,C=fine
   rel=float(np.linalg.norm(np.array(coarse)-fine)/max(1,np.linalg.norm(fine)))
   offsets=np.arange(-3,4);d1=weights(offsets.astype(float),0.,1)/h;d2=weights(offsets.astype(float),0.,2)/h**2;shift=-1 if q[i,9]<0 else 1;adv_offsets=offsets+shift;adv=weights(adv_offsets.astype(float),0.,1)/h
   values={name:[] for name in ['actual_discrete','continuum','direct_stencil','composed_second']}
   for t in theta:
    phase=np.exp(1j*t*(r[ci]-r[i])/h);L=np.column_stack([physical_rows[:,cf==f]@phase[cf==f] for f in range(10)])
    s1=d1@np.exp(1j*t*offsets);s2=d2@np.exp(1j*t*offsets);sa=adv@np.exp(1j*t*adv_offsets);wave=t/h
    models=dict(actual_discrete=L,continuum=A+1j*wave*B-wave**2*C,direct_stencil=A+s1*B+s2*C+q[i,9]*(sa-s1)*np.eye(10),composed_second=A+s1*B+s1**2*C+q[i,9]*(sa-s1)*np.eye(10))
    for name,mat in models.items():
     eigen=np.linalg.eigvals(mat);j=np.argmax(eigen.real);values[name].append([float(eigen[j].real),float(eigen[j].imag)])
   tag=f'n{len(r)}-i{i}';arrays[tag+'-theta']=theta;row=dict(spectrum=str(folder),n=len(r),r=float(r[i]),requested_r=requested,jets_relative_difference=rel,models={})
   for name,v in values.items():
    v=np.array(v);arrays[tag+'-'+name]=v;j=int(v[:,0].argmax());row['models'][name]=dict(max_real=float(v[j,0]),imag_at_max=float(v[j,1]),theta_at_max=float(theta[j]),max_real_times_h=float(v[j,0]*h))
   records.append(row)
  k.close()
 np.savez_compressed(a.output/'symbols.npz',**arrays)
 out=dict(rows=records,inputs=inputs,binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Local frozen Fourier diagnostics, not a variable-coefficient or boundary stability proof. Actual rows include all saved numerical policies. Stencil substitutions hold continuum local q/qprime/qdoubleprime coefficients fixed and omit spatial coefficient variation/reference-chi scaling effects. No PDE or evolution is modified.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(records))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 sizes=sorted({x['n'] for x in records});fig,axes=plt.subplots(1,len(sizes),figsize=(6*len(sizes),4),layout='constrained',squeeze=False)
 for ax,n in zip(axes[0],sizes):
  row=min([x for x in records if x['n']==n],key=lambda x:abs(x['r']-.82));i=int(np.argmin(abs(np.load(Path(row['spectrum'])/'linearization.npz')['r']-row['r'])));tag=f'n{n}-i{i}'
  for name in row['models']:ax.plot(arrays[tag+'-theta']/np.pi,arrays[tag+'-'+name][:,0],label=name.replace('_',' '))
  ax.set(xlabel='Grid phase / pi',ylabel='Maximum real eigenvalue',title=f"N{n}, r={row['r']:.4f}");ax.axhline(0,color='k',lw=.5);ax.grid(alpha=.25);ax.legend(fontsize=8)
 fig.savefig(a.output/'growth.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
