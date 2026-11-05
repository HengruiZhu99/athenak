#!/usr/bin/env python3
"""Linear-only compactly supported D1-squared intervention on saved Jacobians.

This diagnoses an operator mechanism; it is not an implemented spatial scheme.
Coefficients are frozen at the original trajectory. Endpoint rows, first
 derivatives, advection, damping, and dissipation remain exactly as saved.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import eig
from kernel import Kernel
from evolution import make_evolution,weights
from audit_discrete_symbol import jet_jacobians

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--spectra',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 reports=[];arrays={}
 for folder in a.spectra:
  s=np.load(folder/'linearization.npz');cfg=json.loads((folder/'audit.json').read_text())['run_configuration'];k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);e.rhs(s['state']);q,d,dd=[e.last[x].copy() for x in ('q','d','dd')];r=s['r'];h=r[1]-r[0];assert np.allclose(np.diff(r),h)
  active=s['active'];scale=s['scales'];idx=np.full(active.shape,-1);idx[active]=np.arange(active.sum());J=s['J'];n=len(r)
  # C2 smooth compact taper: unchanged endpoint neighbourhoods.
  z=np.minimum((r-.72)/.06,(.89-r)/.03);z=np.clip(z,0,1);taper=z**3*(10-15*z+6*z*z)
  w1=weights(np.arange(-3,4,dtype=float),0.,1);delta=np.convolve(w1,w1);delta[3:10]-=weights(np.arange(-3,4,dtype=float),0.,2);delta/=h*h
  corrections={name:np.zeros_like(J) for name in ('all','lapse','shift','geometry')}
  for i in np.where(taper>0)[0]:
   assert 6<=i<n-6
   C=jet_jacobians(k,r[i],q[i],d[i],dd[i],e.source[i],e.independent,3e-6)[2]
   # Auxiliary D receives actual Lambda RHS, including its second derivatives.
   C[9]=C[5]
   for f in range(10):
    cols=idx[i+np.arange(-6,7),f];assert np.all(cols>=0)
    reference_scale=(r[i]/r[i+np.arange(-6,7)])**cfg.get('chi_reference_power',0) if f==0 else np.ones(13)
    block=taper[i]*C[:,f,None]*delta[None,:]*reference_scale[None,:]*scale[i+np.arange(-6,7),f][None,:]/scale[i,:,None]
    corrections['all'][np.ix_(idx[i],cols)]+=block
    group='lapse' if f==6 else 'shift' if f==7 else 'geometry'
    corrections[group][np.ix_(idx[i],cols)]+=block
  assert np.allclose(corrections['all'],corrections['lapse']+corrections['shift']+corrections['geometry'])
  report=dict(n=n,input=str(folder),input_sha256=hashlib.sha256((folder/'linearization.npz').read_bytes()).hexdigest(),models={})
  for name,correction in [('original',np.zeros_like(J)),*corrections.items()]:
   mat=J+correction;v,V=eig(mat);order=np.argsort(v.real)[::-1];v=v[order];mode=V[:,order[0]];res=np.linalg.norm(mat@mode-v[0]*mode)/max(1,np.linalg.norm(mat)*np.linalg.norm(mode));physical=np.zeros_like(s['state'],dtype=complex);physical[active]=mode*scale[active];peak=np.unravel_index(abs(physical).argmax(),physical.shape)
   report['models'][name]=dict(leading_eigenvalues=np.column_stack((v[:12].real,v[:12].imag)).tolist(),residual=float(res),peak_r=float(r[peak[0]]),peak_field=int(peak[1]),relative_correction_norm=float(np.linalg.norm(correction)/np.linalg.norm(J)))
   arrays[f'n{n}-{name}-eigenvalues']=v;arrays[f'n{n}-{name}-mode']=physical
   print(json.dumps(dict(n=n,model=name,leading=report['models'][name]['leading_eigenvalues'][0],peak_r=float(r[peak[0]]))),flush=True)
  arrays[f'n{n}-r']=r;arrays[f'n{n}-taper']=taper;reports.append(report);k.close()
 np.savez_compressed(a.output/'operators.npz',**arrays)
 out=dict(reports=reports,binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('audit_discrete_symbol.py')]},qualification='Linear-only diagnostic intervention, not the Jacobian of a tested nonlinear scheme. D1 squared replaces direct D2 only in a compact interior window, holding original background coefficients fixed. No evolution or stability claim.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
