#!/usr/bin/env python3
"""Restricted mapped singular controls and exact periodic centered-difference nulls."""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--family',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
report=dict(scope='Restricted 42-mode singular controls, not a full mapped spectrum or spectral-gap proof; no solve/evolution/operator adoption',cases={},periodic_controls={},inputs_sha256={})
for f in sorted(a.family.glob('n*-s*-r*.stdout')):
 r=json.loads(f.read_text());case={k:v for k,v in r.items()if k!='normal_gram'};case['restricted_singular_controls']={}
 for name,raw in r['normal_gram'].items():
  g=np.array(raw);scale=np.max(abs(g));eig,v=np.linalg.eigh((g+g.T)/(2*scale));eig*=scale
  case['restricted_singular_controls'][name]=dict(raw_eigenvalues_AtA=eig.tolist(),negative_eigenvalues=int((eig<0).sum()),singular_values=np.sqrt(np.maximum(0,eig)).tolist(),smallest=np.sqrt(max(0,eig[0])),largest=np.sqrt(max(0,eig[-1])),condition=float(np.sqrt(eig[-1]/eig[0]))if eig[0]>0 else None,smallest_right_trial_coefficients=v[:,0].tolist(),matrix_symmetry_defect=float(np.max(abs(g-g.T))),roundoff_indicator=float(np.finfo(float).eps*len(g)*scale),full_gap_proved=False)
 report['cases'][f.stem]=case;report['inputs_sha256'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
for n in (32,64,128):
 h=1/n;i=np.arange(n)
 def d1(q):return (np.roll(q,2)-8*np.roll(q,1)+8*np.roll(q,-1)-np.roll(q,-2))/(12*h)
 def d2(q):return (-np.roll(q,2)+16*np.roll(q,1)-30*q+16*np.roll(q,-1)-np.roll(q,-2))/(12*h*h)
 row={}
 for k in (1,n//2-1,n//2):
  q=np.exp(2j*np.pi*k*i/n);C=-d1(d1(q));E=-d2(q);lc=np.sum(q.conjugate()*C)/n;le=np.sum(q.conjugate()*E)/n;poisson=4*np.sin(np.pi*k/n)**2/h**2
  row[str(k)]=dict(theta_over_pi=2*k/n,composed_eigenvalue_real=float(lc.real),expanded_eigenvalue_real=float(le.real),imaginary_roundoff=max(float(abs(lc.imag)),float(abs(le.imag))),composed_rightMG_ratio=float(lc.real/poisson),expanded_rightMG_ratio=float(le.real/poisson),low_frequency_composed_relative_error=float(abs(lc.real/(2*np.pi*k)**2-1))if k==1 else None)
 checker=(-1.)**i;row['exact_real_Nyquist']=dict(D1_max=float(np.max(abs(d1(checker)))),D1_squared_max=float(np.max(abs(d1(d1(checker))))),directD2_max=float(np.max(abs(d2(checker)))))
 report['periodic_controls'][str(n)]=row
report['periodic_derivation']='D1 symbol i sin(theta)(4-cos(theta))/(3h); -D1^2 zero at theta=pi. Relative to -D2_second, ratio cos(theta/2)^2*(4-cos(theta))^2/9; direct fourth-D2 ratio1+sin(theta/2)^2/3. This periodic null does not assert a null on the nonperiodic mapped domain.'
report['analysis_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
with tempfile.TemporaryDirectory(dir=out)as cache:
 os.environ['MPLCONFIGDIR']=cache;os.environ['XDG_CACHE_HOME']=cache
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained');theta=np.linspace(0,np.pi,500);axes[0].plot(theta/np.pi,1+np.sin(theta/2)**2/3,label='Expanded fourth-D2');axes[0].plot(theta/np.pi,np.cos(theta/2)**2*(4-np.cos(theta))**2/9,label='Composed D1 squared');axes[0].set(xlabel='Periodic frequency / Nyquist',ylabel='Right scalar-Poisson ratio',title='Exact periodic null at Nyquist');axes[0].legend()
 for n in (6,8,10):
  row=report['cases'][f'n{n}-s7-r0'];v=row['restricted_singular_controls']
  for name,ls in [('expanded_rightMG','-'),('composed_rightMG','--')]:axes[1].semilogy(np.arange(1,len(v[name]['singular_values'])+1),v[name]['singular_values'],ls,label=f'N{n} '+name.replace('_rightMG',''))
 axes[1].set(xlabel='Restricted singular index',ylabel='Singular value of actual A P on trial space',title='42 low/high modes; no full spectral-gap claim');axes[1].legend(fontsize=7)
 for ax in axes:ax.grid(alpha=.2)
 fig.suptitle('Unadopted composed gauge: periodic kernel and bounded mapped controls');fig.savefig(out/'frequency-controls.png',dpi=160);plt.close(fig)
print(json.dumps({k:{n:dict(smin=v['smallest'],smax=v['largest'],condition=v['condition'])for n,v in x['restricted_singular_controls'].items()if 'rightMG'in n}for k,x in report['cases'].items()}))
