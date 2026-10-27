"""Read-only analysis of saved profiles. Does not import or execute an evolution."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]; OUT=ROOT/'output/elliptic-shift-1d'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[];sources=[];runs=[];saved={}
for nu,group,prefix in [(1.,'wormhole-physical-log-compiled-v1','n'),(.25,'wormhole-slicing-front-controls-v1','nu025-n'),(0.,'wormhole-slicing-front-controls-v1','harmonic-n')]:
 for n in ([513,1025,2049,4097] if nu==1 else [513,1025,2049]):
  folder=OUT/group/f'{prefix}{n}';mp=folder/'manifest.json';m=json.loads(mp.read_text());assert m['status']=='finished'
  sources.append(dict(path=str(mp.relative_to(ROOT)),sha256=sha(mp)))
  (HERE/f'manifest-nu{nu}-n{n}.json').write_text(json.dumps(m,indent=2)+'\n')
  runs.append(dict(nu=nu,n=n,status=m['status'],time=m['time'],stages=m['all_RK_stages']))
  for p in sorted(folder.glob('profile-*.npz')):
   with np.load(p,allow_pickle=False) as z:r=z['r'];d=z['diagnostic'];q=z['q_physical'];stored=z['q'];b=z['beta'];t=float(z['t'])
   if not (.5<=t<=3.01):continue
   O=(1-r*r)/2; mask=(r>.03)&(r<.3);ids=np.flatnonzero(mask);h=r[1]-r[0]
   lapse=d[mask,7]/O[mask];grad=np.gradient(lapse,r[mask],edge_order=2);j=ids[np.argmax(abs(grad))]
   F=d[:,7]**2+4*nu*O**2*d[:,7];cm=-b-np.sqrt(F*d[:,8]);cp=-b+np.sqrt(F*d[:,8]);dc=np.gradient(cm,r,edge_order=2)
   hi=ids[np.argmax(abs(d[mask,0]))]
   rec=dict(nu=nu,n=n,t=t,gradient=float(max(abs(grad))),h_gradient=float(h*max(abs(grad))),r_gradient=float(r[j]),H_local=float(abs(d[hi,0])),r_H=float(r[hi]),cminus=float(cm[j]),dcminus=float(dc[j]),profile=str(p.relative_to(ROOT)),sha256=sha(p))
   if abs(t-2.5)<1e-9:
    saved[(nu,n)]=(r.copy(),d.copy(),cm.copy(),cp.copy());K=q[hi,3]-3+2*q[hi,4];chi=d[hi,8];x=r[hi];o=O[hi]
    terms={}
    for width in [7,9]:
     ii=np.arange(hi-width//2,hi+width//2+1);xx=(r[ii]-x)/h;M=xx[None,:]**np.arange(width)[:,None]
     rhs=np.zeros(width);rhs[1]=1;w1=np.linalg.solve(M,rhs)/h;rhs[:]=0;rhs[2]=2;w2=np.linalg.solve(M,rhs)/h**2
     z=stored[ii,0]-stored[hi,0];u1=w1@z;u2=w2@z;D=x**4+o**4;D1=4*x**3-4*x*o**3;D2=12*x*x-4*o**3+12*x*x*o*o
     l1=4/x-D1/D+u1;l2=-4/x**2-D2/D+(D1/D)**2+u2;ch1=chi*l1;ch2=chi*(l1*l1+l2)
     values=dict(K2=2*K*K/3,metric0=-6*chi,metric1=2*o*x*ch1,metric2=o*o*(2*ch2+4*ch1/x-2.5*ch1*ch1/chi),A2=-1.5*o*o*q[hi,2]**2)
     terms[str(width)]=dict(terms=values,sum=float(sum(values.values())),stored_H=float(d[hi,0]))
    rec['H_decomposition_at_peak']=terms
   records.append(rec)
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(10,7.2),layout='constrained');colors=['#0072B2','#D55E00','#009E73','#CC79A7']
for n,c in zip([513,1025,2049,4097],colors):
 a=[x for x in records if x['nu']==1 and x['n']==n];axs[0,0].semilogy([x['t'] for x in a],[x['gradient'] for x in a],color=c,label=f'N={n}')
 r,d,cm,cp=saved[(1.,n)];mask=(r>.105)&(r<.135);axs[0,1].plot(r[mask],cm[mask],'.-',color=c,markersize=3,label=f'N={n}')
axs[0,0].set(xlabel='Time',ylabel=r'$\max_{0.03<r<0.3}|\partial_r\widetilde\alpha|$',title=r'(a) Front sharpening, $\nu=1$');axs[0,0].legend(fontsize=8)
axs[0,1].set(xlabel='r',ylabel=r'$c_-=-\beta-\sqrt{F\chi}$',title=r'(b) Ingoing gauge speed at $t=2.5$')
for nu,c,label in [(1.,colors[0],r'$\nu=1$'),(.25,colors[1],r'$\nu=0.25$'),(0.,colors[2],r'$\nu=0$ (harmonic principal part)')]:
 a=[x for x in runs if x['nu']==nu];ns=[x['n'] for x in a]
 for ax,key in zip(axs[1],['max_H_rms','max_M_rms']):ax.loglog(ns,[x['stages'][key] for x in a],'o-',color=c,label=label)
for ax,title,yl in zip(axs[1],['(c) Hamiltonian: whole-run maximum','(d) Momentum: remaining error'],[r'$\max_{\rm RK\ stages}\|\mathcal{H}\|_r$',r'$\max_{\rm RK\ stages}\|\mathcal{M}_r\|_r$']):
 ax.set(xlabel='Grid points N',ylabel=yl,title=title);ax.set_xticks([513,1025,2049,4097]);ax.set_xticklabels(['513','1025','2049','4097']);ax.minorticks_off()
axs[1,0].legend(fontsize=8)
for ax in axs.flat:ax.grid(alpha=.2)
fig.savefig(HERE/'diagnosis.pdf');fig.savefig(HERE/'diagnosis.png',dpi=180)
(HERE/'review.json').write_text(json.dumps(dict(scope='Existing finished trajectories only. Gradient and speed derivatives use a second-order diagnostic. No characteristic-ray integration or shock theorem. Stage maxima are not accepted-state-only errors. Varying nu also varies its consistent restoring source. H decomposition uses the stored AF log metric and independently generated 7/9-point derivatives at the local H peak.',runs=runs,records=records,sources=sources,analysis_sha256=sha(Path(__file__))),indent=2)+'\n')
for x in records:
 if abs(x['t']-2.5)<1e-9:print(json.dumps(x))
