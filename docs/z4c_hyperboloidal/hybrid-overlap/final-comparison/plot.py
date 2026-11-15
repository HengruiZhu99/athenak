"""Reproduce final history comparisons: python3 plot.py (NumPy, Matplotlib)."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
manifest=json.loads((R/'input-manifest.json').read_text())
for f,m in manifest.items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==m['sha256']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
colors={'legacy':'#596575','full':'#167db3','masked':'#d48629'}
labels={'legacy':'Legacy','full':'Unmasked + full agreement','masked':'Previous mask + agreement'}
titles=['Hamiltonian H','Momentum M','Theta','Z']
scale=np.array([1/9,1/9,1/3,1/3])
H={};S={};assessment={};duplicates=[]
for n,dt in [(12,.00046944203735617445),(16,.00035208152801713085)]:
 for mode in colors:
  for kind,dest in [('z4c_hyperboloidal',H),('scri',S)]:
   rows={}
   for p in sorted((R/'histories'/f'N{n}'/mode).glob(f'segment-*/*.{kind}.hst')):
    a=np.loadtxt(p,ndmin=2);assert np.isfinite(a).all() and np.all(a[:,1]==dt),(p,'finite/dt')
    for r in a:
     c=round(r[0]/dt)
     if c in rows:
      # Constraint rows are instantaneous and must agree across a restart.
      # Scri rows are maxima since the last dump and may span different windows.
      if kind=='z4c_hyperboloidal':
       error=float(np.max(abs(rows[c]-r)/np.maximum(abs(rows[c]),1e-30)))
       assert error<1e-9,(p,c,'duplicate mismatch exceeds tolerance')
       if error:duplicates.append({'N':n,'mode':mode,'cycle':c,'max_relative_difference':error})
     rows[c]=r
   dest[n,mode]=rows
 new=H[n,'full'];old=H[n,'legacy'];common=sorted(c for c in new.keys()&old.keys() if new[c][0]*3>=.25 and new[c][0]*3<=30)
 t=np.array([new[c][0]*3 for c in common]);a=np.array([new[c] for c in common]);b=np.array([old[c] for c in common]);assert np.max(np.abs(a[:,0]-b[:,0]))<1e-9
 ratios=a[:,8:12]/b[:,8:12]
 whole_ratios=a[:,3:7]/b[:,3:7]
 np.savetxt(R/f'N{n}-matched-ratios.csv',np.column_stack([t,ratios,whole_ratios]),delimiter=',',header='t_M0,exterior_H,exterior_M,exterior_Theta,exterior_Z,whole_H,whole_M,whole_Theta,whole_Z')
 result=json.loads((R/f'N{n}-final-result.json').read_text());assert result['complete'] and max(new)==result['last']['cycle']
 assessment[str(n)]={'last_time_M0':result['last']['time_M0'],'last_exact_common_time_M0':float(t[-1]),'exterior_full_over_legacy':ratios[-1].tolist(),'whole_full_over_legacy':whole_ratios[-1].tolist(),'last_exterior_M0_units':(new[max(new)][8:12]*scale).tolist(),'matched_samples':len(common),'max_common_time_mismatch_native':float(np.max(np.abs(a[:,0]-b[:,0]))),'max_relative_exterior_volume_difference':float(np.max(abs(a[:,7]/b[:,7]-1))),'max_relative_whole_volume_difference':float(np.max(abs(a[:,2]/b[:,2]-1))),'finite_histories':True,'dt':dt,'last_masked_history_M0':float(H[n,'masked'][max(H[n,'masked'])][0]*3)}
 for domain,offset in [('exterior',8),('whole-domain',3)]:
  fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
  for j,ax in enumerate(axs.flat):
   for mode in ['legacy','masked','full']:
    data=np.array([r for c,r in sorted(H[n,mode].items()) if r[0]*3<=30.001])
    ax.semilogy(data[:,0]*3,np.ma.masked_less_equal(data[:,offset+j]*scale[j],0),c=colors[mode],lw=1.8,label=labels[mode])
   ax.set(title=titles[j],xlabel=r'$t/M_0$',ylabel=r'$M_0^2$ RMS' if j<2 else r'$M_0$ RMS',xlim=(0,30));ax.grid(alpha=.2)
  axs[0,0].legend(fontsize=8)
  note='Previous masked N12 failed at 18.75 M₀' if n==12 else 'Previous masked N16 stopped at user request; not a demonstrated numerical failure'
  fig.suptitle(f'N{n}: {domain} constraint norms\n{note}',fontsize=13)
  for ext in ['png','pdf']:fig.savefig(R/f'N{n}-{domain}.{ext}',dpi=180)
  plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for j,ax in enumerate(axs.flat):
 for n,c in [(12,'#167db3'),(16,'#aa3e65')]:
  d=np.loadtxt(R/f'N{n}-matched-ratios.csv',delimiter=',');use=d[:,0]>=12
  ax.plot(d[use,0],d[use,j+1],label=f'N{n}',c=c,lw=2)
 ax.axhline(1,c='#596575',ls=':',lw=1.4);ax.set(title=titles[j],xlabel=r'$t/M_0$',ylabel='Full agreement / legacy exterior RMS',xlim=(12,30));ax.grid(alpha=.2)
axs[0,0].legend();fig.suptitle('Late-time constraints at identical recorded times\nBelow 1 means lower than legacy',fontsize=14)
for ext in ['png','pdf']:fig.savefig(R/f'late-time-ratios.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for row,n in enumerate([12,16]):
 for col,(idx,name) in enumerate([(6,r'$|\chi Q-1|$ at scri'),(8,r'$|\Theta|$ at scri')]):
  ax=axs[row,col]
  for mode in ['legacy','full']:
   if not S[n,mode]:continue
   d=np.array([v for c,v in sorted(S[n,mode].items()) if v[0]*3<=30.001]);ax.semilogy(d[:,0]*3,np.ma.masked_less_equal(d[:,idx],0),label=labels[mode],c=colors[mode],lw=1.8)
  ax.set(title=f'N{n}: '+name,xlabel=r'$t/M_0$',ylabel='Native conformal units',xlim=(0,30));ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8);fig.suptitle('Scri diagnostic maxima over nodes and RK stages since previous dump\nRestart segments can change accumulation windows; curves are not exact paired ratios',fontsize=12)
for ext in ['png','pdf']:fig.savefig(R/f'scri-diagnostics.{ext}',dpi=180)
plt.close(fig)
# Cross-resolution estimates use each N12 recorded time within N16 coverage.
# N16 norms are linearly interpolated in time, never extrapolated.
convergence={}
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for mode in ['legacy','full']:
 a=np.array([r for c,r in sorted(H[12,mode].items()) if .25<=r[0]*3<=30])
 b=np.array([r for c,r in sorted(H[16,mode].items()) if r[0]*3<=30.001])
 keep=(a[:,0]>=b[0,0])&(a[:,0]<=b[-1,0]);a=a[keep]
 aligned=np.column_stack([np.interp(a[:,0],b[:,0],b[:,j]) for j in range(b.shape[1])])
 exterior=np.log(a[:,8:12]/aligned[:,8:12])/np.log(16/12)
 whole=np.log(a[:,3:7]/aligned[:,3:7])/np.log(16/12)
 # Record exact coincident samples separately as an interpolation cross-check.
 exact=[]
 for row in a:
  k=int(np.argmin(abs(b[:,0]-row[0])))
  if abs(b[k,0]-row[0])<1e-9:
   exact.append({'time_M0':float(row[0]*3),'exterior_p':(np.log(row[8:12]/b[k,8:12])/np.log(16/12)).tolist()})
 convergence[mode]={'last_time_M0':float(a[-1,0]*3),'exterior_p':exterior[-1].tolist(),'whole_p':whole[-1].tolist(),'last_exact_coincident_sample':exact[-1] if exact else None,'exact_coincident_count':len(exact),'method':'Linear interpolation of N16 norms to N12 times within shared coverage; p=ln(C12/C16)/ln(16/12). No extrapolation.'}
 np.savetxt(R/f'{mode}-apparent-order.csv',np.column_stack([a[:,0]*3,exterior,whole]),delimiter=',',header='t_M0,exterior_H_p,exterior_M_p,exterior_Theta_p,exterior_Z_p,whole_H_p,whole_M_p,whole_Theta_p,whole_Z_p')
 for j,ax in enumerate(axs.flat):
  ax.plot(a[:,0]*3,exterior[:,j],c=colors[mode],lw=1.8,label=labels[mode])
for j,ax in enumerate(axs.flat):
 ax.axhline(0,c='black',ls=':',lw=1);ax.axhline(4,c='#999999',ls='--',lw=1)
 ax.set(title=titles[j],xlabel=r'$t/M_0$',ylabel='Apparent constraint-norm order p',xlim=(0,30));ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8)
fig.suptitle('Two-resolution estimate: p = ln(C₁₂/C₁₆) / ln(16/12)\nTime-aligned exterior norms; two resolutions do not establish asymptotic order',fontsize=13)
for ext in ['png','pdf']:fig.savefig(R/f'apparent-convergence-order.{ext}',dpi=180)
plt.close(fig)
assessment['apparent_convergence']=convergence
assessment['duplicate_policy']={'choice':'Later numbered segment wins at identical cycle; discrepancies bounded by 1e-9 relative, not asserted bitwise.','nonidentical_rows':duplicates}
assessment['limits']=['Histories use original frozen science executable, not optimized performance executable.','Exterior norms use each run\u2019s diagnostic horizon mask. Whole-domain plots remove this selection; masked runs still have different activity geometry.','N12 and N16 full runs reached their last timestep below 30 M0; legacy final checkpoint is one timestep later. Ratios use identical recorded times only.','No order-of-convergence claim from two resolutions.','Agreement mismatch needs a common checkpoint observer; production full runs did not record it.']
(R/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
print(json.dumps(assessment,indent=2))
