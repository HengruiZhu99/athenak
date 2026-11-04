"""Reproduce frozen-coefficient numerical rank evidence, not a nonlinear theorem."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists());RAW=ROOT/'output/refined-convergence/principal-probe';OUT=HERE/'principal-audit';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def project(q):
 P=np.array(q['P']);g=np.array(q['inverse_metric']).reshape(3,3);c=np.array([g[0,0],2*g[0,1],2*g[0,2],g[1,1],2*g[1,2],g[2,2]])
 C=np.zeros((2,28));C[0,1:7]=c;C[1,7:13]=c;Q=np.linalg.svd(C,full_matrices=True)[2][2:].T
 return np.einsum('ia,ij,jb->ab',Q,P,Q),float(np.max(abs(np.einsum('ai,ij,jb->ab',C,P,Q))))
summary={};fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,name in zip(axes,['central6_stationary_matched_plateau','central20_stationary_smr025_radial9']):
 sweep=[json.loads(x) for x in (RAW/(name+'.jsonl')).read_text().splitlines()];cross=[json.loads(x) for x in (RAW/(name+'-crossings.jsonl')).read_text().splitlines()];quarter=[json.loads(x) for x in (RAW/(name+'-crossings-quarter.jsonl')).read_text().splitlines()]
 amp=max(float(np.max(abs(np.array(q['P'])-np.array(w['P'])))) for q,w in zip(cross,quarter));assert amp<1e-12
 roots=[]
 for q in cross:
  if abs(q['shift_speed2']-q['lapse_speed2'])>1e-12:continue
  A,leak=project(q);assert leak<1e-10
  for sign in [-1,1]:
   lam=q['beta_x']+sign*np.sqrt(q['shift_speed2']);T=A-lam*np.eye(len(A));T2=np.einsum('ij,jk->ik',T,T)
   s1=np.linalg.svd(T,compute_uv=False);s2=np.linalg.svd(T2,compute_uv=False);ev=np.linalg.eigvals(A)
   # Absolute thresholds are stated explicitly; report singular values too.
   n1=int(sum(s1<1e-10));n2=int(sum(s2<1e-10));assert n1==1 and n2==2
   roots.append(dict(r=q['r'],sign=sign,lambda_value=float(lam),speed_gap=q['shift_speed2']-q['lapse_speed2'],nullity_T=n1,nullity_T_squared=n2,smallest_singular_T=s1[-3:].tolist(),smallest_singular_T_squared=s2[-3:].tolist(),nearest_eigenvalue_distances=sorted(abs(ev-lam).tolist())[:3],algebraic_trace_tangent_leak=leak))
 base=[q for q in sweep if q['override']==0];strong=[q for q in sweep if q['override']==1]
 r=np.array([q['r'] for q in base]);gap=np.array([q['shift_speed2']-q['lapse_speed2'] for q in base]);g=base[0]['inner_gamma'];s=np.array([q['inner_weight'] for q in base]);double_gap=gap+4/3*s*g
 ax.plot(r,gap,label='normalized inner Gamma = k²');ax.plot(r,double_gap,label='2k² speed-formula control',ls='--');ax.axhline(0,color='k',lw=.7);ax.set(xlabel='compact r',ylabel='c² shift − c² lapse',title='Moderate center' if 'central6' in name else 'Enlarged center');ax.legend(fontsize=8);ax.grid(alpha=.25)
 eigen_imag=0.;leakmax=0.
 for q in sweep:
  A,leak=project(q);leakmax=max(leakmax,leak);eigen_imag=max(eigen_imag,float(np.max(abs(np.linalg.eigvals(A).imag))))
 summary[name]=dict(unit_vs_quarter_amplitude_max_difference=amp,roots=roots,sweep_max_eigen_imaginary_part=eigen_imag,sweep_max_trace_tangent_leak=leakmax,sampled_min_gap_normalized=float(min(gap)),sampled_min_gap_double_normalized=float(min(double_gap)),sampled_min_gap_gamma1=min(q['shift_speed2']-q['lapse_speed2'] for q in strong),sweep_sha256=sha(RAW/(name+'.jsonl')))
 for suffix in ['-crossings.jsonl','-crossings-quarter.jsonl']:shutil.copy2(RAW/(name+suffix),OUT/(name+suffix))
 # Preserve the full matrix sweep compressed; no source information discarded.
 import gzip
 with gzip.open(OUT/(name+'.jsonl.gz'),'wb') as f:f.write((RAW/(name+'.jsonl')).read_bytes())
summary['qualification']='Initial native stationary states; radial frozen-coefficient pseudodifferential symbol from actual production RHS, no KO/upwind. Algebraic determinant/tracefree constraints projected, differential constraints not imposed. Rank(T)=25 and rank(T²)=24 at the identified repeated nonzero modes is numerical Jordan-block evidence. Not a nonlinear well-posedness proof, not a sole-cause attribution of evolved errors. Gamma=1 and doubled Gamma are analysis-only controls; no campaign changed. Inner telegrapher equivalence means the inner crossing may also belong to the corresponding spatial gauge; it is not uniquely a compactification defect.'
(OUT/'verification.json').write_text(json.dumps(summary,indent=2)+'\n');fig.suptitle('Gauge-speed crossings: equal speeds alone are insufficient; rank tests establish the defect');fig.savefig(OUT/'speed-gaps.png',dpi=160);plt.close(fig)
for n in ['build.json','build-crossings.json','build-amplitude-control.json']:
 if (RAW/n).exists():shutil.copy2(RAW/n,OUT/n)
print(json.dumps(summary,indent=2))
