"""Localize two-resolution residual deterioration using preserved admitted diagnostics."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; OUT=HERE/'convergence-localization';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
files={16:HERE/'n16-comparison/endpoint-radial.json',20:HERE/'n16-n20/endpoint-radial.json'}
data={n:read(p) for n,p in files.items()}
for n,d in data.items():assert d['n']==n and d['checkpoint_schema']=='HYP3D_MPI_BN1' and d['t_over_M']==10 and d['diagnostic_outer_first_order']==0
rows={n:{r['cutoff']:r for r in d['radial_cumulative']} for n,d in data.items()}
regions=[(.3,.4),(.4,.6),(.6,.775),(.775,.9),(.9,1)]
summary={'input_sha256':{str(p.relative_to(HERE)):sha(p) for p in files.values()},'regions':[],'cutoff_orders':[],'qualifications':['Fixed coordinate domains, not invariant physical domains. r>.3 exceeds all recorded complete candidate enclosing radii.','Regional values are squared residual integrals, not norms. Effective orders are RMS residual ratios, not field self-convergence. Space and time refine together.','Endpoint localization identifies where errors reside, not where they originated. No failed state or new evolution is used.']}
for lo,hi in regions:
 d={'lo':lo,'hi':hi}
 for key in ['H2','M2']:
  a,b=[rows[n][lo][key]-rows[n][hi][key] for n in [16,20]]
  d[key]={'N16':a,'N20':b,'percent_change':100*(b/a-1),'N20_fraction_of_exterior':b/rows[20][.3][key],'delta':b-a}
 summary['regions'].append(d)
for cut in sorted(set(rows[16])&set(rows[20])):
 if cut<.3 or cut>=1:continue
 d={'cutoff':cut}
 for key in ['H2','M2']:
  a,b=[np.sqrt(rows[n][cut][key]/rows[n][cut]['volume']) for n in [16,20]]
  d[key]={'N16_rms':a,'N20_rms':b,'p':float(np.log(a/b)/np.log(20/16))}
 summary['cutoff_orders'].append(d)
# Exact diagnostic times from first segments only: no interpolation or restart rows.
proof=read(HERE/'n16-n20/comparison.json')['provenance'];hist={}
for n,name in [(16,'n16-completed'),(20,'n20-segment01')]:
 p=HERE/name/'puncture.history.csv';assert sha(p)==proof[str(n)][0]['hashes']['puncture.history.csv'];hist[n]=np.genfromtxt(p,delimiter=',',names=True)
matched=[]
for a in hist[16]:
 inds=np.where(abs(hist[20]['t_over_M']-a['t_over_M'])<1e-8)[0]
 if len(inds)!=1:continue
 b=hist[20][inds[0]];matched.append({'t_M':float(a['t_over_M']),'H_ratio':float(b['H_exterior_L2']/a['H_exterior_L2']),'M_ratio':float(b['M_exterior_L2']/a['M_exterior_L2']),'cut16':float(a['exterior_cutoff']),'cut20':float(b['exterior_cutoff'])})
summary['pre_restart_own_candidate_history']=matched
summary['history_qualification']='Own-candidate cuts differ; these ratios are not fixed-domain convergence. Deterioration before restart rules out restart as a necessary cause of that history behavior, not as an influence on the final endpoint.'
fig,axs=plt.subplots(1,3,figsize=(14,4.4))
labels=['.30–.40','.40–.60','.60–.775','.775–.90','>.90']
for ax,key,title in zip(axs[:2],['H2','M2'],['Hamiltonian','Momentum']):
 x=np.arange(len(regions))
 for j,n in enumerate([16,20]):ax.bar(x+(j-.5)*.34,[d[key][f'N{n}'] for d in summary['regions']],width=.34,label=f'N{n}')
 ax.set_xticks(x,labels,rotation=25);ax.set(yscale='log',ylabel='Regional squared residual integral',xlabel='Compact-radius region',title=title);ax.grid(axis='y',alpha=.2)
axs[0].legend()
for key,label in [('H2','H'),('M2','M')]:axs[2].plot([d['cutoff'] for d in summary['cutoff_orders']],[d[key]['p'] for d in summary['cutoff_orders']],'.-',label=label)
axs[2].axhline(0,color='black',lw=.8);axs[2].axhline(4,color='gray',ls=':');axs[2].set(xlabel='Exterior cutoff radius',ylabel='Effective N16→N20 residual order',title='Dependence on excluded region');axs[2].legend();axs[2].grid(alpha=.2)
fig.suptitle('BN1 at 10M: inner exterior deterioration masks outer-region improvement\nSame native diagnostic operators; coordinate domains; two-resolution residual scaling only')
fig.tight_layout();fig.savefig(OUT/'localization.png',dpi=170);fig.savefig(OUT/'localization.pdf');plt.close(fig)
(OUT/'regional-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({'regions':summary['regions'],'selected_orders':[d for d in summary['cutoff_orders'] if d['cutoff'] in [.3,.4,.6,.9]],'last_pre_restart_matches':matched[-4:]},indent=2))
