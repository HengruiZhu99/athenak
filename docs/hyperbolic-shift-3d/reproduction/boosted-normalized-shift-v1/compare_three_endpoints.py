"""Add admitted N24 to preserved N16/N20 same-operator coordinate-domain diagnostics."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists());O=D/'n16-n20-n24';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
raw=R/'output/boosted-normalized-shift/aurora';camp=raw/'campaigns/n24-boosted-normalized-shift-v1-s24-evolution-segment02';admit=read(camp/'checkpoint-verified.json');assert admit['verified'] and admit['cases'][0]['completed']
cp=raw/admit['cases'][0]['checkpoint'];before=sha(cp);assert before==admit['cases'][0]['sha256']
for name in ['n16-completed','n20-segment01','n20-segment02','n24-segment01','n24-segment02']:
 a=np.genfromtxt(D/name/'puncture.history.csv',delimiter=',',names=True);assert np.max(a['exterior_cutoff'][a['horizon_complete_surface']==1])<.3
name='hyperbolic_shift_3d_checkpoint_constraints_native_control_broad_smooth_six_shells_boosted_free_gauge_refined_core_matched_cfl_normalized_shift';binary=R/'build-boosted-refined-core-v1/tst/hyperbolic_shift_3d'/name
flags=(binary.parent/'CMakeFiles'/f'{name}.dir/flags.make').read_text();assert 'HYPERBOLIC_NORMALIZED_INNER_SHIFT=1' in flags
radial=O/'n24-endpoint-radial.json';cmd=[str(binary),'24',str(cp),str(1/6000),'.3',str(radial),'0,.03,.06,.1,.12,.16,.18,.2,.25,.3,.305,.4,.45,.55,.6,.675,.7,.725,.75,.775,.8,.85,.875,.9,.95,1']
prov=dict(command=cmd,binary_sha256=sha(binary),checkpoint_sha256=before,flags=flags,qualification='Same native diagnostic operators; r>.3 exceeds all recorded complete candidate enclosures, but is a coordinate domain, not an invariant physical domain or accepted final horizon. Legacy diagnostic horizon_acceptance_assumed is not horizon evidence.')
(O/'endpoint-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
if not radial.exists():
 with (O/'n24-endpoint-radial.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
assert sha(cp)==before
new=read(radial);report=read(camp/'output/puncture.report.json')
for k in ['H_L2','M_L2','Mr_L2']:assert abs(new['norms']['total'][k]/report[k]-1)<1e-9
all_data={16:read(D/'n16-comparison/endpoint-radial.json'),20:read(D/'n16-n20/endpoint-radial.json'),24:new}
N=np.array([16,20,24]);summary=dict(norms={n:q['norms']['exterior'] for n,q in all_data.items()},orders={},regions={},qualification=prov['qualification'],order_qualification='Constraint residual scaling, not Richardson field self-convergence. Space/time refined together; two adjacent orders and unweighted log residual versus log N fit do not establish an asymptotic regime.')
for key in ['H_L2','M_L2','Mr_L2']:
 E=np.array([summary['norms'][n][key] for n in N]);summary['orders'][key]=dict(pair16_20=float(np.log(E[0]/E[1])/np.log(20/16)),pair20_24=float(np.log(E[1]/E[2])/np.log(24/20)),fit=float(-np.polyfit(np.log(N),np.log(E),1)[0]))
for n,q in all_data.items():
 rr=q['radial_cumulative'];summary['regions'][n]=[dict(lo=a['cutoff'],hi=b['cutoff'],**{k:a[k]-b[k] for k in ['H2','M2','Mr2']}) for a,b in zip(rr[:-1],rr[1:])]
fig,axs=plt.subplots(1,3,figsize=(14,4.5))
for ax,key in zip(axs[:2],['H2','M2']):
 for n,rr in summary['regions'].items():ax.stairs([x[key] for x in rr],[rr[0]['lo']]+[x['hi'] for x in rr],label=f'N{n}')
 ax.axvline(.3,ls='--',color='gray');ax.set(yscale='log',xlabel='Compact r',ylabel=key+' regional integral');ax.grid(alpha=.2)
for key,label in [('H_L2','H'),('M_L2','M'),('Mr_L2','Mr')]:axs[2].plot(N,[summary['norms'][n][key] for n in N],'.-',label=label)
axs[2].set(xscale='log',yscale='log',xlabel='N',ylabel='r > .3 RMS residual');axs[2].set_xticks(N,labels=N);axs[2].xaxis.set_minor_locator(matplotlib.ticker.NullLocator());axs[2].legend();axs[2].grid(alpha=.2);axs[0].legend()
fig.suptitle('BN1 admitted 10M endpoints: same native operators; coordinate-domain comparison');fig.tight_layout();fig.savefig(O/'endpoint-radial.png',dpi=170);fig.savefig(O/'endpoint-radial.pdf');plt.close(fig)
(O/'endpoint-comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='regions'},indent=2))
