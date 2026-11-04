"""Read admitted BN1 N20 endpoint; compare preserved BN1 N16 at r>.3."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
OUT=HERE/'n16-n20';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
raw=ROOT/'output/boosted-normalized-shift/aurora'
camp=raw/'campaigns/n20-boosted-normalized-shift-v1-s20-evolution-segment02'
admit=read(camp/'checkpoint-verified.json');assert admit['verified'] and admit['cases'][0]['completed']
cp=raw/admit['cases'][0]['checkpoint'];assert sha(cp)==admit['cases'][0]['sha256']
report=read(camp/'output/puncture.report.json')
target='hyperbolic_shift_3d_checkpoint_constraints_native_control_broad_smooth_six_shells_boosted_free_gauge_refined_core_matched_cfl_normalized_shift'
binary=ROOT/'build-boosted-refined-core-v1/tst/hyperbolic_shift_3d'/target
flags=binary.parent/'CMakeFiles'/f'{target}.dir/flags.make'
assert 'HYPERBOLIC_NORMALIZED_INNER_SHIFT=1' in flags.read_text()
radial=OUT/'endpoint-radial.json'
cmd=[str(binary),'20',str(cp),'.0002','.3',str(radial),'0,.03,.06,.1,.12,.16,.18,.2,.25,.3,.305,.4,.45,.55,.6,.675,.7,.725,.75,.775,.8,.85,.875,.9,.95,1']
for d in [HERE/'n16-completed',HERE/'n20-segment01',HERE/'n20-segment02']:
 a=np.genfromtxt(d/'puncture.history.csv',delimiter=',',names=True)
 assert np.max(a['exterior_cutoff'][a['horizon_complete_surface']==1])<.3
prov=dict(command=cmd,binary_sha256=sha(binary),checkpoint_sha256=sha(cp),flags=flags.read_text(),cutoff=.3,qualification='r>.3 exceeds all recorded complete candidate enclosing radii in both histories. This is a coordinate-domain comparison, not an accepted final horizon or invariant physical-domain comparison. Legacy horizon_acceptance_assumed in diagnostic is not horizon evidence.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tst/hyperbolic_shift_3d/checkpoint_constraints.cpp',ROOT/'tst/hyperbolic_shift_3d/coupled_mpi.cpp']})
save('endpoint-provenance.json',prov)
if not radial.exists():
 with (OUT/'endpoint-radial.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
assert sha(cp)==prov['checkpoint_sha256']
new=read(radial);old=read(HERE/'n16-comparison/endpoint-radial.json')
for k in ['H_L2','M_L2','Mr_L2']:assert abs(new['norms']['total'][k]/report[k]-1)<1e-9
regions={}
for label,r in [('BN1 N16',old),('BN1 N20',new)]:
 rr=r['radial_cumulative'];regions[label]=[dict(lo=a['cutoff'],hi=b['cutoff'],**{k:a[k]-b[k] for k in ['H2','M2','Mr2']}) for a,b in zip(rr[:-1],rr[1:])]
summary={'N16':old['norms']['exterior'],'N20':new['norms']['exterior'],'relative_change':{k:new['norms']['exterior'][k]/old['norms']['exterior'][k]-1 for k in ['H_L2','M_L2','Mr_L2']},'effective_order':{k:float(np.log(old['norms']['exterior'][k]/new['norms']['exterior'][k])/np.log(20/16)) for k in ['H_L2','M_L2','Mr_L2']},'order_qualification':'Two-point residual scaling with space/time refined together; not field self-convergence or fourth-order proof.','regions':regions,'qualification':prov['qualification']}
save('endpoint-comparison.json',summary)
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,key in zip(axes,['H2','M2']):
 for label,rr in regions.items():
  ax.stairs([v[key] for v in rr],[rr[0]['lo']]+[v['hi'] for v in rr],label=label)
 ax.set(yscale='log',xlabel='Compact radius r',ylabel=key+' regional integral',title='Admitted 10M endpoints');ax.axvline(.3,ls='--',color='gray');ax.grid(alpha=.2)
axes[0].legend();fig.suptitle('Same native constraint operators; common exterior cutoff r>0.3')
fig.tight_layout();fig.savefig(OUT/'radial.png',dpi=170);fig.savefig(OUT/'radial.pdf');plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='regions'},indent=2))
