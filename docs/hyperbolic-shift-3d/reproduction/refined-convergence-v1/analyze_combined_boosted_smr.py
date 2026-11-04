"""Read admitted short checkpoints; do not evolve or replace frozen science."""
from pathlib import Path
import json,importlib.util
import numpy as np
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
E=D/'sealed-topology-controls';B=R/'output/refined-convergence/sealed-topology-candidate-v1'
spec=importlib.util.spec_from_file_location('reader',D/'check_smr_restart.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
old=R/'output/refined-convergence/smr-tube-candidate-v1/checks/boost-mpi1-step2'
new=B/'physics/full-smr-radial24-angular2'
a,at,aa,af=reader.read(str(old));b,bt,ba,bf=reader.read(str(new))
assert at==bt and aa==ba and a['time']==b['time'] and a['dt']==b['dt'] and a['leaves']==b['leaves']==400
n=12;size=400*n**3;u=af[0][:31*size].reshape(400,n,n,n,31);v=bf[0][:31*size].reshape(u.shape)
leaves=np.frombuffer(at,dtype='<i4').reshape(-1,4)
errors=np.zeros(31);inner=np.zeros(31);different=0;inner_different=0
q=(np.arange(n)+.5)/n
for m,(lx,ly,lz,level) in enumerate(leaves):
 width=.8/(4*2**(int(level)-2));x=-.4+(lx+q)*width;y=-.4+(ly+q)*width;z=-.4+(lz+q)*width
 radius=np.sqrt(z[:,None,None]**2+y[None,:,None]**2+x[None,None,:]**2)
 error=np.abs(u[m]-v[m])/(1+np.maximum(np.abs(u[m]),np.abs(v[m])))
 errors=np.maximum(errors,error.max(axis=(0,1,2)));different+=np.count_nonzero(u[m]!=v[m])
 if np.any(radius<.3):
  inner=np.maximum(inner,error[radius<.3].max(axis=0));inner_different+=np.count_nonzero(u[m][radius<.3]!=v[m][radius<.3])
assert np.isfinite(u).all() and np.isfinite(v).all()
def history(prefix):return [json.loads(x) for x in Path(str(prefix)+'.history.jsonl').read_text().splitlines()]
x=history(old);y=history(new);assert [r['time'] for r in x]==[r['time'] for r in y]
records=[]
for r,s in zip(x,y):
 oldnorm=next(z for z in r['fixed'] if z['cut']==.3);newnorm=next(z for z in s['fixed'] if z['cut']==.3)
 records.append(dict(time_M=s['t_over_M'],old_fixed03=oldnorm,new_fixed03=newnorm,reduction={f:oldnorm[f]/newnorm[f] for f in ['H','M','Mr']},old_mass=r['mass'],new_mass=s['mass'],old_horizon_rms=r['horizon_trial_expansion_rms'],new_horizon_rms=s['horizon_trial_expansion_rms'],new_candidate_finest_margin=s['candidate_finest_margin'],new_tracker=s['tracker']))
oldperf=json.loads((R/'output/refined-convergence/boosted-radial24-control/state.rank0.performance.json').read_text());newperf=json.loads((E/'uniform-parity.rank0.performance.json').read_text())
fullperf=json.loads((E/'full-smr-radial24-angular2.rank0.performance.json').read_text())
result=dict(old_checkpoint=a,new_checkpoint=b,nominal_cartesian_max_scaled_by_field=errors.tolist(),inner_r03_max_scaled_by_field=inner.tolist(),nominal_cartesian_different_values=int(different),inner_r03_different_values=int(inner_different),history=records,matched_uniform_peak_rss=dict(old=oldperf['peak_rss_bytes'],new=newperf['peak_rss_bytes'],fraction_reduction=1-newperf['peak_rss_bytes']/oldperf['peak_rss_bytes']),full_smr_native_performance=fullperf,qualification='Same400-leaf Cartesian topology and equations; added radial/angular transition resolution. Two steps only (0.00025M). Native conformal M norm; no long-time, GPU, global convergence or speedup claim. Peak-RSS pair uses separate fresh uniform controls; full-SMR grid cost is not a paired timing benchmark.')
(E/'combined-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(core_max=float(errors.max()),inner_max=float(inner.max()),initial=records[0]['reduction'],step2=records[-1]['reduction']),indent=2))
