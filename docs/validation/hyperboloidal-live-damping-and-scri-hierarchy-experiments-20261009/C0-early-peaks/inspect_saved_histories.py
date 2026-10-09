"""Read existing C0 histories only: sampled peaks and endpoint trends, no rerun."""
from pathlib import Path
import hashlib,json
w=Path(__file__).resolve().parent;old=w.parent/'full-tensor-propagator/full22-v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
agreement=old/'canonical-vs-krylov-all-states.json'
r={'scope':'Read-only inspection of existing t0..2 histories at dt_output=.025; sampled peaks only, no all-time growth/decay classification',
 'constraint_source':'actual canonical Taylor propagated states and native H/M/Z diagnostic receipts',
 'field_units_source':'existing native field-derivative analysis on independent Arnoldi states; all81times/2seeds independently agree with canonical states within1.76e-13 relative',
 'field_units_norm':'configuration H1 plus momenta L2 component scaling, not a physical energy or proved symmetrizer',
 'canonical_vs_Arnoldi_receipt_sha256':sha(agreement),'gauges':{}}
def inspect(rows,value):
 peak=max(rows,key=value);last=rows[-1];prev=rows[-2]
 earlier=min(rows,key=lambda x:abs(x['time']-1.9));assert abs(earlier['time']-1.9)<1e-14
 a,b,c,d=map(value,[peak,last,prev,earlier])
 return {'sampled_peak':a,'peak_time':peak['time'],'t2':b,'t1_975':c,'t1_9':d,
         'endpoint_over_sampled_peak':b/a if a else None,
         'last_step_difference':b-c,'last_step_secant_slope':(b-c)/(last['time']-prev['time']),
         'last_0_1_secant_slope':(b-d)/(last['time']-earlier['time']),
         'last_step_trend':'rising' if b>c else 'falling' if b<c else 'unchanged',
         'last_0_1_net_trend':'rising' if b>d else 'falling' if b<d else 'unchanged'}
for g in ['production','spatialnorm']:
 p=old/f'{g}-projected-expm-analysis.json';f=old/f'{g}-projected-krylov-field-analysis.json'
 a=json.loads(p.read_text());b=json.loads(f.read_text());out=[]
 for h,hf in zip(a['histories'],b['histories']):
  assert h['name']==hf['name'];v=h['history'];u=hf['history'];assert len(v)==len(u)==81
  assert all(x['time']==y['time'] for x,y in zip(v,u))
  metrics={name:inspect(v,lambda x,k=k:x['native_H_M_Z_rms'][k]) for k,name in enumerate(['H','M','Z'])}
  metrics['free20_Euclidean_amplification']=inspect(v,lambda x:x['euclidean_component_amplification'])
  metrics['all22_reference_volume_H1_amplification']=inspect(v,lambda x:x['reference_volume_component_H1_amplification'])
  metrics['configuration_H1_momentum_L2_amplification']=inspect(u,lambda x:x['configuration_H1_momentum_L2_amplification'])
  out.append({'seed':h['name'],'metrics':metrics})
 r['gauges'][g]={'canonical_constraint_history_sha256':sha(p),'existing_field_units_history_sha256':sha(f),'seeds':out}
(w/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for g,a in r['gauges'].items():
 for row in a['seeds']:
  print(g,row['seed'])
  for name,d in row['metrics'].items():print(name,'peak',d['sampled_peak'],'@',d['peak_time'],'t2',d['t2'],'laststep',d['last_step_trend'],'last.1',d['last_0_1_net_trend'])
print('results_sha256',sha(w/'results.json'))
