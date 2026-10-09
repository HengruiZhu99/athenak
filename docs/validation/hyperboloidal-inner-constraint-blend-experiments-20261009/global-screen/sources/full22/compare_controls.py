"""Matched-time native constraint/component comparisons, no stability claim."""
from pathlib import Path
import argparse,hashlib,json
w=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--stop',type=float,default=2.);a=p.parse_args();old=w.parents[1]/'full-tensor-propagator/full22-v2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();res={'time':a.stop,'long_semantics':'exploratory only; local Arnoldi truncation check, no long independent canonical comparison','gauges':{}}
for g in ['production','spatialnorm']:
 c=json.loads((w/f'{g}-projected-krylov-t{a.stop}-analysis.json').read_text());f=json.loads((w/f'{g}-projected-krylov-t{a.stop}-field-analysis.json').read_text());base=json.loads((old/f'{g}-projected-expm-analysis.json').read_text());bf=json.loads((old/f'{g}-projected-krylov-field-analysis.json').read_text());rows=[]
 for k,(h,hf) in enumerate(zip(c['histories'],f['histories'])):
  assert h['name']==hf['name']==base['histories'][k]['name']
  before=min(base['histories'][k]['history'],key=lambda x:abs(x['time']-a.stop));bfield=min(bf['histories'][k]['history'],key=lambda x:abs(x['time']-a.stop));after=h['history'][-1];field=hf['history'][-1];assert abs(before['time']-a.stop)<1e-14 and abs(after['time']-a.stop)<1e-14
  rows.append({'seed':h['name'],'time':a.stop,'C0_H_M_Z':before['native_H_M_Z_rms'],'blend_H_M_Z':after['native_H_M_Z_rms'],'ratios_H_M_Z':[x/y for x,y in zip(after['native_H_M_Z_rms'],before['native_H_M_Z_rms'])],'C0_field_units_amp':bfield['configuration_H1_momentum_L2_amplification'],'blend_field_units_amp':field['configuration_H1_momentum_L2_amplification'],'C0_free20_Euclidean_amp':before['euclidean_component_amplification'],'blend_free20_Euclidean_amp':after['euclidean_component_amplification'],'blend_outer_H_M_Z_squared_fraction':after['outer_r09_squared_constraints_fraction'],'blend_peak_H_M_Z_radius':after['peak_H_M_Z_radius'],'blend_sampled_max_field_amp':max((r['configuration_H1_momentum_L2_amplification'],r['time']) for r in hf['history'])})
 res['gauges'][g]={'C0_native_analysis_sha256':sha(old/f'{g}-projected-expm-analysis.json'),'rows':rows}
name='pilot-vs-frozen-C0.json' if a.stop==.05 else f'exploratory-t{a.stop}-vs-frozen-C0.json';(w/name).write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
