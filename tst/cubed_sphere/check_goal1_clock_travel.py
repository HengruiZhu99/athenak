#!/usr/bin/env python3
"""Oracle retarded-centre and immutable-reference null travel integrals only."""
from pathlib import Path
import json,hashlib,importlib.util,argparse
import mpmath as m
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();m.mp.dps=80;R=Path(__file__).resolve().parents[2];E=R/'docs/validation/goal1-minkowski-20261018';cache=json.loads((E/'native-global-clock-v1/reused-height-cache.json').read_text());src=R/'docs/validation/hyperboloidal-reference-wave-map-flat-IVP-values-20261009/values-source/flat_ivp_values.py';spec=importlib.util.spec_from_file_location('height_source_only',src);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);recipe=json.loads((src.parent/'values-recipe.json').read_text());height=module.Layer(recipe,128);fine=module.Layer(recipe,96);eps=m.mpf('5e-6');sigma=m.mpf('.2');rows=[]
for r in [m.mpf('.15'),m.sqrt(3)*m.mpf('.15'),m.mpf('.3'),m.mpf('.6'),m.mpf('.7'),m.mpf('.75'),m.mpf(1)]:
 u=m.mpf(cache['outer_constant'])if r==1 else height.defect(r)
 if r==1:radius=m.inf;delta=m.mpf(0);incoming=m.inf
 else:
  omega=1 if r<=m.mpf('.05')else 1-r*r if r>=m.mpf('.95')else 1-r*r/(1+m.exp(1/((r-m.mpf('.05'))/m.mpf('.9'))-1/(1-(r-m.mpf('.05'))/m.mpf('.9'))))
  radius=r/omega;delta=eps*(1-m.exp(-(2*radius/sigma)**2))/radius;incoming=2*radius+u
 rows.append({'compact_radius':str(r),'inertial_reference_radius':str(radius),'uref':str(u),'reference_outgoing_from_origin':str(-u),'oracle_U0_crossing':str(-u+delta),'incoming_advanced_reference_value':str(incoming),'height128_vs96':str(abs(u-(m.mpf(cache['outer_constant'])if r==1 else fine.defect(r))))})
lookup={m.mpf(x['compact_radius']):x for x in rows};delays=[]
for rb,ri in [('.7','.6'),('.7','.3'),('.75','.6'),('.75','.3')]:
 vb=m.mpf(lookup[m.mpf(rb)]['incoming_advanced_reference_value']);vi=m.mpf(lookup[m.mpf(ri)]['incoming_advanced_reference_value']);delays.append({'outer_cut':rb,'interior_radius':ri,'reference_incoming_null_delay':str(vb-vi)})
report={'scope':'Coefficient/oracle travel markers for pilot duration planning; no measured native packet crossing or causal boundary acceptance. Native lapse/shift waveform lobes are not identical to the U=0 potential marker.','epsilon':str(eps),'sigma':str(sigma),'map_reference':'S1,a.5,layer.05,.95; cubeL.15','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reused_height_source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'rows':rows,'finite_cut_reference_incoming_delays':delays,'no_evolution':True};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'outgoing_markers':[{k:x[k]for k in ['compact_radius','oracle_U0_crossing']}for x in rows],'delays':delays},indent=2))
