#!/usr/bin/env python3
"""Compare invariant physical units without changing literal amplification caps."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();source=a.evidence/'probe.csv.units.csv';d=np.genfromtxt(source,delimiter=',',names=True);cart=d['face']<0
r={'normalization_original':'h=1/(N max_A||J_A||), never J00','Cartesian_direction_spacing_defect':float(max(np.max(abs(d['h_direction_'+a][cart]/d['h_common'][cart]-1)) for a in ['x','y','z'])),'Cartesian_minimum_spacing_defect':float(np.max(abs(d['h_min_singular'][cart]/d['h_common'][cart]-1))),'all_scalar_rows':len(d),'directional':{},'minimum_singular':{},'raw_physical':{},'faces':{},'no_cap_relaxation':True}
for name,pre in [('directional','directional'),('minimum_singular','minimum')]:
 for method in ['base','new']:
  g=d[pre+'_'+method+'_gradient'];h=d[pre+'_'+method+'_hessian'];bad=(g>32)|(h>64);r[name][method]={'gradient_max':float(max(g)),'hessian_max':float(max(h)),'literal32_64_violations':int(sum(bad)),'Cartesian_violations':int(sum(bad&cart))}
for method in ['base','new']:r['raw_physical'][method]={'gradient_L1_max':float(max(d['raw_'+method+'_gradient'])),'hessian_L1_max':float(max(d['raw_'+method+'_hessian']))}
keys=['raw_base_gradient','raw_new_gradient','raw_base_hessian','raw_new_hessian','directional_base_gradient','directional_new_gradient','directional_base_hessian','directional_new_hessian','minimum_base_gradient','minimum_new_gradient','minimum_base_hessian','minimum_new_hessian']
for face in range(6):r['faces'][face]={key:float(max(d[key][d['face']==face])) for key in keys}
r['six_face_axis_permutation_relative_defects']={key:float((max(r['faces'][f][key] for f in range(6))-min(r['faces'][f][key] for f in range(6)))/max(r['faces'][f][key] for f in range(6))) for key in keys};peak=int(np.argmax(d['minimum_new_hessian']));r['peak_minimum_hessian_row']={k:float(d[k][peak]) for k in d.dtype.names};r['input_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();assert r['Cartesian_direction_spacing_defect']<1e-13 and r['Cartesian_minimum_spacing_defect']<1e-13 and max(r['six_face_axis_permutation_relative_defects'].values())<1e-10;r['six_face_permutation_and_Cartesian_contract_passed']=True;r['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(a.evidence/'geometry-analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'directional':r['directional'],'minimum_singular':r['minimum_singular'],'axis_permutation_max_defect':max(r['six_face_axis_permutation_relative_defects'].values())}))
