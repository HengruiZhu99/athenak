#!/usr/bin/env python3
"""Retain all-cell and regionwise original/discrete connection residuals."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'scope':'Same actual affine evolved geometry; accepted affine beta versus freshly solved old-operator beta; no new evolution','weights':{'physical':'stored positive normalized map volume*sqrt(det(g))/chi^(3/2), no Omega factor; each region renormalized by its full weight','conformal':'stored map volume*sqrt(det(g)); each region renormalized by its full weight','surface':'uniform point vector RMS; no physical volume interpretation at Omega=0'},'all_cells_retained':True,'cases':{},'source_sha256':sha(Path(__file__)),'inputs':{}}
fields=['expanded_original','actual_discrete','discrete_minus_expanded','product_leibniz','physical_alpha_chart_difference','A_jet_difference','curvature_product_total','beta_D2_vs_D1composition','shift_metric_products_and_halos','decomposition_closure','metric_built_V','independent_Lambda']
for case in ['actual','old_operator']:
 item={}
 for location in ['bulk','surface']:
  path=a.input/(case+'.'+location+'.csv.gz');q=np.genfromtxt(path,delimiter=',',names=True);report['inputs'][str(path)]=sha(path);size=len(q);block=q['point'].astype(int)//216;r=q['r'];regions={'all':np.ones(size,dtype=bool)}
  if location=='bulk':
   regions.update({'core_r_lt_0.1':r<.1,'cartesian':block<8,'shells':block>=8,'transition_0.62_to_0.70':(r>=.62)&(r<=.70),'outer_r_ge_0.8':r>=.8})
   for layer in range(7):regions['shell_layer_'+str(layer)]=(block>=8)&((block-8)//6==layer)
  else:
   for face in range(6):regions['face_'+str(face)]=(q['point'].astype(int)//36==face)
  rows={}
  for region,mask in regions.items():
   data={'count':int(mask.sum()),'physical_weight_sum':float(q['weight'][mask].sum()),'conformal_weight_sum':float(q['conformal_weight'][mask].sum()),'fields':{}}
   for name in fields:
    v=np.stack([q[name+'_'+str(j)] for j in range(3)],axis=1);squared=np.sum(v*v,axis=1);maximum=np.max(np.abs(v),axis=1);f={}
    for measure,key in [('physical','weight'),('conformal','conformal_weight')]:f[measure+'_vector_rms']=float(np.sqrt(np.sum(q[key][mask]*squared[mask])/np.sum(q[key][mask])))
    f['point_component_max']=float(maximum[mask].max());f['point_vector_rms']=float(np.sqrt(squared[mask].mean()));indices=np.flatnonzero(mask);idx=indices[np.argsort(maximum[mask])[-10:][::-1]];f['top10']=[{'point':int(q['point'][i]),'block':int(block[i]) if location=='bulk' else None,'xyz':[float(q[t][i]) for t in ['x','y','z']],'r':float(r[i]),'components':v[i].tolist(),'chi':float(q['chi'][i]),'alpha':float(q['alpha'][i])} for i in idx];data['fields'][name]=f
   rows[region]=data
  item[location]=rows
 report['cases'][case]=item
# Cross-mode files share exactly the same nodal geometry/positive fields and weights.
A=np.genfromtxt(a.input/'actual.bulk.csv.gz',delimiter=',',names=True);B=np.genfromtxt(a.input/'old_operator.bulk.csv.gz',delimiter=',',names=True)
report['same_geometry_and_scalar_state_bitwise']={name:bool(np.array_equal(A[name],B[name])) for name in ['point','x','y','z','r','weight','conformal_weight','chi','alpha']}
(a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({case:{loc:{region:{name:data['fields'][name]['physical_vector_rms'] for name in ['actual_discrete','beta_D2_vs_D1composition','product_leibniz']} for region,data in item[loc].items() if region in ['all','cartesian','transition_0.62_to_0.70']} for loc in item} for case,item in report['cases'].items()},indent=2))
