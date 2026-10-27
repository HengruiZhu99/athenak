#!/usr/bin/env python3
"""All-region MMS error report; finite/constant checks do not imply accuracy pass."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory;rows=[json.loads(f.read_text()) for f in sorted(d.glob('mms*.stdout'))];out=dict(results=rows,all_constants_exact=all(x['constant_max']==0 for x in rows),maximum_normal_jet_change=max(x['original_pure_normal_jet_difference_max'] for x in rows),all_cells_under50000=all(x['cells']<=50000 for x in rows),prototype_not_adopted=True,accuracy_gate='provisional/underresolved; no all-region fourth-order acceptance claim',observed_orders=[])
for layers in [1,2,7]:
 selected=sorted((x for x in rows if x['layers']==layers and x['subdivisions']==1 and x['radial_nodes']==6),key=lambda x:x['n'])
 for x,y in zip(selected,selected[1:]):
  pair=dict(layers=layers,n_from=x['n'],n_to=y['n'],orders={})
  for region in ['all_surface','edge_band','corner_band','exact_patch_seams_and_corners','analytic_pullback_all_surface','analytic_pullback_exact_seams']:
   if region not in x:continue
   pair['orders'][region]={k:math.log(x[region][k]/y[region][k])/math.log(y['n']/x['n']) for k in ['first_max','second_max','normal_tangent_hessian_max'] if x[region][k]>0 and y[region][k]>0}
  out['observed_orders'].append(pair)
(d/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['results','observed_orders']}))
