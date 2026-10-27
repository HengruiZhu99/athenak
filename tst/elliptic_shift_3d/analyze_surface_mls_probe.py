#!/usr/bin/env python3
"""Compare field-independent endpoint MLS controls and original frozen poles."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory;table=d/'atlas-poles.csv';table=table if table.exists() else table.with_suffix('.csv.gz');data=np.genfromtxt(table,delimiter=',',names=True);fields=['chi','Khat','Theta','alpha','Axx','Axy','Axz','Ayy','Ayz','Azz','LambdaX','LambdaY','LambdaZ'];out=dict(prototype_not_adopted=True,mixed_volume_transfer_unresolved=True,analytic_results=[json.loads(x) for x in (d/'analytic.stdout').read_text().splitlines()],snapshot=json.loads((d/'snapshot.stdout').read_text()),pole_modes={})
for mode in ['withinface','crossface_all','crossface_metric']:
 v=np.column_stack([data[mode+'_'+f] for f in fields]);i,j=np.unravel_index(np.argmax(abs(v)),v.shape);out['pole_modes'][mode]=dict(maximum=float(abs(v[i,j])),field=fields[j],row=int(data['row'][i]),face=int(data['face'][i]),original_worst_LambdaY=float(data[mode+'_LambdaY'][54]),per_field_max={f:float(np.max(abs(data[mode+'_'+f]))) for f in fields})
out['pole_gate_passed']=out['pole_modes']['crossface_all']['maximum']<=1e-7;out['negative_control']='At N6 native-node corner-band Hessian error0.235678 is worse than original owner-face0.101124; no derivative/evolution accuracy acceptance from improved pole alone.';(d/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(maximum=out['pole_modes']['crossface_all']['maximum'],pole_gate_passed=out['pole_gate_passed'],maximum_condition=max(x['MLS_max_QR_R_condition_infinity'] for x in out['analytic_results']))))
