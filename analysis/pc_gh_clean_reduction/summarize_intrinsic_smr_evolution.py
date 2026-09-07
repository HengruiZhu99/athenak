#!/usr/bin/env python3
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result={}
for folder in sorted(a.runs.glob('n*-*')):
 if not folder.is_dir():continue
 with (folder/'intrinsic-health-rank0.csv').open() as f:rows=list(csv.DictReader(f))
 health={k:min(float(row[k]) for row in rows) for k in ['min_w','min_rho','min_alpha','min_metric_eigenvalue','min_4_alpha2w2','margin_2_alpha']};health['max_frobenius_condition']=max(float(row['max_frobenius_condition']) for row in rows);assert all(v>0 for v in health.values())
 history=[]
 for file in folder.glob('intrinsic-diagnostics-*.csv'):
  with file.open() as f:data=list(csv.DictReader(f))
  row=dict(time=float(data[0]['time']))
  for group,indices in [('H',[0]),('M',range(1,4)),('GH',range(7,11)),('E',range(11,41)),('Omega',range(41,71)),('OmegaQ',range(71,89))]:row[group]=float(np.sqrt(sum(float(data[i]['RMS'])**2 for i in indices)))
  history.append(row)
 history.sort(key=lambda x:x['time']);result[folder.name]=dict(health=health,history=history)
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print('Summarized',len(result),'runs')
