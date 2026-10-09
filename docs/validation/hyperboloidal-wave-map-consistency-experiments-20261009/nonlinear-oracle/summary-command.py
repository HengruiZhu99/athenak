from pathlib import Path
import json
import hashlib
import mpmath as mp
p=Path(__file__).resolve().parent
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
results={}
mp.mp.dps=130
for digits in (80,110):
 data=json.loads((p/('attempt002/oracle-%d.json'%digits)).read_text())
 for row in data['cases']:
  for c in row['checks']:
   name=c['name']
   kind=next((k for k in ('native_ref','reference_connection','implicit_first','implicit_second','implicit_third','ADM_g00','det_gtilde','A_trace','Lambda_identity','physical_H','physical_M','conformal_source','wave_map_alpha','wave_map_beta','wave_map','harmonic_Yhat') if name.startswith(k)),name)
   v=mp.mpf(c['scaled'])
   previous=results.get(kind)
   if previous is None or v>mp.mpf(previous['scaled_max']):
    results[kind]={'scaled_max':c['scaled'],'absolute_at_scaled_max':c['absolute'],'point':row['point_name'],'epsilon':row['epsilon'],'digits':digits,'passed':c['passed']}
summary={'kind':'Compact saved-check category readback','payloads':{'oracle-%d.json'%d:sha(p/('attempt002/oracle-%d.json'%d)) for d in (80,110)},'source_command':'Exact saved-data reduction shown in captured summary-command.py','categories':results}
(p/'category-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(sha(p/'category-summary.json'))
