#!/usr/bin/env python3
"""Verify the archived operator exporter from the native research header."""
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();out=a.evidence
rows=[json.loads(x) for x in (out/'operator-native.jsonl').read_text().splitlines() if x.startswith('{')]
r={'identity':[x for x in rows if x['type']=='identity'],'accuracy':[x for x in rows if x['type']=='accuracy'],'KO':[],'orders':[]}
for x in rows:
    if x['type']!='KO':continue
    L=np.array(x['matrix']);h=np.array(x['norm_volume']);n=len(h);D3=np.zeros((n-3,n))
    for i in range(n-3):D3[i,i:i+4]=[-1,3,-3,1]
    W=h[:,None]*L;expected=-.1/64*np.einsum('ji,j,jk->ik',D3,np.array(x['line_weight']),D3)
    defect=np.linalg.norm(W-expected)/max(1,np.linalg.norm(expected));eig=np.linalg.eigvalsh(W+W.T)
    r['KO'].append({'nodes':n,'stretch':x['stretch'],'positive_norm':bool(np.all(h>0)),
        'relative_energy_identity':float(defect),'normalized_max_energy_eigenvalue':float(eig[-1]/max(1,np.linalg.norm(W))),
        'passed':bool(np.all(h>0) and defect<=1e-10 and eig[-1]/max(1,np.linalg.norm(W))<=1e-10)})
for k in sorted(set(q['stretch'] for q in r['accuracy'])):
    x=sorted([q for q in r['accuracy'] if q['stretch']==k],key=lambda q:q['nodes'])
    assert len(x)==3
    orders={key:math.log2(x[-2][key]/x[-1][key]) for key in ['gradient','hessian','mixed']}
    r['orders'].append({'stretch':k,'fine_pair_order':orders,'passed':all(v>=1.5 for v in orders.values())})
r['passed']=all(x['D1_scaled']<=1e-10 and x['D2_scaled']<=1e-10 and x['min_norm']>0 for x in r['identity']) and all(x['constant']==0 for x in r['accuracy']) and all(x['passed'] for x in r['KO']+r['orders']) and max(x['omitted_H'] for x in r['accuracy'])>1e-3
r['scope']='Exact isolated research header: whole-radial SBP/compatibleD2 and volume-weighted variable-coefficient KO; smooth map tests include all active radial rows and face/edge/corner angular locations using analytic ghosts. Actual donor interpolation/native coupled energy not tested.'
r['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('tst/cubed_sphere/check_goal1_native_sbp.cpp'),Path('tst/cubed_sphere/goal1_radial_sbp.hpp')]}
r['payload_sha256']=hashlib.sha256((out/'operator-native.jsonl').read_bytes()).hexdigest()
(out/'operator-review.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'passed':r['passed'],'orders':r['orders'],'max_SBP_defect':max(x['D1_scaled'] for x in r['identity']),'max_compatible_D2_defect':max(x['D2_scaled'] for x in r['identity'])},indent=2))
