import json,sys,hashlib
from pathlib import Path
import mpmath as mp
p=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(p))
import nonlinear_oracle as n
mp.mp.dps=80
plan=json.loads((p/'plan.json').read_text());d=next(v for v in plan['points'] if v['name']=='core')
point=[n.exact(float.fromhex(d['time_hex']))]+[n.exact(float.fromhex(x)) for x in d['cartesian_hex']]
class CoreHeight:
 def value(self,r):return mp.mpf(0)
y,y1,y2,y3,o=n.embedding(point,CoreHeight());x,x1,x2,x3,iv=n.inverse_map(y,y1,y2,y3,mp.mpf('.025'));g=n.metric(x1,x2,x3)
gamma=[row[1:] for row in g[1:]];det=n.det3(gamma);chi=det**(-mp.mpf(1)/3);gt=[[chi*v for v in row] for row in gamma]
report={'D':str(iv['determinant']),'gamma':[[str(v.v) for v in row] for row in gamma],'det_gamma_jet':str(det.v),'det_gamma_direct':str(mp.det(mp.matrix([[v.v for v in row] for row in gamma]))),'chi':str(chi.v),'expected_chi':str(det.v**(-mp.mpf(1)/3)),'gt':[[str(v.v) for v in row] for row in gt],'det_gt_jet':str(n.det3(gt).v),'det_gt_direct':str(mp.det(mp.matrix([[v.v for v in row] for row in gt]))),'source_sha':hashlib.sha256((p/'nonlinear_oracle.py').read_bytes()).hexdigest(),'algebra_sha':hashlib.sha256((p/'jet_algebra.py').read_bytes()).hexdigest()}
print(json.dumps(report,indent=2))
