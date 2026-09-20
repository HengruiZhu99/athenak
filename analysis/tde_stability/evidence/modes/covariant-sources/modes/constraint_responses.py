from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parents[1]/'mode-analysis';sys.path.insert(0,str(OLD))
from mode_operator import Operator,SHAPE,ACTIVE
op=Operator(str(ROOT/'physical_constraint_probes'),binary=OLD/'athena-constraint-probe',input_file=OLD/'input-constraints.athinput')
def dx(v,axis):
 out=np.zeros((16,16,16))
 for o,c in [(-3,-1/60),(3,1/60),(-2,3/20),(2,-3/20),(-1,-3/4),(1,3/4)]:
  sl=[slice(4,20)]*3;sl[2-axis]=slice(4+o,20+o);out+=c*v[tuple(sl)]/.25
 return out
def Q(v):
 g=[[1,2,3],[2,4,5],[3,5,6]];tr=v[1]+v[4]+v[6]
 return np.stack([v[14+i,4:20,4:20,4:20]-sum(dx(v[g[i][j]],j) for j in range(3))+.5*dx(tr,i) for i in range(3)])
records=[]
for item in json.loads((ROOT/'old-mode-responses.json').read_text()):
 case=item['case'];n=item['old_mode'];v=np.fromfile(ROOT/f'{case}-oldmode{n}-response.bin').reshape(SHAPE);cc=[]
 for sign in [1,-1]:
  label=f'{case}_mode{n}_sign{sign}';op.advance(sign*1e-3*v,0,label=label)
  cc.append(np.fromfile(op.directory/label/'output.bin.constraints.bin').reshape(7,24,24,24))
 c=(cc[0]-cc[1])/2e-3;ref=np.load(OLD/f'baseline-mode{n}-physical-constraints.npz')
 rec={'case':case,'old_mode':n,'interval_M':3,'constraints':{}}
 for name,a,b in [('H',ref['H'],c[1,4:20,4:20,4:20]),('M_cov',ref['M'],c[4:7,4:20,4:20,4:20]),('Q_contrav',ref['Q'],Q(v))]:
  gain=float(np.vdot(a,b)/np.vdot(a,a))
  rec['constraints'][name]={'norm_gain':float(np.linalg.norm(b)/np.linalg.norm(a)),'projected_gain':gain,'shape_change':float(np.linalg.norm(b-gain*a)/np.linalg.norm(b))}
 records.append(rec)
(ROOT/'old-mode-physical-constraint-responses.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
