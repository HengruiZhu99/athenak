"""Compare immutable native control histories at common times; no snapshot interpolation of fields."""
from pathlib import Path
import hashlib,json
import numpy as np
root=Path(__file__).resolve().parents[3];work=Path(__file__).resolve().parent
control=root/'build-layer-research/clean-wide-kappa10-long';candidate=work/'native-kappa10-t2'
cp=control/'finite-angular-N24/hyp.z4c.user.hst';npth=candidate/'finite-angular-N24/hyp.z4c.user.hst';old=np.atleast_2d(np.loadtxt(cp));new=np.atleast_2d(np.loadtxt(npth))
rows=[]
for time in [.1,.2,.25,.5,.75,1.,1.25,1.5,1.75,2.]:
 if min(old[-1,0],new[-1,0])+1e-14<time:continue
 a=[float(np.interp(time,old[:,0],old[:,i])) for i in [2,3,4]];b=[float(np.interp(time,new[:,0],new[:,i])) for i in [2,3,4]]
 rows.append({'time':time,'control_H_M_Z':a,'projected_H_M_Z':b,'projected_over_control':[b[i]/a[i] for i in range(3)]})
receipt={'scope':'Native d2 N24 wide .05-.95 a=.5 physical-P finite angular pulse .1/.02 width.5, kappa10 pole.03. Values linearly interpolate histories to common times; exact histories retained. Full field spatial budgets are handled separately.','control_receipt':str(control/'results.json'),'control_receipt_sha256':hashlib.sha256((control/'results.json').read_bytes()).hexdigest(),'control_history_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'candidate_history_sha256':hashlib.sha256(npth.read_bytes()).hexdigest(),'latest_candidate_time':float(new[-1,0]),'cases':rows}
(work/'history-comparison.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(rows,indent=2))
