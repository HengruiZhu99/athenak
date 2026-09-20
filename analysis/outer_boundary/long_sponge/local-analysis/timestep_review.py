"""Independently read existing timestep checkpoints; no evolution is launched."""
from pathlib import Path
import argparse,importlib.util,os,sys,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--regression',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.regression));os.environ['ATHENA_REGRESSION_PATH']=str(a.regression);out=Path(__file__).resolve().parent;study=out.parent
from z4c_background_restart import checkpoint,cohort
spec=importlib.util.spec_from_file_location('checker',study/'check_trumpet_checkpoint.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
raw=json.loads((study/'timestep/results.json').read_text());records={};states={};checks={}
for dt in [.6,1.6,3.2]:
 run=study/'timestep'/str(dt).replace('.','p');q=checkpoint(max((run/'rst/rank_00000000').glob('*.rst'),key=lambda p:checkpoint(p)['cycle']));assert abs(q['time']-640)<1e-8
 checks[str(dt)]=c.validate(run,1);assert checks[str(dt)]['passed']
 states[dt]=np.asarray(q['state'])[:,q['active']];records[str(dt)]={'time_M':q['time'],'cycle':q['cycle'],'checkpoint_dt':q['dt'],'input_sha256':hashlib.sha256((run/'input.athinput').read_bytes()).hexdigest()}
for dt in [1.6,3.2]:
 diff=states[dt]-states[.6];rel=float(np.linalg.norm(diff)/np.linalg.norm(states[.6]));maximum=float(np.max(abs(diff)));assert np.isclose(rel,raw[str(dt)]['active_L2_relative_difference'],rtol=1e-14);assert maximum==raw[str(dt)]['max_abs_difference'];records[str(dt)].update(active_residual_L2_relative_difference_vs_dt0p6=rel,max_abs_active_difference=maximum)
result={'scope':'Existing t640M amplitude1e-6 weak-field pulse, serial single16cubed block; actual discrete time comparison only. Raw mixed-variable L2 is not a physical energy or constraint norm. Reference dt.6 is not exact. checkpoint_dt is not necessarily the last shortened timestep. Long controls use amplitude1e-8 and eight8cubed blocks on four ranks.','cases':records,'relative_error_ratio_dt3p2_over_dt1p6':records['3.2']['active_residual_L2_relative_difference_vs_dt0p6']/records['1.6']['active_residual_L2_relative_difference_vs_dt0p6'],'checkpoint_validation':checks}
(out/'timestep-review.json').write_text(json.dumps(result,indent=2)+'\n');print({dt:{k:v for k,v in r.items()if k!='input_sha256'}for dt,r in records.items()})
