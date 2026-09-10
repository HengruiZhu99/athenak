import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,'/pscratch/sd/h/hzhu/corner-fix-tests-20260910');import bin_convert
root=Path('/pscratch/sd/h/hzhu/chi-truncation-amr-20260910/validation_symmetry');report=[]
for n in [128,512]:
 files=sorted((root/f'N{n}'/'bin/rank_00000000').glob('*state*.bin'))
 if not files:continue
 p=files[-1];b=bin_convert.read_binary(str(p));g=b['mb_geometry'];lookup={tuple(q):i for i,q in enumerate(g)};pairs=[]
 for i,q in enumerate(g):
  key=tuple([q[0],q[1],-q[3],-q[2],q[4],q[5]])
  if key in lookup:pairs.append((i,lookup[key]))
 result=dict(N=n,time=b['time'],source=str(p),blocks=len(g),paired_blocks=len(pairs),precision='float32 saved fields',scalar_errors={})
 for name in ['z4c_chi','z4c_alpha','z4c_Khat','z4c_Theta']:
  v=np.asarray(b['mb_data'][name],dtype=np.float64);err=max((np.max(abs(v[i]-v[j,:,::-1,:])) for i,j in pairs),default=float('nan'));scale=float(np.max(abs(v)));result['scalar_errors'][name]=dict(max_abs=float(err),max_abs_field=scale,relative_to_max=float(err/scale) if scale else float(err))
 report.append(result)
(root/'field-reflection.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
