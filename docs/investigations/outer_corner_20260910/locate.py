import numpy as np,json,sys
from pathlib import Path
import bin_convert
for p in sorted((Path(sys.argv[1] if len(sys.argv)>1 else 'bjorhus90_110')/'bin/rank_00000000').glob('*curvature*.bin')):
 try:d=bin_convert.read_binary(str(p))
 except Exception:continue
 v=np.asarray(d['mb_data']['z4c_Kretschmann']);m,k,j,i=map(int,np.unravel_index(np.nanargmax(abs(v)),v.shape));g=d['mb_geometry'][m]
 print(json.dumps(dict(time=d['time'],maxAbsKret=float(abs(v[m,k,j,i])),rho=float(g[0]+i*(g[1]-g[0])/(v.shape[-1]-1)),z=float(g[2]+j*(g[3]-g[2])/(v.shape[-2]-1)))))
