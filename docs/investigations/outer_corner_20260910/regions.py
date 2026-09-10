from pathlib import Path
import sys,json,numpy as np
import bin_convert
root=Path(__file__).resolve().parent
for name in sys.argv[1:]:
 files=sorted((root/name/'bin/rank_00000000').glob('*curvature*.bin'))
 if not files:continue
 p=files[-1];d=bin_convert.read_binary(str(p));v=np.asarray(d['mb_data']['z4c_Kretschmann'])
 result=dict(case=name,file=str(p),time=d['time'],nonfinite=int((~np.isfinite(v)).sum()),regions={})
 for m,g in enumerate(d['mb_geometry']):
  x=np.linspace(g[0],g[1],v.shape[-1]);z=np.linspace(g[2],g[3],v.shape[-2]);xx,zz=np.meshgrid(x,z)
  masks={'whole':np.ones_like(xx,dtype=bool),'outer_band':(xx>=120)|(abs(zz)>=120),'corners':(xx>=120)&(abs(zz)>=120),'interior':(xx<120)&(abs(zz)<120)}
  for region,mask in masks.items():
   if not mask.any():continue
   w=np.where(mask,abs(v[m,0]),-np.inf);j,i=np.unravel_index(np.argmax(w),w.shape);value=float(w[j,i])
   if region not in result['regions'] or value>result['regions'][region]['maxAbsKret']:
    result['regions'][region]=dict(maxAbsKret=value,rho=float(x[i]),z=float(z[j]))
 print(json.dumps(result))
