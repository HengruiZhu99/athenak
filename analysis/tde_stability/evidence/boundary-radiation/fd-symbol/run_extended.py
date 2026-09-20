import json
from pathlib import Path
from fd_boundary import matrix,vector_matrix,summary
out=[]
for sector,builder,nfields in [('scalar',matrix,8),('vector',vector_matrix,4)]:
 for beta in [0.,.2]:
  for damping in [False,True]:
   for degree in [1,3]:
    for mode in ['zero_rate','radiation']:
     for inner in ([4] if mode=='zero_rate' else [2,4,'volume']):
      for tau in ([1.] if mode=='zero_rate' else [1.,.5,.25]):
       p=dict(n=32,h=.125,beta=beta,damping=damping,degree=degree,mode=mode,inner=inner,tau=tau)
       L,_=builder(**p);r=dict(sector=sector,**p,**summary(L,p['h'],nfields=nfields))
       out.append(r);print(json.dumps(r),flush=True)
Path('extended-results.json').write_text(json.dumps(out,indent=2)+'\n')
