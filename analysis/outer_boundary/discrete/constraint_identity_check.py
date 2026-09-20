"""Verify constraint propagation identities against independent full20 symbol."""
import sys,json
from pathlib import Path
import numpy as np
import production_halfspace as p

out=[]
for cfg in [p.Config(),p.Config(alpha=.999,chi=.998001,beta_n=.00032),p.Config(alpha=.7,chi=.49,beta_n=.2,G=2.)]:
 for s,k in [(.07+.02j,.1),(.31-.2j,.4)]:
  V,Q=p.volume(s,k,cfg);D=np.array([s,1j*k,0]);a,c=cfg.alpha,cfg.chi;sig=a*cfg.kappa
  D0=V-cfg.beta_n*s*np.eye(20);I=np.eye(20)
  h=np.array([[I[1],I[3],I[4]],[I[3],I[2],I[5]],[I[4],I[5],-I[1]-I[2]]])
  A=np.array([[I[8],I[10],I[11]],[I[10],I[9],I[12]],[I[11],I[12],-I[8]-I[9]]])
  H=c*np.einsum('i,j,ijm->m',D,D,h)+2*(D@D)*I[0]
  M=np.einsum('j,ijm->im',D,A)-2/3*D[:,None]*(I[6]+2*I[7])
  th=I[7];divQ=D@Q;divM=D@M;lap=D@D
  errors={
   'H':np.max(abs(H@D0+2*a*c*divM)),
   'M':np.max(abs(M@D0-(-a/2*D[:,None]*H+a*c/2*(lap*Q-D[:,None]*divQ)+2*sig*D[:,None]*th))),
   'Theta':np.max(abs(th@D0-(a/2*H+a*c/2*divQ-2*sig*th))),
   'Q':np.max(abs(Q@D0-(2*a*M+2*a*D[:,None]*th-2*sig*Q))),
   'Q_wave':np.max(abs(Q@D0@D0-a*a*c*lap*Q+2*sig*Q@D0)),
   'Theta_wave':np.max(abs(th@D0@D0-a*a*c*lap*th+2*sig*th@D0+a*c*sig*divQ))}
  out.append(dict(config=p.asdict(cfg),s=[s.real,s.imag],k=k,errors={n:float(e) for n,e in errors.items()}))
assert max(e for row in out for e in row['errors'].values())<1e-12
Path('constraint-identities.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS',max(e for row in out for e in row['errors'].values()))
