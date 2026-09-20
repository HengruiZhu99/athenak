from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT.parent/'radial-volume'))
import radial_operator as volume
from boundary_operator import gauge_characteristic
p=np.array([2,3,4,5]);q=np.array([0,1,6,7]);out=[]
for r in (.2,.5,1.,2.,4.):
 a=r/(1+r);chi=a*a;b=r/(1+r)**2
 A2,A1,A0=volume.coefficients(r)
 P=np.block([[A1[np.ix_(p,p)],A2[np.ix_(p,q)]],[A0[np.ix_(q,p)],A1[np.ix_(q,q)]]])
 rows=[]
 for mode in ('lapse','shift'):
  for sign in (-1,1):
   # N=1 artificial derivative row 1 extracts characteristic d coefficients.
   # Separate value and derivative parts by using N=2 with D[0,1]=1.
   raw=gauge_characteristic(np.array([r,r]),np.array([[0.,1.],[0.,0.]]),0,mode,sign).reshape(8,2)
   L=np.r_[raw[p,0],raw[q,1]]
   speed=b+sign*(np.sqrt(2*a*chi) if mode=='lapse' else np.sqrt(8/3))
   err=np.linalg.norm(L@P-speed*L)/(np.linalg.norm(L)*(np.linalg.norm(P,2)+abs(speed)))
   rows.append({'mode':mode,'sign':sign,'expected_derivative_eigenvalue':speed,'relative_left_eigenrow_error':float(err)})
 for sign in (-1,1):
  L1=np.array([0,sign*a,0,chi/2,1,0,0,0]);L2=np.array([sign*4/(3*a),sign*2/(3*a),-sign*2/a,-1,0,1,0,0]);speed=b+sign*a*a
  for name,L in [('C1',L1),('C2',L2)]:
   err=np.linalg.norm(L@P-speed*L)/(np.linalg.norm(L)*(np.linalg.norm(P,2)+abs(speed)))
   rows.append({'mode':name,'sign':sign,'expected_derivative_eigenvalue':speed,'relative_left_eigenrow_error':float(err)})
 out.append({'r_M':r,'rows':rows,'principal_eigenvalues':np.linalg.eigvals(P).tolist()})
assert max(t['relative_left_eigenrow_error'] for row in out for t in row['rows'])<1e-12
(ROOT/'principal-crosscheck.json').write_text(json.dumps(out,indent=2)+'\n');print(max(t['relative_left_eigenrow_error'] for row in out for t in row['rows']))
