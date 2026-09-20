"""Second-order differential Bjørhus boundary pilot; inspect convergence evidence."""
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy import linalg
from first_order import PREV,characteristics
sys.path.insert(0,str(PREV/'radial-full-boundary'))
from full_spectrum import build as old_build,inspect
from boundary_operator import gauge_characteristic

def build(n):
 _,_,raw,r,D,C,rows,labels,replaced=old_build(n);N=len(r);out=raw.copy()
 L,lam,lab=characteristics(r[0]);wp=np.array([2,3,4,5])*N
 win=np.array([gauge_characteristic(r,D,0,'lapse',1),gauge_characteristic(r,D,0,'shift',1)])
 delta=np.array([-win[0]@raw,-win[1]@raw,-lam[2]*rows[2]/(r[0]/(1+r[0])),lam[3]*rows[3]/(r[0]/(1+r[0]))**2])
 out[wp]+=np.linalg.solve(L[:4,:4],delta)
 # One incoming inner shift condition; correct Gamma only as the minimal p-only analogue.
 wi=gauge_characteristic(r,D,N-1,'shift',-1);idx=6*N-1;out[idx]-=wi@raw/wi[idx]
 return out,raw,r,D,C,rows,labels,replaced

def spectrum(n):
 out,raw,r,D,C,rows,labels,replaced=build(n)
 val,vec=linalg.eig(out);ii=np.argsort(val.real)[::-1]
 def mode(j,profiles=False):return inspect(val[j],vec[:,j],raw,r,D,C,rows,labels,replaced,'areal',profiles)
 return {'degree':n,'max_real':float(val.real.max()),'positive_count_1e-8':int(np.sum(val.real>1e-8)),'fastest':[mode(j) for j in ii[:5]],'near_constraint_reference':[mode(j,True) for j in np.argsort(abs(val+.072790693446))[:3]]}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--degree',type=int,default=32);p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=spectrum(a.degree);a.output.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
