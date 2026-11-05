#!/usr/bin/env python3
import argparse,ctypes as C,json
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import ChartEvolution
from physical_constraints import evaluate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--legacy-library',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);k=Kernel(a.library,separated=True);r0=k.config[2]
 old=C.CDLL(str(Path(a.legacy_library).resolve()));arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
 old.spherical_create.argtypes=[C.c_int,C.c_double,C.c_double,C.c_int];old.spherical_create.restype=C.c_void_p
 old.spherical_initial.argtypes=[C.c_void_p,C.c_int]+[arr]*4;old.spherical_initial.restype=C.c_int
 old.spherical_destroy.argtypes=[C.c_void_p];v=old.spherical_create(1,20.,2.,1)
 rows=[]
 for eps in (1e-4,1e-5,1e-6,1e-7):
  r=r0+eps*np.array([-2.,-1.,0.,1.,2.]);q,*_=k.initial(r);Q=np.zeros((5,12));src=np.zeros((5,6));omega=np.zeros((5,3))
  assert old.spherical_initial(v,5,r,Q,src,omega)==0
  rows.append(dict(epsilon=eps,new_value_gap=float(q[3,8]-q[1,8]),old_value_gap=float(Q[3,8]-Q[1,8]),
   new_old_lapse_difference=float(np.max(abs(q[:,8]-Q[:,8]))),new_slope_gap=float((-3*q[2,8]+4*q[3,8]-q[4,8])/(2*eps)-(3*q[2,8]-4*q[1,8]+q[0,8])/(2*eps))))
 assert abs(rows[-1]['new_value_gap'])<1e-7
 assert abs(rows[-1]['new_slope_gap'])<1e-6
 assert abs(rows[-1]['old_value_gap'])<1e-7
 assert max(v['new_old_lapse_difference'] for v in rows)<1e-13
 old.spherical_destroy(v)
 e=ChartEvolution(k,128,normal_policy='analytic',reference_jets=True)
 rate=e.rhs(e.initial);x=e.last;physical=evaluate(e.r,x['q'],x['d'],x['dd'],e.omega)
 convert=e.omega[:-1,0]*np.sqrt(x['q'][:-1,0]/x['q'][:-1,1]);kernel=x['diag'][:-1,:4].copy();kernel[:,[1,3]]*=convert[:,None]
 error=np.max(abs(physical-kernel),axis=0);assert max(error)<1e-9,error
 core=e.r<k.config[2];outer=(e.r>k.config[8])&(e.r<.995)
 core_rate=float(np.max(abs(rate[core])));assert core_rate>1
 result=dict(passed=True,lapse_join=rows,independent_physical_constraint_errors=error.tolist(),
  core_actual_rhs_max=core_rate,outer_actual_rhs_max=float(np.max(abs(rate[outer]))),
  reference_derivative_refinement_max=np.max(abs(e.reference_jet_difference),axis=1).tolist(),
  qualification='Initial representation and finite-interior formula checks; reference derivatives remain numerical and the core is not stationary.')
 (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));k.close()
if __name__=='__main__':main()
