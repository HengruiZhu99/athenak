"""Compare the implemented DO1 symbol with the independently compiled audit proposal."""
import argparse
import ctypes as C
import json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--implementation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
libs=[C.CDLL(x) for x in [a.audit,a.implementation]]
for lib in libs: lib.inner_gauge_symbol.argtypes=[C.c_int]+[C.c_double]*3+[fp,fp,fp]
rng=np.random.default_rng(3261);error=0.;count=0
for r in list(np.linspace(0,1,201))+list(rng.uniform(0,1,200)):
    A=rng.normal(size=(3,3));g=A@A.T+np.eye(3);g/=np.linalg.det(g)**(1/3)
    n=rng.normal(size=3);n/=np.linalg.norm(n);alpha=rng.uniform(.05,1.5);chi=rng.uniform(.001,2)
    matrices=[]
    for lib,mode in zip(libs,[2,1]):
        P=np.zeros((26,26));assert lib.inner_gauge_symbol(mode,r,alpha,chi,g,n,P)==0;matrices.append(P)
    error=max(error,float(np.max(np.abs(matrices[0]-matrices[1]))));count+=1
assert error<2e-10,error
result=dict(passed=True,states=count,max_matrix_difference=error,description='Independent audit candidate versus implemented evolution gauge; not a global hyperbolicity proof')
open(a.output,'w').write(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
