#!/usr/bin/env python3
"""Default parity and nonstationary compatibility for the alternative lift."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import make_evolution,ReferenceChiEvolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--before',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 base=json.loads((a.before/'configuration.json').read_text());old=np.load(a.before/'before.npz');parity={}
 for name,outflow in [('standard',False),('outflow',True)]:
  cfg=dict(base,outflow_boundary=outflow);k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);rhs=e.rhs(old[name+'_q']);assert np.array_equal(rhs,old[name+'_rhs']);parity[name]=True;k.close()
 cfg=dict(base,outflow_boundary=True,lapse_normal_lift='second-neutral');k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);r=e.r;state=e.initial.copy()
 state[:,0]=.8+.1*r;state[:,1]=1+.01*r*r;state[:,2]=.03*r;state[:,3]=-3+.02*r*r;state[:,4]=.01*r
 state[:,5]=.05*r+.02*r*r;state[:,6]=.7+.1*r;state[:,7]=-.1*r;state[:,9]=state[:,5]+.3*r+.02*r*r-.1*r**3
 q=e.materialize(state);original_d,original_dd=ReferenceChiEvolution.spatial_jets(e,q);d,dd=e.spatial_jets(q)
 normal=k.analytic_normal(q[-1],d[-1],dd[-1]);_,dep=k.scri(q[-1],d[-1],dd[-1],normal,e.dr[-1],e.penalty)
 assert abs(dep[1])<1e-9 and abs(dep[0]-q[-1,11])<1e-9
 second=float(dd[-1,8]-original_dd[-1,8]);assert abs(second)<1e-9
 assert abs(d[-1,8]-original_d[-1,8])>1 and np.array_equal(q[:,6],state[:,4]) and np.array_equal(q[:,7],state[:,5])
 z=(r[-7:]-1)/e.dr[-1];P=np.polynomial.Polynomial.fromroots(z);P/=P.deriv()(0);L=P*np.polynomial.Polynomial([1.,-.5*P.deriv(2)(0)])
 assert max(abs(L(z)))<1e-8 and abs(L.deriv()(0)-1)<1e-12 and abs(L.deriv(2)(0))<1e-12
 e.rhs(state);assert abs(e.last['clock'])<1e-9
 out=dict(default_RHS_bitwise_equal=parity,clock=float(dep[1]),dependent_D_residual=float(dep[0]-q[-1,11]),endpoint_second_derivative_change=second,endpoint_first_derivative_change=float(d[-1,8]-original_d[-1,8]),normal_corrections=e.boundary_derivative_correction.tolist(),lift_nodal_residual=float(max(abs(L(z)))),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Polynomial lifting and algebraic nonstationary compatibility only. Full operator and evolution stability remain separate gates.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));k.close()
if __name__=='__main__':main()
