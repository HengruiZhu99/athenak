#!/usr/bin/env python3
"""Changing-geometry check of outgoing auxiliary/shift-normal compatibility."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import make_evolution,ReferenceChiEvolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--outer-lapse-damping',type=float,default=1.);ap.add_argument('--penalty',type=float,default=.6510416666660156);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=dict(penalty=a.penalty,outer_lapse_damping=a.outer_lapse_damping,n=64,algebraic_chart=True,chi_reference_power=2,compact_end=.97,separated=True,normal_policy='analytic',reference_jets=True,reference_step=.000125,reference_tail='scri',outflow_boundary=True,boundary_normal_system='shift')
 k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);r=e.r;state=e.initial.copy()
 state[:,0]=.8+.1*r;state[:,1]=1+.01*r*r;state[:,2]=.03*r;state[:,3]=-3+.02*r*r;state[:,4]=.01*r
 state[:,5]=.05*r+.02*r*r;state[:,6]=.7+.1*r;state[:,7]=-.1*r;state[:,9]=state[:,5]+.3*r+.02*r*r-.1*r**3
 q=e.materialize(state);original_d,original_dd=ReferenceChiEvolution.spatial_jets(e,q);d,dd=e.spatial_jets(q)
 assert np.array_equal(original_d[:,8],d[:,8]) and np.array_equal(original_dd[:,8],dd[:,8])
 normal=k.analytic_normal(q[-1],d[-1],dd[-1]);rate,dep=k.scri(q[-1],d[-1],dd[-1],normal,e.dr[-1],e.penalty)
 # The ABI's analytic normal packs geometry/lapse only. Evaluate the shift
 # pole normal independently from three nearby shared-kernel pole values.
 nodes=np.ascontiguousarray(1.-1e-5*np.arange(3));dx=nodes-1
 near=q[-1]+dx[:,None]*d[-1]+.5*dx[:,None]**2*dd[-1]
 _,src,_=k.initial(nodes);_,poles,_,_=k.parts(nodes,near,np.tile(d[-1],(3,1)),np.tile(dd[-1],(3,1)),src)
 shift_normal=(3*poles[0,9]-4*poles[1,9]+poles[2,9])/(2e-5)
 R,_,_,_=k.parts(r,q,d,dd,e.source);raw_shift=R[-1,9]-shift_normal
 assert abs(raw_shift-rate[9])<1e-9 and abs(dep[0]-q[-1,11])<1e-9
 assert abs(q[-1,11]-q[-1,7]-.22)<1e-10
 assert np.array_equal(q[:,6],state[:,4]) and np.array_equal(q[:,7],state[:,5])
 null_rate=rate[0]/q[-1,1]-q[-1,0]*rate[1]/q[-1,1]**2-2*q[-1,9]*rate[9]
 assert abs(null_rate)<1e-10
 out=dict(penalty=a.penalty,outer_lapse_damping=a.outer_lapse_damping,lapse_jets_unchanged=True,clock_residual=float(dep[1]),dependent_D_residual=float(dep[0]-q[-1,11]),raw_minus_projected_shift_rate=float(raw_shift-rate[9]),null_time_derivative=float(null_rate),outgoing_trace=float(q[-1,11]-q[-1,7]),normal_corrections=e.boundary_derivative_correction.tolist(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='The clock residual is deliberately measured, not constrained by a second normal condition. Its convergence and differential auxiliary compatibility must be tested in evolved regular data. This arbitrary off-constraint fixture is not a regularity or stability proof.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));k.close()
if __name__=='__main__':main()
