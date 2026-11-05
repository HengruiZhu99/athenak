#!/usr/bin/env python3
"""Attribute the last accepted innermost chi rate without rescuing the state."""
import argparse,json
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import ChartEvolution,puncture_dissipation

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((a.run/'manifest.json').read_text())['config'];saved=np.load(a.run/'last-accepted.npz');k=Kernel.from_config(a.library,cfg)
 e=ChartEvolution(k,cfg['n'],cfg['grading'],cfg['penalty'],cfg['ko'],cfg['floors'],cfg['normal_policy'],cfg['reference_jets'],cfg['reference_step'],cfg['reference_tail'],cfg['grid_map'])
 state=saved['independent_state'];rate=e.rhs(state);base=e.last;q=base['q'];d,dd=e.spatial_jets(q);R,S,_,_=k.parts(e.r,q,d,dd,e.source)
 continuum=R[0,0]+S[0,0]/e.omega[0,0];adv=e.advection_jets(q);advective=q[0,9]*(adv[0,0]-d[0,0]);ko=puncture_dissipation(q,e.r,e.h,e.ko,.8,order=8,include_first=True)[0,0]
 assert abs(rate[0,0]-continuum-advective-ko)<1e-12
 out=dict(time_M=float(saved['t_M']),r0=float(e.r[0]),chi=float(q[0,0]),chi_rate=float(rate[0,0]),centered_continuum_rate=float(continuum),biased_advection_correction=float(advective),KO_rate=float(ko),beta=float(q[0,9]),chi_prime_centered=float(d[0,0]),chi_prime_biased=float(adv[0,0]),linear_zero_time_increment_M=float(-3*q[0,0]/rate[0,0]),qualification='Last accepted discrete rate decomposition only. No state floor, constraint reset, or causal attribution beyond the saved terms.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(a.output/'inner.npz',r=e.r[:12],q=q[:12],d=d[:12],dd=dd[:12],rhs=rate[:12]);print(json.dumps(out));k.close()
if __name__=='__main__':main()
