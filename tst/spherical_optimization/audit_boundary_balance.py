#!/usr/bin/env python3
"""Decompose the current scri D algebraic/differential mismatch on saved states."""
import argparse,json
from pathlib import Path
import numpy as np
from kernel import Kernel,FIELDS
from evolution import make_evolution,weights
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--runs',type=Path,nargs='+',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
 for folder in a.runs:
  cfg=json.loads((folder/'manifest.json').read_text())['config'];saved=np.load(folder/'last-accepted.npz');k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);state=saved['independent_state'];rate=e.rhs(state);b=e.last.copy();q=b['q'];d=b['d'];dd=b['dd']
  R,S,T,_=k.parts(e.r,q,d,dd,e.source);normal=k.analytic_normal(q[-1],d[-1],dd[-1]);raw=R[-1]-normal;pen=np.zeros(12)
  for f in [0,3,4,5,6,7]:pen[f]=cfg['penalty']/e.dr[-1]*(4*S[-1,f]-8*T[-1,f])
  adv=e.advection_jets(q);advection=q[-1,9]*(adv[-1,11]-adv[-1,7]);damping=-cfg.get('auxiliary_damping',1)*q[-1,11]
  assert abs(raw[7]+pen[7]+advection+damping-b['auxiliary_trace_pde'])<1e-10
  # Recover mu from the shared exact outer longitudinal speed; restoring=4
  # is the fixed parameter of all current laboratory configurations.
  mu=.75*k.gauge_speeds(1.,q[-1])[1]*q[-1,1];speed=np.sqrt(q[-1,0]/q[-1,1])
  geometry=-.5*speed*(raw[0]/q[-1,0]-raw[1]/q[-1,1])/mu
  penalty=-.5*speed*pen[0]/(mu*q[-1,0]);shift=(1-q[-1,9]*d[-1,9])/mu;restoration=-4*(d[-1,9]+1)/mu
  assert abs(geometry+penalty+shift+restoration-b['dependent_D'])<1e-10
  trace_rates=[]
  for eps0 in [1e-5,3e-6,1e-6]:
   eps=eps0/max(1.,np.max(abs(rate)));e.rhs(state+eps*rate);plus=e.last['dependent_D'];e.rhs(state-eps*rate);minus=e.last['dependent_D'];trace_rates.append((plus-minus)/(2*eps))
  contributions={}
  for j,f in enumerate(e.independent):
   direction=np.zeros_like(rate);direction[:,j]=rate[:,j];eps=3e-6/max(1.,np.max(abs(direction)))
   e.rhs(state+eps*direction);plus=e.last['dependent_D'];e.rhs(state-eps*direction);minus=e.last['dependent_D'];contributions[FIELDS[f]]=(plus-minus)/(2*eps)
  extrap=[]
  for width in [3,5,7]:
   x=e.r[-width-1:-1];values=q[-width-1:-1];value=weights(x,1.)@values;derivative=weights(x,1.,1)@values
   extrap.append(dict(width=width,D_value_mismatch=float(q[-1,11]-value[11]),D_value_mismatch_over_h=float((q[-1,11]-value[11])/e.dr[-1]),Lambda_value_mismatch=float(q[-1,7]-value[7]),alternative_auxiliary_rate=float(raw[7]+pen[7]+q[-1,9]*(derivative[11]-derivative[7])+damping)))
  row=dict(run=str(folder),n=cfg['n'],time_M=float(saved['t_M']),h=float(e.dr[-1]),D=float(q[-1,11]),Lambda=float(q[-1,7]),algebraic_D_terms=dict(raw_geometry=float(geometry),geometric_penalty=float(penalty),shift_advection=float(shift),restoration=float(restoration)),bulk_D_terms=dict(raw_Lambda=float(raw[7]),Lambda_penalty=float(pen[7]),D_minus_Lambda_advection=float(advection),D_damping=float(damping)),trace_time_derivative_probes=list(map(float,trace_rates)),trace_derivative_by_field=contributions,trace_derivative_sum_difference=float(sum(contributions.values())-trace_rates[1]),differential_defect=float(trace_rates[-1]-b['auxiliary_trace_pde']),clock=float(b['clock']),pole_max=float(max(abs(S[-1]))),regularity_trace_max=float(max(abs(T[-1]))),chi_ratio_residual=float(q[-1,0]/q[-1,1]-1),tail_extrapolations=extrap)
  rows.append(row);np.savez_compressed(a.output/(folder.name+'-tail.npz'),r=e.r[-16:],q=q[-16:],d=d[-16:],dd=dd[-16:]);k.close()
 (a.output/'balance.json').write_text(json.dumps(dict(runs=rows,qualification='Diagnostics of the existing discrete boundary extension only. Excluding endpoint values is an alternative derivative probe, not a changed evolution or a cure. No regularity/constraint reset.'),indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
