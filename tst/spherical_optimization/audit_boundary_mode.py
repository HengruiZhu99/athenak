#!/usr/bin/env python3
"""Attribute a saved rejected mode to Hermite gauge-jet feedback, without evolving."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import make_evolution,ReferenceChiEvolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--spectra',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
 for path in a.spectra:
  data=np.load(path/'linearization.npz');audit=json.loads((path/'audit.json').read_text());cfg=audit['run_configuration']
  assert cfg['outflow_boundary'];k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);state=data['state'];mode=data['leading_mode'];assert np.max(abs(mode.imag))<1e-12
  mode=mode.real;mode/=mode[-1,2]
  def terms(shift,policy):
   q=e.materialize(state+shift*mode);d,dd=e.spatial_jets(q);plain_d,plain_dd=ReferenceChiEvolution.spatial_jets(e,q)
   for f,name in [(8,'alpha'),(9,'beta')]:
    if policy not in ('both',name):d[:,f]=plain_d[:,f];dd[:,f]=plain_dd[:,f]
   R,S,T,_=k.parts(e.r,q,d,dd,e.source);normal=k.analytic_normal(q[-1],d[-1],dd[-1]);f=3
   result=np.array([R[-1,f],-normal[f],cfg['penalty']/e.dr[-1]*(4*S[-1,f]-8*T[-1,f])])
   return result,np.array([d[-1,8],dd[-1,8],d[-1,9],dd[-1,9]])
  probes=[]
  for epsilon in (1e-5,3e-6,1e-6):
   row=dict(epsilon=epsilon)
   for policy in ('both','alpha','beta','neither'):
    plus,jp=terms(epsilon,policy);minus,jm=terms(-epsilon,policy);response=(plus-minus)/(2*epsilon)
    row[policy]=dict(regular=float(response[0]),pole_normal=float(response[1]),penalty=float(response[2]),total=float(sum(response)),gauge_jet_response=((jp-jm)/(2*epsilon)).tolist())
   assert abs(row['both']['total']-audit['max_real_eigenvalue'])<1e-4*(1+abs(audit['max_real_eigenvalue']))
   probes.append(row)
  rows.append(dict(spectrum=str(path),n=cfg['n'],eigenvalue=audit['max_real_eigenvalue'],eigenmode_normalization='endpoint Ar=1',hermite_second_derivative_factor=float(e.hermite_dd[-1]*e.dr[-1]),probes=probes,input_sha256=hashlib.sha256((path/'linearization.npz').read_bytes()).hexdigest()));k.close()
 out=dict(rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),qualification='Directional derivative of endpoint Ar RHS along the saved rejected eigenmode. Removing individual Hermite jet corrections is diagnostic only and violates their clock/auxiliary compatibility; these mixed jets are not proposed evolutions.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
