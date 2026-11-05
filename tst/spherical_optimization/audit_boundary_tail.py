#!/usr/bin/env python3
"""Compare smooth, logarithmic and fractional fits to raw near-scri residuals.

Finite-range fits are diagnostics, not proof of an asymptotic singularity.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[];inputs={};arrays={}
 for folder in a.runs:
  for name in ('last-accepted.npz','initial.npz'):
   p=folder/name;inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  final=np.load(folder/'last-accepted.npz');initial=np.load(folder/'initial.npz');r=final['r'];x=1-r;residual=final['q']-initial['q'];residual-=residual[-1];n=len(r)
  for outer in (.015,.03,.06):
   for exclude in (0,2):
    mask=(x>exclude*(r[-1]-r[-2]))&(x<=outer)
    if sum(mask)<10:continue
    z=x[mask]/outer
    for f,name in [(0,'chi'),(5,'Khat'),(8,'lapse'),(9,'shift')]:
     y=residual[mask,f]/x[mask]
     def solve(columns):
      design=np.column_stack(columns);coef=np.linalg.lstsq(design,y,rcond=None)[0];error=design@coef-y
      return float(np.linalg.norm(error)/np.sqrt(len(y))),coef,float(np.linalg.cond(design)),error
     def fractional(p):return solve([np.ones_like(z),z**p,z,z*z])
     fit=minimize_scalar(lambda p:fractional(p)[0],bounds=(.05,.95),method='bounded',options={'xatol':1e-10});models={}
     for label,answer in [('fractional',fractional(fit.x)),('logarithmic',solve([np.ones_like(z),z*np.log(z),z,z*z])),('polynomial',solve([np.ones_like(z),z,z*z,z**3]))]:
      rms,coef,cond,error=answer;models[label]=dict(rms=rms,relative_to_variation=rms/max(np.ptp(y),1e-300),coefficients=coef.tolist(),condition=cond)
      tag=f'n{n}-outer{outer}-exclude{exclude}-{name}-{label}';arrays[tag+'-x']=x[mask];arrays[tag+'-error']=error
     rows.append(dict(n=n,outer=outer,excluded_cells=exclude,field=name,points=int(sum(mask)),fractional_ratio_power=float(fit.x),implied_field_power=float(1+fit.x),models=models))
 out=dict(rows=rows,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Fits to endpoint-subtracted (field-initial field)/(1-r). Fractional model a+b*x^p+c*x+d*x² has an extra fitted exponent; polynomial and x*log(x) models have four linear coefficients. Finite resolution, endpoint error and finite fit windows prevent proving differentiability or an asymptotic exponent.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(a.output/'fit_residuals.npz',**arrays)
 print(json.dumps([dict(n=r['n'],field=r['field'],outer=r['outer'],p=r['fractional_ratio_power'],fractional=r['models']['fractional']['rms'],logarithmic=r['models']['logarithmic']['rms'],polynomial=r['models']['polynomial']['rms']) for r in rows if r['field']=='lapse' and r['excluded_cells']==0]))
if __name__=='__main__':main()
