#!/usr/bin/env python3
"""Check actual compiled black-hole psi_t against the independent solution."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as leg
from scipy.interpolate import PchipInterpolator
from scalar_evidence import snapshot
from reference_interpolation import lagrange6
from validate_schwarzschild import norms

def reference_velocity(data,profile,kerr=False,sampler='pchip'):
 rho=np.linalg.norm(data[:,:3],axis=1)
 interpolate=lambda x,v,q:lagrange6(x,v,q) if sampler=='lagrange6' else PchipInterpolator(x,v,axis=0)(q)
 if not kerr:return data[:,0]*data[:,1]/(rho*rho)*interpolate(profile['rho'],profile['v'],rho)
 radius=np.unique(profile['rho']);z=np.unique(profile['z']);nr,na=len(radius),len(z)
 if len(profile)!=nr*na or not np.array_equal(profile['z'].reshape(nr,na),np.tile(z,(nr,1))):raise RuntimeError('Unexpected independent Kerr tensor grid')
 inv=np.linalg.inv(leg.legvander(z,na-1));values=(profile['v_real']+1j*profile['v_imag']).reshape(nr,na);result=np.empty(len(data))
 for start in range(0,len(data),65536):
  sl=slice(start,start+65536);r=rho[sl];n=data[sl,:3]/r[:,None];weights=leg.legvander(n[:,2],na-1)@inv;V=np.sum(weights*(interpolate(radius,values.real,r)+1j*interpolate(radius,values.imag,r)),axis=1);result[sl]=2*((n[:,0]+1j*n[:,1])**2*V).real
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs=3,type=Path,required=True);p.add_argument('--audits',nargs=3,type=Path,required=True);p.add_argument('--oracle-waveform',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-sampler',choices=['pchip','lagrange6'],default='pchip');a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing RHS review is immutable')
 parameters=[json.loads((d/'parameters.json').read_text()) for d in a.runs];ns=[r['n'] for r in parameters];spins=[r['a_over_M'] for r in parameters]
 if spins not in [[0,0,0],[.5,.5,.5]]:raise RuntimeError('Expected one declared scalar background')
 source={name:hashlib.sha256((a.source/name).read_bytes()).hexdigest() for name in ['CMakeLists.txt','main.cpp','geometry.hpp','sbp63.hpp']};integrity=[];records=[];checks=[];inputs=[]
 for run,audit,params in zip(a.runs,a.audits,parameters):
  m=json.loads((audit/'actual-rhs.json').read_text());g=json.loads((run/'frozen-integrity.json').read_text());keys=['physical_before','reference_before','physical_derivatives_before','reference_derivatives_before'];integrity.append(dict(run=str(run),passed=m['frozen_unchanged'] and g['unchanged'] and all(m[k]==g[k] for k in keys) and m['Pi_restore_max']<1e-12 and m['n']==params['n'] and m['spin']==params['a_over_M'] and m['excision']==params['excision'],Pi_restore_max=m['Pi_restore_max']))
 for t in [2,6,12,100]:
  profilefile=Path(str(a.oracle_waveform)+f'-profile-{t}.csv');profile=np.genfromtxt(profilefile,delimiter=',',names=True);inputs.append(profilefile);errors=[]
  for run,audit in zip(a.runs,a.audits):
   data=snapshot(run,t);ratefile=audit/f'actual-rhs-t{t:.3f}.f64';rate=np.fromfile(ratefile);inputs.append(ratefile)
   if len(rate)!=len(data):raise RuntimeError('Actual RHS shape differs from native snapshot')
   mask=np.linalg.norm(data[:,:3],axis=1)>=1.2;exact=reference_velocity(data[mask],profile,spins[0]!=0,a.reference_sampler);errors.append(dict(RHS=norms(rate[mask]-exact,data[mask,3]),reconstruction=norms(data[mask,5]-exact,data[mask,3])))
  records.append(dict(time=t,errors=errors))
  for field in ['RHS','reconstruction']:
   for norm in ['L2','Linf']:
    values=[e[field][norm] for e in errors];orders=[math.log(values[k]/values[k+1])/math.log(ns[k+1]/ns[k]) for k in [0,1]];checks.append(dict(time=t,field=field,norm=norm,errors=values,orders=orders,passed=values[0]>values[1]>values[2] and min(orders)>=1.5))
 result=dict(stage='schwarzschild' if spins[0]==0 else 'kerr',passed=all(c['passed'] for c in checks+integrity),resolutions=ns,runs=[str(r.resolve()) for r in a.runs],audits=[str(r.resolve()) for r in a.audits],delivery_source_sha256=source,records=records,checks=checks,integrity=integrity,input_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},scope='Actual compiled psi_t and centred-gradient time derivative reconstruction, common physical exterior, independent solution, both norms at2/6/12/100. No time stepping; exact native frozen cache identity and full t100 Pi restoration checked.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 result['reference_sampler']=a.reference_sampler;result['interpolator_sha256']=hashlib.sha256(Path(__file__).with_name('reference_interpolation.py').read_bytes()).hexdigest()
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],minimum_order=min(min(c['orders']) for c in checks)),indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
