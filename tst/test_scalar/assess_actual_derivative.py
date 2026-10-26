#!/usr/bin/env python3
"""Require three-grid convergence of the actual compiled scalar psi RHS."""
import argparse,csv,hashlib,json,math
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs=3,type=Path,required=True);p.add_argument('--audits',nargs=3,type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing derivative review is immutable')
 ns=[json.loads((r/'parameters.json').read_text())['n'] for r in a.runs];tables=[{(float(r['time']),r['region']):r for r in csv.DictReader((d/'actual-time-derivative.csv').open())} for d in a.audits];checks=[]
 for t in [2,6,12]:
  for region in ['all','bulk','shell']:
   for field in ['RHS','reconstruction']:
    for norm in ['L2','Linf']:
     values=[float(rows[t,region][field+'_'+norm]) for rows in tables];orders=[math.log(values[k]/values[k+1])/math.log(ns[k+1]/ns[k]) for k in [0,1]];checks.append(dict(time=t,region=region,field=field,norm=norm,errors=values,orders=orders,passed=values[0]>values[1]>values[2] and min(orders)>=1.5))
 restore=max(float(r['Pi_restore_max']) for rows in tables for r in rows.values());source={name:sha(a.source/name) for name in ['CMakeLists.txt','main.cpp','geometry.hpp','sbp63.hpp']};result=dict(passed=ns==[24,32,40] and all(c['passed'] for c in checks) and restore<1e-12,resolutions=ns,runs=[str(d.resolve()) for d in a.runs],audits=[str(d.resolve()) for d in a.audits],delivery_source_sha256=source,checks=checks,Pi_restore_max=restore,files={str(d/'actual-time-derivative.csv'):sha(d/'actual-time-derivative.csv') for d in a.audits},scope='Actual compiled upwind psi_t and centred-gradient velocity reconstruction both converge in all regions and both norms. Full t12 scalar checkpoint bounds the Pi reconstruction. Audit executables check all four frozen-cache fingerprints before returning.',source_sha256=sha(Path(__file__)))
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],minimum_order=min(min(c['orders']) for c in checks),Pi_restore_max=restore),indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
