#!/usr/bin/env python3
"""Audit a collected CPU control before mixing architectures in a matrix."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scalar_evidence import snapshot

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--collected',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--case-name',default='backend-minkowski-n24-t2');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing backend audit is immutable')
 remote=json.loads((a.collected/'remote-manifest.json').read_text());collection=json.loads((a.collected/'collection-receipt.json').read_text());plan=json.loads((a.prepared/'cpu-plan.json').read_text());manifest=json.loads((a.prepared/'portable-source-manifest.json').read_text());result=a.collected/'result';run=result/a.case_name;receipt=json.loads((run/'receipt.json').read_text());params=json.loads((run/'parameters.json').read_text());reference=json.loads((a.reference/'parameters.json').read_text());frozen=json.loads((run/'frozen-integrity.json').read_text());project=Path(__file__).resolve().parents[2]
 source_match=remote['source_files']==manifest and not remote['source_mismatches'] and all(sha(project/'src/test_scalar'/name)==digest for name,digest in plan['source_sha256'].items())
 dependencies={name:sha(project/name) for name in ['src/athena.hpp','src/mesh/cubed_sphere/map.hpp']};dependency_match=all(sha(a.prepared/'source'/name)==digest for name,digest in dependencies.items());binary_digest=sha(result/'test_scalar');recorded_binary=(result/'executable.sha256').read_text().split()[0];preserved=plan.get('binary_sha256');scheduler=next(iter(remote['scheduler']['Jobs'].values()));build_binding=(binary_digest==preserved if preserved else (result/'configure.log').is_file() and (result/'build.log').is_file() and scheduler.get('Exit_status')==0);binary_match=binary_digest==recorded_binary==json.loads((run/'invocation.json').read_text())['binary'] and build_binding;mismatch=[name for name,digest in receipt['files'].items() if not (run/name).is_file() or sha(run/name)!=digest];parameter_match=all(params[k]==v for k,v in reference.items() if k!='tlim') and params['tlim']==2
 initial=snapshot(a.reference,0);scale=max(float(np.max(abs(initial[:,4:9]))),1e-12);checks=[]
 for t in [0,2]:
  x,y=snapshot(run,t),snapshot(a.reference,t);coordinate=float(np.max(abs(x[:,:4]-y[:,:4])/(1+abs(y[:,:4]))));delta=float(np.max(abs(x[:,4:]-y[:,4:]))/scale);checks.append(dict(time=t,coordinate_weight_relative=coordinate,joint_initial_scale=scale,scaled_max_field_derivative_difference=delta,passed=coordinate<1e-11 and delta<1e-9))
 passed=collection['passed'] and source_match and dependency_match and binary_match and not mismatch and parameter_match and receipt['returncode']==0 and (run/'complete.txt').exists() and frozen['unchanged'] and all(c['passed'] for c in checks)
 audit=dict(passed=passed,source_manifest_match=source_match,dependency_match=dependency_match,dependency_source_sha256=dependencies,validated_binary_sha256=sha(result/'test_scalar'),parameters=params,binary_match=binary_match,parameters_match=parameter_match,mismatches=mismatch,checks=checks,frozen=frozen,collection_sha256=sha(a.collected/'collection-receipt.json'),plan_sha256=sha(a.prepared/'cpu-plan.json'),scope='Same source and protected Cartesian/cubed-sphere dependencies; actual backend scalar fields/derivatives compared at0/2 before using the remote finite-cap control. All scientific three-grid and sensitivity gates remain required.',source_sha256=sha(Path(__file__)))
 a.output.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(dict(passed=passed,checks=checks),indent=2))
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
