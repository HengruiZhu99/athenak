#!/usr/bin/env python3
"""Bounded matched SMR evolution using the existing initial transfer inputs."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True);runs=[]
for n in [8,16,32]:
 for mode in ['none','residual_shifted']:
  for half in ([False,True] if n==32 else [False]):
   dt=.0000625 if half else .000125;steps=320 if half else 160;cadence=80 if half else 40
   case=f'n{n}-{mode}'+('-halfdt' if half else '');d=a.output/case;d.mkdir()
   text=(a.fixtures/f'n{n}'/f'fd6-2d-{mode}'/'used.athinput').read_text().replace('nlim = 0',f'nlim = {steps}').replace('research_dt_ceiling = 0.001',f'research_dt_ceiling = {dt}').replace('intrinsic_diagnostics = false','intrinsic_diagnostics = true').replace('intrinsic_diagnostic_dcycle = 5',f'intrinsic_diagnostic_dcycle = {cadence}')
   (d/'used.athinput').write_text(text);command=[str(a.binary.resolve()),'-i','used.athinput'];start=time.monotonic()
   with (d/'run.log').open('w') as log:r=subprocess.run(command,cwd=d,stdout=log,stderr=subprocess.STDOUT,timeout=180)
   row=dict(case=case,command=command,returncode=r.returncode,seconds=time.monotonic()-start,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),input_sha256=hashlib.sha256((d/'used.athinput').read_bytes()).hexdigest());runs.append(row);(a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');print(json.dumps(row),flush=True)
   if r.returncode:raise RuntimeError('Preserve failed state; stop campaign: '+case)
