#!/usr/bin/env python3
"""Bounded ACTUAL athena Driver/Task smoke and accepted native restart controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import resource
import subprocess
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--extended-controls',action='store_true');args=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);binary=root/'build-shift-3d/src/athena';input=root/'inputs/z4c/composite_minkowski.athinput'
 manifest=dict(scope='Actual Serial native vacuum Driver RK/Task integration with canonical global host coefficients and verified native physical core mirror; engine-owned shells are NOT native MeshBlocks; no AMR/device/MPI/wormhole claim',binary_sha256=sha(binary),input_sha256=sha(input),cases=[])
 def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 def run(name,arguments,expected_success=True):
  folder=out/name;folder.mkdir();command=[str(binary),*map(str,arguments),'-d',str(folder)];row=dict(name=name,command=command,maximum_seconds=180,expected_success=expected_success);manifest['cases'].append(row);save();start=time.monotonic()
  with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
   process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,start_new_session=True);row['owned_group']=process.pid
   try:row['exit_code']=process.wait(timeout=180)
   except subprocess.TimeoutExpired:
    row['exit_code']=124
    try:
     if os.getpgid(process.pid)!=process.pid:raise RuntimeError('refusing unowned group')
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=5)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
     row['cleanup']='owned group terminated'
    except Exception as error:row['cleanup_error']=str(error)
  row['seconds']=time.monotonic()-start;row['expectation_met']=(row['exit_code']==0)==expected_success;save();print(name,row['exit_code'],row['seconds'],flush=True)
  if not row['expectation_met']:raise RuntimeError('Unexpected native smoke result: '+name)
  return folder
 full=run('two-shell',['-i',input]);single=run('one-shell',['-i',input,'z4c/composite_radii=1'])
 midpoint=full/'rst/native_composite.00001.rst';assert midpoint.exists();restarted=run('restart',['-r',midpoint])
 compare={}
 for suffix in ['.csv','.scri.csv']:
  a=full/('native_composite.00002'+suffix);b=restarted/a.name;compare[suffix]=dict(identical=a.read_bytes()==b.read_bytes(),uninterrupted_sha256=sha(a),restart_sha256=sha(b))
 manifest['accepted_restart_physical_state_comparison']=compare;save();assert all(v['identical'] for v in compare.values())
 # Native driver must reject unsupported paths and corrupt/mismatched restart data.
 run('reject-adm-alias',['-i',input,'adm/separate_z4c_gauge=false'],False)
 run('reject-native-amr',['-i',input,'mesh_refinement/refinement=adaptive'],False)
 sidecar=Path(str(midpoint)+'.composite');bad_sidecar=out/'corrupt-sidecar.composite';data=bytearray(sidecar.read_bytes());data[len(data)//2]^=1;bad_sidecar.write_bytes(data)
 run('reject-sidecar-corruption',['-r',midpoint,'z4c/composite_restart_file='+str(bad_sidecar)],False)
 # With zoom/matter disabled, native restart payload ends in native B_z ghost.
 # Change ONLY this copy; a valid global sidecar must not silently erase the edit.
 bad_native=out/'edited-native-core.rst';data=bytearray(midpoint.read_bytes());data[-8:]=struct.pack('d',1e-6);bad_native.write_bytes(data)
 run('reject-native-mirror-edit',['-r',bad_native],False)
 run('reject-restart-rk-change',['-r',midpoint,'time/integrator=rk3'],False)
 if args.extended_controls:
  pulse_comparisons={}
  for name,angular in [('radial-pulse','false'),('angular-pulse','true')]:
   pulse=run(name,['-i',input,'z4c/composite_initial_lapse_amplitude=1e-5','z4c/composite_initial_angular='+angular]);mid=pulse/'rst/native_composite.00001.rst';again=run(name+'-restart',['-r',mid]);checks={}
   for suffix in ['.csv','.scri.csv']:
    a=pulse/('native_composite.00002'+suffix);b=again/a.name;checks[suffix]=dict(identical=a.read_bytes()==b.read_bytes(),before_sha256=sha(a),after_sha256=sha(b))
   pulse_comparisons[name]=checks;manifest['nonzero_pulse_restart_comparisons']=pulse_comparisons;save();assert all(v['identical'] for v in checks.values())
   run(name+'-reject-changed-amplitude',['-r',mid,'z4c/composite_initial_lapse_amplitude=2e-5'],False)
  run('extended-minkowski',['-i',input,'time/nlim=40','time/tlim=.02','z4c/composite_dt=.0005','z4c/composite_output_every=10','output1/dt=.005','output2/dt=.005','output3/dt=.005'])
 manifest['binary_unchanged']=sha(binary)==manifest['binary_sha256'];manifest['input_unchanged']=sha(input)==manifest['input_sha256'];save();print(json.dumps(dict(cases=len(manifest['cases']),restart_bitwise=all(v['identical'] for v in compare.values()),binary_unchanged=manifest['binary_unchanged'])),flush=True)


if __name__=='__main__':main()
