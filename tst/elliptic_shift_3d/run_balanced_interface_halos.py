#!/usr/bin/env python3
"""Bounded seven-query global free-background response probe; no adoption."""
import argparse,hashlib,json,os,resource,shutil,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(exist_ok=False);old=root/'docs/elliptic-shift-3d/evidence/transition-complete-balanced-20261027';capsule=out/'source';shutil.copytree(old/'source',capsule,ignore=shutil.ignore_patterns('__pycache__'));shutil.copyfile(root/'tst/elliptic_shift_3d/probe_balanced_interface_halos.cpp',capsule/'tst/elliptic_shift_3d/probe_balanced_interface_halos.cpp');shutil.copyfile(root/'tst/elliptic_shift_3d/mixed_interface_polynomial_jet.hpp',capsule/'tst/elliptic_shift_3d/mixed_interface_polynomial_jet.hpp');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={str(f.relative_to(capsule)):sha(f) for f in capsule.rglob('*') if f.is_file()};m={'scope':'Five-query balanced/positive-volume mixed Cartesian-shell P5 value halos, legacyacceptedplans first, no adoption','predeclared_budget':{'one_CPU_per_process':True,'maximum_seconds_per_command':180,'mesh_active_cells':47088,'query_points':[0,6,126,14657,17922],'initial_profile_MMS':True,'value_support_order':[125,175,256],'all_56_P5_columns':True,'value_L1_cap':16,'geometry_catalog':['legacy Euclidean first','Euclidean positivephysicalvolumes','quarter-shell balanced positivevolumes','half-shell balanced positivevolumes'],'new_balanced_maximum_distance_in_local_h':8,'composed_gradient_Hessian_caps':[32,64],'physical_caps_unchanged':[16,32,64],'condition':1e8,'reproduction':1e-10},'source_sha256':files,'commands':[]};resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
def run(name,command):
 row={'name':name,'argv':list(map(str,command)),'maximum_seconds':180};m['commands'].append(row);save();start=time.monotonic()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  process=subprocess.Popen(list(map(str,command)),stdout=so,stderr=se,start_new_session=True);row['owned_process_group']=process.pid;save()
  try:row['exit_code']=process.wait(timeout=180)
  except subprocess.TimeoutExpired:
   assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
   row['exit_code']=124;row['owned_process_reaped']=True
 row['seconds']=time.monotonic()-start;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True);assert row['exit_code']==0
command=[str(x).replace(str(old/'source'),str(capsule)).replace(str(old/'diagnostic.d'),str(out/'diagnostic.d')).replace(str(old/'diagnostic'),str(out/'diagnostic')).replace('qualify_transition_background_balanced.cpp','probe_balanced_interface_halos.cpp') for x in json.loads((old/'manifest.json').read_text())['commands'][0]['argv']];run('compile',command);m['binary_sha256']=sha(out/'diagnostic');save();initial=root/'docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat';exact=root/'docs/elliptic-shift-3d/evidence/transition-jet-localization-20261027/exact.dat';m['initial_sha256']=sha(initial);m['exact_sha256']=sha(exact);save();run('probe',[out/'diagnostic',initial,exact]);m['sources_unchanged']=all(sha(capsule/f)==h for f,h in files.items());assert m['sources_unchanged'];save()
