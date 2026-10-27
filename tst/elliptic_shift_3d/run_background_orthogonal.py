#!/usr/bin/env python3
"""Bounded fixed/free initial-background derivative falsifier using frozen v15 headers/data."""
import argparse,hashlib,json,os,resource,shutil,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(exist_ok=False);previous=root/'docs/elliptic-shift-3d/evidence/transition-jet-localization-20261027';capsule=out/'source';shutil.copytree(previous/'source',capsule,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['diagnose_transition_p5.cpp','transition_polynomial_jet.hpp','diagnose_transition_background.cpp','transition_background_jet.hpp','diagnose_background_normalization.cpp','transition_background_projection.hpp','transition_orthogonal_background_jet.hpp','diagnose_background_orthogonal.cpp']:shutil.copyfile(root/'tst/elliptic_shift_3d'/name,capsule/'tst/elliptic_shift_3d'/name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={str(f.relative_to(capsule)):sha(f) for f in capsule.rglob('*') if f.is_file()};m={'scope':'Independent handful-query P5-orthogonal background residual probe; full physical RHS retained, no evolution or production adoption','source_sha256':files,'predeclared_budget':{'maximum_seconds_per_command':180,'one_CPU_per_process':True,'queries':4,'donor_supports':[125,175,256],'field_specific_plans':24,'basis_preconditioning':'weighted P5 projection and normalized residual; identical span','normalization_physical_weight_tolerance':1e-9,'guards':{'condition':1e8,'reproduction':1e-10,'physical_functional_l1':[16,32,64],'background_coefficient_l1':10000}},'commands':[]};resource.setrlimit(resource.RLIMIT_CORE,(0,0))
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
old=json.loads((previous/'manifest.json').read_text());command=[str(v).replace(str(previous/'source'),str(capsule)).replace(str(previous/'diagnostic.d'),str(out/'diagnostic.d')).replace(str(previous/'diagnostic'),str(out/'diagnostic')).replace('diagnose_transition_jets.cpp','diagnose_background_orthogonal.cpp') for v in old['commands'][0]['argv']];run('compile',command);initial=root/'docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat';exact=previous/'exact.dat';m['binary_sha256']=sha(out/'diagnostic');m['initial_sha256']=sha(initial);m['exact_sha256']=sha(exact);save();run('probe',[out/'diagnostic',initial,exact]);m['sources_unchanged']=all(sha(capsule/f)==h for f,h in files.items());m['exact_unchanged']=sha(exact)==m['exact_sha256'];assert m['sources_unchanged'] and m['exact_unchanged'];save()
