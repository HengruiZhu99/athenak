#!/usr/bin/env python3
"""Bounded seven-query global free-background response probe; no adoption."""
import argparse,hashlib,json,os,resource,shutil,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(exist_ok=False);old=root/'docs/elliptic-shift-3d/evidence/transition-complete-balanced-20261027';capsule=out/'source';shutil.copytree(old/'source',capsule,ignore=shutil.ignore_patterns('__pycache__'));shutil.copyfile(root/'tst/elliptic_shift_3d/evaluate_global_background_stage.cpp',capsule/'tst/elliptic_shift_3d/evaluate_global_background_stage.cpp');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={str(f.relative_to(capsule)):sha(f) for f in capsule.rglob('*') if f.is_file()};m={'scope':'One actual original-vector stage solve with global scalar corrections; oldv15FD; no production adoption','predeclared_budget':{'one_CPU_per_process':True,'maximum_seconds_per_command':180,'all_active_cells':47088,'scalar_jet_overrides':94176,'all_global_coefficient_donors':47088,'physical_caps_unchanged':[16,32,64],'condition':1e8,'reproduction':1e-10},'source_sha256':files,'commands':[]};resource.setrlimit(resource.RLIMIT_CORE,(0,0))
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
command=[str(x).replace(str(old/'source'),str(capsule)).replace(str(old/'diagnostic.d'),str(out/'diagnostic.d')).replace(str(old/'diagnostic'),str(out/'diagnostic')).replace('qualify_transition_background_balanced.cpp','evaluate_global_background_stage.cpp') for x in json.loads((old/'manifest.json').read_text())['commands'][0]['argv']];run('compile',command);initial=root/'docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat';exact=root/'docs/elliptic-shift-3d/evidence/transition-jet-localization-20261027/exact.dat';m['initial_sha256']=sha(initial);m['exact_sha256']=sha(exact);m['binary_sha256']=sha(out/'diagnostic');functional=root/'docs/elliptic-shift-3d/evidence/global-background-scalar-proof-r2-20261027/functionals.dat';m['functional_sha256']=sha(functional);save();run('probe',[out/'diagnostic',initial,exact,functional,out/'probe.csv']);m['sources_unchanged']=all(sha(capsule/f)==h for f,h in files.items());m['initial_unchanged']=sha(initial)==m['initial_sha256'];rows=[json.loads(line) for line in (out/'probe.stdout').read_text().splitlines() if line.startswith('{')];m['norms']=next(row for row in rows if row.get('phase')=='full_initial_norms');m['shift_solve']=next(row for row in rows if row.get('phase')=='actual_shift_solve');assert m['sources_unchanged'] and m['initial_unchanged'];save()
