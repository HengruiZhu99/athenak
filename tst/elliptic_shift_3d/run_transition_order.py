#!/usr/bin/env python3
"""Bounded sixth-order transition-only study; frozen v15 physical equations."""
import argparse,hashlib,json,os,resource,shlex,shutil,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(exist_ok=False);capsule=out/'source';previous=root/'docs/elliptic-shift-3d/evidence/transition-jet-localization-20261027';shutil.copytree(previous/'source',capsule,ignore=shutil.ignore_patterns('__pycache__'));shutil.copyfile(root/'tst/elliptic_shift_3d/diagnose_transition_order.cpp',capsule/'tst/elliptic_shift_3d/diagnose_transition_order.cpp');resource.setrlimit(resource.RLIMIT_CORE,(0,0));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={str(f.relative_to(capsule)):sha(f) for f in capsule.rglob('*') if f.is_file()}
cases=[('n6_amr3',6,3,root/'docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat'),('n6_static',6,0,root/'docs/elliptic-shift-3d/evidence/native-accuracy-controls-20261027/dt_half/initial.hidden-after-fresh-run.dat'),('n8_static',8,0,root/'docs/elliptic-shift-3d/evidence/native-accuracy-controls-20261027/n8/initial.hidden-after-fresh-run.dat'),('n10_static',10,0,root/'docs/elliptic-shift-3d/evidence/native-accuracy-controls-20261027/n10/initial.hidden-after-fresh-run.dat')]
m={'scope':'Standalone transition-only sixth-order chi/lapse derivative prototype, no production adoption/evolution/shift solve','physical_source_capsule_manifest_sha256':sha(previous/'manifest.json'),'source_sha256':files,'predeclared_budget':{'maximum_seconds_per_command':180,'one_CPU_per_process':True,'maximum_active_cells':50000,'cases':[{'name':k,'n':n,'refinement':ref,'cells':47088 if ref else 50*n**3} for k,n,ref,_ in cases]},'commands':[]}
def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
def run(name,command):
 row={'name':name,'argv':list(map(str,command)),'maximum_seconds':180};m['commands'].append(row);save();start=time.monotonic()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  process=subprocess.Popen(list(map(str,command)),cwd=root,stdout=so,stderr=se,start_new_session=True);row['owned_process_group']=process.pid;save()
  try:row['exit_code']=process.wait(timeout=180)
  except subprocess.TimeoutExpired:
   assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
   row['exit_code']=124;row['owned_process_reaped']=True
 row['seconds']=time.monotonic()-start;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True);assert row['exit_code']==0
old=json.loads((previous/'manifest.json').read_text());command=[str(v).replace(str(previous/'source'),str(capsule)).replace(str(previous/'diagnostic.d'),str(out/'diagnostic.d')).replace(str(previous/'diagnostic'),str(out/'diagnostic')).replace('diagnose_transition_jets.cpp','diagnose_transition_order.cpp') for v in old['commands'][0]['argv']];run('compile',command);binary=out/'diagnostic';m['binary_sha256']=sha(binary);save()
for name,n,ref,initial in cases:
 case=out/name;case.mkdir();m.setdefault('initial_sha256',{})[name]=sha(initial);save();run(name+'-catalog',[binary,'catalog',initial,case/'catalog.csv',n,ref]);run(name+'-analytic-oracle',[sys.executable,capsule/'tst/elliptic_shift_3d/wormhole_exact_jets.py',case/'catalog.csv',case/'exact.dat']);run(name+'-diagnose',[binary,'diagnose',initial,case/'exact.dat',n,ref,case/'diagnostic.csv'])
m['source_unchanged']=all(sha(capsule/f)==h for f,h in files.items());m['initials_unchanged']=all(sha(initial)==m['initial_sha256'][name] for name,_,_,initial in cases);assert m['source_unchanged'] and m['initials_unchanged'];save()
