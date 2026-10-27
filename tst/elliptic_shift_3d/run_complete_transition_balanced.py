#!/usr/bin/env python3
"""Preflight then bounded face batches; never proceed past unsupported geometry."""
import argparse,hashlib,json,os,resource,shutil,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(exist_ok=False);old=root/'docs/elliptic-shift-3d/evidence/transition-background-orthogonal-20261027';capsule=out/'source';shutil.copytree(old/'source',capsule,ignore=shutil.ignore_patterns('__pycache__'))
for name in ['qualify_transition_background.cpp','qualify_transition_background_balanced.cpp','wormhole_background_jets.py']:shutil.copyfile(root/'tst/elliptic_shift_3d'/name,capsule/'tst/elliptic_shift_3d'/name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files={str(f.relative_to(capsule)):sha(f) for f in capsule.rglob('*') if f.is_file()};m={'scope':'Complete i0/i1 cube-to-sphere interface qualification, selected M1/3/Minkowski profile; no production/evolution adoption','predeclared_budget':{'one_CPU_per_process':True,'maximum_seconds_per_command':180,'active_cells':47088,'interface_queries':432,'scalar_field_plans':864,'preflight_queries':36,'donor_support_order':[125,175,256],'all_donors':'global active Cartesian AND shell cells','unchanged_physical_caps':[16,32,64],'condition_gate':1e8,'reproduction_gate':1e-10,'coefficient_normalization':'normalized known-background residual; nuisance coefficient recorded separately, not a raw-scale coverage gate','unknown_noise_amplitude':1e-12,'B_nodal_noise_amplitude':1e-12,'B_scaled_query_jet_noise_amplitude':1e-12,'uncertainty_source_change_gate':1e-7,'uncertainty_scaled_jet_change_gate':1e-8,'exact_basis_mode_source_error_gate':1e-7,'stop_on_preflight_failure':True,'explicit_geometry_catalog':[{'metric':'Euclidean physical distance','support_order':[125,175,256]},{'metric':'target-Jacobian logical-cell distance','support_order':[125,175,256]}],'catalog_note':'Only after every Euclidean support fails unchanged guards, try Jacobian supports. No lower-degree or source/value fallback; all attempts logged.'},'source_sha256':files,'commands':[]};resource.setrlimit(resource.RLIMIT_CORE,(0,0))
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
 row['seconds']=time.monotonic()-start;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True);return row['exit_code']
previous=root/'docs/elliptic-shift-3d/evidence/transition-jet-localization-20261027';compile_reference=json.loads((old/'manifest.json').read_text())['commands'][0]['argv'];command=[str(v).replace(str(old/'source'),str(capsule)).replace(str(old/'diagnostic.d'),str(out/'diagnostic.d')).replace(str(old/'diagnostic'),str(out/'diagnostic')).replace('diagnose_background_orthogonal.cpp','qualify_transition_background_balanced.cpp') for v in compile_reference];assert run('compile',command)==0
background=out/'background192.dat';assert run('B192-independent-oracle',[sys.executable,capsule/'tst/elliptic_shift_3d/wormhole_background_jets.py',previous/'catalog.csv',background,'--quadrature','192'])==0
initial=root/'docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat';exact=previous/'exact.dat';m['initial_unknown96_sha256']=sha(initial);m['B96_exact_oracle_sha256']=sha(exact);m['B192_oracle_sha256']=sha(background);m['binary_sha256']=sha(out/'diagnostic');save()
code=run('preflight',[out/'diagnostic',initial,exact,background,'preflight','-1',out/'preflight.supports']);rows=[json.loads(s) for s in (out/'preflight.stdout').read_text().splitlines() if s.startswith('{')];m['preflight_summary']=next((r for r in rows if r['phase']=='summary'),None);save()
if code!=0:m['status']='stopped_at_preflight_failure';save();raise SystemExit(2)
m['face_summaries']={}
for face in range(6):
 code=run('face'+str(face),[out/'diagnostic',initial,exact,background,'face',str(face),out/('face'+str(face)+'.supports')]);rows=[json.loads(s) for s in (out/('face'+str(face)+'.stdout')).read_text().splitlines() if s.startswith('{')];m['face_summaries'][str(face)]=next((r for r in rows if r['phase']=='summary'),None);save()
 if code!=0:m['status']='stopped_at_face_failure';save();raise SystemExit(2)
m['complete_interface_coverage']=sum(r['selected_field_plans'] for r in m['face_summaries'].values())==864 and all(r['passed'] for r in m['face_summaries'].values());assert m['complete_interface_coverage'];m['status']='complete_interface_qualified';m['source_unchanged']=all(sha(capsule/f)==h for f,h in files.items());m['initial_unknown96_unchanged']=sha(initial)==m['initial_unknown96_sha256'];assert m['source_unchanged'] and m['initial_unknown96_unchanged'];save()
