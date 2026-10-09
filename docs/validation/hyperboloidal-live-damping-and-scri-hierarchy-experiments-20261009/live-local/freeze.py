from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent;repo=p.parents[2];dest=p/'immutable-live-damping-local-20261009'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert not dest.exists();r=json.loads((p/'receipt.json').read_text());assert r['passed_finite_Omega_local_numerical_gate'] and r['source_count']==384 and r['command_count']==17
assert all(sha(repo/k)==v for k,v in r['source_before'].items())
for x in r['commands']:
 assert x['returncode']==0 and not x['stderr']
 if 'stdout_file' in x:assert sha(p/x['stdout_file'])==x['stdout_sha256']
dest.mkdir()
names=['live_damping_profile.hpp','profile_helpers.hpp','subsidiary_c0.hpp','subsidiary_profile.hpp','constraint_tangent.cpp','full20_profile.cpp','tensor_gate.cpp','principal_profile.cpp','gradient_gate.cpp','kernel_symbol_copy.cpp','full20_RK_all_a.cpp','run_audit.py','supplement.py','check_live.py','finalize.py','freeze.py','DERIVATION.md','REPORT.md','compile-receipt.json','supplement-receipt.json','receipt.json','leading-pole.json','tensor-gate.json','tensor-gate-debug.json','gradient-gate.json','principal.log','check-report.json','check.log','check.stderr','check-final.log','run.log','finalize.log','finalize.stderr','commands-in-progress.json']
for name in names:shutil.copy2(p/name,dest/name)
shutil.copy2(repo/'build-layer-research/spatial-norm-native-controls/N36-t0.2/finite-angular-long-N36/layer.athinput',dest/'authoritative-native-input.athinput')
shutil.copy2(p.parent/'live-damping-assessment/immutable-live-math-20261009/index.json',dest/'separate-math-assessment-index.json')
external={}
for name in ['full20.json','full20-RK-all-a.json','constraint-profile.json','constraint-base.json','constraint-base-rerun.json']+list(r['binary_sha256']):external[name]={'sha256':sha(p/name),'bytes':(p/name).stat().st_size,'original_repo_path':str((p/name).relative_to(repo))}
index={'scope':'Frozen live C0 V(.15,.3) finite-positive-Omega local actual-kernel gate only; no native/global/energy/nonlinear-bound/scri/BH acceptance. Original compile and supplemental receipts preserved.','files':{str(f.relative_to(dest)):sha(f) for f in sorted(dest.rglob('*')) if f.is_file()},'large_outputs_outside_snapshot':external,'source_count':384,'production_src_and_root_CMake_count':365,'commands':17,'gate_receipt_sha256':sha(dest/'receipt.json'),'helper_sha256':sha(dest/'live_damping_profile.hpp'),'runtime_implementation':r['runtime_implementation'],'compiled_launch_head':r['compiled_launch_head'],'finalize_head':r['finalize_head'],'all_frozen_hashes_verified':True}
(dest/'index.json').write_text(json.dumps(index,indent=2)+'\n')
assert all(sha(dest/f)==h for f,h in index['files'].items())
print('FROZEN',str(dest.relative_to(repo)),len(index['files']),'files',len(external),'large/binary records')
print('index',sha(dest/'index.json'));print('receipt',sha(dest/'receipt.json'));print('helper',sha(dest/'live_damping_profile.hpp'))
