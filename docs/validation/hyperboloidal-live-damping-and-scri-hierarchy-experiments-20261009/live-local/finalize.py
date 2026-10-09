from pathlib import Path
import json,hashlib,subprocess,sys,time
p=Path(__file__).resolve().parent;repo=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
base=json.loads((p/'compile-receipt.json').read_text());supp=json.loads((p/'supplement-receipt.json').read_text())
assert base['passed_compiled_commands'] and supp['passed_supplement_commands']
files=[repo/k for k in supp['source_before']]+[p/'check_live.py',p/'finalize.py',repo/'build-layer-research/spatial-norm-native-controls/N36-t0.2/finite-angular-long-N36/layer.athinput']
before={str(f.relative_to(repo)):sha(f) for f in files}
assert all(before[k]==v for k,v in base['source_before'].items()) and all(before[k]==v for k,v in supp['source_before'].items())
assert len(set(before))==len(files)
results=[]
# Original exploratory baseline data are preserved by hash before an independently
# recorded exact rerun; no compile receipt or prior scientific output is rewritten.
old_base=sha(p/'constraint-base.json');start=time.monotonic()
for cmd,out in [([str(p/'constraint_tangent'),'base'],'constraint-base-rerun.json'),([sys.executable,str(p/'check_live.py')],'check-final.log')]:
 t=time.monotonic()
 with (p/out).open('w') as f:r=subprocess.run(cmd,cwd=repo,stdout=f,stderr=subprocess.PIPE,text=True)
 result={'command':cmd,'returncode':r.returncode,'stderr':r.stderr,'seconds':time.monotonic()-t,'stdout_file':out,'stdout_sha256':sha(p/out)};results.append(result)
 if r.returncode:
  (p/'final-failed-receipt.json').write_text(json.dumps({'source_before':before,'commands':results},indent=2)+'\n');r.check_returncode()
 assert not r.stderr
 if out=='constraint-base-rerun.json':assert sha(p/out)==old_base
report=json.loads((p/'check-report.json').read_text());assert report['passed_finite_Omega_local_gate']
after={k:sha(repo/k) for k in before};assert before==after
for result in base['commands']+supp['commands']:
 if 'stdout_file' in result:assert sha(p/result['stdout_file'])==result['stdout_sha256']
receipt={'passed_finite_Omega_local_numerical_gate':True,'global_native_or_scri_stability_accepted':False,'source_before':before,'source_after':after,'source_count':len(before),'production_src_and_root_CMake_count':365,'sources_unchanged':True,'runtime_implementation':base['runtime_implementation'],'compiled_launch_head':base['launch_head'],'finalize_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'compile_receipt_sha256':sha(p/'compile-receipt.json'),'supplement_receipt_sha256':sha(p/'supplement-receipt.json'),'original_baseline_sha256':old_base,'baseline_rerun_byte_identical':True,'check_report_sha256':sha(p/'check-report.json'),'helper_sha256':sha(p/'live_damping_profile.hpp'),'commands':base['commands']+supp['commands']+results,'command_count':len(base['commands'])+len(supp['commands'])+len(results),'binary_sha256':base['binary_sha256']|{'full20_RK_all_a':sha(p/'full20_RK_all_a')},'toolchain':{'compiler':base['compiler'],'python':base['python'],'numpy':base['numpy']},'scope':'Actual20 and linear Einstein-reference eight-constraint local gates, finite Omega. All positive local/primitive roots retained. No native run, global/energy bound, nonlinear admissibility preservation, BH compatibility or exact scri closure is accepted.','seconds_finalize':time.monotonic()-start}
(p/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print('PASS final live gate',sha(p/'receipt.json'),'inputs',len(before),'commands',receipt['command_count'])
