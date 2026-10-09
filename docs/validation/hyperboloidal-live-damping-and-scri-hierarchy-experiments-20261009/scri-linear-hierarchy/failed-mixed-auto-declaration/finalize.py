from pathlib import Path
import subprocess,hashlib,json,time,sys
p=Path(__file__).resolve().parent;repo=p.parents[2];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
base=json.loads((p.parent/'live-damping-control/receipt.json').read_text());sources={k:v for k,v in base['source_before'].items() if k.startswith('src/') or k=='CMakeLists.txt' or k.endswith(('profile_helpers.hpp','live_damping_profile.hpp','dual_helpers.hpp'))}
for f in ['taylor_kernel.cpp','run_audit.py','firstjet_gate.cpp','check_hierarchy.py','finalize.py']:sources[str((p/f).relative_to(repo))]=sha(p/f)
assert all(sha(repo/k)==v for k,v in sources.items());flags=base['commands'][0]['command'][:-3]
debug=flags[:];debug[debug.index('-O3')]='-O1';debug.remove('-DNDEBUG');debug+=['-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
old=sha(p/'kernel.json');commands=[(flags+[str(p/'taylor_kernel.cpp'),'-o',str(p/'taylor_kernel')],None),([str(p/'taylor_kernel')],'kernel.json'),(flags+[str(p/'firstjet_gate.cpp'),'-o',str(p/'firstjet_gate')],None),(debug+[str(p/'firstjet_gate.cpp'),'-o',str(p/'firstjet_gate_debug')],None),([str(p/'firstjet_gate')],'firstjet.json'),([str(p/'firstjet_gate_debug')],'firstjet-debug.json'),([sys.executable,str(p/'check_hierarchy.py')],'check-final.log')];rows=[]
for cmd,out in commands:
 t=time.monotonic()
 if out:
  with (p/out).open('w') as f:r=subprocess.run(cmd,cwd=repo,stdout=f,stderr=subprocess.PIPE,text=True)
 else:r=subprocess.run(cmd,cwd=repo,capture_output=True,text=True)
 row={'command':cmd,'returncode':r.returncode,'stderr':r.stderr,'seconds':time.monotonic()-t};rows.append(row)
 if out:row.update(stdout_file=out,stdout_sha256=sha(p/out))
 else:row['stdout']=r.stdout
 if r.returncode or r.stderr:
  (p/'final-failed-receipt.json').write_text(json.dumps({'sources':sources,'commands':rows},indent=2)+'\n');r.check_returncode();assert not r.stderr
assert sha(p/'kernel.json')==old
j=json.loads((p/'firstjet.json').read_text());assert j==json.loads((p/'firstjet-debug.json').read_text());assert j['operator_rows']==640 and j['full_R0_formula_error']<1e-12 and j['rows']==160 and j['all_pole_firstjet_formula_error']<1e-12 and j['nonfree_angular_chi_jet_error']<1e-14 and j['Theta_corner_actual_limit_error']<1e-8
assert all(sha(repo/k)==v for k,v in sources.items())
report=json.loads((p/'check-report.json').read_text());report['general_firstjet_actual_gate']=j;(p/'check-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
receipt={'passed_actual_kernel_linear_hierarchy_gate':True,'PDE_finite_Q_amplitude_blowup_claim':False,'closed_scri_PDE_accepted':False,'sources':sources,'source_count':len(sources),'sources_unchanged':True,'kernel_rerun_byte_identical':True,'commands':rows,'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'runtime_implementation':base['runtime_implementation'],'check_report_sha256':sha(p/'check-report.json'),'kernel_sha256':sha(p/'kernel.json'),'binary_sha256':{f:sha(p/f) for f in ['taylor_kernel','firstjet_gate','firstjet_gate_debug']},'scope':'Reference linear Taylor compatibility and actual20 frozen-normal amplitude caveat only, no native run or PDE closure/amplitude bound. Initial serialization-only failed checker and earlier source/output passes are preserved.'}
(p/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS final Taylor hierarchy',sha(p/'receipt.json'),j)
