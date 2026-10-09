"""Fresh scratch-only actual tensor/gauge gate. No native integration."""
import hashlib,json,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
PY='/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
flags=json.loads((ROOT/'build-layer-research/continuum/live-damping-control/receipt.json').read_text())['commands'][0]['command'][:-3]
receipt={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','commands':[]}
prod=subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=ROOT,text=True).splitlines()
inputs=prod+[str(p.relative_to(ROOT)) for p in P.rglob('*') if p.suffix in ['.cpp','.hpp','.py']]
receipt['source_before']={p:sha(ROOT/p) for p in inputs}
def run(cmd,name):
 t=time.monotonic();r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
 (P/(name+'.stdout')).write_text(r.stdout);(P/(name+'.stderr')).write_text(r.stderr)
 receipt['commands'].append({'command':cmd,'returncode':r.returncode,'seconds':time.monotonic()-t,'stdout':name+'.stdout','stderr':name+'.stderr','stdout_sha256':sha(P/(name+'.stdout')),'stderr_sha256':sha(P/(name+'.stderr'))})
 (P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
 return r.stdout
for name in ['full20','nonlinear','principal','prior_corner']:
 run(flags+[str(P/(name+'.cpp')),'-o',str(P/name)],'compile-'+name)
 (P/(name+'.json')).write_text(run([str(P/name)],'run-'+name))
run([PY,str(ROOT/'tst/hyperboloidal/check_kernel_symbol.py'),str(P/'principal')],'check-principal')
debugflags=[f for f in flags if f!='-DNDEBUG'];debugflags=[('-O0' if f=='-O3' else f) for f in debugflags]+['-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
for name in ['full20','nonlinear']:
 run(debugflags+[str(P/(name+'.cpp')),'-o',str(P/(name+'-debug'))],'compile-'+name+'-debug')
 (P/(name+'-debug.json')).write_text(run([str(P/(name+'-debug'))],'run-'+name+'-debug'))
run([PY,str(P/'check_gate.py')],'check-gate')
for name in ['full20','nonlinear']:
 assert json.loads((P/(name+'.json')).read_text())==json.loads((P/(name+'-debug.json')).read_text()),name
receipt['passed_local_pole_source_principal_corner_gates']=True
receipt['release_debug_json_equal']=True
receipt['native_or_global_accepted']=False
receipt['binary_sha256']={name:sha(P/name)for name in ['full20','nonlinear','principal','prior_corner','full20-debug','nonlinear-debug']}
receipt['source_after']={p:sha(ROOT/p) for p in inputs};receipt['sources_unchanged']=receipt['source_before']==receipt['source_after']
(P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
