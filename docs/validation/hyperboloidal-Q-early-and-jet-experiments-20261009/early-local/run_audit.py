from pathlib import Path
import hashlib,json,subprocess,time
P=Path(__file__).resolve().parent;ROOT=P.parents[2];PY='/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
flags=json.loads((ROOT/'build-layer-research/continuum/conformal-q-null-feedback/immutable-conformal-Q-null-feedback-local-20261009/receipt.json').read_text())['commands'][0]['command'][:-3]
paths=subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=ROOT,text=True).splitlines()+[str(p.relative_to(ROOT))for p in P.rglob('*')if p.suffix in ['.cpp','.hpp','.py']]
r={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_before':{f:sha(ROOT/f)for f in paths},'commands':[]}
def run(cmd,name):
 t=time.monotonic();q=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True);(P/(name+'.stdout')).write_text(q.stdout);(P/(name+'.stderr')).write_text(q.stderr);r['commands'].append(dict(command=cmd,returncode=q.returncode,seconds=time.monotonic()-t,stdout=name+'.stdout',stderr=name+'.stderr'));(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');assert q.returncode==0,q.stderr;return q.stdout
for name in ['fourier','source','principal']:
 run(flags+[str(P/(name+'.cpp')),'-o',str(P/name)],'compile-'+name);(P/(name+'.json')).write_text(run([str(P/name)],'run-'+name))
run([PY,str(ROOT/'tst/hyperboloidal/check_kernel_symbol.py'),str(P/'principal')],'check-principal')
r['source_after']={f:sha(ROOT/f)for f in paths};r['sources_unchanged']=r['source_before']==r['source_after'];(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
