from pathlib import Path
import json,hashlib,subprocess,time
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'commands':[]}
paths=subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=ROOT,text=True).splitlines()+[str(f.relative_to(ROOT))for f in P.rglob('*')if f.suffix in ['.cpp','.hpp','.py'] and 'history' not in f.relative_to(P).parts]
r['source_before']={f:sha(ROOT/f)for f in paths}
flags=json.loads((ROOT/'build-layer-research/continuum/conformal-q-null-feedback/immutable-conformal-Q-null-feedback-local-20261009/receipt.json').read_text())['commands'][0]['command'][:-3]
def run(cmd,name):
 t=time.monotonic();q=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True);(P/(name+'.stdout')).write_text(q.stdout);(P/(name+'.stderr')).write_text(q.stderr);r['commands'].append(dict(command=cmd,returncode=q.returncode,seconds=time.monotonic()-t,stdout=name+'.stdout',stderr=name+'.stderr'));(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');assert q.returncode==0,q.stderr;return q.stdout
run(flags+[str(P/'jet_map.cpp'),'-o',str(P/'jet-map')],'compile')
(P/'actual.json').write_text(run([str(P/'jet-map')],'actual'))
debugflags=json.loads((ROOT/'build-layer-research/continuum/conformal-q-null-feedback/immutable-conformal-Q-null-feedback-local-20261009/receipt.json').read_text())['commands'][9]['command'][:-3]
run(debugflags+[str(P/'jet_map.cpp'),'-o',str(P/'jet-map-debug')],'compile-debug')
(P/'actual-debug.json').write_text(run([str(P/'jet-map-debug')],'actual-debug'))
assert json.loads((P/'actual-debug.json').read_text())==json.loads((P/'actual.json').read_text())
run(['/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python',str(P/'check_maps.py')],'check')
r['source_after']={f:sha(ROOT/f)for f in paths};r['sources_unchanged']=r['source_before']==r['source_after'];(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
