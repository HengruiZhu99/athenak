from pathlib import Path
import json,hashlib,subprocess,time
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'commands':[]};paths=subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=ROOT,text=True).splitlines()+[str(f.relative_to(ROOT))for f in P.rglob('*')if f.suffix in ['.cpp','.hpp','.py']and'history'not in f.relative_to(P).parts];r['source_before']={f:sha(ROOT/f)for f in paths}
old=json.loads((ROOT/'build-layer-research/continuum/conformal-q-null-feedback/immutable-conformal-Q-null-feedback-local-20261009/receipt.json').read_text())
def run(cmd,name):
 t=time.monotonic();q=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True);(P/(name+'.stdout')).write_text(q.stdout);(P/(name+'.stderr')).write_text(q.stderr);r['commands'].append(dict(command=cmd,returncode=q.returncode,seconds=time.monotonic()-t,stdout=name+'.stdout',stderr=name+'.stderr'));(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');assert q.returncode==0,q.stderr;return q.stdout
for mode,index in [('release',0),('debug',9)]:
 flags=old['commands'][index]['command'][:-3];run(flags+[str(P/'angular.cpp'),'-o',str(P/('angular-'+mode))],'compile-'+mode);(P/('actual-'+mode+'.json')).write_text(run([str(P/('angular-'+mode))],'run-'+mode))
assert json.loads((P/'actual-release.json').read_text())==json.loads((P/'actual-debug.json').read_text())
r['source_after']={f:sha(ROOT/f)for f in paths};r['sources_unchanged']=r['source_before']==r['source_after'];(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
