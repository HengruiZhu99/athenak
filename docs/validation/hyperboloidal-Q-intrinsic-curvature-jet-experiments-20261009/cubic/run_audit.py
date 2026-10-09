from pathlib import Path
import hashlib,json,subprocess,time
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=ROOT/'build-layer-research/continuum/conformal-q-null-feedback/immutable-conformal-Q-null-feedback-local-20261009'
prior_receipt=json.loads((prior/'receipt.json').read_text())
paths=subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=ROOT,text=True).splitlines()
paths+=[str(p.relative_to(ROOT))for p in P.rglob('*')if p.suffix in ['.hpp','.cpp','.py']and'history'not in p.relative_to(P).parts and not any(x.startswith('immutable-')for x in p.relative_to(P).parts)]
paths.append(str((prior/'index.json').relative_to(ROOT)))
r={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
   'compiled_production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2',
   'source_before':{p:sha(ROOT/p)for p in paths},'commands':[]}
def run(command,name):
    start=time.monotonic();q=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (P/(name+'.stdout')).write_text(q.stdout);(P/(name+'.stderr')).write_text(q.stderr)
    r['commands'].append({'command':command,'returncode':q.returncode,'seconds':time.monotonic()-start,'stdout':name+'.stdout','stderr':name+'.stderr'})
    (P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
    assert q.returncode==0,q.stderr
    return q.stdout
for mode,command_index in [('release',0),('debug',9)]:
    flags=prior_receipt['commands'][command_index]['command'][:-3]
    run(flags+[str(P/'curvature.cpp'),'-o',str(P/('curvature-'+mode))],'compile-'+mode)
    (P/('actual-'+mode+'.json')).write_text(run([str(P/('curvature-'+mode))],'run-'+mode))
assert json.loads((P/'actual-release.json').read_text())==json.loads((P/'actual-debug.json').read_text())
run(['/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python',str(P/'check_curvature.py')],'check')
r['source_after']={p:sha(ROOT/p)for p in paths};r['sources_unchanged']=r['source_before']==r['source_after']
r['actual_Release_ASan_UBSan_json_equal']=True
r['passed']=r['sources_unchanged'] and all(c['returncode']==0 for c in r['commands'])
(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');assert r['passed']
print(json.dumps({'passed':r['passed'],'inputs':len(paths),'commands':len(r['commands']),'seconds':sum(c['seconds']for c in r['commands'])},indent=2))
