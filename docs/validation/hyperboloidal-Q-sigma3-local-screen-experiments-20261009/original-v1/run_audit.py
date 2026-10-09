from pathlib import Path
import hashlib,json,subprocess,time
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
core=ROOT/'build-layer-research/continuum/conformal-q-null-feedback/immutable-conformal-Q-null-feedback-local-20261009'
old=ROOT/'build-layer-research/continuum/conformal-q-followup/immutable-Q-null-finite-frequency-negative-20261009'
early=ROOT/'build-layer-research/continuum/q-null-early-feedback/immutable-Q-null-early-feedback-local-20261009'
prior=json.loads((core/'receipt.json').read_text())
paths=subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=ROOT,text=True).splitlines()
paths+=[str(p.relative_to(ROOT))for p in P.rglob('*')if p.suffix in ['.hpp','.cpp','.py']and'history'not in p.relative_to(P).parts and not any(x.startswith('immutable-')for x in p.relative_to(P).parts)]
paths+=[str(p.relative_to(ROOT))for p in [core/'index.json',core/'receipt.json',old/'index.json',old/'fourier.json',early/'index.json',early/'fourier.json']]
assert len(paths)==len(set(paths))
r={'exploration_pilot_launch_HEAD':'a7260897b92d12d5fc1ceb2f3dc6a886676a663f','final_runner_launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'compiled_production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','source_before':{p:sha(ROOT/p)for p in paths},'commands':[]}
def run(command,name):
    t=time.monotonic();q=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (P/(name+'.stdout')).write_text(q.stdout);(P/(name+'.stderr')).write_text(q.stderr)
    r['commands'].append({'command':command,'returncode':q.returncode,'seconds':time.monotonic()-t,'stdout':name+'.stdout','stderr':name+'.stderr'})
    (P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
    assert q.returncode==0,q.stderr
    return q.stdout
for mode,command_index in [('release',0),('debug',9)]:
    flags=prior['commands'][command_index]['command'][:-3]
    run(flags+[str(P/'fourier22.cpp'),'-o',str(P/('fourier22-'+mode))],'compile-'+mode)
    (P/('metadata-'+mode+'.json')).write_text(run([str(P/('fourier22-'+mode)),str(P/('matrices-'+mode+'.bin'))],'run-'+mode))
run(['/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python',str(P/'check_fourier22.py')],'check')
r['source_after']={p:sha(ROOT/p)for p in paths};r['sources_unchanged']=r['source_before']==r['source_after'];assert r['sources_unchanged']
r['Release_ASan_UBSan_metadata_and_matrix_bytes_equal']=True
r['artifacts']={str(p.relative_to(P)):{'sha256':sha(p),'bytes':p.stat().st_size}for p in [P/'fourier22-release',P/'fourier22-debug',P/'matrices-release.bin',P/'matrices-debug.bin',P/'metadata-release.json',P/'metadata-debug.json',P/'roots.npz',P/'root-maxima.json',P/'check-report.json']}
r['passed_local_gate']=all(c['returncode']==0 for c in r['commands']) and r['sources_unchanged'];r['native_global_admission']=False
(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'passed_local_gate':r['passed_local_gate'],'source_inputs':len(paths),'commands':len(r['commands']),'seconds':sum(c['seconds']for c in r['commands'])},indent=2))
