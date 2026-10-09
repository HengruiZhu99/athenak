from pathlib import Path
import json,hashlib,subprocess,time,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2];G=ROOT/'build-layer-research/continuum/q-sigma3-frozen-fourier/immutable-Q-sigma3-raw22-intrinsic20-frozen-Fourier-20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((G/'receipt.json').read_text());index=json.loads((G/'index.json').read_text())
for n,d in index['files'].items():assert sha(G/n)==d['sha256']
for n,d in old['source_before'].items():assert sha(ROOT/n)==d
paths=list(old['source_before'])+[str((G/'index.json').relative_to(ROOT))]+[str((P/n).relative_to(ROOT))for n in ['check_fourier22.py','run_reanalysis.py','freeze.py']]
r={'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'compiled_production_implementation':old['compiled_production_implementation'],'original_exploration_HEAD':old['exploration_pilot_launch_HEAD'],'original_final_runner_HEAD':old['final_runner_launch_HEAD'],'original_frozen_index_sha256':sha(G/'index.json'),'all_46_v1_frozen_files_verified':True,'all_381_original_source_inputs_unchanged':True,'source_before':{n:sha(ROOT/n)for n in paths},'original_warning_command':old['commands'][4]['command'],'original_warning_stderr_sha256':sha(G/'check.stderr'),'original_warning_stderr_bytes':(G/'check.stderr').stat().st_size,'original_cpp_commands_clean_and_payloads_unchanged':all((G/c['stderr']).stat().st_size==0 for c in old['commands'][:4]),'commands':[]}
command=['/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python',str(P/'check_fourier22.py')]
t=time.monotonic();q=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);(P/'check.stdout').write_text(q.stdout);(P/'check.stderr').write_text(q.stderr);r['commands'].append({'command':command,'returncode':q.returncode,'seconds':time.monotonic()-t,'stdout':'check.stdout','stderr':'check.stderr'})
assert q.returncode==0 and not q.stderr,q.stderr
x=json.loads((G/'root-maxima.json').read_text());y=json.loads((P/'root-maxima.json').read_text());assert len(x)==len(y)==5880
r['positive_root_counts_identical_every_parameter_row']=all(d['positive_roots']==e['positive_roots']for d,e in zip(x,y));assert r['positive_root_counts_identical_every_parameter_row']
r['max_real_root_maximum_change']=max(abs(d['max_real']-e['max_real'])for d,e in zip(x,y));assert r['max_real_root_maximum_change']<1e-9
old_roots=np.load(G/'roots.npz');new_roots=np.load(P/'roots.npz');r['root_set_comparison']={}
for k in old_roots.files:
 distance=np.abs(old_roots[k][:,:,None]-new_roots[k][:,None,:]);hausdorff=float(np.maximum(distance.min(axis=1).max(axis=1),distance.min(axis=2).max(axis=1)).max());assert hausdorff<1e-9
 r['root_set_comparison'][k]={'array_equal':bool(np.array_equal(old_roots[k],new_roots[k])),'max_bidirectional_nearest_root_distance':hausdorff}
r['source_after']={n:sha(ROOT/n)for n in paths};r['sources_unchanged']=r['source_before']==r['source_after'];assert r['sources_unchanged']
r['passed_warning_free_additive_reanalysis']=True;r['native_global_admission']=False
(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'passed':True,'inputs':len(paths),'commands':1,'seconds':r['commands'][0]['seconds'],'max_real_root_maximum_change':r['max_real_root_maximum_change'],'root_set_comparison':r['root_set_comparison']},indent=2))
