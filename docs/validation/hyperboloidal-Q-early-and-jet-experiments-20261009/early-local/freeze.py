from pathlib import Path
import hashlib,json,shutil
P=Path(__file__).resolve().parent;R=P.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'receipt.json').read_text());assert r['sources_unchanged'];assert len(r['source_before'])==376 and len(r['commands'])==8
assert all(sha(R/f)==s for f,s in r['source_before'].items())
assert sha(P/'check_early.py')==r['postprocessing_source_sha256']
assert all(c['returncode']==0 and not (P/c['stderr']).read_bytes()for c in r['commands'])
c=json.loads((P/'check-report.json').read_text());assert c['passed_source_principal_and_outer_pole_identity_gates'] and not c['sampled_frozen_spectrum_nonpositive'] and not c['global_native_accepted']
O=P/'immutable-Q-null-early-feedback-local-20261009';assert not O.exists();O.mkdir()
files={}
for f in sorted(P.rglob('*')):
 if not f.is_file() or O in f.parents or f.name in ['fourier','source','principal']:continue
 rel=f.relative_to(P);q=O/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,q);files[str(rel)]=sha(q)
large={name:{'sha256':sha(P/name),'bytes':(P/name).stat().st_size,'original_repo_path':str((P/name).relative_to(R))}for name in ['fourier','source','principal']}
idx={'scope':'Earlier prescribed-feedback local attribution gate; positive resolved primitive roots retained, no global/native/stability admission. Original late/core freezes unchanged.','files':files,'input_count':376,'command_count':8,'matrix_count':560,'matched_parameter_points':280,'production_files_unchanged':365,'receipt_sha256':sha(P/'receipt.json'),'helper_sha256':sha(P/'early_feedback.hpp'),'core_index_sha256':'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96','late_negative_index_sha256':'adaa2b1437054f4ba4cf6471eafb6f83af96a31e9ea376d723a2666abcf5e72a','postprocessing_source_sha256':r['postprocessing_source_sha256'],'large_outputs_outside_snapshot':large,'file_count':len(files),'bytes':sum((O/f).stat().st_size for f in files)}
(O/'index.json').write_text(json.dumps(idx,indent=2)+'\n');assert all(sha(O/f)==s for f,s in files.items());print(json.dumps({'index':str(O/'index.json'),'sha256':sha(O/'index.json'),'receipt_sha256':sha(O/'receipt.json'),'files':len(files),'bytes':idx['bytes']},indent=2))
