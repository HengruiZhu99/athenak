"""Freeze exact exploratory sources, gates and completed native receipts."""
from pathlib import Path
import hashlib,json,shutil,subprocess
p=Path(__file__).resolve().parent
repo=p.parents[4]
out=p/'immutable-spatial-norm-20261009'
assert not out.exists(), 'refuse to overwrite immutable completed experiment'
summary=json.loads((p/'native-summary.json').read_text())
assert summary['final']['time']==2. and summary['exit_status']==0
build=json.loads((p/'native-build-receipt.json').read_text())
for name,h in build['source_sha256'].items():
 assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==h,name
for name,h in build['overlay_sha256'].items():
 f=Path(name);f=f if f.is_absolute() else p/f
 assert hashlib.sha256(f.read_bytes()).hexdigest()==h,name
out.mkdir()
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
files=[]
for glob in ['*.hpp','*.cpp','*.py','*.cmake','*.athinput','*.log']:
 files.extend(p.glob(glob))
files.extend(p/x for x in ['receipt.json','source-before.json','native-build-receipt.json','native-overlay-gate-receipt.json','BH-gate-receipt-pass-20261009.json','BH-gate-first-failure-observation.json','native-experiment-report.json','native-summary.json','constraint-budgets.json','REPORT.md'])
files.extend((p/'BH-gate-complete-first-failure-20261009').glob('*'))
for phase in ['reference','short','long']:
 d=p/('native-'+phase)
 files.append(d/'results.json')
 for name in ['manifest.json','changes.patch']:files.append(d/'source-at-launch'/name)
 for case in json.loads((d/'results.json').read_text())['cases']:
  cd=d/case['name']
  for name in ['layer.athinput','run.log','hyp.z4c.user.hst']:files.append(cd/name)
archived={}
for f in sorted(set(files)):
 assert f.is_file(),f
 rel=f.relative_to(p);target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(f,target);target.chmod(0o444)
 archived[str(rel)]={'sha256':sha(target),'bytes':target.stat().st_size}
large={}
for name in ['poles.json','fourier.json','report.json','overlay-poles.json','overlay-fourier.json']:
 f=p/name;large[str(f)]={'sha256':sha(f),'bytes':f.stat().st_size}
exe=repo/'build-layer-spatial-norm-native/src/athena'
large[str(exe)]={'sha256':sha(exe),'bytes':exe.stat().st_size}
for phase in ['reference','short','long']:
 for case in json.loads((p/('native-'+phase)/'results.json').read_text())['cases']:
  for f in sorted((p/('native-'+phase)/case['name']/'bin').glob('*')):
   if f.is_file():large[str(f)]={'sha256':sha(f),'bytes':f.stat().st_size}
manifest={'scope':'Immutable exploratory live spatial-norm-feedback source/native experiment. No production option, global PDE stability, nonlinear regularity closure or BH evolution claim.',
 'compiled_implementation':summary['implementation_commit'],'launch_head':summary['launch_head'],
 'native_executable_sha256':summary['native_executable_sha256'],
 'archived_files':archived,'large_artifact_hashes':large,
 'tracked_runtime_diff_from_27c19d20':subprocess.check_output(['git','diff','27c19d20','--','src','CMakeLists.txt'],cwd=repo,text=True),
 'git_status':subprocess.check_output(['git','status','--short'],cwd=repo,text=True)}
assert manifest['tracked_runtime_diff_from_27c19d20']==''
assert manifest['git_status']==''
(out/'immutable-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'immutable-manifest.json').chmod(0o444)
print('Archived',len(archived),'small files; hashed',len(large),'large artifacts')
print('Manifest',sha(out/'immutable-manifest.json'))
