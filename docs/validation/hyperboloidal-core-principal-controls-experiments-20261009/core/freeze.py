"""Freeze completed local core evidence; no new scientific calculation."""
from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import sys

P=Path(__file__).resolve().parent
ROOT=P.parents[2]
D=P/'immutable-total-J-flat-core-envelope-20261009'
assert not D.exists(), 'Never rewrite frozen evidence'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
symbolic=json.loads((P/'symbolic-report.json').read_text())
local=json.loads((P/'local-report.json').read_text())
cart=json.loads((P/'cartesian-oracle-report.json').read_text())
assert symbolic['passed_exact_flat_core_envelope_derivation']
assert local['passed_local_flat_core_envelope_gate'] and all(local['checks'].values())
assert cart['passed_declared_cartesian_core_oracle']
assert sha(P/'derive_core.py')==symbolic['source_sha256']
assert sha(P/'check_local.py')==local['source_sha256']
assert sha(P/'core_envelope.hpp')==symbolic['generated_header_sha256']
assert sha(P/'core-envelope-blocks.npz')==symbolic['block_npz_sha256']
assert sha(P/'local-actions.npz')==local['actions_npz_sha256']
pins=json.loads((P/'source-pins.json').read_text())
for record in pins.values():assert sha(record['path'])==record['sha256']
for key in ('basis_index','angular_index'):
    source=Path(pins[key]['path']);index=json.loads(source.read_text())
    for record in index['files']:
        file=source.parent/record['path']
        assert sha(file)==record['sha256']

receipts=[]
for name in ('release','asan'):
    receipts.append(P/'local-attempt-001'/('build-'+name+'.json'))
for name in ('release','debug'):
    receipts.append(P/'cartesian-oracle-attempts'/name/'receipt.json')
for value in local['original_native_bridge_dependency_verification'].values():
    source=Path(value['path']);assert sha(source)==value['sha256']
    target=P/'inputs/original-native-build'/source.parent.name/'receipt.json'
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    receipts.append(source)
dependencies={}
for source in receipts:
    r=json.loads(source.read_text())
    for path,digest in r['compiler_dependency_hashes'].items():
        assert sha(path)==digest,path
        dependencies[path]=digest
        file=Path(path)
        if file.is_relative_to(ROOT):
            rel=file.relative_to(ROOT)
            if rel.parts[0] in ('src','build-layer-research'):
                target=P/'compiled-sources'/rel
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(file,target)
                assert sha(target)==digest
    for path,digest in r.get('link_archive_hashes',{}).items():
        assert sha(path)==digest
for name,r in local['runs'].items():
    assert sha(r['command'][0])==r['executable_sha256']
    assert r['exit_code']==0 and r['stderr_bytes']==0
    assert sha(P/(name+'.txt'))==r['output_sha256']
assert not (P/'symbolic.stderr').read_bytes()
assert not (P/'local.stderr').read_bytes()
assert not (P/'cartesian.stderr').read_bytes()

def finite(value):
    if isinstance(value,float):assert math.isfinite(value)
    elif isinstance(value,dict):
        for x in value.values():finite(x)
    elif isinstance(value,list):
        for x in value:finite(x)

runtime={'documentation_HEAD_at_freeze':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'git_status_porcelain':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
         'python':sys.version,'python_executable':sys.executable,'python_executable_sha256':sha(sys.executable),
         'source_pins_unchanged':True,'original_basis_angular_indexes_readback_passed':True,
         'reverified_unique_compiler_dependencies':len(dependencies),
         'no_actual_radial_operator_boundary_eigen_or_evolution':True,
         'reproduction_commands':[
             'OPENBLAS_NUM_THREADS=1 /Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python derive_core.py',
             'OPENBLAS_NUM_THREADS=1 /Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python check_local.py',
             'OPENBLAS_NUM_THREADS=1 /Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python run_cartesian_oracle.py'],
         'reproduction_note':'Use a fresh copy with the documented original include paths; attempt directories deliberately reject overwrite. Original native bridge executables are reused only after hash/dependency verification.'}
(P/'freeze-verification.json').write_text(json.dumps(runtime,indent=2)+'\n')
excluded_names={'core-envelope-release','core-envelope-asan','cartesian-oracle-release','cartesian-oracle-debug'}
files=[file for file in P.rglob('*') if file.is_file() and D not in file.parents and '__pycache__' not in file.parts]
D.mkdir()
small=[];large=[];json_count=0
for file in sorted(files):
    rel=file.relative_to(P)
    record={'path':str(rel),'bytes':file.stat().st_size,'sha256':sha(file)}
    if file.stat().st_size>1000000 or file.name in excluded_names or any(part.endswith('.dSYM') for part in rel.parts):
        record['external_path']=str(file)
        large.append(record)
    else:
        target=D/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
        assert sha(target)==record['sha256']
        if file.suffix=='.json':finite(json.loads(file.read_text()));json_count+=1
        small.append(record)
index={'scope':local['scope'],'passed_exact_symbolic_and_local_core_gate':True,
       'actual_radial_operator_boundary_eigen_or_evolution_admitted':False,
       'files':small,'large_files_metadata_only':large,'small_file_count':len(small),
       'small_file_bytes':sum(v['bytes'] for v in small),'finite_json_count':json_count,
       'all_copied_and_external_hashes_verified':True,'all_failures_preserved':True,
       'report_sha256':sha(P/'REPORT.md'),'symbolic_source_sha256':sha(P/'derive_core.py'),
       'generated_header_sha256':sha(P/'core_envelope.hpp'),'plan_sha256':sha(P/'plan.json'),
       'symbolic_report_sha256':sha(P/'symbolic-report.json'),'local_report_sha256':sha(P/'local-report.json')}
(D/'index.json').write_text(json.dumps(index,indent=2)+'\n')
for record in small:assert sha(D/record['path'])==record['sha256']
for record in large:assert sha(record['external_path'])==record['sha256']
print(json.dumps({'index':str(D/'index.json'),'sha256':sha(D/'index.json'),
                  'files':len(small),'bytes':index['small_file_bytes'],'finite_json':json_count,
                  'large_metadata_only':len(large),'verified':True},indent=2))
