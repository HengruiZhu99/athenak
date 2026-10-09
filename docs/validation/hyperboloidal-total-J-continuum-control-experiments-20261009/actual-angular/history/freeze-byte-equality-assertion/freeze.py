"""Freeze compact local/angular evidence, verifying all accepted source pins."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys

P=Path(__file__).resolve().parent
ROOT=P.parents[2]
DEST=P/'immutable-C0-spatialnorm-total-J-local-angular-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not DEST.exists()
prepare=json.loads((P/'preparation.json').read_text())
for row in prepare['input_pins']:
    assert sha(row['path'])==row['sha256'],row['path']
held=P.parent/'total-j-cartesian-control-held'
assert sha(held/'preparation-receipt.json')=='bf79cca0eb151c56c955e5390c3b12ef0568c615cb129fa507aff0ef6cad50e9'
for row in json.loads((held/'preparation-receipt.json').read_text())['files']:
    assert sha(row['path'])==row['sha256'],row['path']
for row in json.loads((held/'source-pins.json').read_text())['files']:
    assert sha(row['absolute_path'])==row['sha256'],row['absolute_path']
basis=ROOT/'build-layer-research/continuum/total-j-harmonic-basis/immutable-Cartesian-total-J-basis-20261009'
assert sha(basis/'index.json')=='414f241e986e0c46b7d94820fb060b704166d973284854c53d2e8c64ee6c489e'
for row in json.loads((basis/'index.json').read_text())['files']:
    assert sha(basis/row['path'])==row['sha256']
accepted=[]
for mode in ('release','debug'):
    latest=json.loads((P/('build-'+mode+'-latest.json')).read_text());b=Path(latest['attempt'])/'receipt.json'
    assert sha(b)==latest['receipt_sha256'];accepted.append(b)
    local=json.loads((P/('local-'+mode+'-latest.json')).read_text());r=Path(local['attempt'])/'receipt.json'
    assert sha(r)==local['receipt_sha256'];assert json.loads(r.read_text())['passed_all_local_gates']
    accepted.append(P/'core-oracle-attempts'/mode/'receipt.json')
external_deps={}
for p in accepted:
    record=json.loads(p.read_text());assert record.get('exit_code',record.get('compiler_exit_code'))==0
    for path,h in record['compiler_dependency_hashes'].items():
        assert sha(path)==h,path
        if str(path).startswith(str(ROOT/'src')+'/') or ('/build-layer-research/' in path and not path.startswith(str(P)+'/')):
            external_deps[path]=h
    for path,h in record['link_archive_hashes'].items():assert sha(path)==h
angular=json.loads((P/'angular-analysis-002.json').read_text());assert angular['passed_all_angular_gates']
assert (P/'angular-warning-free.stderr').stat().st_size==0
warning=json.loads((P/'warning-free-control.json').read_text());assert warning['coefficient_relative_difference_max']==0
assert angular['source_sha256']==sha(P/'bridge.cpp')
assert sha(P/'angular-actions.txt')==angular['runtime']['output_sha256']
local_release=Path(json.loads((P/'local-release-latest.json').read_text())['attempt'])/'stdout.json'
local_debug=Path(json.loads((P/'local-debug-latest.json').read_text())['attempt'])/'stdout.json'
assert local_release.read_bytes()==local_debug.read_bytes()
for mode in ('release','debug'):
    c=json.loads((P/'core-oracle-attempts'/mode/'stdout.json').read_text());assert c['cases']==1404 and c['rhs_scaled_error']<=5e-12 and c['physical8_constraint_scaled_error']<=5e-12
environment={'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'python':sys.version,'python_binary':sys.executable,'python_binary_sha256':sha(sys.executable),'platform':platform.platform(),'OpenBLAS_threads_requested':'1','scientific_environment':'run_angular.py used pinned workspace python-deps NumPy2; no eigensolve/propagation'}
(P/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
DEST.mkdir()
large=[];files=[]
binary_names={'bridge-release','bridge-debug','core-oracle-release','core-oracle-debug'}
for source in sorted(P.rglob('*')):
    if not source.is_file() or DEST in source.parents:continue
    rel=source.relative_to(P)
    if source.name in binary_names or any(str(part).endswith('.dSYM') for part in rel.parts) or source.stat().st_size>5_000_000:
        large.append({'path':str(source),'relative_path':str(rel),'sha256':sha(source),'bytes':source.stat().st_size});continue
    target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    files.append({'path':str(rel),'sha256':sha(target),'bytes':target.stat().st_size})
for path,h in sorted(external_deps.items()):
    source=Path(path)
    if str(source).startswith(str(ROOT/'src')+'/'):rel=Path('compiled-sources/production')/source.relative_to(ROOT)
    else:rel=Path('compiled-sources/external-research')/source.relative_to(ROOT/'build-layer-research')
    target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha(target)==h
    files.append({'path':str(rel),'sha256':h,'bytes':target.stat().st_size})
finite_json=0
def finite(v):
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,list):return all(finite(x) for x in v)
    return not isinstance(v,(int,float)) or math.isfinite(v)
for row in files:
    f=DEST/row['path'];assert sha(f)==row['sha256']
    if f.suffix=='.json':assert finite(json.loads(f.read_text())),f;finite_json+=1
index={'kind':'actual-C0-spatialnorm-continuum-total-J-local-angular-gate','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_HEAD':environment['freeze_HEAD'],'runtime_source_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2','root_release_sha256':'0c4d415d79e6bb50acfe6f0d6262c4bcda263a537b98221282931cede87c742c','basis_index_sha256':'414f241e986e0c46b7d94820fb060b704166d973284854c53d2e8c64ee6c489e','held_preparation_preserved':True,'files':files,'large_local_records_metadata_only':large,'small_file_count':len(files),'small_bytes':sum(r['bytes'] for r in files),'finite_json_count':finite_json,'passed_actual_double_dual_and_local_gates':True,'passed_release_and_ASan_UBSan':True,'passed_independent_flat_core_full22_physical8_cases_per_build':1404,'passed_tested_J0_J1_J2_angular_closure':True,'angular_query_rows':96720,'fit_and_heldout_worst_scaled_error':max(angular['worst_scaled_errors'].values()),'source_sha256':sha(P/'bridge.cpp'),'report_sha256':sha(P/'REPORT.md'),'accepted_angular_report_sha256':sha(P/'angular-analysis-002.json'),'warnings_first_analysis_preserved':True,'warning_free_same_batch_control_passed':True,'no_radial_operator_boundary_evolution_or_stability_admission':True,'scope':'Local continuum C0 spatialnorm angular coefficients only at positive Omega. Native Cartesian anisotropic stencils/ghosts not reproduced by a fixed-J radial control.'}
(DEST/'index.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps({'index':str(DEST/'index.json'),'sha256':sha(DEST/'index.json'),'small_files':len(files),'small_bytes':index['small_bytes'],'finite_json':finite_json,'large_metadata_records':len(large),'passed':True},indent=2))
