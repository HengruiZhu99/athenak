"""One-shot freeze of completed finite-positive-lapse reference/short native gates."""
import hashlib, importlib.util, json, shutil, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'immutable-native-lapse-preflight-compact-20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not OUT.exists()
spec=importlib.util.spec_from_file_location('fresh_lapse_auditor',HERE/'audit_lapse_native.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
build=audit.verify_build();audit.load_helper()
receipts={}
for name in ['reference','short']:
    r=json.loads((HERE/'audit'/f'{name}.json').read_text());launch=json.loads((HERE/f'{name}-launch.json').read_text())
    assert r['status']=='PASS' and r['build_verification']==build
    assert r['auditor_sha256']==sha(HERE/'audit_lapse_native.py')
    assert launch['native_build_receipt_sha256']==audit.BUILD_RECEIPT
    assert launch['native_executable_sha256']==audit.PRIVATE_EXE
    assert launch['launch_script_sha256']==sha(HERE/'launch_preflight.py')
    assert launch['lapse_gate_index_sha256']==audit.LAPSE_INDEX
    assert launch['override_sha256']==sha(HERE/'native.athinput')
    for path,record in r['all_run_files'].items():
        assert sha(ROOT/path)==record['sha256'] and (ROOT/path).stat().st_size==record['bytes']
    cmd=launch['command']
    assert Path(cmd[2])==HERE/'native-build/athena-inner-lapse-advection'
    assert Path(cmd[3])==HERE/name
    assert float(cmd[cmd.index('--duration')+1])==r['exact_final_comparison_time']
    receipts[name]=r
assert json.loads((HERE/'short-launch.json').read_text())['reference_audit_sha256']==sha(HERE/'audit/reference.json')
explicit=['build_lapse_native.py','launch_preflight.py','audit_lapse_native.py','freeze_preflight_compact.py','native.athinput',
          'reference-launch.json','short-launch.json','audit-reference.log','audit-short.log',
          'audit/reference.json','audit/short.json','native-build/build-source.py','native-build/build-receipt.json','native-build/build.log']
sources={HERE/name:name for name in explicit}
for p in (HERE/'native-build').glob('attempt-*.json'):sources[p]=str(p.relative_to(HERE))
for folder in ['native-build/include','precompile-recipe-syntax-failure','precompile-receipt-key-failure','audit-index-schema-failure']:
    for p in (HERE/folder).rglob('*'):
        if p.is_file():sources[p]=str(p.relative_to(HERE))
sources[audit.FAMILY/'native-build-receipt.json']='dependencies/original-norm-native-build-receipt.json'
large=set((HERE/'native-build').glob('*.o'))|set((HERE/'native-build').glob('*.d'))|{HERE/'native-build/athena-inner-lapse-advection'}
for name in ['reference','short']:
    for p in (HERE/name).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.suffix in ['.rst','.bin'] or p.name=='athena-validation':large.add(p)
        else:sources[p]=str(p.relative_to(HERE))
# Existing run capture contains duplicate staged documentation, not compiled source.
# Preserve its complete manifest/hash metadata while avoiding a second copy of prior archives.
original=HERE/'immutable-native-lapse-preflight-20261009/index.json'
assert sha(original)=='adfe7a0c18b08b1f5754285132ce475db1d10738140ed37ac2e1b22dc483e86d'
omitted={}
for source,name in list(sources.items()):
    if '/source-at-launch/files/docs/' in '/'+name:
        omitted[str(source.relative_to(ROOT))]={'bytes':source.stat().st_size,'sha256':sha(source)}
        del sources[source]
sources[original]='historical-full-copy-index.json'
OUT.mkdir();files={}
for source,name in sorted(sources.items()):
    target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assert source.read_bytes()==target.read_bytes()
    files[name]={'source':str(source.relative_to(ROOT)),'bytes':target.stat().st_size,'sha256':sha(target)}
report=OUT/'REPORT.json'
report.write_text(json.dumps({'scope':'Finite-positive-lapse reference/short integrity only; no stable pulse, puncture, scri or BH acceptance.',
    'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'build':build,'source_operation':'Only helper include plus regular gauge_rhs.alpha addition; removal restores original full header exactly.',
    'reference_max_drift':max(x['full_precision_drift_from_initial_max'] for x in receipts['reference']['private_snapshots']),
    'short_comparisons':receipts['short']['original_HST_diagnostics'],
    'caveats':['Original geometric/gauge poles unchanged.','Additive source can cancel when lapse collapses; future BH implementation requires direct equivalent advection evaluation and fresh gates.',
               'Initial precompile syntax, precompile receipt key and auditor index-schema failures preserved; none altered scientific source or evolution.',
               'Historical physical_metric_eigen keys are Penrose gtilde/chi magnitudes; SPD positivity equivalent for Omega>0.']},indent=2,allow_nan=False)+'\n')
files[report.name]={'bytes':report.stat().st_size,'sha256':sha(report)}
index={'immutable':True,'native_global_or_scri_stability_accepted':False,'small_files':files,
       'omitted_duplicate_documentation_capture_metadata_only':omitted,
       'large_files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(large)},
       'lapse_gate_index_sha256':audit.LAPSE_INDEX,'build_receipt_sha256':audit.BUILD_RECEIPT,
       'reference_audit_sha256':sha(HERE/'audit/reference.json'),'short_audit_sha256':sha(HERE/'audit/short.json')}
(OUT/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'large_hashes':len(large),'index_sha256':sha(OUT/'index.json')}))
