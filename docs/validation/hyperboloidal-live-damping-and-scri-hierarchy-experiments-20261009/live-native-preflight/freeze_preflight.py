"""One-shot freeze of completed live-damping reference/short native gates."""
import hashlib, importlib.util, json, shutil, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'immutable-native-live-damping-preflight-20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not OUT.exists()
spec=importlib.util.spec_from_file_location('fresh_lapse_auditor',HERE/'audit_live_native.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
build=audit.verify_build();audit.load_helper()
receipts={}
for name in ['reference','short']:
    r=json.loads((HERE/'audit'/f'{name}.json').read_text());launch=json.loads((HERE/f'{name}-launch.json').read_text())
    assert r['status']=='PASS' and r['build_verification']==build
    assert r['auditor_sha256']==sha(HERE/'audit_live_native.py')
    assert launch['native_build_receipt_sha256']==audit.BUILD_RECEIPT
    assert launch['native_executable_sha256']==audit.PRIVATE_EXE
    assert launch['launch_script_sha256']==sha(HERE/'launch_preflight.py')
    assert launch['live_gate_index_sha256']==audit.LIVE_INDEX
    assert launch['override_sha256']==sha(HERE/'native.athinput')
    for path,record in r['all_run_files'].items():
        assert sha(ROOT/path)==record['sha256'] and (ROOT/path).stat().st_size==record['bytes']
    cmd=launch['command']
    assert Path(cmd[2])==HERE/'native-build/athena-live-damping-c0'
    assert Path(cmd[3])==HERE/name
    assert float(cmd[cmd.index('--duration')+1])==r['exact_final_comparison_time']
    receipts[name]=r
assert json.loads((HERE/'short-launch.json').read_text())['reference_audit_sha256']==sha(HERE/'audit/reference.json')
explicit=['build_live_native.py','launch_preflight.py','audit_live_native.py','freeze_preflight.py','native.athinput',
          'reference-launch.json','short-launch.json','audit-reference.log','audit-short.log',
          'audit/reference.json','audit/short.json','native-build/build-source.py','native-build/build-receipt.json','native-build/build.log']
sources={HERE/name:name for name in explicit}
for p in (HERE/'native-build').glob('attempt-*.json'):sources[p]=str(p.relative_to(HERE))
for folder in ['native-build/include','kappa-audit']:
    for p in (HERE/folder).rglob('*'):
        if p.is_file() and p.suffix!='.raw' and p.name!='check_snapshot':sources[p]=str(p.relative_to(HERE))
sources[audit.FAMILY/'native-build-receipt.json']='dependencies/original-norm-native-build-receipt.json'
large=set((HERE/'native-build').glob('*.o'))|set((HERE/'native-build').glob('*.d'))|{HERE/'native-build/athena-live-damping-c0'}
for name in ['reference','short']:
    for p in (HERE/name).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.suffix in ['.rst','.bin'] or p.name=='athena-validation':large.add(p)
        else:sources[p]=str(p.relative_to(HERE))
kr=HERE/'kappa-audit/snapshot-results.json'
assert sha(kr)=='79e28da214ef1fb3d9bda186ab10cdcae3bc7f904a6865d3e44d11cdd30f1e4a'
kappa=json.loads(kr.read_text());assert kappa['status']=='PASS' and len(kappa['rows'])==6
assert kappa['private_build_receipt_sha256']==audit.BUILD_RECEIPT
assert kappa['utility_executable_sha256']==sha(HERE/'kappa-audit/check_snapshot')
for row in kappa['rows']:
    assert row['returncode']==0 and row['stderr']==''
    assert sha(ROOT/row['restart'])==row['restart_sha256']
    assert sha(ROOT/row['raw_file'])==row['raw_sha256']
    assert row['within_flat_frozen_coefficient_interval']
large.update((HERE/'kappa-audit').glob('*.raw'));large.add(HERE/'kappa-audit/check_snapshot')
review=ROOT/'build-layer-research/continuum/live-damping-native-review/receipt.json'
assert sha(review)=='abfc2b30d8a1acae89ad23fcd44de9a026af5e82b5e384468d35baaf3a389dd3'
sources[review]='dependencies/independent-native-recipe-review.json'
OUT.mkdir();files={}
for source,name in sorted(sources.items()):
    target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assert source.read_bytes()==target.read_bytes()
    files[name]={'source':str(source.relative_to(ROOT)),'bytes':target.stat().st_size,'sha256':sha(target)}
report=OUT/'REPORT.json'
report.write_text(json.dumps({'scope':'Finite-Omega live-damping reference/short integrity and six sampled coefficient bounds only; no stable pulse, puncture, scri or BH acceptance.',
    'freeze_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'build':build,'source_operation':'Only helper include and four live/reference evolution/diagnostic kappa2 arguments; removal restores original full header exactly.',
    'reference_max_drift':max(x['full_precision_drift_from_initial_max'] for x in receipts['reference']['private_snapshots']),
    'short_comparisons':receipts['short']['original_HST_diagnostics'],
    'live_coefficient_audit_sha256':sha(kr),
    'all_six_sampled_snapshots_within_flat_interval':True,
    'kappa2_range':[-.19715077315037724,0],
    'caveats':['Original physical-P/spatial-norm gauge and kappa1/reference normalization unchanged.',
               'V(.15,.3) equals one beyond .3; sigma cancellation is live algebraic identity.',
               'Initial interval proof and six sampled snapshot bounds do not establish evolution preservation or a stability/energy bound.',
               'Historical physical_metric_eigen keys are Penrose gtilde/chi magnitudes; SPD positivity equivalent for Omega>0.']},indent=2,allow_nan=False)+'\n')
files[report.name]={'bytes':report.stat().st_size,'sha256':sha(report)}
index={'immutable':True,'native_global_or_scri_stability_accepted':False,'small_files':files,
       'large_files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(large)},
       'live_gate_index_sha256':audit.LIVE_INDEX,'build_receipt_sha256':audit.BUILD_RECEIPT,
       'reference_audit_sha256':sha(HERE/'audit/reference.json'),'short_audit_sha256':sha(HERE/'audit/short.json')}
(OUT/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'large_hashes':len(large),'index_sha256':sha(OUT/'index.json')}))
