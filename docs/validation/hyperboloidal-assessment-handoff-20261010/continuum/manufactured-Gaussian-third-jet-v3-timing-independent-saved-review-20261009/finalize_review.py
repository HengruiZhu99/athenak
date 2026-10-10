#!/usr/bin/env python3
"""Finalize compact additive saved review; no candidate imports or payload decode."""
from pathlib import Path
import hashlib,json,os,sys
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def require(v,s):
    if not v:raise ValueError(s)
def write(name,x):
    with (HERE/name).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
require(sys.flags.isolated==1 and sys.flags.dont_write_bytecode==1 and sys.flags.optimize==0 and os.environ.get('PYTHONOPTIMIZE')=='0','isolated stdlib route')
capture=load(HERE/'capture.json');readback=load(HERE/'saved-readback.json')
for row in capture['inputs']:
    for p in (Path(row['origin']),HERE/row['copy']):require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'original/copy drift')
for row in capture['external_metadata_only']:
    p=Path(row['origin']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'external metadata drift')
require(readback['passed'] is True and readback['inputs_unchanged'] is True and readback['scientific_timing_passed'] is True,'successful saved audit')
require((HERE/'review-v2.stderr').stat().st_size==0,'successful reader empty stderr')
env={k:os.environ.get(k) for k in ('PYTHONOPTIMIZE','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
write('commands.json',{'environment':env,'interpreter':sys.executable,'interpreter_resolved':str(Path(sys.executable).resolve()),'interpreter_sha256':sha(Path(sys.executable).resolve()),'commands':[[sys.executable,'-I','-B',str(HERE/'capture_inputs.py')],[sys.executable,'-I','-B',str(HERE/'review_saved_timing.py')],[sys.executable,'-I','-B',str(HERE/'review_saved_timing_v2.py')],[sys.executable,'-I','-B',str(Path(__file__).resolve())]],'successful_reader_returncode':0,'first_reader_returncode':1,'successful_reader_stdout':'review-v2.stdout','successful_reader_stderr':'review-v2.stderr','shell_preparation_read_failures':[{'command':'rg -n registry|def scientific|timing|progress|identity|precision_checks .../driver.py','reason':'actual indexed driver is run_oracle.py; path did not exist'},{'command':'cat .../inputs/031-timing001-result.json','reason':'capture names are enumerated; actual captured result is042-timing001-result.json'}],'metadata_only':True,'no_candidate_import_or_target_rerun':True})
write('receipt.json',{**readback,'source_review_only':False,'review_kind':'independent compact saved-output provenance/count audit plus empirical cost assessment','reviewed_source_index_sha256':readback['source_index_sha256'],'timing_receipt_sha256':readback['child_receipt_sha256'],'timing_result_sha256':readback['result_sha256'],'full_stage_source_review_passed':True,'full_stage_cost_admission':False,'cost_decision_reserved_to_root':True,'cost_assessment_sha256':sha(HERE/'cost-assessment.json'),'REVIEW_sha256':sha(HERE/'REVIEW.md'),'commands_sha256':sha(HERE/'commands.json'),'reader_v1_failure_preserved':True,'reader_v2_diff_sha256':sha(HERE/'reader-schema-only.diff'),'original_owner_root_and_prior_failure_bytes_unchanged':True,'executed_target_or_candidate':False})
files=[]
for p in sorted(HERE.rglob('*')):
    if p.is_file() and p!=HERE/'index.json':
        require(p.stat().st_size<=1048576 and p.suffix not in ('.jsonl','.npz','.npy'),'compact policy')
        if p.suffix=='.json':load(p)
        files.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)})
write('index.json',{'schema':'independent-saved-review-v1','source_only_review_execution':False,'saved_only':True,'passed':True,'file_count':len(files),'bytes':sum(x['bytes'] for x in files),'files':files,'external_payloads_metadata_only':capture['external_metadata_only'],'source_index_sha256':readback['source_index_sha256'],'full_stage_admitted':False})
print(json.dumps({'passed':True,'index_sha256':sha(HERE/'index.json'),'receipt_sha256':sha(HERE/'receipt.json'),'cost_sha256':sha(HERE/'cost-assessment.json'),'files':len(files),'bytes':sum(x['bytes'] for x in files)}))
