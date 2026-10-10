"""Source-only stdlib metadata generation; no candidate or numerical oracle import."""
from pathlib import Path
import ast
import hashlib
import json

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OWNER=BASE/'reference-wave-map-far-dual-source002-held-20261009'
READBACK=BASE/'reference-wave-map-far-dual-Release-failure-saved-review-20261009'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def main():
    for name in ('recipe.json','input-pins.json','source-index.json','source-preparation.json','authorization-schema.json'):
        if (HERE/name).exists():raise RuntimeError('fresh metadata only')
    for name in ('oracle.py','run_once.py','prepare_metadata.py'):ast.parse((HERE/name).read_text())
    expected={OWNER/'source-index.json':'79aa9a75b26687d97da135eacf8efc3b577f52c21726304db0c8f17834b6a1e5',
        OWNER/'attempts/Release001/receipt.json':'839d123c40023f502d96804116c708ad48c02d8d572663ec4afbdc5d728d445d',
        OWNER/'attempts/Release001/oracle-report.json':'b08c324e8f964b10c98ac04218f4ee9174d7b163a2654b8b3e838930994e6ab4',
        READBACK/'index.json':'7b0bd0e5020b47b955c33d138fc7a47b73af892ba6707f49603e47eabe8dc434'}
    for path,digest in expected.items():
        if sha(path)!=digest:raise RuntimeError('fixed evidence changed')
    old=load(OWNER/'recipe.json');pins={row['path']:row['sha256'] for row in load(old['input_pins'])+load(OWNER/'source-index.json')['files']}
    for path in expected:pins[str(path)]=expected[path]
    for row in load(READBACK/'index.json')['files']:pins[row['path']]=row['sha256']
    for name in ('direct.jsonl','closed.jsonl','dependencies.json','dependencies.d','far-dual-probe-release'):
        path=OWNER/'attempts/Release001'/name;pins[str(path)]=sha(path)
    for path,digest in pins.items():
        if sha(path)!=digest:raise RuntimeError('protected input changed: '+path)
    report=load(READBACK/'attempt001/summary.json')
    if not(report['saved_readback_passed'] and report['failure_count']==63 and report['unique_failed_rows']==26):raise RuntimeError('saved association prerequisite missing')
    contexts=load(READBACK/'attempt001/selected-consumed-contexts.json');selected=[row['base_index'] for row in contexts]
    if selected!=[3,7,11,15,19,27,35,42,43,47,50,51,55,59,63,67,71,75,83,91,98,99,103,106,107,111]:raise RuntimeError('fixed26 registry differs')
    recipe={'source_only':True,'execution_admitted':False,'scope':'Fixed26-row observational inverse/flux/pole arithmetic decomposition only',
        'selected_bases':selected,'seed_index':13,'zero_primal_gradients':False,'expected_rows':26,'expected_helper_calls':52,
        'original_source_index':str(OWNER/'source-index.json'),'original_direct_jsonl':str(OWNER/'attempts/Release001/direct.jsonl'),
        'original_oracle_report':str(OWNER/'attempts/Release001/oracle-report.json'),'original_child_receipt':str(OWNER/'attempts/Release001/receipt.json'),
        'saved_failure_association_index':str(READBACK/'index.json'),'compiler':old['compiler'],'compile_flags':old['compile_flags']['release'],
        'python':old['python'],'repository':old['repository'],'environment':old['environment'],
        'attempt':str(HERE/'attempts/diagnostic001'),'thresholds':{'original_saved_MP_target_consistency':'2e-10','input_and_original_output_bits':'exact','intermediate_accuracy':'reported, no pass criterion'},
        'original63_failures_and8zero_labels_preserved':True,'no_main_registry_oracle_helper_changes':True,
        'future_command':[old['python'],'-I','-B',str(HERE/'run_once.py'),'--authorization','ROOT_AUTHORIZATION.json','--authorization-sha256','ROOT_HASH']}
    write(HERE/'recipe.json',recipe);write(HERE/'input-pins.json',pins)
    write(HERE/'authorization-schema.json',{'schema_only':True,'metric_flux_diagnostic_local_admitted':False,'recipe_sha256':'EXACT_REVIEWED','source_index_sha256':'EXACT_REVIEWED','runner_sha256':sha(HERE/'run_once.py')})
    write(HERE/'source-preparation.json',{'source_only':True,'candidate_imported_or_compiled':False,'targets_evaluated':False,'original_payloads_streaming_hash_only':True,'protected_paths':len(pins),'original_helpers_unchanged':True,'exact_failed_registry':selected})
    rows=[{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(HERE.rglob('*')) if path.is_file()]
    if any(row['bytes']>1048576 for row in rows):raise RuntimeError('source compact file cap exceeded')
    write(HERE/'source-index.json',{'source_only':True,'scientific_execution':False,'execution_authorized':False,'files':rows,'file_count':len(rows)})
    print(json.dumps({'source_only_ready':True,'index_sha256':sha(HERE/'source-index.json'),'recipe_sha256':sha(HERE/'recipe.json'),'pins':len(pins)}))

if __name__=='__main__':main()
