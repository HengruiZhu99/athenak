#!/usr/bin/env python3
"""One-shot stdlib-only source correction/pin preparation. No science."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/hz0693/research/hyperboloidal')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source001-held-20261009'
OUTER=ROOT/'build-layer-research/inner-joint-nonlinear-source001-root-release-20261009'
REVIEW=ROOT/'build-layer-research/continuum/inner-joint-nonlinear-helper-source001-independent-review-20261009'

def pin(path):
    path=Path(path).resolve(); data=path.read_bytes()
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
def read(path): return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path)
    if path.exists(): raise RuntimeError('Refuse overwrite '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
def text_new(path,text):
    path=Path(path)
    if path.exists(): raise RuntimeError('Refuse overwrite '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
def copy_new(source,target):
    if target.exists(): raise RuntimeError('Refuse overwrite '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(source.read_bytes())
def main():
    if (HERE/'source-index.json').exists(): raise RuntimeError('Source002 already indexed')
    if pin(OLD/'source-index.json')['sha256']!='6b10f2625f831843745b662a18726c4587701f681f155dc858bfe6b230f225b4':
        raise RuntimeError('Original source index drift')
    original=read(OLD/'source-index.json'); protected={}
    def add(row):
        actual=pin(row['path'])
        if actual['sha256']!=row['sha256'] or ('bytes' in row and actual['bytes']!=row['bytes']):
            raise RuntimeError('Protected input drift '+row['path'])
        protected[actual['path']]=actual
    for row in read(OLD/'input-pins.json')+original['files']+[pin(OLD/'source-index.json')]: add(row)
    if pin(OLD/'attempts/Release001/receipt.json')['sha256']!='e965eda26ff716e62d30764a210953550572eca452b06efd542e045502eaead6':
        raise RuntimeError('Original failed receipt drift')
    if pin(OLD/'attempts/Release001/compile.stderr')['sha256']!='27e417a29bda31d849a6f189bbbe7563b4ff3dd6984496d9a8661ea274919fdd':
        raise RuntimeError('Original compiler stderr drift')
    failure=read(OLD/'attempts/Release001/receipt.json')
    if failure['completed'] or failure['passed'] or failure['returncode']!=1 or not failure['source_inputs_unchanged']:
        raise RuntimeError('Expected preserved pre-query compile failure')
    if [c['name'] for c in failure['commands']]!=['compiler-version','launch-HEAD','python-version','compile']:
        raise RuntimeError('Unexpected original command scope')
    outer=read(OUTER/'release-invocation001/receipt.json')
    if outer['completed'] or outer['accepted_local_gate'] or outer['returncode']!=1 or not outer['inputs_unchanged']:
        raise RuntimeError('Expected preserved outer failure')
    # Read and protect the completed original source review, without treating it as a compile result.
    if pin(REVIEW/'index.json')['sha256']!='c23f20097718f0e0486bb61a53dfa3a7dac47344b5549573157417e3f8499cae':
        raise RuntimeError('Original independent source review drift')
    for tree in [OUTER,REVIEW,OLD/'attempts/Release001']:
        for path in sorted(tree.rglob('*')):
            if path.is_file(): add(pin(path))
    # Copy exactly the 25 indexed files. Generated metadata is rebound below.
    replace={'probe.cpp','recipe.json','input-pins.json','authorization-schema.json','source-preparation.json'}
    for row in original['files']:
        source=Path(row['path']); relative=source.relative_to(OLD)
        if str(relative) not in replace: copy_new(source,HERE/relative)
    old_probe=(OLD/'probe.cpp').read_text()
    before='const auto qp=Parts(q),fp=Fields(f);'
    after='const auto qp=Parts(q);const auto fp=Fields(f);'
    if old_probe.count(before)!=1: raise RuntimeError('Declaration correction is not unique')
    new_probe=old_probe.replace(before,after)
    text_new(HERE/'probe.cpp',new_probe)
    if new_probe.replace(after,before)!=old_probe: raise RuntimeError('Probe changed beyond declaration split')
    recipe=read(OLD/'recipe.json')
    # Only private absolute prefix references are rebound; every scientific setting is retained.
    recipe=json.loads(json.dumps(recipe).replace(str(OLD),str(HERE)))
    recipe['prior_compile_failure_receipt']=pin(OLD/'attempts/Release001/receipt.json')
    recipe['prior_compile_failure_stderr']=pin(OLD/'attempts/Release001/compile.stderr')
    recipe['prior_source_index']=pin(OLD/'source-index.json')
    recipe['correction_scope']='one C++ auto declaration split; mechanical fresh-prefix/pin/failure-history rebinding only'
    write(HERE/'recipe.json',recipe)
    auth=read(OLD/'authorization-schema.json'); auth['recipe_sha256']=pin(HERE/'recipe.json')['sha256']
    write(HERE/'authorization-schema.json',auth)
    write(HERE/'input-pins.json',sorted(protected.values(),key=lambda row:row['path']))
    # Exact history copies are additive and the original sources/attempt remain untouched.
    for source,target in [
        (OLD/'source-index.json',HERE/'history/source001-source-index.json'),
        (OLD/'attempts/Release001/receipt.json',HERE/'history/source001-Release001-receipt.json'),
        (OLD/'attempts/Release001/compile.stderr',HERE/'history/source001-Release001-compile.stderr'),
        (OUTER/'release-invocation001/receipt.json',HERE/'history/root-source001-outer-receipt.json'),
        (OUTER/'release-invocation001/stdout.log',HERE/'history/root-source001-outer.stdout'),
        (OUTER/'release-invocation001/stderr.log',HERE/'history/root-source001-outer.stderr')]: copy_new(source,target)
    diff=[]
    for name in ['probe.cpp','recipe.json','authorization-schema.json']:
        diff.extend(difflib.unified_diff((OLD/name).read_text().splitlines(True),(HERE/name).read_text().splitlines(True),fromfile='source001/'+name,tofile='source002/'+name))
    text_new(HERE/'source001-to-source002.diff',''.join(diff))
    write(HERE/'failure-history.json',{
        'source001_compile_failed_before_any_scientific_query':True,
        'original_child_receipt':pin(OLD/'attempts/Release001/receipt.json'),
        'original_compile_stderr':pin(OLD/'attempts/Release001/compile.stderr'),
        'original_root_outer_receipt':pin(OUTER/'release-invocation001/receipt.json'),
        'original_source_review_is_not_a_compilation_gate':True,
        'original_source_review_index':pin(REVIEW/'index.json'),
        'original_sources_attempt_dependencies_logs_and_outer_tree_protected':True,
        'original_debug_execution_not_performed':True,
        'no_failed_compilation_relabelled_as_scientific_gate_failure':True})
    unchanged={}
    for row in original['files']:
        relative=Path(row['path']).relative_to(OLD)
        if str(relative) not in replace:
            current=pin(HERE/relative)
            if current['sha256']!=row['sha256']: raise RuntimeError('Copied byte drift '+str(relative))
            unchanged[str(relative)]=current['sha256']
    write(HERE/'source-preparation.json',{
        'source_only':True,'execution_admitted':False,
        'protected_inputs':len(protected),'protected_inputs_unchanged':True,
        'original_source001_files_unchanged':True,
        'probe_only_scientific_source_diff':'split auto qp=array8 and fp=array4 into separate declarations',
        'helper_oracle_runner_case_thresholds_frozen_RWM_and_held_plan_unchanged':True,
        'byte_identical_copied_files':unchanged,
        'no_scientific_import_syntax_compile_query_array_load_execution':True,
        'preparation_script_scope':'stdlib-only copy/hash/JSON/diff; original prepare_source.py retained byte-identical as history, not executed'})
    for row in list(protected.values()): add(row)
    files=[pin(path) for path in sorted(HERE.rglob('*')) if path.is_file()]
    write(HERE/'source-index.json',{'source_only':True,'execution_admitted':False,
        'scope':'uncompiled private source002 declaration-only correction with preserved source001 compile failure',
        'files':files,'file_count':len(files),'protected_inputs':len(protected),'inputs_unchanged':True,
        'held_plan_sha256':original['held_plan_sha256']})
    print(json.dumps({'source_index':pin(HERE/'source-index.json'),'recipe':pin(HERE/'recipe.json'),
        'probe':pin(HERE/'probe.cpp'),'diff':pin(HERE/'source001-to-source002.diff'),
        'files':len(files),'protected_inputs':len(protected),'execution_admitted':False},sort_keys=True))

if __name__=='__main__': main()
