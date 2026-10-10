"""Strict standard-library admission and metadata; no arithmetic imports."""
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
ROOT=P.parents[2]


def require(condition,message):
    if not condition: raise RuntimeError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): value.update(block)
    return value.hexdigest()


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def check(entry):
    path=Path(entry['path'])
    require(path.is_file() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],
            'input drift: '+str(path))


def load(path):
    def reject(value): raise ValueError('nonfinite JSON token: '+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def reviewed_index_pins(index_path):
    index_path=Path(index_path).resolve()
    entries=load(index_path)['files']
    if isinstance(entries,dict):
        entries=[dict(value,path=value.get('path',name)) for name,value in entries.items()]
    require(isinstance(entries,list),'review index files must be list or pin mapping')
    result=[]
    for entry in entries:
        path=Path(entry['path'])
        path=(path if path.is_absolute() else index_path.parent/path).resolve()
        require(path.is_relative_to(index_path.parent),'review index path escapes its capsule')
        item=dict(entry,path=str(path));check(item);result.append(item)
    return result


def admit(recipe_path,authorization_path):
    require(sys.flags.isolated==1 and sys.dont_write_bytecode and sys.flags.optimize==0,
            'require -I -B and optimization0')
    require(Path(recipe_path).resolve()==P/'recipe.json','only exact local recipe admitted')
    recipe,index,auth=load(P/'recipe.json'),load(P/'source-index.json'),load(authorization_path)
    require(auth.get('execution_released') is True and auth.get('scope')==recipe['scope']
            and auth.get('source_index_sha256')==sha(P/'source-index.json')
            and auth.get('recipe_sha256')==sha(P/'recipe.json')
            and auth.get('attempt')==str(P/'attempts/units001'),'exact root release absent')
    require(auth.get('source_review_passed') is True,'explicit passed driver/source review required')
    protected=index['files']+load(P/'external-pins.json')+[
        auth['review_receipt'],auth['review_index'],pin(P/'source-index.json'),pin(authorization_path)]
    protected+=reviewed_index_pins(auth['review_index']['path'])
    for entry in protected: check(entry)
    review=load(auth['review_receipt']['path'])
    require((review.get('passed') is True or review.get('passed_source_review') is True)
            and review.get('reviewed_source_index_sha256')==sha(P/'source-index.json'),
            'independent driver review does not pass/bind exact candidate')
    for key,value in recipe['environment'].items(): require(os.environ.get(key)==value,'environment: '+key)
    for key in recipe['sanitized_environment']: require(key not in os.environ,'forbidden environment: '+key)
    require(str(Path(sys.executable).resolve())==recipe['python']['path'],'interpreter identity')
    check(recipe['python'])
    compiler=recipe['compiler']
    require(compiler['path']=='/Library/Developer/CommandLineTools/usr/bin/clang++','literal clang++ required')
    require(str(Path(compiler['path']).resolve())==compiler['resolved_path'],'compiler resolved target drift')
    require(sha(compiler['path'])==compiler['sha256'] and sha(compiler['resolved_path'])==compiler['resolved_sha256'],
            'compiler invocation/target bytes drift')
    require(recipe['fixed_cases']==82 and recipe['base_cases']==74 and recipe['carry_range_cases']==8
            and recipe['command_timeout_seconds']==60 and recipe['root_process_group_cap_seconds']==120,
            'fixed count/resource contract mismatch')
    old=load(recipe['completed_signed_product_receipt']['path'])
    require(old.get('completed') is True and old.get('passed') is True and old.get('returncode')==0
            and old.get('inputs_unchanged') is True and old.get('fixed_cases')==70,
            'closed baseline actual old units not PASS')
    source_review=load(recipe['backend_source_review']['path'])
    require(source_review.get('passed') is True and source_review.get('source_math_review_passed') is True,
            'backend pencil/source prerequisite not PASS')
    return recipe,auth,protected
