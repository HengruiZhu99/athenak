"""Stdlib-only exact source and successful actual-main admission guards."""
import hashlib,json
from pathlib import Path

def pin(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return {'path':str(path),'sha256':h.hexdigest(),'bytes':path.stat().st_size}
def read(path):return json.loads(Path(path).read_text())
def write_new(path,obj):
    with Path(path).open('x') as stream:stream.write(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def guard(rows):
    for row in rows:
        if pin(row['path'])!=row:raise RuntimeError('Protected input drift '+row['path'])
def admitted(here,authorization,build):
    recipe=read(here/'recipe.json');index=read(here/'source-index.json');auth=read(authorization)
    if not(auth.get('arithmetic_supplement_execution_admitted') is True and auth.get('recipe_sha256')==pin(here/'recipe.json')['sha256'] and auth.get('source_index_sha256')==pin(here/'source-index.json')['sha256'] and build in auth.get('allowed_builds',[])):
        raise RuntimeError('Exact fresh root supplement release required')
    protected=read(here/'input-pins.json');guard(protected);guard(index['files'])
    main_pin=auth['actual_main003_release_receipt']
    if main_pin['path']!=recipe['actual_main003_release_receipt'] or pin(main_pin['path'])!=main_pin:
        raise RuntimeError('Exact actual source003 Release receipt binding required')
    main=read(main_pin['path'])
    if not(main.get('completed') is True and main.get('passed') is True and main.get('returncode')==0 and main.get('source_inputs_unchanged') is True and main.get('build')=='release' and main.get('source_index_sha256')==recipe['main003_source_index_sha256']):
        raise RuntimeError('Actual main source003 Release PASS is mandatory')
    oracle_pin=main['oracle_report']
    guard([oracle_pin]);oracle=read(oracle_pin['path'])
    if oracle.get('passed') is not True:raise RuntimeError('Main actual oracle PASS required')
    protected=protected+[main_pin,oracle_pin,pin(authorization)]+main['output_inventory']
    guard(protected)
    return recipe,index,protected
