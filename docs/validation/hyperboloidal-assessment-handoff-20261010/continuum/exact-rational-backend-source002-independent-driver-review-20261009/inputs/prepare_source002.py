"""Metadata-only JSON-registry guard sibling; no candidate import/evaluation."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil,sys
P=Path(__file__).resolve().parent
OLD=P.parent/'exact-rational-backend-source001-held-20261009'
OLDROOT=P.parents[1]/'exact-rational-backend-source001-root-release-20261009'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def pin(path):
    path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def load(path):return json.loads(Path(path).read_text())


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    if not(sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0):raise RuntimeError('metadata flags')
    if (P/'source-index.json').exists():raise RuntimeError('one-shot already frozen')
    if sha(OLD/'source-index.json')!='c703481f9c8da663baf5c894aee2ff285d0e427572dbe990a4467874b9d5e4b5':raise RuntimeError('001 source drift')
    oldindex=load(OLD/'source-index.json');external={}
    for item in oldindex['files']+load(OLD/'external-pins.json'):
        if sha(item['path'])!=item['sha256']:raise RuntimeError('001 protected drift')
        external[item['path']]=item
    history=P/'history/source001';history.mkdir(parents=True)
    for path in sorted(OLD.iterdir()):
        if path.is_file():
            shutil.copyfile(path,history/path.name)
            if path.name!='source-index.json':shutil.copyfile(path,P/path.name)
    external[str(OLD/'source-index.json')]=pin(OLD/'source-index.json')
    for path in sorted(OLDROOT.iterdir()):
        if path.is_file():external[str(path)]=pin(path)
    original=(OLD/'oracle_driver.py').read_text()
    new=original.replace('import argparse\n','import argparse\nimport json\n',1)
    before="load(attempt/'registry.json')==cases"
    after="load(attempt/'registry.json')==json.loads(json.dumps(cases,allow_nan=False))"
    if original.count(before)!=1:raise RuntimeError('nonunique guard anchor')
    new=new.replace(before,after,1)
    (P/'oracle_driver.py').write_text(new)
    reverse=new.replace('import argparse\nimport json\n','import argparse\n',1).replace(after,before,1)
    if reverse!=original:raise RuntimeError('guard reverse bytes')
    recipe=load(OLD/'recipe.json')
    recipe['source001_ineligible_context']=dict(source_index=pin(OLD/'source-index.json'),root_source_index=pin(OLDROOT/'source-index.json'),
        reason='JSON-loaded operation arrays are lists; original regenerated operations are tuples; direct equality is always false',
        compiler_or_registry_executed=False)
    recipe['local_sources']+=['prepare_source002.py','JSON-REGISTRY-ERRATUM.md','guard-reverse-proof.json','source001-to-source002.diff']
    save(P/'recipe.json',recipe)
    save(P/'external-pins.json',sorted(external.values(),key=lambda item:item['path']))
    (P/'source001-to-source002.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='source001/oracle_driver.py',tofile='source002/oracle_driver.py')))
    unchanged=['exact_dyadic_ratio.hpp','probe.cpp','fraction_units.py','carry_range_units.py','gate_common.py','run_gate.py']
    proofs=[]
    for name in unchanged:
        if (OLD/name).read_bytes()!=(P/name).read_bytes():raise RuntimeError('unchanged source differs')
        item=dict(file=name,byte_equal=True)
        if name.endswith('.py'):item['AST_equal']=ast.dump(ast.parse((OLD/name).read_text()),include_attributes=False)==ast.dump(ast.parse((P/name).read_text()),include_attributes=False)
        proofs.append(item)
    save(P/'guard-reverse-proof.json',dict(oracle_guard_reverse_byte_equal=True,
        oracle_guard_reverse_AST_equal=ast.dump(ast.parse(reverse),include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False),
        unchanged_source_proofs=proofs,source001_preserved=True,registry_generation=False,compiler_or_units=False))
    (P/'JSON-REGISTRY-ERRATUM.md').write_text('''Source001 remains unexecuted and ineligible. The original oracle driver compared a JSON-loaded registry (lists for operation arrays) to regenerated Python tuples. That structural equality would fail independently of numerical correctness. Source002 adds only the json import and canonical JSON round-trip of the regenerated registry in this guard. The complete saved case/ID/operand/protocol guard is retained; no comparison or threshold is removed. All three reviewed WIP files, the eight extra carry/range controls, runner, admission and numerical arithmetic are byte-identical. Original001/root001 indexes and files are protected as source-only failure history. No compiler, candidate import, registry, Fraction evaluation or unit has executed.\n''')
    (P/'PLAN.md').write_text((OLD/'PLAN.md').read_text()+'\nSource002 corrects only the JSON-registry structural guard, as documented in JSON-REGISTRY-ERRATUM.md. Source001 is preserved ineligible and unexecuted.\n')
    for path in P.rglob('*.py'):ast.parse(path.read_text(),filename=str(path))
    for item in external.values():
        if sha(item['path'])!=item['sha256']:raise RuntimeError('final protected drift')
    save(P/'source-preparation002-receipt.json',dict(passed=True,source_only=True,metadata_AST_only=True,
        original001_unchanged=True,original001_execution=False,candidate_import_or_registry_Fraction_compiler_units=False,
        external_unique_pins=len(external)))
    files=sorted(path for path in P.rglob('*') if path.is_file() and path!=P/'source-index.json')
    save(P/'source-index.json',dict(source_only=True,execution_admitted=False,files=[pin(path) for path in files],
        external_pins_file='external-pins.json',external_unique_pins=len(external),fixed_cases=82,
        preserved_base_cases=74,new_carry_range_cases=8,source001_preserved_ineligible=True,no_compile_import_registry_or_unit_execution=True))
    print(json.dumps(dict(index=sha(P/'source-index.json'),recipe=sha(P/'recipe.json'),files=len(files),external=len(external))))


if __name__=='__main__':main()
