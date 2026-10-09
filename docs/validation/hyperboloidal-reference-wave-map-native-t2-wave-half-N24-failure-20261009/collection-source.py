"""One-shot byte-exact preservation of the terminated wave-map-half-N24-large-t2 run."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path('/Users/hz0693/research/hyperboloidal')
LONG=ROOT/'build-layer-research/wave-map-native-t2-root-20261009'
CASE=LONG/'batch001/wave-map-half-N24-large-t2'
DEST=ROOT/'docs/validation/hyperboloidal-reference-wave-map-native-t2-wave-half-N24-failure-20261009'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
launch=load(CASE/'launch-receipt.json')
assert launch['returncode']==-6 and launch['passed_native_process_and_provenance'] is False
assert launch['sources_before_after_equal'] is True
assert (CASE/'protected-inputs-before.json').read_bytes()==(CASE/'protected-inputs-after.json').read_bytes()
inventory={}
for p in sorted(CASE.rglob('*')):
    if p.is_file():inventory[str(p.relative_to(CASE))]={'source':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
for rel,item in launch['outputs'].items():
    got=inventory['output/'+rel]
    assert got['bytes']==item['bytes'] and got['sha256']==item['sha256']
assert inventory['native.stdout']['sha256']==launch['run_log_sha256']
assert inventory['native.stderr']['sha256']==launch['stderr_sha256']
DEST.mkdir(parents=True,exist_ok=False)
copied={};omitted={}
sources={name:Path(item['source']) for name,item in inventory.items()}
sources.update({'execution-release.json':LONG/'release.json','execution-launcher.py':LONG/'run_preflights.py','original-input.athinput':Path(launch['input_path']),'collection-source.py':Path(__file__).resolve()})
for rel,p in sorted(sources.items()):
    meta={'source':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
    if p.suffix.lower() in ['.bin','.rst','.npy','.npz','.jsonl','.o','.a','.dylib','.so'] or meta['bytes']>1024*1024:
        omitted[rel]=dict(meta,role='large_payload');continue
    content=p.read_bytes();content.decode('utf-8')
    q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(content);copied[rel]=meta
for rel,item in inventory.items():
    p=Path(item['source']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
for item in list(copied.values())+list(omitted.values()):
    p=Path(item['source']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
catalog={'case':'wave-map-half-N24-large-t2','scope':'Terminated native process provenance only; partial outputs are not completed evolution acceptance.','copied':copied,'omitted_payloads':omitted}
dump(DEST/'catalog.json',catalog)
receipt={'passed_failure_preservation':True,'original_native_returncode':-6,'original_process_passed':False,'copied_files':len(copied),'omitted_files':len(omitted),'original_case_files':len(inventory),'all_originals_rehashed_before_after':True,'catalog_sha256':sha(DEST/'catalog.json'),'launch_receipt_sha256':sha(CASE/'launch-receipt.json'),'native_stderr_sha256':sha(CASE/'native.stderr'),'source_sha256':sha(__file__),'new_scientific_queries':0,'new_native_steps':0}
dump(DEST/'preservation-receipt.json',receipt)
(DEST/'README.md').write_text('# Failed half-timestep N24 physical-reference wave-map run\n\nThe original native process terminates with return code -6 at mesh time\n0.79280598958281545, cycle 24355. Its stderr reports an invalid physical ADM\nstate, with negative lapse and chi at xyz=(.1375,-.9625,-.2291666666666667),\nOmega=.0021701388888885099. Exact values remain in native.stderr.\nThe original command, executable/source/input binding, stdout hash, stderr,\nbefore/after protected-input identities, output inventory and native history\nare preserved. Source hashes stayed equal; the native process gate failed.\n\nAll copied sources/logs/receipts are byte-exact. Restart/visualization arrays\nand any >1MiB console log are hash/size/origin metadata only. This capsule\ndoes not replay arrays or perform a partial-field analysis. It does not turn\nthe failed t2 run into completed acceptance or identify a PDE/boundary cause.\nOther cases in the fixed t2 matrix remain independently recorded.\n')
print(json.dumps(receipt))
