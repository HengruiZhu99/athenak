"""One-shot compact copy of completed independent document review only."""
from pathlib import Path
import hashlib,json,shutil
R=Path('/Users/hz0693/research/hyperboloidal');S=R/'build-layer-research/boundary/reference-wave-map-final9-document-review-20261009';D=R/'docs/validation/hyperboloidal-reference-wave-map-final-matrix-document-review-20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=['receipt.json','review001.stdout','review_saved.py','PLAN.md','review001.stderr','inputs-before.json','numbers.json','recipe.json','invocation001-receipt.json','inputs-after.json']
assert {p.name for p in S.iterdir() if p.is_file()}==set(names)
x=json.loads((S/'receipt.json').read_text());assert x['passed_independent_saved_number_source_scope_review'] is True and x['checked_claims']==58 and x['source_inputs']==45 and x['inputs_unchanged'] is True and x['corrections']==[]
assert sha(S/'receipt.json')=='7a170b234880137510821d9fff6aae8c485250cba8e1233b2f2444f7c7cb26a0';assert sha(R/'docs/hyperboloidal-reference-wave-map-final-matrix.md')==x['draft_sha256']
inv=json.loads((S/'invocation001-receipt.json').read_text());assert inv['returncode']==0
originals={n:{'source':str(S/n),'bytes':(S/n).stat().st_size,'sha256':sha(S/n)} for n in names};source=Path(__file__).resolve();originals['collection-source.py']={'source':str(source),'bytes':source.stat().st_size,'sha256':sha(source)}
assert not D.exists();D.mkdir()
for n,m in originals.items():
 assert m['bytes']<=1048576;shutil.copyfile(m['source'],D/n);assert sha(D/n)==m['sha256']
catalog={'scope':'Completed independent final-matrix document review only; no new scientific calls','files':originals,'external_final_document':{'path':str(R/'docs/hyperboloidal-reference-wave-map-final-matrix.md'),'sha256':x['draft_sha256']},'all_original_bytes_preserved':True}
(D/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n');(D/'README.md').write_text('# Final native-matrix document review\n\nAll 58 saved-data/text checks passed, with 45 unchanged inputs and no corrections. No scientific query, array decoding or native advance occurred. Exact source, logs and failure-free review receipts are retained. The current final document is an external byte identity, not a duplicate frozen document.\n')
for n,m in originals.items():assert sha(Path(m['source']))==m['sha256'] and sha(D/n)==m['sha256']
(D/'collection-receipt.json').write_text(json.dumps({'completed':True,'new_scientific_calls':0,'copied_original_files':len(originals),'sources_unchanged':True,'catalog_sha256':sha(D/'catalog.json')},indent=2)+'\n');print('completed document-review capsule',sha(D/'catalog.json'))
