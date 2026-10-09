"""Standard-library saved-only public-document factual readback; no science calls."""
from pathlib import Path
from decimal import Decimal
import hashlib,json
P=Path(__file__).resolve().parent;R=P.parents[2]
A=R/'docs/validation/hyperboloidal-derivative-pilots-inner-pencil-20261009'
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
def load(q):return json.loads(q.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
inputs=load(P/'inputs-before-review.json')['inputs']
for row in inputs:
 if row is not inputs[0]:assert sha(Path(row['source']))==row['sha256']
 assert sha(Path(row['copy']))==row['sha256']
doc=(P/'inputs/00-hyperboloidal-derivative-pilots-inner-gauge.md').read_text()
cat=load(P/'inputs/01-catalog.json')
assert sha(A/'catalog.json')=='79b850b690bb21b17898506b932370e1f48fa630b0023e0760de64243c4a1b5f'
for name,row in cat['files'].items():
 q=A/name;assert sha(q)==row['sha256'];assert q.stat().st_size==row['bytes']
files=sorted(q for q in A.rglob('*') if q.is_file());finite=0
for q in files:
 b=q.read_bytes();assert len(b)<=1048576;b.decode('utf-8')
 assert q.suffix.lower() not in {'.npz','.npy','.jsonl','.exe','.o','.a','.bin','.rst'}
 if q.suffix=='.json':load(q);finite+=1
assert (len(files),sum(q.stat().st_size for q in files),finite)==(287,6100522,152)
assert len(cat['omitted_large_payloads'])==7
external=0
for rel in cat['external_dependency_parts']:
 x=load(A/rel)
 for z in x['inputs']:
  q=Path(z['source']);assert sha(q)==z['sha256'];assert q.stat().st_size==z['bytes'];external+=1
assert external==315
old=load(P/'inputs/04-receipt.json');checks=load(P/'inputs/05-checks.json')
new=load(P/'inputs/06-receipt.json');read=load(P/'inputs/07-saved-readback001.json')
assert old['passed_tiny_timing_identity_gate'] and new['passed_compact_root_comparison_gate']
assert old['ray_rows']==new['ray_rows']==480 and old['group_rows']==new['group_rows']==24
assert len(checks)==4744 and all(z['passed'] is True for z in checks)
assert new['checks']==38344 and old['sources_unchanged'] and new['sources_unchanged']
assert old['seconds']==193.820702625 and new['seconds']==18.980866792
assert read['passed_independent_saved_decimal_readback'] and read['ray_rows']==480
assert read['jet_components_independently_compared']==28800
assert read['methods']=={'initial_exact':160,'compact_safeguarded_Newton':160,'outer_exact':160}
assert Decimal(read['maximum_scaled_jet_difference'])==Decimal('3.8369807166e-50')
assert Decimal(read['maximum_scaled_metric_difference'])<Decimal('1.065114715835e-50')
assert Decimal(read['maximum_original_root_residual'])<Decimal('5.649684602436e-52')
run=load(P/'inputs/10-receipt.json');ind=load(P/'inputs/11-receipt.json')
a=load(P/'inputs/12-analyze-release.stdout');b=load(P/'inputs/13-analyze-debug.stdout')
assert a==b and a['passed'] and a['actual20_cases']==118
assert run['passed'] and run['completed'] and run['inputs_unchanged'] and run['release_debug_stdout_byte_equal']
assert run['exact_scalar_summary']['exact_scalar_cases']==18
assert a['maxima']['matrix']==5.346834086594754e-13
assert a['maxima']['basis_inverse']==4.440892098500626e-16
assert a['maxima']['basis_condition_inf']==113.32241771251452
assert ind['passed'] and ind['saved_only'] and ind['independent_matrix_max']==a['maxima']['matrix']
for text in ('480 rays in 24','4,744 checks','38,344 checks','287 files, 6,100,522 bytes and 152','315 external','118 actual-kernel','18 exact rational','5.346834086594754e-13','4.440892098500626e-16','113.32241771251452','wormhole-to-trumpet','Minkowski hyperboloidal reference'):
 assert text in doc,text
out={'saved_only':True,'passed_factual_readback':True,'evidence_inputs_unchanged':True,'document_original_sha256':inputs[0]['sha256'],'document_current_sha256':sha(Path(inputs[0]['source'])),'document_revision_after_capture':'Parent accepted integrand wording correction; original reviewed bytes preserved','catalog_sha256':sha(A/'catalog.json'),'archive_files':287,'archive_bytes':6100522,'finite_JSONs':152,'metadata_only_payloads':7,'external_dependencies':315,'old_checks':4744,'new_checks':38344,'rays':480,'groups':24,'saved_jet_components':28800,'actual20_cases_per_build':118,'exact_scalar_cases':18,'principal_maxima':a['maxima'],'scientific_calls':0,'scope':'Saved JSON/text/hash factual readback; math/source review is separate; no oracle, API, compiler, array, eigen, propagation or evolution execution'}
(P/'saved-factual-readback.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(out,indent=2))
