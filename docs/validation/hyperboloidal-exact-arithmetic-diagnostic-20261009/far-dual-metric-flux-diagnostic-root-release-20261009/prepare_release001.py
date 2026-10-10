"""Bind exact independent source receipt before a separate bounded local replay."""
from pathlib import Path
import argparse,hashlib,json,sys
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
ap=argparse.ArgumentParser();ap.add_argument('--review',required=True,type=Path);ap.add_argument('--receipt-sha256',required=True);ap.add_argument('--index-sha256',required=True);args=ap.parse_args()
assert sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0
review=args.review.resolve();assert sha(review/'receipt.json')==args.receipt_sha256 and sha(review/'index.json')==args.index_sha256
independent=load(review/'receipt.json')
assert independent['source_math_admission_passed'] and independent['protected_inputs_before_after_equal']
assert independent['status']=='PASS_SOURCE_ONLY_HELD' and not independent['execution_released']
assert independent['owner_index_sha256']=='3da906bcddd42dafcee590a775ee7c142cf78767784c08fae7deb924134ff6a5'
assert independent['owner_recipe_sha256']=='022a924a8958d0a85d401476272316fe69d1d714dd9180f24f0445a7b3e7241a'
pins=load(HERE/'source-pins001.json')
for row in load(review/'index.json')['files']:pins[row['path']]=row['sha256']
pins[str(review/'receipt.json')]=args.receipt_sha256;pins[str(review/'index.json')]=args.index_sha256
for p,h in pins.items():assert sha(p)==h,p
rootreview=load(HERE/'source-review001.json');assert rootreview['source_review_passed'] and rootreview['original_far_Release_remains_FAILED']
write(HERE/'independent-review-bound001.json',{'review':str(review),'receipt_sha256':args.receipt_sha256,'index_sha256':args.index_sha256,'root_has_read_full_review':True})
SRC=HERE.parents[1]/'build-layer-research/boundary/reference-wave-map-far-dual-metric-flux-diagnostic-held-20261009'
auth={'metric_flux_diagnostic_local_admitted':True,'recipe_sha256':sha(SRC/'recipe.json'),'source_index_sha256':sha(SRC/'source-index.json'),'runner_sha256':sha(SRC/'run_once.py'),'scope':'Exactly26 observational contexts/52 unchanged helper calls; independent Fraction diagnostic only; original63 FAIL immutable; no fix/native/Debug/evolution admission.'}
write(HERE/'authorization.json',auth)
for name in ('source-pins001.json','source-review001.json','prepare_source_review001.py','independent-review-bound001.json','prepare_release001.py','launch.py'):pins[str(HERE/name)]=sha(HERE/name)
write(HERE/'release.json',{'owner':str(SRC),'pins':pins,'authorization_sha256':sha(HERE/'authorization.json'),'review_receipt_sha256':args.receipt_sha256,'review_index_sha256':args.index_sha256})
print(json.dumps({'released_observational_contexts':26,'helper_calls':52,'pins':len(pins),'authorization_sha256':sha(HERE/'authorization.json')}))
