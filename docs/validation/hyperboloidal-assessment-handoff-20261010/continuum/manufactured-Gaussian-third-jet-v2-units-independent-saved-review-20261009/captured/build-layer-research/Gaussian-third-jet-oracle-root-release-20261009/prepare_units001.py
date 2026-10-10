"""Release only corrected Gaussian analytic units after independent source gate."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
OWNER=BASE/'continuum/manufactured-angular-Gaussian-third-jet-oracle-v2-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-index-sha256',required=True);ap.add_argument('--review-receipt-sha256',required=True);args=ap.parse_args()
review=args.review.resolve();assert sha(review/'index.json')==args.review_index_sha256 and sha(review/'receipt.json')==args.review_receipt_sha256
rv=load(review/'receipt.json');indexsha='92a412f36e668bb2e2414bd65e5dee381e42534d94e2df712fd7e9f3a52ce5d3'
assert rv['passed'] and rv['reviewed_source_index_sha256']==indexsha
assert sha(OWNER/'source-index.json')==indexsha
root=load(HERE/'source-review002.json');assert root['passed'] and root['root_full_source_math_and_admission_review'] and root['source_index_sha256']==indexsha
pins=load(HERE/'source-pins002.json');reviewpins={}
for p in review.rglob('*'):
 if p.is_file():reviewpins[str(p)]=sha(p)
for name in ('source-review001.json','REVIEW-source001.md','source-review002.json','prepare_review002.py','finalize_review002.py','prepare_units001.py','launch_units.py'):
 reviewpins[str(HERE/name)]=sha(HERE/name)
for p,h in reviewpins.items():
 assert p not in pins or pins[p]==h,p
 pins[p]=h
for p,h in pins.items():assert sha(p)==h,p
recipe=load(OWNER/'recipe.json');assert recipe['expected_counts']['units']==318 and recipe['outer_seconds']['units']==120 and recipe['stage_seconds']['units']==60
output=OWNER/'attempts/units001';outer=HERE/'units-outer001';assert not output.exists() and not outer.exists()
auth={'Gaussian_third_jet_oracle_stage_authorized':'units','source_index_sha256':indexsha,'recipe_sha256':sha(OWNER/'recipe.json'),'driver_sha256':sha(OWNER/'run_oracle.py'),'output':str(output),'outer_output':str(outer),'review_pins':reviewpins,'scope':'Only318 corrected-source analytic units; root process-group60s cap additionally enforces nominal unit limit. Timing/full/native queries remain held.'}
write(HERE/'units-authorization.json',auth)
write(HERE/'units-release.json',{'owner':str(OWNER),'stage':'units','pins':pins,'authorization_sha256':sha(HERE/'units-authorization.json'),'output':str(output),'outer_output':str(outer)})
print(json.dumps({'prepared':True,'pins':len(pins),'authorization_sha256':sha(HERE/'units-authorization.json'),'units_only':True}))
