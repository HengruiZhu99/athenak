"""Release exactly reviewed source002 standalone units, no gauge calls."""
from pathlib import Path
import argparse,hashlib,json,sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SRC=REPO/'build-layer-research/continuum/exact-signed-product-sum-source002-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
ap=argparse.ArgumentParser();ap.add_argument('--review',required=True,type=Path);ap.add_argument('--receipt-sha256',required=True);ap.add_argument('--index-sha256',required=True);a=ap.parse_args()
assert sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0
assert sha(a.review/'receipt.json')==a.receipt_sha256 and sha(a.review/'index.json')==a.index_sha256
independent=load(a.review/'receipt.json');assert (independent.get('passed') is True or independent.get('passed_source_review') is True) and independent['reviewed_source_index_sha256']==sha(SRC/'source-index.json')
assert load(HERE/'source002-review.json')['passed_source_review']
pins=load(HERE/'source002-pins.json')
for row in load(a.review/'index.json')['files']:pins[row['path']]=row['sha256']
pins[str((a.review/'receipt.json').resolve())]=a.receipt_sha256;pins[str((a.review/'index.json').resolve())]=a.index_sha256
for p,h in pins.items():assert sha(p)==h,p
r=load(SRC/'recipe.json');review=(a.review/'receipt.json').resolve()
auth={'execution_released':True,'scope':r['scope'],'source_index_sha256':sha(SRC/'source-index.json'),'recipe_sha256':sha(SRC/'recipe.json'),'source_review_passed':True,'review_receipt':{'path':str(review),'bytes':review.stat().st_size,'sha256':a.receipt_sha256},'no_RWM_adoption_or_failure_upgrade':True}
write(HERE/'units-authorization.json',auth)
for name in ('review_source001.py','source001-review.json','review_source002.py','source002-review.json','source002-pins.json','prepare_units001.py','launch_units.py'):pins[str(HERE/name)]=sha(HERE/name)
write(HERE/'units-release.json',{'owner':str(SRC),'pins':pins,'authorization_sha256':sha(HERE/'units-authorization.json'),'independent_index_sha256':a.index_sha256,'independent_receipt_sha256':a.receipt_sha256})
print(json.dumps({'units_released':70,'pins':len(pins),'authorization_sha256':sha(HERE/'units-authorization.json')}))
