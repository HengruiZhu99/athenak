"""Prepare one 20-event timing sample after independent saved units review."""
from pathlib import Path
import argparse,hashlib,json
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-index-sha256',required=True);ap.add_argument('--review-receipt-sha256',required=True);a=ap.parse_args()
review=a.review.resolve()
assert sha(review/'index.json')==a.review_index_sha256 and sha(review/'receipt.json')==a.review_receipt_sha256
rv=load(review/'receipt.json');assert rv.get('passed') is True
old=load(HERE/'units-release.json');owner=Path(old['owner']);indexsha='92a412f36e668bb2e2414bd65e5dee381e42534d94e2df712fd7e9f3a52ce5d3'
assert sha(owner/'source-index.json')==indexsha
unit=owner/'attempts/units001/receipt.json';assert sha(unit)=='2cac8cc11cdef7993ea8cb18e2af82d14cfa062d42b8df3c1b0c3ba4e4fed774'
u=load(unit);assert u['completed'] and u['passed'] and u['sources_unchanged'] and u['stage']=='units' and u['source_index_sha256']==indexsha
assert load(unit.parent/'result.json')==dict(passed=True,records=0,checks=318,failed=[],failed_total=0)
root=load(HERE/'units-invocation001/receipt.json');assert root['completed'] and root['accepted_stage'] and root['inputs_unchanged'] and root['root_process_group_cap_seconds']==60 and root['elapsed_seconds']<60
assert rv['reviewed_source_index_sha256']==indexsha and rv['child_receipt_sha256']==sha(unit) and rv['result_sha256']==sha(unit.parent/'result.json')
assert rv['root_receipt_sha256']==sha(HERE/'units-invocation001/receipt.json') and rv['outer_receipt_sha256']==sha(HERE/'units-outer001/receipt.json')
assert rv['saved_only'] is True and rv['checks']==318 and rv['failed_total']==0 and rv['root_process_group_cap_seconds']==60
pins=dict(old['pins']);reviewpins={}
for folder in (review,HERE/'units-invocation001',HERE/'units-outer001',unit.parent):
 for p in folder.rglob('*'):
  if p.is_file():reviewpins[str(p)]=sha(p)
for n in ('units-authorization.json','units-release.json','prepare_timing001.py','launch_timing.py'):
 reviewpins[str(HERE/n)]=sha(HERE/n)
for p,h in reviewpins.items():
 assert p not in pins or pins[p]==h,p
 pins[p]=h
for p,h in pins.items():assert sha(p)==h,p
recipe=load(owner/'recipe.json');assert len(recipe['timing_keys'])==10 and recipe['outer_seconds']['timing']==660 and recipe['stage_seconds']['timing']==600
output=owner/'attempts/timing001';outer=HERE/'timing-outer001';assert not output.exists() and not outer.exists() and not (HERE/'timing-invocation001').exists()
auth={'Gaussian_third_jet_oracle_stage_authorized':'timing','source_index_sha256':indexsha,'recipe_sha256':sha(owner/'recipe.json'),'driver_sha256':sha(owner/'run_oracle.py'),'output':str(output),'outer_output':str(outer),'unit_receipt':{'path':str(unit),'sha256':sha(unit)},'review_pins':reviewpins,'scope':'Only20 fixed timing records,600s scientific/660s outer/690s root group caps. Full stage requires separate actual measured-cost review and release. No native queries.'}
write(HERE/'timing-authorization.json',auth)
write(HERE/'timing-release.json',{'owner':str(owner),'stage':'timing','pins':pins,'authorization_sha256':sha(HERE/'timing-authorization.json'),'output':str(output),'outer_output':str(outer),'independent_unit_review_receipt_sha256':sha(review/'receipt.json')})
print(json.dumps({'prepared':True,'pins':len(pins),'authorization_sha256':sha(HERE/'timing-authorization.json'),'timing_records_only':20}))
