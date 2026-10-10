"""Seal the root static findings; no candidate imports or numeric execution."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
pins=json.loads((HERE/'source-pins001.json').read_text())
for p,h in pins.items():assert sha(p)==h,p
result={'source_math_admission_passed':False,'execution_admitted':False,'source_index_sha256':'2348bbf4391444606dc2f946067a3a75eb55d472cc0c738795d8f0617f161d88','protected_inputs':len(pins),'inputs_unchanged':True,'candidate_imported':False,'numerical_failures_observed':0,'findings':['physical versus conformal reference connection normalization','nominal rational radius admission must not depend on rounded reconstructed norm','full stage requires separately pinned measured timing review'],'review_sha256':sha(HERE/'REVIEW-source001.md')}
with (HERE/'source-review001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
