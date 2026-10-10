"""Source/text-only root review receipt; no scientific imports or targets."""
from pathlib import Path
import ast,hashlib,json,sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
SRC=REPO/'build-layer-research/boundary/reference-wave-map-far-dual-metric-flux-diagnostic-held-20261009'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0
assert sha(SRC/'source-index.json')=='3da906bcddd42dafcee590a775ee7c142cf78767784c08fae7deb924134ff6a5'
assert sha(SRC/'recipe.json')=='022a924a8958d0a85d401476272316fe69d1d714dd9180f24f0445a7b3e7241a'
pins=load(SRC/'input-pins.json')
for row in load(SRC/'source-index.json')['files']:pins[row['path']]=row['sha256']
pins[str(SRC/'source-index.json')]=sha(SRC/'source-index.json')
for p,h in pins.items():assert sha(p)==h,p
for name in ('oracle.py','run_once.py','prepare_metadata.py'):ast.parse((SRC/name).read_text())
r=load(SRC/'recipe.json')
assert r['expected_rows']==26 and r['expected_helper_calls']==52 and r['seed_index']==13
assert r['zero_primal_gradients'] is False and r['source_only'] and not r['execution_admitted']
old=load(r['original_child_receipt'])
assert old['completed'] is False and old['passed'] is False and old['returncode']==1 and old['source_inputs_unchanged']
oracle=(SRC/'oracle.py').read_text();probe=(SRC/'probe.cpp').read_text()
assert 'dG' not in probe and '#include "/Users/hz0693/research/hyperboloidal/build-layer-research/boundary/reference-wave-map-far-dual-source002-held-20261009/probe.cpp"' in probe
assert 'assert x-y==(z-y)+(t-z)+(x-t)' in oracle
assert "if not equal_bits(row[key],old[key])" in oracle
assert "failed_labels_checked==63" in oracle
review={
 'source_review_passed':True,'execution_released':False,'source_index_sha256':sha(SRC/'source-index.json'),
 'recipe_sha256':sha(SRC/'recipe.json'),'protected_paths':len(pins),'protected_inputs_unchanged':True,
 'root_full_reads':['PLAN.md','probe.cpp','oracle.py','run_once.py','prepare_metadata.py','authorization-schema.json','source-preparation.json','recipe.json','frozen original probe/State/CastPoint/dual operators/GaugeFar/LegacyGauge/Assemble/Product','production cofactor/determinant/inverse Geometry'],
 'mathematical_checks':[
  'Exact cofactor inverse uses deleted row j/column i with sign(-1)^(i+j), followed by exact dG=-G dg G independent consistency check.',
  'Complete physical-P RWM model has original beta_d[j][i] convention, reference alpha/chi gradients and nonflat scaled connection; parts R0..3/S0..3 precede one R+S/Omega assembly.',
  'All exported products have identical frozen Product factor order and original gradient signs; db/dV/Lh/dL and pole unsigned factors match GaugeFar.',
  'Full submitted input/reference/connection and actual new+legacy outputs bind bitwise including signed zero before diagnostic interpretation.',
  'Held-native-inverse and held-native-auxiliary exact models define valid entrywise telescoping identities; they report observational effects, not independent causal percentages or corrections.',
  'Fraction targets match all original63 MP tangent targets under unchanged2e-10 gates;8 original exact-zero targets retained.',
  'Term bounds follow absolute-sum propagation through recorded rational expression tree; division uses exact nonzero denominator magnitude; this is a diagnostic scale, not a rounding theorem.',
  'Fixed26 original metric-STF contexts only,52 unchanged helper calls; no FD/new state/threshold repair or native/Debug evolution acceptance.'
 ],
 'provenance_read_order':'Root substantive text reads preceded this new receipt capture; original source-index hashes were supplied and are now fully reverified. No claim of pre-pinning those first reads.',
 'metadata_only_inputs':'Original scientific JSONL/executable streamed only for root source preparation; no payload target evaluation, candidate import or compile.',
 'original_far_Release_remains_FAILED':True,'independent_source_review_required_before_release':True
}
write(HERE/'source-pins001.json',pins);write(HERE/'source-review001.json',review)
print(json.dumps({'source_review_passed':True,'pins':len(pins),'execution_released':False,'review_sha256':sha(HERE/'source-review001.json')}))
