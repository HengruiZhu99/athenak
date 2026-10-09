"""Compact saved-artifact review; no source queries, builds, or large-payload arithmetic."""
from pathlib import Path
import hashlib,json,math
P=Path(__file__).resolve().parent;cat=json.loads((P/'catalog.json').read_text())
def finite(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,list):
        for v in x:finite(v)
finite(cat);count=0
for name,e in cat['files'].items():
    p=P/name;b=p.read_bytes();assert len(b)==e['bytes']and hashlib.sha256(b).hexdigest()==e['sha256'];count+=1
    if p.suffix=='.json':finite(json.loads(b))
wave=P/'wave-map-local';idx=json.loads((wave/'index.json').read_text());a=wave/idx['accepted_attempt']
r=json.loads((a/'receipt.json').read_text());out=json.loads((a/'release.json').read_text());recipe=json.loads((a/'release-recipe.json').read_text())
assert r['passed_local_gate']and r['sources_unchanged']and r['release_debug_equal']and not r['operators_or_evolution_run']
assert out==json.loads((a/'debug.json').read_text())
for key,tol in recipe['thresholds'].items():
    value=out[key];value=value[-1]if isinstance(value,list)else value;assert value<=tol,key
assert out['factored_reference_max']==0
assert json.loads((P/'CPP-composition-review/attempt001/receipt.json').read_text())['passed']
assert json.loads((P/'wave-map-root-review/receipt.json').read_text())['passed']
assert not json.loads((P/'reference-and-failed-coordinate/index.json').read_text())['complete_cartesian_coordinate_gate_passed']
print(json.dumps({'passed':True,'checked_files':count,'scope':'compact hashes/finite JSON/saved thresholds only; omitted full comparison arithmetic and executable replay remain local'}))
