"""Strict saved-output endpoint/core and tail coverage readback."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OWNER = ROOT/'build-layer-research/boundary/einstein-coordinate-gauge-local-20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
recipe = json.loads((HERE/'recipe.json').read_text())
for item in recipe['inputs']:
    assert sha(ROOT/item['path']) == item['sha256']
native = [[float(v) for v in line.split()]
          for line in (OWNER/'reference-gate-attempt001/reference-release.stdout').read_text().splitlines()]
assert (OWNER/'reference-gate-attempt001/reference-release.stdout').read_bytes() == (
    OWNER/'reference-gate-attempt001/reference-debug.stdout').read_bytes()
offset = {}
column = 3
owner_recipe = json.loads((OWNER/'reference-local-recipe.json').read_text())
for field in owner_recipe['radial_fields_in_output_order']:
    offset[field['name']] = (column, len(field['ordinary_derivatives']))
    column += len(field['ordinary_derivatives'])
assert column == 57
checks = []
for row in native:
    r = row[0]
    assert all(math.isfinite(v) for v in row)
    for key in ('chi', 'g_radial', 'alpha', 'L', 'omega'):
        assert row[offset[key][0]] > 0
    if r <= .05 or r >= .95:
        for name, (col, count) in offset.items():
            for order in range(count):
                zero = ((r <= .05 and (order > 0 or name in
                                      ('A_radial', 'A_tangent', 'P', 'b', 'beta', 'lambda', 'weight')))
                        or (r >= .95 and ((name in ('A_radial', 'A_tangent', 'complement', 'lambda'))
                                          or (name in ('P', 'chi', 'g_radial', 'weight') and order > 0)
                                          or (name in ('omega', 'L', 'alpha') and order > 2)
                                          or (name in ('b', 'beta') and order > 1))))
                if zero:
                    assert row[col+order] == 0, (r, name, order, row[col+order])
                    checks.append((r, name, order))
comparisons = json.loads((HERE/'radial-attempt002/radial-comparisons.json').read_text())
assert len(comparisons) == 2052 and all(row['passed'] for row in comparisons)
tail = [row for row in comparisons if row['tail_nonzero_required']]
receipt = dict(kind='Strict saved reference branch readback',
               source_sha256=sha(Path(__file__)), radial_receipt_sha256=sha(HERE/'radial-attempt002/receipt.json'),
               owner_recipe_sha256=sha(OWNER/'reference-local-recipe.json'),
               owner_output_sha256=sha(OWNER/'reference-gate-attempt001/reference-release.stdout'),
               exact_branch_zero_checks=len(checks), exact_branch_rows=sum(row[0] <= .05 or row[0] >= .95 for row in native),
               finite_entries=38*57, positive_radial_values=38*5,
               tail_nonzero_checks=len(tail),
               smallest_required_tail=min(abs(row['native']) for row in tail),
               release_debug_bitwise_equal=True, passed=True,
               scope='Saved-output exact branch/finite/radial positivity and coverage; no new owner executable invocation or Cartesian lift gate.')
(HERE/'branch-readback.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
