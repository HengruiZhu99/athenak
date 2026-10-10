"""Metadata-only freeze of a completed manual source/math review."""
from pathlib import Path
import hashlib
import json

here = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def write(p, d):
    p.write_text(json.dumps(d, indent=2, sort_keys=True, allow_nan=False) + '\n')


before = json.loads((here/'source-before.json').read_text())
assert not (here/'receipt.json').exists()
for path, r in before['originals'].items():
    assert sha(path) == r['sha256'] and Path(path).stat().st_size == r['bytes'], path
for r in before['exact_copies']:
    assert sha(r['path']) == r['sha256'] and Path(r['path']).stat().st_size == r['bytes']
write(here/'receipt.json', {
    'passed':True,'inputs_unchanged':True,'source_review_only':True,
    'manual_mathematical_source_inspection':True,
    'reviewed_header_sha256':sha(here/'inputs/exact_gauge_rows.hpp'),
    'included_backend_sha256':sha(here/'inputs/exact_dyadic_ratio.hpp'),
    'review_sha256':sha(here/'REVIEW.md'),
    'source_before_sha256':sha(here/'source-before.json'),
    'exact_context_copies':6,'blocking_corrections':[],
    'scientific_execution':False,'compiler_invoked':False,
    'candidate_imports':False,'arithmetic_or_targets_recomputed':False,
    'native_binding_or_evolution':False,
    'scope':'Uncompiled exact whole-row WIP source/layout/algebra/range review only; backend and integration execution remain separately held.',
    'original_63_failed_comparisons_preserved':True,
    'resource_bound_scope':'Mathematical source upper bounds only, no executable capacity test.'})
files = []
for p in sorted(here.rglob('*')):
    if p.is_file() and p.name != 'index.json':
        files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
write(here/'index.json',{'passed':True,'source_review_only':True,
    'files':files,'file_count':len(files),'scientific_execution':False,
    'original_sources_unchanged':True})
print(json.dumps({'index_sha256':sha(here/'index.json'),
    'receipt_sha256':sha(here/'receipt.json'),
    'review_sha256':sha(here/'REVIEW.md'),
    'file_count':len(files),'bytes':sum(v['bytes'] for v in files)}))
