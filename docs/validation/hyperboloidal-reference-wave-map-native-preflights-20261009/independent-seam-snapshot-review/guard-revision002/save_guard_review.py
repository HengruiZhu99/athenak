"""Exact-byte final guard source review metadata; no scientific code import/run."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OWNER=ROOT/'build-layer-research/reference-wave-map-native-held-20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
index_path=HERE/'analyzer-guard-review-index.json'
assert sha(index_path)=='defa29818d8f8c2ce95d32018ce7bc59989083c9f1429734548d10b8b93fb882'
index=json.loads(index_path.read_text())
for name,item in index['files'].items():
    assert sha(HERE/'source-copies'/name)==sha(OWNER/name)==item['sha256'],name
old=(HERE.parent/'source-copies/analyze_snapshots.py').read_text()
new=(HERE/'source-copies/analyze_snapshots.py').read_text()
assert old== (HERE/'source-copies/review-history/pre-seam-and-postpins-guard/analyze_snapshots.py').read_text()
for text in ["assert seam['passed_compile_and_fixed_t0_seam'] is True", "assert seam['probe_executable_sha256']==auth['probe_executable_sha256']", "assert seam['recipe_sha256']==auth['probe_recipe_sha256']", "assert seam['seam_cases_sha256']==sha(HERE/'seam-cases.json')", "dump(output/'protected-inputs-before.json',protected)", "dump(output/'protected-inputs-after.json',protected)", "'protected_inputs_before_after_equal':True"]:
    assert text in new,text
seam=json.loads((HERE/'source-copies/probe-attempts/compile-and-seam-001/receipt.json').read_text())
assert seam['passed_compile_and_fixed_t0_seam'] is True
assert seam['recipe_sha256']=='83640cb37b7132eb32b026801e96331768aade94fce9f89ddfade39768f44932'
assert seam['probe_executable_sha256']==index['probe_executable_sha256']
receipt={
 'passed_independent_source_only_review':True,
 'scientific_execution_or_recomputation':False,
 'owner_guard_index_sha256':sha(index_path),
 'reviewed_files':index['files'],
 'analyzer_sha256':index['analyzer_sha256'],
 'source_review_report_sha256':sha(HERE/'REVIEW.md'),
 'source_review_program_sha256':sha(Path(__file__)),
 'prior_captured_v2_review_receipt_sha256':sha(HERE.parent/'receipt.json'),
 'successful_seam_receipt_verified_by_hash_and_schema_only':True,
 'owner_seam_receipt_sha256':'54a7f3f955c294bcc45307986ec33f4d7a2fb5e10d6626c89df94a68061a2b73',
 'prior_findings_closed':['successful-seam-prerequisite','end-of-case-source-pins'],
 'remaining_scope_note':'Assertions before inner try need outer launcher command/stdout/stderr capture, as explicitly documented.',
 'blocking_source_math_layout_transport_issues':[],
 'owner_bytes_unchanged_at_review_completion':True,
 'gate_owner':'root',
 'scope':'Final exact guard-only source review with prior full source/math/layout/transport review. No probe query, analyzer call, compile, saved-array recomputation, operator, or native evolution.'}
(HERE/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({'receipt_sha256':sha(HERE/'receipt.json'),'report_sha256':sha(HERE/'REVIEW.md'),'passed_source_only_review':True},indent=2))
