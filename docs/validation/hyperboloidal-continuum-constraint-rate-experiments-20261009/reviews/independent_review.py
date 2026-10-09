"""Read-only frozen evidence/draft review, with saved arithmetic only."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P = ROOT/'build-layer-research/continuum/finite-rb-constraint-rate-oracle/immutable-finite-rb-C0-constraint-rates-20261009'
OUT = HERE/'independent-draft-review.json'
assert not OUT.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'index.json') == '3d4c613a814a8a3325a7f980c2e20dcabf3ea08ddbcb10d42026ddca732d4e2f'
index = json.loads((P/'index.json').read_text())
summary = json.loads((P/'summary.json').read_text())
saved = json.loads((HERE/'independent-saved-readback.stdout.json').read_text())
assert saved['passed_saved_data_readback'] and saved['verified_index_files'] == 117
assert saved['total_cases'] == 7896 and not saved['explicitly_omitted_large_payloads']
assert (HERE/'independent-saved-readback.stderr').stat().st_size == 0
assert (HERE/'summary.json').read_bytes() == (P/'summary.json').read_bytes()
norm = lambda a: math.sqrt(math.fsum(x*x for x in a))
sub = lambda a, b: [x-y for x, y in zip(a, b)]
classification = {}
calls_total = 0
for stage in ('core', 'gauge', 'shell'):
    calls = json.loads((P/'payloads'/f'{stage}-calls.json').read_text())
    assert len(calls) == summary['stages'][stage]['api_calls']
    assert all(c['returncode'] == 0 and c['stderr_bytes'] == 0 for c in calls)
    calls_total += len(calls)
    if stage == 'core':
        continue
    counts = {'actual': Counter(), 'subsidiary': Counter()}
    for line in (P/'payloads'/f'{stage}-cases.jsonl').open():
        row = json.loads(line)
        for side in ('actual', 'subsidiary'):
            seq = row[side+'_rate_sequence']
            inc = [norm(sub(b, a)) for a, b in zip(seq, seq[1:])]
            scale = [max(1., norm(a), norm(b)) for a, b in zip(seq, seq[1:])]
            evidence = any(inc[j] > 1e-10*scale[j] and inc[j+1] > 1e-10*scale[j+1]
                           and inc[j] >= 8*inc[j+1] for j in range(3))
            small = all(v/s <= 2e-7 for v, s in zip(inc, scale))
            status = ('classified_at_least_one_pair' if evidence else
                      'within_tolerance_order_unclassified' if small else 'unresolved')
            assert status == row[side+'_sequence']['order_status'] != 'unresolved'
            counts[side][status] += 1
    classification[stage] = {k: dict(v) for k, v in counts.items()}
    for side in counts:
        assert dict(counts[side]) == summary['stages'][stage][side+'_order_status_counts']
assert calls_total == summary['total_scientific_api_calls'] == 23688
paths = [HERE/'audit-draft.md', HERE/'archive-README.md', P/'index.json',
         P/'summary.json', P/'REPORT.md', P/'verify_frozen.py',
         P/'prove_flat_core.py', P/'context/boundary/constraint_rate_api.hpp',
         P/'run_constraint_rates_v2.py', HERE/'independent-saved-readback.stdout.json']
pins = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths]
result = {
    'status': 'PASS_read_only_math_source_saved_data_and_draft_review',
    'reviewer': '/root/literature_gauge',
    'review_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'source_sha256': sha(Path(__file__)), 'input_pins': pins,
    'frozen_index_files_rehashed_by_saved_data_verifier': 117,
    'frozen_index_bytes': index['total_bytes'],
    'saved_cases_recomputed_by_verifier': 7896,
    'API_receipts_rechecked_empty_stderr_exit_zero': calls_total,
    'independently_recomputed_order_classification': classification,
    'corrections': [],
    'math_source_scope': [
        'Core commuting-derivative C(d)L(d)T=B(d)C(d)T identity, physical K=P+2Theta, alpha_t=-3P, beta_t=3Lambda/8 and all kappa signs match the earlier independent source review. The complete trace/determinant tangent T is essential.',
        'ConstraintRateBatch evaluates DC_ref on the full raw22 source jet without a hidden projection; SubsidiaryBatch retains physical Cartesian H,M,Z,Theta and the frozen coefficient-aware C0 equations.',
        'Transition/collar h stencils remain strictly inside rb=.98 and differentiate complete sampled functions; no unavailable higher analytic reference jets are invented.',
        'Saved final/extrapolated discrepancies and all classified/unclassified counts agree. A classified sequence means at least one non-floor increment ratio>=8, not uniform fourth-order convergence over every h.',
        'Pointwise/sample Euclidean statistics do not define an integrated physical/constraint energy. Selected stationary-reference tests do not establish nonlinear live subsidiary closure, discrete Bianchi, CPBC, exact-scri regularity or stability.'
    ],
    'provenance_scope': [
        'Launch and freeze HEADs are distinguished; core/Debug launch context is explicitly reconstructed additively, while FD stages record launch directly.',
        'The frozen package binds API binaries/source/output and preserves the failed Debug executable-permission launch; exact API compiler/dependency attempt context is being collected separately by root.',
        'Metadata-only publication must retain explicit omitted-large-payload entries and must not claim full numerical readback from omitted bytes.'
    ],
    'scientific_kernel_or_SymPy_rerun': False,
    'generator_spectrum_or_propagation': False,
    'scope': 'Completed continuum point-rate checkpoint only. Radial projection/SAT constraint production, rb=.995 and later wormhole-to-trumpet BH with Minkowski reference remain separate unaccepted gates.'
}
OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'status': result['status'], 'receipt': str(OUT), 'sha256': sha(OUT),
                  'bytes': OUT.stat().st_size}, indent=2))
