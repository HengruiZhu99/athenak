"""Read-only early-support source and saved-matrix contribution audit."""
from pathlib import Path
import ast
import hashlib
import json
import math
import time

import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
GATE = P.parent/'q-null-early-feedback/immutable-Q-null-early-feedback-local-20261009'
PIN = 'PENDING'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
start = time.monotonic()
assert PIN != 'PENDING' and sha(GATE/'index.json') == PIN
index = json.loads((GATE/'index.json').read_text())
for name, entry in index['files'].items():
    digest = entry if isinstance(entry, str) else entry['sha256']
    assert sha(GATE/name) == digest
assert len(index['files']) == index['file_count']
receipt = json.loads((GATE/'receipt.json').read_text())
assert receipt['sources_unchanged'] and receipt['source_before'] == receipt['source_after']
for name, digest in receipt['source_after'].items():
    assert sha(ROOT/name) == digest
for row in receipt['commands']:
    assert row['returncode'] == 0 and (GATE/row['stderr']).read_bytes() == b''
for entry in index.get('large_outputs_outside_snapshot', {}).values():
    path = ROOT/entry['original_repo_path']
    assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes']
assert sha(GATE/'early_feedback.hpp') == '562d01b4382c5f9c36afc83b29208f5db3d90ca37f12890c92c03b101eb36a69'
helper = (GATE/'early_feedback.hpp').read_text()
assert 'qnf::Gauge(p,u,g,{.85,.95,0,true})' in helper
assert 'par.weight<0||par.weight>1||par.sigma<0' in helper
assert 'if(!(norm>T(0)))' in helper
assert 'qnf::WeightedNullDifference(p,u)' in helper
assert 'weight*T(par.sigma)*p.domega[i]*delta/norm' in helper
source = json.loads((GATE/'source.json').read_text())
assert source['source_rows'] == 192 and source['outer_rows'] == 32
assert source['pole_columns'] == 160
assert source['reference_fixedpoint_error'] < 1e-12
assert source['generic_4D_Box_feedback_delta_error'] < 1e-11
assert source['identical_outer_gauge_error'] == source['identical_leading_gauge_pole_error'] == 0.
assert source['min_sampled_gradient_norm2_on_support'] > 0.
principal = json.loads((GATE/'check-principal.stdout').read_text())
assert principal['passed_kernel_cases'] == 504 and principal['max_kernel_symbol_error'] < 1e-12
assert principal['harmonic_endpoint_complete']

# Reuse only the two pinned scalar math functions from the independent
# negative-screen review. Execute their definitions alone; never import or
# rerun its top-level driver or write any prior frozen directory.
previous = P.parent/'independent-q-followup-review/immutable-independent-Q-finite-negative-review-20261009'
assert sha(previous/'index.json') == '62d9e68a59858026e09a750f981ddd820ace5d2395208ac769ce7818e8541cc4'
previous_index = json.loads((previous/'index.json').read_text())
assert sha(previous/'review_finite.py') == previous_index['files']['review_finite.py']['sha256']
tree = ast.parse((previous/'review_finite.py').read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('cutoff', 'ref')]
assert len(functions) == 2
environment = {'math': math}
exec(compile(ast.Module(body=functions, type_ignores=[]), '<pinned scalar functions>', 'exec'), environment)
cutoff, reference = environment['cutoff'], environment['ref']
oldgate = P.parent/'conformal-q-followup/immutable-Q-null-finite-frequency-negative-20261009'
assert sha(oldgate/'index.json') == 'adaa2b1437054f4ba4cf6471eafb6f83af96a31e9ea376d723a2666abcf5e72a'
oldindex = json.loads((oldgate/'index.json').read_text())
assert sha(oldgate/'fourier.json') == oldindex['files']['fourier.json']
old = {}
for row in json.loads((oldgate/'fourier.json').read_text()):
    raw = np.array(row['M'])
    old[tuple(row[k] for k in ('a', 'r', 'k', 'dir', 'form'))] = raw[:, :, 0]+1j*raw[:, :, 1]
data = json.loads((GATE/'fourier.json').read_text())
roots = json.loads((GATE/'fourier-roots.json').read_text())
assert len(data) == len(roots) == 560
error = outer_error = core_error = root_error = 0.
positive_count = 0
positive_matrices = 0
worst = {}
for row, recorded in zip(data, roots):
    a, r, k, direction, form = (row[n] for n in ('a', 'r', 'k', 'dir', 'form'))
    assert form in (4, 5)
    assert tuple(row[n] for n in ('a', 'r', 'k', 'dir', 'form')) == tuple(recorded[n] for n in ('a', 'r', 'k', 'dir', 'form'))
    raw = np.array(row['M'])
    matrix = raw[:, :, 0]+1j*raw[:, :, 1]
    assert matrix.shape == (20, 20) and np.isfinite(matrix).all()
    O, dO, h, hp, beta, chi, gr = reference(r, a)
    oldweight, _ = cutoff(r, .85, .95)
    newweight, _ = cutoff(r, .45, .85) if form == 4 else cutoff(r, .65, .85)
    deltaweight = newweight-oldweight
    wn = -beta*dO/h
    delta = np.zeros((20, 20), dtype=complex)
    delta[4, 0] = 2*deltaweight*5*h*wn*wn/(O*dO)
    delta[4, 1] = deltaweight*5*h*h*dO/(O*gr)
    delta[4, 4] = 2*deltaweight*5*h*wn/O
    delta[4, 7] = -deltaweight*5*h*h*chi*dO/(O*gr*gr)
    base = old[a, r, k, direction, 2]
    error = max(error, float(np.max(np.abs(matrix-base-delta))))
    if r >= .95:
        outer_error = max(outer_error, float(np.max(np.abs(matrix-base))))
    if r == .45:
        core_error = max(core_error, float(np.max(np.abs(matrix-old[a, r, k, direction, 3])))))
    ev = np.linalg.eigvals(matrix)
    largest = float(ev.real.max())
    root_error = max(root_error, abs(largest-recorded['max_real']))
    count = int(np.sum(ev.real > 1e-8))
    positive_count += count
    positive_matrices += count > 0
    if (a, form) not in worst or largest > worst[a, form]['max_real']:
        worst[a, form] = {**{n: row[n] for n in ('a', 'r', 'Omega', 'k', 'dir', 'form')},
                           'max_real': largest, 'positive_roots': count}
assert error < 1e-8 and outer_error == core_error == 0.
assert root_error < 1e-8
assert all(abs(worst[.5, form]['max_real']-20.168940777918813) < 1e-9 for form in (4, 5))
out = {'status': 'PASS_READ_ONLY_EARLY_SUPPORT_LOCAL_REVIEW',
       'seconds': time.monotonic()-start, 'frozen_index_sha256': PIN,
       'files_rechecked': len(index['files']), 'sources_rechecked': len(receipt['source_after']),
       'commands_rechecked': len(receipt['commands']), 'helper_sha256': sha(GATE/'early_feedback.hpp'),
       'matrix_count': 560, 'rank_one_weight_change_Jacobian_max_error': error,
       'outer_and_W0_matrix_equality_error': max(outer_error, core_error),
       'saved_max_root_recomputation_error': root_error,
       'positive_primitive_count_threshold': 1e-8,
       'positive_primitive_root_counts_sum': positive_count,
       'matrices_with_positive_primitive_roots': positive_matrices,
       'worst_by_a_form': list(worst.values()),
       'source_and_principal': {'source': source, 'principal': principal},
       'support_admissibility': 'Target S1,a>=.5,geometry(.05,.95),gauge(.45,.85),finite sigma5. Omega prime is strictly negative outside geometric core: wprime*(Omegaout-1)+w*Omegaoutprime<0. Both feedback supports lie there. Base qnf still requires suppliedg.r1<=.85; mode1 has fixed.65/.85. No arbitrary-radius or NaN/inf parameter admission.',
       'source_scope': 'Generic deltaBox=V*sigma*deltaNraw/Omega holds for the new feedback at any admitted support point. Full preferred-source Box identity remains W1-only; no transition identity is inherited.',
       'scope': 'Read-only local algebra/source and saved Fourier matrix review. Positive primitive roots retained; no tensor rerun, global/native/constraint-subsidiary/energy/BH or smooth-hierarchy claim.'}
(P/'early-review-result.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
