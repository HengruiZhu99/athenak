"""Read-only finite-Omega Fourier/coefficient review, without tensor rerun."""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
GATE = P.parent/'conformal-q-followup/immutable-Q-null-finite-frequency-negative-20261009'
PIN = 'adaa2b1437054f4ba4cf6471eafb6f83af96a31e9ea376d723a2666abcf5e72a'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
start = time.monotonic()
assert sha(GATE/'index.json') == PIN
index = json.loads((GATE/'index.json').read_text())
for name, digest in index['files'].items():
    assert sha(GATE/name) == digest
assert len(index['files']) == index['file_count'] == 19
assert sum((GATE/n).stat().st_size for n in index['files']) == index['bytes'] == 12749009
receipt = json.loads((GATE/'receipt.json').read_text())
assert receipt['sources_unchanged'] and receipt['source_before'] == receipt['source_after']
assert len(receipt['source_after']) == 371
for name, digest in receipt['source_after'].items():
    assert sha(ROOT/name) == digest
assert len(receipt['commands']) == 3
for row in receipt['commands']:
    assert row['returncode'] == 0 and (GATE/row['stderr']).read_bytes() == b''
for entry in index['large_outputs_outside_snapshot'].values():
    assert sha(ROOT/entry['original_repo_path']) == entry['sha256']
data = json.loads((GATE/'fourier.json').read_text())
roots = json.loads((GATE/'fourier-roots.json').read_text())
assert len(data) == len(roots) == 1120
matrices = {}
root_error = 0.
root_counts = 0
for row, recorded in zip(data, roots):
    key = tuple(row[n] for n in ('a', 'r', 'k', 'dir', 'form'))
    assert key == tuple(recorded[n] for n in ('a', 'r', 'k', 'dir', 'form'))
    raw = np.array(row['M'])
    m = raw[:, :, 0]+1j*raw[:, :, 1]
    assert m.shape == (20, 20) and np.isfinite(m).all() and key not in matrices
    matrices[key] = m
    ev = np.linalg.eigvals(m)
    root_error = max(root_error, abs(float(ev.real.max())-recorded['max_real']))
    assert int(np.sum(ev.real > 1e-8)) == recorded['positive_roots']
    root_counts += recorded['positive_roots']
assert root_error < 1e-8

def cutoff(r, r0, r1):
    if r <= r0:
        return 0., 0.
    if r >= r1:
        return 1., 0.
    t = (r-r0)/(r1-r0)
    u = (r1-r)/(r1-r0)
    g = -1/t+1/u
    exponential = math.exp(-abs(g))
    w = exponential/(1+exponential) if g <= 0 else 1/(1+exponential)
    derivative = exponential/(1+exponential)**2*(1/t**2+1/u**2)/(r1-r0)
    return w, derivative

def ref(r, a):
    # Independently evaluate only the scalar reference quantities needed by
    # the algebraic source Jacobians, not any tensor or derivative kernel.
    if r >= .95:
        return (1-r)*(1+r)/(2*a), -r/a, (1+r*r)/(2*a), r/a, -r/a, 1., 1.
    w, wp = cutoff(r, .05, .95)
    out = (1-r)*(1+r)/(2*a)
    O = (1-w)+w*out
    dO = wp*(out-1)-w*r/a
    b, bp = r*w/a, (w+r*wp)/a
    L = O-r*dO
    alpha = math.hypot(O, b)
    da = (O*dO+b*bp)/alpha
    beta = -b*alpha/L
    chi = (alpha/L)**(2/3)
    gr = chi*L*L/(alpha*alpha)
    return O, dO, alpha, da, beta, chi, gr

blend_error = feedback_error = equality_error = 0.
for a in (.5, .75, 1., 2.):
    for r in (.45, .65, .8, .85, .9, .95, .98):
        O, dO, h, hp, beta, chi, gr = ref(r, a)
        W, _ = cutoff(r, .45, .85)
        e, xi = 1-W, 1/a
        Bh, adv = beta*dO, beta*hp
        # (1-W)*(physicalP_alpha-Q_alpha), from direct differentiation of
        # the two algebraic lapse sources at the reference. All other rows,
        # live derivative coefficients, and the P column cancel exactly.
        blend = np.zeros((20, 20), dtype=complex)
        blend[0, 0] = e*(adv/h+(-2*W*xi*h-W*Bh-3*(h+2*e)*Bh/h)/O)
        blend[0, 4] = e*(hp+(-W*h+3*(h+2*e))*dO/O)
        # sigma5-minus0 feedback is rank one at this aligned reference point.
        V, _ = cutoff(r, .85, .95)
        wn = -Bh/h
        feedback = np.zeros((20, 20), dtype=complex)
        feedback[4, 0] = 2*V*5*h*wn*wn/(O*dO)
        feedback[4, 1] = V*5*h*h*dO/(O*gr)
        feedback[4, 4] = 2*V*5*h*wn/O
        feedback[4, 7] = -V*5*h*h*chi*dO/(O*gr*gr)
        for k in (0., 4., 16., 64., 256.):
            for direction in (0, 1):
                m = [matrices[a, r, k, direction, form] for form in range(4)]
                blend_error = max(blend_error, float(np.max(np.abs(m[2]-m[1]-blend))))
                feedback_error = max(feedback_error, float(np.max(np.abs(m[1]-m[0]-feedback))))
                if r >= .85:
                    equality_error = max(equality_error, float(np.max(np.abs(m[2]-m[1]))))
                if r == .85:
                    equality_error = max(equality_error, float(np.max(np.abs(m[1]-m[0]))))
                if r == .45:
                    equality_error = max(equality_error, float(np.max(np.abs(m[2]-m[3]))))
assert blend_error < 1e-8 and feedback_error < 1e-8 and equality_error == 0.
target = max((r for r in roots if r['a'] == .5 and r['form'] == 2), key=lambda r: r['max_real'])
baseline = max((r for r in roots if r['a'] == .5 and r['form'] == 3), key=lambda r: r['max_real'])
assert target['r'] == .85 and target['k'] == 4 and target['dir'] == 0
assert abs(target['max_real']-22.467615596732056) < 1e-10
assert abs(baseline['max_real']-1.5319523359620948) < 1e-10
out = {'status': 'PASS_READ_ONLY_NEGATIVE_FOURIER_AND_COEFFICIENT_REVIEW',
       'seconds': time.monotonic()-start, 'frozen_index_sha256': PIN,
       'file_count': 19, 'frozen_bytes': 12749009, 'source_inputs_rechecked': 371,
       'commands_rechecked': 3, 'matrix_count': 1120,
       'recorded_max_real_recomputation_error': root_error,
       'positive_primitive_root_counts_sum': root_counts,
       'independent_alpha_blend_Jacobian_max_error': blend_error,
       'independent_feedback_Jacobian_max_error': feedback_error,
       'endpoint_onset_matrix_equality_error': equality_error,
       'target_worst': target, 'matched_baseline_worst': baseline,
       'phase_convention': 'exp(+i*k*n.x), forward column action dq/dt=Lq; temporal growth is Re(lambda) in coordinate generator units.',
       'scope': 'Read-only source/coefficient and saved-matrix eigenscreen verification. No tensor rerun, physical subsidiary classification, global/PDE/native/energy/BH inference.'}
(P/'finite-review-result.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
