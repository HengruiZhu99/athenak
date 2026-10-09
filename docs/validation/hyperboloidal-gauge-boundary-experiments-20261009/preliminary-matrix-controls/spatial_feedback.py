"""Preliminary spatial-normal-norm shift feedback on actual-kernel matrices.

xi=1/a and S_beta=-eta*W[delta_beta+C*n_out*(G/G_ref-1)],
G=chi*gtilde_inverse^ij*Omega_i*Omega_j, C=(S/a)[1-S/(eta*a^2)].
This value-only feedback preserves the Minkowski fixed point. It cancels
geometric BH leading shift gauge motion when the mass-corrected initial height
matches the outer Minkowski branch, without subtracting any BH RHS. The
current matrix sweep is algebraic, rather than an independently compiled
candidate source or a proof of nonlinear scri closure or global stability.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'build-layer-research/continuum/preferred/native-overlay'
SOURCES = {
    'fourier-matrices.json':'bf779ebb2a0ff3701b0e73ee9354d564ee79134e3a1c8046fe9b693954389687',
    'fourier-mode-matrices.json':'e8fed96d9dd1b0845618913644e271a804adb148f4f922c254559a5e3bbdae5b'}
ALL = {}
for name, expected in SOURCES.items():
    source = BASE/name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    for row in json.loads(source.read_text()):
        if not row['candidate'] and row['eps'] == 1e-6 and row['kappa'] == 10 and row['wide']:
            ALL[row['r'], row['k'], row['oblique']] = row


def cutoff(r, low, high):
    if r <= low:
        return 0, 0
    if r >= high:
        return 1, 0
    s = (r-low)/(high-low)
    e = math.exp(-1/s)
    f = math.exp(-1/(1-s))
    w = e/(e+f)
    return w, w*(1-w)*(1/s**2+1/(1-s)**2)/(high-low)


def reference(case):
    r = case['r']
    w, dw = cutoff(r, .05, .95)
    om = 1-w*r*r
    L = om-r*(-dw*r*r-2*w*r)
    alpha = math.hypot(om, 2*r*w)
    chi = (alpha/L)**(2/3)
    grr = chi*L*L/(alpha*alpha)
    return om, alpha, chi, grr


OUTPUT = {'scope':__doc__, 'sources_sha256':SOURCES, 'cases':[]}
for eta in [4,5,5.5,6,6.5,7,8,10,15,20]:
    coeff = 2*(1-4/eta)
    modes = []
    for case in ALL.values():
        om, alpha, chi, grr = reference(case)
        W, _ = cutoff(case['r'], .45, .85)
        B = np.asarray(case['B']).copy()
        C = np.asarray(case['C'])
        B[0,0] -= 2*alpha*W*.5/om
        for j in [4,5,6]:
            B[j,j] -= eta*W/om
        B[4,1] -= eta*W*coeff/chi/om
        B[4,7] += eta*W*coeff/grr/om
        spectrum = np.linalg.eigvals(B+1j*C)
        root = spectrum[np.argmax(spectrum.real)]
        modes.append({'r':case['r'], 'k':case['k'], 'oblique':case['oblique'],
                      'max_real':float(root.real), 'root_imag':float(root.imag)})
    row = {'xi':2, 'eta':eta, 'normal_norm_coefficient':coeff, 'modes':modes,
           'worst':max(modes,key=lambda v:v['max_real']),
           'high_frequency_worst':max((v for v in modes if v['k']>=32),
                                      key=lambda v:v['max_real'])}
    OUTPUT['cases'].append(row)
    print(eta,coeff,row['worst'])
(ROOT/'build-layer-research/null-tangent-control/spatial_feedback_results.json').write_text(
    json.dumps(OUTPUT,indent=2,allow_nan=False)+'\n')
