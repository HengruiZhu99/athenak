"""Preliminary first-jet-compatible xi/eta control spectra.

Uses saved actual-kernel Fourier matrices; this is algebraic linearization
rather than an independently compiled source or a global stability argument.
The relation eta*a^2/S=2*(1+xi*a)/(3+xi*a) preserves beta/null tangency only.
The stronger initial zero-jet gauge compatibility candidate is xi=1/a and
eta=S/a^2, retaining original geometric BH first jets.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT/'build-layer-research/continuum/preferred/native-overlay/fourier-mode-matrices.json'
EXPECTED = 'e8fed96d9dd1b0845618913644e271a804adb148f4f922c254559a5e3bbdae5b'
assert hashlib.sha256(PATH.read_bytes()).hexdigest() == EXPECTED
DATA = json.loads(PATH.read_text())


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


def alpha_at(case):
    w, _ = cutoff(case['r'], .05 if case['wide'] else .2,
                  .95 if case['wide'] else .8)
    return math.hypot(case['omega'], 2*case['r']*w)


CASES = [v for v in DATA if not v['candidate'] and v['eps'] == 1e-6
         and v['kappa'] == 10 and v['wide']]
OUTPUT = {'scope':__doc__, 'matrix_sha256':EXPECTED, 'cases':[]}
parameters = [(xi,8*(1+.5*xi)/(3+.5*xi),'beta/null first-jet relation')
              for xi in [0,.5,1,1.5,2,3,14/3,6,10,20,40,80]]
parameters += [(xi,10,'eta10 independent control') for xi in [2,3,5,10,20,40,80]]
for xi, eta, scope in parameters:
    modes = []
    for case in CASES:
        W, _ = cutoff(case['r'], .45, .85)
        B = np.asarray(case['B']).copy()
        C = np.asarray(case['C'])
        B[0,0] -= 2*alpha_at(case)*W*(xi-1.5)/case['omega']
        for component in [4,5,6]:
            B[component,component] -= eta*W/case['omega']
        spectrum = np.linalg.eigvals(B+1j*C)
        root = spectrum[np.argmax(spectrum.real)]
        modes.append({'r':case['r'], 'k':case['k'], 'oblique':case['oblique'],
                      'max_real':float(root.real), 'root_imag':float(root.imag)})
    row = {'xi':xi,'eta':eta,'parameter_scope':scope,'modes':modes,
           'worst':max(modes,key=lambda v:v['max_real'])}
    OUTPUT['cases'].append(row)
    print(xi,eta,scope,row['worst'])
(ROOT/'build-layer-research/null-tangent-control/paired_results.json').write_text(
    json.dumps(OUTPUT,indent=2,allow_nan=False)+'\n')
