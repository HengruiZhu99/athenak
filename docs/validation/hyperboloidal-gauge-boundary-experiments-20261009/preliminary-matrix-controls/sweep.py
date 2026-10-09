"""Preliminary algebraic feedback applied to saved actual-kernel matrices.

At a spherical reference point, additional alpha and normal beta RHS poles
are proportional to (-c, -beta_ref/alpha_ref*c) in the alpha column and
((1-c)*Omega_r/wn_ref, -(1-c)) in the radial beta column. This is the
linearization of a null-tangent lapse/normal-shift value control. It does not
replace a compiled candidate kernel audit or establish continuum stability.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / ('build-layer-research/continuum/preferred/native-overlay/'
               'fourier-mode-matrices.json')
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


def reference(case):
    r = case['r']
    w, dw = cutoff(r, .05 if case['wide'] else .2,
                    .95 if case['wide'] else .8)
    omega = 1-w*r*r
    domega = -dw*r*r-2*w*r
    L = omega-r*domega
    boost = 2*r*w
    alpha = math.hypot(omega, boost)
    beta = -boost*alpha/L
    wn = -beta*domega/alpha
    assert abs(omega-case['omega']) < 1e-14
    return omega, domega, alpha, beta, wn


CASES = [v for v in DATA if not v['candidate'] and v['eps'] == 1e-6
         and v['kappa'] == 10 and v['wide']]
OUTPUT = {'scope': __doc__, 'matrix_path': str(PATH),
          'matrix_sha256': EXPECTED, 'cases': []}
for c in [0, .25, .5, .75, 1]:
    for rate in [1, 2, 5, 10, 20, 40, 80]:
        for tangent in [0, 10]:
            modes = []
            for case in CASES:
                omega, domega, alpha, beta, wn = reference(case)
                W, _ = cutoff(case['r'], .45, .85)
                B = np.asarray(case['B']).copy()
                C = np.asarray(case['C'])
                B[0, 0] -= rate*W*c/omega
                B[0, 4] += rate*W*(1-c)*domega/wn/omega
                B[4, 0] -= rate*W*c*beta/alpha/omega
                B[4, 4] -= rate*W*(1-c)/omega
                B[5, 5] -= tangent*W/omega
                B[6, 6] -= tangent*W/omega
                spectrum = np.linalg.eigvals(B+1j*C)
                root = spectrum[np.argmax(spectrum.real)]
                modes.append({'r':case['r'], 'k':case['k'],
                              'oblique':case['oblique'],
                              'max_real':float(root.real),
                              'root_imag':float(root.imag)})
            OUTPUT['cases'].append({'c':c, 'rate':rate, 'tangent_rate':tangent,
                                    'modes':modes,
                                    'worst':max(modes,key=lambda v:v['max_real'])})
for case in sorted(OUTPUT['cases'], key=lambda v:v['worst']['max_real'])[:10]:
    print(case['c'],case['rate'],case['tangent_rate'],case['worst'])
(ROOT/'build-layer-research/null-tangent-control/results.json').write_text(
    json.dumps(OUTPUT,indent=2,allow_nan=False)+'\n')
