"""Independent initial ADM/normal-Box projection; saved-kernel rows only."""
from pathlib import Path
import hashlib
import json
import sys
import mpmath as mp
import sympy as s

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
G = P.parent/'q-null-invariant-jets/immutable-Q-sigma5-Einstein-null-noninvariance-20261009'
R = ROOT/'build-layer-research/q-null-independent-box-derivation/immutable-independent-ADM-Box-null-identity-20261009'
GPIN = '4cfbc8ed743c46787f617c33fdbe9875ec092fd117e82f1daddf5e165f2ef29a'
RPIN = '78c6870117cf1fb74e1175cda6abc38b08b5ac9092f3fce7d904bf8dd958e585'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


indices = []
for gate, pin in ((G, GPIN), (R, RPIN)):
    assert sha(gate/'index.json') == pin
    idx = json.loads((gate/'index.json').read_text())
    for name, entry in idx['files'].items():
        digest = entry if isinstance(entry, str) else entry['sha256']
        assert sha(gate/name) == digest
        if isinstance(entry, dict):
            assert (gate/name).stat().st_size == entry['bytes']
    indices.append(idx)
receipt = json.loads((G/'receipt.json').read_text())
assert receipt['source_before'] == receipt['source_after']
assert len(receipt['source_before']) == 372
for name, digest in receipt['source_before'].items():
    assert sha(ROOT/name) == digest
for command in receipt['commands']:
    assert command['returncode'] == 0 and (G/command['stderr']).read_text() == ''
for name, entry in indices[0]['outside_binaries'].items():
    assert sha(G.parent/name) == entry['sha256']
    assert (G.parent/name).stat().st_size == entry['bytes']
assert (G/'actual-release.json').read_bytes() == (G/'actual-debug.json').read_bytes()
root_receipt = json.loads((R/'receipt.json').read_text())
assert root_receipt['exit_code'] == 0 and (R/'run.stderr').read_text() == ''

# Projection identity: Box=Delta+Kbar*wn-n(wn)+acceleration.dOmega.
# Future normal n=(partial_t-beta.grad)/alpha, physical K=-3/a.
r, a, sigma = s.symbols('r a sigma', positive=True)
m = s.symbols('m', positive=True, integer=True)
O = (1-r*r)/(2*a)
h = (1+r*r)/(2*a)
beta = -r/a
w = -r*r/(a*a*h)
Kbar = -3/(a*h)
cb, ca = r/(a*h), r*r/(a*a*h*h)
F, B = s.Function('F')(r), s.Function('B')(r)


def projected_rate(f, b):
    dw = cb*b+ca*f
    dn = -2*w*dw
    # Inverse spatial-metric rate from the physical ADM metric equation.
    dg = -2*r*r*s.diff(b, r)/a**2-2*r**3*b/(a**3*O)-2*r*r*f/(a**3*O)
    # Background stationarity removes variation of the outside lapse factor.
    dwt = (b*s.diff(w, r)+beta*s.diff(dw, r)-f*beta*s.diff(w, r)/h
           +h*(-3*w*dw/O+Kbar*dw+s.diff(f/h, r)*s.diff(O, r)-sigma*dn/O))
    return s.factor(dn), s.factor(dg-2*w*dwt), s.factor(dwt)


dn, ndot, _ = projected_rate(F, B)
T = -(r**4+6*r*r+1)/(4*a*r)
C = (r**6+16*r**4*sigma-29*r**4+15*r*r-3)/(4*a*r*r*(r*r-1))
assert s.factor(ndot-T*s.diff(dn, r)-C*dn) == 0
weighted = s.factor(C+2*T*s.diff(O, r)/O)
weighted3 = -(r*r+3)*(3*r*r-1)/(4*a*r*r)
assert s.factor(weighted.subs(sigma, 3)-weighted3) == 0
assert s.limit(weighted3, r, 1) == -2/a
f0 = s.Function('f')(r)
common = projected_rate(h*f0, beta*f0)
assert common[0] == 0 and common[1] == 0
# Tangential shift cancels between trace and spatial-divergence terms.
Bh, divT = s.symbols('Bh divT')
assert s.simplify((Bh*Bh/h**2)*(2*divT)+(2*Bh/h**2)*(-Bh*divT)) == 0

b = O**m
dn, nt, wt = projected_rate(s.S(0), b)
af = 3*h*r*b/(a*O)  # Q lapse: -h^2 delta(P-3wn)/Omega.
bf = s.factor((wt-ca*af)/cb)
bf_expected = (m-2*sigma)*r*r*O**(m-1)/a**2+(-6/a+2*r*r/(a*a*h))*b
assert s.factor(bf-bf_expected) == 0
cf = s.factor(-s.Rational(2, 3)*(s.diff(b, r)+2*b/r)-2*r*b/(a*O))
q = 2*(s.diff(b, r)-b/r)
lf = s.Rational(4, 3)*(s.diff(b, r, 2)+2*s.diff(b, r)/r-2*b/r**2)
fields = [af, cf, 0, 0, bf, 0, 0, 2*q/3, 0, 0, -q/3, 0,
          0, 0, 0, 0, 0, lf, 0, 0]
normal = s.factor(nt/O**(m-1))
qlimit = s.factor((-3*wt)/O**(m-1))
limits = {}
for mi in (2, 3, 4):
    value = s.simplify(s.limit(normal.subs(m, mi), r, 1))
    assert value == 4*(mi+1-sigma)/a**3
    limits[str(mi)] = str(value)
assert s.simplify(s.limit(qlimit, r, 1)+3*(m+3-2*sigma)/a**2) == 0
assert s.factor(dn/O**m).subs(r, 1) == 2/a
matchedN, matchedRate, _ = projected_rate(s.S(1), -s.S(1))
assert s.limit(matchedN/O**2, r, 1) == -a
assert s.limit(matchedRate/O, r, 1) == 2*(sigma-3)/a

mp.mp.dps = 80
evaluate = s.lambdify((r, a, sigma, m), fields+[dn, -3*cb*b, nt, -3*wt], 'mpmath')
witness_rate = projected_rate(O, -O)[1]
evaluate_witness = s.lambdify((r, a, sigma), witness_rate, 'mpmath')
data = json.loads((G/'actual-release.json').read_text())
errors = {'full20': 0.0, 'initial_N_Q': 0.0, 'Ndot_Qnumdot': 0.0,
          'next_N1': 0.0, 'common_rescaling_Ndot': 0.0,
          'earlier_witness_Ndot': 0.0, '4D_Box': 0.0, 'inferred_Rbar': 0.0}
for row in data['rows']:
    assert all(value == 0 for value in row['C'])
    aa, sg, mi = mp.mpf(str(row['a'])), mp.mpf(str(row['sigma'])), row['m']
    oo = mp.mpf(str(row['Omega']))
    if oo == 0:
        if row['which'] == 0:
            prediction = 4*(3-sg)/aa**3 if mi == 2 else mp.mpf(0)
            errors['next_N1'] = max(errors['next_N1'], float(abs(prediction-row['next_N1'])))
            assert max(map(abs, row['next_R0'])) < 2e-14
        continue
    rr = mp.sqrt(1-2*aa*oo)
    if row['which'] == 0:
        values = evaluate(rr, aa, sg, mi)
        errors['full20'] = max(errors['full20'], max(float(abs(x-y)) for x, y in zip(values[:20], row['F'])))
        errors['initial_N_Q'] = max(errors['initial_N_Q'], float(abs(values[20]-row['Npert'])), float(abs(values[21]-row['Qnumpert'])))
        errors['Ndot_Qnumdot'] = max(errors['Ndot_Qnumdot'], float(abs(values[22]-row['Ndot'])), float(abs(values[23]-row['Qnumdot'])))
        reg = row['initial_4D_regular']
        errors['4D_Box'] = max(errors['4D_Box'], float(abs(reg[0]-sg*values[20]/oo**2)))
        errors['inferred_Rbar'] = max(errors['inferred_Rbar'], abs(reg[3]-(-6*reg[0]+12*reg[1])))
    elif row['which'] == 1:
        errors['common_rescaling_Ndot'] = max(errors['common_rescaling_Ndot'], abs(row['Ndot']))
    else:
        pred = evaluate_witness(rr, aa, sg)
        errors['earlier_witness_Ndot'] = max(errors['earlier_witness_Ndot'], float(abs(pred-row['Ndot'])))
assert max(errors.values()) < 2e-12
assert data['summary']['initial_constraints_error'] == 0
report = {'passed': True, 'sympy_version': s.__version__,
          'sympy_module_path': s.__file__, 'sympy_module_sha256': sha(s.__file__),
          'mpmath_version': mp.__version__, 'oracle_decimal_digits': mp.mp.dps,
          'python_executable': sys.executable, 'python_executable_sha256': sha(sys.executable),
          'python_version': sys.version, 'reviewed_indices': [GPIN, RPIN],
          'verified_frozen_files': [len(j['files']) for j in indices],
          'verified_actual_inputs': 372, 'verified_actual_commands': 5,
          'actual_rows': len(data['rows']), 'radial_rows': sum(x['which'] == 0 for x in data['rows']),
          'independent_errors': errors, 'normalized_limits': limits,
          'initial_gauge_only_transport': str(T), 'initial_gauge_only_reaction': str(C),
          'weighted_sigma3_reaction': str(weighted3),
          'Release_ASan_UBSan_JSON_byte_identical': True,
          'scope': 'Negative sigma5 smooth-time quadratic-null Taylor tangency gate on initial Einstein reference geometry; initial-gauge-only identity. No nonlinear ideal, finiteOmega blowup, sigma3 candidate or PDE stability admission.',
          'no_kernel_compiler_native_or_propagation_executed': True}
(P/'result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
