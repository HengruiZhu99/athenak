"""Independent frozen Einstein pullback review; no actual kernel rerun."""
from pathlib import Path
import hashlib
import json
import sys
import sympy as s
import mpmath as mp

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
G = ROOT/'build-layer-research/continuum/q-null-spatial-diffeo/immutable-Q-spatial-Einstein-pullback-timejet-20261009'
PIN = 'de31bb51e55c159100a58d14acaae0f00231a7d6128bc6c3771d535b75ff5662'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(G/'index.json') == PIN
index = json.loads((G/'index.json').read_text())
for row in index['files']:
    path = G/row['path']
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
receipt = json.loads((G/'receipt.json').read_text())
assert receipt['source_before'] == receipt['source_after'] and receipt['sources_unchanged']
assert len(receipt['source_before']) == 374 and len(receipt['commands']) == 6
assert all(row['returncode'] == 0 for row in receipt['commands'])
for name, pin in receipt['source_before'].items():
    assert sha(ROOT/name) == pin
assert (G/'actual-release.json').read_bytes() == (G/'actual-debug.json').read_bytes()
actual = json.loads((G/'actual-release.json').read_text())
basis = json.loads((G/'basis.json').read_text())
assert len(basis) == 120 and len(actual['controls']) == 1920

# A finite radial coordinate pullback provides a separate route to the
# stationary four-metric inverse and volume, before taking its tangent.
r, a, sigma, eps = s.symbols('r a sigma eps', positive=True)
xi = s.Function('xi')(r)
O = (1-r*r)/(2*a)
h = (1+r*r)/(2*a)
beta = -r/a
R = r+eps*xi
OR = (1-R*R)/(2*a)
hR = (1+R*R)/(2*a)
Jac = 1+eps*s.diff(xi, r)
Grr = 1-beta*beta/h**2
volume = h*r*r
inverse_pulled = (OR/O)**2*(1-R*R/(a*a*hR*hR))/Jac**2
volume_pulled = (O/OR)**4*hR*R*R*Jac
alpha_pulled = O/OR*hR
beta_pulled = -R/(a*Jac)
dG = s.factor(s.diff(inverse_pulled, eps).subs(eps, 0))
dvolume = s.factor(s.diff(volume_pulled, eps).subs(eps, 0))
dalpha = s.factor(s.diff(alpha_pulled, eps).subs(eps, 0))
dbeta = s.factor(s.diff(beta_pulled, eps).subs(eps, 0))
assert s.simplify(dalpha-(xi*s.diff(h,r)-h*xi*s.diff(O,r)/O)) == 0
assert s.simplify(dbeta-(xi*s.diff(beta,r)-beta*s.diff(xi,r))) == 0
N = s.factor(Grr*s.diff(O,r)**2)
Box = lambda F: s.diff(volume*Grr*s.diff(F,r),r)/volume
B = s.factor(Box(O))
dN = s.factor(dG*s.diff(O,r)**2)
dB = s.factor(s.diff((dvolume*Grr+volume*dG)*s.diff(O,r),r)/volume-dvolume*B/volume)
w = xi*s.diff(O,r)
f = w/O
assert s.factor(dN-(xi*s.diff(N,r)-2*Grr*s.diff(w,r)*s.diff(O,r)+2*f*N)) == 0
assert s.factor(dB-(xi*s.diff(B,r)-Box(w)+2*f*B-2*Grr*s.diff(f,r)*s.diff(O,r))) == 0
radial = []
mp.mp.dps = 100
numeric = []
for xi0, label in [(O, 'xi=Omega*n'), (r*O, 'xi=Omega*x')]:
    sub = {xi: xi0, s.diff(xi,r): s.diff(xi0,r), s.diff(xi,r,2): s.diff(xi0,r,2)}
    dn = s.factor(dN.subs(sub))
    db = s.factor(dB.subs(sub))
    rate = s.factor(2*r*r*(db-sigma*dn/O)/(a*a))
    n2 = s.factor(s.limit(dn/O**2, r, 1))
    box1 = s.factor(s.limit(db/O, r, 1))
    n1t = s.factor(s.limit(rate/O, r, 1))
    assert n2 == -2/a and box1 == -4/a
    assert s.simplify(n1t-4*(sigma-2)/a**3) == 0
    radial.append({'generator': label, 'N2': str(n2), 'stationary_Box1': str(box1),
                   'N1_time': str(n1t), 'exact_Ndot': str(rate)})
    fn = s.lambdify((r,a,sigma), rate/O, 'mpmath')
    for aa in [mp.mpf('.5'), mp.mpf('.75'), mp.mpf(1), mp.mpf(2)]:
        for sig in [3,5]:
            rr = mp.sqrt(1-2*aa*mp.mpf('1e-40'))
            err = abs(fn(rr,aa,sig)-4*(sig-2)/aa**3)
            assert err < mp.mpf('1e-35')
            numeric.append({'generator': label, 'a': str(aa), 'sigma': sig,
                            'Omega': '1e-40', 'absolute_limit_error': mp.nstr(err, 15)})

# Independent angular leading map: for homogeneous Y=x_j X of degree d,
# Box(Y) at scri is Delta_S(Y).  Product-rule terms are kept before the limit.
o, degree, Y, lapY = s.symbols('o degree Y lapY')
rr2 = 1-2*a*o
hh = 1/a-o
Nref = rr2*o**2/(a*a*hh*hh)
Boxref = s.factor(B.subs(r*r, rr2))
coef = 1/(a*hh)-4/(a*a*hh*hh)+rr2/(a**3*hh**3)
BoxY = lapY-degree*(degree-1)*Y/(a*a*hh*hh)+coef*degree*Y
gradY_gradO = -degree*o**2*Y/(a*hh*hh)
angularN = -o*Y*s.diff(Nref,o)/a+2*o*gradY_gradO/a
angularBox = -o*Y*s.diff(Boxref,o)/a+o*BoxY/a-Y*Boxref/a+4*gradY_gradO/a
angular_rate = s.factor(s.limit(2*rr2*(angularBox-sigma*angularN/o)/(a*a*o),o,0))
expected_rate = 2*(lapY+(2*sigma-4-degree*(degree+1))*Y)/a**3
assert s.simplify(angular_rate-expected_rate) == 0
x,y,z = s.symbols('x y z')
xyz = [x,y,z]
expected = []
for item in basis:
    field = xyz[item['axis']]*s.sympify(item['monomial'], locals={'x':x,'y':y,'z':z})
    d = s.Poly(field,*xyz).total_degree()
    lap = sum(s.diff(field,v,2) for v in xyz)
    expected.append((s.lambdify(xyz,field,'math'),s.lambdify(xyz,lap,'math'),d))
errors = [0.0]*3
for row in actual['controls']:
    item = basis[row['col']]
    n = [.36,-.48,.8] if row['dir'] else [1,0,0]
    fn,lap,d = expected[row['col']]
    rate = 2*(lap(*n)+(2*row['sigma']-4-d*(d+1))*fn(*n))/row['a']**3 if item['m']==1 else 0.0
    for k in range(3):
        errors[k] = max(errors[k],abs(row['N1_t_actual'][k]-rate))
assert max(errors)<2e-5
report = {'passed_read_only_independent_spatial_pullback_review': True,
          'scientific_index_sha256': PIN, 'scientific_files_verified': len(index['files']),
          'source_inputs_verified': 374, 'commands_all_zero': 6,
          'release_ASan_UBSan_debug_byte_equal': True,
          'fields': 120, 'controls': 1920, 'sampled_actual_kernel_points': 23040,
          'finite_coordinate_pullback_radial_proof': radial,
          '100_digit_radial_limit_checks': numeric,
          'angular_rate': str(angular_rate), 'actual_angular_N1_errors': errors,
          'actual_summary': actual['summary'],
          'boundary_errors': {key: max(abs(t[key]) for t in actual['controls'])
                              for key in ['initial_N0_boundary','initial_N1_boundary',
                                          'initial_Q0_boundary','next_R0_error']},
          'frame_distinction': 'For xi=Omega X e_j, f=xi.dOmega/Omega -> -Y/a and delta q_AB=2Y q_AB/a. Fixing this frame is an additional unproved restriction, not a consequence of initial Einstein constraints or smoothness.',
          'scope': 'Linear smooth-time quadratic-null compatibility obstruction on exact-Einstein initial pullback tangents. No finite-Omega amplitude instability, universal candidate rejection, invariant fixed-frame restriction, radiative-data closure, PDE stability or evolution admission inferred.',
          'python': sys.version, 'sympy_version': s.__version__,
          'sympy_module_sha256': sha(Path(s.__file__)), 'mpmath_version': mp.__version__}
(P/'review-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
