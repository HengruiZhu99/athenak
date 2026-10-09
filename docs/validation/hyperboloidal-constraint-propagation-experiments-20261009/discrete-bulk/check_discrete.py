"""Actual full20 bulk stencil audit; no boundary or nonlinear stability claim."""
import json
from pathlib import Path
import subprocess
import sys
import importlib.util

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    'original_basis', ROOT / 'tst/hyperboloidal/check_kernel_symbol.py')
basis_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(basis_module)


def z(a):
    a = np.asarray(a)
    return a[..., 0] + 1j*a[..., 1]


def canonical_matrix(alpha, w):
    f = 1 + 2*(1-w)/alpha
    mu = (1-w)*3/8+w
    m = np.zeros((20, 20))
    m[:8, :8] = basis_module.scalar_matrix(f, mu, w, w/2)
    vector = [[0, -2, 0, 1], [-1/2, 0, 1/2, 0],
              [0, 0, 0, 1], [0, 0, mu, 0]]
    for s in [8, 12]:
        m[s:s+4, s:s+4] = vector
    for s in [16, 18]:
        m[s:s+2, s:s+2] = [[0, -2], [-1/2, 0]]
    return m


def weight(r):
    if r <= .45:
        return 0.
    if r >= .85:
        return 1.
    s = (r-.45)/.4
    g = -1/s+1/(1-s)
    return 1/(1+np.exp(-g))


def transform(case, p):
    """The actual kernel's orthonormal, derivative-based constrained reduction."""
    alpha, chi = case['alpha'], case['chi']
    b = np.array([[1.3, .2, -.1], [0, .8, .12], [0, 0, 1/1.04]])
    if not case['spd']:
        b = np.eye(3)
    g = b.T@b
    e0 = b/np.sqrt(chi)
    v = np.linalg.solve(e0.T, p)
    s = np.linalg.norm(v)
    n = v/s
    transverse = np.eye(3)[np.argmin(abs(n))]
    t = transverse-np.dot(n, transverse)*n
    t /= np.linalg.norm(t)
    q = np.stack([n, t, np.cross(n, t)])
    e = q@e0
    inv = np.linalg.inv(e)
    gi = np.linalg.inv(g)
    tr = np.zeros((20, 20), dtype=complex)
    ti, tj = [0, 0, 0, 1, 1], [0, 1, 2, 1, 2]
    for col in range(20):
        h, a = np.zeros((3, 3)), np.zeros((3, 3))
        if 7 <= col < 12:
            i, j = ti[col-7], tj[col-7]
            h[i, j] = h[j, i] = 1
            h[2, 2] = -np.sum(gi*h)/gi[2, 2]
        if 12 <= col < 17:
            i, j = ti[col-12], tj[col-12]
            a[i, j] = a[j, i] = 1
            a[2, 2] = -np.sum(gi*a)/gi[2, 2]
        hf, af = inv.T@h@inv/chi, inv.T@a@inv/chi
        lf, bf = np.zeros(3), np.zeros(3)
        if col >= 17:
            lf = chi*e[:, col-17]
        if 4 <= col < 7:
            bf = e[:, col-4]/alpha
        tr[0, col] = 1j*s/alpha if col == 0 else 0
        tr[1, col] = 1j*s/chi if col == 1 else 0
        tr[2, col], tr[5, col] = 1j*s*hf[0, 0], af[0, 0]
        tr[3, col], tr[4, col] = float(col == 2), float(col == 3)
        tr[6, col], tr[7, col] = lf[0], 1j*s*bf[0]
        for v in range(2):
            start = 8+4*v
            tr[start:start+4, col] = [1j*s*hf[0, 1+v], af[0, 1+v],
                                     lf[1+v], 1j*s*bf[1+v]]
        tr[16:20, col] = [1j*s*(hf[1, 1]-hf[2, 2])/2,
                          (af[1, 1]-af[2, 2])/2,
                          1j*s*hf[1, 2], af[1, 2]]
    return tr, s


def exact_and_manufactured():
    x = sp.symbols('x')
    d = {-2: sp.Rational(1, 12), -1: -sp.Rational(2, 3),
         1: sp.Rational(2, 3), 2: -sp.Rational(1, 12)}
    c = {k: sum(a*b for i, a in d.items() for j, b in d.items()
                if i+j == k) for k in range(-4, 5)}
    expected = [1, -16, 64, 16, -130, 16, 64, -16, 1]
    assert [c[k] for k in range(-4, 5)] == [sp.Rational(v, 144) for v in expected]
    for n in range(6):
        target = 2 if n == 2 else 0
        assert sum(c[k]*k**n for k in c) == target
    # exp(x): composed second derivative has leading -h^4/15 error.
    assert sp.simplify(sum(c[k]*k**6/sp.factorial(6) for k in c)) == -sp.Rational(1, 15)
    hs = [.2, .1, .05, .025]
    errors = [abs(sum(float(c[k])*np.exp(k*h) for k in c)/h**2-1) for h in hs]
    ratios = [errors[i]/errors[i+1] for i in range(len(errors)-1)]
    assert min(ratios) > 15.8 and max(ratios) < 16.3, (errors, ratios)
    theta = sp.symbols('theta', real=True)
    t = sp.symbols('t', real=True)
    # Substitute cos(theta)=1-2t and sin²(theta)=4t(1-t).
    native = (-2*(2*(1-2*t)**2-1)+32*(1-2*t)-30)/12
    composed = -4*t*(1-t)*(4-(1-2*t))**2/9
    delta = sp.factor(native-composed)
    assert sp.simplify(delta+16*t**3*(2+t)/9) == 0
    return {'coefficients_over_144': expected, 'radius': 4,
            'manufactured_exp_h': hs, 'manufactured_exp_errors': errors,
            'manufactured_exp_ratios': ratios,
            'delta_h2': str(delta), 'composed_leading_error': '-h^4 f^(6)/15'}


def main(path):
    data = json.loads(Path(path).read_text())
    maximum = {'D_symbol': 0., 'S_symbol': 0., 'KO_symbol': 0.,
               'upwind_symbol': 0., 'upwind_positive_real': 0., 'canonical_matrix': 0.,
               'complete_basis_residual': 0., 'complete_basis_condition': 0.,
               'flat_gauge_formula': 0., 'compatible_gauge_leakage': 0.,
               'common_scalar_basis_residual': 0., 'nyquist_nilpotence': 0.}
    basis_count = gauge_count = zero_count = 0
    jordan_bound = {'h_max': .125, 'ko_epsilon': .1,
                    'max_propagator_Dplus_norm_sampled': 0.,
                    'max_uniform_h_analytic_bound': 0.,
                    'maximum_roundoff_removed': 0.}
    native_peak = {'H': 0., 'M': 0., 'Z': 0.}
    velocity = np.array([.3, -.2, .1])
    for case in data:
        theta, h = np.array(case['theta']), case['h']
        d, S, L, Q = z(case['D']), z(case['S']), z(case['L']), z(case['Q'])
        p = np.sin(theta)*(4-np.cos(theta))/(3*h)
        expected_S = -np.outer(p, p).astype(complex)
        if not case['compatible']:
            for i in range(3):
                expected_S[i, i] = (-2*np.cos(2*theta[i])+32*np.cos(theta[i])-30)/(12*h*h)
        maximum['D_symbol'] = max(maximum['D_symbol'], float(np.max(abs(d-1j*p))))
        maximum['S_symbol'] = max(maximum['S_symbol'], float(np.max(abs(S-expected_S))))
        ko = z(case['KO'])
        maximum['KO_symbol'] = max(maximum['KO_symbol'],
                                   float(abs(ko+sum(np.sin(theta/2)**6)/h)))
        up = z(case['upwind'])
        t = np.sin(theta/2)**2
        up_exact = np.sum(-8*abs(velocity)*t**3/(3*h)
                          +1j*velocity*np.sin(theta)
                          *(np.cos(theta)**2-3*np.cos(theta)+5)/(3*h))
        maximum['upwind_symbol'] = max(maximum['upwind_symbol'], float(abs(up-up_exact)))
        maximum['upwind_positive_real'] = max(maximum['upwind_positive_real'], up.real)
        # Constant flat, alpha=chi=1 is an independent exact full-kernel C_hL_h gate.
        if not case['spd'] and case['alpha'] == case['chi'] == 1:
            CL = np.einsum('ij,jk->ik', Q, L)
            delta = np.diag(S)-d*d
            wanted = np.zeros((8, 4), complex)  # alpha,beta_x,beta_y,beta_z
            for i in range(3):
                wanted[1+i, 0] = d[i]*sum(delta[j] for j in range(3) if j != i)
                wanted[0, 1+i] = -2*d[i]*sum(delta[j] for j in range(3) if j != i)
                wanted[4+i, 1+i] = sum(delta)/2+delta[i]/6
            observed = CL[:, [0, 4, 5, 6]]
            maximum['flat_gauge_formula'] = max(maximum['flat_gauge_formula'],
                                                float(np.max(abs(wanted-observed))))
            if case['compatible']:
                maximum['compatible_gauge_leakage'] = max(
                    maximum['compatible_gauge_leakage'], float(np.max(abs(observed))))
            else:
                native_peak['H'] = max(native_peak['H'], float(np.max(abs(observed[0]))))
                native_peak['M'] = max(native_peak['M'], float(np.max(abs(observed[1:4]))))
                native_peak['Z'] = max(native_peak['Z'], float(np.max(abs(observed[4:7]))))
            gauge_count += 1
        if not case['compatible']:
            continue
        if np.linalg.norm(p) < 1e-11:
            # At exactly zero modified covector primitive lower-order nilpotent
            # links survive. KO shifts the Jordan block; it does not diagonalize it.
            maximum['nyquist_nilpotence'] = max(maximum['nyquist_nilpotence'],
                                                float(np.max(abs(np.einsum('ij,jk->ik', L, L)))))
            if np.any(abs(theta) > 1):
                assert (.1*ko+up).real < 0
                N = L.copy()
                N[abs(N) < 1e-11] = 0  # exact sin(pi)=0, discard trig roundoff
                jordan_bound['maximum_roundoff_removed'] = max(
                    jordan_bound['maximum_roundoff_removed'], float(np.max(abs(N-L))))
                np.testing.assert_allclose(np.einsum('ij,jk->ik', N, N), 0, atol=1e-14)
                positions = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]
                momenta = [i for i in range(20) if i not in positions]
                assert not np.any(N[:, positions])
                assert not np.any(N[momenta, :])
                m = int(np.count_nonzero(abs(theta) > 1))
                norm_N = np.linalg.norm(N, 2)
                # ||e^(-gamma t)(I+t N)|| <= 1+||N||/(e gamma).
                # In ||u||_Dplus, position weight zeta=sqrt(1+4m/h²),
                # N maps momenta->positions, so ||N||_Dplus=zeta||N||.
                # gamma >= epsilon*m/h. This bound is uniform for 0<h<=h_max.
                bound = 1+norm_N*np.sqrt(.125**2+4*m)/(.1*m*np.e)
                jordan_bound['max_uniform_h_analytic_bound'] = max(
                    jordan_bound['max_uniform_h_analytic_bound'], float(bound))
                for hj in [.125, .0625, .015625, .00390625]:
                    zeta = np.sqrt(1+4*m/hj**2)
                    for tau in [0, .01, .1, .3, 1, 3, 10, 30, 100]:
                        # t=h*tau, conservative KO only, upwind omitted here.
                        propagator = np.exp(-.1*m*tau)*(np.eye(20)+hj*tau*zeta*N)
                        norm = float(np.linalg.norm(propagator, 2))
                        assert norm <= bound*(1+2e-14)
                        jordan_bound['max_propagator_Dplus_norm_sampled'] = max(
                            jordan_bound['max_propagator_Dplus_norm_sampled'], norm)
            zero_count += 1
            continue
        T, s = transform(case, p)
        reduced = np.einsum('ij,jk,kl->il', T, L, np.linalg.inv(T))
        alpha, w = case['alpha'], weight(case['r'])
        M = canonical_matrix(alpha, w)
        target = 1j*alpha*s*M+1j*np.dot(velocity, p)*np.eye(20)
        maximum['canonical_matrix'] = max(maximum['canonical_matrix'],
                                         float(np.max(abs(reduced-target))))
        B, speeds = basis_module.canceled_basis(alpha, w)
        maximum['complete_basis_residual'] = max(maximum['complete_basis_residual'],
            float(np.max(abs(np.einsum('ij,jk->ik', B, reduced)-(1j*alpha*s*speeds+1j*np.dot(velocity,p))[:, None]*B))))
        maximum['complete_basis_condition'] = max(maximum['complete_basis_condition'],
                                                  float(np.linalg.cond(B)))
        assert np.linalg.matrix_rank(B) == 20
        shifted = reduced+(up-1j*np.dot(velocity, p)+.1*ko)*np.eye(20)
        eigenvalues = 1j*alpha*s*speeds+up+.1*ko
        maximum['common_scalar_basis_residual'] = max(maximum['common_scalar_basis_residual'],
            float(np.max(abs(np.einsum('ij,jk->ik', B, shifted)-eigenvalues[:, None]*B))))
        assert max(eigenvalues.real) < 1e-12
        basis_count += 1
    assert maximum['D_symbol'] < 3e-14, maximum
    assert maximum['S_symbol'] < 3e-13, maximum
    assert maximum['KO_symbol'] < 2e-14, maximum
    assert maximum['upwind_positive_real'] < 2e-14, maximum
    assert maximum['upwind_symbol'] < 3e-14, maximum
    assert maximum['canonical_matrix'] < 3e-10, maximum
    assert maximum['complete_basis_residual'] < 3e-10, maximum
    assert maximum['common_scalar_basis_residual'] < 3e-10, maximum
    assert maximum['flat_gauge_formula'] < 2e-10, maximum
    assert maximum['compatible_gauge_leakage'] < 2e-10, maximum
    assert maximum['nyquist_nilpotence'] < 3e-10, maximum
    manufactured = json.loads(HERE.joinpath('manufactured.json').read_text())
    manufactured_errors = []
    for sample in manufactured:
        a = np.array([.7, -.4, .2])
        exact = sample['exact_value']*np.outer(a, a)
        manufactured_errors.append(float(np.max(abs(sample['S']-exact))))
    manufactured_ratios = [manufactured_errors[i]/manufactured_errors[i+1]
                           for i in range(len(manufactured_errors)-1)]
    assert min(manufactured_ratios) > 15.8 and max(manufactured_ratios) < 16.3
    report = {'passed': True, 'full20_cases': len(data),
              'complete_nonzero_modified_covector_cases': basis_count,
              'flat_gauge_constraint_cases': gauge_count,
              'zero_modified_covector_cases': zero_count,
              'maximum_errors': maximum, 'native_flat_gauge_defect_peaks': native_peak,
              'nyquist_ko_damped_Jordan': jordan_bound,
              'exact_stencil_and_manufactured': exact_and_manufactured(),
              'actual_Dx_composition_manufactured_errors': manufactured_errors,
              'actual_Dx_composition_manufactured_ratios': manufactured_ratios,
              'scope': 'Constant-coefficient interior principal and gauge-constraint columns; '
                       'no nonflat product rule, ghost, nonlinear or global energy claim.'}
    HERE.joinpath('check-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
