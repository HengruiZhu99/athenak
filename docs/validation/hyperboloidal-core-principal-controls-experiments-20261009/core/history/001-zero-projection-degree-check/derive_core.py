"""Exact regular total-J envelope action of the frozen independent flat core.

No radial discretization, boundary map, spectrum or evolution is constructed.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import sympy as s

P = Path(__file__).resolve().parent
pins = json.loads((P / 'source-pins.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for record in pins.values():
    assert sha(record['path']) == record['sha256'], record['path']
spec = importlib.util.spec_from_file_location('frozen_totalj', pins['basis_python']['path'])
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
x = b.xyz
rho_xyz = b.rho
rho = s.Symbol('rho', nonnegative=True)
w = s.symbols('W W_rho W_rhorho', real=True)
sqrt3 = s.sqrt(3)
started = time.monotonic()


def zero_jet():
    return (s.S.Zero, (s.S.Zero,) * 3, ((s.S.Zero,) * 3,) * 3)


def jet(poly):
    value = poly * w[0]
    d = tuple(s.diff(poly, x[i]) * w[0] + 2*x[i]*poly*w[1] for i in range(3))
    dd = tuple(tuple(s.diff(poly, x[i], x[j])*w[0]
                    + 2*(x[i]*s.diff(poly, x[j]) + x[j]*s.diff(poly, x[i])
                         + (poly if i == j else 0))*w[1]
                    + 4*x[i]*x[j]*poly*w[2] for j in range(3)) for i in range(3))
    return value, d, dd


def lap(v):
    return sum(v[2][i][i] for i in range(3))


def stf(matrix):
    trace = sum(matrix[3*i+i] for i in range(3))
    return tuple(s.expand(v - (trace/3 if k in (0, 4, 8) else 0)) for k, v in enumerate(matrix))


def flat_action(J, m, column):
    """Independent physical-variable rewrite of frozen full22 FlatFormula.

    tau = tr(delta bar-gamma)/sqrt(3); delta chi=-tau/sqrt(3),
    delta g=h_STF, and independent_A is STF at the exact Minkowski core.
    P and Theta are the actual stored P/physical Theta, without a redefinition.
    """
    layout = b.channel_layout(J)
    seed = layout[column]
    fields = {'alpha': [zero_jet()], 'metric_trace': [zero_jet()],
              'P': [zero_jet()], 'Theta_phys': [zero_jet()],
              'beta': [zero_jet() for _ in range(3)],
              'Lambda': [zero_jet() for _ in range(3)],
              'metric_STF': [zero_jet() for _ in range(9)],
              'independent_A': [zero_jet() for _ in range(9)]}
    fields[seed['name']] = [jet(poly) for poly in b.basis(J, m, seed['spin'], seed['L'])]
    a, tau, p, th = (fields[k][0] for k in ('alpha', 'metric_trace', 'P', 'Theta_phys'))
    beta, lam, h, A = (fields[k] for k in ('beta', 'Lambda', 'metric_STF', 'independent_A'))
    divbeta = sum(beta[i][1][i] for i in range(3))
    divlambda = sum(lam[i][1][i] for i in range(3))
    out = {
        'alpha': (-3*p[0],),
        'metric_trace': (-2*(p[0]+2*th[0]-divbeta)/sqrt3,),
        'P': (-lap(a)+10*th[0],),
        'Theta_phys': (-lap(tau)/sqrt3 + divlambda/2 - 20*th[0],),
        'beta': tuple(3*lam[i][0]/8 for i in range(3)),
        'Lambda': tuple(lap(beta[i]) + sum(beta[j][2][i][j] for j in range(3))/3
                        - 4*p[1][i]/3 - 2*th[1][i]/3
                        - 10*(lam[i][0]-sum(h[3*i+j][1][j] for j in range(3)))
                        for i in range(3)),
        'metric_STF': tuple(-2*A[3*i+j][0] + beta[j][1][i] + beta[i][1][j]
                            - (2*divbeta/3 if i == j else 0)
                            for i in range(3) for j in range(3)),
    }
    apart = stf(tuple(-a[2][i][j] - tau[2][i][j]/(2*sqrt3)
                     + (lam[j][1][i]+lam[i][1][j])/2
                     for i in range(3) for j in range(3)))
    out['independent_A'] = tuple(apart[k]-lap(h[k])/2 for k in range(9))
    return {kind: tuple(s.expand(v) for v in values) for kind, values in out.items()}


def homogeneous(values):
    result = {}
    for i, expr in enumerate(values):
        for powers, coefficient in s.Poly(expr, *x).terms():
            if coefficient == 0:
                continue
            degree = sum(powers)
            if degree not in result:
                result[degree] = [s.S.Zero] * len(values)
            result[degree][i] += coefficient * s.prod(x[k]**powers[k] for k in range(3))
    return {degree: tuple(values) for degree, values in result.items()}


def project(J, column, action):
    """Exact homogeneous polynomial angular projection; no radial division."""
    layout = b.channel_layout(J)
    coeff = {}
    for derivative in range(3):
        homogeneous_by_kind = {kind: homogeneous(tuple(s.expand(expr).coeff(w[derivative]) for expr in values))
                               for kind, values in action.items()}
        reconstructed = {kind: [s.S.Zero] * len(values) for kind, values in action.items()}
        for row, output in enumerate(layout):
            poly = b.basis(J, 0, output['spin'], output['L'])
            for degree, values in homogeneous_by_kind[output['name']].items():
                difference = degree-output['L']
                if difference < 0 or difference % 2:
                    continue
                power = difference//2
                assert power <= 2, 'Unexpected rho degree; no workaround admitted'
                amplitude = s.simplify(b.inner(poly, values))
                if amplitude == 0:
                    continue
                assert amplitude.is_real is True, amplitude
                coeff[row, derivative, power] = amplitude
                for k, component in enumerate(poly):
                    reconstructed[output['name']][k] += amplitude*rho_xyz**power*component
        for kind, actual in action.items():
            expected = tuple(s.expand(expr).coeff(w[derivative]) for expr in actual)
            for lhs, rhs in zip(reconstructed[kind], expected):
                assert s.Poly(s.expand(lhs-rhs), *x).is_zero, (J, column, derivative, kind, lhs-rhs)
    return coeff


def verify_m(J, column, coefficients, m):
    layout = b.channel_layout(J)
    action = flat_action(J, m, column)
    reconstructed = {kind: [s.S.Zero]*len(values) for kind, values in action.items()}
    for (row, derivative, power), amplitude in coefficients.items():
        output = layout[row]
        poly = b.basis(J, m, output['spin'], output['L'])
        for k, component in enumerate(poly):
            reconstructed[output['name']][k] += amplitude*rho_xyz**power*w[derivative]*component
    for kind, actual in action.items():
        for lhs, rhs in zip(reconstructed[kind], actual):
            assert s.Poly(s.expand(lhs-rhs), *x, *w).is_zero, (J, m, column, kind)
    return len(action)


blocks = {}
records = []
counts = defaultdict(int)
for J in range(3):
    layout = b.channel_layout(J)
    n = len(layout)
    assert n == (8, 16, 20)[J]
    numerical = np.zeros((3, 3, n, n))  # derivative, rho power, output, input
    all_coefficients = []
    for column in range(n):
        action = flat_action(J, 0, column)
        coefficients = project(J, column, action)
        all_coefficients.append(coefficients)
        for (row, derivative, power), amplitude in sorted(coefficients.items()):
            assert not amplitude.has(*x, *w, rho)
            numerical[derivative, power, row, column] = float(amplitude)
            records.append({'J': J, 'output': row, 'input': column, 'derivative': derivative,
                            'rho_power': power, 'exact': str(amplitude), 'real': float(amplitude)})
        for m in range(-J, J+1):
            verify_m(J, column, coefficients, m)
            counts['exact_all_m_input_columns'] += 1
            counts['exact_all_m_independent_jet_actions'] += 3
        counts['primary_columns'] += 1
        print('exact J%d column%d/%d' % (J, column+1, n), flush=True)
    blocks['J%d' % J] = numerical
np.savez_compressed(P/'core-envelope-blocks.npz', **blocks)

header = ['// Exact flat C0/core total-J envelope formula; no radial discretization.',
          '#ifndef RESEARCH_TOTALJ_FLAT_CORE_ENVELOPE_HPP_',
          '#define RESEARCH_TOTALJ_FLAT_CORE_ENVELOPE_HPP_', '#include <array>', '#include <stdexcept>',
          'namespace core_envelope {',
          'struct Entry {int J,out,in,derivative,power; double coefficient;};',
          'inline constexpr Entry entries[] = {']
for record in records:
    header.append('  {%d,%d,%d,%d,%d,%.17g},' % tuple(record[k] for k in ('J','output','input','derivative','rho_power','real')))
header += ['};', 'inline int Channels(int J) {if(J<0||J>2) throw std::invalid_argument("J"); return J==0?8:(J==1?16:20);}',
           'template<class T> std::array<T,20> Apply(int J,T rho,const std::array<std::array<T,20>,3>&w) {',
           '  Channels(J); std::array<T,20> out{}; const std::array<T,3> powers={T(1),rho,rho*rho};',
           '  for(const auto&e:entries) if(e.J==J) out[e.out]+=T(e.coefficient)*powers[e.power]*w[e.derivative][e.in];',
           '  return out;', '}', '} // namespace core_envelope', '#endif', '']
(P/'core_envelope.hpp').write_text('\n'.join(header))

report = {'passed_exact_flat_core_envelope_derivation': True, 'scope': json.loads((P/'plan.json').read_text())['scope'],
          'source_sha256': sha(__file__), 'plan_sha256': sha(P/'plan.json'), 'pins_sha256': sha(P/'source-pins.json'),
          'runtime_seconds': time.monotonic()-started, 'sympy': s.__version__, 'python': sys.version,
          'counts': dict(counts), 'sparse_polynomial_entries': len(records), 'maximum_rho_degree': max(r['rho_power'] for r in records),
          'all_exact_cartesian_residuals_zero': True, 'r_or_rho_denominators': False,
          'imposed_cross_L_envelope_conditions': False, 'all_negative_m_directly_verified': True,
          'channel_layouts': {str(J): b.channel_layout(J) for J in range(3)}, 'entries': records,
          'block_npz_sha256': sha(P/'core-envelope-blocks.npz'), 'generated_header_sha256': sha(P/'core_envelope.hpp'),
          'actual_radial_PDE_global_matrix_boundary_or_evolution_admitted': False}
(P/'symbolic-report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key: value for key, value in report.items() if key not in ('entries','channel_layouts')}, indent=2), flush=True)
