"""HELD native-time Gaussian inverse/admissibility screen; stdlib until admission."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s:
                      (_ for _ in ()).throw(ValueError(s)))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('changed pinned input '+path)


def execute(recipe, out):
    sys.path.insert(0, recipe['mpmath_parent'])
    import mpmath as mp
    if str(Path(mp.__file__).resolve()) != recipe['mpmath_init']:
        raise RuntimeError('unexpected imported mpmath origin')
    sys.path.insert(0, str(HERE))
    import values_context as height_context
    import gaussian_radial as radial_context
    if Path(height_context.__file__).resolve() != HERE/'values_context.py':
        raise RuntimeError('unexpected height module origin')
    if Path(radial_context.__file__).resolve() != HERE/'gaussian_radial.py':
        raise RuntimeError('unexpected radial module origin')
    Layer = height_context.Layer
    derivatives, radial = radial_context.functions(mp)

    def rat(value):
        f = Fraction(value)
        return mp.mpf(f.numerator)/f.denominator

    def number(value):
        if not mp.isfinite(value):
            raise ArithmeticError('nonfinite multiprecision result')
        return mp.nstr(value, mp.mp.dps)

    def scaled(x, y):
        return abs(x-y)/max(1, abs(x), abs(y))

    def invert(function, lo, hi, center, lower_j, kind):
        """Numerical sign bracket, 16 safeguarded Newton + <=512 bisections.

        A local sign-probed bracket is tried once Newton's residual and the
        global positive derivative bound make its prescribed width plausible.
        Computed signs are not rigorous interval arithmetic enclosures.
        """
        tolerance = rat(recipe['root_absolute_tolerance'])
        width_tolerance = rat(recipe['root_width_tolerance'])
        calls = 0

        def evaluate(x):
            nonlocal calls
            calls += 1
            g, j = function(x)
            if not (mp.isfinite(g) and mp.isfinite(j) and j > 0):
                raise ArithmeticError('nonfinite or nonpositive inverse Jacobian')
            return g, j

        glo, _ = evaluate(lo)
        ghi, _ = evaluate(hi)
        if not (lo <= hi and glo <= 0 <= ghi):
            raise ArithmeticError('initial inverse numerical bracket signs fail')
        initial = dict(lower=number(lo), upper=number(hi),
                       g_lower=number(glo), g_upper=number(ghi))
        if lo == hi:
            if abs(glo) > tolerance:
                raise ArithmeticError('analytic collapsed-bracket residual fails')
            return lo, dict(kind=kind, initial=initial, final=initial,
                            residual=number(abs(glo)), width='0.0',
                            newton_steps=0, bisection_steps=0, evaluations=calls,
                            formal_interval_enclosure=False)
        x = min(hi, max(lo, center))
        newton_steps = 0
        bisection_steps = 0
        for phase, count in [('newton', recipe['newton_iterations']),
                             ('bisection', recipe['bisection_iterations'])]:
            for _ in range(count):
                g, j = evaluate(x)
                if phase == 'newton':
                    newton_steps += 1
                else:
                    bisection_steps += 1
                if g < 0:
                    lo, glo = x, g
                elif g > 0:
                    hi, ghi = x, g
                # Avoid hundreds of bisections after Newton has located the
                # root: confirm both signs at a fixed quarter-width distance.
                if abs(g) <= lower_j*width_tolerance/8:
                    trial_lo = max(lo, x-width_tolerance/4)
                    trial_hi = min(hi, x+width_tolerance/4)
                    trial_glo, _ = evaluate(trial_lo)
                    trial_ghi, _ = evaluate(trial_hi)
                    if trial_glo <= 0 <= trial_ghi:
                        lo, hi, glo, ghi = trial_lo, trial_hi, trial_glo, trial_ghi
                if hi-lo <= width_tolerance:
                    answer = (lo+hi)/2
                    residual, _ = evaluate(answer)
                    if abs(residual) <= tolerance:
                        return answer, dict(
                            kind=kind, initial=initial,
                            final=dict(lower=number(lo), upper=number(hi),
                                       g_lower=number(glo), g_upper=number(ghi)),
                            residual=number(abs(residual)), width=number(hi-lo),
                            newton_steps=newton_steps, bisection_steps=bisection_steps,
                            evaluations=calls, formal_interval_enclosure=False)
                if phase == 'newton':
                    proposal = x-g/j
                    x = proposal if lo < proposal < hi else (lo+hi)/2
                else:
                    x = (lo+hi)/2
        raise ArithmeticError('fixed inverse iteration/residual/width gate fails')

    def compact_wave(u, z, sigma):
        advanced = u+2/z
        f, g = derivatives(u, sigma, 3), derivatives(advanced, sigma, 3)
        phi = f[2]-g[2]+3*z*(f[1]+g[1])+3*z*z*(f[0]-g[0])
        plus = (-f[2]-6*z*f[1]-9*z*z*f[0]-2*g[3]/z
                +7*g[2]-12*z*g[1]+9*z*z*g[0])
        minus = (2*f[3]+7*z*f[2]+12*z*z*f[1]+9*z**3*f[0]
                 -z*g[2]+6*z*z*g[1]-9*z**3*g[0])
        return phi, plus, minus, advanced

    def sphere_frame(p):
        n1 = mp.sqrt((1+mp.sqrt(1-4*p*p))/2)
        n = [n1, p/n1, mp.mpf(0)]
        tangent = [n[1]-2*p*n[0], n[0]-2*p*n[1], mp.mpf(0)]
        return n, tangent

    def adm_values(omega, lapse_ref, L, J, D, wr, wt, n,
                   compact=None):
        # Values only: no curvature, P/A/Lambda jets or time rates are claimed.
        if not D > 0:
            return None
        pi = [[mp.mpf(i == j)-n[i]*n[j] for j in range(3)] for i in range(3)]
        if compact is None:
            lapse = omega/mp.sqrt(D)
            detbar = L*L*D/(omega*omega*J*J)
            rr = L*L/(omega*omega)*(1-wr*wr/(J*J))
            gamma = [[pi[i][j]+rr*n[i]*n[j]
                      -L*wr/(omega*J*J)*(n[i]*wt[j]+wt[i]*n[j])
                      -wt[i]*wt[j]/(J*J) for j in range(3)] for i in range(3)]
            beta = [-omega*omega*wr/(L*D)*n[i]-omega*wt[i]/D for i in range(3)]
        else:
            r, z, delta, km, jp, epsilon, phi, tangent = compact
            lapse = r/mp.sqrt(delta)
            detbar = L*L*delta/(r*r*J*J)
            gamma = [[pi[i][j]+L*L*km*jp/(r*r*J*J)*n[i]*n[j]
                      +epsilon*L*z*wr*phi/(r*J*J)
                      *(n[i]*tangent[j]+tangent[i]*n[j])
                      -epsilon*epsilon*z**4*phi*phi/(J*J)*tangent[i]*tangent[j]
                      for j in range(3)] for i in range(3)]
            beta = [-r*r*wr/(L*delta)*n[i]
                    +epsilon*omega*phi/delta*tangent[i] for i in range(3)]
        chi = detbar**(-mp.mpf(1)/3)
        gtilde = [[chi*gamma[i][j] for j in range(3)] for i in range(3)]
        return dict(alpha_bar=number(lapse), alpha_bar_over_reference=number(lapse/lapse_ref),
                    beta=[number(v) for v in beta], chi=number(chi),
                    det_bar_gamma=number(detbar),
                    bar_gamma=[[number(v) for v in row] for row in gamma],
                    gtilde=[[number(v) for v in row] for row in gtilde])

    comparisons = {}
    profile_summaries = []
    level_context = []
    total_rows = 0
    maximum_outer_identity = mp.mpf(0)
    maximum_inverse_residual = mp.mpf(0)
    maximum_inverse_width = mp.mpf(0)
    maximum_original_map_residual = mp.mpf(0)
    started = time.monotonic()
    with (out/'samples.jsonl').open('x') as sink:
        for level in recipe['levels']:
            digits, order, terms = level['digits'], level['height_order'], level['series_terms']
            mp.mp.dps = digits
            layer = Layer(recipe['layer'], order)
            a = rat(recipe['a'])
            level_key = str(digits)+'-'+str(order)
            current = {}
            minimum_by_profile = {}
            level_context.append(dict(level=level_key, digits=digits, height_order=order,
                                      series_terms=terms, outer_height_constant=number(layer.outer_constant)))
            for sigma_s in recipe['sigma']:
                sigma = rat(sigma_s)
                for eps_s in recipe['epsilon']:
                    epsilon = rat(eps_s)
                    lower_j = 1-4*epsilon/mp.pi
                    if not lower_j > 0:
                        raise ArithmeticError('fixed Gaussian global J lower bound fails')
                    for radius in recipe['native_radii']:
                        r = rat(radius['value'])
                        omega, _, b, lapse_ref, L = layer.reference(r)
                        R = r/omega
                        h = b/lapse_ref
                        d0 = (omega/lapse_ref)**2
                        outer = r >= layer.r1
                        for t_s in recipe['native_times']:
                            native_t = rat(t_s)
                            for p_s in recipe['p_values']:
                                p = rat(p_s)
                                n, tangent = sphere_frame(p)
                                if outer:
                                    z = omega/r
                                    s = native_t+layer.outer_constant
                                    center = a*a/(mp.sqrt(1+a*a*z*z)+1)
                                    bound = abs(epsilon*p)*(2*sigma*sigma
                                            +6*abs(z)*sigma**3+6*z*z*sigma**4)

                                    def equation(c_ret):
                                        u = s+z*c_ret
                                        phi, plus, minus, _ = compact_wave(u, z, sigma)
                                        J = 1+epsilon*p*z*(minus+z*plus)/2
                                        return c_ret+epsilon*p*phi-center, J

                                    c_ret, root = invert(equation, center-bound, center+bound,
                                                         center, lower_j, 'outer_c_ret')
                                    u = s+z*c_ret
                                    T = R+u
                                    phi, plus, minus, advanced = compact_wave(u, z, sigma)
                                    J = 1+epsilon*p*z*(minus+z*plus)/2
                                    wr = h+epsilon*p*z*(minus-z*plus)/2
                                    wt = [-epsilon*z*z*phi*v for v in tangent]
                                    eta = a*a/(mp.sqrt(1+a*a*z*z)*(mp.sqrt(1+a*a*z*z)+1))
                                    km = eta+epsilon*p*plus
                                    jp = 1+h+epsilon*p*z*minus
                                    tau = 1-4*p*p
                                    delta = km*jp-epsilon*epsilon*z*z*phi*phi*tau
                                    D = z*z*delta
                                    D_ratio = delta/(a*a*h*h)
                                    # Independent original vector-gradient definition,
                                    # including direct high-precision T-R/T+R evaluation.
                                    C, CT, CR, radial_branch = radial(T, R, sigma, terms)
                                    FT = R*R*p*CT
                                    grad = [R*C*(n[1] if i == 0 else n[0] if i == 1 else 0)
                                            +R*R*p*CR*n[i] for i in range(3)]
                                    w_direct = [h*n[i]-epsilon*grad[i] for i in range(3)]
                                    J_direct = 1+epsilon*FT
                                    D_direct = J_direct*J_direct-mp.fsum(v*v for v in w_direct)
                                    identity = scaled(D_direct/d0, D_ratio)
                                    maximum_outer_identity = max(maximum_outer_identity, identity)
                                    original_map_residual = abs((T-R)+epsilon*R*R*p*C
                                                                -(native_t+layer.defect(r)))
                                    adm = adm_values(omega, lapse_ref, L, J, D, wr, wt, n,
                                                    (r, z, delta, km, jp, epsilon, phi, tangent))
                                    inverse_unknown = c_ret
                                else:
                                    Y0 = native_t+layer.height(r)
                                    bound = 16*epsilon*sigma/(3*mp.pi)

                                    def equation(T):
                                        C, CT, _, _ = radial(T, R, sigma, terms)
                                        return T+epsilon*R*R*p*C-Y0, 1+epsilon*R*R*p*CT

                                    if R == 0 or epsilon == 0 or p == 0 or Y0 == 0:
                                        lo = hi = Y0
                                    else:
                                        lo, hi = max(mp.mpf(0), Y0-bound), Y0+bound
                                    T, root = invert(equation, lo, hi, Y0, lower_j,
                                                     'core_transition_physical_T')
                                    u = T-R
                                    advanced = T+R
                                    C, CT, CR, radial_branch = radial(T, R, sigma, terms)
                                    FT = R*R*p*CT
                                    FR = R*p*(2*C+R*CR)
                                    J = 1+epsilon*FT
                                    wr = h-epsilon*FR
                                    wt = [-epsilon*R*C*v for v in tangent]
                                    grad = [R*C*(n[1] if i == 0 else n[0] if i == 1 else 0)
                                            +R*R*p*CR*n[i] for i in range(3)]
                                    w_direct = [h*n[i]-epsilon*grad[i] for i in range(3)]
                                    D = J*J-mp.fsum(v*v for v in w_direct)
                                    D_ratio = D/d0
                                    identity = mp.mpf(0)
                                    original_map_residual = abs(T+epsilon*R*R*p*C-Y0)
                                    adm = adm_values(omega, lapse_ref, L, J, D, wr, wt, n)
                                    inverse_unknown = T
                                if not (J > 0 and T >= 0):
                                    raise ArithmeticError('future inverse/J condition fails')
                                maximum_inverse_residual = max(maximum_inverse_residual, rat(root['residual']))
                                maximum_inverse_width = max(maximum_inverse_width, rat(root['width']))
                                maximum_original_map_residual = max(maximum_original_map_residual,
                                                                    original_map_residual)
                                key = (sigma_s, eps_s, radius['label'], t_s, p_s)
                                metrics = dict(D_ratio=number(D_ratio), J=number(J),
                                               T=number(T), u=number(u), inverse_unknown=number(inverse_unknown),
                                               alpha_ratio=None if adm is None else adm['alpha_bar_over_reference'])
                                current[key] = metrics
                                record = dict(level=level_key, digits=digits, height_order=order,
                                              sigma=sigma_s, epsilon=eps_s, native_radius=radius,
                                              native_time=t_s, p=p_s, s_angular='1',
                                              physical_R=number(R), physical_T=number(T),
                                              retarded_u=number(u), advanced_v=number(advanced),
                                              D=number(D), D_over_reference=number(D_ratio),
                                              J=number(J), passed_positive=bool(D > 0),
                                              outer_factored=outer, radial_branch=radial_branch,
                                              outer_direct_identity=number(identity), inverse=root,
                                              original_map_residual=number(original_map_residual),
                                              ADM_values=adm, no_ADM_reason=None if adm else 'D_nonpositive')
                                sink.write(json.dumps(record, allow_nan=False)+'\n')
                                total_rows += 1
                                profile = (sigma_s, eps_s)
                                if profile not in minimum_by_profile:
                                    minimum_by_profile[profile] = dict(samples=0, all_positive=True,
                                                                      minimum=D_ratio, worst=record)
                                summary = minimum_by_profile[profile]
                                summary['samples'] += 1
                                summary['all_positive'] = summary['all_positive'] and D > 0
                                if D_ratio < summary['minimum']:
                                    summary.update(minimum=D_ratio, worst=record)
                        sink.flush()
                        save(out/'progress.json', dict(level=level_key, sigma=sigma_s, epsilon=eps_s,
                                                      native_radius=radius['label'], total_rows=total_rows))
            if len(current) != recipe['records_per_level']:
                raise ArithmeticError('fixed event registry count differs')
            comparisons[level_key] = current
            for profile, summary in minimum_by_profile.items():
                profile_summaries.append(dict(level=level_key, sigma=profile[0], epsilon=profile[1],
                                              samples=summary['samples'],
                                              all_sampled_D_positive=bool(summary['all_positive']),
                                              minimum_D_over_reference=number(summary['minimum']),
                                              worst=summary['worst']))
    mp.mp.dps = max(level['digits'] for level in recipe['levels'])
    maxima = []
    for kind, left, right in recipe['comparison_pairs']:
        maximum = mp.mpf(0)
        worst = None
        if set(comparisons[left]) != set(comparisons[right]):
            raise ArithmeticError('precision/height registry differs')
        for key in comparisons[left]:
            x, y = comparisons[left][key], comparisons[right][key]
            for name in ('D_ratio', 'J', 'T', 'u', 'inverse_unknown', 'alpha_ratio'):
                if (x[name] is None) != (y[name] is None):
                    raise ArithmeticError('precision/height D sign decision differs')
                if x[name] is None:
                    continue
                error = scaled(rat(x[name]), rat(y[name]))
                if error > maximum:
                    maximum, worst = error, dict(key=list(key), field=name,
                                                 left=x[name], right=y[name])
        maxima.append(dict(kind=kind, left=left, right=right,
                           maximum_scaled=number(maximum), worst=worst))
    height_max = mp.mpf(0)
    for kind, left, right in recipe['comparison_pairs']:
        x = next(row for row in level_context if row['level'] == left)
        y = next(row for row in level_context if row['level'] == right)
        height_max = max(height_max, scaled(rat(x['outer_height_constant']), rat(y['outer_height_constant'])))
    if total_rows != recipe['total_records']:
        raise ArithmeticError('fixed total event count differs')
    comparison_max = max(rat(row['maximum_scaled']) for row in maxima)
    tolerance = rat(recipe['comparison_tolerance'])
    passed = (comparison_max <= tolerance and height_max <= tolerance
              and maximum_outer_identity <= tolerance
              and maximum_original_map_residual <= rat(recipe['root_absolute_tolerance']))
    result = dict(checks_passed=bool(passed), total_records=total_rows,
                  records_per_level=recipe['records_per_level'], levels=level_context,
                  comparison_pairs=maxima, maximum_comparison_scaled=number(comparison_max),
                  maximum_height_constant_comparison=number(height_max),
                  maximum_outer_direct_identity=number(maximum_outer_identity),
                  maximum_inverse_residual=number(maximum_inverse_residual),
                  maximum_inverse_width=number(maximum_inverse_width),
                  maximum_original_map_residual=number(maximum_original_map_residual),
                  all_sampled_D_positive=all(row['all_sampled_D_positive'] for row in profile_summaries),
                  profiles=profile_summaries, negative_events_preserved=True,
                  formal_interval_certificate=False, global_timelike_proven=False,
                  jets_kernel_PDE_native_evolution_or_stability_accepted=False,
                  scope=recipe['scope'], seconds=time.monotonic()-started)
    save(out/'result.json', result)
    if not passed:
        raise ArithmeticError('fixed precision/height/outer identity gate fails')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--authorization', type=Path, required=True)
    ap.add_argument('--authorization-sha256', required=True)
    args = ap.parse_args()
    out = HERE/'attempt001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = dict(completed=False, returncode=1,
                   scope='Finite native-time manufactured Gaussian inverse/value screen only')
    pins = {}
    before = None
    try:
        if sys.flags.optimize != 0 or not sys.flags.isolated or not sys.dont_write_bytecode:
            raise RuntimeError('isolated unoptimized -B runtime required')
        if sha(args.authorization) != args.authorization_sha256:
            raise RuntimeError('authorization digest differs')
        auth = load(args.authorization)
        recipe, index = load(HERE/'recipe.json'), load(HERE/'source-index.json')
        if not (auth.get('native_Gaussian_inverse_screen_authorized') is True
                and auth.get('source_index_sha256') == sha(HERE/'source-index.json')
                and auth.get('recipe_sha256') == sha(HERE/'recipe.json')
                and auth.get('screen_source_sha256') == sha(__file__)):
            raise RuntimeError('exact root source/math authorization required')
        if sha(Path(sys.executable).resolve()) != recipe['python_runtime_sha256']:
            raise RuntimeError('actual interpreter digest differs')
        for key, value in recipe['environment'].items():
            if os.environ.get(key) != value:
                raise RuntimeError('fixed environment differs '+key)
        pins.update(recipe['pins'])
        for row in index['files']:
            pins[row['path']] = row['sha256']
        pins[str(HERE/'source-index.json')] = sha(HERE/'source-index.json')
        pins[str(args.authorization.resolve())] = args.authorization_sha256
        verify(pins)
        before = dict(pins)
        save(out/'pins-before.json', before)
        result = execute(recipe, out)
        receipt.update(completed=True, returncode=0, checks_passed=result['checks_passed'],
                       all_sampled_D_positive=result['all_sampled_D_positive'])
    except BaseException as exc:
        receipt['failure'] = repr(exc)
        (out/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:
            verify(pins)
            receipt['inputs_unchanged'] = bool(before)
        except BaseException as exc:
            receipt.update(inputs_unchanged=False, post_pin_failure=repr(exc))
        if not receipt.get('inputs_unchanged'):
            receipt.update(completed=False, returncode=1)
        receipt['seconds'] = time.monotonic()-started
        save(out/'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
