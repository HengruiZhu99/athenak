#!/usr/bin/env python3
"""Equation-preserving puncture-chart algebra and conditioning controls.

These are analytic/manufactured transformation tests, not Einstein evolutions
or evidence that any listed exponent is universal. No endpoint parity or beta
condition is imposed. Deliberately unsuccessful scaling controls are retained.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return 'nonfinite:' + str(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def scale_jet(r, n, velocity, power=None, length=.25):
    """Return log(f), its Cartesian gradient/Laplacian and fixed-x time rate."""
    if power is not None:
        logf = power * math.log(r)
        gradient = power * n / r
        laplacian = power / r**2
    else:
        e = math.exp(-(r / length)**2)
        f = 1 + 3 * e  # fixed 1 <= f <= 4; no vanishing or Omega floor
        gradf = -6 * e * r * n / length**2
        gradient = gradf / f
        lapf = e * (-18 / length**2 + 12 * r*r / length**4)
        laplacian = lapf / f - float(np.dot(gradient, gradient))
        logf = math.log(f)
    return logf, gradient, laplacian, -float(np.dot(gradient, velocity))


def chain_rule_controls():
    """Check all temporal/spatial terms against independently prescribed U jets.

    The scalar advection-diffusion equation is only an algebraic test vehicle.
    Its manufactured source makes the chosen U an exact solution. This does
    not identify diffusion with any particular Einstein equation.
    """
    center = np.array([.03, -.02, .01])
    beta0 = np.array([.04, -.03, .02])
    matrix = np.array([[.13, .02, -.01], [-.03, -.07, .01], [.01, -.02, .04]])
    directions = [np.array(v, float) / np.linalg.norm(v)
                  for v in ([1, 2, 3], [-2, 1, 1], [3, -1, 2])]
    rows = []
    worst = 0.
    for velocity in [np.zeros(3), -beta0]:
        for exponent in [4., 3., 2., 1.7, 1.09]:
            for power in [2., 4., 6., None]:
                for sign in [-1, 1]:  # w=f^sign U: divide or multiply unknown
                    error = reconstruction = 0.
                    for r in np.geomspace(1e-2, 1e-6, 9):
                        for n in directions:
                            x = center + r*n
                            beta = beta0 + r*matrix@n
                            H = .13*x[0]-.09*x[1]+.07*x[2]+.02*np.dot(x, x)
                            logu = exponent*math.log(r)+H
                            grad = exponent*n/r + np.array([.13, -.09, .07])+.04*x
                            lap = exponent/r**2 + .12
                            time = -.17*math.log(r)-exponent*np.dot(n, velocity)/r+.04
                            sigma, a, lap_sigma, sigma_t = scale_jet(r, n, velocity, power)
                            q = grad+sign*a
                            lapw_over_w = lap+sign*lap_sigma+np.dot(q, q)
                            diffusivity = .03
                            source_over_u = time-np.dot(beta, grad)-diffusivity*(lap+np.dot(grad, grad))
                            terms = [np.dot(beta, q), diffusivity*lapw_over_w,
                                     -2*diffusivity*sign*np.dot(a, q),
                                     sign*sigma_t-sign*np.dot(beta, a),
                                     diffusivity*(np.dot(a, a)-sign*lap_sigma), source_over_u]
                            expected = time+sign*sigma_t
                            residual = abs(sum(terms)-expected)/max(1., sum(abs(t) for t in terms)+abs(expected))
                            # Independent inverse product-rule jet reconstruction.
                            recovered_lap = lapw_over_w-2*sign*np.dot(a, q)+np.dot(a, a)-sign*lap_sigma
                            lap_error = abs(recovered_lap-(lap+np.dot(grad, grad))) / max(1., abs(lap)+np.dot(grad, grad))
                            recovered_time = (time+sign*sigma_t)-sign*sigma_t
                            time_error = abs(recovered_time-time)/max(1., abs(time))
                            # Actual exponential reconstruction, not just symbolic logs.
                            value = math.exp(logu+sign*sigma)*math.exp(-sign*sigma)
                            value_error = abs(value/math.exp(logu)-1)
                            error = max(error, residual)
                            reconstruction = max(reconstruction, lap_error, time_error, value_error)
                    worst = max(worst, error, reconstruction)
                    rows.append(dict(exponent=exponent, power=power, operation='multiply' if sign==1 else 'divide',
                                     center_velocity=velocity.tolist(), normalized_PDE_error=error,
                                     inverse_jet_relative_error=reconstruction))
    assert worst < 3e-12, worst
    return dict(cases=len(rows), exponents_are_manufactured_examples=True,
                diffusion_is_only_chain_rule_test_vehicle=True, maximum_error=worst, cases_detail=rows)


def conditioning_controls():
    rows = []
    radii = np.geomspace(1e-2, 1e-24, 45)
    families = [('chi', [4., 3.5, 3., 2.5, 2.], 4.),
                ('alpha', [2., 1.7, 1.3, 1.09], 2.)]
    for family, exponents, historical_power in families:
        for exponent in exponents:
            modes = [('direct', 0.), ('log', None), ('divide_fixed', historical_power),
                     ('divide_generic_factor', historical_power),
                     ('multiply_4', 4.), ('multiply_6', 6.), ('multiply_8', 8.),
                     ('multiply_16_stress', 16.), ('divide_bounded', None)]
            for mode, power in modes:
                u = radii**exponent  # every physical value is positive/representable
                with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
                    if mode == 'direct':
                        stored = u.copy(); reconstructed = stored.copy(); jac = np.ones_like(u)
                        stored_exponent = exponent; log_gain = -exponent*np.log10(radii)
                    elif mode == 'log':
                        stored = np.log(u); reconstructed = np.exp(stored); jac = u.copy()
                        stored_exponent = None; log_gain = np.zeros_like(u)
                    elif mode == 'divide_bounded':
                        f = 1+3*np.exp(-(radii/.25)**2)
                        stored = u/f; reconstructed = f*stored; jac = f
                        stored_exponent = exponent; log_gain = np.log10(f)-exponent*np.log10(radii)
                    elif mode == 'divide_generic_factor':
                        omega = (1-radii*radii)/2
                        if family == 'chi':
                            f = radii**4/(radii**4+omega**4)
                        else:
                            f = (1+radii*radii)/2*radii**2/(radii**2+omega**4)
                        stored = u/f; reconstructed = stored*f; jac = f
                        stored_exponent = exponent-historical_power
                        log_gain = np.log10(f)-exponent*np.log10(radii)
                    else:
                        f = radii**power
                        if mode.startswith('divide'):
                            stored = u/f; reconstructed = f*stored; jac = f
                            stored_exponent = exponent-power
                            log_gain = (power-exponent)*np.log10(radii)
                        else:
                            stored = f*u; reconstructed = stored/f; jac = 1/f
                            stored_exponent = exponent+power
                            log_gain = -(power+exponent)*np.log10(radii)
                    relative = abs(reconstructed/u-1)
                finite = np.isfinite(relative)
                guarantee = None if stored_exponent is None else (math.ceil(stored_exponent)-1 if stored_exponent>0 else -1)
                rows.append(dict(field=family, physical_power=exponent, mode=mode, fixed_power=power,
                    stored_power=stored_exponent, generic_angular_Ck_guarantee=guarantee,
                    exponent_guarantee_excludes_special_radial_polynomial_cancellations=True,
                    stored_min=float(np.nanmin(stored)), stored_max=float(np.nanmax(stored)),
                    stored_zeros=int(np.sum(stored==0)), stored_nonfinite=int(np.sum(~np.isfinite(stored))),
                    reconstruction_nonfinite=int(np.sum(~np.isfinite(reconstructed))),
                    maximum_finite_relative_roundtrip_error=float(np.max(relative[finite])) if np.any(finite) else None,
                    reconstruction_absolute_Jacobian_at_min_r=float(jac[-1]),
                    log10_relative_physical_gain_per_unit_absolute_stored_error_at_min_r=float(log_gain[-1]),
                    linear_rescaling_relative_condition_number=1. if mode!='log' else None,
                    log_absolute_error_to_relative_physical_error_gain=1. if mode=='log' else None))
    stress_failures = [r for r in rows if r['stored_zeros'] or r['stored_nonfinite'] or r['reconstruction_nonfinite']]
    assert stress_failures, 'Stress control did not retain expected scaling failures'
    return dict(radii=radii.tolist(), physical_values_all_representable=True,
                common_absolute_stored_error_is_a_norm_conditioning_stress_not_a_roundoff_model=True,
                failures_retained=stress_failures, rows=rows)


def discrete_product_rule_controls():
    rows = []
    for exponent in [4., 2., 1.09]:
        for power in [0., 2., 4., 6., 8.]:
            samples = []
            for n in [32, 64, 128, 256]:
                h = 1./n
                for location, r in [('first_positive_point', h/2), ('fixed_bulk', .25+h/2)]:
                    x = r+h*np.arange(-2, 3)
                    # Exact manufactured radial values on a line through the
                    # punctured Cartesian domain; this is NOT a parity BC.
                    W = abs(x)**(exponent+power)
                    D1 = (W[0]-8*W[1]+8*W[3]-W[4])/(12*h)
                    D2 = (-W[0]+16*W[1]-30*W[2]+16*W[3]-W[4])/(12*h*h)
                    transformed = r**(-power)*(D2+(2-2*power)*D1/r+power*(power-1)*W[2]/r**2)
                    target = exponent*(exponent+1)*r**(exponent-2)
                    # Exact discrete conjugation reconstructs U at every donor
                    # before applying the physical stencil; it is algebraically
                    # identical to the unscaled physical discretization.
                    U = W/abs(x)**power
                    u1 = (U[0]-8*U[1]+8*U[3]-U[4])/(12*h)
                    u2 = (-U[0]+16*U[1]-30*U[2]+16*U[3]-U[4])/(12*h*h)
                    conjugated = u2+2*u1/r
                    samples.append(dict(n=n, location=location, r=r, physical_target=target,
                        continuum_product_rule_FD=transformed,
                        physical_residual_absolute=abs(transformed-target),
                        weighted_equation_residual_absolute=r**power*abs(transformed-target),
                        relative_error=abs(transformed-target)/abs(target),
                        exact_discrete_conjugation_relative_error=abs(conjugated-target)/abs(target)))
            rows.append(dict(physical_power=exponent, multiply_unknown_power=power, samples=samples))
    bad = next(r for r in rows if r['physical_power']==2 and r['multiply_unknown_power']==6)
    first = [s for s in bad['samples'] if s['location']=='first_positive_point']
    assert min(s['relative_error'] for s in first)>1e-2
    assert max(s['exact_discrete_conjugation_relative_error'] for s in first)<1e-10
    return dict(method='Fourth-order centered radial derivative identities on an exactly prescribed manufactured scalar',
        no_parity_or_endpoint_condition_imposed=True,
        naive_fixed_overregularization_rejected=True,
        reason='Smooth weighted data do not make a separately differenced continuum product rule equal to discrete conjugation.',
        rows=rows)


def vector_scaling_controls():
    """Independent Cartesian Lame product rule and metric-symbol checks."""
    first = np.array([[.13, .02, -.01], [-.03, -.07, .01], [.01, -.02, .04]])
    second = np.array([[[.03, .01, -.02], [.01, -.04, .01], [-.02, .01, .02]],
                       [[-.02, .02, .01], [.02, .01, -.01], [.01, -.01, .04]],
                       [[.01, -.01, .02], [-.01, .03, .01], [.02, .01, -.02]]])
    direction = np.array([1., 2., -3.])/math.sqrt(14)
    worst = 0.; cases = 0
    for r in np.geomspace(1e-2, 1e-6, 7):
        y = r*direction
        value = np.array([.2, -.1, .3])+first@y+.5*np.einsum('iab,a,b->i', second, y, y)
        gradient = first.T+np.einsum('iab,b->ai', second, y)
        hessian = np.transpose(second, (1, 2, 0))
        def lame(h):
            return np.array([sum(h[j, j, i] for j in range(3))
                             +sum(h[i, j, j] for j in range(3))/3 for i in range(3)])
        for power in [-4., -2., 0., 2., 4., 6.]:
            f = r**power; a = power*direction/r
            hf_over_f = (power*np.eye(3)+power*(power-2)*np.outer(direction, direction))/r**2
            beta_hessian = f*(hf_over_f[:, :, None]*value[None, None, :]
                +a[:, None, None]*gradient[None, :, :]
                +a[None, :, None]*gradient[:, None, :]+hessian)
            direct = lame(beta_hessian)/f
            term1 = lame(hessian)
            term2 = 2*np.einsum('a,ai->i', a, gradient)
            term3 = power*(power+1)/r**2*value
            term4 = (hf_over_f@value+gradient@a+a*np.trace(gradient))/3
            expected = term1+term2+term3+term4
            error = float(np.max(abs(direct-expected))/max(1., np.max(abs(term1)+abs(term2)+abs(term3)+abs(term4))))
            worst = max(worst, error); cases += 1
    assert worst < 3e-13, worst
    metrics = [np.eye(3), np.diag([2., .5, 1.]),
               np.array([[1.3, .2, .1], [.2, .9, -.05], [.1, -.05, 1.1]])]
    symbols = []
    for metric in metrics:
        metric = metric/np.linalg.det(metric)**(1./3)
        inverse = np.linalg.inv(metric);a = direction@inverse@direction
        symbol = a*np.eye(3)+np.outer(inverse@direction, direction)/3
        eigen = np.sort(np.linalg.eigvals(symbol).real)
        expected = np.sort(np.array([a, a, 4*a/3]))
        assert max(abs(eigen-expected)) < 1e-13
        assert np.max(abs(metric@symbol-(metric@symbol).T)) < 1e-13
        symbols.append(dict(metric=metric.tolist(), determinant=float(np.linalg.det(metric)),
                            metric_eigenvalues=np.linalg.eigvalsh(metric).tolist(),
                            symbol_eigenvalues=eigen.tolist(), chi_or_lapse_enters_principal_symbol=False))
    return dict(product_rule_cases=cases, maximum_normalized_error=worst,
                metric_symbol_checks=symbols,
                source_power_examples=[dict(lapse_power=q, curvature_power=b,
                    leading_divergence_source_power=q+b-1,
                    exponents_are_illustrations_not_universal=True)
                    for q in [2., 1.09] for b in [2., 0.]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = dict(scope='Analytic equation-preserving transformation/conditioning tests only; no evolution or universal exponent claim',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                chain_rule=chain_rule_controls(), conditioning=conditioning_controls(),
                discrete_product_rule=discrete_product_rule_controls(),
                shift_scaling=vector_scaling_controls(),
                recommendation='Keep physical beta unscaled; prefer exponent-independent positive/log fields with fixed bounded positive normalization. Treat vanishing-power overregularization as an unvalidated auxiliary representation unless exact reconstruction/discrete conjugation and physical residual controls are demonstrated.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finite_json(data), indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(chain_rule_maximum_error=data['chain_rule']['maximum_error'],
        transformation_cases=data['chain_rule']['cases'],
        conditioning_rows=len(data['conditioning']['rows']),
        retained_stress_failures=len(data['conditioning']['failures_retained']),
        vector_product_rule_error=data['shift_scaling']['maximum_normalized_error'],
        naive_fixed_overregularization_rejected=True)))


if __name__ == '__main__':
    main()
