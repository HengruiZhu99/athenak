"""Bounded original-boundary real-root search at 16 cells/tangential wavelength."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize

from volume import Config, QP, asdict, boundary, schur
from boundary import assess_mode, profile
from validate import mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = Config(alpha=.999665896620774, chi=.9993319048666159,
                 beta_n=.0003157233701617295)
    k = np.pi/(8*32)
    xs = np.geomspace(1e-7, .05, 250)
    ys = [assess_mode(x, k, cfg)['sigma_min'] for x in xs]
    found = []

    def canonical(x):
        U, D, _, _ = schur(x, k, cfg)
        # Incoming q trace is nonsingular in this bounded search. This
        # canonical representation removes arbitrary Schur-basis phases.
        return boundary(U, D, cfg) @ np.linalg.inv(U[QP])

    for i in range(1, len(xs)-1):
        if not (ys[i] < ys[i-1] and ys[i] < ys[i+1]):
            continue
        fit = optimize.minimize_scalar(
            lambda x: assess_mode(x, k, cfg)['sigma_min'],
            bracket=xs[i-1:i+2], method='brent', options={'xtol': 1e-13})
        if fit.fun > 1e-5:
            continue
        E = canonical(fit.x)
        norms = np.linalg.norm(E, axis=1)
        left, _, vh = np.linalg.svd(E/norms[:, None])
        lvec, rvec = left[:, -1], vh[-1].conj()
        # Refine with a signed singular-vector residual; verify the actual
        # full boundary matrix and PDE afterward, not just this projection.
        scalar = lambda x: float(np.real(np.vdot(lvec, canonical(x)@rvec/norms)))
        root = optimize.brentq(scalar, xs[i-1], xs[i+1], xtol=1e-16, rtol=1e-14)
        result = mode(root, k, cfg, 'zero_rate')
        assert result['boundary_scaled'] < 1e-10 and result['bulk_relative'] < 1e-10
        result['fit_bracket'] = [float(xs[i-1]), float(xs[i+1])]
        result['profile'] = [{key: val for key, val in row.items() if key != 'state'}
                             for row in profile(root, k, cfg)['profile']]
        found.append(result)
        print(f'lambda={root:.16g}, boundary={result["boundary_scaled"]:.3g}, '
              f'bulk={result["bulk_relative"]:.3g}')
    args.output.write_text(json.dumps({
        'scope': __doc__, 'config': asdict(cfg), 'k': k, 'dx_M': 32,
        'cells_per_tangential_wavelength': 16, 'real_search': [1e-7, .05],
        'real_samples': len(xs), 'roots': found}, indent=2)+'\n')


if __name__ == '__main__':
    main()
