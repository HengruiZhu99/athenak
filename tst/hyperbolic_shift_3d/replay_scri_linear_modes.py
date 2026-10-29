#!/usr/bin/env python3
"""Independent saved-mode residual replay, including a pure-NumPy BLAS audit.

Uses non-BLAS reductions for residuals/Rayleigh quotients. No evolution occurs.
"""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np

from scri_linear_audit import Probe, sha, write


def norm(x):return float(np.sqrt(np.sum(np.abs(x)**2)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();campaign=args.campaign.resolve()
    rng=np.random.default_rng(5)
    q=rng.normal(size=(4104,512));q/=np.sqrt(np.sum(q*q,axis=0));z=rng.normal(size=4104)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always');product=q.T@z;gram=q.T@q
    # These explicit contraction paths do not invoke the BLAS matmul routine.
    reference=np.einsum('ij,i->j',q,z,optimize=False)
    gram_reference=np.einsum('ij,ik->jk',q,q,optimize=False)
    output=dict(numpy=np.__version__,scope='Saved linear modes only, no evolution',
        pure_numpy_matmul=dict(native_library_loaded=False,warnings=[str(w.message) for w in notices],
            all_outputs_finite=bool(np.isfinite(product).all() and np.isfinite(gram).all()),
            vector_relative_error=norm(product-reference)/norm(reference),
            gram_max_absolute_error=float(np.max(np.abs(gram-gram_reference)))),cases=[])
    for scheme in ['v2','v3']:
        report_path=campaign/scheme/'report.json';report=json.loads(report_path.read_text())
        modes_path=campaign/scheme/'local-modes.npz';modes=np.load(modes_path)['modes']
        probe=Probe(campaign/'probe.dylib',report['n'],scheme=='v3',100,report['volume_dimension']>0)
        case=dict(scheme=scheme,report_sha256=sha(report_path),modes_sha256=sha(modes_path),
                  library_sha256=sha(campaign/'probe.dylib'),modes=[])
        try:
            for mode,old in zip(modes,report['modes']):
                value=complex(*old['eigenvalue']);mode=mode/norm(mode);checks=[]
                for eps in [1e-4,5e-5,2e-5,1e-5,5e-6,2e-6,1e-6]:
                    action=probe.jvp(mode,eps);rayleigh=np.sum(np.conj(mode)*action)
                    checks.append(dict(epsilon=eps,relative_original_ritz_residual=norm(action-value*mode)/(norm(action)+abs(value)),
                                       rayleigh=[float(rayleigh.real),float(rayleigh.imag)],
                                       all_finite=bool(np.isfinite(action).all())))
                case['modes'].append(dict(original_eigenvalue=old['eigenvalue'],checks=checks))
            case['native_rhs_evaluations']=probe.calls
        finally:probe.close()
        output['cases'].append(case)
    write(args.output,output)
    print(json.dumps(dict(output=str(args.output),blas_relative_error=output['pure_numpy_matmul']['vector_relative_error'])))


if __name__=='__main__':main()
