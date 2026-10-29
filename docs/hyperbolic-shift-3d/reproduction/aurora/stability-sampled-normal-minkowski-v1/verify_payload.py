#!/usr/bin/env python3
"""Read-only V5 duration payload verification and endpoint-fit sensitivity.

Requires the complete locally archived campaign, including its checkpoint.
Never admits or evolves a checkpoint. The original comparison is unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AURORA = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output must be new')
    decoder_path = AURORA/'preflight-sampled-normal-v1/verification/verify.py'
    analysis_path = AURORA/'stability-dissipative-minkowski-v1/analysis/analyze.py'
    base = load_module('duration_base', analysis_path)
    decoder = load_module('v5_decoder', decoder_path)
    base.require(sha(analysis_path) == '685f65de88ea4065dde8f213f8151c518f4c4aa63411e7f55067e10ef18a666d',
                 'Changed frozen analysis')
    # Decoder identity is bound to the already verified preflight evidence.
    preflight = base.load(AURORA/'preflight-sampled-normal-v1/admission.json')
    base.require(sha(decoder_path) == preflight['verifier_sha256'], 'Changed V5 decoder')
    identity = base.load(HERE/'analysis/identity.json')
    base.require(sha(args.raw/'plan.json') == identity['plan_sha256'], 'Wrong plan')
    case, history = base.verify_case(('v5', 'V5 sampled normal', args.raw.name,
        identity['source_manifest_sha256'], identity['binary_sha256']), args.raw, verify_payloads=True)
    cp = decoder.checkpoint(args.raw/'output/minkowski.checkpoint')
    report = base.load(args.raw/'output/minkowski.report.json')
    base.require(cp['n'] == 16 and cp['kind'] == 0 and cp['time'] == report['t'] == .6
        and cp['dt'] == report['dt'] == .0005 and cp['steps'] == report['steps'] == 1201
        and cp['admitted'] == report['checkpoint_admitted']
        and cp['scri_dissipation_epsilon'] == 2, 'Checkpoint/report configuration differs')
    base.require(case['terminal']['terminal_verified'], 'Not terminal/released')
    base.require(report['status'] == 'completed' and case['execution_succeeded']
        and {g['field'] for g in case['v4_limits_on_recorded_history'] if g['status'] == 'failed'}
            == {'state_deviation', 'M_L2'}, 'Unexpected scientific outcome')
    # The accepted tiny final step yields two near-identical .6 diagnostics.
    # Retain the original estimator and report this separate sensitivity.
    keep = base.np.ones(len(history['t']), dtype=bool)
    keep[1:] = base.np.diff(history['t']) > report['dt'] * 1e-6
    filtered = {key: values[keep] for key, values in history.items()}
    result = dict(evidence_verified=True, scientifically_admitted=False,
        reason='Completed native evolution violates unchanged state and momentum limits',
        checkpoint=cp, artifact_verification=case['artifacts'], terminal=case['terminal'],
        native_completion_is_not_scientific_admission=True,
        original_fit=base.growth(history),
        endpoint_sensitivity=dict(rule='Drop later sample when adjacent time gap <= dt*1e-6',
            removed_indices=base.np.flatnonzero(~keep), fit=base.growth(filtered)),
        decoder_sha256=sha(decoder_path), analysis_sha256=sha(analysis_path),
        verifier_sha256=sha(Path(__file__)))
    base.write(args.out, result)


if __name__ == '__main__':
    main()
