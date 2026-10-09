"""Saved-receipt summary of exactly the eight short/reference preflights.

No probe, native executable, stencil, operator or evolution calls.
"""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFLIGHT = ROOT / 'build-layer-research/wave-map-native-preflight-root-20261009'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def finite(x):
    if isinstance(x, dict):
        for value in x.values(): finite(value)
    elif isinstance(x, list):
        for value in x: finite(value)
    elif isinstance(x, float):
        assert math.isfinite(x)

def load(p):
    value = json.loads(Path(p).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    finite(value)
    return value

def dump(p, value):
    finite(value)
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def main():
    output = HERE / 'snapshot-preflight-summary001'
    output.mkdir(exist_ok=False)
    (output / 'summarizer.py').write_bytes(Path(__file__).read_bytes())
    release = load(PREFLIGHT / 'release.json')
    assert sha(PREFLIGHT / 'release.json') == 'dc5787c5b579b0137cb96cea3953cdb706f22ebbd94f5e2fd12c03c1fc54ff9f'
    cases = []
    pins = {str(Path(__file__)): sha(Path(__file__)),
        str(PREFLIGHT / 'release.json'): sha(PREFLIGHT / 'release.json')}
    for spec in release['cases']:
        folder = HERE / 'snapshot-attempts' / (spec['name'] + '-001')
        outer = load(folder / 'receipt.json')
        analysis = load(folder / 'analysis/receipt.json')
        data = load(folder / 'analysis/snapshots.json')
        assert outer['passed_fixed_snapshot_readback'] is True
        assert analysis['passed_saved_snapshot_finite_and_diagnostic_gates'] is True
        assert analysis['protected_inputs_before_after_equal'] is True
        assert outer['protected_before_after_equal'] is True
        assert outer['analyzer_receipt_sha256'] == sha(folder / 'analysis/receipt.json')
        assert analysis['snapshots_sha256'] == sha(folder / 'analysis/snapshots.json')
        assert len(data) == analysis['saved_arrays']
        assert (folder / 'protected-inputs-before.json').read_bytes() == (folder / 'protected-inputs-after.json').read_bytes()
        for relative in ['receipt.json', 'authorization.json', 'stdout', 'stderr',
                         'analysis/receipt.json', 'analysis/snapshots.json']:
            p = folder / relative
            pins[str(p)] = sha(p)
        profile = 0 if analysis['reference'] else 1 if 'small' in spec['name'] else 2
        q = {'case': spec['name'], 'mode': analysis['mode'], 'N': analysis['N'],
            'saved_arrays': len(data), 'target_time': analysis['target_time'],
            'final_H_Mcon_Zcon_Theta_physical': data[-1]['rms_H_Mcon_Zcon_Theta'],
            'max_saved_H_Mcon_Zcon_Theta_physical': [max(x['rms_H_Mcon_Zcon_Theta'][i] for x in data) for i in range(4)],
            'min_saved_alpha': min(x['alpha_min'] for x in data),
            'min_saved_chi': min(x['chi_min'] for x in data),
            'min_saved_conformal_metric_eigenvalue': min(x['minimum_conformal_metric_eigenvalue'] for x in data),
            'max_saved_det_error': max(x['det_max'] for x in data),
            'max_saved_trace_error': max(x['trace_max'] for x in data),
            'max_saved_reference_full25_deviation': max(max(x['reference_deviation_max25']) for x in data),
            't0_actual_initializer_max_error': data[0]['initial_profile_max_error_reference_small_large'][profile],
            'max_history_native_kernel_scaled_RMS_error': max(x['history_scaled_rms_error'] for x in data),
            'saved_time': [x['time'] for x in data],
            'saved_cycle': [x['cycle'] for x in data],
            'saved_history_dt_fullprecision': [x['history_dt'] for x in data],
            'saved_restart_header_dt_fullprecision': [x['restart_header_dt'] for x in data],
            'outer_receipt': str(folder / 'receipt.json'),
            'outer_receipt_sha256': sha(folder / 'receipt.json'),
            'analyzer_receipt_sha256': sha(folder / 'analysis/receipt.json')}
        cases.append(q)
    assert len(cases) == 8
    for name, digest in pins.items(): assert sha(name) == digest
    dump(output / 'summary.json', {'passed_all_eight_short_reference_snapshot_gates': True,
        'cases': cases, 'analyzer_sha256': 'c1b9487717e85e920b274b7dcb429290eed6b0d96b3b6cbd8e00d6ea352f7c72',
        'native_probe_sha256': '584e74bc257e7661310fff684af6d5ccf12c18dd24886a7e0ae9941c87161a54',
        'scope': 'Actual native reference/short pulse binary64 fields and native diagnostic gates only. No long-time stability, relative pulse acceptance, finite-k/PDE theorem, exact scri closure, BH core choice or wormhole-to-trumpet evolution claim. No exact Cauchy-core grid cell at N16/24/32 span2.2; separate local core gate remains the evidence there. Restart/history dt semantics differ from the six-digit console and from live-state recomputed caps.'})
    dump(output / 'receipt.json', {'passed_saved_receipt_summary': True, 'source_sha256': sha(Path(__file__)),
        'summary_sha256': sha(output / 'summary.json'), 'saved_input_pins': pins,
        'new_probe_queries': 0, 'new_native_steps': 0})
    print(sha(output / 'summary.json'))

if __name__ == '__main__':
    main()
