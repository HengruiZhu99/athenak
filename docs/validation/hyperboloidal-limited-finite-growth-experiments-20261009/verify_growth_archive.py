"""Verify saved compact evidence only; never assemble or propagate an operator."""
from pathlib import Path
import argparse, hashlib, json, math

def finite(x):
    if isinstance(x, float): assert math.isfinite(x)
    elif isinstance(x, dict):
        for v in x.values(): finite(v)
    elif isinstance(x, list):
        for v in x: finite(v)

def read(p):
    x=json.loads(p.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    finite(x); return x

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path); a=p.parse_args()
    c=read(a.root/'catalog.json'); checked=0
    for name, spec in c['files'].items():
        f=a.root/name
        assert f.stat().st_size==spec['bytes'] and sha(f)==spec['sha256'], name
        if f.suffix=='.json':
            if spec.get('empty_diagnostic_log'):
                assert name=='original-held-growth-source/synthetic-preflight-001/history/001-bundled-import-probe/stdout.json' and f.stat().st_size==0
            else: read(f)
        assert f.suffix not in ('.npz','.npy','.o','.a','.bin') and f.stat().st_size<=1048576
        checked+=1
    actual={str(f.relative_to(a.root)) for f in a.root.rglob('*') if f.is_file()}
    assert actual==set(c['files'])|{'catalog.json'}
    assert all(not (a.root/name).exists() for name in c['omitted_large_payloads'])
    growth=[]
    for n in (8,12,16):
        d=read(a.root/f'growth/J0-N{n}-rb98-growth001/receipt.json')
        assert d['passed_finite_ODE_numerical_checks'] is True and d['error'] is None
        assert d['input_pins_unchanged'] is True and d['inputs_before']==d['inputs_after']
        assert d['original_full_projection_defect_gate_passed'] is False
        assert d['general_nongauge_continuum_comparator_unresolved'] is True
        assert d['both_original_FD_attempts_remain_failed'] is True
        assert d['original_SciPy_expm_attempt_remains_failed'] is True
        assert [r['time'] for r in d['propagation']]==[0.,.25,.5,1.,2.,4.,6.]
        maximum=0.
        for row in d['propagation']:
            for i, value in enumerate(row['seed_energy_amplifications']):
                expected=row['seed_energy_norms'][i]/d['seeds'][i]['initial_energy_norm']
                delta=abs(value-expected)/max(1.,abs(value),abs(expected))
                maximum=max(maximum,delta); assert delta<=5e-14
        assert max(v[0] for v in d['spectrum'])==d['finite_matrix_spectral_abscissa']>0.
        growth.append({'N':n,'spectral_abscissa_estimate':d['finite_matrix_spectral_abscissa'],
          't6_lapse_energy_amplification':d['propagation'][-1]['seed_energy_amplifications'][0],
          't6_shift_energy_amplification':d['propagation'][-1]['seed_energy_amplifications'][4],
          'max_saved_ratio_reconstruction':maximum})
    h=read(a.root/'actual-high-precision/reference-001/results/receipt.json')
    assert h['status']=='PASS_finite_matrix_accuracy_only'
    assert h['inputs_before']==h['inputs_after']
    assert h['full_propagator_comparison']['scaled_frobenius']<2e-7
    assert h['saved_modal_seed_comparison']['scaled_frobenius']<2e-7
    s=read(a.root/'saved-point-readbacks/index.json')
    assert s['general_nongauge_continuum_comparator_unresolved'] is True
    print(json.dumps({'passed_saved_compact_checks':True,'checked_blobs':checked,
      'omitted_payloads':len(c['omitted_large_payloads']),'growth':growth,
      'point_case_arithmetic':'Original full-local verification receipt retained; JSONL cases omitted here. No point arithmetic rerun in compact verification.',
      'scope':'Rounded finite matrices only; no eigenvalue forward certificate, continuum convergence, native/nonlinear/scri/BH acceptance.'},indent=2,allow_nan=False))

if __name__=='__main__': main()
