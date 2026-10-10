"""Stdlib saved-record audit only. No scientific imports or payload decoding."""
from pathlib import Path
import collections
import hashlib
import json
import math
import time
import traceback

HERE = Path(__file__).resolve().parent
LIMIT = 1 << 20
CASES = ('primary', 'radial_pair', 'angular_pair')
OUTER = ('E', 'Ks', 'Kw', 'G', 'loads')
AUDITS = (
    'tiny-normalization-rounding.json',
    'tiny-bilinear-weighting-rounding.json',
    'tiny-derivative-product-rounding.json',
    'tiny-weak-product-rounding.json',
    'tiny-retained-basis-division-rounding.json',
) + tuple('tiny-outer-measure-' + k + '-rounding.json' for k in OUTER)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def finite_json(x):
    if isinstance(x, float):
        assert math.isfinite(x), 'nonfinite saved JSON number'
    elif isinstance(x, dict):
        for v in x.values(): finite_json(v)
    elif isinstance(x, list):
        for v in x: finite_json(v)


def load(path):
    p = Path(path)
    assert p.suffix == '.json' and p.stat().st_size <= LIMIT, ('metadata-only file', str(p))
    d = json.loads(p.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    finite_json(d)
    return d


def write(path, d):
    finite_json(d)
    Path(path).write_text(json.dumps(d, indent=2, sort_keys=True, allow_nan=False) + '\n')


def record(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def check_pins(pins):
    for p, expected in pins.items():
        assert sha(p) == expected, ('input changed', p)


def check_index(path):
    p = Path(path)
    d = load(p)
    fs = d.get('files', [])
    assert isinstance(fs, list)
    for item in fs:
        q = Path(item['path'])
        if not q.is_absolute(): q = p.parent / q
        assert sha(q) == item['sha256'], ('index mismatch', str(q))
        if 'bytes' in item: assert q.stat().st_size == item['bytes']
    return len(fs)


def numeric_recount(result, case, recipe_case):
    assert case != 'radial_pair', 'oversized radial report must never be decoded'
    assert result['case'] == case and result['passed'] is True
    assert result['generator_eigenvalues_computed'] is False
    assert result['quadrature_and_energy_symmetric_eigensolves_only'] is True
    assert result['root_pinned_outer_receipt'] == recipe_case['outer_receipt_sha256']
    checks = result['checks']
    assert len(checks) == 3534 and all(v['passed'] is True for v in checks)
    schemas = collections.Counter()
    selected = {}
    family = []
    for item in checks:
        if 'tolerance' in item:
            assert item['tolerance'] > 0 and item['scaled'] <= item['tolerance']
            schemas['saved_scaled_vs_saved_tolerance'] += 1
        elif 'rank' in item:
            assert item['rank'] == item['expected']
            schemas['saved_rank_vs_expected'] += 1
        else:
            assert item['name'] == 'dissipative_boundary_work'
            schemas['recorded_boundary_work_acceptance_only'] += 1
        name = item['name']
        if name.startswith('direct_forced_family_'):
            family.append(item)
        if name in ('weak_strong', 'volume_trace_identity', 'nodal_congruence',
                    'retained_family_solved', 'dissipative_boundary_work') or name.startswith('incoming_rank_'):
            selected[name] = item
    assert schemas == {'saved_scaled_vs_saved_tolerance':3529,
                       'saved_rank_vs_expected':4,
                       'recorded_boundary_work_acceptance_only':1}
    assert len(family) == 33
    labels = [v['name'].removeprefix('direct_forced_family_') for v in family]
    assert len(set(labels)) == 33 and labels.count('mixed') == 1
    powers = collections.defaultdict(set)
    for label in labels:
        if label != 'mixed':
            name, power = label.rsplit('_rho_power_', 1)
            powers[name].add(int(power))
    assert len(powers) == 8 and all(v == {0,1,2,3} for v in powers.values())
    source = result['source_checks']
    assert set(source) == {'raw_action','normalized_action','normalized_values',
        'normalized_configuration_derivative','manual_normal','source_condition',
        'normal_input_scaled','normal_output_scaled'}
    assert all(v <= (1e4 if k == 'source_condition' else 5e-11) for k,v in source.items())
    assert result['source_passed'] is True
    assert result['energy_min'] > 0 and result['energy_condition'] <= 1e12
    return {'recounted_checks':len(checks), 'schemas':dict(schemas),
        'forcing_family_fields':33,'forcing_channels':sorted(powers),
        'worst_forcing_saved_scaled':max(family,key=lambda v:v['scaled']),
        'source_checks':source,'source_absolute_maxima':result['source_absolute_maxima'],
        'energy_min':result['energy_min'],'energy_condition':result['energy_condition'],
        'selected_saved_metrics':selected,
        'limitations':'Saved scalar comparisons only; no array reconstruction, target recomputation or independent boundary-work scale recomputation.'}


def audit_summary(d):
    # Exact rounding error fractions are retained as recorded strings, never evaluated.
    for k, v in d['counts'].items(): assert isinstance(v,int) and v >= 0
    out = {k:d[k] for k in ('scope','rounding','counts','labels',
        'maximum_absolute_local_rounding_error_exact','sum_absolute_local_rounding_errors_upper_bound',
        'error_bound_scope') if k in d}
    if 'fast_weighting_proof' in d: out['fast_weighting_proof'] = d['fast_weighting_proof']
    return out


def main():
    t0 = time.monotonic()
    recipe_before_sha256 = sha(HERE/'recipe.json')
    recipe = load(HERE/'recipe.json')
    pins = load(HERE/'input-pins.json')
    assert sha(HERE/'review_saved.py') == recipe['review_source_sha256']
    assert sha(HERE/'input-pins.json') == recipe['input_pins_sha256']
    output = HERE/'attempt001'
    output.mkdir(exist_ok=False)
    receipt = {'completed':False,'passed':False,'returncode':1,
        'saved_data_only':True,'scientific_execution':False,
        'radial_large_result_decoded':False,'generator_execution':False}
    try:
        check_pins(pins)
        owner = Path(recipe['owner']); root = Path(recipe['root'])
        release = load(root/'release.json'); authorization = load(root/'authorization.json')
        prep = load(root/'preparation001/receipt.json')
        assert len(release['pins']) == 4639
        assert prep['completed'] is True and prep['returncode'] == 0 and prep['inputs_unchanged'] is True
        assert prep['scientific_execution'] is False and prep['pins'] == 4639
        assert prep['authorization_sha256'] == sha(root/'authorization.json') == release['authorization_sha256']
        assert prep['release_sha256'] == sha(root/'release.json')
        assert release['process_group_cap_seconds'] == authorization['process_group_cap_seconds'] == 900
        assert release['cap_is_new_not_historical'] is True
        assert release['generator_spectrum_or_propagation_authorized'] is False
        assert authorization['generator_eigenvalues_authorized'] is False
        assert authorization['child_review_gate_required'] is True
        assert authorization['no_automatic_retry_or_cap_increase'] is True
        assert authorization['source_index_sha256'] == sha(owner/'source-index.json')
        assert authorization['recipe_sha256'] == sha(owner/'recipe.json')
        assert authorization['readback_source_sha256'] == sha(owner/'verify_retained.py')
        index_counts = {'owner':check_index(owner/'source-index.json'),
                        'root_launcher':check_index(root/'launcher-source-index.json')}
        independent = load(release['independent_review_receipt'])
        assert sha(release['independent_review_receipt']) == release['independent_review_receipt_sha256']
        assert sha(release['independent_review_index']) == release['independent_review_index_sha256']
        assert all(independent[k] is True for k in ('passed','inputs_unchanged','source_review_only'))
        assert independent['reviewed_source_index_sha256'] == sha(owner/'source-index.json')
        recipe_owner = load(owner/'recipe.json')
        result_summary = {}
        parsed_primary = None
        for case in CASES:
            rt = root/(case+'-invocation001')
            child = owner/'attempts'/('independent-'+case+'001')
            rr = load(rt/'receipt.json'); cr = load(child/'receipt.json')
            assert rr['case'] == case
            for k in ('completed','child_completed','child_inputs_unchanged','child_passed',
                      'source_runtime_review_pins_unchanged','cap_is_new_not_historical'):
                assert rr[k] is True, (case,k)
            assert rr['returncode'] == 0 and rr['cap_reached'] is False
            assert rr['process_group_cap_seconds'] == 900
            assert rr['generator_spectrum_or_propagation_admitted'] is False
            assert all(cr[k] is True for k in ('completed','passed','inputs_unchanged'))
            assert cr['returncode'] == 0
            assert sha(child/'receipt.json') == rr['child_receipt_sha256'] == recipe['expected_cases'][case]['child_receipt_sha256']
            assert sha(child/'result.json') == rr['result_sha256'] == recipe['expected_cases'][case]['result_sha256']
            assert (rt/'stderr.log').stat().st_size == 0
            for key in ('stdout','stderr'): assert sha(rt/(key+'.log')) == rr[key+'_sha256']
            before = load(rt/'pins-before.json'); after = load(rt/'pins-after.json')
            assert before == after and all(before.get(p) == h for p,h in release['pins'].items())
            child_before = load(child/'pins-before.json')
            assert all(before.get(p) == h for p,h in child_before.items())
            command = load(rt/'command.json')
            assert command['new_process_group'] is True and command['process_group_cap_seconds'] == 900
            assert command['cap_is_new_not_historical'] is True
            cmd = command['actual_after_stdlib_review_gate']
            assert cmd[1:3] == ['-B','-s'] and str(child) == cmd[-1]
            assert command['command'][1:3] == ['-I','-B']
            env = command['environment']
            assert all(env[k] == '1' for k in ('OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE'))
            assert env['PYTHONOPTIMIZE'] == '0'
            audits = {name:load(child/name) for name in AUDITS}
            for name in OUTER:
                assert sha(child/('tiny-outer-measure-'+name+'-rounding.json')) == rr['five_outer_measure_audit_files'][name]
            summed = {name:audit_summary(d) for name,d in audits.items()}
            outer_counts = {name:audits['tiny-outer-measure-'+name+'-rounding.json']['counts'] for name in OUTER}
            if case == 'radial_pair':
                assert (child/'result.json').stat().st_size == 1276457 > LIMIT
                counts = outer_counts['E']
                assert counts['fallback_multiply'] == counts['inexact'] == 28
                assert counts['rounded_zero'] == 20 and counts['rounded_nonzero_subnormal'] == 8
                assert all(not outer_counts[k] for k in ('Ks','Kw','G','loads'))
                numerical = {'independent_final_numerical_recount':False,
                    'acceptance_association':'Actual successful root and child receipts only; oversized result is stream-hashed metadata, never decoded.',
                    'result_bytes':1276457}
            else:
                assert (child/'result.json').stat().st_size <= LIMIT
                d = load(child/'result.json')
                numerical = numeric_recount(d,case,recipe_owner['cases'][case])
                for name in OUTER: assert d['tiny_outer_measure_arithmetic'][name] == audits['tiny-outer-measure-'+name+'-rounding.json']
                direct = {'tiny_normalization_arithmetic':AUDITS[0],
                    'tiny_bilinear_weighting_arithmetic':AUDITS[1],
                    'tiny_derivative_product_arithmetic':AUDITS[2],
                    'tiny_weak_product_arithmetic':AUDITS[3],
                    'tiny_retained_basis_division_arithmetic':AUDITS[4]}
                for key, filename in direct.items(): assert d[key] == audits[filename]
                assert all(not v for v in outer_counts.values())
                if case == 'primary': parsed_primary = d
                else: assert d['source_checks'] == parsed_primary['source_checks']
            result_summary[case] = {'root_receipt':record(rt/'receipt.json'),
                'child_receipt':record(child/'receipt.json'),'result_metadata_only':record(child/'result.json'),
                'root_seconds':rr['seconds'],'child_wall_seconds':rr['child_wall_seconds'],
                'child_recorded_seconds':cr['seconds'],'root_pins_before_after_count':len(before),
                'child_pins_before_count':len(child_before),'stderr_bytes':0,'cap_reached':False,
                'actual_acceptance_associated':True,'numerical_review':numerical,
                'ten_rounding_audits':summed}
        history = {}
        for name, path in recipe['historical_failures'].items():
            d = load(path)
            assert d['completed'] is False and d['returncode'] != 0
            history[name] = {'receipt':record(path),'recorded_completed':False,
                'recorded_returncode':d['returncode'],'recorded_error':d.get('error',d.get('failure')),
                'scope':'Original failure remains unchanged; no reinterpretation as a successful scientific run.'}
        check_pins(pins)
        assert sha(HERE/'recipe.json') == recipe_before_sha256
        assert sha(HERE/'input-pins.json') == recipe['input_pins_sha256']
        write(output/'summary.json',{'passed':True,'saved_data_only':True,
            'source_runtime_review_pins_rehashed_before_after':len(pins),
            'root_protected_pins':4639,'source_index_file_counts':index_counts,
            'cases':result_summary,'historical_failures':history,
            'all_three_actual_pass_receipts_associated':True,
            'primary_angular_saved_checks_recounted':7068,
            'radial_final_numerical_recount_performed':False,
            'no_arrays_maps_jsonl_or_large_json_decoded':True,
            'no_targets_or_scientific_queries_recomputed':True,
            'generator_spectrum_or_propagation_admitted':False,
            'scope':'Independent compact saved-scalar and provenance audit, not continuum/native stability or a new operator result.'})
        receipt.update(completed=True,passed=True,returncode=0,inputs_unchanged=True,
            protected_input_count=len(pins),all_three_actual_pass_receipts_associated=True,
            primary_angular_checks_recounted=7068,radial_final_numerical_recount=False)
    except BaseException as exc:
        (output/'failure.txt').write_text(traceback.format_exc())
        receipt['error'] = str(exc)
        try: check_pins(pins); receipt['inputs_unchanged'] = True
        except BaseException as pin_exc: receipt['inputs_unchanged'] = False; receipt['post_pin_error'] = str(pin_exc)
        raise
    finally:
        receipt['seconds'] = time.monotonic()-t0
        receipt['review_source_sha256'] = sha(HERE/'review_saved.py')
        receipt['recipe_sha256'] = sha(HERE/'recipe.json')
        receipt['recipe_unchanged'] = receipt['recipe_sha256'] == recipe_before_sha256
        write(output/'receipt.json',receipt)
        print(json.dumps(receipt,sort_keys=True))


if __name__ == '__main__': main()
