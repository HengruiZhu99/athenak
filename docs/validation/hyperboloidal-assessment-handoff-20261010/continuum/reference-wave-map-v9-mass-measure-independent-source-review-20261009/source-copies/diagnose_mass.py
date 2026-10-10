"""HELD one-site saved-operand diagnostic; imports only after exact release."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback
import warnings

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s:
        (_ for _ in ()).throw(ValueError(s)))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def verify(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('changed protected input: ' + path)


def rational(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def bits(value):
    return struct.pack('>d', float(value)).hex()


def classify(recipe, out):
    # No scientific import can occur before main's complete admission checks.
    import numpy as np
    from scipy.linalg.blas import dgemm
    from fast_weighting import FastWeightingArithmetic
    from tiny_normalization import possible_tiny, rounded_binary64, MIN_NORMAL, MIN_SUBNORMAL
    from operand_graph import build_mass_operand
    if str(Path(np.__file__).resolve().parent) != recipe['numpy_root']:
        raise RuntimeError('unexpected NumPy origin')
    np.seterr(all='raise')
    warnings.filterwarnings('error', category=RuntimeWarning)
    if not (recipe['first_radius'] == 609 and recipe['last_radius'] == 640 and
            recipe['total_radii'] == 769 and recipe['N'] == 8 and recipe['rb'] == .98):
        raise RuntimeError('fixed radius/degree scope changed')
    old_receipt = load(recipe['failed_receipt'])
    progress = load(recipe['failed_progress'])
    if not (old_receipt.get('completed') is False and old_receipt.get('returncode') == 1
            and old_receipt.get('inputs_unchanged') is True and
            old_receipt.get('failure') == 'FloatingPointError: underflow encountered in multiply'
            and progress['radius_index'] == 608 and progress['total_radii'] == 769):
        raise RuntimeError('exact actual failed radial provenance absent')
    expression = 'E+=measure*(bilinear(y,hy,angles)+bilinear(u,u,angles))'
    trace = Path(recipe['failed_trace']).read_text()
    if 'line 431, in scientific_readback' not in trace or expression not in trace:
        raise RuntimeError('traceback is not the declared outer E product')
    selected = recipe['selected_npz_arrays']
    if selected != ['source_coefficient_radii', 'radial_weights', 'angular_weights', 'source_reference_rows']:
        raise RuntimeError('selected-array scope changed')
    with np.load(recipe['operator_npz'], allow_pickle=False) as retained:
        # Lazy NpzFile accesses only these four operands. No operator matrix key.
        arrays = {key: retained[key] for key in selected}
    radii = arrays['source_coefficient_radii']
    weights = arrays['radial_weights']
    angles = arrays['angular_weights']
    refs = arrays['source_reference_rows']
    if not (radii.shape == (769,) and weights.shape == (768,) and
            angles.shape == (288,) and refs.shape == (769,17)):
        raise RuntimeError('retained operand shapes differ')
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError('nonfinite selected operand')
    if not (np.all(weights > 0) and np.all(angles > 0) and np.all(radii > 0)
            and np.all(np.diff(radii) > 0) and radii[-1] == .98 and np.all(refs[:,9] > 0)):
        raise RuntimeError('positive radii/measure/reference domain differs')
    layout = load(recipe['basis_data'])['channel_layouts']['0']
    ell = [row['L'] for row in layout]
    if len(ell) != 8:
        raise RuntimeError('J0 eight-channel layout differs')
    shape = (769,288,8,3,50)
    if Path(recipe['input_map']).stat().st_size != math.prod(shape)*8:
        raise RuntimeError('input map byte schema differs')
    maps = np.memmap(recipe['input_map'], mode='r', dtype='<f8', shape=shape)
    weighting = FastWeightingArithmetic(np)
    counts = Counter()
    rows = []
    examples = []
    first_exception = last_exception = first_tiny = last_tiny = None
    maximum_error = Fraction(0)
    maximum_case = None
    for ir in range(609,641):
        if not np.isfinite(maps[ir]).all():
            raise RuntimeError('nonfinite declared input-map window')
        measure, mass_sum = build_mass_operand(np, dgemm, weighting, radii, weights, refs,
                                             maps, angles, ell, ir)
        if mass_sum.shape != (64,64) or not np.isfinite(mass_sum).all():
            raise RuntimeError('mass operand shape/finite differs')
        if not (math.isfinite(float(measure)) and measure > 0):
            raise RuntimeError('finite positive outer measure required')
        ordinary = None
        row_counts = Counter()
        try:
            ordinary = measure*mass_sum
        except FloatingPointError as error:
            if 'underflow' not in str(error):
                raise
            event = dict(radius_index=ir, operation='measure*mass_sum', exception=str(error))
            row_counts['whole_array_underflow_observed'] += 1
            if first_exception is None:
                first_exception = event
            last_exception = event
        for i in range(64):
            for j in range(64):
                a, b = float(measure), float(mass_sum[i,j])
                row_counts['multiply_components'] += 1
                if not possible_tiny(a,b,'multiply'):
                    row_counts['zero_or_provably_normal_products'] += 1
                    continue
                exact = Fraction.from_float(a)*Fraction.from_float(b)
                rounded = rounded_binary64(exact)
                error = abs(Fraction.from_float(rounded)-exact)
                row_counts['potential_tiny_products'] += 1
                row_counts['exact_below_min_normal'] += int(abs(exact) < MIN_NORMAL)
                row_counts['exact_below_min_subnormal'] += int(abs(exact) < MIN_SUBNORMAL)
                row_counts['inexact_tiny_before_rounding'] += int(abs(exact) < MIN_NORMAL and error != 0)
                row_counts['inexact_tiny_after_rounding'] += int(abs(Fraction.from_float(rounded)) < MIN_NORMAL and error != 0)
                row_counts['rounded_zero'] += int(rounded == 0)
                row_counts['rounded_nonzero_subnormal'] += int(0 < abs(Fraction.from_float(rounded)) < MIN_NORMAL)
                record = dict(radius_index=ir, row=i, column=j, a_hex=a.hex(), b_hex=b.hex(),
                    rounded_hex=rounded.hex(), exact_product=rational(exact), rounding_error=rational(error))
                try:
                    observed = np.multiply(np.float64(a),np.float64(b))
                except FloatingPointError as error_flag:
                    if 'underflow' not in str(error_flag):
                        raise
                    row_counts['strict_scalar_underflow_observed'] += 1
                    record['strict_scalar_observation'] = str(error_flag)
                else:
                    if bits(observed) != bits(rounded):
                        raise ArithmeticError('strict scalar product differs from exact rounded target')
                    row_counts['strict_scalar_returned_exact_target'] += 1
                if ordinary is not None and bits(ordinary[i,j]) != bits(rounded):
                    raise ArithmeticError('strict vector product differs from exact rounded target')
                if error > maximum_error:
                    maximum_error, maximum_case = error, record
                if error and abs(exact) < MIN_NORMAL:
                    if first_tiny is None:
                        first_tiny = record
                    last_tiny = record
                if len(examples) < recipe['example_limit']:
                    examples.append(record)
        counts.update(row_counts)
        rows.append(dict(radius_index=ir, radius_hex=float(radii[ir]).hex(),
            measure_hex=float(measure).hex(), counts=dict(row_counts)))
        write(out/'progress.json',dict(last_radius_completed=ir, counts=dict(counts)))
    if counts['multiply_components'] != recipe['expected_components'] or len(rows) != 32:
        raise RuntimeError('bounded component/radius registry incomplete')
    if first_exception is None:
        raise RuntimeError('fixed window did not reproduce the declared strict product exception')
    if maximum_error > MIN_SUBNORMAL/2:
        raise ArithmeticError('local possible-tiny nearest-even error exceeds half-minsubnormal')
    write(out/'operand-weighting-rounding.json',weighting.summary())
    result = dict(diagnostic_completed=True, original_v9_radial_readback_passed=False,
        scope=recipe['scope'], counts=dict(counts), expected_components=recipe['expected_components'],
        first_strict_array_exception=first_exception, last_strict_array_exception=last_exception,
        first_inexact_tiny=first_tiny, last_inexact_tiny=last_tiny,
        maximum_local_rounding_error_exact=rational(maximum_error), maximum_error_component=maximum_case,
        examples=examples, radii=rows, selected_npz_arrays_accessed=selected,
        input_map_radius_window=[609,640], E_accumulated=False, operator_matrices_loaded=False,
        source_tables_loaded=False, SVD_executed=False, query_or_generator_executed=False,
        limitations=['Only the outer E measure product is classified. Exact onset outside the fixed window remains unknown.',
            'The bilinear operands retain the actual saved map, original modal recurrence, H action, tested weighting and same BLAS path.',
            'A local rounding bound is not a matrix/solve error certificate, tolerance change or radial readback qualification.'])
    write(out/'result.json',result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--authorization-sha256',required=True)
    args = parser.parse_args()
    recipe = load(HERE/'recipe.json')
    out = Path(recipe['attempt'])
    out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    record = dict(completed=False,returncode=1,scientific_imports_started=False,
        original_v9_radial_readback_passed=False,scope=recipe['scope'])
    pins = {str(HERE/'recipe.json'):sha(HERE/'recipe.json'),str(Path(__file__).resolve()):sha(__file__)}
    try:
        if not (sys.flags.no_user_site and sys.dont_write_bytecode and not sys.flags.optimize):
            raise RuntimeError('require unoptimized -B -s runtime')
        if str(Path(sys.executable).resolve()) != recipe['resolved_python']:
            raise RuntimeError('interpreter differs')
        for key,value in recipe['environment'].items():
            if os.environ.get(key) != value:
                raise RuntimeError('fixed environment differs: ' + key)
        if os.environ.get('PYTHONHOME') or os.environ.get('PYTHONWARNINGS'):
            raise RuntimeError('unset PYTHONHOME/PYTHONWARNINGS required')
        if sha(args.authorization) != args.authorization_sha256:
            raise RuntimeError('authorization identity differs')
        auth = load(args.authorization)
        if not (auth.get('bounded_saved_mass_diagnostic_authorized') is True and
                auth.get('source_index_sha256') == sha(HERE/'source-index.json') and
                auth.get('recipe_sha256') == sha(HERE/'recipe.json') and
                auth.get('diagnostic_source_sha256') == sha(__file__) and
                auth.get('only_selected_operands_no_E_accumulation_no_SVD_no_queries') is True and
                isinstance(auth.get('independent_review_receipt'),dict)):
            raise RuntimeError('exact bounded root release absent')
        review_pin = auth['independent_review_receipt']
        if sha(review_pin['path']) != review_pin['sha256']:
            raise RuntimeError('independent review receipt changed')
        review = load(review_pin['path'])
        if not (review.get('passed') is True and
                review.get('reviewed_source_index_sha256') == auth['source_index_sha256']):
            raise RuntimeError('passed review of exact source absent')
        pins.update(load(HERE/'input-pins.json'))
        for row in load(HERE/'source-index.json')['files']:
            if row['path'] in pins and pins[row['path']] != row['sha256']:
                raise RuntimeError('conflicting source pin')
            pins[row['path']] = row['sha256']
        pins[str(HERE/'source-index.json')] = auth['source_index_sha256']
        pins[str(args.authorization.resolve())] = args.authorization_sha256
        pins[review_pin['path']] = review_pin['sha256']
        verify(pins)
        write(out/'pins-before.json',pins)
        record['scientific_imports_started'] = True
        result = classify(recipe,out)
        record.update(completed=True,returncode=0,diagnostic_completed=result['diagnostic_completed'])
    except BaseException as error:
        record['failure'] = type(error).__name__ + ': ' + str(error)
        (out/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:
            verify(pins)
            record['inputs_unchanged'] = True
        except BaseException as error:
            record.update(inputs_unchanged=False,post_pin_failure=str(error),returncode=1)
        record['seconds'] = time.monotonic()-started
        write(out/'receipt.json',record)
    print(json.dumps(record),flush=True)
    if not (record['completed'] and record['inputs_unchanged'] and record['returncode'] == 0):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
