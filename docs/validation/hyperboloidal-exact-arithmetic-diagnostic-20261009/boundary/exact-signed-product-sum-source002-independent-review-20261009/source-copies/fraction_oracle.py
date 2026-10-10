"""HELD independent exact-Fraction target/bit oracle; run only via pinned gate."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import sys


def require(test, message):
    if not test:
        raise ValueError(message)


def load(path):
    def reject(value):
        raise ValueError('nonfinite JSON token: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(word):
    return (int(word, 16) >> 52) & 2047 != 2047


def number(word):
    return Fraction.from_float(struct.unpack('>d', bytes.fromhex(word))[0])


def bits(value):
    return struct.pack('>d', value).hex()


def power2(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << (-exponent))


MAXFINITE = Fraction((1 << 53) - 1) * (1 << 971)


def validation(case):
    ts = case['terms']
    if len(ts) > 32:
        return 'term_cap'
    if ts and case['null_input']:
        return 'null_input'
    for t in ts:
        if t['sign'] not in (-1, 1):
            return 'invalid_sign'
        if t['shift'] not in (-1, 0):
            return 'invalid_shift'
        if not 0 <= t['arity'] <= 4:
            return 'factor_cap'
        for atom in t['atoms'][:t['arity']]:
            if not finite(atom[0]) or (case['mode'] == 'dual' and not finite(atom[1])):
                return 'invalid_atom'
    return 'ok'


def polynomial_targets(case):
    primal, tangent = Fraction(0), Fraction(0)
    primal_terms, tangent_terms = [], []
    for t in case['terms']:
        coefficient = t['sign'] * power2(t['shift'])
        # Independent exact polynomial recurrence in a formal variable e.
        # Coefficient of e is updated without factor division or seed tests.
        p, d = Fraction(1), Fraction(0)
        atoms = [(number(a[0]), number(a[1]) if case['mode'] == 'dual' else Fraction(0))
                 for a in t['atoms'][:t['arity']]]
        for v, dv in atoms:
            d, p = d * v + p * dv, p * v
        primal += coefficient * p
        tangent += coefficient * d
        primal_terms.append(coefficient * p)
        # A separate expanded list checks only the candidate's declared counters.
        for j in range(t['arity']):
            product = coefficient
            for k, (v, dv) in enumerate(atoms):
                product *= dv if k == j else v
            tangent_terms.append(product)
    require(sum(tangent_terms, Fraction(0)) == tangent, 'independent dual recurrences disagree')
    return [(primal, primal_terms), (tangent, tangent_terms)]


def check_result(result, exact, monomials, label):
    expected_status = 'exact_overflow' if abs(exact) > MAXFINITE else 'ok'
    require(result['status'] == expected_status, label + ': status')
    if expected_status != 'ok':
        require(result['bits'] == '0000000000000000', label + ': overflow value sentinel')
        require(result['audit']['monomials'] == len(monomials), label + ': overflow count')
        return dict(status=expected_status, exact_numerator=str(exact.numerator),
                    exact_denominator=str(exact.denominator))
    # CPython Fraction->float uses its own rational conversion, structurally
    # different from the candidate's fixed limbs/guard/sticky rounder.
    rounded = float(exact)
    expected_bits = bits(rounded)
    require(result['bits'] == expected_bits, label + ': rounded bits')
    audit = result['audit']
    require(audit['monomials'] == len(monomials), label + ': monomial count')
    require(audit['zero_monomials'] == sum(x == 0 for x in monomials), label + ': zero count')
    require(audit['nonzero_monomials'] == sum(x != 0 for x in monomials), label + ': nonzero count')
    require(audit['exact_zero'] is (exact == 0), label + ': exact-zero audit')
    rounded_exact = Fraction.from_float(rounded)
    inexact = rounded_exact != exact
    require(audit['inexact'] is inexact, label + ': inexact audit')
    qzero = exact != 0 and rounded_exact == 0
    require(audit['rounded_to_zero'] is qzero, label + ': rounded-zero audit')
    require(audit['negative_rounded_zero'] is (qzero and exact < 0), label + ': signed-zero audit')
    b = int(expected_bits, 16)
    subnormal = b & 0x7ff0000000000000 == 0 and b & 0xfffffffffffff != 0
    require(audit['subnormal_result'] is subnormal, label + ': subnormal audit')
    if exact != 0:
        x = abs(exact)
        if x >= power2(-1022):
            e = x.numerator.bit_length() - x.denominator.bit_length()
            if x < power2(e):
                e -= 1
            exponent = e - 53
        else:
            exponent = -1075
        require(audit['half_grid_exponent'] == exponent, label + ': rounding-grid exponent')
        require(abs(rounded_exact - exact) <= power2(exponent), label + ': exact half-grid bound')
    else:
        require(expected_bits == '0000000000000000', label + ': exact-zero sign')
    return dict(status='ok', bits=expected_bits, inexact=inexact,
                exact_numerator=str(exact.numerator), exact_denominator=str(exact.denominator))


def main():
    require(sys.flags.optimize == 0 and sys.flags.isolated == 1 and sys.dont_write_bytecode,
            'isolated unoptimized -I -B launch required')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--registry', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--report', required=True)
    args = ap.parse_args()
    registry = load(args.registry)
    raw = Path(args.output).read_text().splitlines()
    require(len(raw) == registry['case_count'] == 70, 'fixed70 cases required')
    reports = []
    for case, line in zip(registry['cases'], raw):
        row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        expected_echo = {key: case[key] for key in ('id', 'mode', 'null_input', 'terms')}
        require({key: row[key] for key in expected_echo} == expected_echo, case['id'] + ': input echo')
        valid = validation(case)
        hand_status = case.get('hand_status')
        if case['mode'] == 'scalar':
            if valid != 'ok':
                require(row['result']['status'] == valid, case['id'] + ': rejected input')
                require(hand_status == valid, case['id'] + ': fixed invalid expectation')
                reports.append(dict(id=case['id'], validation=valid))
                continue
            exact, monomials = polynomial_targets(case)[0]
            report = check_result(row['result'], exact, monomials, case['id'])
            if 'hand_bits' in case:
                require(report.get('bits') == case['hand_bits'], case['id'] + ': hand tie/control')
            if hand_status is not None:
                require(report['status'] == hand_status, case['id'] + ': hand status')
            reports.append(dict(id=case['id'], validation=valid, result=report))
        else:
            require(row['validation'] == valid, case['id'] + ': dual validation')
            if valid != 'ok':
                require(hand_status == valid and row['generated_tangent_terms'] == 0,
                        case['id'] + ': fixed invalid dual expectation')
                reports.append(dict(id=case['id'], validation=valid))
                continue
            targets = polynomial_targets(case)
            require(row['generated_tangent_terms'] == sum(t['arity'] for t in case['terms']),
                    case['id'] + ': complete unsimplified derivative count')
            rs = [check_result(row[key], exact, terms, case['id'] + ':' + key)
                  for key, (exact, terms) in zip(('primal', 'tangent'), targets)]
            if 'hand_bits' in case:
                require([r.get('bits') for r in rs] == case['hand_bits'], case['id'] + ': hand dual control')
            if hand_status is not None:
                require([r['status'] for r in rs] == hand_status, case['id'] + ': hand dual status')
            reports.append(dict(id=case['id'], validation=valid, primal=rs[0], tangent=rs[1],
                                generated_tangent_terms=row['generated_tangent_terms']))
    report = dict(passed=True, fixed_cases=70, scalar_cases=44, dual_cases=26,
                  registry_sha256=sha(args.registry), output_sha256=sha(args.output),
                  oracle='exact Fraction polynomial recurrence and CPython rational-to-binary64 conversion',
                  hand_tie_controls=True, no_gauge_or_PDE_acceptance=True, cases=reports)
    Path(args.report).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('passed', 'fixed_cases', 'scalar_cases', 'dual_cases')}))


if __name__ == '__main__':
    main()
