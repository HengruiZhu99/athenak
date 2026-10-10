"""Unexecuted fixed74-unit Fraction oracle for the private integer backend.

This module compiles/runs nothing. A future separately reviewed driver must
build Release and sanitizer Debug probes and call check_output on saved text.
"""
from fractions import Fraction
import json
import struct

EXPECTED_CASES = 74
ONE = "3ff0000000000000"
TWO = "4000000000000000"
THREE = "4008000000000000"
MIN = "0000000000000001"
MAX = "7fefffffffffffff"


def registry():
    cases = []

    def case(name, atoms, ops=(), numerator=0, denominator=1, category="finite"):
        cases.append({"id": name, "atoms": list(atoms), "operations": list(ops),
                      "numerator": numerator, "denominator": denominator,
                      "category": category})

    atom_bits = (
        "0000000000000000", "8000000000000000", MIN, "8000000000000001",
        "000fffffffffffff", "800fffffffffffff", "0010000000000000", "8010000000000000",
        ONE, "bff0000000000000", TWO, "c000000000000000", "3fd5555555555555", "bfd5555555555555",
        MAX, "ffefffffffffffff", "3ff0000000000001", "bff0000000000001",
        "3fefffffffffffff", "bfefffffffffffff", "0123456789abcdef", "8123456789abcdef",
        "6fedcba987654321", "efedcba987654321")
    for i, bits in enumerate(atom_bits):
        case("atom_%02d" % i, [bits, ONE])

    ratios = ((ONE, THREE), (TWO, THREE), ("0010000000000000", THREE), (MIN, TWO),
              (MIN, "c000000000000000"), ("8000000000000001", TWO), (MAX, MAX),
              ("0010000000000000", MIN), (MAX, TWO), (MAX, "3fe0000000000000"),
              (MIN, MAX), (MAX, MIN))
    for i, atoms in enumerate(ratios):
        case("ratio_%02d" % i, atoms, category="exact_overflow" if i in (9, 11) else "finite")

    for i, (anchor, half) in enumerate((
            (ONE, "3ca0000000000000"), ("3ff0000000000001", "3ca0000000000000"),
            (TWO, "3cb0000000000000"), ("4000000000000001", "3cb0000000000000"),
            ("bff0000000000000", "bca0000000000000"), ("bff0000000000001", "bca0000000000000"),
            ("c000000000000000", "bcb0000000000000"), ("c000000000000001", "bcb0000000000000"))):
        case("normal_tie_%02d" % i, [anchor, half, ONE], [("+", 0, 1)], 3, 2)

    for i, bits in enumerate((MIN, "0000000000000003", "0000000000000005",
                               "0000000000000007", "8000000000000001", "8000000000000003")):
        case("subnormal_tie_%02d" % i, [bits, TWO])

    case("cancel_huge_square", [MAX, ONE], [("*", 0, 0), ("-", 2, 2)], 3, 1)
    case("retain_tiny_after_square", [MAX, MIN, ONE],
         [("*", 0, 0), ("+", 3, 1), ("-", 4, 3)], 5, 2)
    case("associativity", ["3ff0000000000001", "3fefffffffffffff", "6fedcba987654321", ONE],
         [("*", 0, 1), ("*", 4, 2), ("*", 1, 2), ("*", 0, 6), ("-", 5, 7)], 8, 3)
    case("retain_tiny_after_sum", [MAX, MIN, ONE], [("+", 0, 1), ("-", 3, 0)], 4, 2)
    case("signed_cancellation", [MAX, "ffefffffffffffff", ONE],
         [("-", 0, 1), ("-", 1, 0), ("+", 3, 4)], 5, 2)
    case("top_binade_difference", [MAX, "7feffffffffffffe", ONE], [("-", 0, 1)], 3, 2)

    for depth in (1, 2, 4, 6, 8, 10):
        ops, current = [], 0
        for i in range(depth):
            ops.append(("*", current, current))
            current = 2 + i
        case("odd_power_%02d" % depth, ["3ff0000000000001", ONE], ops, current, 1)

    for degree in (14, 16, 20, 32, 48, 60):
        # Force exact accumulation across as many as125880 binary places.
        # Final ratio is finite; full numerator/denominator bits are checked.
        ops, large, small = [], 0, 1
        for _ in range(degree - 1):
            ops.append(("*", large, 0)); large = 3 + len(ops) - 1
            ops.append(("*", small, 1)); small = 3 + len(ops) - 1
        ops.append(("+", large, small)); total = 3 + len(ops) - 1
        case("wide_gap_%02d" % degree, [MAX, MIN, ONE], ops, large, total)

    case("shift_capacity", [ONE, ONE], [("L", 0, 131072)], 2, 1, "resource_failure")
    ops, current = [], 0
    for i in range(8):
        ops.append(("*", current, current)); current = 2 + i
    case("exponent_capacity", [MAX, ONE], ops, current, 1, "resource_failure")
    case("nan_rejected", ["7ff8000000000001", ONE], category="domain_failure")
    case("infinity_rejected", ["7ff0000000000000", ONE], category="domain_failure")
    case("zero_denominator", [ONE, "0000000000000000"], category="domain_failure")
    case("zero_over_zero", ["0000000000000000", "8000000000000000"], category="domain_failure")
    if len(cases) != EXPECTED_CASES or len({c["id"] for c in cases}) != EXPECTED_CASES:
        raise RuntimeError("fixed74 registry/count/unique IDs drifted")
    return cases


def protocol(cases):
    lines = []
    for case in cases:
        tokens = [case["id"], str(len(case["atoms"])), str(len(case["operations"])),
                  str(case["numerator"]), str(case["denominator"])] + case["atoms"]
        for op, a, b in case["operations"]:
            tokens.extend((op, str(a), str(b)))
        lines.append(" ".join(tokens))
    return "\n".join(lines) + "\n"


def atom(bits):
    # Independent CPython exact binary64->Fraction conversion; no native codec.
    value = struct.unpack(">d", int(bits, 16).to_bytes(8, "big"))[0]
    return Fraction.from_float(value)


def canonical(value):
    if value == 0:
        return (0, 0, "0")
    sign, n, d = (-1 if value < 0 else 1), abs(value.numerator), value.denominator
    if d & (d - 1):
        raise RuntimeError("unit expression lost dyadic denominator")
    trailing = (n & -n).bit_length() - 1
    return sign, trailing - (d.bit_length() - 1), format(n >> trailing, "x")


def expected(case):
    category = case["category"]
    if category in ("domain_failure", "resource_failure"):
        # These are fixed negative controls. Numerical expressions are not
        # evaluated to simulate the candidate's capacity or domain checks.
        return {"id": case["id"], "category": category}
    values = [atom(bits) for bits in case["atoms"]]
    for op, a, b in case["operations"]:
        if op == "+": result = values[a] + values[b]
        elif op == "-": result = values[a] - values[b]
        elif op == "*": result = values[a] * values[b]
        elif op == "N": result = -values[a]
        elif op == "L": result = values[a] * (1 << b)
        else: raise RuntimeError("unknown oracle operation")
        values.append(result)
    numerator, denominator = values[case["numerator"]], values[case["denominator"]]
    quotient = numerator / denominator
    maxfinite = atom(MAX)
    overflow = abs(quotient) > maxfinite
    actual_category = "exact_overflow" if overflow else "finite"
    if actual_category != category:
        raise RuntimeError("fixed unit category is mathematically wrong: " + case["id"])
    # Fraction.__float__ performs one correctly rounded rational conversion.
    # Strong range admission is checked before that independent conversion.
    rounded = 0 if overflow else int.from_bytes(struct.pack(">d", float(quotient)), "big")
    return {"id": case["id"], "category": category, "bits": "%016x" % rounded,
            "numerator": canonical(numerator), "denominator": canonical(denominator)}


def check_output(text):
    cases = registry()
    lines = text.splitlines()
    if len(lines) != EXPECTED_CASES:
        raise RuntimeError("actual native output count differs")
    checks = []
    for case, line in zip(cases, lines):
        want = expected(case)
        fields = line.split("\t")
        if fields[:2] != [want["id"], want["category"]]:
            raise RuntimeError("native ID/category mismatch for " + case["id"])
        if want["category"] in ("domain_failure", "resource_failure"):
            if len(fields) != 3 or not fields[2]:
                raise RuntimeError("missing explicit negative-control failure")
        else:
            if len(fields) != 9 or fields[2] != want["bits"]:
                raise RuntimeError("native rounded bits mismatch for " + case["id"])
            num = (int(fields[3]), int(fields[4]), fields[5])
            den = (int(fields[6]), int(fields[7]), fields[8])
            if num != want["numerator"] or den != want["denominator"]:
                raise RuntimeError("full exact operand bits mismatch for " + case["id"])
        checks.append({"id": case["id"], "category": want["category"], "passed": True})
    return {"passed": True, "checks": len(checks), "expected_checks": EXPECTED_CASES,
            "exact_full_operand_comparisons": sum(c["category"] not in ("domain_failure", "resource_failure") for c in cases),
            "rows": checks}


if __name__ == "__main__":
    raise SystemExit("Source-only module: use a separately reviewed driver; no direct execution")
