#!/usr/bin/env python3
"""Fail closed on forbidden production PC-GH source dependencies."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "pc_gh"
FORBIDDEN = {
    "second-derivative finite-difference helper": re.compile(r"\bD(?:xx|xy)\s*<"),
    "legacy GH or Z4c include": re.compile(
        r'^\s*#\s*include\s*[<\"](?:fo_gh|ref_gh|z4c)/', re.MULTILINE
    ),
}
PREFERRED_EVOLUTION = {
    "pc_gh_calcrhs.cpp",
    "pc_gh_constraints.cpp",
    "pc_gh_newdt.cpp",
    "pc_gh_projection.cpp",
}
PUNCTURE_FIELD_DIVISION = re.compile(
    r"/(?:\s*(?:w|rho|alpha|chi|A)(?![A-Za-z0-9_])"
    r"|\s*\([^\n)]*\b(?:w|rho|alpha|chi|A)\b[^\n)]*\))"
)
NEGATIVE_PUNCTURE_POWER = re.compile(
    r"(?:pow|POWER)\s*\(\s*(?:w|rho|alpha|chi|A)\s*,\s*-"
)


def main():
    files = sorted([*SOURCE.rglob("*.cpp"), *SOURCE.rglob("*.hpp")])
    if not files:
        raise AssertionError(f"no production PC-GH sources found below {SOURCE}")
    failures = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        for policy, pattern in FORBIDDEN.items():
            for match in pattern.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                failures.append(f"{path.relative_to(ROOT)}:{line}: {policy}")
        if path.name in PREFERRED_EVOLUTION:
            code = re.sub(r"/\*.*?\*/", lambda match: "\n"*match.group().count("\n"),
                          source, flags=re.DOTALL)
            code = re.sub(r"//[^\n]*", "", code)
            for match in PUNCTURE_FIELD_DIVISION.finditer(code):
                line = code.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: puncture-field division"
                )
            for match in NEGATIVE_PUNCTURE_POWER.finditer(code):
                line = code.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: negative puncture-field power"
                )
    if failures:
        raise AssertionError("\n".join(failures))
    print(f"PASS: source policy audit over {len(files)} PC-GH production files")


if __name__ == "__main__":
    main()
