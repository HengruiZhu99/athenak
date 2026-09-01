#!/usr/bin/env python3
"""Verify the normalized PC-GH KO symbol is nonpositive for every stencil."""

from __future__ import annotations

import math


WEIGHTS = {
    2: [1, -4, 6, -4, 1],
    3: [1, -6, 15, -20, 15, -6, 1],
    4: [1, -8, 28, -56, 70, -56, 28, -8, 1],
}


def main() -> None:
    for stencil, weights in WEIGHTS.items():
        offsets = range(-stencil, stencil + 1)
        coefficient = 2.0**(-2*stencil)*(-1.0 if stencil % 2 == 0 else 1.0)
        for sample in range(1025):
            theta = math.pi*sample/1024.0
            raw_real = sum(weight*math.cos(offset*theta)
                           for offset, weight in zip(offsets, weights, strict=True))
            raw_imag = sum(weight*math.sin(offset*theta)
                           for offset, weight in zip(offsets, weights, strict=True))
            normalized = coefficient*raw_real
            expected = -math.sin(0.5*theta)**(2*stencil)
            if abs(raw_imag) > 2.0e-14 or abs(normalized - expected) > 3.0e-14:
                raise AssertionError(
                    f"stencil {stencil}: KO symbol mismatch at theta={theta}")
            if normalized > 3.0e-14:
                raise AssertionError(
                    f"stencil {stencil}: antidissipative symbol {normalized}")
        print(f"PASS: stencil={stencil} normalized symbol=-sin(theta/2)^{2*stencil}")
    print("PASS: every supported PC-GH KO symbol is nonpositive")


if __name__ == "__main__":
    main()
