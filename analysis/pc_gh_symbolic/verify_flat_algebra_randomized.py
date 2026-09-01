#!/usr/bin/env python3
"""Seeded randomized flat-algebra gate for PC-GH metric/state operations."""

from __future__ import annotations

import numpy as np


TRIALS = 10_000
SEED = 20260901


def spatial_det(g: np.ndarray) -> float:
    return float(
        g[0, 0]*(g[1, 1]*g[2, 2] - g[1, 2]*g[1, 2])
        - g[0, 1]*(g[0, 1]*g[2, 2] - g[0, 2]*g[1, 2])
        + g[0, 2]*(g[0, 1]*g[1, 2] - g[0, 2]*g[1, 1]))


def spatial_inv(g: np.ndarray) -> np.ndarray:
    inv_det = 1.0/spatial_det(g)
    result = np.empty((3, 3))
    result[0, 0] = (g[1, 1]*g[2, 2] - g[1, 2]*g[1, 2])*inv_det
    result[0, 1] = result[1, 0] = (g[0, 2]*g[1, 2] - g[0, 1]*g[2, 2])*inv_det
    result[0, 2] = result[2, 0] = (g[0, 1]*g[1, 2] - g[0, 2]*g[1, 1])*inv_det
    result[1, 1] = (g[0, 0]*g[2, 2] - g[0, 2]*g[0, 2])*inv_det
    result[1, 2] = result[2, 1] = (g[0, 1]*g[0, 2] - g[0, 0]*g[1, 2])*inv_det
    result[2, 2] = (g[0, 0]*g[1, 1] - g[0, 1]*g[0, 1])*inv_det
    return result


def symmetric(rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(3, 3))
    return 0.5*(raw + raw.T)


def main() -> None:
    rng = np.random.default_rng(SEED)
    maxima = {name: 0.0 for name in (
        "inverse", "det_projection", "tf_A", "tf_Q", "adm_metric",
        "adm_curvature", "adm_chi", "adm_K")}
    for _ in range(TRIALS):
        factor = rng.normal(size=(3, 3))
        raw_g = factor@factor.T + np.eye(3)*10.0**rng.uniform(-2.0, 1.0)
        raw_g *= 10.0**rng.uniform(-3.0, 3.0)
        if not np.all(np.linalg.eigvalsh(raw_g) > 0.0):
            raise AssertionError("SPD generator failed")
        inverse = spatial_inv(raw_g)
        maxima["inverse"] = max(maxima["inverse"],
                                float(np.max(np.abs(inverse - np.linalg.inv(raw_g)))))
        if not np.allclose(inverse@raw_g, np.eye(3), rtol=2.0e-11, atol=2.0e-11):
            raise AssertionError("cofactor inverse failed")

        raw_a = symmetric(rng)
        raw_q = np.stack([symmetric(rng) for _ in range(3)])
        scale = spatial_det(raw_g)**(-1.0/3.0)
        g = scale*raw_g
        a = raw_a - raw_g*np.einsum("ij,ij->", inverse, raw_a)/3.0
        q = np.empty_like(raw_q)
        for d in range(3):
            q[d] = scale*(raw_q[d]
                          - raw_g*np.einsum("ij,ij->", inverse, raw_q[d])/3.0)
        g_inv = spatial_inv(g)
        trace_a = abs(np.einsum("ij,ij->", g_inv, a))
        trace_a_scale = np.linalg.norm(g_inv)*np.linalg.norm(a)
        relative_trace_a = trace_a/max(trace_a_scale, np.finfo(float).tiny)
        relative_trace_q = [
            abs(np.einsum("ij,ij->", g_inv, q[d]))
            /max(np.linalg.norm(g_inv)*np.linalg.norm(q[d]), np.finfo(float).tiny)
            for d in range(3)]
        maxima["det_projection"] = max(maxima["det_projection"], abs(spatial_det(g) - 1.0))
        maxima["tf_A"] = max(maxima["tf_A"], relative_trace_a)
        maxima["tf_Q"] = max(maxima["tf_Q"], max(relative_trace_q))
        if not np.isclose(spatial_det(g), 1.0, rtol=2.0e-11, atol=2.0e-11):
            raise AssertionError("determinant projection failed")
        if relative_trace_a > 2.0e-12:
            raise AssertionError("Atilde trace-free projection failed")
        if any(value > 2.0e-12 for value in relative_trace_q):
            raise AssertionError("Q trace projection failed")

        chi = 10.0**rng.uniform(-4.0, 4.0)
        trace_k = rng.normal()
        gamma = g/chi
        curvature = (a + g*trace_k/3.0)/chi
        gamma_inv = spatial_inv(gamma)
        chi_back = spatial_det(gamma)**(-1.0/3.0)
        trace_k_back = np.einsum("ij,ij->", gamma_inv, curvature)
        g_back = chi_back*gamma
        a_back = chi_back*(curvature - gamma*trace_k_back/3.0)
        maxima["adm_metric"] = max(maxima["adm_metric"],
                                   float(np.max(np.abs(g_back - g))))
        maxima["adm_curvature"] = max(maxima["adm_curvature"],
                                      float(np.max(np.abs(a_back - a))))
        maxima["adm_chi"] = max(maxima["adm_chi"], abs(chi_back - chi))
        maxima["adm_K"] = max(maxima["adm_K"], abs(trace_k_back - trace_k))
        if not np.allclose(g_back, g, rtol=3.0e-10, atol=3.0e-10):
            raise AssertionError("PC-GH/ADM metric round trip failed")
        if not np.allclose(a_back, a, rtol=3.0e-10, atol=3.0e-10):
            raise AssertionError("PC-GH/ADM curvature round trip failed")
        if not np.isclose(chi_back, chi, rtol=3.0e-10, atol=3.0e-10):
            raise AssertionError("PC-GH/ADM chi round trip failed")
        if not np.isclose(trace_k_back, trace_k, rtol=3.0e-10, atol=3.0e-10):
            raise AssertionError("PC-GH/ADM K round trip failed")

    print(f"PASS: {TRIALS} seeded SPD inverse/projection/ADM round-trip trials")
    for name, value in maxima.items():
        print(f"  max_{name}={value:.6e}")


if __name__ == "__main__":
    main()
