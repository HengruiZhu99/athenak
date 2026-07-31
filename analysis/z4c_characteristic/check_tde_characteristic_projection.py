#!/usr/bin/env python3
"""Check the independent TDE characteristic projector against closed forms."""

import importlib.util
import math
import pathlib
import types

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "compare_tde_boundaries", HERE / "compare_tde_boundaries.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main():
    nx = 33
    ny = 33
    dx = 0.1
    dy = 0.1
    x1min = 8.35
    x2min = -1.65
    x = x1min + (np.arange(nx) + 0.5) * dx
    y = x2min + (np.arange(ny) + 0.5) * dy
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    center_x = 10.0

    fields = {
        name: np.zeros((ny, nx), dtype=np.float64)
        for name in MODULE.Z4C_VARIABLES
    }

    def linear(constant, slope_x=0.0, slope_y=0.0):
        return (
            constant + slope_x * (grid_x - center_x)
            + slope_y * grid_y
        )

    fields["z4c_Khat"] = linear(0.11)
    fields["z4c_Theta"] = linear(-0.07)
    fields["z4c_Axx"] = linear(0.03)
    fields["z4c_Axy"] = linear(-0.04)
    fields["z4c_Axz"] = linear(0.05)
    fields["z4c_Ayy"] = linear(-0.025)
    fields["z4c_Ayz"] = linear(0.025)
    fields["z4c_Azz"] = linear(-0.005)
    fields["z4c_Gamx"] = linear(0.02)
    fields["z4c_Gamy"] = linear(-0.06)
    fields["z4c_Gamz"] = linear(0.08)
    fields["z4c_alpha"] = linear(0.10, 0.013)
    fields["z4c_chi"] = linear(0.05, -0.017)
    fields["z4c_betax"] = linear(0.12, 0.021)
    fields["z4c_betay"] = linear(-0.03, -0.023)
    fields["z4c_betaz"] = linear(0.04, 0.019)
    fields["z4c_gxx"] = linear(0.0, 0.014)
    fields["z4c_gyy"] = linear(0.0, -0.006)
    fields["z4c_gzz"] = linear(0.0, -0.008)
    fields["z4c_gxy"] = linear(0.0, 0.012)
    fields["z4c_gxz"] = linear(0.0, -0.011)
    fields["z4c_gyz"] = linear(0.0, 0.009)

    block = {
        "level": 0,
        "logical_location": (0, 0, 0, 0),
        "x1min": x1min,
        "x1max": x1min + nx * dx,
        "x2min": x2min,
        "x2max": x2min + ny * dy,
        "nx": nx,
        "ny": ny,
        "fields": fields,
    }
    arguments = types.SimpleNamespace(
        background_mass=0.0,
        background_spin=0.0,
        background_center_x=0.0,
        background_center_y=0.0,
        background_center_z=0.0,
        boundary_axis=1,
        boundary_side=1,
        # The synthetic metric has an xz slope away from the checked center;
        # only the central point has the exactly x-directed normal used below.
        maximum_unresolved_normal=0.1,
        residual_lapse_f=1.0,
        lapse_oplog=2.0,
        lapse_harmonicf=1.0,
        lapse_harmonic=0.0,
        shift_driver=1.0,
    )
    measured, measured_outgoing, measured_x, measured_y = (
        MODULE.characteristic_fields(
            block, arguments, include_outgoing=True))
    center = (ny // 2, nx // 2)
    if not (
        math.isclose(measured_x[center], center_x, abs_tol=1.0e-14)
        and math.isclose(measured_y[center], 0.0, abs_tol=1.0e-14)
    ):
        raise SystemExit("synthetic characteristic cell is not centered")

    alpha = 1.10
    chi = 1.05
    alpha_bg = 1.0
    beta_full = 0.12
    beta_bg = 0.0
    lapse_driver = 2.0
    shift_driver = 1.0
    beta_difference = beta_full - beta_bg
    lapse_root = 0.5 * (
        beta_full + beta_bg
        + math.sqrt(beta_difference ** 2 + 4.0 * chi * lapse_driver)
    )
    lapse_root_out = 0.5 * (
        beta_full + beta_bg
        - math.sqrt(beta_difference ** 2 + 4.0 * chi * lapse_driver)
    )
    shift_long_root = 0.5 * (
        beta_full + beta_bg
        + math.sqrt(beta_difference ** 2 + 16.0 * shift_driver / 3.0)
    )
    shift_long_root_out = 0.5 * (
        beta_full + beta_bg
        - math.sqrt(beta_difference ** 2 + 16.0 * shift_driver / 3.0)
    )
    shift_transverse_root = 0.5 * (
        beta_full + beta_bg
        + math.sqrt(beta_difference ** 2 + 4.0 * shift_driver)
    )
    shift_transverse_root_out = 0.5 * (
        beta_full + beta_bg
        - math.sqrt(beta_difference ** 2 + 4.0 * shift_driver)
    )
    shift_mu = shift_long_root - beta_full
    shift_delta_bg = shift_long_root - beta_bg
    lapse_shift_separation = (
        chi * lapse_driver - shift_mu * shift_delta_bg
    )
    light_shift_separation = chi * alpha ** 2 - shift_mu ** 2
    shift_mu_out = shift_long_root_out - beta_full
    shift_delta_bg_out = shift_long_root_out - beta_bg
    light_shift_separation_out = chi * alpha ** 2 - shift_mu_out ** 2
    shift_q = 4.0 * shift_driver / 3.0

    p = (0.11, -0.07, 0.03, 0.02)
    d = (-0.017, 0.014, 0.013, 0.021)
    sqrt_chi = math.sqrt(chi)
    expected = {
        "lapse": -(lapse_root - beta_bg) * p[0] / chi + d[2],
        "shift_longitudinal": (
            alpha * shift_delta_bg ** 2 * light_shift_separation * p[0]
            + 0.5 * alpha * shift_q * lapse_shift_separation * p[1]
            + 0.25 * shift_delta_bg
            * (4.0 * chi * alpha ** 2 - 3.0 * shift_mu ** 2)
            * lapse_shift_separation * p[3]
            + 0.5 * alpha ** 2 * shift_delta_bg
            * lapse_shift_separation * d[0]
            - chi * alpha * shift_delta_bg
            * light_shift_separation * d[2]
            + lapse_shift_separation * light_shift_separation * d[3]
        ),
        "constraint_scalar_theta": (
            sqrt_chi * p[1] + 0.5 * chi * p[3] + d[0]
        ),
        "constraint_scalar_z": (
            4.0 * p[0] / (3.0 * sqrt_chi)
            + 2.0 * p[1] / (3.0 * sqrt_chi)
            - 2.0 * p[2] / sqrt_chi - p[3] + d[1]
        ),
        "shift_transverse_1": (
            (shift_transverse_root - beta_bg) * (-0.06) - 0.023
        ),
        "shift_transverse_2": (
            (shift_transverse_root - beta_bg) * 0.08 + 0.019
        ),
        "constraint_transverse_1": (
            -2.0 * (-0.04) / sqrt_chi - (-0.06) + 0.012
        ),
        "constraint_transverse_2": (
            -2.0 * 0.05 / sqrt_chi - 0.08 - 0.011
        ),
        "tt_plus": -2.0 * (-0.01) / sqrt_chi + 0.001,
        "tt_cross": -2.0 * 0.025 / sqrt_chi + 0.009,
    }
    expected_outgoing = {
        "lapse": (
            -(lapse_root_out - beta_bg) * p[0] / chi + d[2]
        ),
        "shift_longitudinal": (
            alpha * shift_delta_bg_out ** 2
            * light_shift_separation_out * p[0]
            + 0.5 * alpha * shift_q * lapse_shift_separation * p[1]
            + 0.25 * shift_delta_bg_out
            * (4.0 * chi * alpha ** 2 - 3.0 * shift_mu_out ** 2)
            * lapse_shift_separation * p[3]
            + 0.5 * alpha ** 2 * shift_delta_bg_out
            * lapse_shift_separation * d[0]
            - chi * alpha * shift_delta_bg_out
            * light_shift_separation_out * d[2]
            + lapse_shift_separation * light_shift_separation_out * d[3]
        ),
        "constraint_scalar_theta": (
            -sqrt_chi * p[1] + 0.5 * chi * p[3] + d[0]
        ),
        "constraint_scalar_z": (
            -4.0 * p[0] / (3.0 * sqrt_chi)
            - 2.0 * p[1] / (3.0 * sqrt_chi)
            + 2.0 * p[2] / sqrt_chi - p[3] + d[1]
        ),
        "shift_transverse_1": (
            (shift_transverse_root_out - beta_bg) * (-0.06) - 0.023
        ),
        "shift_transverse_2": (
            (shift_transverse_root_out - beta_bg) * 0.08 + 0.019
        ),
        "constraint_transverse_1": (
            2.0 * (-0.04) / sqrt_chi - (-0.06) + 0.012
        ),
        "constraint_transverse_2": (
            2.0 * 0.05 / sqrt_chi - 0.08 - 0.011
        ),
        "tt_plus": 2.0 * (-0.01) / sqrt_chi + 0.001,
        "tt_cross": 2.0 * 0.025 / sqrt_chi + 0.009,
    }
    if set(measured) != set(expected):
        raise SystemExit(
            "synthetic characteristic modes differ: measured={} expected={}".
            format(sorted(measured), sorted(expected)))
    if set(measured_outgoing) != set(expected_outgoing):
        raise SystemExit(
            "synthetic outgoing modes differ: measured={} expected={}".
            format(
                sorted(measured_outgoing), sorted(expected_outgoing)))
    maximum_error = max(
        max(
            abs(measured[name][center] - value)
            for name, value in expected.items()
        ),
        max(
            abs(measured_outgoing[name][center] - value)
            for name, value in expected_outgoing.items()
        ),
    )

    # Rotate the same local state so the normal is z and the resolved slice is
    # x-z.  This exercises the thin-z TDE analysis path without relying on a
    # boundary-kernel diagnostic.
    grid_xz_x, grid_xz_z = np.meshgrid(
        np.linspace(-1.6, 1.6, nx),
        np.linspace(center_x - 1.6, center_x + 1.6, ny),
        indexing="xy",
    )

    def linear_z(constant, slope_z=0.0):
        return constant + slope_z * (grid_xz_z - center_x)

    fields_xz = {
        name: np.zeros((ny, nx), dtype=np.float64)
        for name in MODULE.Z4C_VARIABLES
    }
    fields_xz["z4c_Khat"] = linear_z(0.11)
    fields_xz["z4c_Theta"] = linear_z(-0.07)
    fields_xz["z4c_Azz"] = linear_z(0.03)
    fields_xz["z4c_Axz"] = linear_z(-0.04)
    fields_xz["z4c_Ayz"] = linear_z(0.05)
    fields_xz["z4c_Axx"] = linear_z(-0.025)
    fields_xz["z4c_Axy"] = linear_z(0.025)
    fields_xz["z4c_Ayy"] = linear_z(-0.005)
    fields_xz["z4c_Gamz"] = linear_z(0.02)
    fields_xz["z4c_Gamx"] = linear_z(-0.06)
    fields_xz["z4c_Gamy"] = linear_z(0.08)
    fields_xz["z4c_alpha"] = linear_z(0.10, 0.013)
    fields_xz["z4c_chi"] = linear_z(0.05, -0.017)
    fields_xz["z4c_betaz"] = linear_z(0.12, 0.021)
    fields_xz["z4c_betax"] = linear_z(-0.03, -0.023)
    fields_xz["z4c_betay"] = linear_z(0.04, 0.019)
    fields_xz["z4c_gzz"] = linear_z(0.0, 0.014)
    fields_xz["z4c_gxx"] = linear_z(0.0, -0.006)
    fields_xz["z4c_gyy"] = linear_z(0.0, -0.008)
    fields_xz["z4c_gxz"] = linear_z(0.0, 0.012)
    fields_xz["z4c_gyz"] = linear_z(0.0, -0.011)
    fields_xz["z4c_gxy"] = linear_z(0.0, 0.009)
    block_xz = {
        "level": 0,
        "logical_location": (0, 0, 0, 0),
        "plane_axes": (0, 2),
        "fixed_axis": 1,
        "column_min": -1.65,
        "column_max": 1.65,
        "row_min": 8.35,
        "row_max": 11.65,
        "ncolumn": nx,
        "nrow": ny,
        "fields": fields_xz,
    }
    arguments_xz = types.SimpleNamespace(**vars(arguments))
    arguments_xz.boundary_axis = 3
    arguments_xz.slice_fixed_coordinate = 0.0
    measured_xz, measured_xz_outgoing, measured_x, measured_z = (
        MODULE.characteristic_fields(
            block_xz, arguments_xz, include_outgoing=True))
    if not (
        math.isclose(measured_x[center], 0.0, abs_tol=1.0e-14)
        and math.isclose(measured_z[center], center_x, abs_tol=1.0e-14)
    ):
        raise SystemExit("synthetic x-z characteristic cell is not centered")
    if set(measured_xz) != set(expected):
        raise SystemExit(
            "synthetic x-z characteristic modes differ: "
            "measured={} expected={}".format(
                sorted(measured_xz), sorted(expected)))
    if set(measured_xz_outgoing) != set(expected_outgoing):
        raise SystemExit(
            "synthetic x-z outgoing modes differ: "
            "measured={} expected={}".format(
                sorted(measured_xz_outgoing),
                sorted(expected_outgoing)))
    maximum_error = max(
        maximum_error,
        max(
            abs(measured_xz[name][center] - value)
            for name, value in expected.items()
        ),
        max(
            abs(measured_xz_outgoing[name][center] - value)
            for name, value in expected_outgoing.items()
        ),
    )
    background_arguments = types.SimpleNamespace(**vars(arguments))
    background_arguments.background_mass = 1.0
    ks_metric, ks_alpha, ks_chi, ks_beta = (
        MODULE.kerr_schild_background(
            np.asarray(12.8), np.asarray(0.0), np.asarray(0.0),
            background_arguments)
    )
    h = 1.0 / 12.8
    determinant = 1.0 + 2.0 * h
    expected_chi = determinant ** (-1.0 / 3.0)
    expected_metric = expected_chi * np.diag((determinant, 1.0, 1.0))
    background_error = max(
        float(np.max(np.abs(ks_metric - expected_metric))),
        abs(float(ks_alpha) - determinant ** -0.5),
        abs(float(ks_chi) - expected_chi),
        float(np.max(np.abs(
            ks_beta - np.asarray((2.0 * h / determinant, 0.0, 0.0))))),
    )
    maximum_error = max(maximum_error, background_error)
    if not math.isfinite(maximum_error) or maximum_error > 1.0e-12:
        raise SystemExit(
            "TDE characteristic projection error {:.8e}".format(
                maximum_error))
    print(
        "PASS tde_characteristic_projection_error={:.8e} "
        "background_error={:.8e} incoming_modes={} outgoing_modes={}".format(
            maximum_error, background_error, len(expected),
            len(expected_outgoing))
    )


if __name__ == "__main__":
    main()
