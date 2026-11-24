"""Generate matched cyclic-zoom runs from the aligned MAD Athena parameter dump."""

import argparse
import math
from pathlib import Path


def read_header(path):
    blocks = {}
    current = None
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("<"):
            current = line.strip("<>")
            blocks[current] = {}
        elif "=" in line and current:
            name, value = line.split("=", 1)
            blocks[current][name.strip()] = value.strip()
    return blocks


def write_blocks(b):
    return "\n".join(
        f"<{k}>\n" + "".join(f"{n} = {v}\n" for n, v in values.items()) for k, values in b.items()
    )


def refine_disk(b, level, inner, outer, height):
    # Select parent blocks by cylindrical radius. Merge each contiguous row so
    # the input remains small and the Cartesian corners stay at the coarser level.
    parent_width = int(b["meshblock"]["nx1"]) * 128 / 2 ** (level - 1)
    limit = math.ceil(outer / parent_width)
    index = sum(name.startswith("refined_region") for name in b)
    for j in range(-limit, limit):
        y = (j + 0.5) * parent_width
        spans = []
        for i in range(-limit, limit):
            if inner <= math.hypot((i + 0.5) * parent_width, y) < outer:
                if spans and spans[-1][1] == i:
                    spans[-1][1] = i + 1
                else:
                    spans.append([i, i + 1])
        for xlo, xhi in spans:
            index += 1
            b[f"refined_region{index}"] = {
                "level": level,
                "x1min": xlo * parent_width,
                "x1max": xhi * parent_width,
                "x2min": j * parent_width,
                "x2max": (j + 1) * parent_width,
                "x3min": -height,
                "x3max": height,
            }


def make_input(header, sep, mode, large_blocks=False, wide_sweep=False, separation_layout=False):
    period = 2 * math.pi * sep**1.5
    b = {name: dict(header[name]) for name in ("mesh", "meshblock", "coord", "mhd", "problem")}
    p = b["problem"]
    expected = {
        "r_edge": 60,
        "r_peak": 240,
        "potential_r_pow": 2,
        "potential_falloff": 1500,
        "q": 1,
        "a1": 0.7,
        "a2": 0.7,
        "potential_beta_min": 100,
    }
    if not all(float(p[k]) == v for k, v in expected.items()) or p["vertical_field"] != "true":
        raise ValueError("Expected the aligned +0.7 vertical-field reference dump")
    tailored = separation_layout and sep in (15, 35)
    if tailored:
        if wide_sweep:
            raise ValueError("The separation-specific disk annuli use the three-zone sweep")
        large_blocks = True
        p.update(r_edge=2 * sep, r_peak={15: 126, 35: 275}[sep])
    b["job"] = {"basename": f"mad{sep}"}
    p.update(
        sep=sep,
        initial_condition="torus",
        corotating="true",
        spin_ramp="false",
        flux_modeled_horizons="true",
        flux_ks_radius1=5,
        flux_ks_radius2=5,
        flux_diagnostics="true",
        flux_monitor_ct="true",
        flux_min_cells_per_radius=3,
        flux_rsurf_inner=2.4 * sep,
        flux_rsurf_outer=2.4 * sep,
        flux_dr_surf=1,
        flux_shell_budget="false",
    )
    b["coord"].update(excise_to_horizon="false", excise_1_rad=0.45, excise_2_rad=0.45)
    b["adm"] = {"dynamic": "false"}
    b["mhd"]["calculate_tmunu"] = "false"
    for d in (1, 2, 3):
        b["mesh"][f"nx{d}"] = 64
        b["meshblock"][f"nx{d}"] = 16
    b["time"] = {
        "evolution": "dynamic",
        "integrator": "rk2",
        "cfl_number": 0.4,
        "nlim": -1,
        "tlim": 200 * period,
        "ndiag": 200,
    }
    b["output1"] = {"file_type": "hst", "dt": 2, "data_format": "%24.16e"}
    b["output2"] = {"file_type": "rst", "dt": 20 * period}
    if mode in ("inertial-ic", "rotating-ic"):
        for d in (1, 2, 3):
            b["mesh"][f"x{d}min"] = -384
            b["mesh"][f"x{d}max"] = 384
            b["mesh"][f"nx{d}"] = 32
        p.update(
            corotating="true" if mode == "rotating-ic" else "false",
            pert_amp=0,
            flux_modeled_horizons="false",
            flux_ks_radius1=0,
            flux_ks_radius2=0,
        )
        b["adm"]["dynamic"] = "true" if mode == "inertial-ic" else "false"
        b["time"]["nlim"] = 0
        b["output2"] = {"file_type": "bin", "variable": "mhd_w_bcc", "dt": 1}
    else:
        b["mesh_refinement"] = {
            "refinement": "adaptive",
            "num_levels": 11,
            "max_nmb_per_rank": 1960,
            "ncycle_check": 1,
            "refinement_interval": 1,
        }
        index = 0

        def box(level, x, half, zhalf):
            nonlocal index
            index += 1
            b[f"refined_region{index}"] = {
                "level": level,
                "x1min": x - half,
                "x1max": x + half,
                "x2min": -half,
                "x2max": half,
                "x3min": -zhalf,
                "x3max": zhalf,
            }

        for level, half, zhalf in [
            (1, 2048, 1024),
            (2, 1024, 512),
            (3, 640, 320),
            (4, 384, 192),
            (5, 256, 128),
            (6, 96, 48),
        ]:
            box(level, 0, half, zhalf)
        for level, half in [(7, 20), (8, 9), (9, 4), (10, 2)]:
            for x in (-sep / 2, sep / 2):
                box(level, x, half, half)
        if mode != "control":
            b["meshblock"]["nx3"] = 8
            if large_blocks:
                b["meshblock"].update(nx1=32, nx2=32)
                b["mesh_refinement"]["max_nmb_per_rank"] = 736
            if tailored:
                # Refine the inner magnetized disk vertically and retain about
                # ten cells per initial MRI wavelength near its pressure maximum.
                rings = {
                    15: [(4, 320, 512, 64), (5, 160, 384, 48), (6, 0, 240, 32),
                         (6, 208, 288, 16), (7, 32, 176, 16), (8, 40, 104, 8)],
                    35: [(4, 480, 896, 64), (5, 256, 704, 48), (6, 0, 400, 48),
                         (6, 352, 512, 16), (7, 64, 320, 16), (8, 96, 144, 8)],
                }[sep]
                for ring in rings:
                    refine_disk(b, *ring)
            else:
                refine_disk(b, 4, 480, 768, 64)
                refine_disk(b, 5, 256, 672 if large_blocks else 608, 48)
                refine_disk(b, 6, 304, 480 if large_blocks else 432, 16)
                refine_disk(b, 6, 0, 352, 48)
                refine_disk(b, 7, 48, 288 if large_blocks else 272, 16)
                refine_disk(b, 8, 80, 152, 8)
        if mode == "run":
            for i, normal in enumerate(("x3", "x2"), 3):
                b[f"output{i}"] = {
                    "file_type": "bin",
                    "variable": "mhd_w_bcc",
                    "id": "midplane" if normal == "x3" else "meridional",
                    f"slice_{normal}": 0,
                    "dt": 2 * period,
                }
        # The largest frozen ball ends inside the original disk edge for all three binaries.
        b["amr_criterion1"] = z = {
            "method": "cyclic_zoom",
            "nlevels": 6 if mode == "control" else 3,
            "ncenters": 2,
            "x1c": sep / 2,
            "x1c2": -sep / 2,
            "r_0": 1,
            "f_ref": 8,
            "f_exc": 0,
            "trun_pow": 0,
            "add_emf": "true",
        }
        # Bridging zones use several local dynamical times. The outer residence
        # samples disk inflow over a substantial fraction of t_in(3a) ~ 20 Pbin.
        dwells = [0.6 * period, 12, 4 * period]
        if wide_sweep and mode != "control":
            z.update(nlevels=4, r_0=0.375, f_ref=64)
            dwells = [0.6 * period, 12, 32, 4 * period]
        if mode == "control":
            dwells = [0.6 * period, 12, 32, 96, 256, 4 * period]
        for i, dwell in enumerate(dwells):
            z[f"trun_fac_{i}"] = 0.06 if mode == "pilot" else dwell
        b["cyclic_zoom"] = {
            "verbose": "false",
            "max_nzmb_per_dvce": 1500,
            "max_nzmb_per_host": 1850,
            "restart_reset_finest_dwell": "false",
        }
        if mode != "control":
            # Coarsen complete families near each hole without stripping the
            # thin disk rings during the four-orbit outer residence.
            z["r_ref_max"] = 192 if wide_sweep else 32
            if large_blocks:
                limit = 416 if tailored and sep == 15 else 736
                device_limit = {15: 384, 35: 704}[sep] if tailored else 640
                b["mesh_refinement"]["max_nmb_per_rank"] = limit
                b["cyclic_zoom"].update(max_nzmb_per_dvce=device_limit, max_nzmb_per_host=limit)
        if mode == "scale":
            b["time"].update(nlim=56, ndiag=8)
            b.pop("output2")
        if mode == "pilot":
            b["output3"] = {
                "file_type": "bin",
                "variable": "mhd_w_d",
                "slice_x3": 0,
                "id": "mesh_audit",
                "dcycle": 1,
            }
            b["time"].update(tlim=20, nlim=100, ndiag=1)
            b["output1"] = {"file_type": "hst", "dcycle": 1, "data_format": "%24.16e"}
            b.pop("output2")
            if tailored:
                for i, name, normal in [(4, "midplane", "x3"), (5, "meridional", "x2")]:
                    b[f"output{i}"] = {
                        "file_type": "bin", "variable": "mhd_w_bcc", "id": name,
                        f"slice_{normal}": 0, "dt": 2 * period,
                    }
    return write_blocks(b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("header", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--large-blocks", action="store_true", help="Use 32 x 32 x 8 disk blocks")
    parser.add_argument(
        "--wide-sweep",
        action="store_true",
        help="Coarsen the inner disk annulus for the outer dwell",
    )
    parser.add_argument(
        "--separation-layout", action="store_true",
        help="Generate the revised 15M and 35M disks with their own large-block annuli",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    header = read_header(args.header)
    for sep in (25, 15, 35):
        for mode in (
            ("run", "pilot", "scale", "control", "inertial-ic", "rotating-ic")
            if sep == 25
            else ("run", "pilot")
        ):
            (args.out / f"mad{sep}-{mode}.athinput").write_text(
                make_input(header, sep, mode, args.large_blocks, args.wide_sweep, args.separation_layout)
            )
