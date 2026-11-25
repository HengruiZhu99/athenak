"""Write matched 100:1 embedded-secondary, fixed-mesh and cyclic-zoom startup tests."""

import argparse
import math
from pathlib import Path

from inputs import make_input, parse_header, read_header, write_blocks


def generate(out, mass_ratio=100, shallow=False):
    out.mkdir(parents=True, exist_ok=True)
    header = read_header(Path(__file__).with_name("aligned_reference.par"))
    seed = out / "cyclic.athinput"
    b = parse_header(make_input(header, 25, "run"))
    b = {k: v for k, v in b.items() if not k.startswith(("refined_region", "output"))}
    sep = 25
    q = 1 / mass_ratio  # AthenaK convention is M2/M1.
    m1, m2 = 1 / (1 + q), q / (1 + q)
    x1, x2 = sep * m2, -sep * m1
    b["job"]["basename"] = "embedded"
    b["problem"].update(
        q=q,
        r_edge=12,
        r_peak=28,
        pert_amp=0,
        flux_radius1=5 * m1,
        flux_radius2=20 * m2,
        flux_rsurf_inner=40,
        flux_rsurf_outer=40,
        flux_ks_radius1=0,
        flux_ks_radius2=0,
        flux_ntheta=24,
        flux_nphi=48,
    )
    # Reset the puncture core and smoothly taper before the modeled horizon.
    b["coord"].update(
        excise_1_rad=1.35 * m1,
        excise_2_rad=1.35 * m2,
        smooth_excision_puncture_width_fraction=0.25,
        smooth_excision_puncture_weight_exponent=1,
        puncture_flux_excision_radius_factor=1.15,
    )
    for d in (1, 2, 3):
        b["mesh"].update({f"x{d}min": -128, f"x{d}max": 128, f"nx{d}": 32})
    secondary_level = math.ceil(math.log2(8 / (0.13 * m2)))
    primary_level = math.ceil(math.log2(8 / (0.13 * m1)))
    b["mesh_refinement"].update(num_levels=secondary_level + 1, max_nmb_per_rank=1632)
    b["cyclic_zoom"].update(max_nzmb_per_host=900, max_nzmb_per_dvce=800)

    def box(level, x, half, height):
        key = f"refined_region{1 + sum(k.startswith('refined_region') for k in b)}"
        b[key] = dict(
            level=level,
            x1min=x - half,
            x1max=x + half,
            x2min=-half,
            x2max=half,
            x3min=-height,
            x3max=height,
        )

    for level, half, height in [(1, 80, 32), (2, 62, 16), (3, 45, 8), (4, 34, 4)]:
        box(level, 0, half, height)
    for level in range(5, primary_level + 1):
        half = 5 if level == 5 else 2.4 * 2 ** (6 - level)
        box(level, x1, half, half)
    for level in range(5, secondary_level + 1):
        dx = 8 / 2**level
        box(level, x2, 23 * dx, 23 * dx)
    # The reference dwell and output times use M units for m2 = 1/101.
    time_scale = m2 * 101
    b["amr_criterion1"].update(
        nlevels=4,
        ncenters=1,
        x1c=x2,
        r_0=1.35 * m2,
        f_ref=32,
        r_ref_max=min(2 * time_scale, 6),
        trun_fac_0=0.024 * time_scale,
        trun_fac_1=0.012 * time_scale,
        trun_fac_2=0.018 * time_scale,
        trun_fac_3=0.192 * time_scale,
    )
    if shallow:
        criterion = b["amr_criterion1"]
        criterion.update(nlevels=2, trun_fac_0=0.080 * time_scale, trun_fac_1=0.160 * time_scale)
        del criterion["trun_fac_2"], criterion["trun_fac_3"]
    b["time"].update(tlim=0.94 * time_scale, ndiag=8)
    b["output1"] = dict(file_type="hst", dt=0.006 * time_scale, data_format="%24.16e")
    b["output2"] = dict(
        file_type="bin", variable="mhd_w_bcc", id="midplane", slice_x3=0, dt=0.094 * time_scale
    )
    seed.write_text(write_blocks(b))
    b.pop("amr_criterion1")
    b.pop("cyclic_zoom")
    b["mesh_refinement"]["refinement"] = "static"
    (out / "fixed.athinput").write_text(write_blocks(b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mass-ratio", type=int, choices=(1, 100), default=100)
    parser.add_argument(
        "--shallow", action="store_true", help="Use two zones and a longer finest dwell"
    )
    args = parser.parse_args()
    generate(args.directory, args.mass_ratio, args.shallow)
