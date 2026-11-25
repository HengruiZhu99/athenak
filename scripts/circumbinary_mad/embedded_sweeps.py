"""Generate the 100:1 timestep probes, compact-mask sweeps and disk-relaxation tests."""

import argparse
from pathlib import Path

from embedded_pilot import generate
from inputs import read_header, write_blocks


def make_sweeps(out):
    generate(out)
    primary_x = 25 / 101
    for name in (
        "single-probe",
        "dual-probe",
        "live3",
        "live4",
        "return-check",
        "relax-compact",
        "relax-secondary",
        "relax-both",
        "relax-long",
        "fine-control",
    ):
        static = name == "fine-control"
        deck = read_header(out / ("fixed.athinput" if static else "cyclic.athinput"))
        if not static:
            z = deck["amr_criterion1"]
            for key in list(z):
                if key.startswith("trun_fac_"):
                    del z[key]
            z["trun_fac"] = 0
        if name.endswith("probe"):
            z.update(
                nlevels=10, ncenters=2 if name == "dual-probe" else 1, x1c2=primary_x, r_ref_max=12
            )
            deck["time"].update(nlim=20, tlim=100, ndiag=1)
            for key in ("output1", "output2"):
                del deck[key]["dt"]
                deck[key]["dcycle"] = 1
            deck["output2"]["variable"] = "mhd_w_d"
            deck["cyclic_zoom"]["max_nzmb_per_host"] = 1600
        elif name in ("live3", "live4"):
            # Keep the AMR search boxes and puncture excision fixed while shrinking
            # the frozen primitive sphere. Allow the surrounding gas to evolve.
            levels = 3 if name == "live3" else 4
            z.update(nlevels=levels, r_0=0.3375 / 101, f_ref=128, trun_fac_0=0.014)
            z[f"trun_fac_{levels-1}"] = 0.084
        else:
            # Resolve the primary's interpolation stencils on the finest visits.
            deck["refined_region16"] = dict(
                level=7,
                x1min=primary_x - 2.2,
                x1max=primary_x + 2.2,
                x2min=-2.2,
                x2max=2.2,
                x3min=-2.2,
                x3max=2.2,
            )
            deck["mesh_refinement"]["max_nmb_per_rank"] = 2100
            if not static:
                levels = 7 if name == "relax-compact" else 10
                z.update(
                    nlevels=levels,
                    ncenters=1 if name in ("relax-secondary", "relax-compact") else 2,
                    x1c2=primary_x,
                    r_0=0.45 / 101,
                    f_ref=96,
                    r_ref_max=12,
                    trun_fac_0=0.060,
                )
                z[f"trun_fac_{levels-1}"] = 0.12 if name == "return-check" else 6.0
                deck["cyclic_zoom"]["max_nzmb_per_host"] = 1900
            if name.startswith("relax"):
                deck["time"].update(tlim=52, ndiag=4)
                deck["output1"]["dt"] = 0.018
                deck["output2"]["dt"] = 1.5
            if name == "relax-long":
                # Test longer outer residence in isolation, with eight fine returns.
                z["trun_fac_9"] = 24.0
                deck["time"]["tlim"] = 198
        (out / (name + ".athinput")).write_text(write_blocks(deck))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    make_sweeps(parser.parse_args().output)
