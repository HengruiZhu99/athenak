"""Separate eta and lapse damping effects at the matched radial normal profile."""
from pathlib import Path
from screen import defaults, run

out = Path(__file__).resolve().parent / "gauge-factorial"
out.mkdir(exist_ok=True)
for eta, lapse in [(2., .01), (.02, .1)]:
    for angle in [0., 1/64, 1/32, 1/8]:
        target = out / f"eta{eta}-lapse{lapse}-q{angle}.json"
        if target.exists():
            continue
        params = defaults(n=64, h=64, width=1536, rate=.001, kappa=0, angle=angle)
        params.update(sponge_end_cells=4., eta=eta, lapse_damping=lapse)
        run(params, target)
