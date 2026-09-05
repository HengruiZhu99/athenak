"""Summarize completed single-puncture screens without declaring qualification."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from analyze_native_puncture import analyze

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'analysis/pc_gh_localization'))
from plot_qualification import regional_rms, athena_read

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('group', type=Path)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
rows = []
for done in sorted(args.group.glob('pcgh-*/completed.json')):
    run = done.parent
    bounds = np.atleast_1d(np.genfromtxt(next(run.glob('*.pcgh-boundedness.dat')), names=True))
    if not all(np.isfinite(bounds[name]).all() for name in bounds.dtype.names):
        raise ValueError(f'Nonfinite full-volume boundedness data in {run}')
    hist = athena_read.hst(str(next(run.glob('*.pcgh.hst'))))
    if not all(np.isfinite(values).all() for values in hist.values()):
        raise ValueError(f'Nonfinite history data in {run}')
    files = sorted((run/'bin').glob('*.bin'))
    native = analyze(files[-1], run/'native-analysis')
    t = float(bounds['time'][-1])
    if abs(native['time']-t) > 1e-4:
        raise ValueError(f'Last native slice does not match completed evolution in {run}')
    row = dict(run=run.name, time=t, native=native,
               bounds_final={name: float(bounds[name][-1]) for name in bounds.dtype.names},
               bounds_min={name: float(bounds[name].min()) for name in bounds.dtype.names},
               bounds_max={name: float(bounds[name].max()) for name in bounds.dtype.names},
               history_time=float(hist['time'][-1]), regional_constraints={})
    for region in ['all', 'chi', 'r05', 'r1', 'r2', 'ah']:
        rms = regional_rms(hist, region)
        if not all(np.isfinite(v).all() for v in rms.values()):
            raise ValueError(f'Nonfinite regional RMS in {run}, region {region}')
        row['regional_constraints'][region] = {name: float(v[-1]) for name, v in rms.items()}
    rows.append(row)
    print(run.name, 'time', t, 'min SPD', row['bounds_min']['min_eigenvalue'],
          'max Z', row['bounds_max']['max_Z'], 'Qcurl', row['bounds_final']['pcgh_curl_Q_max'],
          'exterior r1', row['regional_constraints']['r1'])
report = dict(runs=rows, scope='Completed screening runs; no automatic puncture acceptance')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2)+'\n')
