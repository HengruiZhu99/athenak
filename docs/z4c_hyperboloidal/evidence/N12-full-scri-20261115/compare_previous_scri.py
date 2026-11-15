"""Matched-window scri maxima from the previous full-overlap N12/N16 runs."""
from pathlib import Path
import hashlib
import json
import numpy as np

here = Path(__file__).resolve().parent
root = here.parents[1]/'hybrid-overlap/final-comparison'
manifest = json.loads((root/'input-manifest.json').read_text())
arrays = {}; files = {}
for n in (12, 16):
    rows = []; files[n] = {}
    for path in sorted((root/'histories'/f'N{n}'/'full').glob('segment-*/*.scri.hst')):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest[str(path.relative_to(root))]['sha256']
        files[n][str(path.relative_to(root))] = digest
        data = np.loadtxt(path, ndmin=2)
        assert np.isfinite(data).all()
        rows.append(data)
    arrays[n] = np.concatenate(rows)
start = 27.500383990360667/3
end = 29.00025129971555/3
peaks = {}; bounds = {}
for n, data in arrays.items():
    lo = data[np.argmin(abs(data[:, 0]-start)), 0]
    hi = data[np.argmin(abs(data[:, 0]-end)), 0]
    assert abs(lo-start) < 1e-9 and abs(hi-end) < 1e-9
    selected = (data[:, 0] > lo+1e-10) & (data[:, 0] <= hi+1e-10)
    # Retain duplicate/finalization rows: their union preserves split windows.
    peaks[n] = np.max(data[selected], axis=0)
    bounds[n] = [3*float(lo), 3*float(hi)]
result = dict(window_M0=bounds, method='Max over all native subwindow rows inside matched endpoint interval; retain duplicate rows in the union; no interpolation of extrema.', quantities={}, source_sha256=files)
for column, name in [(2, 'solved_compatibility'), (3, 'clock'), (6, 'chiQ_minus_one'),
                     (7, 'P_plus_three'), (8, 'Theta'), (11, 'connection_regularity'),
                     (12, 'shear_regularity')]:
    a, b = peaks[12][column], peaks[16][column]
    result['quantities'][name] = dict(N12=float(a), N16=float(b), N16_over_N12=float(b/a),
                                     two_grid_apparent_p=float(np.log(a/b)/np.log(16/12)))
(here/'previous-N12-N16-scri.json').write_text(json.dumps(result, indent=2)+'\n')
