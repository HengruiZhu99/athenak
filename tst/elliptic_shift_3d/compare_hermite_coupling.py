#!/usr/bin/env python3
"""Isolate the optional boundary sampler on the fixed matched radial pilot."""
import json
from pathlib import Path
import sys
import compare_coupled_family as comparison

comparison.PROFILE_GRID_NOTE = 'same mesh, boundary derivative control'
comparison.CASES = (
    ('supported', 'Fixed puncture basis', 'supported-puncture-coupled-20261027',
     'coupled.csv', 'pilot.stdout', 'manifest.json', [.45,.55,.60,.65,.70,.75,1]),
    ('hermite', '+ boundary Hermite jets', 'hermite-supported-coupled-20261027',
     'coupled.csv', 'pilot.stdout', 'manifest.json', [.45,.55,.60,.65,.70,.75,1]))

if __name__ == '__main__':
    comparison.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    path = output/'comparison.json'
    data = json.loads(path.read_text())
    data['scope'] = ('Same47088-cell full3D RK4 at t=.0002, positive logs, fixed '
                     'puncture basis and radius.24 halo support; only boundary '
                     'derivative sampler differs. All cells and raw angular '
                     'scatter retained. Both scientific FAIL; no late collapse qualification.')
    data['wrapper_source'] = str(Path(__file__).resolve())
    data['wrapper_sha256'] = comparison.digest(Path(__file__))
    path.write_text(json.dumps(data, indent=2)+'\n')
