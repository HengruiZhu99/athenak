#!/usr/bin/env python3
"""Fixed-mesh positive-log derivative control; retains all-cell/raw-angle errors."""
import json
from pathlib import Path
import sys
import compare_coupled_family as comparison

comparison.PROFILE_GRID_NOTE = 'same mesh, fixed puncture derivative control'
comparison.CASES = (
    ('baseline', 'Polynomial + log', 'amr-expanded-halo-coupled-20261027',
     'coupled.csv', 'pilot.stdout', 'manifest.json', [.45,.55,.60,.65,.70,.75,1]),
    ('supported_rho', 'Polynomial + log + rho', 'supported-puncture-coupled-20261027',
     'coupled.csv', 'pilot.stdout', 'manifest.json', [.45,.55,.60,.65,.70,.75,1]))

if __name__ == '__main__':
    comparison.main()
    output = Path(sys.argv[sys.argv.index('--output')+1])
    path = output/'comparison.json'
    data = json.loads(path.read_text())
    data['scope'] = ('Same47088-cell full3D RK4 mesh at t=.0002, positive logs, '
                     'radius.24 halo support; only puncture derivative basis differs. '
                     'Every active cell and raw angular scatter retained. Both '
                     'scientific FAIL; no late collapse or generic moving qualification.')
    data['wrapper_source'] = str(Path(__file__).resolve())
    data['wrapper_sha256'] = comparison.digest(Path(__file__))
    path.write_text(json.dumps(data, indent=2)+'\n')
