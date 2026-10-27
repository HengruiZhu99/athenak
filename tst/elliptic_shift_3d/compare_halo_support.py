#!/usr/bin/env python3
"""Same-mesh halo controls versus the independently completed radial pilot."""
import json
from pathlib import Path
import sys
import compare_coupled_family as comparison

comparison.PROFILE_GRID_NOTE = 'same mesh, halo support controls'
comparison.CASES = (
    ('amr_core_n6', 'Original halos', 'amr-core-pilot-20261027',
     'coupled.csv', 'coupled.stdout', 'coupled-command.json', [.45,.55,.60,.65,.70,.75,1]),
    ('radius_012', 'Log halos: radius .12', 'amr-log-halo-coupled-20261027',
     'coupled.csv', 'pilot.stdout', 'command.json', [.45,.55,.60,.65,.70,.75,1]),
    ('radius_024', 'Log halos: radius .24', 'amr-expanded-halo-coupled-20261027',
     'coupled.csv', 'pilot.stdout', 'manifest.json', [.45,.55,.60,.65,.70,.75,1]))

if __name__ == '__main__':
    comparison.main()
    # The reused analyzer also serves differing-grid families. Bind this run's
    # more specific interpretation without rerunning or altering any evolution.
    output = Path(sys.argv[sys.argv.index('--output')+1])
    path = output/'comparison.json'
    data = json.loads(path.read_text())
    data['scope'] = ('Same 47088-cell full3D RK4 mesh at t=.0002 with original, '
                     'radius.12 and radius.24 positive-log halo controls. All '
                     'scientific FAIL. Unchanged independent N2049/N1025 radial '
                     'references, every active cell and raw angular scatter '
                     'retained. This is a halo-discretization comparison, not '
                     'spatial convergence or late collapse reproduction.')
    data['wrapper_source'] = str(Path(__file__).resolve())
    data['wrapper_sha256'] = comparison.digest(Path(__file__))
    path.write_text(json.dumps(data, indent=2)+'\n')
