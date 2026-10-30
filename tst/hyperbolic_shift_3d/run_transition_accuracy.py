#!/usr/bin/env python3
"""Bounded analytic-field transition audit; no evolution or scheduler actions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
binary = a.binary.resolve(); deadline = time.monotonic()+400
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
repo = Path(__file__).resolve().parents[2]
sources = ['tst/hyperbolic_shift_3d/transition_accuracy.cpp',
           'tst/hyperbolic_shift_3d/fixed_composite_mesh.hpp',
           'src/mesh/cubed_sphere/composite.hpp',
           'src/mesh/cubed_sphere/composite_difference.hpp',
           'src/mesh/cubed_sphere/map.hpp']
(a.out/'source-hashes.json').write_text(json.dumps({name: sha(repo/name) for name in sources}, indent=2)+'\n')
records = []
for scheme in ['legacy-v7', 'matched-nonoverlap', 'control-l3', 'thin-l1']:
    for n in [12, 16, 24]:
        name = f'{scheme}-n{n}'; command = [str(binary), scheme, str(n), str(a.out/(name+'.json'))]
        start = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, timeout=min(120, deadline-start))
        (a.out/(name+'.log')).write_text(result.stdout+result.stderr)
        records.append(dict(name=name, command=command, returncode=result.returncode,
                            seconds=time.monotonic()-start, binary_sha256=sha(binary)))
        (a.out/'commands.json').write_text(json.dumps(records, indent=2)+'\n')
        assert result.returncode == 0, (name, result.stderr)
        data = json.loads((a.out/(name+'.json')).read_text())
        assert data['scheme'] == scheme and data['n'] == n and not data['vertex_centered']
        assert data['constant_error'] < 1e-10 and data['minimum_radius'] > 0
        assert len(data['errors']) == 16
        print(name, 'completed spatial measurement', flush=True)
(a.out/'completion.json').write_text(json.dumps(dict(status='completed', cases=len(records),
    scope='Twelve manufactured spatial measurements. Completion is not an evolution stability or penalty qualification.'), indent=2)+'\n')
