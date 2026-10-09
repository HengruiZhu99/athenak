#!/usr/bin/env python3
"""Source hash inventory only; no CAS, numerical/kernel or native calls."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert not (HERE / 'receipt.json').exists()
sources = {
    'S01': 'build-layer-research/reference-wave-map-proposal-20261009/DERIVATION.md',
    'S02': 'build-layer-research/continuum/reference-wave-map-gauge-20261009/reference_wave_map.hpp',
    'S03': 'src/z4c/hyperboloidal/layer_reference.hpp',
    'S04': 'build-layer-research/continuum/inner-wave-map-blend-feasibility-20261009/ASSESSMENT.md',
    'S05': 'build-layer-research/continuum/inner-wave-map-blend-feasibility-20261009/index.json'}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
pins = {key: {'path': value, 'sha256': sha(ROOT / value)} for key, value in sources.items()}
assert pins['S02']['sha256'] == '56d61c56bf37bf33591bec74c029c686c7d39f7abbe62dd090becda8e7176e28'
receipt = {'prepared_independent_pencil_derivation': True,
    'sources': pins, 'derivation_sha256': sha(HERE / 'DERIVATION.md'),
    'inventory_source_sha256': sha(Path(__file__)),
    'scope': 'Conditional stationary spherical exact-Einstein asymptotic necessity; not existence, evolution, off-constraint closure or adoption.',
    'CAS_or_numerical_calculation': False, 'kernel_queries': 0, 'native_steps': 0,
    'source_only_hash_inventory_command': 'python3 build-layer-research/continuum/conformal-reference-stationary-BH-pencil-20261009/pin_sources.py',
    'python_version': sys.version,
    'key_necessary_equations': ['R^2 F psi_prime Omega(f)^4=D; smooth end implies D=0.',
        'Spatial logarithmic balance 2c log(R/Rstar)-5c+6M+2d=0; c=0,d=-3M.',
        'Smooth future hyperboloidal height requires coefficient 2M of log(R/ell).'],
    'full_stationary_solution_established': False,
    'Minkowski_reference_retained': True,
    'later_BH_requirement': 'Wormhole-to-trumpet inner transition with Minkowski hyperboloidal reference throughout.'}
(HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
files = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
         for path in sorted(HERE.iterdir()) if path.is_file()}
(HERE / 'index.json').write_text(json.dumps({'scope': receipt['scope'], 'files': files,
    'files_excluding_index': len(files), 'receipt_sha256': sha(HERE / 'receipt.json')}, indent=2) + '\n')
print(json.dumps({'index_sha256': sha(HERE / 'index.json'),
                  'receipt_sha256': sha(HERE / 'receipt.json'),
                  'derivation_sha256': sha(HERE / 'DERIVATION.md')}, indent=2))
