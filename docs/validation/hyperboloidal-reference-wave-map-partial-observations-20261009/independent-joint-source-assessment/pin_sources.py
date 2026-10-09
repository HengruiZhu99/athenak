#!/usr/bin/env python3
"""Hash inventory for pencil note only; no scientific calculation or query."""
from pathlib import Path
import hashlib
import json
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
assert not (HERE / 'receipt.json').exists()
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
sources = {
    'S01': 'build-layer-research/continuum/conformal-reference-stationary-BH-pencil-20261009/DERIVATION.md',
    'S01-index': 'build-layer-research/continuum/conformal-reference-stationary-BH-pencil-20261009/index.json',
    'S02': 'build-layer-research/reference-wave-map-proposal-20261009/DERIVATION.md',
    'S03': 'build-layer-research/continuum/reference-wave-map-gauge-20261009/reference_wave_map.hpp',
    'S04': 'build-layer-research/continuum/inner-wave-map-blend-feasibility-20261009/ASSESSMENT.md'}
pins = {key: {'path': value, 'sha256': sha(ROOT / value)} for key, value in sources.items()}
assert pins['S01-index']['sha256'] == '815a1a489bf9bf9843817e9291259030254513bb57aa5238074d15ee86f0ed6d'
receipt = {'independent_pencil_necessary_matching_only': True,
    'sources': pins, 'derivation_sha256': sha(HERE / 'DERIVATION.md'),
    'inventory_source_sha256': sha(Path(__file__)), 'python_version': sys.version,
    'CAS_or_numerical_calculation': False, 'kernel_queries': 0, 'native_steps': 0,
    'necessary_radial_offset': 'd=-3M from preferred BoxOmega through Omega^2.',
    'necessary_compact_temporal_source_limits': {'physical_reference_base': '+6M/S^2',
                                               'conformal_reference_base': '-6M/S^2'},
    'full_stationary_solution_or_offconstraint_closure_established': False,
    'command': 'python3 build-layer-research/continuum/preferred-Box-mass-log-source-pencil-20261009/pin_sources.py'}
(HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
files = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
         for path in sorted(HERE.iterdir()) if path.is_file()}
(HERE / 'index.json').write_text(json.dumps({'scope': 'Immutable pencil-only joint source necessity; no adoption.',
    'files': files, 'receipt_sha256': sha(HERE / 'receipt.json')}, indent=2) + '\n')
print(json.dumps({'index_sha256': sha(HERE / 'index.json'), 'receipt_sha256': sha(HERE / 'receipt.json'),
                  'derivation_sha256': sha(HERE / 'DERIVATION.md')}, indent=2))
