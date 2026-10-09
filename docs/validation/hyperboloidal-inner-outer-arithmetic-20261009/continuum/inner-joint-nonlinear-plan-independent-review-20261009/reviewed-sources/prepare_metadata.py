#!/usr/bin/env python3
"""Source-only plan pinning. No numerical import, compilation or query."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/Users/hz0693/research/hyperboloidal')
HERE = Path(__file__).resolve().parent
PRINCIPAL = ROOT / 'build-layer-research/continuum/inner-joint-principal-gate-v3-held-20261009'
RWM = ROOT / 'build-layer-research/continuum/reference-wave-map-gauge-20261009/immutable-local-reference-wave-map-20261009'


def digest(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data)}


def read(path):
    return json.loads(Path(path).read_text())


def write_new(name, data):
    path = HERE / name
    if path.exists():
        raise RuntimeError('Refuse overwrite: ' + str(path))
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def main():
    if (HERE / 'source-index.json').exists():
        raise RuntimeError('One-shot metadata source already indexed')
    required = {
        PRINCIPAL / 'source-index.json': 'ea06abf5a285ede3fb4224904a7b8932f2664dbc07c5b31718c19353570c7b4a',
        PRINCIPAL / 'gauge_proposal.hpp': '57f4801e6713448629c0f289de1aa8da6d39f98b9a12186b86a04ff71f76027e',
        PRINCIPAL / 'attempts/gate001/receipt.json': 'bf778caf62d6596261d5c15a7307b791bb39be321f0bc0bc834adfdeeef4d83b',
        RWM / 'index.json': '25f09ff04e1f7067147b9bb75ff916750f989beb4366477b1db8748fde74dd31',
        RWM / 'reference_wave_map.hpp': '56d61c56bf37bf33591bec74c029c686c7d39f7abbe62dd090becda8e7176e28',
    }
    pins = {}

    def add(path, expected=None, role='protected input'):
        row = digest(path)
        if expected is not None and row['sha256'] != expected:
            raise RuntimeError('Input drift: ' + row['path'])
        row['role'] = role
        prior = pins.get(row['path'])
        if prior and prior['sha256'] != row['sha256']:
            raise RuntimeError('Conflicting pin: ' + row['path'])
        pins[row['path']] = row

    for path, expected in required.items():
        add(path, expected, 'explicit scientific prerequisite')
    receipt = read(PRINCIPAL / 'attempts/gate001/receipt.json')
    assert receipt['completed'] is True and receipt['passed'] is True
    assert receipt['returncode'] == 0 and receipt['inputs_unchanged'] is True
    assert receipt['exact_scalar_summary']['passed'] is True
    assert receipt['exact_scalar_summary']['exact_scalar_cases'] == 18
    assert all(s['passed'] is True and s['actual20_cases'] == 118
               for s in receipt['analysis_summaries'].values())

    principal_index = read(PRINCIPAL / 'source-index.json')
    for key in ('files', 'external_inputs'):
        for row in principal_index[key]:
            add(row['path'], row['sha256'], 'principal source/index dependency')
    principal_recipe = read(PRINCIPAL / 'recipe.json')
    add(PRINCIPAL / 'recipe.json', role='principal compile recipe')
    for row in principal_recipe['protected_inputs']:
        add(row['path'], row['sha256'], 'principal protected compiler/source input')
    rwm_index = read(RWM / 'index.json')
    for row in rwm_index['files']:
        add(RWM / row['path'], row['sha256'], 'frozen RWM source/result dependency')
    for rel, expected in rwm_index['external_source_inputs'].items():
        add(ROOT / rel, expected, 'compiled production/source identity')
    for mode, dependencies in rwm_index['external_compiler_dependencies'].items():
        for path, expected in dependencies.items():
            add(path, expected, 'RWM ' + mode + ' compiler dependency')

    # Byte copies are review aids, never executable replacements of the inputs.
    copies = HERE / 'inputs'
    copies.mkdir()
    copied = []
    for path, name in [
        (PRINCIPAL / 'gauge_proposal.hpp', 'principal-only-gauge_proposal.hpp'),
        (RWM / 'reference_wave_map.hpp', 'frozen-reference_wave_map.hpp'),
        (RWM / 'test_support.hpp', 'frozen-test_support.hpp'),
        (RWM / 'dual_helpers.hpp', 'frozen-dual_helpers.hpp'),
        (RWM / 'nonlinear_values.hpp', 'frozen-nonlinear_values.hpp'),
    ]:
        target = copies / name
        shutil.copyfile(path, target)
        row = digest(target)
        assert row['sha256'] == digest(path)['sha256']
        copied.append({'source': str(path), **row})

    before = sorted(pins.values(), key=lambda x: x['path'])
    for row in before:
        assert digest(row['path'])['sha256'] == row['sha256']
    write_new('input-pins.json', before)
    write_new('source-preparation.json', {
        'source_only': True, 'execution_admitted': False,
        'scientific_helper_implemented': False,
        'metadata_only_standard_library': True,
        'launch_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                             cwd=ROOT, text=True).strip(),
        'compiled_production_identity': rwm_index['production_implementation'],
        'principal_receipt_completed_passed_inputs_unchanged': True,
        'principal_actual20_cases': 118, 'principal_exact_scalar_cases': 18,
        'protected_inputs': len(before), 'protected_inputs_unchanged': True,
        'review_only_copies': copied,
        'no_scientific_import_compile_query_array_load_or_execution': True,
    })
    files = [digest(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
    write_new('source-index.json', {
        'source_only': True, 'execution_admitted': False,
        'scope': 'held inner nonlinear helper equations/cases/thresholds only',
        'scientific_helper_implemented': False,
        'files': files, 'file_count': len(files),
        'protected_inputs': len(before),
        'protected_inputs_unchanged': True,
        'no_native_or_BH_adoption': True,
    })
    print(json.dumps({'source_index': digest(HERE / 'source-index.json'),
                      'protected_inputs': len(before), 'files': len(files),
                      'execution_admitted': False}, sort_keys=True))


if __name__ == '__main__':
    main()
