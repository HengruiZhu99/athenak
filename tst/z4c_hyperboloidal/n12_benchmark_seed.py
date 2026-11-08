#!/usr/bin/env python3
"""Make a private SP1 benchmark seed from admitted native cycle2486.

Only the legacy field payload is replaced. Historical SP1 header/identity and
unused compensation bytes are preserved. All eight native chi-record members
are exported separately because SP1 has no chi-history storage. No evolution,
projection, interpolation, source checkpoint mutation or job is performed.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from n12_checkpoint import (DT, NBULK, NSURFACE, NF, NR, decode, expected_time,
                            load_mapping, parse_native, require, sha256,
                            validate_header, chi_summary)
from n12_checkpoint_gpu_control import old_checkpoint, resolved_comparison

STEP = 2486
END = 2556
NATIVE = 'dd4b86ab42c10827f14bd339930cb4d2ffaf9fb5e2b43953d1e5d7924a9fc1ec'
SCIENCE = 'babb035acdcb505d2f8407815981731168ae2c75c3217dcc43ba2ce56c363bd9'
REVIEW = '6322dcb077c8ac51bc105791d024cd93686994cbc89ffaf90b5819a9f3b99d51'
TEMPLATE = '0bac8c0e4e1d0378c58faf1ff5b15926a93710c3915d36922b12b2aa475ba731'
CONFIG = '9c786cc8c3312ddf93389a4b3f1f6e32921811a61f4d131ee53990fb80a1d363'


def convert(args):
    output = args.output.resolve()
    require(not output.exists(), 'new private output directory required')
    inputs = {name: getattr(args, name).resolve() for name in
              ('native', 'science', 'review', 'template', 'configuration')}
    for name, digest in [('native', NATIVE), ('science', SCIENCE), ('review', REVIEW),
                         ('template', TEMPLATE), ('configuration', CONFIG)]:
        require(sha256(inputs[name]) == digest, 'unqualified input: '+name)
    science = json.loads(inputs['science'].read_text())
    review = json.loads(inputs['review'].read_text())
    require(science['passed'] and review['passed'] and
            science['end_cycle'] == review['end_cycle'] == STEP and
            review['science_admission_sha256'] == SCIENCE and
            science['checkpoint_sha256'] == review['checkpoint_sha256'] == NATIVE,
            'cycle2486 admission/review binding differs')
    manifest, mapping = load_mapping(args.mapping)
    require(manifest['mapping_sha256'] ==
            '484bd5e77d83c537ce564e2f02b68bd3544680ced65d81abdf8a69216434b15d',
            'actual Aurora mapping required')
    info = parse_native(inputs['native']); validate_header(info, manifest, STEP)
    values = decode(info, mapping)
    # The existing parser validates the complete historical SP1 framing and
    # snapshots, including all compensation values, before any private copy.
    offset = inputs['template'].stat().st_size-2*(NBULK+NSURFACE)*NF*8
    old = np.memmap(inputs['template'], dtype='<f8', mode='r', offset=offset)
    bulk_size, surface_size = NBULK*NF, NSURFACE*NF
    original = old_checkpoint(inputs['template'], STEP,
        old[:bulk_size].reshape(NBULK, NF),
        old[2*bulk_size:2*bulk_size+surface_size].reshape(NSURFACE, NF))
    require(resolved_comparison(manifest['physics_identity'],
            original['physics_identity'])['all_common_exact'],
            'native/legacy resolved physics differ')
    output.mkdir(parents=True)
    result = dict(schema='native-n12-matched-benchmark-seed-v1', passed=False,
                  start_cycle=STEP, end_cycle=END, time=info['time'], dt=DT,
                  input_sha256={str(p):sha256(p) for p in inputs.values()},
                  mapping_sha256=manifest['mapping_sha256'])
    try:
        target = output/'restart.bin'; shutil.copyfile(inputs['template'], target)
        data = np.memmap(target, dtype='<f8', mode='r+', offset=offset)
        data[:bulk_size] = values['bulk'][:, :NF].reshape(-1)
        data[2*bulk_size:2*bulk_size+surface_size] = values['surface'][:, :NF].reshape(-1)
        data.flush(); del data
        converted = old_checkpoint(target, STEP, values['bulk'][:, :NF],
                                   values['surface'][:, :NF])
        with inputs['template'].open('rb') as f: prefix = f.read(offset)
        with target.open('rb') as f: require(f.read(offset) == prefix, 'SP1 prefix changed')
        converted_data = np.memmap(target, dtype='<f8', mode='r', offset=offset)
        for lo, hi in [(bulk_size, 2*bulk_size),
                       (2*bulk_size+surface_size, 2*(bulk_size+surface_size))]:
            require(np.array_equal(old[lo:hi].view('<u8'), converted_data[lo:hi].view('<u8')),
                    'unused compensation payload changed')
        for region, filename in [('bulk', 'chi-records.bin'),
                                 ('surface', 'surface-chi-records.bin')]:
            values[region][:, NF:].astype('<f8').tofile(output/filename)
            actual = np.memmap(output/filename, dtype='<f8', mode='r',
                               shape=(values[region].shape[0], NR))
            require(np.array_equal(actual.view('<u8'), values[region][:, NF:].view('<u8')),
                    'chi-record copy changed')
        shutil.copyfile(inputs['configuration'], output/'configuration.cfg')
        # Native restart suppresses an initial horizon search. Give the legacy
        # final-only finder exactly the native persisted seed and finder policy.
        fastflow = {k.split('/', 1)[1]:v for k,v in info['parameters'].items()
                    if k.startswith('fastflow/')}
        require(float(fastflow['last_a0_0']) > 0 and fastflow['ah_found_a0_0'] in ('1','true'),
                'missing accepted native horizon seed')
        lines = ['<time>', 'tlim = 2', 'nlim = '+str(END), 'dt = '+repr(DT),
                 '<mesh>', 'cells_per_axis = 12', '<numerics>', 'ko = .1',
                 'chi_floor = 1e-10', '<tracker>', 'enabled = false',
                 '<problem>', 'configuration = configuration.cfg', '<fastflow>']
        lines += [name+' = '+value for name,value in sorted(fastflow.items())]
        (output/'run.in').write_text('\n'.join(lines)+'\n')
        names = ('restart.bin','chi-records.bin','surface-chi-records.bin',
                 'configuration.cfg','run.in')
        result.update(passed=True, field_values=NF*(NBULK+NSURFACE),
            record_values=NR*(NBULK+NSURFACE), initial_fields_bitwise_native=True,
            initial_records_bitwise_native=True, native_chi_summary=chi_summary(values,mapping),
            header_and_identity_preserved=True, compensation_bytes_preserved=True,
            compensation_scope='SP1 spherical-policy RK branches return before compensated updates; preserved bytes are not active evolution state in this formulation.',
            legacy_checkpoint=converted, fastflow_parameters=fastflow,
            warmup=10, window_steps=20, windows=3,
            output_sha256={name:sha256(output/name) for name in names},
            scope='Private format conversion only; exact admitted native2486 fields and all8 records. No numerical evaluation or performance admission.')
        for path, digest in result['input_sha256'].items():
            require(sha256(path) == digest, 'source input changed during conversion')
    except Exception as error:
        result['validation_error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        result['source_sha256'] = {str(Path(__file__).with_name(name).resolve()):
            sha256(Path(__file__).with_name(name)) for name in
            ('n12_benchmark_seed.py','n12_checkpoint.py','n12_checkpoint_gpu_control.py')}
        (output/'seed.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native','science','review','template','configuration','mapping','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = convert(args)
    print(json.dumps({k:result[k] for k in ('passed','start_cycle','end_cycle',
          'field_values','record_values','initial_fields_bitwise_native',
          'initial_records_bitwise_native')},indent=2))
