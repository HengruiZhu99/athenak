#!/usr/bin/env python3
"""Compare all native N12 fields and ChiRecords across shared-file MPI writers."""
import argparse
import json
from pathlib import Path

import numpy as np

from n12_checkpoint import (NF, NR, compare_fields, decode, load_mapping,
    parse_native, require, sha256, validate_header, chi_summary)
from n12_checkpoint_cross_platform import identity_difference
from n12_checkpoint_gpu_control import native_coordinates


def complete_records(actual, expected, keys):
    require(actual.shape == expected.shape and actual.shape[1] == NR, 'record coverage differs')
    different = actual.view('<u8') != expected.view('<u8')
    fields = []
    for column in range(NR):
        delta = np.abs(actual[:, column]-expected[:, column]); worst = int(delta.argmax())
        fields.append(dict(column=column, different_values=int(different[:, column].sum()),
            max_absolute=float(delta[worst]), worst=dict(legacy_ordinal=worst,
                native_key=keys[worst].tolist(), actual=float(actual[worst, column]),
                expected=float(expected[worst, column]))))
    return dict(comparisons=int(actual.size), bitwise_equal=not different.any(),
                different_values=int(different.sum()), fields=fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference-ranks', type=int, required=True)
    parser.add_argument('--candidate-ranks', type=int, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--reference-coordinates', type=Path)
    parser.add_argument('--candidate-coordinates', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    manifest, mapping = load_mapping(args.mapping)
    reference, candidate = parse_native(args.reference), parse_native(args.candidate)
    # Across ranks on the same backend, exact numerical identity is mandatory.
    # CPU sidecar differences remain an explicitly retained separate audit.
    same_backend = dict(manifest, physics_identity=reference['identity'].decode())
    validate_header(reference, same_backend, args.step); validate_header(candidate, same_backend, args.step)
    require(reference['metadata'][4] == args.reference_ranks and
            candidate['metadata'][4] == args.candidate_ranks, 'writer ranks differ from requested comparison')
    ref, got = decode(reference, mapping), decode(candidate, mapping)
    report = dict(passed=True, step=args.step, time=candidate['time'], dt=candidate['dt'],
        gate='bitwise equality of every owned bulk/scri field and every ChiRecord member',
        same_backend_numerical_identity_exact=True,
        cpu_sidecar_identity_differences_preserved=identity_difference(manifest['physics_identity'], reference['identity'].decode()),
        reference=dict(path=str(args.reference.resolve()), sha256=sha256(args.reference), ranks=args.reference_ranks),
        candidate=dict(path=str(args.candidate.resolve()), sha256=sha256(args.candidate), ranks=args.candidate_ranks),
        mapping_sha256=manifest['mapping_sha256'], fields={}, records={},
        source_sha256={str(Path(__file__).with_name(name).resolve()): sha256(Path(__file__).with_name(name))
            for name in ('n12_checkpoint_rank_compare.py', 'n12_checkpoint.py',
                         'n12_checkpoint_cross_platform.py', 'n12_checkpoint_gpu_control.py')},
        candidate_chi_summary=chi_summary(got, mapping), reference_chi_summary=chi_summary(ref, mapping))
    for region in ('bulk', 'surface'):
        fields = compare_fields(got[region][:, :NF], ref[region][:, :NF], mapping[region+'_keys'],
                                mapping[region+'_xyz'], mapping[region+'_weight'],
                                'short' if args.step <= 8 else 'long')
        fields['different_values'] = int(np.count_nonzero(got[region][:, :NF].copy().view('<u8') !=
                                                         ref[region][:, :NF].copy().view('<u8')))
        records = complete_records(got[region][:, NF:], ref[region][:, NF:], mapping[region+'_keys'])
        report['fields'][region] = fields; report['records'][region] = records
        report['passed'] &= fields['bitwise_equal'] and records['bitwise_equal']
    require(bool(args.reference_coordinates) == bool(args.candidate_coordinates),
            'both actual coordinate directories are required together')
    if args.reference_coordinates:
        a, ah = native_coordinates(args.reference_coordinates, mapping)
        b, bh = native_coordinates(args.candidate_coordinates, mapping)
        changed = a.view('<u8') != b.view('<u8')
        report['coordinates'] = dict(comparisons=int(a.size), bitwise_equal=not changed.any(),
            different_components=int(changed.sum()), max_absolute=float(abs(a-b).max()),
            reference_sha256=ah, candidate_sha256=bh,
            scope='actual frozen saved3551 replay geometry on each decomposition; bulk only')
        report['passed'] &= report['coordinates']['bitwise_equal']
    report['passed'] = bool(report['passed'])
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], step=args.step,
        field_differences={k:v['different_values'] for k,v in report['fields'].items()},
        record_differences={k:v['different_values'] for k,v in report['records'].items()}), indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
