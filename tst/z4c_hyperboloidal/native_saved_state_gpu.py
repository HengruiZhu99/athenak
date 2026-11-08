#!/usr/bin/env python3
"""Build a test-only saved-state device replay from native CMake provenance.

No scheduler submission or MPI launch is performed. Launch the resulting binary
with the site's existing tile wrapper. Strict replay failures remain failures.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import struct
import subprocess

MACROS = ('HYPERBOLIC_BROAD_SMOOTH_A', 'HYPERBOLIC_CONFIGURED_FORMULATION',
          'HYPERBOLIC_DIRECT_OUTER', 'HYPERBOLIC_INNER_TELEGRAPHER',
          'HYPERBOLIC_NORMALIZED_INNER_SHIFT', 'HYPERBOLIC_RESTORE_INNER_DAMPING',
          'HYPERBOLIC_SHORT_INNER_GAUGE', 'HYPERBOLIC_SMOOTH_GAUGE_BLEND',
          'HYPERBOLIC_SPHERICAL_POLICY')
MASK_COUNTS = ('all', 'positive_weight', 'device_outer', 'host_outer', 'near_any',
               'near_positive', 'device1_host0', 'device0_host1', 'weighted_1_0',
               'weighted_0_1', 'invalid_any', 'invalid_zero_weight',
               'exact_device_cut', 'exact_host_cut')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    directory, legacy, output = args.build.resolve(), args.legacy.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_suffix('.cpp').resolve()
    database = (args.compile_database or directory/'compile_commands.json').resolve()
    entries = json.loads(database.read_text())
    selected = [entry for entry in entries if
                Path(entry['file']).as_posix().endswith('z4c_hyperboloidal/diagnostics.cpp')]
    if len(selected) != 1:
        raise RuntimeError('compile_commands.json must contain exactly one native diagnostics.cpp')
    entry = selected[0]
    root = Path(entry['file']).resolve().parents[2]
    compile_cwd = Path(entry['directory']).resolve()
    original = entry.get('arguments') or shlex.split(entry['command'])
    obj, executable = output/'native_saved_state_gpu.o', output/'native_saved_state_gpu'
    command, seen_source, seen_output = [], 0, 0
    index = 0
    while index < len(original):
        arg = original[index]
        if arg == '-o':
            command += ['-o', str(obj)]; seen_output += 1; index += 2; continue
        if arg in ('-MF', '-MT', '-MQ'):
            index += 2; continue
        if arg in ('-MD', '-MMD', '-MP'):
            index += 1; continue
        candidate = Path(arg) if arg.startswith('/') else compile_cwd/arg
        if not arg.startswith('-') and candidate.resolve() == Path(entry['file']).resolve():
            command.append(str(source)); seen_source += 1
        else:
            command.append(arg)
        index += 1
    if seen_source != 1 or seen_output != 1 or '-c' not in command:
        raise RuntimeError('unrecognized native compile command; refusing inferred compiler/flags')
    if '-ffp-contract=off' not in command:
        raise RuntimeError('native diagnostics compile entry does not preserve pinned contraction policy')
    command += ['-D'+name+'=1' for name in MACROS]
    command += ['-I'+str(legacy/'src'), '-I'+str(legacy), '-MMD', '-MF',
                str(output/'native_saved_state_gpu.d'), '-MT', str(obj)]
    link_path = directory/'src/CMakeFiles/athena.dir/link.txt'
    link = shlex.split(link_path.read_text())
    main = [i for i, token in enumerate(link) if token.endswith('athena.dir/main.cpp.o')]
    if len(main) != 1:
        raise RuntimeError('native link.txt must contain exactly one main.cpp.o')
    link[main[0]] = str(obj)
    link[link.index('-o')+1] = str(executable)
    link_cwd = link_path.parents[2]
    object_paths = [Path(p) if Path(p).is_absolute() else link_cwd/p
                    for p in link if p.endswith(('.o', '.a', '.so', '.dylib')) and p != str(obj)]
    source_paths = sorted(list((root/'src').rglob('*.cpp'))+list((root/'src').rglob('*.hpp')))
    source_paths += [source, Path(__file__).resolve(),
                     Path(__file__).with_name('native_saved_state_gpu_contract.md').resolve()]
    provenance = dict(compile_database=str(database), compile_database_sha256=sha(database),
                      native_compile_entry=entry, compile_command=command, link_command=link,
                      link_file_sha256=sha(link_path), legacy_root=str(legacy),
                      sources={str(p): sha(p) for p in source_paths},
                      objects={str(p.resolve()): sha(p) for p in object_paths})
    if args.reference:
        reference = args.reference.resolve()
        provenance['reference'] = {str(p): sha(p) for p in sorted(reference.iterdir()) if p.is_file()}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    makefile = output/'replay.mk'
    makefile.write_text('.PHONY: all\nall: '+str(executable)+'\n'+str(obj)+':\n\t'
                       +'cd '+shlex.quote(str(compile_cwd))+' && '+shlex.join(command)+'\n'
                       +str(executable)+': '+str(obj)+'\n\t'
                       +'cd '+shlex.quote(str(link_cwd))+' && '+shlex.join(link)+'\n')
    with (output/'build.log').open('w') as log:
        subprocess.run(['make', '-j64', '-B', '-f', str(makefile)], stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    dependencies = (output/'native_saved_state_gpu.d').read_text().replace('\\\n', ' ')
    dependencies = shlex.split(dependencies.split(':', 1)[1])
    legacy_dependencies = [Path(p) if Path(p).is_absolute() else compile_cwd/p for p in dependencies]
    provenance['legacy_dependencies'] = {str(p.resolve()): sha(p) for p in legacy_dependencies
                                        if p.resolve().is_relative_to(legacy)}
    provenance['executable_sha256'] = sha(executable)
    provenance['harness_object_sha256'] = sha(obj)
    provenance['sources_unchanged'] = all(sha(Path(p)) == h for p, h in provenance['sources'].items())
    provenance['objects_unchanged'] = all(sha(Path(p)) == h for p, h in provenance['objects'].items())
    assert provenance['sources_unchanged'] and provenance['objects_unchanged']
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({k: provenance[k] for k in
                     ('executable_sha256', 'harness_object_sha256', 'sources_unchanged', 'objects_unchanged')}, indent=2))


def records(folder):
    result = {}
    for path in sorted(folder.glob('points-rank*.bin')):
        raw = path.read_bytes()
        if raw[:16] != b'NATIVE_DIAG_V1\0\0':
            raise RuntimeError('invalid diagnostic dump magic: '+str(path))
        count, keys, values = struct.unpack_from('<QII', raw, 16)
        assert (keys, values) == (8, 26)
        stride = keys*4+values*8
        assert len(raw) == 32+count*stride
        for offset in range(32, len(raw), stride):
            key = struct.unpack_from('<8i', raw, offset)
            assert key[0] not in result
            value = raw[offset+32:offset+stride]
            assert all(math.isfinite(x) for x in struct.unpack('<26d', value))
            result[key[0]] = (key, value)
    assert set(result) == set(range(229392)), 'incomplete/duplicate canonical diagnostic coverage'
    return result


def audit(args):
    folder = args.output.resolve()
    report = json.loads((folder/'results.json').read_text())
    assert report['step'] == 3551 and report['time'] == 1.6669886746517166
    assert len(report['strict_checks']) == len(report['history_checks']) == 8
    assert [(q['region'], q['field']) for q in report['strict_checks']] == [
        (region, field) for region in range(2) for field in range(4)]
    assert len(report['raw_device']) == len(report['raw_device_host_mask_control']) == 10
    expected_pass = (report['readonly'] and report['mask_counts'][0] == 229392
                     and report['mask_counts'][10] == 0
                     and all(q['passed'] for q in report['strict_checks']))
    assert report['strict_replay_passed'] == expected_pass
    assert report['passed'] == (expected_pass and all(q['passed'] for q in report['history_checks'])
                                and report['horizon']['passed'])
    data = records(folder)
    report['mask_count_names'] = dict(zip(MASK_COUNTS, report['mask_counts']))
    report['device_minus_host_cut_raw'] = [a-b for a, b in zip(
        report['raw_device'][5:], report['raw_device_host_mask_control'][5:])]
    digest = hashlib.sha256()
    for key in sorted(data):
        indices, values = data[key]
        digest.update(struct.pack('<8i', *indices));digest.update(values)
    report['canonical_dump_sha256'] = digest.hexdigest()
    report['dump_files_sha256'] = {p.name: sha(p) for p in sorted(folder.glob('points-rank*.bin'))}
    if args.compare:
        other = records(args.compare.resolve())
        components = [0]*26; mask_differences = [0]*3; max_errors = [0.]*26
        for key, (indices, value) in data.items():
            ref_indices, reference = other[key]
            assert indices[:5] == ref_indices[:5], 'native leaf/index identity changed'
            for f in range(3):
                mask_differences[f] += indices[5+f] != ref_indices[5+f]
            a, b = struct.unpack('<26d', value), struct.unpack('<26d', reference)
            for f in range(26):
                components[f] += value[8*f:8*(f+1)] != reference[8*f:8*(f+1)]
                max_errors[f] = max(max_errors[f], abs(a[f]-b[f]))
        report['cross_rank'] = dict(reference=str(args.compare.resolve()),
            field_bit_differences=components, field_max_absolute=max_errors,
            fixed_device_host_AH_mask_differences=mask_differences,
            exact=not any(components+mask_differences))
    (folder/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['passed'] or (args.compare and not report['cross_rank']['exact']):
        raise SystemExit(2)


parser = argparse.ArgumentParser(description=__doc__)
sub = parser.add_subparsers(dest='mode', required=True)
p = sub.add_parser('build')
p.add_argument('--build', type=Path, required=True)
p.add_argument('--legacy', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--compile-database', type=Path)
p.add_argument('--reference', type=Path)
p = sub.add_parser('audit')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--compare', type=Path)
args = parser.parse_args()
build(args) if args.mode == 'build' else audit(args)
