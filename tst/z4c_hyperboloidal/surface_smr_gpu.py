#!/usr/bin/env python3
"""Build/verify the unchanged refined-surface oracle; never launch GPU work.

Reuse the coherent AMR candidate and the actual diagnostics.cpp SYCL recipe.
Only the original main object and link output are replaced. The C++ fixture,
input and numerical gates are pinned to their existing CPU-qualified bytes.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from native_amr_gpu import check_inputs, require, sha
from native_amr_candidate_cmake import check as check_candidate

BASELINE = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'
CANDIDATE = '442594b847ff0eb56407513da48e96930ec41dbe537a69c202fd802da4ea6a31'
PROOF = 'f4941ccd95ebee95aac62842aa6e06266872b65621a80f43e8ad5ae4532eb340'
FIXTURE = '2f4641583d28e2d875c4234b1e2c3386710c3f20fe84e664b8e66ad43345fad3'
INPUT = '0a1ac2023d0abb6ad17931090a61c97a3505b97e7fd4f472f6d5a9342ebc2719'


def recipe(database, source, obj, dependency):
    entries = json.loads(database.read_text())
    selected = [e for e in entries if Path(e['file']).as_posix().endswith(
        '/z4c_hyperboloidal/diagnostics.cpp')]
    require(len(selected) == 1, 'need one coherent diagnostics.cpp recipe')
    entry = selected[0]
    cwd = Path(entry['directory']).resolve()
    native = (cwd / entry['file']).resolve()
    original = entry.get('arguments') or shlex.split(entry['command'])
    command, source_count, output_count, i = [], 0, 0, 0
    while i < len(original):
        token = original[i]
        if token == '-o':
            command += ['-o', str(obj)]; output_count += 1; i += 2; continue
        if token in ('-MF', '-MT', '-MQ'):
            i += 2; continue
        if token in ('-MD', '-MMD', '-MP'):
            i += 1; continue
        if not token.startswith('-') and (cwd / token).resolve() == native:
            command.append(str(source)); source_count += 1
        else:
            command.append(token)
        i += 1
    require(source_count == output_count == 1 and '-c' in command,
            'unrecognized native compile recipe')
    for flag in ('-fsycl', '-fp-model=precise', '-ffp-contract=off', '-O2'):
        require(flag in command, 'qualified SYCL/FP flag absent: '+flag)
    command += ['-MMD', '-MF', str(dependency), '-MT', str(obj)]
    return entry, cwd, native.parents[2], command


def link_recipe(path, obj, executable):
    command = shlex.split(path.read_text())
    indices = [i for i, token in enumerate(command)
               if token.endswith('athena.dir/main.cpp.o')]
    require(len(indices) == 1 and command.count('-o') == 1,
            'unrecognized coherent link recipe')
    command[indices[0]] = str(obj)
    command[command.index('-o')+1] = str(executable)
    return command


def verify_record(record):
    require(record['schema'] == 'native-surface-smr-gpu-build-v1' and
            record['status'] == 'built' and record['build_exit_code'] == 0,
            'surface SMR helper did not build successfully')
    require(record['candidate_build_sha256'] == PROOF and
            record['native_executable_sha256'] == CANDIDATE and
            record['baseline_native_executable_sha256'] == BASELINE,
            'wrong coherent candidate/baseline identity')
    require(sha(record['fixture']) == FIXTURE and sha(record['input']) == INPUT,
            'original fixture/input changed')
    check_inputs(record)  # Includes full coherent candidate and baseline checks.
    for name in ('executable', 'harness_object'):
        require(sha(record[name]) == record[name+'_sha256'], 'changed '+name)
    entry, cwd, root, command = recipe(Path(record['compile_database']),
        Path(record['fixture']), Path(record['harness_object']), Path(record['dependency_file']))
    require(command == record['compile_command'] and str(cwd) == record['compile_cwd']
            and entry == record['native_compile_entry'], 'compile recipe changed')
    require(link_recipe(Path(record['link_file']), Path(record['harness_object']),
                        Path(record['executable'])) == record['link_command'],
            'link changed beyond main-object/output replacement')


def build(args):
    directory, output = args.build.resolve(), args.output.resolve()
    candidate_path = args.candidate_build.resolve()
    require(sha(candidate_path) == PROOF, 'not the qualified coherent candidate proof')
    candidate = json.loads(candidate_path.read_text()); check_candidate(candidate)
    require(candidate['executable_sha256'] == CANDIDATE and
            candidate['baseline_native_executable_sha256'] == BASELINE,
            'wrong candidate/baseline executable')
    source = Path(__file__).with_name('surface_smr.cpp').resolve()
    inp = source.with_suffix('.athinput')
    require(sha(source) == FIXTURE and sha(inp) == INPUT, 'fixture/input bytes changed')
    output.mkdir(parents=True, exist_ok=False)
    obj, executable = output/'surface_smr.o', output/'surface_smr'
    dependency = output/'surface_smr.d'
    database = directory/'compile_commands.json'
    entry, cwd, root, command = recipe(database, source, obj, dependency)
    require(Path(candidate['root']) == root and
            Path(candidate['executable']) == directory/'src/athena',
            'candidate source/object/header tree differs')
    for name in ('surface_smr.cpp', 'surface_smr.athinput'):
        require((source.parent/name).read_bytes() ==
                (root/'tst/z4c_hyperboloidal'/name).read_bytes(),
                'staged fixture differs from coherent source: '+name)
    link_file = directory/'src/CMakeFiles/athena.dir/link.txt'
    link_cwd = link_file.parents[2]
    link = link_recipe(link_file, obj, executable)
    objects = [(link_cwd / token).resolve() for token in link
               if token.endswith(('.o', '.a', '.so', '.dylib')) and token != str(obj)]
    baseline_build = Path(candidate['baseline_build']).resolve()
    require(all(not p.is_relative_to(baseline_build) for p in objects),
            'baseline object linked into coherent helper')
    sources = sorted((root/'src').rglob('*.cpp')) + sorted((root/'src').rglob('*.hpp'))
    sources += [source, inp, Path(__file__).resolve(),
                source.with_name('surface_smr_gpu_contract.md')]
    sources += [source.with_name(name) for name in ('native_amr_gpu.py',
        'native_amr_candidate_cmake.py', 'native_amr_admission_candidate.py')]
    record = dict(schema='native-surface-smr-gpu-build-v1', status='building',
        compile_database=str(database), compile_database_sha256=sha(database),
        native_compile_entry=entry, compile_command=command, compile_cwd=str(cwd),
        link_file=str(link_file), link_file_sha256=sha(link_file), link_command=link,
        link_cwd=str(link_cwd), native_executable=str(directory/'src/athena'),
        native_executable_sha256=sha(directory/'src/athena'),
        baseline_native_executable_sha256=BASELINE, candidate_build=str(candidate_path),
        candidate_build_sha256=sha(candidate_path), fixture=str(source), input=str(inp),
        sources={str(p):sha(p) for p in sources}, objects={str(p):sha(p) for p in objects},
        executable=str(executable), harness_object=str(obj), dependency_file=str(dependency),
        fixture_unchanged=True, input_unchanged=True, gpu_executed=False,
        runtime_scope='coherent candidate objects; only original main object/output replaced')
    proof = output/'provenance.json'
    proof.write_text(json.dumps(record, indent=2)+'\n')
    check_inputs(record)
    makefile = output/'surface_smr.mk'
    makefile.write_text('.PHONY: all\nall:\n\tcd '+shlex.quote(str(cwd))+' && '+
        shlex.join(command)+'\n\tcd '+shlex.quote(str(link_cwd))+' && '+shlex.join(link)+'\n')
    start = time.monotonic()
    with (output/'build.log').open('w') as log:
        result = subprocess.run(['make', '-j64', '-f', str(makefile)], stdout=log,
                                stderr=subprocess.STDOUT)
    record.update(build_seconds=time.monotonic()-start, build_exit_code=result.returncode,
                  status='failed' if result.returncode else 'built')
    proof.write_text(json.dumps(record, indent=2)+'\n')
    check_inputs(record); result.check_returncode()
    text = dependency.read_text().replace('\\\n', ' ')
    dependencies = [(cwd/token).resolve() for token in shlex.split(text.split(':', 1)[1])]
    record.update(dependencies={str(p):sha(p) for p in dependencies},
        executable_sha256=sha(executable), harness_object_sha256=sha(obj),
        runtime_inputs_unchanged=True)
    proof.write_text(json.dumps(record, indent=2)+'\n')
    verify_record(record)
    print(json.dumps({k:record[k] for k in ('status', 'build_exit_code',
        'executable_sha256', 'native_executable_sha256', 'runtime_inputs_unchanged')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    for name in ('build', 'candidate-build', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('verify'); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build': build(args)
    else:
        verify_record(json.loads((args.output/'provenance.json').read_text()))
        print('Verified unchanged surface SMR fixture, coherent candidate and frozen baseline')
