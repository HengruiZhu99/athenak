#!/usr/bin/env python3
"""Build/verify a test-only boundary route metadata helper against unchanged native objects.

Compiler, device options and link inputs come from the actual AthenaK CMake
build. This tool does not rebuild the runtime, submit jobs or launch evolution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_inputs(provenance):
    for group in ('sources', 'objects', 'dependencies'):
        for path, expected in provenance.get(group, {}).items():
            require(sha(path) == expected, 'changed ' + group + ': ' + path)
    for key in ('compile_database', 'link_file', 'native_executable'):
        require(sha(provenance[key]) == provenance[key + '_sha256'], 'changed ' + key)


def build(args):
    directory, output = args.build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_suffix('.cpp').resolve()
    require(source.is_file(), 'missing native boundary route audit source')
    database = (args.compile_database or directory / 'compile_commands.json').resolve()
    entries = json.loads(database.read_text())
    selected = [entry for entry in entries if
                Path(entry['file']).as_posix().endswith('z4c_hyperboloidal/diagnostics.cpp')]
    require(len(selected) == 1, 'need exactly one native diagnostics.cpp compile entry')
    entry = selected[0]
    compile_cwd = Path(entry['directory']).resolve()
    native_source = Path(entry['file'])
    if not native_source.is_absolute():
        native_source = compile_cwd / native_source
    native_source = native_source.resolve()
    root = native_source.parents[2]
    original = entry.get('arguments') or shlex.split(entry['command'])
    obj = output / 'native_bvals_route_audit.o'
    executable = output / 'native_bvals_route_audit'
    dependency_file = output / 'native_bvals_route_audit.d'
    command, source_count, output_count, index = [], 0, 0, 0
    while index < len(original):
        token = original[index]
        if token == '-o':
            command.extend(['-o', str(obj)])
            output_count += 1
            index += 2
            continue
        if token in ('-MF', '-MT', '-MQ'):
            index += 2
            continue
        if token in ('-MD', '-MMD', '-MP'):
            index += 1
            continue
        path = Path(token) if token.startswith('/') else compile_cwd / token
        if not token.startswith('-') and path.resolve() == native_source:
            command.append(str(source))
            source_count += 1
        else:
            command.append(token)
        index += 1
    require(source_count == output_count == 1 and '-c' in command,
            'unrecognized native compile command; refusing guessed flags')
    require('-ffp-contract=off' in command, 'native contraction policy is absent')
    command.extend(['-MMD', '-MF', str(dependency_file), '-MT', str(obj)])
    link_file = directory / 'src/CMakeFiles/athena.dir/link.txt'
    link = shlex.split(link_file.read_text())
    main = [i for i, token in enumerate(link) if token.endswith('athena.dir/main.cpp.o')]
    require(len(main) == 1 and link.count('-o') == 1, 'unrecognized native link command')
    link[main[0]] = str(obj)
    link[link.index('-o') + 1] = str(executable)
    link_cwd = link_file.parents[2]
    objects = [(Path(token) if Path(token).is_absolute() else link_cwd / token).resolve()
               for token in link if token.endswith(('.o', '.a', '.so', '.dylib'))
               and token != str(obj)]
    sources = sorted(list((root / 'src').rglob('*.cpp')) + list((root / 'src').rglob('*.hpp')))
    sources.extend([source, Path(__file__).resolve()])
    contract = source.with_name('native_bvals_route_audit_contract.md')
    if contract.exists():
        sources.append(contract)
    native_executable = directory / 'src/athena'
    provenance = dict(
        schema='native-bvals-route-build-v1', status='building',
        compile_database=str(database), compile_database_sha256=sha(database),
        native_compile_entry=entry, compile_command=command, compile_cwd=str(compile_cwd),
        link_file=str(link_file), link_file_sha256=sha(link_file),
        link_command=link, link_cwd=str(link_cwd),
        native_executable=str(native_executable), native_executable_sha256=sha(native_executable),
        sources={str(path): sha(path) for path in sources},
        objects={str(path): sha(path) for path in objects},
        executable=str(executable), harness_object=str(obj))
    record = output / 'provenance.json'
    record.write_text(json.dumps(provenance, indent=2) + '\n')
    makefile = output / 'audit.mk'
    makefile.write_text('.PHONY: all\nall:\n\tcd ' + shlex.quote(str(compile_cwd))
                       + ' && ' + shlex.join(command) + '\n\tcd '
                       + shlex.quote(str(link_cwd)) + ' && ' + shlex.join(link) + '\n')
    start = time.monotonic()
    with (output / 'build.log').open('w') as log:
        result = subprocess.run(['make', '-j64', '-f', str(makefile)],
                                stdout=log, stderr=subprocess.STDOUT)
    provenance['build_seconds'] = time.monotonic() - start
    provenance['build_exit_code'] = result.returncode
    provenance['status'] = 'failed' if result.returncode else 'built'
    record.write_text(json.dumps(provenance, indent=2) + '\n')
    check_inputs(provenance)
    result.check_returncode()
    dependency_text = dependency_file.read_text().replace('\\\n', ' ')
    dependencies = [Path(token) if Path(token).is_absolute() else compile_cwd / token
                    for token in shlex.split(dependency_text.split(':', 1)[1])]
    provenance['dependencies'] = {str(path.resolve()): sha(path) for path in dependencies}
    provenance['executable_sha256'] = sha(executable)
    provenance['harness_object_sha256'] = sha(obj)
    provenance['runtime_inputs_unchanged'] = True
    record.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({key: provenance[key] for key in
                      ('status', 'executable_sha256', 'native_executable_sha256',
                       'runtime_inputs_unchanged')}, indent=2))


def verify(args):
    provenance = json.loads((args.output.resolve() / 'provenance.json').read_text())
    require(provenance['schema'] == 'native-bvals-route-build-v1'
            and provenance['status'] == 'built', 'route audit build did not pass')
    check_inputs(provenance)
    for key in ('executable', 'harness_object'):
        require(sha(provenance[key]) == provenance[key + '_sha256'], 'changed ' + key)
    print('Verified boundary route audit and unchanged native source/object/executable inputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build_parser = commands.add_parser('build')
    build_parser.add_argument('--build', type=Path, required=True)
    build_parser.add_argument('--output', type=Path, required=True)
    build_parser.add_argument('--compile-database', type=Path)
    verify_parser = commands.add_parser('verify')
    verify_parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    {'build': build, 'verify': verify}[arguments.command](arguments)
