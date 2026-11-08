#!/usr/bin/env python3
"""Build/verify complete-cycle timing against unchanged native numerical objects.

The driver-only source overlay inserts host timing callbacks at whole-cycle
boundaries. It is invertible byte-for-byte and preserves the driver FP flags.

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
    for key in ('compile_database', 'link_file', 'native_executable', 'original_driver_object'):
        require(sha(provenance[key]) == provenance[key + '_sha256'], 'changed ' + key)


def transform(original, header):
    """Insert callbacks without changing any original byte or instruction text."""
    additions = [
        ('#include "utils/utils.hpp"\n', '#include '+json.dumps(str(header))+'\n'),
        ('void Driver::Initialize(Mesh *pmesh, ParameterInput *pin, Outputs *pout, bool res_flag) {\n',
         '  native_cycle_timing::BeginInitialize(pmesh, this, res_flag);\n'),
        ('void Driver::Execute(Mesh *pmesh, ParameterInput *pin, Outputs *pout, bool wdflag) {\n',
         '  native_cycle_timing::BeginExecute(wdflag);\n'),
        ('           (elapsed_time < wall_time)) {\n',
         '      native_cycle_timing::BeforeCycle();\n'),
        ('      pmesh->NewTimeStep(tlim);\n      RefreshSTSCycleState(pmesh);\n',
         '      native_cycle_timing::AfterCycle();\n'),
        ('void Driver::Finalize(Mesh *pmesh, ParameterInput *pin, Outputs *pout) {\n',
         '  native_cycle_timing::BeginFinalize();\n'),
        ('  if (pm->ncycle % ndiag == 0) {\n',
         '    native_cycle_timing::Console();\n')]
    transformed=original; edits=[]
    def insert(anchor, addition, before=False):
        nonlocal transformed
        a,b=anchor.encode(),addition.encode()
        require(transformed.count(a)==1 and b not in transformed,
                'driver insertion anchor absent/ambiguous or timing already installed')
        replacement=b+a if before else a+b
        transformed=transformed.replace(a,replacement,1)
        edits.append(dict(anchor=anchor,addition=addition,before=before))
    for anchor,addition in additions: insert(anchor,addition)
    insert('          out->LoadOutputData(pmesh);\n',
           '          native_cycle_timing::Output();\n',True)
    # Function endings are selected by their following unique documentation.
    for function,call in [('Execute','EndInitialize(pout)'),('Finalize','EndExecute()'),
                          ('OutputCycleDiagnostics','EndFinalize()')]:
        marker='//! \\fn Driver::'+function+'()'
        pos=transformed.index(marker.encode())
        end=transformed.rfind(b'  return;\n}',0,pos)
        require(end>=0,'driver function end not found: '+function)
        anchor=transformed[end:pos+len(marker)].decode()
        insert(anchor,'  native_cycle_timing::'+call+';\n',True)
    return transformed,edits


def reverse_transform(transformed, edits):
    for edit in reversed(edits):
        anchor,addition=edit['anchor'].encode(),edit['addition'].encode()
        combined=addition+anchor if edit.get('before') else anchor+addition
        require(transformed.count(combined)==1,'driver overlay insertion changed/repeated')
        transformed=transformed.replace(combined,anchor,1)
    return transformed


def check_overlay(provenance):
    overlay = provenance['overlay']
    original = Path(overlay['original_source']).read_bytes()
    pristine = Path(overlay['original_copy']).read_bytes()
    transformed = Path(overlay['transformed_source']).read_bytes()
    require(original == pristine and sha(overlay['original_source']) == overlay['original_sha256'],
            'original driver changed or pristine copy differs')
    require(sha(overlay['transformed_source']) == overlay['transformed_sha256'] and
            reverse_transform(transformed,overlay['additions']) == original,
            'timed driver is not the exact insertion overlay')


def build(args):
    directory, output = args.build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    header = Path(__file__).with_suffix('.hpp').resolve()
    require(header.is_file(), 'missing native cycle timing header')
    database = (args.compile_database or directory / 'compile_commands.json').resolve()
    entries = json.loads(database.read_text())
    selected = [entry for entry in entries if
                Path(entry['file']).as_posix().endswith('/src/driver/driver.cpp')]
    require(len(selected) == 1, 'need exactly one original src/driver/driver.cpp compile entry')
    entry = selected[0]
    compile_cwd = Path(entry['directory']).resolve()
    native_source = Path(entry['file'])
    if not native_source.is_absolute():
        native_source = compile_cwd / native_source
    native_source = native_source.resolve()
    root = native_source.parents[2]
    original_bytes = native_source.read_bytes()
    source = output/'driver-timed.cpp'
    pristine = output/'driver-original.cpp'
    overlay, edits = transform(original_bytes, header)
    pristine.write_bytes(original_bytes)
    source.write_bytes(overlay)
    require(reverse_transform(overlay, edits) == original_bytes,
            'driver overlay is not exactly invertible')
    original = entry.get('arguments') or shlex.split(entry['command'])
    obj = output / 'driver-timed.o'
    executable = output / 'athena-cycle-timing'
    dependency_file = output / 'driver-timed.d'
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
    # Do not add/change FP flags: this is the original driver.cpp compilation.
    # Relocating the TU removes its implicit quoted-include directory. Restore
    # that original search location (e.g. "driver.hpp"), without changing FP.
    command.extend(['-iquote', str(native_source.parent), '-MMD', '-MF',
                    str(dependency_file), '-MT', str(obj)])
    link_file = directory / 'src/CMakeFiles/athena.dir/link.txt'
    link = shlex.split(link_file.read_text())
    driver = [i for i, token in enumerate(link) if token.endswith('athena.dir/driver/driver.cpp.o')]
    require(len(driver) == 1 and link.count('-o') == 1, 'unrecognized native link command')
    link_cwd = link_file.parents[2]
    original_driver = Path(link[driver[0]])
    if not original_driver.is_absolute():
        original_driver = link_cwd/original_driver
    original_driver = original_driver.resolve()
    link[driver[0]] = str(obj)
    link[link.index('-o') + 1] = str(executable)
    objects = [(Path(token) if Path(token).is_absolute() else link_cwd / token).resolve()
               for token in link if token.endswith(('.o', '.a', '.so', '.dylib'))
               and token != str(obj)]
    sources = sorted(list((root / 'src').rglob('*.cpp')) + list((root / 'src').rglob('*.hpp')))
    sources.extend([header, Path(__file__).resolve()])
    contract = header.with_name('native_cycle_timing_contract.md')
    if contract.exists():
        sources.append(contract)
    native_executable = directory / 'src/athena'
    provenance = dict(
        schema='native-cycle-timing-build-v1', status='building',
        compile_database=str(database), compile_database_sha256=sha(database),
        native_compile_entry=entry, compile_command=command, compile_cwd=str(compile_cwd),
        original_compile_command=original, original_driver_fp_flags_preserved=True,
        restored_original_quote_include_directory=str(native_source.parent),
        overlay=dict(original_source=str(native_source), original_copy=str(pristine),
            original_sha256=sha(native_source), transformed_source=str(source),
            transformed_sha256=sha(source), additions=edits, inverse_byte_exact=True),
        link_file=str(link_file), link_file_sha256=sha(link_file),
        link_command=link, link_cwd=str(link_cwd),
        native_executable=str(native_executable), native_executable_sha256=sha(native_executable),
        original_driver_object=str(original_driver), original_driver_object_sha256=sha(original_driver),
        sources={str(path): sha(path) for path in sources},
        objects={str(path): sha(path) for path in objects},
        executable=str(executable), harness_object=str(obj))
    record = output / 'provenance.json'
    record.write_text(json.dumps(provenance, indent=2) + '\n')
    makefile = output / 'timing.mk'
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
    check_overlay(provenance)
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
    require(provenance['schema'] == 'native-cycle-timing-build-v1'
            and provenance['status'] == 'built', 'timing build did not pass')
    check_inputs(provenance)
    check_overlay(provenance)
    for key in ('executable', 'harness_object'):
        require(sha(provenance[key]) == provenance[key + '_sha256'], 'changed ' + key)
    print('Verified invertible original-driver timing and unchanged native source/object/executable inputs')


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
