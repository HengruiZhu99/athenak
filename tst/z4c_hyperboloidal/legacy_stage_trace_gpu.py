#!/usr/bin/env python3
"""Build a test-only observer of exact pinned predecessor Evolution using its flags.

Only the entry translation unit/object/output is replaced. No original source,
object, library or executable is modified. No jobs or evolution are launched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check(p):
    for group in ('sources', 'objects', 'dependencies'):
        for path, expected in p.get(group, {}).items():
            require(sha(path) == expected, 'changed ' + group + ': ' + path)
    require(sha(p['original_executable']) == p['original_executable_sha256'],
            'changed immutable original executable')


def build(args):
    output, legacy, cwd = args.output.resolve(), args.legacy.resolve(), args.link_cwd.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_suffix('.cpp').resolve()
    require(source.is_file(), 'missing observer C++')
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    require(len(manifest['commands']) == 1, 'ambiguous original command record')
    entry = manifest['commands'][0]
    original_compile, original_link = entry['compile'], entry['link']
    entry_source = legacy / 'tst/hyperbolic_shift_3d/endpoint_policy/run.cpp'
    require(original_compile.count(str(entry_source)) == 1, 'original source path differs')
    original_binary, original_hash = next(iter(manifest['binaries'].items()))
    require(original_hash == '494ec8e38fb34806eb3981fb19bdb5e3bc75f46615b05a21d8314a0db6583753',
            'wrong original binary admission')
    require(sha(original_binary) == original_hash, 'original executable changed')
    # Match every surviving source in the archived build record before using it.
    sources = {}
    for relative, expected in manifest['sources'].items():
        path = legacy / relative
        require(sha(path) == expected, 'pinned source differs: ' + relative)
        sources[str(path)] = expected
    sources.update({str(p): sha(p) for p in (source, Path(__file__).resolve(), manifest_path)})
    contract = source.with_name('legacy_stage_trace_gpu_contract.md')
    if contract.is_file():
        sources[str(contract)] = sha(contract)
    obj = output / 'legacy_stage_trace_gpu.o'
    executable = output / 'legacy_stage_trace_gpu'
    depfile = output / 'legacy_stage_trace_gpu.d'
    command = list(original_compile)
    require(command.count('-o') == 1 and '-c' in command and '-ffp-contract=off' in command,
            'unrecognized original compile command')
    old_obj = command[command.index('-o') + 1]
    command[command.index('-o') + 1] = str(obj)
    command[command.index(str(entry_source))] = str(source)
    command.extend(['-I' + str(legacy), '-MMD', '-MF', str(depfile), '-MT', str(obj)])
    link = list(original_link)
    require(link.count(old_obj) == 1 and link.count('-o') == 1, 'unrecognized original link')
    link[link.index(old_obj)] = str(obj)
    link[link.index('-o') + 1] = str(executable)
    objects = {}
    for token in link:
        if token.endswith(('.o', '.a', '.so', '.dylib')) and token != str(obj):
            p = (Path(token) if token.startswith('/') else cwd / token).resolve()
            objects[str(p)] = sha(p)
    original_library = cwd / 'libcurrent_spherical.so'
    require(sha(original_library) == 'ad4c6321b18a44ee89e42cdd4307e760fe59506ec0293a655e97416811acff2d',
            'pinned original spherical library changed')
    p = dict(schema='original-stage-accessor-build-v1', status='building',
             original_executable=original_binary, original_executable_sha256=original_hash,
             original_compile_command=original_compile, original_link_command=original_link,
             compile_command=command, link_command=link, working_directory=str(cwd),
             sources=sources, objects=objects, executable=str(executable), harness_object=str(obj))
    record = output / 'provenance.json'
    record.write_text(json.dumps(p, indent=2) + '\n')
    check(p)
    mk = output / 'observer.mk'
    mk.write_text('.PHONY: all\nall:\n\tcd ' + shlex.quote(str(cwd)) + ' && ' + shlex.join(command)
                  + '\n\tcd ' + shlex.quote(str(cwd)) + ' && ' + shlex.join(link) + '\n')
    started = time.monotonic()
    with (output / 'build.log').open('w') as log:
        result = subprocess.run(['make', '-j64', '-f', str(mk)], stdout=log, stderr=subprocess.STDOUT)
    p.update(build_seconds=time.monotonic()-started, build_exit_code=result.returncode,
             status='failed' if result.returncode else 'built')
    record.write_text(json.dumps(p, indent=2) + '\n')
    check(p)
    result.check_returncode()
    deptext = depfile.read_text().replace('\\\n', ' ')
    dependencies = [(Path(t) if t.startswith('/') else cwd / t).resolve()
                    for t in shlex.split(deptext.split(':', 1)[1])]
    p['dependencies'] = {str(path): sha(path) for path in dependencies}
    p.update(executable_sha256=sha(executable), harness_object_sha256=sha(obj),
             original_runtime_inputs_unchanged=True)
    record.write_text(json.dumps(p, indent=2) + '\n')
    print(json.dumps({k: p[k] for k in ('status', 'executable_sha256',
                                       'original_runtime_inputs_unchanged')}, indent=2))


def verify(args):
    p = json.loads((args.output.resolve() / 'provenance.json').read_text())
    require(p['schema'] == 'original-stage-accessor-build-v1' and p['status'] == 'built',
            'observer build failed or unknown')
    check(p)
    for k in ('executable', 'harness_object'):
        require(sha(p[k]) == p[k + '_sha256'], 'changed observer ' + k)
    print('Verified accessor and exact pinned predecessor source/object/library/executable inputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--manifest', type=Path, required=True)
    b.add_argument('--legacy', type=Path, required=True)
    b.add_argument('--link-cwd', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    v = sub.add_parser('verify');v.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'build': build, 'verify': verify}[args.command](args)
