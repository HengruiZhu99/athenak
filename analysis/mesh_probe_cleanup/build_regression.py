#!/usr/bin/env python3
"""Relink a separate executable and constructor regression; never modify old build.

Requires the original CMake Unix Makefiles build, including object files.
Only meshblock_pack.cpp is recompiled. All other objects and libraries are reused
with their original flags, hashed, and left untouched. Python 3.6 compatible.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--fixed-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--test-source', type=Path,
                   default=Path(__file__).with_name('test_constructor.cpp'))
    a = p.parse_args()
    build = a.build.resolve(); out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cwd = build / 'src'; cm = cwd / 'CMakeFiles/athena.dir'
    cache = (build / 'CMakeCache.txt').read_text()
    link = shlex.split((cm / 'link.txt').read_text())
    compiler = link[0]
    home = Path(re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$', cache, re.M).group(1))
    flags = dict(re.findall(r'^(CXX_\w+) = (.*)$', (cm / 'flags.make').read_text(), re.M))
    compile_args = [compiler]
    for k in ['CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS']:
        compile_args += shlex.split(flags[k])
    compile_args += ['-I' + str(home / 'src/mesh')]
    link = [x for x in link if not x.startswith('-Wl,--dependency-file=')]
    mesh_obj = 'CMakeFiles/athena.dir/mesh/meshblock_pack.cpp.o'
    main_obj = 'CMakeFiles/athena.dir/main.cpp.o'
    assert link.count(mesh_obj) == 1 and link.count(main_obj) == 1
    old_binary = cwd / 'athena'
    report = {'old_executable': str(old_binary), 'old_sha256': digest(old_binary),
              'fixed_source_sha256': digest(a.fixed_source), 'commands': [],
              'baseline_objects_and_libraries': {}}
    for x in link:
        if x.endswith(('.o', '.a', '.dylib', '.so')):
            f = (cwd / x).resolve()
            report['baseline_objects_and_libraries'][str(f)] = digest(f)
    def run(cmd, name, expected=0):
        report['commands'].append(cmd)
        with (out / (name + '.log')).open('w') as f:
            r = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        print(name, r.returncode, flush=True)
        if r.returncode != expected:
            raise RuntimeError(name + ': inspect ' + str(out / (name + '.log')))
    def linked(obj, output, test=False):
        cmd = [str(obj) if x == mesh_obj else x for x in link]
        if test: cmd = [str(out / 'test.o') if x == main_obj else x for x in cmd]
        cmd[cmd.index('-o') + 1] = str(output)
        return cmd
    run(compile_args + ['-c', str(a.test_source.resolve()), '-o', str(out / 'test.o')], 'compile-test')
    run(linked(cwd / mesh_obj, out / 'test-old', True), 'link-test-old')
    run([str(out / 'test-old')], 'test-old', expected=1)
    run(compile_args + ['-c', str(a.fixed_source.resolve()), '-o', str(out / 'meshblock_pack.cpp.o')], 'compile-fixed')
    run(linked(out / 'meshblock_pack.cpp.o', out / 'test-fixed', True), 'link-test-fixed')
    run([str(out / 'test-fixed')], 'test-fixed')
    run(linked(out / 'meshblock_pack.cpp.o', out / 'athena'), 'link-athena')
    assert digest(old_binary) == report['old_sha256'], 'Original executable changed'
    for path, expected in report['baseline_objects_and_libraries'].items():
        assert digest(Path(path)) == expected, 'Original object/library changed: ' + path
    report.update(new_executable=str(out / 'athena'), new_sha256=digest(out / 'athena'),
                  old_test_expected_failure=True, fixed_test_passed=True,
                  original_objects_and_executable_unchanged=True)
    (out / 'build-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS', report['new_sha256'], flush=True)

if __name__ == '__main__':
    main()
