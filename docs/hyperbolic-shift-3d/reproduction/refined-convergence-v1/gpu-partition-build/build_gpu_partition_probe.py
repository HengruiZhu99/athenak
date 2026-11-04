"""Compile the locally qualified ownership probe against frozen Aurora libraries.

Run inside a new package directory with oneAPI loaded. This never changes or
rebuilds the production source, object files, libraries, or executable.
"""
from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
out = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'launcher-v7'))
from storage import capacity

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(name, value):
    p = out / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)

with (out / 'started.json').open('x') as f:
    json.dump(dict(pid=os.getpid(), started=time.time()), f)
result = dict(status='building', started=time.time())
try:
    # Reserve the active physics job's output/atomic budget in addition to
    # this small compile, including when its output is not yet on disk.
    write('storage-before.json', capacity(root, planned=500_000_000,
                                          reserve=3_000_000_000))
    package = json.loads((out / 'source.json').read_text())
    assert digest(root / 'source-manifest.json') == package['base_manifest_sha256']
    manifest = json.loads((root / 'source-manifest.json').read_text())
    assert manifest['source_commit'] == package['base_source_commit']
    for name, record in manifest['files'].items():
        assert digest(root / 'source' / name) == record['sha256'], name
    for name, record in package['files'].items():
        assert digest(out / name) == record['sha256'], name
        assert (out / name).stat().st_size == record['bytes'], name
    science = json.loads((root / 'science.json').read_text())
    assert digest(root / science['binary']) == science['binary_sha256']
    target = root / 'build/tst/hyperbolic_shift_3d'
    cmake = target / 'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir'
    assert digest(cmake / 'flags.make') == science['flags_sha256']
    assert digest(root / 'build/CMakeCache.txt') == science['build_cache_sha256']
    flags = {}
    for line in (cmake / 'flags.make').read_text().splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1)
            flags[key] = shlex.split(value)
    link = shlex.split((cmake / 'link.txt').read_text())
    compiler = link[0]
    command = ([compiler] + flags['CXX_DEFINES'] +
               ['-DHYPERBOLIC_MEASURED_PARTITION=1'] +
               flags['CXX_INCLUDES'] + flags['CXX_FLAGS'] +
               ['-I' + str(root / 'source/tst/hyperbolic_shift_3d'),
                '-c', str(out / 'amr_evolution.cpp'), '-o', str(out / 'probe.o')])
    old_object = 'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/amr_evolution.cpp.o'
    assert link.count(old_object) == 1
    link[link.index(old_object)] = str(out / 'probe.o')
    link[link.index('-o') + 1] = str(out / 'probe')
    libraries = {str((target / x).resolve()): digest(target / x)
                 for x in link if x.endswith('.a')}
    write('build-commands.json', dict(compile=command, link=link,
          cwd=str(target), frozen_libraries=libraries,
          production_binary_sha256=science['binary_sha256'],
          source_manifest_sha256=package['base_manifest_sha256']))
    with (out / 'compiler.txt').open('w') as f:
        subprocess.run([compiler, '--version'], stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    with (out / 'build.log').open('w') as f:
        for cmd in (command, link):
            subprocess.run(cmd, cwd=target, stdout=f,
                           stderr=subprocess.STDOUT, check=True)
    for name, expected in libraries.items():
        assert digest(name) == expected
    assert digest(root / science['binary']) == science['binary_sha256']
    result.update(status='completed', binary_sha256=digest(out / 'probe'),
                  binary_bytes=(out / 'probe').stat().st_size,
                  production_unchanged=True,
                  qualification='Ownership-only static SMR analysis binary; no GPU run or speedup claim.')
    write('storage-after.json', capacity(root, reserve=3_000_000_000))
except Exception as e:
    result.update(status='failed', error=repr(e))
    raise
finally:
    result['finished'] = time.time()
    write('build-result.json', result)
