#!/usr/bin/env python3
"""Source-locked, single-process CPU probes for the 3D takeover; no adoption."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import subprocess
import tarfile
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=int, default=180, choices=range(1, 301))
p.add_argument('arguments', nargs=argparse.REMAINDER)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
source = a.source.resolve()
source.relative_to(root)
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
build = root / 'build-shift-3d'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
m = {'scope': 'Standalone bounded CPU diagnostic; not production adoption or scientific validation',
     'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
     'runs': {}}
def save():
    (out / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
def run(name, cmd, limit=60):
    row = {'argv': list(map(str, cmd)), 'maximum_seconds': limit}
    m['runs'][name] = row
    save()
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    with (out / (name+'.stdout')).open('w') as so, (out / (name+'.stderr')).open('w') as se:
        child = subprocess.Popen(row['argv'], stdout=so, stderr=se, cwd=out,
                                 env=env, start_new_session=True)
        row['owned_process_group'] = child.pid
        save()
        try:
            row['exit_code'] = child.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            assert os.getpgid(child.pid) == child.pid
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            row['exit_code'] = 124
    row['seconds'] = time.monotonic() - start
    save()
    print(name, row['exit_code'], round(row['seconds'], 3), flush=True)
    return row['exit_code']
flags = (build/'CMakeFiles/elliptic_shift_3d_composite_regrid_tests.dir/flags.make').read_text().splitlines()
args = []
for key in ('CXX_DEFINES =', 'CXX_INCLUDES ='):
    args += shlex.split(next(x.split('=', 1)[1] for x in flags if x.startswith(key)))
if run('dependencies', ['/usr/bin/c++', '-std=c++17', *args, '-M', source]):
    raise SystemExit(1)
def dependencies(name):
    return [Path(s).resolve() for s in shlex.split((out/name).read_text().replace('\\\n', ' '))[1:]]
deps = dependencies('dependencies.stdout')
frozen = out / 'source'
project = {str(f.relative_to(root)): sha(f) for f in deps + [Path(__file__).resolve()]
           if f.is_file() and f.is_relative_to(root)}
for relative, digest in project.items():
    target = frozen / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((root/relative).read_bytes())
    assert sha(target) == digest
m['project_source_hashes'] = project
fargs = [v.replace(str(root), str(frozen)) for v in args]
fsource = frozen / source.relative_to(root)
if run('frozen-dependencies', ['/usr/bin/c++', '-std=c++17', *fargs, '-M', fsource]):
    raise SystemExit(1)
actual = dependencies('frozen-dependencies.stdout')
m['live_project_header_leaks'] = [str(f) for f in actual if f.is_relative_to(root) and not f.is_relative_to(frozen)]
assert not m['live_project_header_leaks']
libraries = [build/f'kokkos/{k}/src/libkokkos{k}.a' for k in ('containers', 'algorithms', 'core', 'simd')]
m['libraries'] = {str(f): sha(f) for f in libraries}
with tarfile.open(out/'compiled-source.tar.gz', 'w:gz') as archive:
    for relative in sorted(project):
        archive.add(frozen/relative, arcname=relative)
m['source_archive_sha256'] = sha(out/'compiled-source.tar.gz')
(out/'.gitignore').write_text('/probe\n/source/\n/*.csv\n/*.bin\n')
code = run('build', ['/usr/bin/c++', '-O3', '-std=c++17', *fargs, fsource, '-o', out/'probe', *libraries])
if not code:
    m['binary_sha256'] = sha(out/'probe')
    argv = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    code = run('probe', [out/'probe', *argv], a.seconds)
m['project_sources_unchanged'] = all(sha(root/f) == digest for f, digest in project.items())
m['libraries_unchanged'] = all(sha(Path(f)) == digest for f, digest in m['libraries'].items())
for f in out.glob('*.csv'):
    f.with_suffix(f.suffix+'.gz').write_bytes(gzip.compress(f.read_bytes(), mtime=0))
save()
assert m['project_sources_unchanged'] and m['libraries_unchanged']
raise SystemExit(code)
