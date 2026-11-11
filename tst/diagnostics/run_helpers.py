"""Compile focused tests against production headers, using the configured build flags."""
import argparse, pathlib, shlex, subprocess
p=argparse.ArgumentParser();p.add_argument('build',type=pathlib.Path);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[2]
flags=(a.build/'src/CMakeFiles/athena.dir/flags.make').read_text().splitlines()
inc=next(x.split(' = ',1)[1] for x in flags if x.startswith('CXX_INCLUDES ='))
exe=a.build.resolve()/'diagnostic-helpers'
libs=[a.build.resolve()/'kokkos/core/src/libkokkoscore.a',a.build.resolve()/'kokkos/containers/src/libkokkoscontainers.a']
subprocess.run(['c++','-std=c++17',*shlex.split(inc),str(root/'tst/diagnostics/helpers.cpp'),'-o',str(exe),*map(str,libs)],check=True)
subprocess.run([str(exe)],check=True)
