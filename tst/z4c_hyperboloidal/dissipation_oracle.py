#!/usr/bin/env python3
"""Compare all native N12 KO rows/rates with unchanged legacy builder/operator."""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=pathlib.Path, required=True)
parser.add_argument('--legacy', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--ranks', type=int, default=2)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
build, legacy, output = args.build.resolve(), args.legacy.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
source = (legacy/'src/z4c/hyperboloidal/hyperbolic_composite_device.hpp').read_text()
begin = source.index('    auto indices=Kokkos::create_mirror_view(active_);')
end = source.index('\n  }\n\n  template<class Model>', begin)
(output/'ko_legacy_builder.hpp').write_text(source[begin:end]+'\n')
begin = source.index('        using S=CompositeDissipation;Real value=0;')
end = source.index('\n      }', begin)
(output/'ko_legacy_evaluate.hpp').write_text(source[begin:end]+'\n')
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
obj, exe = output/'dissipation_oracle.o', output/'dissipation_oracle'
native_flags = shlex.split(re.search(r'^CXX_FLAGS = (.*)$', flags, re.M).group(1))
compile = ['/usr/bin/c++']+native_flags+['-ffp-contract=off']+includes
compile += ['-I'+str(legacy/'src'), '-I'+str(output), '-c',
            str(root/'tst/z4c_hyperboloidal/dissipation_oracle.cpp'), '-o', str(obj)]
link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(obj) if x=='CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
assert str(obj) in link
link[link.index('-o')+1] = str(exe)
makefile = output/'oracle.mk'
makefile.write_text('all:\n\t'+shlex.join(compile)+'\n\t'+shlex.join(link)+'\n')
with (output/'build.log').open('w') as log:
    subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src',
                   stdout=log, stderr=subprocess.STDOUT, check=True)
with (output/'run.log').open('w') as log:
    result = subprocess.run(['mpiexec', '-n', str(args.ranks), str(exe),
                             str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput')],
                            cwd=output, stdout=log, stderr=subprocess.STDOUT)
text = (output/'run.log').read_text()
report = {'passed': result.returncode == 0, 'local_mpi_ranks': args.ranks,
          'scope': 'all N12 KO line coefficients, volumes, donors, and 31-field rates',
          'input': 'deterministic nonsmooth physical values in owned and required halo nodes',
          'halo_interpolation_tested': False,
          'native_executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
          'legacy_source_sha256': hashlib.sha256(source.encode()).hexdigest()}
match = re.search(r'PASS native KO: terms=(\d+) rates=(\d+) bitwise=(\d+) max_scaled_error=(\S+)', text)
if match:
    report.update(terms=int(match[1]), rates=int(match[2]), bitwise_rates=int(match[3]),
                  max_scaled_error=float(match[4]))
(output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
print(text)
result.check_returncode()
