#!/usr/bin/env python3
"""Link a full native module regression against an already built AthenaK executable.

This reuses its native object files and link line; no second driver is implemented.
The independent legacy initializer is included only by the test translation unit.
"""
import argparse
import pathlib
import re
import shlex
import subprocess
parser=argparse.ArgumentParser()
parser.add_argument('--build',type=pathlib.Path,required=True)
parser.add_argument('--legacy',type=pathlib.Path,required=True)
parser.add_argument('--output',type=pathlib.Path,required=True)
parser.add_argument('--ranks',type=int,default=2)
args=parser.parse_args()
root=pathlib.Path(__file__).resolve().parents[2]
build,legacy,output=args.build.resolve(),args.legacy.resolve(),args.output.resolve()
output.mkdir(parents=True,exist_ok=True)
flags=(build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M).group(1))
options=['BROAD_SMOOTH_A','CONFIGURED_FORMULATION','DIRECT_OUTER','INNER_TELEGRAPHER',
         'NORMALIZED_INNER_SHIFT','RESTORE_INNER_DAMPING','SHORT_INNER_GAUGE',
         'SMOOTH_GAUGE_BLEND','SPHERICAL_POLICY']
obj=output/'module_oracle.o';exe=output/'module_oracle'
compile=['/usr/bin/c++','-std=c++17','-O2','-ffp-contract=off']+includes
compile += ['-DHYPERBOLIC_'+x+'=1' for x in options]
compile += ['-I'+str(legacy/'src'),'-I'+str(legacy),'-c',
            str(root/'tst/z4c_hyperboloidal/module_oracle.cpp'),'-o',str(obj)]
link=shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link=[str(obj) if x=='CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
assert str(obj) in link
link[link.index('-o')+1]=str(exe)
makefile=output/'module_oracle.mk'
makefile.write_text('all:\n\t'+shlex.join(compile)+'\n\t'+shlex.join(link)+'\n')
with (output/'build.log').open('w') as log:
 subprocess.run(['make','-j64','-f',str(makefile)],cwd=build/'src',stdout=log,
                stderr=subprocess.STDOUT,check=True)
command=['mpiexec','-n',str(args.ranks),str(exe),
         str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput')]
with (output/'run.log').open('w') as log:
 subprocess.run(command,cwd=output,stdout=log,stderr=subprocess.STDOUT,check=True)
print((output/'run.log').read_text())
