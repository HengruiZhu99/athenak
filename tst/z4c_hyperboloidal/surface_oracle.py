#!/usr/bin/env python3
"""Link a native scri regression against an already built AthenaK executable.

This reuses its native object files and link line; no second driver is implemented.
The independent legacy initializer is included only by the test translation unit.
"""
import argparse
import json
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
obj=output/'surface_oracle.o';exe=output/'surface_oracle'
compile=['/usr/bin/c++','-std=c++17','-O2','-ffp-contract=off']+includes
compile += ['-DHYPERBOLIC_'+x+'=1' for x in options]
compile += ['-I'+str(legacy/'src'),'-I'+str(legacy),'-c',
            str(root/'tst/z4c_hyperboloidal/surface_oracle.cpp'),'-o',str(obj)]
link=shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link=[str(obj) if x=='CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
assert str(obj) in link
link[link.index('-o')+1]=str(exe)
for name in ['surface_exchange','mapped_halo']:
 assert any(x.endswith('/'+name+'.cpp.o') for x in link), 'Build native '+name+' before the oracle'
makefile=output/'surface_oracle.mk'
makefile.write_text('all:\n\t'+'\n\t'.join(shlex.join(c) for c in [compile,link])+'\n')
with (output/'build.log').open('w') as log:
 subprocess.run(['make','-j64','-f',str(makefile)],cwd=build/'src',stdout=log,
                stderr=subprocess.STDOUT,check=True)
command=['mpiexec','-n',str(args.ranks),str(exe),
         str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput')]
with (output/'run.log').open('w') as log:
 result=subprocess.run(command,cwd=output,stdout=log,stderr=subprocess.STDOUT)
text=(output/'run.log').read_text()
metrics={}
for m in re.finditer(r'^(\w+) comparisons=(\d+) max_scaled=([^ ]+) max_absolute=([^ ]+) failures=(\d+)$',text,re.M):
 metrics[m.group(1)]={'comparisons':int(m.group(2)),'max_scaled_error':float(m.group(3)),
  'max_absolute_error':float(m.group(4)),'failed_values':int(m.group(5))}
status={'passed':result.returncode==0,'local_mpi_ranks':args.ranks,'scenarios':2,
 'gate':'abs(delta)<=1e-12+5e-10*abs(reference)','metrics':metrics,
 'native_build':str(build),'legacy_readonly_source':str(legacy)}
m=re.search(r'normal_compatibility=([^ ]+) zero_face_ranks=(\d+)',text)
if m:status.update(normal_compatibility=float(m.group(1)),zero_face_ranks=int(m.group(2)))
cache=re.search(r'endpoint_cache comparisons=(\d+) poison_refresh=passed default_direct=passed bitwise=passed',text)
if cache:status['endpoint_cache']={'comparisons':int(cache[1]),'poison_refresh':True,
 'default_direct':True,'bitwise':True}
layout=re.search(r'cache_layout rows=(\d+) fields=(\d+) jet_bytes=(\d+) bytes=(\d+)',text)
if layout:
    status['cache_layout']=dict(zip(('rows','fields','jet_bytes','bytes'),map(int,layout.groups())))
cache_valid=bool(cache and int(cache[1])==2249856 and layout and tuple(map(int,layout.groups()))==(2592,26,104,2592*26*104))
status['passed']=status['passed'] and cache_valid
(output/'results.json').write_text(json.dumps(status,indent=2)+'\n')
print(text)
result.check_returncode()

assert cache_valid, 'missing/incomplete radial-cache poison/allocation coverage'
