#!/usr/bin/env python3
"""Link a complete native stage RHS regression against an already built AthenaK executable.

This reuses its native object files and link line; no second driver is implemented.
The independent legacy initializer and complete CPU RHS are included only by the
test translation unit. This local regression does not evolve an RK step.
"""
import argparse
import hashlib
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
tracked=sorted((root/'src/z4c_hyperboloidal').glob('*.[ch]pp'))
tracked += [root/p for p in ['src/mesh/chart_atlas.cpp','src/mesh/chart_atlas.hpp',
  'src/bvals/mapped_exchange.cpp','src/bvals/mapped_exchange.hpp',
  'tst/z4c_hyperboloidal/coupled_rhs_oracle.cpp']]
source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
flags=(build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M).group(1))
options=['BROAD_SMOOTH_A','CONFIGURED_FORMULATION','DIRECT_OUTER','INNER_TELEGRAPHER',
         'NORMALIZED_INNER_SHIFT','RESTORE_INNER_DAMPING','SHORT_INNER_GAUGE',
         'SMOOTH_GAUGE_BLEND','SPHERICAL_POLICY']
obj=output/'coupled_rhs_oracle.o';exe=output/'coupled_rhs_oracle'
compile=['/usr/bin/c++','-std=c++17','-O2','-ffp-contract=off']+includes
compile += ['-DHYPERBOLIC_'+x+'=1' for x in options]
compile += ['-I'+str(legacy/'src'),'-I'+str(legacy),'-c',
            str(root/'tst/z4c_hyperboloidal/coupled_rhs_oracle.cpp'),'-o',str(obj)]
link=shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link=[str(obj) if x=='CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
assert str(obj) in link
link[link.index('-o')+1]=str(exe)
for name in ['surface_exchange','mapped_halo','dissipation_plan',
             'z4c_hyperboloidal_tasks','z4c_hyperboloidal_rhs']:
 assert any(x.endswith('/'+name+'.cpp.o') for x in link), \
     'required stage source absent from original AthenaK executable: '+name
makefile=output/'coupled_rhs_oracle.mk'
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
for m in re.finditer(r'^(\w+) comparisons=(\d+) max_scaled=([^ ]+) max_absolute=([^ ]+) max_tolerance_ratio=([^ ]+) failures=(\d+) weighted_rms=([^ ]+) weight_sum=([^\n]+)$',text,re.M):
 metrics[m.group(1)]={'comparisons':int(m.group(2)),'max_scaled_error':float(m.group(3)),
  'max_absolute_error':float(m.group(4)),'max_tolerance_ratio':float(m.group(5)),
  'failed_values':int(m.group(6)),'weighted_rms':float(m.group(7)),
  'weight_sum':float(m.group(8)),'per_field':[],'worst_location':{}}
for m in re.finditer(r'^(\w+) field=(\d+) max_absolute=([^ ]+) max_scaled=([^ ]+) weighted_rms=([^ ]+) failures=(\d+)$',text,re.M):
 metrics[m.group(1)]['per_field'].append({'field':int(m.group(2)),
  'max_absolute_error':float(m.group(3)),'max_scaled_error':float(m.group(4)),
  'weighted_rms':float(m.group(5)),'failed_values':int(m.group(6))})
for m in re.finditer(r'^(\w+) worst=(absolute|scaled) field=(\d+) gid=(\d+) ijk=([^ ]+) xyz=([^ ]+) scenario=(\d+) rank=(\d+) actual=([^ ]+) expected=([^\n]+)$',text,re.M):
 metrics[m.group(1)]['worst_location'][m.group(2)]={'field':int(m.group(3)),
  'native_gid':int(m.group(4)),'cell_indices':[int(x) for x in m.group(5).split(',')],
  'coordinates':[float(x) for x in m.group(6).split(',')],
  'scenario':int(m.group(7)),'rank':int(m.group(8)),
  'actual':float(m.group(9)),'expected':float(m.group(10))}
status={'passed':result.returncode==0,'local_mpi_ranks':args.ranks,'scenarios':2,
 'gate':'abs(delta)<=1e-12+5e-10*abs(reference); chi<=1e-14+5e-10*abs(reference)','metrics':metrics,
 'native_build':str(build),'legacy_readonly_source':str(legacy),
 'source_sha256':source_hashes,'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
 'native_compile_flags':flags,'oracle_compile_command':compile,
 'rms_weights':{'bulk':'legacy grid.volumes for all independently evolved points, including overlap and the outer endpoint-grid half weight',
  'surface':'legacy physical cell volume at scri divided by its local radial spacing (surface area weight)',
  'aggregation':'sum(weight*delta^2)/sum(weight), square root; category aggregate includes all31fields and both scenarios; per-field RMS separately'}}
status['source_unchanged_during_run']=source_hashes=={
 str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
m=re.search(r'coordinate_mismatches=(\d+) coordinate_max_absolute=([^\n]+)',text)
if m:status.update(coordinate_mismatches=int(m.group(1)),coordinate_max_absolute=float(m.group(2)))
m=re.search(r'legacy_normal_compatibility=([^\n]+)',text)
if m:status.update(legacy_normal_compatibility=float(m.group(1)))
(output/'results.json').write_text(json.dumps(status,indent=2)+'\n')
print(text)
result.check_returncode()
assert status['source_unchanged_during_run'], 'native/test sources changed during oracle run'
assert set(metrics)=={'initial_bulk','initial_surface','bulk_rhs','surface_rhs'}
assert all(len(x['per_field'])==31 and len(x['worst_location'])==2 for x in metrics.values())
