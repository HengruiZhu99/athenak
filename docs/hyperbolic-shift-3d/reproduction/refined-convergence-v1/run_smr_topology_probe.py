"""Reproduce one fresh topology-only probe with existing MPI/native AMR machinery."""
from pathlib import Path
import argparse,json,hashlib,shlex,subprocess,time
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['cube','offset-tube','symmetric-tube'],required=True);a.add_argument('--output',required=True);a=a.parse_args();O=Path(a.output).resolve();O.mkdir(parents=True,exist_ok=False)
T=R/'build-refined-convergence/tst/hyperbolic_shift_3d';target=T/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir';flags={}
for line in (target/'flags.make').read_text().splitlines():
 if ' = ' in line:k,v=line.split(' = ',1);flags[k]=shlex.split(v)
src=D/'probe_smr_topology.cpp';obj=O/'probe.o';binary=O/'probe';cmd=['/usr/bin/c++']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-I'+str(R/'tst/hyperbolic_shift_3d'),'-c',str(src),'-o',str(obj)]
subprocess.run(cmd,check=True);link=shlex.split((target/'link.txt').read_text());link[link.index('CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/amr_evolution.cpp.o')]=str(obj);link[link.index('-o')+1]=str(binary);subprocess.run(link,cwd=T,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(O/'probe-source.cpp').write_bytes(src.read_bytes());(O/'build.json').write_text(json.dumps(dict(compile=cmd,link=link,source_sha256=sha(src),binary_sha256=sha(binary)),indent=2)+'\n')
config=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial18_reference.cfg';command=['mpiexec','-n','2',str(binary),str(config),'.05',str(O/'topology.json')]
if a.mode!='cube':command.append('tube' if a.mode=='offset-tube' else 'symmetric-tube')
began=time.monotonic()
with (O/'native.log').open('w') as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=300)
r=dict(command=command,returncode=p.returncode,elapsed_seconds=time.monotonic()-began,config_sha256=sha(config),qualification='Topology probe only, no evolution or checkpoint. Analysis allocation2048/ranks+8; production capacity unchanged. Tube flags not a production runtime policy.')
(O/'command.json').write_text(json.dumps(r,indent=2)+'\n');assert p.returncode==0;print(json.dumps(r,indent=2))
