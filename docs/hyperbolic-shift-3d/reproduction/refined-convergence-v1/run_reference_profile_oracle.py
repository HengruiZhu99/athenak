"""Build/run a separate read-only native profile oracle; preserve all inputs."""
from pathlib import Path
import json,hashlib,shlex,subprocess,time,csv
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists());O=R/'output/refined-convergence/reference-profile-oracle';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
T=R/'build-refined-convergence/tst/hyperbolic_shift_3d';target=T/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir';flags={}
for line in (target/'flags.make').read_text().splitlines():
 if ' = ' in line:k,v=line.split(' = ',1);flags[k]=shlex.split(v)
src=D/'reference_profile_oracle.cpp';obj=O/'reader.o';binary=O/'reader'
compile=['/usr/bin/c++']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-I'+str(R/'tst/hyperbolic_shift_3d'),'-c',str(src),'-o',str(obj)]
subprocess.run(compile,check=True);link=shlex.split((target/'link.txt').read_text());link[link.index('CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/amr_evolution.cpp.o')]=str(obj);link[link.index('-o')+1]=str(binary);subprocess.run(link,cwd=T,check=True)
inputs=sorted((R/'output/refined-convergence/reference-endpoint-reader').glob('segment01.rank*.csv'));radii=set()
for p in inputs:
 for row in csv.DictReader(p.open()):
  if int(row['ray']) in (0,3) and .3<float(row['r'])<.999:radii.add(float(row['r']))
# Dense native reference sampling resolves the support/shape; evolved rays remain unsmoothed.
radii.update(.85+.11*i/1100 for i in range(1101));(O/'radii.txt').write_text('\n'.join(format(x,'.17g') for x in sorted(radii))+'\n')
config=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial12_reference.cfg';cmd=[str(binary),str(config),str(O/'radii.txt'),str(O/'initial.csv')]
record=dict(compile=compile,link=link,command=cmd,source_sha256=sha(src),binary_sha256=sha(binary),config_sha256=sha(config),radii_sha256=sha(O/'radii.txt'),input_profiles={str(p.relative_to(R)):sha(p) for p in inputs},qualification='Separate local native exact initial/reference oracle, no evolution or checkpoint mutation. Derivatives by independent five-point probes at1e-5 and5e-6. Spacelike margin f/(f+R²S²) refers to initial stationary slice outside horizon, not an evolved time-coordinate invariant.')
start=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True);record.update(returncode=p.returncode,elapsed_seconds=time.monotonic()-start,stdout=p.stdout,stderr=p.stderr);(O/'command.json').write_text(json.dumps(record,indent=2)+'\n');assert p.returncode==0
print(json.dumps(record,indent=2))
