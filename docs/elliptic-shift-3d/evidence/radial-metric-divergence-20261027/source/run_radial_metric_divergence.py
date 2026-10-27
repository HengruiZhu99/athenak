#!/usr/bin/env python3
"""Build one disjoint diagnostic from the existing build flags, no CMake edits."""
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);build=a.build.resolve();out=a.output.resolve();root=Path.cwd()
flags={}
for line in (build/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make').read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
link=shlex.split((build/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/link.txt').read_text())
libraries=[str(build/t) for t in link if t.endswith('.a')]
source=root/'tst/elliptic_shift_3d/test_radial_metric_divergence.cpp';binary=out/'radial-metric-divergence'
command=[link[0],*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(source),'-o',str(binary),*libraries]
dependencies=[source,root/'src/z4c/hyperboloidal/composite_metric_divergence.hpp',root/'src/z4c/hyperboloidal/elliptic_shift.hpp',root/'src/z4c/hyperboloidal/conformal_constraints.hpp',root/'src/z4c/hyperboloidal/conformal_rhs.hpp',root/'src/mesh/cubed_sphere/composite_difference.hpp',root/'src/mesh/cubed_sphere/composite.hpp',root/'src/mesh/cubed_sphere/scri_surface.hpp',root/'src/mesh/cubed_sphere/map.hpp',a.data.resolve()]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
manifest=dict(n=8,timeout_per_process_seconds=180,commands=[],source_hashes={str(f):sha(f) for f in dependencies},native_integration=False,source_policy='actual sampled wormhole alpha*A, unchanged physical source')
def run(name,cmd):
 start=time.monotonic()
 try:
  result=subprocess.run(cmd,text=True,capture_output=True,timeout=180);code=result.returncode;stdout=result.stdout;stderr=result.stderr
 except subprocess.TimeoutExpired as e:
  code=124;stdout=e.stdout or '';stderr=e.stderr or ''
  if isinstance(stdout,bytes):stdout=stdout.decode()
  if isinstance(stderr,bytes):stderr=stderr.decode()
 (out/(name+'.stdout')).write_text(stdout);(out/(name+'.stderr')).write_text(stderr)
 receipt=dict(name=name,command=cmd,exit_code=code,seconds=time.monotonic()-start)
 try:receipt['result']=json.loads(stdout)
 except ValueError:pass
 manifest['commands'].append(receipt);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(name,code,round(receipt['seconds'],3),flush=True);return code
if run('build',command):raise SystemExit(1)
manifest['binary_sha256']=sha(binary)
for radial in [0,1]:
 run(f'radial-interface-{radial}',[str(binary),'8',str(a.data.resolve()),str(radial),str(out/f'radial-interface-{radial}.points.csv')])
manifest['sources_unchanged']=all(sha(Path(f))==h for f,h in manifest['source_hashes'].items());manifest['binary_unchanged']=sha(binary)==manifest['binary_sha256']
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
raise SystemExit(int(any(c['exit_code'] for c in manifest['commands'])))
