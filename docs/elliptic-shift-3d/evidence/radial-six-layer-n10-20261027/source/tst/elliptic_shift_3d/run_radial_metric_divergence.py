#!/usr/bin/env python3
"""Build a disjoint bounded static diagnostic, optionally regenerate exact data."""
import argparse,hashlib,json,re,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--data',type=Path)
p.add_argument('--output',type=Path,required=True);p.add_argument('--radii',default='.62,.70,1')
p.add_argument('--datum-ghosts',type=int,choices=[2,4],default=2);p.add_argument('--radial-flags',type=int,choices=[0,1],nargs='+',default=[0,1])
p.add_argument('--centered-only',action='store_true');p.add_argument('--n',type=int,choices=[6,8,10],default=8);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);build=a.build.resolve();out=a.output.resolve();root=Path.cwd()
if (out/'manifest.json').exists():raise FileExistsError('Choose an unused evidence directory')
flags={}
for line in (build/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make').read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
link=shlex.split((build/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/link.txt').read_text())
libraries=[str(build/t) for t in link if t.endswith('.a')]
source=root/'tst/elliptic_shift_3d/test_radial_metric_divergence.cpp';binary=out/'radial-metric-divergence'
command=[link[0],*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(source),'-o',str(binary),*libraries]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
# Freeze quote-included project headers before compilation, including generated
# config and unused template helper includes; retain exact bytes, not only hashes.
dependencies=set();queue=[source]
while queue:
 f=queue.pop().resolve()
 if f in dependencies:continue
 dependencies.add(f)
 for name in re.findall(r'^\s*#include\s+"([^"]+)"',f.read_text(),re.M):
  for candidate in [f.parent/name,root/'src'/name,build/name]:
   if candidate.is_file():queue.append(candidate);break
generator=root/'tst/elliptic_shift_3d/wormhole_exact_jets.py'
if a.data is None:
 dependencies.update((root/'tst/elliptic_shift_1d').glob('*.py'))
 dependencies.update([generator,root/'tst/elliptic_shift_3d/radial_reference.py'])
dependencies.add(Path(__file__).resolve())
if a.data is not None:dependencies.add(a.data.resolve())
manifest=dict(n=a.n,radii=a.radii,timeout_per_process_seconds=180,commands=[],source_hashes={str(f):sha(f) for f in sorted(dependencies)},
 library_hashes={f:sha(Path(f)) for f in libraries},native_integration=False,source_policy='actual sampled wormhole alpha*A, unchanged physical source')
archive=out/'source';archive.mkdir()
for f in sorted(dependencies):
 if f.suffix=='.dat':continue
 target=archive/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(f.read_bytes())
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
if a.data is None:
 coordinates=out/'coordinates.csv';datum=out/'exact.dat';datum_ghosts=4
 if run('coordinates',[str(binary),str(a.n),'--coordinates',str(a.radial_flags[0]),str(coordinates),a.radii,'4',str(int(a.centered_only))]):raise SystemExit(1)
 datum_ghosts=manifest['commands'][-1]['result']['datum_ghosts']
 if run('generate-independent-data',['python3',str(generator),str(coordinates),str(datum)]):raise SystemExit(1)
 manifest['input_hashes']={str(coordinates):sha(coordinates),str(datum):sha(datum)}
else:datum=a.data.resolve();datum_ghosts=a.datum_ghosts
for radial in a.radial_flags:
 run(f'radial-interface-{radial}',[str(binary),str(a.n),str(datum),str(radial),str(out/f'radial-interface-{radial}.points.csv'),a.radii,str(datum_ghosts),str(int(a.centered_only))])
manifest['sources_unchanged']=all(sha(Path(f))==h for f,h in manifest['source_hashes'].items());manifest['binary_unchanged']=sha(binary)==manifest['binary_sha256']
manifest['libraries_unchanged']=all(sha(Path(f))==h for f,h in manifest['library_hashes'].items())
if 'input_hashes' in manifest:manifest['inputs_unchanged']=all(sha(Path(f))==h for f,h in manifest['input_hashes'].items())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
raise SystemExit(int(any(c['exit_code'] for c in manifest['commands'])))
