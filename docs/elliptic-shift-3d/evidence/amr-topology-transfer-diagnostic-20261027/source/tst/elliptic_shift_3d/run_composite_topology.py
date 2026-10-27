#!/usr/bin/env python3
"""Freeze, build and run the isolated topology transfer test, at most 180s each."""
import argparse,hashlib,json,re,shlex,subprocess,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path.cwd();build=a.build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'manifest.json').exists():raise FileExistsError('Choose unused evidence directory')
flags={}
flagfile=build/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make'
linkfile=flagfile.parent/'link.txt'
for line in flagfile.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
link=shlex.split(linkfile.read_text());libraries=[(build/t).resolve() for t in link if t.endswith('.a')]
source=root/'tst/elliptic_shift_3d/test_composite_topology.cpp';binary=out/'composite-topology'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
dependencies=set();queue=[source]
while queue:
 f=queue.pop().resolve()
 if f in dependencies:continue
 dependencies.add(f)
 for name in re.findall(r'^\s*#include\s+"([^"]+)"',f.read_text(),re.M):
  for candidate in [f.parent/name,root/'src'/name,build/name]:
   if candidate.is_file():queue.append(candidate);break
dependencies.update({Path(__file__).resolve(),flagfile,linkfile})
records=[]
for f in sorted(dependencies):
 relative=f.relative_to(root);destination=out/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(f.read_bytes())
 records.append({'path':str(relative),'sha256':sha(f)})
command=[link[0],*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(source),'-o',str(binary),*[str(f) for f in libraries]]
manifest={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source':records,'libraries':[{'path':str(f),'sha256':sha(f)} for f in libraries],
 'build_command':command,'run_command':[str(binary)],'timeout_seconds_per_process':180,'maximum_active_cells':33104,'dense_matrix_allocated':False,'production_integration':False}
try:
 for label,cmd in [('build',command),('test',[str(binary)])]:
  start=time.monotonic()
  try:
   run=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
   (out/f'{label}.stdout').write_text(run.stdout);(out/f'{label}.stderr').write_text(run.stderr)
   manifest[label]={'exit_code':run.returncode,'seconds':time.monotonic()-start}
   if run.returncode:raise RuntimeError(f'{label} exited {run.returncode}; see evidence')
  except subprocess.TimeoutExpired as e:
   (out/f'{label}.stdout').write_text((e.stdout or b'').decode() if isinstance(e.stdout,bytes) else e.stdout or '')
   (out/f'{label}.stderr').write_text((e.stderr or b'').decode() if isinstance(e.stderr,bytes) else e.stderr or '')
   manifest[label]={'timeout':True,'seconds':time.monotonic()-start};raise
 manifest['binary_sha256']=sha(binary);manifest['results']=[json.loads(line) for line in (out/'test.stdout').read_text().splitlines()]
finally:
 manifest['source_unchanged']=all(sha(root/r['path'])==r['sha256'] for r in records)
 manifest['libraries_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in manifest['libraries'])
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
