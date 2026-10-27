#!/usr/bin/env python3
"""Freeze and run targeted AMR core derivative controls; <=180s per process."""
import argparse,ast,hashlib,json,re,shlex,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path.cwd();build=a.build.resolve();out=a.output.resolve();initial=a.initial.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'manifest.json').exists():raise FileExistsError('Choose unused evidence directory')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();flags={}
flagfile=build/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make';linkfile=flagfile.parent/'link.txt'
for line in flagfile.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
link=shlex.split(linkfile.read_text());libraries=[(build/t).resolve() for t in link if t.endswith('.a')]
source=root/'tst/elliptic_shift_3d/test_amr_core_derivatives.cpp';generator=root/'tst/elliptic_shift_3d/wormhole_exact_jets.py';dependencies=set();queue=[source,generator,Path(__file__).resolve()]
while queue:
 f=queue.pop().resolve()
 if f in dependencies:continue
 dependencies.add(f);names=re.findall(r'^\s*#include\s+"([^"]+)"',f.read_text(),re.M)
 if f.suffix=='.py':
  for node in ast.walk(ast.parse(f.read_text())):
   if isinstance(node,ast.ImportFrom) and node.module:names.append(node.module.split('.')[0]+'.py')
   if isinstance(node,ast.Import):names.extend(n.name.split('.')[0]+'.py' for n in node.names)
 for name in names:
  for candidate in [f.parent/name,root/'src'/name,build/name,root/'tst/elliptic_shift_1d'/name,root/'tst/elliptic_shift_3d'/name]:
   if candidate.is_file():queue.append(candidate);break
dependencies.update({flagfile,linkfile});records=[]
for f in sorted(dependencies):
 relative=f.relative_to(root);dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes());records.append({'path':str(relative),'sha256':sha(f)})
binary=out/'amr-core-derivatives';catalog=out/'target-coordinates.csv';exact=out/'exact-jets.dat'
compile_cmd=[link[0],*flags['CXX_DEFINES'],'-I'+str(out/'source/src'),'-I'+str(out/'source'/build.relative_to(root)),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(out/'source'/source.relative_to(root)),'-o',str(binary),*[str(f) for f in libraries]]
commands={'build':compile_cmd,'catalog':[str(binary),'catalog',str(initial),str(catalog)],'oracle':[sys.executable,str(out/'source'/generator.relative_to(root)),str(catalog),str(exact)],'diagnostic':[str(binary),'diagnose',str(initial),str(exact)]}
manifest={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source':records,'libraries':[{'path':str(f),'sha256':sha(f)} for f in libraries],
 'initial':{'path':str(initial),'sha256':sha(initial)},'maximum_active_cells':50000,'timeout_seconds_per_process':180,'compiled_from_frozen_archive':True,'runs':{},'physics_rhs_changed':False}
try:
 for label,cmd in commands.items():
  start=time.monotonic()
  try:
   r=subprocess.run(cmd,text=True,capture_output=True,timeout=180);(out/f'{label}.stdout').write_text(r.stdout);(out/f'{label}.stderr').write_text(r.stderr)
   manifest['runs'][label]={'command':cmd,'exit_code':r.returncode,'seconds':time.monotonic()-start}
   if r.returncode:raise RuntimeError(f'{label} failed: see evidence')
  except subprocess.TimeoutExpired as e:
   for kind,value in [('stdout',e.stdout),('stderr',e.stderr)]:
    (out/f'{label}.{kind}').write_text(value.decode() if isinstance(value,bytes) else value or '')
   manifest['runs'][label]={'command':cmd,'timeout':True,'seconds':time.monotonic()-start};raise
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 manifest['binary_sha256']=sha(binary);manifest['catalog_sha256']=sha(catalog);manifest['oracle_sha256']=sha(exact)
 results=[json.loads(line) for line in (out/'diagnostic.stdout').read_text().splitlines()]
 (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
finally:
 manifest['source_unchanged']=all(sha(root/r['path'])==r['sha256'] for r in records);manifest['libraries_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in manifest['libraries']);manifest['initial_unchanged']=sha(initial)==manifest['initial']['sha256']
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k not in ('source','libraries')},indent=2))
