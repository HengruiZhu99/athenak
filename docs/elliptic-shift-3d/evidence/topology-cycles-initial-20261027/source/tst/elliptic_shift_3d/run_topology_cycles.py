#!/usr/bin/env python3
"""Frozen standalone global hierarchy qualification; no root pilot/CMake build."""
import argparse,hashlib,json,re,shlex,subprocess,time
from pathlib import Path

cases={}
for equation in ('scalar','vector'):
 for n in (6,8):
  for shells in (1,2,7):cases[f'{equation}_n{n}_s{shells}']=['--equation',equation,'--n',str(n),'--shells',str(shells)]
  cases[f'{equation}_n{n}_mixed']=['--equation',equation,'--n',str(n),'--shells','2','--mixed','1']
  cases[f'{equation}_n{n}_subdivided']=['--equation',equation,'--n',str(n),'--shells','2','--subdivisions','2']
cases['scalar_classic_n8']=['--equation','scalar','--n','8','--shells','2','--topology','0']
cases['vector_continuum_n6']=['--equation','vector','--n','6','--shells','2','--continuum','1']
cases['vector_flat_n6']=['--equation','vector','--n','6','--shells','2','--curved','0']
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--cases',nargs='+',choices=sorted(cases),default=['scalar_n6_s1','vector_n6_s2']);a=p.parse_args()
root=Path.cwd();build=a.build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'manifest.json').exists():raise FileExistsError('Choose unused evidence directory')
flags={};flagfile=build/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make';linkfile=flagfile.parent/'link.txt'
for line in flagfile.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
link=shlex.split(linkfile.read_text());libraries=[(build/t).resolve() for t in link if t.endswith('.a')]
source=root/'tst/elliptic_shift_3d/test_topology_cycles.cpp';binary=out/'topology-cycles'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
dependencies=set();queue=[source]
while queue:
 f=queue.pop().resolve()
 if f in dependencies:continue
 dependencies.add(f)
 for name in re.findall(r'^\s*#include\s+"([^"]+)"',f.read_text(),re.M):
  for candidate in [f.parent/name,root/'src'/name,build/name]:
   if candidate.is_file():queue.append(candidate);break
dependencies.update({Path(__file__).resolve(),flagfile,linkfile});records=[]
for f in sorted(dependencies):
 relative=f.relative_to(root);destination=out/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(f.read_bytes())
 records.append({'path':str(relative),'sha256':sha(f)})
command=[link[0],*flags['CXX_DEFINES'],'-I'+str(out/'source/src'),'-I'+str(out/'source'/build.relative_to(root)),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(out/'source'/source.relative_to(root)),'-o',str(binary),*[str(f) for f in libraries]]
manifest={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source':records,'libraries':[{'path':str(f),'sha256':sha(f)} for f in libraries],
 'build_command':command,'timeout_seconds_per_process':180,'maximum_active_cells':50000,'compiled_from_frozen_archive':True,'production_integration':False,'runs':{}}
def run(label,cmd):
 start=time.monotonic()
 try:
  r=subprocess.run(cmd,text=True,capture_output=True,timeout=180);(out/f'{label}.stdout').write_text(r.stdout);(out/f'{label}.stderr').write_text(r.stderr)
  result={'exit_code':r.returncode,'seconds':time.monotonic()-start,'command':cmd}
  if label!='build' and r.stdout.strip():
   try:result['result']=json.loads(r.stdout)
   except json.JSONDecodeError:pass
  return result
 except subprocess.TimeoutExpired as e:
  for kind,value in [('stdout',e.stdout),('stderr',e.stderr)]:
   (out/f'{label}.{kind}').write_text(value.decode() if isinstance(value,bytes) else value or '')
  return {'timeout':True,'seconds':time.monotonic()-start,'command':cmd}
try:
 manifest['build']=run('build',command)
 if manifest['build'].get('exit_code')!=0:raise RuntimeError('standalone build failed')
 manifest['binary_sha256']=sha(binary)
 for case in a.cases:
  result=run(case,[str(binary),*cases[case]]);manifest['runs'][case]=result
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  print(json.dumps({'case':case,**result}),flush=True)
  if result.get('exit_code')!=0:break
finally:
 manifest['source_unchanged']=all(sha(root/r['path'])==r['sha256'] for r in records)
 manifest['libraries_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in manifest['libraries'])
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if manifest['build'].get('exit_code')!=0 or any(r.get('exit_code')!=0 for r in manifest['runs'].values()):raise SystemExit(1)
