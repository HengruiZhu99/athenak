#!/usr/bin/env python3
"""Read-only shared sources; explicit frozen-header row-budget experiment."""
import argparse,difflib,hashlib,json,re,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--radius',type=float,default=.24);p.add_argument('--maximum-rows',type=int,default=20000);p.add_argument('--maximum-donors',type=int,choices=[175,256],default=175);a=p.parse_args()
if not 10000<=a.maximum_rows<=20000:p.error('this frozen experiment permits maximum_rows in[10000,20000]')
root=Path.cwd();build=a.build.resolve();out=a.output.resolve();initial=a.initial.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'manifest.json').exists():raise FileExistsError('Choose unused evidence directory')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();flagfile=build/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make';linkfile=flagfile.parent/'link.txt';flags={}
for line in flagfile.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
link=shlex.split(linkfile.read_text());libraries=[(build/t).resolve() for t in link if t.endswith('.a')]
source=root/'tst/elliptic_shift_3d/test_amr_expanded_value_cache.cpp';dependencies=set();queue=[source,Path(__file__).resolve()]
while queue:
 f=queue.pop().resolve()
 if f in dependencies:continue
 dependencies.add(f)
 for name in re.findall(r'^\s*#include\s+"([^"]+)"',f.read_text(),re.M):
  for candidate in [f.parent/name,root/'src'/name,build/name]:
   if candidate.is_file():queue.append(candidate);break
dependencies.update({flagfile,linkfile});records=[]
for f in sorted(dependencies):
 relative=f.relative_to(root);dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);data=f.read_bytes();dest.write_bytes(data);records.append({'path':str(relative),'sha256':hashlib.sha256(data).hexdigest()})
variant=out/'source/src/z4c/hyperboloidal/puncture_log_value_plans.hpp';original=variant.read_text()
changes={'maximum_rows>10000':'maximum_rows>20000','value-plan row budget must lie in[1,10000]':'value-plan row budget must lie in[1,20000]'}
modified=original
for old,new in changes.items():
 if modified.count(old)!=1:raise RuntimeError('expected frozen row-cap guard changed upstream; inspect before patching')
 modified=modified.replace(old,new)
variant.write_text(modified);(out/'frozen-row-budget.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='original/puncture_log_value_plans.hpp',tofile='frozen-variant/puncture_log_value_plans.hpp')))
binary=out/'amr-expanded-values';compile_cmd=[link[0],*flags['CXX_DEFINES'],'-I'+str(out/'source/src'),'-I'+str(out/'source'/build.relative_to(root)),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(out/'source'/source.relative_to(root)),'-o',str(binary),*[str(f) for f in libraries]]
arguments=[str(binary),str(initial),str(a.radius),str(a.maximum_rows),str(a.maximum_donors)]
manifest={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source':records,'libraries':[{'path':str(f),'sha256':sha(f)} for f in libraries],
 'initial':{'path':str(initial),'sha256':sha(initial)},'frozen_variant':{'path':str(variant.relative_to(out)),'sha256':sha(variant),'changes':changes,'shared_header_modified':False},
 'radius':a.radius,'maximum_rows':a.maximum_rows,'maximum_donors':a.maximum_donors,'value_l1_cap':8,'maximum_active_cells':50000,'timeout_seconds_per_process':180,'compiled_from_frozen_archive':True,'runs':{},'physics_rhs_changed':False}
def run(label,cmd):
 start=time.monotonic()
 try:
  r=subprocess.run(cmd,text=True,capture_output=True,timeout=180);(out/f'{label}.stdout').write_text(r.stdout);(out/f'{label}.stderr').write_text(r.stderr)
  manifest['runs'][label]={'command':cmd,'exit_code':r.returncode,'seconds':time.monotonic()-start}
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  if r.returncode:raise RuntimeError(f'{label} failed: see evidence')
 except subprocess.TimeoutExpired as e:
  for kind,value in [('stdout',e.stdout),('stderr',e.stderr)]:
   (out/f'{label}.{kind}').write_text(value.decode() if isinstance(value,bytes) else value or '')
  manifest['runs'][label]={'command':cmd,'timeout':True,'seconds':time.monotonic()-start};raise
try:
 run('build',compile_cmd);manifest['binary_sha256']=sha(binary);run('count',[*arguments,'count']);count=json.loads((out/'count.stdout').read_text());manifest['count']=count
 print(json.dumps(count),flush=True)
 if count['unique_query_spacing_plans']>a.maximum_rows:raise RuntimeError('count exceeds explicit frozen-plan budget; no truncation or construction attempted')
 run('diagnostic',[*arguments,'diagnose']);results=[json.loads(line) for line in (out/'diagnostic.stdout').read_text().splitlines()];(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2),flush=True)
finally:
 manifest['source_unchanged']=all(sha(root/r['path'])==r['sha256'] for r in records);manifest['libraries_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in manifest['libraries']);manifest['initial_unchanged']=sha(initial)==manifest['initial']['sha256'];(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
