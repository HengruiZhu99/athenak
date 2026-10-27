#!/usr/bin/env python3
"""Frozen, bounded single-rho enrichment falsifier. No shared helper edits."""
import argparse,difflib,hashlib,json,re,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--initial',type=Path);p.add_argument('--actual',action='store_true');p.add_argument('--source-noise',action='store_true');p.add_argument('--bounded-actual',action='store_true');p.add_argument('--bounded-source-noise',action='store_true');p.add_argument('--supported',action='store_true');p.add_argument('--supported-actual',action='store_true');p.add_argument('--supported-source-noise',action='store_true');p.add_argument('--adapter',action='store_true');a=p.parse_args()
actual_mode=a.actual or a.source_noise or a.bounded_actual or a.bounded_source_noise or a.supported_actual or a.supported_source_noise or a.adapter
requires_initial=actual_mode and not a.adapter
if sum([a.actual,a.source_noise,a.bounded_actual,a.bounded_source_noise,a.supported,a.supported_actual,a.supported_source_noise,a.adapter])>1:p.error('choose one diagnostic mode')
if requires_initial and not a.initial:p.error('actual-data modes require --initial')
root=Path.cwd();build=a.build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'manifest.json').exists():raise FileExistsError('Choose unused evidence directory')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();flags={};flagfile=build/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make';linkfile=flagfile.parent/'link.txt'
for line in flagfile.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
link=shlex.split(linkfile.read_text());libraries=[(build/t).resolve() for t in link if t.endswith('.a')]
source=root/('tst/elliptic_shift_3d/test_supported_puncture_adapter.cpp' if a.adapter else 'tst/elliptic_shift_3d/test_supported_rho_log.cpp' if a.supported else 'tst/elliptic_shift_3d/test_supported_rho_actual.cpp' if a.supported_actual else 'tst/elliptic_shift_3d/test_supported_rho_source_noise.cpp' if a.supported_source_noise else 'tst/elliptic_shift_3d/test_bounded_rho_log_actual.cpp' if a.bounded_actual else 'tst/elliptic_shift_3d/test_bounded_rho_source_noise.cpp' if a.bounded_source_noise else 'tst/elliptic_shift_3d/test_rho_log_source_noise.cpp' if a.source_noise else 'tst/elliptic_shift_3d/test_rho_log_actual.cpp' if a.actual else 'tst/elliptic_shift_3d/test_rho_log_derivative.cpp');dependencies=set();queue=[source,Path(__file__).resolve()]
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
old=root/'src/z4c/hyperboloidal/odd_radial_log_derivative.hpp';new=root/'src/z4c/hyperboloidal/rho_log_derivative.hpp'
(out/'prototype-derivation.patch').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile='reference/odd_radial_log_derivative.hpp',tofile='prototype/rho_log_derivative.hpp')))
binary=out/'rho-log-diagnostic';compile_cmd=[link[0],*flags['CXX_DEFINES'],'-I'+str(out/'source/src'),'-I'+str(out/'source'/build.relative_to(root)),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(out/'source'/source.relative_to(root)),'-o',str(binary),*[str(f) for f in libraries]]
inputs=[{'path':str(a.initial.resolve()),'sha256':sha(a.initial.resolve())}] if requires_initial else [];run_cmd=[str(binary),str(a.initial.resolve())] if requires_initial else [str(binary)]
manifest={'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source':records,'inputs':inputs,'libraries':[{'path':str(f),'sha256':sha(f)} for f in libraries],
 'maximum_active_cells':50000,'timeout_seconds_per_process':180,'compiled_from_frozen_archive':True,'actual_mesh':actual_mode,'standalone_adapter':a.adapter,'source_noise':a.source_noise or a.bounded_source_noise or a.supported_source_noise,'bounded_variant':a.bounded_actual or a.bounded_source_noise,'support_variant':a.supported or a.supported_actual or a.supported_source_noise,'basis':'geometry-onlyboundedblend' if a.bounded_actual or a.bounded_source_noise else 'degree5Cartesian+logrho+rho','guard_value_l1':16,'guard_gradient_l1':32,'guard_hessian_l1':64,'runs':{},'production_adopted':False}
try:
 for label,cmd in [('build',compile_cmd),('diagnostic',run_cmd)]:
  start=time.monotonic()
  try:
   r=subprocess.run(cmd,text=True,capture_output=True,timeout=180);(out/f'{label}.stdout').write_text(r.stdout);(out/f'{label}.stderr').write_text(r.stderr);manifest['runs'][label]={'command':cmd,'exit_code':r.returncode,'seconds':time.monotonic()-start}
   if r.returncode:raise RuntimeError(f'{label} failed: see frozen evidence')
  except subprocess.TimeoutExpired as e:
   for kind,value in [('stdout',e.stdout),('stderr',e.stderr)]:
    (out/f'{label}.{kind}').write_text(value.decode() if isinstance(value,bytes) else value or '')
   manifest['runs'][label]={'command':cmd,'timeout':True,'seconds':time.monotonic()-start};raise
  (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 manifest['binary_sha256']=sha(binary);results=[json.loads(line) for line in (out/'diagnostic.stdout').read_text().splitlines()];(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'rows':len(results),'summary':results[-1]},indent=2))
finally:
 manifest['source_unchanged']=all(sha(root/r['path'])==r['sha256'] for r in records);manifest['libraries_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in manifest['libraries']);manifest['inputs_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in inputs);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
