#!/usr/bin/env python3
"""Frozen same-engine actual-RK/topology replay of the adaptive Driver case."""
import argparse,hashlib,json,os,resource,shlex,signal,subprocess,time
from pathlib import Path
import numpy as np
from analyze_native_adaptive import sidecar
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('controls',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();control=a.controls.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);native=json.loads((control/'manifest.json').read_text());frozen=root/'docs/elliptic-shift-3d/evidence/native-adaptive-build-preflight-20261027/source';vendor=root/'docs/elliptic-shift-3d/evidence/native-static-smr-final-shared-build-20261027/compiled-source';source=out/'source';source.mkdir();files=['native_adaptive_reference.cpp','native_static_reference.cpp']
 for name in files:(source/name).write_bytes((root/'tst/elliptic_shift_3d'/name).read_bytes())
 flags=[]
 for line in (root/'build-shift-3d/CMakeFiles/elliptic_shift_3d_native_fields_tests.dir/flags.make').read_text().splitlines():
  if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):flags.extend(shlex.split(line.split('=',1)[1]))
 rewritten=[]
 for token in flags:
  if token.startswith('-I'+str(root)):token='-I'+str(vendor/Path(token[2:]).relative_to(root))
  elif token.startswith(str(root)):token=str(vendor/Path(token).relative_to(root))
  rewritten.append(token)
 flags=['-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*rewritten];src=source/files[0];probe=['/usr/bin/c++',*flags,'-M',str(src)];dep=subprocess.run(probe,capture_output=True,text=True,timeout=180);(out/'dependencies.stdout').write_text(dep.stdout);(out/'dependencies.stderr').write_text(dep.stderr);dep.check_returncode();dependencies=[Path(x).resolve() for x in shlex.split(dep.stdout.replace('\\\n',''))[1:]];leaks=[str(p) for p in dependencies if p.is_relative_to(root) and not any(p.is_relative_to(d) for d in [frozen,vendor,source])];assert not leaks,leaks
 libs=list((root/'build-shift-3d/kokkos').glob('*/src/libkokkos*.a'));m={'scope':'Same-engine transaction/RK replay driven by actual native leaf and live Driver coefficient receipts; no independent-PDE claim','live_project_header_leaks':leaks,'dependency_probe':probe,'libraries':[{'path':str(p),'sha256':sha(p)} for p in libs],'runs':[],'reference_source':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in source.iterdir()]}
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 def run(label,cmd):
  t=time.monotonic();row={'name':label,'command':list(map(str,cmd))};m['runs'].append(row);save()
  with (out/(label+'.stdout')).open('w') as stdout,(out/(label+'.stderr')).open('w') as stderr:
   process=subprocess.Popen(row['command'],cwd=root,stdout=stdout,stderr=stderr,start_new_session=True);row['owned_group']=process.pid
   try:row['exit_code']=process.wait(timeout=180)
   except subprocess.TimeoutExpired:
    row['exit_code']=124
    if os.getpgid(process.pid)!=process.pid:raise RuntimeError('unowned group')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
  row['seconds']=time.monotonic()-t;save();print(label,row['exit_code'],round(row['seconds'],3),flush=True);assert row['exit_code']==0
 binary=out/'reference';run('build',['/usr/bin/c++',*flags,src,'-o',binary,*libs]);full=control/'full';prefix='native_adaptive';dts=out/'dts.dat';trace=native['trace'];actual_dts=[sidecar(full/'rst'/f'{prefix}.{i:05d}.rst.composite')['dt'] for i in range(1,5)];dts.write_text(''.join(format(dt,'.17g')+'\n' for dt in actual_dts));m['dt_source']='actual binary nativeRST stage dt; never subtract rounded cumulative times';m['actual_dts']=actual_dts;save();run('reference',[binary,full/(prefix+'.mesh.dat'),full/(prefix+'.rk.dat'),dts,out/'reference',full/(prefix+'.regrid-1.mesh.dat'),full/(prefix+'.regrid-2.mesh.dat'),native['amplitude'],1])
 comparisons={}
 for stem,cycle in [(prefix,c) for c in range(5)]+[(prefix+'.regrid-1',1),(prefix+'.regrid-2',3)]:
  reference_stem=stem.replace(prefix,'reference',1)
  for suffix in ['.csv','.scri.csv']:
   actual=np.genfromtxt(full/f'{stem}.{cycle:05d}{suffix}',delimiter=',',names=True);expected=np.genfromtxt(out/f'{reference_stem}.{cycle:05d}{suffix}',delimiter=',',names=True);fields=[f'state_{f}' for f in range(22)];errors={f:float(np.max(abs(actual[f]-expected[f]))) for f in fields};assert all(np.array_equal(actual[f],expected[f]) for f in fields);comparisons[f'{stem}.{cycle:05d}{suffix}']={'cells':len(actual),'all22_coefficients_bitwise':True,'maximum_error':max(errors.values())}
  snapshot=sidecar(full/(stem+'.composite') if '.regrid-' in stem else full/'rst'/f'{stem}.{cycle:05d}.rst.composite');raw=(out/f'{reference_stem}.{cycle:05d}.shift.bin').read_bytes();count=struct.unpack_from('=Q',raw)[0];shift=np.frombuffer(raw,dtype='=f8',count=count,offset=8);assert np.array_equal(snapshot['shift'],shift);comparisons[f'{stem}.{cycle:05d}.canonical_shift']={'bitwise':True,'maximum_error':float(np.max(abs(snapshot['shift']-shift)))}
 m['comparisons']=comparisons;m['all_values_and_shifts_bitwise']=True;m['libraries_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in m['libraries']);save();assert m['libraries_unchanged'];print(json.dumps({'matched_snapshots':len(comparisons),'all_coefficients_and_canonical_beta_bitwise':True,'live_project_headers':0}))
if __name__=='__main__':
 import struct
 main()
