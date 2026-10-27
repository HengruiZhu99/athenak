#!/usr/bin/env python3
"""Bounded actual static-SMR native Driver controls. Never launches allocations."""
import argparse,hashlib,json,os,resource,shlex,signal,subprocess,sys,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--build-evidence',type=Path,default=Path('docs/elliptic-shift-3d/evidence/native-static-smr-audited-relink-20261027'));p.add_argument('--resume',action='store_true');p.add_argument('--wormhole',action='store_true');p.add_argument('--coordinates',type=Path);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=a.resume);resource.setrlimit(resource.RLIMIT_CORE,(0,0));build=a.build_evidence.resolve();binary=build/'athena-frozen';build_manifest=json.loads((build/'manifest.json').read_text());frozen=Path(build_manifest.get('base_frozen_source',str(build/'source')));prefix='native_smr';scope='wormhole' if a.wormhole else 'Minkowski'
 m={'scope':'Actual Serial static Cartesian SMR Driver integration; engine-owned shells; '+scope,'scientific_acceptance':False,'binary_sha256':sha(binary),'frozen_build_manifest_sha256':sha(build/'manifest.json'),'maximum_seconds_per_process':180,'maximum_active_cells':50000,'cases':[],'shell_native_MeshBlocks':False,'dynamic_regrid_supported':False}
 if a.resume:
  old=(out/'manifest.json').read_bytes();(out/'manifest-before-epoch-control-fix.json').write_bytes(old);m=json.loads(old);m['resume_reason']='complete physical state bitwise; replace overstrict whole-byte test by specified Restore epoch+1 contract';m['runner-resume-source-sha256']=sha(Path(__file__))
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 def run(name,args,success=True,folder=None):
  folder=folder or out/name;folder.mkdir(parents=True,exist_ok=False);row={'name':name,'command':list(map(str,args)),'expected_success':success};m['cases'].append(row);save();t=time.monotonic()
  with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
   process=subprocess.Popen(row['command'],cwd=root,stdout=stdout,stderr=stderr,start_new_session=True);row['owned_group']=process.pid;save()
   try:row['exit_code']=process.wait(timeout=180)
   except subprocess.TimeoutExpired:
    row['exit_code']=124
    if os.getpgid(process.pid)!=process.pid:raise RuntimeError('refusing unowned group')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
    row['cleanup']='only newly owned process group terminated'
  row['seconds']=time.monotonic()-t;row['expectation_met']=(row['exit_code']==0)==success;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True)
  if not row['expectation_met']:raise RuntimeError('unexpected native static result '+name)
  return folder
 template=(root/'inputs/z4c/composite_minkowski.athinput').read_text();template=template.replace('nx1=6','nx1=12',1).replace('nx2=6','nx2=12',1).replace('nx3=6','nx3=12',1).replace('nghost=3','nghost=4').replace('refinement=none','refinement=static\nmax_nmb_per_rank=15\nnum_levels=2').replace('basename=native_composite','basename='+prefix).replace('composite_output=native_composite','composite_output='+prefix).replace('composite_radii=.65,1','composite_radii=.45,.55,.60,.65,.70,.75,1')
 template+='\n<refined_region1>\nlevel=1\nx1min=-.24\nx1max=-.01\nx2min=-.24\nx2max=-.01\nx3min=-.24\nx3max=-.01\n'
 if a.wormhole and not a.resume:
  if not a.coordinates:p.error('--wormhole requires exact actual native coordinate CSV')
  coordinates=a.coordinates.resolve();initial=out/'initial.dat';run('independent-initializer',[sys.executable,root/'tst/elliptic_shift_3d/wormhole_initial.py',coordinates,initial]);m['initializer']=json.loads(initial.with_suffix('.json').read_text());m['initializer']['sha256']=sha(initial)
  template=template.replace('pgen_name=z4c_composite_minkowski','pgen_name=z4c_composite_wormhole').replace('mass=0','mass=.3333333333333333333').replace('composite_checkpoint_tag=native-composite-minkowski-v1','composite_checkpoint_tag=native-static-smr-wormhole-v1')
  template=template.replace('composite_initial_angular=false','composite_initial_angular=false\ncomposite_initial_file='+str(initial)+'\ncomposite_positive_storage=2\ncomposite_telegraph_width=.24')
 input=out/'native_static.athinput'
 if not a.resume:input.write_text(template);m['input_sha256']=sha(input)
 else:assert sha(input)==m['input_sha256']
 save()
 full=out/'full'
 if not a.resume:run('full',[binary,'-i',input,'-d',full],folder=full)
 trace=[json.loads(line) for line in (full/(prefix+'.jsonl')).read_text().splitlines()];m['trace']=trace;save();assert [r['cycle'] for r in trace]==[0,1,2] and trace[-1]['native_update_calls']==8
 assert trace[-1]['cells']==12312
 receipt=(full/(prefix+'.mesh.dat')).read_text().split();assert receipt[0]=='NATIVE_STATIC_COMPOSITE_MESH_V1';n,ng=int(receipt[1]),int(receipt[2]);count=int(receipt[5]);assert(n,ng,count)==(6,4,15)
 if a.wormhole:
  hidden=out/'initial.hidden-after-fresh-run.dat'
  if not a.resume:
   assert (full/(prefix+'.coordinates.csv')).read_bytes()==coordinates.read_bytes();initial.rename(hidden);m['initializer']['hidden_before_restart']=True;save()
 mid=full/'rst'/(prefix+'.00001.rst');again=out/'restart'
 if not a.resume:run('restart',[binary,'-r',mid,'-d',again],folder=again)
 comparisons={}
 for suffix in ['.csv','.scri.csv']:
  x=full/(prefix+'.00002'+suffix);y=again/x.name;comparisons[suffix]={'bitwise':x.read_bytes()==y.read_bytes(),'full_sha256':sha(x),'restart_sha256':sha(y)}
 x=full/'rst'/(prefix+'.00002.rst.composite');y=again/'rst'/x.name;comparisons['complete-sidecar']={'bitwise':x.read_bytes()==y.read_bytes(),'full_sha256':sha(x),'restart_sha256':sha(y)}
 from analyze_native_static_smr import sidecar,restart_compare
 m['restart_payload']=restart_compare(sidecar(x),sidecar(y));m['restart']=comparisons;save();assert comparisons['.csv']['bitwise'] and comparisons['.scri.csv']['bitwise']
 # Compile against precisely the frozen engine/native helper, while root may
 # continue disjoint engine work. The actual Driver receipt supplies RK arrays.
 source=root/'tst/elliptic_shift_3d/native_static_reference.cpp';(out/source.name).write_bytes(source.read_bytes());flags=[]
 for line in (root/'build-shift-3d/CMakeFiles/elliptic_shift_3d_native_fields_tests.dir/flags.make').read_text().splitlines():
  if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):flags.extend(shlex.split(line.split('=',1)[1]))
 flags=build_manifest.get('frozen_compile_flags',['-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*flags]);reference=out/'reference-binary';probe=['/usr/bin/c++',*flags,'-M',str(out/source.name)];dep=subprocess.run(probe,capture_output=True,text=True,timeout=180);(out/'reference-dependencies.stdout').write_text(dep.stdout);(out/'reference-dependencies.stderr').write_text(dep.stderr);dep.check_returncode();dependencies=[Path(x).resolve() for x in shlex.split(dep.stdout.replace('\\\n',''))[1:]];leaked=[str(p) for p in dependencies if p.is_relative_to(root) and not p.is_relative_to(frozen) and not p.is_relative_to(build/'project-headers') and p!=out/source.name];m['reference_dependency_probe']={'command':probe,'live_project_headers':leaked};save();assert not leaked;command=['/usr/bin/c++',*flags,str(out/source.name),'-o',str(reference),*map(str,(root/'build-shift-3d/kokkos').glob('*/src/libkokkos*.a'))];run('compile-reference',command)
 dts=out/'dts.dat';dts.write_text(''.join(format(trace[i+1]['time']-trace[i]['time'],'.17g')+'\n' for i in range(2)));args=[reference,full/(prefix+'.mesh.dat'),full/(prefix+'.rk.dat'),dts,out/'reference',int(a.wormhole),0,0,1e-11]
 if a.wormhole:args.append(hidden)
 run('matched-actual-RK',args)
 matched={}
 for cycle in range(3):
  for surface in [False,True]:
   suffix='.scri.csv' if surface else '.csv';actual=np.genfromtxt(full/f'{prefix}.{cycle:05d}{suffix}',delimiter=',',names=True);expected=np.genfromtxt(out/f'reference.{cycle:05d}{suffix}',delimiter=',',names=True);fields=[f'state_{f}' for f in range(22)]+[f'delta_beta_{i}' for i in range(3)];errors={}
   for field in fields:
    value=actual[field] if field in actual.dtype.names else actual['beta_'+field[-1].replace('0','x').replace('1','y').replace('2','z')]+actual['xyz'[int(field[-1])]]
    # Minkowski CSV stores physical beta=-x+delta. Use the canonical sidecar
    # below for bitwise shift controls; this reporting conversion can round.
    errors[field]=float(np.max(abs(value-expected[field])))
   matched[f'{cycle}{suffix}']={'cells':len(actual),'field_errors':errors,'maximum_error':max(errors.values())}
 m['matched_actual_RK']=matched;save()
 # Identical state/scri coefficients are stronger than rounded physical-beta
 # reconstruction; only the latter can differ through subtraction roundoff.
 assert all(all(err==0 for field,err in row['field_errors'].items() if field.startswith('state_')) for row in matched.values())
 run('independent-native-analysis',[sys.executable,root/'tst/elliptic_shift_3d/analyze_native_static_smr.py',out]);m['independent_analysis']=json.loads((out/'independent-analysis.json').read_text());save()
 run('reject-adaptive',[binary,'-i',input,'mesh_refinement/refinement=adaptive','-d',out/'reject-adaptive'],False)
 run('reject-odd-ghost-SMR',[binary,'-i',input,'mesh/nghost=3','-d',out/'reject-odd-ghost-SMR'],False)
 m['binary_unchanged']=sha(binary)==m['binary_sha256'];m['input_unchanged']=sha(input)==m['input_sha256'];m['scientific_acceptance']=False;save();assert m['binary_unchanged'] and m['input_unchanged'];print(json.dumps({'restart_bitwise':True,'actual_static_cells':12312,'scientific_acceptance':False}))
if __name__=='__main__':main()
