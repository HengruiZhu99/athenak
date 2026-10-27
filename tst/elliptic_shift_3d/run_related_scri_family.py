#!/usr/bin/env python3
"""Frozen classical-RK4 research-engine related N6/N8/N10 scri controls.
No production/shared source, CMake, binary, or original experiment is modified.
"""
import argparse,gzip,hashlib,json,os,resource,shlex,shutil,signal,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,nargs='+',default=[6,8,10]);a=p.parse_args();assert a.cases==sorted(set(a.cases)) and set(a.cases)<={6,8,10};resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir();(out/'.gitignore').write_text('/source/\n/pilot\n/*/coordinates.csv\n/*/initial.dat\n/*/coupled.csv\n/*/coupled.csv.spheres.csv\n/*/coupled.checkpoint\n')
 m=dict(scope='Actual full3D classical-RK4 research engine; not native Driver tasks; fixed uniform eight Cartesian blocks plus7shells, no Cartesian AMR/cache; no altered numerical/scientific tolerance',source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),budget=dict(one_sequential_CPU=True,maximum_seconds_per_command=180,active_cells_max=50000),controls=dict(n=a.cases,radii=[.45,.55,.60,.65,.70,.75,1],dt=.0001,steps=2,time=.0002,tolerance=1e-10,spatial_order=4,positive_storage=2,telegraph_width=.24,bulk_subdivisions=2,refinement=0,radial_interface_stencils=True,radial_line_smoothing=True,topology_coarsening=True,log_halo_cache=False,puncture_scheme=1,boundary_scheme=1,surface_nodes=7),build={},cases={})
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 def run(group,name,cmd,maximum=180):
  row=dict(command=list(map(str,cmd)),maximum_seconds=maximum);group[name]=row;save();start=time.monotonic()
  with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
   child=subprocess.Popen(row['command'],cwd=root,stdout=so,stderr=se,start_new_session=True);row['owned_process_group']=child.pid;save()
   try:row['exit_code']=child.wait(timeout=maximum)
   except subprocess.TimeoutExpired:
    assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
    row.update(exit_code=124,owned_process_reaped=True,cleanup='only this newly owned process group terminated')
  row['seconds']=time.monotonic()-start;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True);return row['exit_code']
 flags=[];flagfile=root/'build-shift-3d/CMakeFiles/elliptic_shift_3d_coupled_pilot.dir/flags.make'
 for line in flagfile.read_text().splitlines():
  if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):flags+=shlex.split(line.split('=',1)[1])
 compiler='/usr/bin/c++';input=root/'tst/elliptic_shift_3d/coupled.cpp';assert run(m['build'],'dependency-discovery',[compiler,*flags,'-M',input])==0
 deps=[Path(x).resolve() for x in (out/'dependency-discovery.stdout').read_text().replace('\\\n',' ').split(':',1)[1].split()];projectdeps=sorted(set(x for x in deps if root in x.parents));externaldeps=sorted(set(x for x in deps if root not in x.parents))
 link=shlex.split((flagfile.parent/'link.txt').read_text());libraries=[(root/'build-shift-3d'/x).resolve() for x in link if x.endswith('.a')]
 extra=[Path(__file__).resolve(),root/'tst/elliptic_shift_3d/wormhole_initial.py',root/'tst/elliptic_shift_1d/wormhole_isotropic.py',root/'tst/elliptic_shift_1d/wormhole_data.py',root/'LICENSE',flagfile,flagfile.parent/'link.txt']
 files=sorted(set(projectdeps+libraries+extra));before={str(x.relative_to(root)):sha(x) for x in files};m['copied_source_hashes']=before;m['external_dependencies_before']={str(x):sha(x) for x in externaldeps};save()
 for f in files:
  target=source/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
 frozenflags=[x.replace(str(root),str(source)) for x in flags];binary=out/'pilot';cmd=[compiler,*frozenflags,str(source/input.relative_to(root)),'-o',str(binary),*[str(source/x.relative_to(root)) for x in libraries]];assert run(m['build'],'frozen-build',cmd)==0;assert run(m['build'],'frozen-dependencies',[compiler,*frozenflags,'-M',source/input.relative_to(root)])==0
 deptext=(out/'frozen-dependencies.stdout').read_text().replace('\\\n',' ').split(':',1)[1].split();live=[x for x in deptext if x.startswith(str(root)+'/') and not x.startswith(str(source)+'/')];assert not live,live;m['live_project_dependencies']=live;m['binary_sha256']=sha(binary);tag=m['binary_sha256'];m['copied_inputs_stable_during_build']=all(sha(root/f)==h for f,h in before.items());assert m['copied_inputs_stable_during_build'];save()
 with tarfile.open(out/'compiled-source.tar.gz','w:gz') as archive:
  for f in sorted(source.rglob('*')):
   if f.is_file():archive.add(f,arcname=str(f.relative_to(source)))
 m['source_archive_sha256']=sha(out/'compiled-source.tar.gz')
 with gzip.open(out/'pilot.gz','wb') as f:f.write(binary.read_bytes())
 save()
 for n in a.cases:
  case=out/f'n{n}';case.mkdir();row=dict(n=n,cells=50*n**3,commands={},scientific_acceptance=False);m['cases'][str(n)]=row;save();coords=case/'coordinates.csv';initial=case/'initial.dat';assert run(row['commands'],f'n{n}-coordinates',[binary,'--coordinates',n,'.45,.55,.60,.65,.70,.75,1',2,0,coords])==0;q=np.genfromtxt(coords,delimiter=',',names=True);assert len(q)==50*n**3 and np.array_equal(q['id'],np.arange(len(q)));tuples=np.column_stack([q[x] for x in ['x','y','z','r']]);row['coordinate_sha256']=sha(coords);row['coordinate_tuple_sha256']=hashlib.sha256(tuples.tobytes()).hexdigest()
  old=root/'docs/elliptic-shift-3d/evidence/native-accuracy-controls-20261027'/({6:'dt_half',8:'n8',10:'n10'}[n]);prior=old/'coordinates.csv';data=old/'initial.hidden-after-fresh-run.dat';matched=False
  if prior.exists() and data.exists():
   oldq=np.genfromtxt(prior,delimiter=',',names=True);oldtuples=np.column_stack([oldq[x] for x in ['x','y','z','r']]);matched=np.array_equal(tuples,oldtuples);row['prior_coordinate_check']=dict(path=str(prior),sha256=sha(prior),tuple_sha256=hashlib.sha256(oldtuples.tobytes()).hexdigest(),all_point_tuples_bitwise_equal=matched)
  if matched:
   values=np.loadtxt(data,skiprows=1);assert values.shape==(len(q),5) and np.array_equal(values[:,0],q['r']) and np.all(values[:,1:3]>0);shutil.copyfile(data,initial);row['initializer_reused']=dict(source=str(data),sha256=sha(data),provenance=str(old/'initial.json'),provenance_sha256=sha(old/'initial.json'))
  else:assert run(row['commands'],f'n{n}-initializer',[sys.executable,source/'tst/elliptic_shift_3d/wormhole_initial.py',coords,initial])==0
  row['initial_sha256']=sha(initial);save()
  cmd=[binary,n,0,'.0001',2,0,7,'1e-10',case/'coupled.csv',case/'coupled.checkpoint',tag,4,1,4,8,1,initial,2,0,'.45,.55,.60,.65,.70,.75,1',7,1,2,'.24',1,1,1,0,0,'.24',1,1];code=run(row['commands'],f'n{n}-coupled',cmd)
  trace=[json.loads(x) for x in (out/f'n{n}-coupled.stdout').read_text().splitlines() if x.startswith('{')];row['exit_code']=code
  if trace:
   result=trace[-1];assert result['time']==.0002 and result['n']==n and result['cells']==50*n**3 and result['steps']==2 and result['bulk_subdivisions']==2 and result['mesh_revision']==0 and not result['log_halo_enrichment'];row['result']=result;row['completed_with_bitwise_restart']=bool(result['restart_equal']);row['gates']={name:dict(value=result[name],target=target,passes=result[name]<=target) for name,target in [('shift_residual_max',1e-10),('shift_residual_point_max',1e-7),('scri_tensor_pole',1e-7),('scri_ghost_clock',1e-13),('scri_ghost_null',1e-12),('scri_ghost_tangent',1e-12),('H_rms',1e-5)]}
  else:row['failure_reason']=(out/f'n{n}-coupled.stderr').read_text()[-2000:];row['checkpoint_written']=(case/'coupled.checkpoint').exists()
  row['initial_unchanged']=sha(initial)==row['initial_sha256'];row['binary_unchanged']=sha(binary)==tag
  for f in [coords,initial,case/'coupled.csv',case/'coupled.csv.spheres.csv',case/'coupled.checkpoint']:
   if f.exists():
    with gzip.open(f.with_name(f.name+'.gz'),'wb') as z:z.write(f.read_bytes())
  save()
 m['external_dependencies_unchanged']=all(sha(Path(x))==h for x,h in m['external_dependencies_before'].items());m['private_capsule_unchanged']=all(sha(source/f)==h for f,h in before.items());save()
if __name__=='__main__':main()
