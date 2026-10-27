#!/usr/bin/env python3
"""Frozen, bounded finite-sphere mass extraction; no evolution/solver call."""
import argparse,gzip,hashlib,json,os,resource,shlex,shutil,signal,subprocess,sys,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot-only',action='store_true');a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();cap=a.capture.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();capture=json.loads((cap/'manifest.json').read_text());(out/'.gitignore').write_text('/source/\n/extract\n/final.bin\n/initial.dat\n/*.csv\n');m=dict(scope='FullCartesian coordinate-sphere Hawkingmass fromactualacceptedN10 t=.0002, no radialaveraging/solve/evolution; same3Dinterpolation initial/Minkcontrols',source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),mass_kernel_commit='8004c20118ac561715c613fad670307b5d0c52ed',capture_manifest_sha256=sha(cap/'manifest.json'),capture_source_archive_sha256=capture['compiled_source_sha256'],source_tag=capture['io_binary_sha256'],budget=dict(one_sequential_worker=True,maximum_seconds_per_process=180,maximum_volume_cells=50000,maximum_query_points_per_process=4096),runs={})
def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
def run(name,cmd):
 row=dict(command=list(map(str,cmd)),maximum_seconds=180);start=time.monotonic()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  child=subprocess.Popen(row['command'],stdout=so,stderr=se,start_new_session=True);row['owned_process_group']=child.pid;m['runs'][name]=row;save()
  try:row['exit_code']=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
   row.update(exit_code=124,owned_process_reaped=True)
 row['seconds']=time.monotonic()-start;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True);return row['exit_code']
flag=root/'build-shift-3d/CMakeFiles/elliptic_shift_3d_coupled_pilot.dir/flags.make';flags=[]
for line in flag.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):flags+=shlex.split(line.split('=',1)[1])
driver=root/'tst/elliptic_shift_3d/extract_surface_mass.cpp';assert run('dependency-discovery',['/usr/bin/c++',*flags,'-M',driver])==0;deps=[Path(x).resolve() for x in (out/'dependency-discovery.stdout').read_text().replace('\\\n',' ').split(':',1)[1].split()];project=sorted(set(x for x in deps if root in x.parents));external=sorted(set(x for x in deps if root not in x.parents));link=shlex.split((flag.parent/'link.txt').read_text());libs=[(root/'build-shift-3d'/x).resolve() for x in link if x.endswith('.a')];extra=[Path(__file__).resolve(),root/'LICENSE',root/'kokkos/LICENSE',root/'tst/elliptic_shift_3d/prepare_surface_mass_reference.py',root/'tst/elliptic_shift_3d/compare_coupled_family.py',root/'build-shift-3d/libelliptic_shift_3d_radial_kernel.dylib',*root.glob('tst/elliptic_shift_1d/*.py')];files=sorted(set(project+libs+extra));m['copied_sources']={str(f.relative_to(root)):sha(f) for f in files};m['external_dependencies_before']={str(f):sha(f) for f in external};save()
for f in files:
 target=source/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
ref=root/'docs/elliptic-shift-3d/evidence/selected-taper-pilot-20261027';private_ref=source/'reference';private_ref.mkdir();m['reference_inputs']={}
for name in ['radial-n1025.npz','radial-n2049.npz','reference-parameters.json']:
 shutil.copyfile(ref/name,private_ref/name);m['reference_inputs'][name]=sha(ref/name)
assert sha(source/'build-shift-3d/libelliptic_shift_3d_radial_kernel.dylib')==json.loads((private_ref/'reference-parameters.json').read_text())['library_sha256']
assert run('prepare-reference',[sys.executable,source/'tst/elliptic_shift_3d/prepare_surface_mass_reference.py','--output',out/'reference','--reference',private_ref,'--library',source/'build-shift-3d/libelliptic_shift_3d_radial_kernel.dylib'])==0
frozenflags=[x.replace(str(root),str(source)) for x in flags];assert run('build',['/usr/bin/c++',*frozenflags,source/driver.relative_to(root),'-o',out/'extract',*[source/x.relative_to(root) for x in libs]])==0;assert run('frozen-dependencies',['/usr/bin/c++',*frozenflags,'-M',source/driver.relative_to(root)])==0;paths=(out/'frozen-dependencies.stdout').read_text().replace('\\\n',' ').split(':',1)[1].split();m['live_project_paths']=[x for x in paths if x.startswith(str(root)+'/') and not x.startswith(str(source)+'/')];assert not m['live_project_paths'];m['binary_sha256']=sha(out/'extract')
(out/'final.bin').write_bytes(gzip.open(cap/'n10/coupled.checkpoint.final.gz','rb').read());initial=root/'docs/elliptic-shift-3d/evidence/related-scri-family-20261027/n10/initial.dat.gz';(out/'initial.dat').write_bytes(gzip.open(initial,'rb').read());m['final_snapshot_sha256']=sha(out/'final.bin');m['initial_sha256']=sha(out/'initial.dat');save()
for points in ([128] if a.pilot_only else [128,512,2048]):
 for mode in (['mink','analytic_initial','initial','final'] if points==128 else ['mink','initial','final']):
  for begin,end in ([(0,8)] if points<=512 else [(0,2),(2,4),(4,6),(6,8)]):
   name=f'{mode}-{points}-{begin}-{end}';cmd=[out/'extract',out/'final.bin',m['source_tag'],out/'initial.dat',out/'reference'/f'quadrature-{points}.dat',out/'reference/initial-oracle.dat',mode,out/(name+'.csv'),begin,end];code=run(name,cmd)
   if (out/(name+'.stdout')).stat().st_size:
    try:m['runs'][name]['result']=json.loads((out/(name+'.stdout')).read_text())
    except json.JSONDecodeError:pass
   if code in [124,3]:save();raise SystemExit(code)
   if code not in [0,2]:save();raise SystemExit(code)
   if (out/(name+'.csv')).exists():
    raw=out/(name+'.csv');target=out/(name+'.csv.gz')
    with gzip.open(target,'wb') as z:z.write(raw.read_bytes())
    assert gzip.open(target,'rb').read()==raw.read_bytes();m['runs'][name]['raw_csv_sha256']=sha(raw);m['runs'][name]['gzip_sha256']=sha(target)
   save()
with tarfile.open(out/'compiled-source.tar.gz','w:gz') as archive:
 for f in sorted(source.rglob('*')):
  if f.is_file():archive.add(f,arcname=str(f.relative_to(source)))
m['compiled_source_sha256']=sha(out/'compiled-source.tar.gz');m['external_dependencies_unchanged']=all(sha(Path(f))==h for f,h in m['external_dependencies_before'].items());m['private_sources_unchanged']=all(sha(source/f)==h for f,h in m['copied_sources'].items());save()
