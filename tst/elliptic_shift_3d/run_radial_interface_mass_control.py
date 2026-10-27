#!/usr/bin/env python3
"""Initial/Mink mass controls only; actual-final sampling explicitly forbidden."""
import argparse,difflib,gzip,hashlib,json,os,resource,shutil,signal,subprocess,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();base=root/'docs/elliptic-shift-3d/evidence/coupled-surface-mass-20261027';mathcap=root/'docs/elliptic-shift-3d/evidence/radial-interface-jet-mms-20261027';out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((base/'manifest.json').read_text());mm=json.loads((mathcap/'manifest.json').read_text());assert sha(mathcap/'compiled-source.tar.gz')==mm['compiled_source_sha256'];assert all(r['exit_code']==0 for r in mm['runs'].values());(out/'.gitignore').write_text('/source/\n/extract\n/final.bin\n/initial.dat\n/*.csv\n')
with tarfile.open(mathcap/'compiled-source.tar.gz') as archive:
 for f in archive.getmembers():
  target=(source/f.name).resolve()
  if f.isdir():continue
  if not f.isfile() or not target.is_relative_to(source):raise RuntimeError('unsafe archive member')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(f).read())
original=source/'tst/elliptic_shift_3d/extract_surface_mass.cpp';old=original.read_text();new=old.replace('#include "mesh/cubed_sphere/composite_difference.hpp"','#include "mesh/cubed_sphere/composite_difference.hpp"\n#include "mesh/cubed_sphere/radial_interface_jet_probe.hpp"');marker='if(grid.active.size()>50000)throw std::runtime_error("bounded50000");';assert new.count(marker)==1;new=new.replace(marker,marker+'c::RadialInterfaceJetProbe radial_jets(grid);');new=new.replace('const std::string mode=argv[6];','const std::string mode=argv[6];if(mode=="final")throw std::runtime_error("actualfinal extraction forbidden before global qualification");');new=new.replace('grid.SampleJets(field,22,x,values,d,dd)','radial_jets.SampleJets(field,22,x,values,d,dd)');target=source/'tst/elliptic_shift_3d/extract_radial_interface_initial.cpp';target.write_text(new);(out/'sampling-only.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='owner-SampleJets.cpp',tofile='AD-global-radial-initial-only.cpp')))
command=[x.replace(str(base/'source'),str(source)).replace(str(base/'extract'),str(out/'extract')) for x in m['runs']['build']['command']];command=[str(target) if x.endswith('/tst/elliptic_shift_3d/extract_surface_mass.cpp') else x for x in command];idx=command.index(str(target));report=dict(scope='ONLYoriginalinitialM1/3andMinkfinite-sphere controls ofnewADglobalradialjets; finalmodecompiledtoREJECT, noPDEchange',mathematical_MMS_capsule_sha256=mm['compiled_source_sha256'],initial_mass_baseline_manifest_sha256=sha(base/'manifest.json'),sampling_patch_sha256=sha(out/'sampling-only.patch'),runs={})
def save():(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
def run(name,cmd):
 start=time.monotonic()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  child=subprocess.Popen(list(map(str,cmd)),stdout=so,stderr=se,start_new_session=True)
  try:code=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
   code=124
 report['runs'][name]=dict(command=list(map(str,cmd)),exit_code=code,seconds=time.monotonic()-start,maximum_seconds=180);save();print(name,code,round(report['runs'][name]['seconds'],3),flush=True);return code
assert run('build',command)==0;assert run('dependencies',command[:idx]+['-M',str(target)])==0
shutil.copyfile(base/'final.bin',out/'final.bin');shutil.copyfile(base/'initial.dat',out/'initial.dat');shutil.copytree(base/'reference',out/'reference');assert sha(out/'initial.dat')==m['initial_sha256'] and sha(out/'final.bin')==m['final_snapshot_sha256']
for points in [128,512,2048]:
 for mode in (['mink','analytic_initial','initial'] if points==128 else ['mink','initial']):
  for begin,end in ([(0,8)] if points<=512 else [(0,2),(2,4),(4,6),(6,8)]):
   name=f'{mode}-{points}-{begin}-{end}';cmd=[out/'extract',out/'final.bin',m['source_tag'],out/'initial.dat',out/'reference'/f'quadrature-{points}.dat',out/'reference/initial-oracle.dat',mode,out/(name+'.csv'),begin,end];code=run(name,cmd)
   if (out/(name+'.stdout')).stat().st_size:
    try:report['runs'][name]['result']=json.loads((out/(name+'.stdout')).read_text())
    except json.JSONDecodeError:pass
   if code not in [0,2]:save();raise SystemExit(code)
   raw=out/(name+'.csv')
   if raw.exists():
    target=out/(name+'.csv.gz')
    with gzip.open(target,'wb') as z:z.write(raw.read_bytes())
    assert gzip.open(target,'rb').read()==raw.read_bytes();report['runs'][name]['raw_csv_sha256']=sha(raw);report['runs'][name]['gzip_sha256']=sha(target)
   save()
# Explicit negative control prevents accidental use of the new sampler on finaldata.
cmd=[out/'extract',out/'final.bin',m['source_tag'],out/'initial.dat',out/'reference/quadrature-128.dat',out/'reference/initial-oracle.dat','final',out/'forbidden.csv'];assert run('forbidden-final',cmd)==2;report['forbidden_final_reason']=(out/'forbidden-final.stderr').read_text().strip()
paths=(out/'dependencies.stdout').read_text().replace('\\\n',' ').split(':',1)[1].split();report['live_project_paths']=[x for x in paths if x.startswith(str(root)+'/') and not x.startswith(str(source)+'/')];assert not report['live_project_paths']
with tarfile.open(out/'compiled-source.tar.gz','w:gz') as archive:
 for f in sorted(source.rglob('*')):
  if f.is_file():archive.add(f,arcname=str(f.relative_to(source)))
report['compiled_source_sha256']=sha(out/'compiled-source.tar.gz');save()
