#!/usr/bin/env python3
"""Freeze IO-only final-state capture; tiny Minkowski then one identical N10.
No existing project source/CMake/binary is edited, no PDE or RK arithmetic change.
"""
import argparse,difflib,gzip,hashlib,json,os,resource,signal,subprocess,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--family',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();parent=a.family.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((parent/'manifest.json').read_text());assert sha(parent/'compiled-source.tar.gz')==m['source_archive_sha256'];(out/'.gitignore').write_text('/source/\n/pilot-io\n/pilot-original\n/diagnostic\n/*/*.csv\n/*/*.dat\n/*/*.checkpoint\n/*/*.checkpoint.final\n')
with tarfile.open(parent/'compiled-source.tar.gz') as archive:
 for f in archive.getmembers():
  target=(source/f.name).resolve()
  if f.isdir():continue
  if not f.isfile() or not target.is_relative_to(source):raise RuntimeError('unsafe archive member')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(f).read())
path=source/'tst/elliptic_shift_3d/coupled.cpp';old=path.read_text();start=old.index('        cubed_sphere::CompositeSnapshot snapshot{');end=old.index('        cubed_sphere::CompositeCheckpoint::Write(checkpoint, snapshot);',start)+len('        cubed_sphere::CompositeCheckpoint::Write(checkpoint, snapshot);');body=old[start:end].replace('static_cast<std::uint64_t>(i+1)','static_cast<std::uint64_t>(steps)').replace('Write(checkpoint, snapshot)','Write(checkpoint+".final", snapshot)');marker='    pilot.RHS(pilot.state, dt);  // accepted-state diagnostics, with a fresh stage solve\n';assert old.count(marker)==1;new=old.replace(marker,marker+'    // IO ONLY: retain the exact accepted state AFTER its fresh shift solve.\n    if (!checkpoint.empty()) {\n'+body+'\n    }\n');assert new.replace(marker+'    // IO ONLY: retain the exact accepted state AFTER its fresh shift solve.\n    if (!checkpoint.empty()) {\n'+body+'\n    }\n',marker)==old;path.write_text(new);(out/'io-only.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='frozen/coupled.cpp',tofile='io-capture/coupled.cpp')))
driver=root/'tst/elliptic_shift_3d/diagnose_final_scri_state.cpp';diag=source/driver.relative_to(root);diag.write_bytes(driver.read_bytes());original=out/'pilot-original';original.write_bytes(gzip.open(parent/'pilot.gz','rb').read());original.chmod(0o755);assert sha(original)==m['binary_sha256'];report=dict(scope='IO-only capture after final accepted fresh RHS; tinyMink control before ONE identicalN10case, no PDE/RK/sampler/tolerance changes',parent_manifest_sha256=sha(parent/'manifest.json'),original_binary_sha256=sha(original),original_compiled_source_sha256=sha(parent/'compiled-source.tar.gz'),io_patch_sha256=sha(out/'io-only.patch'),driver_sha256=sha(driver),runs={},controls={})
def save():(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
def run(name,cmd):
 start=time.monotonic();row=dict(command=list(map(str,cmd)),maximum_seconds=180)
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  child=subprocess.Popen(row['command'],stdout=so,stderr=se,start_new_session=True);row['owned_process_group']=child.pid;report['runs'][name]=row;save()
  try:row['exit_code']=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
   row.update(exit_code=124,owned_process_reaped=True)
 row['seconds']=time.monotonic()-start;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True);return row['exit_code']
command=[x.replace(str(parent/'source'),str(source)).replace(str(parent/'pilot'),str(out/'pilot-io')) for x in m['build']['frozen-build']['command']];assert run('io-build',command)==0
idx=command.index(str(path));assert run('io-dependencies',command[:idx]+['-M',str(path)])==0;command2=command.copy();command2[idx]=str(diag);command2[command2.index('-o')+1]=str(out/'diagnostic');assert run('diagnostic-build',command2)==0;assert run('diagnostic-dependencies',command2[:idx]+['-M',str(diag)])==0
report['io_binary_sha256']=sha(out/'pilot-io');tag=report['io_binary_sha256'];save()
for name,binary in [('mink-original',original),('mink-io',out/'pilot-io')]:
 folder=out/name;folder.mkdir();cmd=[binary,6,0,'.0001',2,0,2,'1e-10',folder/'coupled.csv',folder/'coupled.checkpoint',sha(binary),4,1,4,8,0,'',1,0,'.65,1',7,1,0,'.24',0,1,1,0,0,'.24',0,1];assert run(name,cmd)==0
for suffix in ['', '.spheres.csv']:
 left=out/'mink-original'/('coupled.csv'+suffix);right=out/'mink-io'/left.name;assert left.read_bytes()==right.read_bytes();report['controls']['Mink'+suffix]=dict(bitwise=True,sha256=sha(left))
assert run('mink-final-certificate',[out/'diagnostic',out/'mink-io/coupled.checkpoint.final',tag,out/'mink-io/poles.csv'])==0;report['controls']['Mink_certificate']=json.loads((out/'mink-final-certificate.stdout').read_text());assert report['controls']['Mink_certificate']['saved_beta_original_physical_rms']<=1e-10;assert report['controls']['Mink_certificate']['helper_full_tensor_agreement']<1e-12;save()
folder=out/'n10';folder.mkdir();initial=folder/'initial.dat';initial.write_bytes(gzip.open(parent/'n10/initial.dat.gz','rb').read());assert sha(initial)==m['cases']['10']['initial_sha256'];cmd=m['cases']['10']['commands']['n10-coupled']['command'].copy();cmd[0]=str(out/'pilot-io');cmd[8]=str(folder/'coupled.csv');cmd[9]=str(folder/'coupled.checkpoint');cmd[10]=tag;cmd[16]=str(initial);code=run('n10-io-capture',cmd);assert code in [0,1] and (folder/'coupled.checkpoint.final').exists(), 'N10 capture failed; preserve evidence, no rerun'
report['N10_result']=json.loads((out/'n10-io-capture.stdout').read_text());report['N10_scientific_acceptance']=False
for suffix in ['', '.spheres.csv']:
 baseline=parent/'n10'/('coupled.csv'+suffix+'.gz');current=folder/('coupled.csv'+suffix);assert gzip.open(baseline,'rb').read()==current.read_bytes();report['controls']['N10'+suffix]=dict(bitwise=True,baseline_gzip_sha256=sha(baseline),current_sha256=sha(current))
save();assert run('n10-final-certificate',[out/'diagnostic',folder/'coupled.checkpoint.final',tag,folder/'poles.csv'])==0;report['N10_certificate']=json.loads((out/'n10-final-certificate.stdout').read_text());assert report['N10_certificate']['saved_beta_original_physical_rms']<=1e-10 and report['N10_certificate']['saved_beta_original_conformal_rms']<=1e-10 and report['N10_certificate']['saved_beta_original_point_max']<=1e-7
for name in ['io-dependencies','diagnostic-dependencies']:
 paths=(out/(name+'.stdout')).read_text().replace('\\\n',' ').split(':',1)[1].split();live=[x for x in paths if x.startswith(str(root)+'/') and not x.startswith(str(source)+'/')];assert not live;report[name+'-live-project-paths']=live
with tarfile.open(out/'compiled-source.tar.gz','w:gz') as archive:
 for f in sorted(source.rglob('*')):
  if f.is_file():archive.add(f,arcname=str(f.relative_to(source)))
report['compiled_source_sha256']=sha(out/'compiled-source.tar.gz');report['output_gzip']={}
for folder in [out/'mink-original',out/'mink-io',out/'n10']:
 for f in folder.iterdir():
  if f.is_file() and f.suffix!='.gz':
   target=f.with_name(f.name+'.gz')
   with gzip.open(target,'wb') as z:z.write(f.read_bytes())
   assert gzip.open(target,'rb').read()==f.read_bytes();report['output_gzip'][str(target.relative_to(out))]=dict(sha256=sha(target),raw_sha256=sha(f),lossless=True)
report['external_dependencies_unchanged']=all(sha(Path(x))==h for x,h in m['external_dependencies_before'].items());save()
