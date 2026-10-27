#!/usr/bin/env python3
"""Checksummed topology edits must reject at the actual native restore guards."""
import argparse,hashlib,json,os,resource,signal,struct,subprocess,time
from pathlib import Path
from analyze_native_static_smr import sidecar
from analyze_native_wormhole import checksum

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();controls=a.controls.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);m0=json.loads((controls/'manifest.json').read_text());binary=Path(m0['cases'][0]['command'][0])
 # The first case can be the Python initial-data generator in wormhole controls.
 binary=next(Path(r['command'][0]) for r in m0['cases'] if r['name']=='full');rst=controls/'full/rst/native_smr.00001.rst';source=Path(str(rst)+'.composite');layout=sidecar(source);original=source.read_bytes();m={'binary_sha256':sha(binary),'native_rst_sha256':sha(rst),'original_sidecar_sha256':sha(source),'maximum_seconds_per_process':180,'cases':[]}
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 for name,key,fmt,delta,reason in [('count','native_count','i',-1,'core topology/count mismatch'),('map','composite','i',1,'core pack map mismatch'),('level','level','i',1,'core pack map mismatch'),('bounds','bounds','d',1e-5,'core bounds mismatch')]:
  data=bytearray(original);offset=layout['offsets'][key];value=struct.unpack_from('='+fmt,data,offset)[0];struct.pack_into('='+fmt,data,offset,value+delta);struct.pack_into('=Q',data,len(data)-8,checksum(data[:-8]));bad=out/(name+'.composite');bad.write_bytes(data);folder=out/name;folder.mkdir();command=[str(binary),'-r',str(rst),'z4c/composite_restart_file='+str(bad),'-d',str(folder)];row={'name':name,'command':command,'checksummed_edit_offset':offset,'old':value,'new':value+delta,'expected_reason':reason};m['cases'].append(row);save();t=time.monotonic()
  with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:
   process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,start_new_session=True);row['owned_group']=process.pid
   try:row['exit_code']=process.wait(timeout=180)
   except subprocess.TimeoutExpired:
    row['exit_code']=124
    if os.getpgid(process.pid)!=process.pid:raise RuntimeError('unowned group')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
   row['seconds']=time.monotonic()-t;row['reason_observed']=reason in (out/(name+'.stderr')).read_text();save();assert row['exit_code'] not in [0,124] and row['reason_observed'];print(name,row['exit_code'],round(row['seconds'],3),flush=True)
 # Default uniform ng3 branch must retain its established numerical output.
 uniform=out/'uniform-ng3';uniform.mkdir();input=root/'inputs/z4c/composite_minkowski.athinput';command=[str(binary),'-i',str(input),'-d',str(uniform)];t=time.monotonic()
 with (out/'uniform.stdout').open('w') as stdout,(out/'uniform.stderr').open('w') as stderr:
  result=subprocess.run(command,cwd=root,stdout=stdout,stderr=stderr,timeout=180)
 m['uniform_regression']={'command':command,'exit_code':result.returncode,'seconds':time.monotonic()-t,'input_sha256':sha(input),'files':{}};save();assert result.returncode==0
 baseline=root/'docs/elliptic-shift-3d/evidence/native-composite-cfl-controls-20261027/two-shell'
 for old in sorted(baseline.glob('*.csv')):
  new=uniform/old.name;m['uniform_regression']['files'][old.name]={'bitwise':old.read_bytes()==new.read_bytes(),'baseline_sha256':sha(old),'new_sha256':sha(new)}
 save();assert len(m['uniform_regression']['files'])==6 and all(r['bitwise'] for r in m['uniform_regression']['files'].values());print('uniform-ng3',result.returncode,round(m['uniform_regression']['seconds'],3),'6CSVbitwise',flush=True)
 m['original_inputs_unchanged']=sha(rst)==m['native_rst_sha256'] and sha(source)==m['original_sidecar_sha256'] and sha(binary)==m['binary_sha256'];save();assert m['original_inputs_unchanged']
if __name__=='__main__':main()
