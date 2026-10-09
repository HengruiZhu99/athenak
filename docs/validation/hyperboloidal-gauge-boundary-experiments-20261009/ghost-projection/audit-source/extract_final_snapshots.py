"""Retain exact last-valid κ10 native payloads and quantify original/projected ghost preparation."""
from pathlib import Path
import hashlib,importlib.util,json,struct,subprocess
import numpy as np
root=Path(__file__).resolve().parents[3];work=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reader',root/'vis/python/bin_convert.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
fields=['chi','gxx','gxy','gxz','gyy','gyz','gzz','Khat','Axx','Axy','Axz','Ayy','Ayz','Azz','Gamx','Gamy','Gamz','Theta','alpha','betax','betay','betaz','Bx','By','Bz'];records=[]
for label,directory,index in [('control',root/'build-layer-research/clean-wide-kappa10-long/finite-angular-N24',37),('projected',work/'native-kappa10-t2/finite-angular-N24',74)]:
 checkpoint=directory/'rst'/f'hyp.{index:05d}.rst';binary=directory/'bin'/f'hyp.z4c.{index:05d}.bin';contents=checkpoint.read_bytes();expected=25*30**3*8;offset=len(contents)-expected;assert struct.unpack('<Q',contents[offset-8:offset])[0]==expected
 value=np.frombuffer(contents,dtype='<f8',offset=offset).reshape(25,30,30,30);data=reader.read_binary(str(binary));mask=data['mb_data']['z4c_active'][0].astype(bool)
 for f,name in enumerate(fields):assert np.array_equal(value[f].astype(np.float32)[mask],data['mb_data']['z4c_'+name][0][mask]),name
 payload=work/f'last-valid-kappa10-{label}.double';payload.write_bytes(contents[offset:]);record={'label':label,'time':data['time'],'cycle':data['cycle'],'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(contents).hexdigest(),'payload':str(payload),'payload_sha256':hashlib.sha256(payload.read_bytes()).hexdigest(),'every_active_field_float32_roundtrip_exact':True,'cases':[]}
 for exe in ['quantify-original','quantify-projected']:
  prefix=work/f'last-valid-{label}-{exe}';cmd=[str(work/exe),'2',str(payload),str(prefix)];done=subprocess.run(cmd,capture_output=True,text=True);Path(str(prefix)+'.jsonl').write_text(done.stdout);Path(str(prefix)+'.stderr').write_text(done.stderr);case={'command':cmd,'exit_status':done.returncode,'stdout':done.stdout,'stderr':done.stderr}
  if done.returncode==0:case['measurements']=json.loads(done.stdout)
  record['cases'].append(case)
 records.append(record)
(work/'last-valid-ghost-audit.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records:
 print(r['label'],r['time'],[(c['command'][0].split('/')[-1],c['exit_status'],c.get('measurements',{}).get('invalid_ghost_SPD_count'),c.get('measurements',{}).get('min_ghost_determinant')) for c in r['cases']])
