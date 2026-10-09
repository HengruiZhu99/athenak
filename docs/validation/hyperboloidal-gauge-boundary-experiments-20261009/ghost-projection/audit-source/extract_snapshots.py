"""Extract exact double precision native vacuum one-block restart payloads.

Validates the trailer data_size and every active field against matching float32
binary output. It does not reconstruct geometry from ADM or round native values.
"""
import hashlib,importlib.util,json,struct
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[3]
work=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reader',root/'vis/python/bin_convert.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
fields=['chi','gxx','gxy','gxz','gyy','gyz','gzz','Khat','Axx','Axy','Axz','Ayy','Ayz','Azz','Gamx','Gamy','Gamz','Theta','alpha','betax','betay','betaz','Bx','By','Bz']
records=[]
for degree,base in [(2,'wide-transition-scan/native-kappa5-t0.5'),(4,'wide-degree4/native-kappa5-t0.5')]:
 directory=root/'build-layer-research/boundary'/base/'finite-angular-N24'
 for index in ([0,5,10,20] if degree==2 else [0,5,10]):
  checkpoint=directory/'rst'/f'hyp.{index:05d}.rst'
  binary=directory/'bin'/f'hyp.z4c.{index:05d}.bin'
  content=checkpoint.read_bytes();cells=30**3;expected=25*cells*8;offset=len(content)-expected
  assert struct.unpack('<Q',content[offset-8:offset])[0]==expected
  values=np.frombuffer(content,dtype='<f8',offset=offset).reshape(25,30,30,30)
  saved=reader.read_binary(str(binary));data=saved['mb_data'];mask=data['z4c_active'][0].astype(bool)
  maximum=0
  for f,name in enumerate(fields):
   actual=np.asarray(data['z4c_'+name])[0]
   rounded=values[f].astype(np.float32)
   assert np.array_equal(rounded[mask],actual[mask]),name
   maximum=max(maximum,float(np.max(abs(values[f][mask]-actual[mask]))))
  output=work/f'native-degree{degree}-output{index:05d}.double'
  output.write_bytes(content[offset:])
  records.append({'degree':degree,'time':saved['time'],'cycle':saved['cycle'],'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(content).hexdigest(),'payload_offset':offset,'payload_size':expected,'output':str(output),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'all_active_fields_float32_roundtrip_exact':True,'max_double_minus_float32':maximum})
(work/'snapshot-extraction.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
