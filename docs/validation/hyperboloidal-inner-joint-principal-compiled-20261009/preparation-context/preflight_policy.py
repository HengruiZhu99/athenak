"""Saved-file UTF-8/archive-policy preflight only; no collector/science execution."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
c=json.loads((P/'scope-config.json').read_text())
reject_suffix={'.rst','.bin','.npy','.npz','.jsonl','.o','.obj','.a','.so','.dylib','.dll','.exe','.pyc','.pyo','.h5','.hdf5','.pkl','.pickle'}
binary4={b'\x7fELF',b'\xfe\xed\xfa\xce',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xcf\xfa\xed\xfe',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca'}
files=[]
for r in c['roots']:files.extend(q for q in Path(r['source']).rglob('*') if q.is_file())
files.extend(Path(r['source']) for r in c['selected_files'])
rows=[];copied=0
for q in sorted(set(files)):
 b=q.read_bytes();m=b[:8];reason=None
 if len(b)>1048576:reason='size'
 elif q.suffix.lower() in reject_suffix:reason='suffix'
 elif m[:4] in binary4 or m.startswith(b'!<arch>') or m.startswith(b'\x93NUMPY') or m.startswith(b'\x89HDF'):reason='magic'
 if reason is None:
  b.decode('utf-8');copied+=1
  if m.startswith((b'PK\x03\x04',b'PK\x05\x06',b'PK\x07\x08',b'\x1f\x8b',b'BZh',b'\xfd7zXZ',b'7z\xbc\xaf\x27\x1c')) or b[257:262]==b'ustar':raise ValueError('archive magic '+str(q))
  if q.suffix=='.json':json.loads(b,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
 rows.append({'source':str(q.resolve()),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'metadata_only_reason':reason})
out={'passed':True,'source_only':True,'files':len(rows),'eligible_utf8':copied,'omitted_binary_or_large':len(rows)-copied,'rows':rows}
(P/'policy-preflight.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
