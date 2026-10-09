"""Independent byte/payload QA of the newly completed compact capsule."""
import hashlib,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
D=R/'docs/validation/hyperboloidal-inner-joint-principal-compiled-20261009'
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=load(D/'catalog.json');receipt=load(D/'collection-receipt.json');assert receipt['completed'] is True
assert receipt['inputs_unchanged'] and receipt['source_inventories_unchanged'] and receipt['production_unchanged']
for rel,x in c['files'].items():
 p=D/rel;assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes']
 assert p.read_bytes()==Path(x['source']).read_bytes()
assert sha(D/'README.md')==c['README_sha256']
assert sha(D/'production-source-identity.json')==c['production_identity_sha256']
assert sha(D/'external-current-documents-metadata.json')==c['external_current_documents_metadata_sha256']
assert len(c['omitted_large_payloads'])==3
metadata=[]
for part in c['external_dependency_parts']:metadata+=load(D/part)['inputs']
assert len(metadata)==1413
for x in metadata:
 assert sha(x['source'])==x['sha256'] and Path(x['source']).stat().st_size==x['bytes']
files=sorted(p for p in D.rglob('*') if p.is_file());finite=0
for p in files:
 b=p.read_bytes();assert len(b)<=1048576
 assert p.suffix.lower() not in {'.npz','.npy','.jsonl','.o','.obj','.a','.exe','.dylib','.so','.dll','.bin','.rst','.pyc','.pyo'}
 assert not b.startswith((b'\x7fELF',b'\xfe\xed\xfa',b'\xce\xfa\xed\xfe',b'\xcf\xfa\xed\xfe',b'!<arch>',b'\x93NUMPY',b'MZ'))
 if p.suffix=='.json':load(p);finite+=1
proc=subprocess.run(['git','diff','--exit-code','27c19d20696ea6dd4704032c51dfd026218f64f2','--','src','CMakeLists.txt'],cwd=R,capture_output=True)
assert proc.returncode==0
report=dict(passed=True,files=len(files),bytes=sum(p.stat().st_size for p in files),finite_JSONs=finite,copies=len(c['files']),omissions=3,metadata_dependencies=1413,catalog_sha256=sha(D/'catalog.json'),production_unchanged=True,scientific_execution=False)
with (P/'archive-qa001.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report,indent=2))
