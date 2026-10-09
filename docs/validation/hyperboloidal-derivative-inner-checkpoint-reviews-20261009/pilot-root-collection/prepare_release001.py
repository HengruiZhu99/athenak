"""Root fixed-inventory review; no array decode or scientific execution."""
import ast,hashlib,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
S=R/'build-layer-research/boundary/derivative-pilots-inner-pencil-collector-held-20261009'
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def pin(p):
 p=Path(p).resolve();return dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def check(x):assert pin(x['source'])=={k:x[k] for k in ('source','sha256','bytes')},x['source']
def save(name,x):
 with (P/name).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(S/'source-index.json')['sha256']=='91878140b639f5831cd7c2f2706e0dffc853dd3f332856a340665055d11a4b73'
assert pin(S/'recipe.json')['sha256']=='010c3b1c875515fd4edfe8acef6e9a52d2962d922469014ba029fb0bbf45fe08'
assert pin(S/'collect_once.py')['sha256']=='1bd04e3803377848c18ab4d06825b612bcbf32bf1b9fb3f8df2efcff2dcee741'
ast.parse((S/'collect_once.py').read_text())
i=load(S/'source-index.json');r=load(S/'recipe.json')
assert not Path(r['destination']).exists() and not (S/'invocation001').exists()
assert len(r['roots'])==14 and len(r['planned_files'])==280
assert sum(x['omission_reason'] is None for x in r['planned_files'])==273
deps=[x for part in r['dependency_input_manifests'] for x in load(part['source'])['inputs']]
assert len(deps)==315 and not r['metadata_only_external_files']
review=R/'build-layer-research/boundary/derivative-pilots-inner-pencil-collector-independent-source-review-20261009/receipt.json'
assert pin(review)['sha256']=='c23b9bf684fd3683c837c90ea9c19eb20a0942ddae62497758a0c4917b690a6b'
rows=i['files']+r['planned_files']+deps+[pin(S/'source-index.json'),pin(review),pin(__file__)]
unique={}
for x in rows:
 check(x);v={k:x[k] for k in ('source','sha256','bytes')}
 if v['source'] in unique:assert v==unique[v['source']]
 unique[v['source']]=v
for root in r['roots']:
 actual={str(p.resolve()) for p in Path(root['source']).rglob('*') if p.is_file()}
 assert actual==set(root['inventory']),root['tag']
for g in r['completion_gates']:
 assert pin(g['source'])['sha256']==g['sha256']
 x=load(g['source'])
 for k,v in g['expected'].items():assert x[k]==v,(g['source'],k)
# Recheck the independent selection, including PE signatures and exact policy.
for x in r['planned_files']:
 if x['omission_reason'] is not None:continue
 q=Path(x['source']);b=q.read_bytes();assert len(b)<=1048576
 assert q.suffix.lower() not in {'.npy','.npz','.jsonl','.o','.obj','.a','.so','.dylib','.dll','.exe','.rst','.bin','.pyc','.pyo','.h5','.hdf5','.pkl','.pickle'}
 assert not b.startswith((b'\x7fELF',b'\xfe\xed\xfa',b'\xce\xfa\xed\xfe',b'\xcf\xfa\xed\xfe',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca',b'!<arch>',b'\x93NUMPY',b'\x89HDF',b'MZ'))
 if q.suffix=='.json':load(q)
prod=subprocess.run(['git','diff','--exit-code',r['production_reference_commit'],'--','src','CMakeLists.txt'],cwd=R,capture_output=True)
assert prod.returncode==0 and prod.stderr==b''
report=dict(root_review_passed=True,verified_inputs=list(unique.values()),roots=14,planned_copies=273,omissions=7,metadata_dependencies=315,independent_review=pin(review),production_unchanged=True,no_scientific_calls=True,destination=r['destination'])
save('review.json',report)
save('authorization.json',dict(one_shot_compact_collection_authorized=True,recipe_sha256=pin(S/'recipe.json')['sha256'],source_index_sha256=pin(S/'source-index.json')['sha256'],destination=r['destination'],scope=r['scope'],root_review=pin(P/'review.json')))
print(json.dumps(dict(passed=True,verified_inputs=len(unique),authorization=pin(P/'authorization.json')),indent=2))
