"""Standard-library source-only metadata preparation; never executes collector."""
from pathlib import Path
import ast,hashlib,json,subprocess,time
P=Path(__file__).resolve().parent;R=P.parents[2]
def pin(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'source':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def dump(p,x):
 s=(json.dumps(x,indent=2,allow_nan=False)+'\n').encode();assert len(s)<=1048576;Path(p).write_bytes(s)
def policy(p,size):
 if size>1048576:return 'file exceeds1MiB; original bytes metadata-only'
 if p.suffix.lower() in {'.rst','.bin','.npy','.npz','.jsonl','.o','.obj','.a','.so','.dylib','.dll','.exe','.pyc','.pyo','.h5','.hdf5','.pkl','.pickle'}:return 'array/executable/library/object/JSONL payload metadata-only'
 with p.open('rb') as f:m=f.read(8)
 if m[:4] in {b'\x7fELF',b'\xfe\xed\xfa\xce',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xcf\xfa\xed\xfe',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca'} or m.startswith(b'!<arch>'):return 'compiled binary signature metadata-only'
 if m.startswith(b'\x93NUMPY') or m.startswith(b'\x89HDF'):return 'raw scientific array signature metadata-only'
 return None
config=load(P/'scope-config.json');roots=[];planned=[];seen=set();deps={}
for root in config['roots']:
 src=Path(root['source']).resolve();files=sorted(q.resolve() for q in src.rglob('*') if q.is_file());roots.append({'tag':root['tag'],'source':str(src),'inventory':[str(q) for q in files]})
 for q in files:
  assert str(q) not in seen;seen.add(str(q));v=pin(q);planned.append({**v,'archive_relative':root['tag']+'/'+str(q.relative_to(src)),'role':root['role'],'omission_reason':policy(q,v['bytes'])})
for row in config['selected_files']:
 q=Path(row['source']).resolve();assert str(q) not in seen;seen.add(str(q));v=pin(q);planned.append({**v,'archive_relative':row['archive_relative'],'role':row['role'],'omission_reason':policy(q,v['bytes'])})
external=[{**pin(x['source']),'role':x['role']} for x in config['metadata_only_external_files']]
assert not seen & {x['source'] for x in external}
gates=[]
for g in config['completion_gates']:
 q=Path(g['source']);x=load(q)
 if g.get('sha256'):assert pin(q)['sha256']==g['sha256'],str(q)
 for k,v in g['expected'].items():assert x[k]==v,(str(q),k)
 gates.append({'source':str(q.resolve()),'sha256':pin(q)['sha256'],'expected':g['expected']})
for a,b in config['identical_inventory_pairs']:assert Path(a).read_bytes()==Path(b).read_bytes()
for manifest in config['dependency_manifests']:
 x=load(manifest['source'])
 for key in manifest.get('keys',[]):x=x[key]
 if isinstance(x,dict):items=list(x.items())
 else:items=[(z.get('source',z.get('path')),z['sha256']) for z in x]
 for q,h in items:
  v=pin(q);assert v['sha256']==h,str(q);assert q not in deps or deps[q]==v;deps[q]=v
for x in external:deps[x['source']]={k:x[k] for k in ('source','sha256','bytes')}
parts=[];drows=sorted(deps.values(),key=lambda x:x['source']);(P/'dependency-inputs').mkdir(exist_ok=False)
for j in range(0,len(drows),256):
 q=P/'dependency-inputs'/('part-%03d.json'%(j//256));dump(q,{'inputs':drows[j:j+256]});parts.append(pin(q))
policy_map={'max_copied_bytes_per_file':1048576,'all_arrays_executables_objects_NPY_NPZ_JSONL_metadata_only':True,'binary_signatures_checked':True,'logs_exact_bytes_whitespace_preserved':True,'large_logs_not_truncated_or_normalized':True,'never_overwrite_existing_destination':True,'no_live_parent_derivative_or_new008_tree':True}
recipe={k:config[k] for k in ('scope','destination','production_reference_commit','README')};recipe.update(roots=roots,planned_files=planned,metadata_only_external_files=external,completion_gates=gates,identical_inventory_pairs=config['identical_inventory_pairs'],dependency_input_manifests=parts,policy=policy_map)
dump(P/'recipe.json',recipe)
for q in (P/'collect_once.py',P/'prepare_metadata.py'):ast.parse(q.read_text())
files=[pin(P/n) for n in ('collect_once.py','prepare_metadata.py','scope-config.json','PLAN.md','release-schema.json','recipe.json')]+parts
dump(P/'source-index.json',{'scope':config['scope'],'execution':'HELD','files':files,'new_scientific_calls':0})
summary={'source_only':True,'execution':'HELD','source_index':pin(P/'source-index.json'),'recipe':pin(P/'recipe.json'),'collector':pin(P/'collect_once.py'),'planned_roots':len(roots),'planned_files':len(planned),'planned_copied_files':sum(not x['omission_reason'] for x in planned),'planned_metadata_only_files':sum(bool(x['omission_reason']) for x in planned),'external_dependencies':len(drows),'external_current_document_records':len(external),'archive_exists':Path(config['destination']).exists(),'all_completion_gates_checked':True,'no_scientific_imports_queries_or_array_decodes':True}
assert not summary['archive_exists'];dump(P/'preparation-receipt.json',summary);print(json.dumps(summary,indent=2))
