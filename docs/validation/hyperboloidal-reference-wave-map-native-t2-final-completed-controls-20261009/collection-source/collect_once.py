"""HELD one-shot compact collector of fixed completed saved evidence only."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time
P=Path(__file__).resolve().parent;R=P.parents[2]
def pin(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'source':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def dump(p,x):
 b=(json.dumps(x,indent=2,allow_nan=False)+'\n').encode();assert len(b)<=1048576,'generated metadata exceeds1MiB';Path(p).write_bytes(b)
def verify(rows):
 for x in rows:assert pin(x['source'])=={k:x[k] for k in ('source','sha256','bytes')},x['source']
def file_policy(path,size):
 p=Path(path)
 if size>1048576:return 'file exceeds1MiB; original bytes metadata-only'
 if p.suffix.lower() in {'.rst','.bin','.npy','.npz','.jsonl','.o','.obj','.a','.so','.dylib','.dll','.exe','.pyc','.pyo','.h5','.hdf5','.pkl','.pickle'}:return 'array/executable/library/object/JSONL payload metadata-only'
 with p.open('rb') as f:magic=f.read(8)
 if magic[:4] in {b'\x7fELF',b'\xfe\xed\xfa\xce',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xcf\xfa\xed\xfe',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca'} or magic.startswith(b'!<arch>'):return 'compiled binary signature metadata-only'
 if magic.startswith(b'\x93NUMPY') or magic.startswith(b'\x89HDF'):return 'raw scientific array signature metadata-only'
 return None
def gates(recipe):
 for row in recipe['completion_gates']:
  x=load(row['source']);assert pin(row['source'])['sha256']==row['sha256']
  for k,v in row['expected'].items():assert x[k]==v,(row['source'],k)
 for a,b in recipe['identical_inventory_pairs']:assert Path(a).read_bytes()==Path(b).read_bytes()
def inventories(recipe):
 for root in recipe['roots']:
  actual={str(p.resolve()) for p in Path(root['source']).rglob('*') if p.is_file()}
  assert actual==set(root['inventory']),root['source']
def main():
 a=argparse.ArgumentParser();a.add_argument('--authorization',required=True,type=Path);a.add_argument('--authorization-sha256',required=True);args=a.parse_args()
 inv=P/'invocation001';inv.mkdir(exist_ok=False);started=time.monotonic();record={'completed':False,'source':pin(__file__),'new_scientific_calls':0,'stage':'one_shot_compact_saved_evidence_collection'}
 try:
  assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
  apin=pin(args.authorization);assert apin['sha256']==args.authorization_sha256
  auth=load(args.authorization);recipe=load(P/'recipe.json');index=load(P/'source-index.json')
  assert auth['one_shot_compact_collection_authorized'] is True
  assert auth['recipe_sha256']==pin(P/'recipe.json')['sha256'] and auth['source_index_sha256']==pin(P/'source-index.json')['sha256'] and auth['destination']==recipe['destination']
  deps=[x for part in recipe['dependency_input_manifests'] for x in load(part['source'])['inputs']]
  protected=index['files']+recipe['planned_files']+recipe['metadata_only_external_files']+deps+[apin,pin(P/'source-index.json')]
  verify(protected);gates(recipe);inventories(recipe)
  assert not {x['source'] for x in recipe['planned_files']} & {x['source'] for x in recipe['metadata_only_external_files']},'external current documents must not be copied'
  proc=subprocess.run(['git','diff','--exit-code',recipe['production_reference_commit'],'--','src','CMakeLists.txt'],cwd=R,capture_output=True)
  (inv/'production-diff.stdout').write_bytes(proc.stdout);(inv/'production-diff.stderr').write_bytes(proc.stderr);assert proc.returncode==0
  dest=Path(recipe['destination']);assert not dest.exists(),'fresh destination only; never rerun/overwrite'
  record.update(scope=recipe['scope'],authorization=apin,recipe=pin(P/'recipe.json'),source_index=pin(P/'source-index.json'),destination=str(dest),launch_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip());dump(inv/'receipt.json',record)
  dest.mkdir(parents=True,exist_ok=False);copied={};omitted={};finite_json=0
  for x in recipe['planned_files']:
   reason=file_policy(x['source'],x['bytes']);assert reason==x['omission_reason'];rel=x['archive_relative'];v={k:x[k] for k in ('source','sha256','bytes','role')}
   if reason:omitted[rel]={**v,'reason':reason};continue
   target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(x['source'],target);assert target.read_bytes()==Path(x['source']).read_bytes()
   if target.suffix=='.json':load(target);finite_json+=1
   copied[rel]=v
  for name in ('collect_once.py','prepare_metadata.py','scope-config.json','recipe.json','source-index.json','PLAN.md','release-schema.json'):
   src=P/name;x=pin(src);rel='collection-source/'+name;reason=file_policy(src,x['bytes'])
   if reason:omitted[rel]={**x,'role':'collector source/preparation','reason':reason}
   else:
    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);assert target.read_bytes()==src.read_bytes();copied[rel]={**x,'role':'collector source/preparation'}
  parts=[]
  for j in range(0,len(deps),256):
   rel=f'dependency-metadata/part-{j//256:03d}.json';(dest/'dependency-metadata').mkdir(exist_ok=True);dump(dest/rel,{'scope':'External dependencies metadata only; no payload copies.','inputs':deps[j:j+256]});parts.append(rel)
  dump(dest/'external-current-documents-metadata.json',{'scope':'Current document/overview or prior capsules are external byte identities only. No duplicate frozen prose is implied.','files':recipe['metadata_only_external_files']})
  prod={'runtime_production_commit':recipe['production_reference_commit'],'source_scope':['src','CMakeLists.txt'],'git_diff_returncode':proc.returncode,'launch_HEAD':record['launch_HEAD']};dump(dest/'production-source-identity.json',prod)
  (dest/'README.md').write_text(recipe['README'])
  verify(protected);gates(recipe);inventories(recipe)
  assert subprocess.run(['git','diff','--exit-code',recipe['production_reference_commit'],'--','src','CMakeLists.txt'],cwd=R,capture_output=True).returncode==0
  for rel,x in copied.items():assert pin(dest/rel)['sha256']==x['sha256']
  catalog={'scope':recipe['scope'],'files':copied,'omitted_large_payloads':omitted,'roots':recipe['roots'],'policy':recipe['policy'],'external_dependency_parts':parts,'metadata_only_external_files':recipe['metadata_only_external_files'],'source_recipe':pin(P/'recipe.json'),'collector_source':pin(__file__),'README_sha256':pin(dest/'README.md')['sha256'],'production_identity_sha256':pin(dest/'production-source-identity.json')['sha256'],'external_current_documents_metadata_sha256':pin(dest/'external-current-documents-metadata.json')['sha256']};dump(dest/'catalog.json',catalog)
  record.update(completed=True,inputs_unchanged=True,source_inventories_unchanged=True,production_unchanged=True,copied_files=len(copied),copied_bytes=sum(x['bytes'] for x in copied.values()),omitted_payloads=len(omitted),external_dependency_records=len(deps),external_document_metadata_records=len(recipe['metadata_only_external_files']),finite_copied_JSONs=finite_json,catalog=pin(dest/'catalog.json'),generated_metadata_files=len(parts)+5)
  dump(dest/'collection-receipt.json',record)
 except BaseException as exc:record['failure']=type(exc).__name__+': '+str(exc)
 finally:
  record['seconds']=time.monotonic()-started;dump(inv/'receipt.json',record)
 print(json.dumps(record,indent=2));assert record['completed'],'failure preserved; no retry into an existing destination'
if __name__=='__main__':main()
