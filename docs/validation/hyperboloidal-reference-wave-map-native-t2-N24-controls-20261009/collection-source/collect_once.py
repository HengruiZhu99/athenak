"""HELD one-shot compact collector; no scientific modules/queries or native steps."""
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
 s=(json.dumps(x,indent=2,allow_nan=False)+'\n').encode();assert len(s)<=1048576,'generated compact metadata exceeds1MiB';Path(p).write_bytes(s)
def verify(rows):
 for row in rows:assert pin(row['source'])=={k:row[k] for k in ('source','sha256','bytes')},row['source']
def assert_gates(recipe):
 for g in recipe['completion_gates']:
  x=load(g['source']);assert pin(g['source'])['sha256']==g['sha256']
  for key,value in g['expected'].items():assert x[key]==value,(g['source'],key)
 # Case-specific paired source inventories must remain byte-identical.
 for pair in recipe['identical_inventory_pairs']:assert Path(pair[0]).read_bytes()==Path(pair[1]).read_bytes()
def file_policy(path,size):
 p=Path(path);suffix=p.suffix.lower()
 if size>1048576:return 'file exceeds1MiB; exact original bytes metadata-only'
 if suffix in {'.rst','.bin','.npy','.npz','.jsonl','.o','.obj','.a','.so','.dylib','.dll','.exe','.pyc','.pyo','.h5','.hdf5','.pkl','.pickle'}:return 'raw array/compiled/object/JSONL payload metadata-only'
 with p.open('rb') as f:magic=f.read(8)
 if magic[:4] in {b'\x7fELF',b'\xfe\xed\xfa\xce',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xcf\xfa\xed\xfe',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca'} or magic.startswith(b'!<arch>'):return 'compiled binary signature metadata-only'
 if magic.startswith(b'\x93NUMPY') or magic.startswith(b'\x89HDF'):return 'raw scientific array signature metadata-only'
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--authorization',required=True,type=Path);ap.add_argument('--authorization-sha256',required=True);a=ap.parse_args()
 INV=P/'invocation001';INV.mkdir(exist_ok=False)
 record={'completed':False,'source':pin(__file__),'stage':'compact_saved_evidence_collection','new_scientific_calls':0,'started_ns':time.time_ns(),'scope':'One fixed archive of completed N24 native/partial/readback evidence only; no live tree/operator source or evolution.'};started=time.monotonic();dest=None
 try:
  assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
  authpin=pin(a.authorization);assert authpin['sha256']==a.authorization_sha256
  recipe=load(P/'recipe.json');index=load(P/'source-index.json');auth=load(a.authorization)
  assert auth['one_shot_compact_collection_authorized'] is True and auth['recipe_sha256']==pin(P/'recipe.json')['sha256'] and auth['source_index_sha256']==pin(P/'source-index.json')['sha256']
  assert auth['destination']==recipe['destination']
  protected=index['files']+recipe['planned_files']+[row for part in recipe['dependency_input_manifests'] for row in load(part['source'])['inputs']]+[authpin,pin(P/'source-index.json')]
  verify(protected);assert_gates(recipe)
  # Exact fixed source-tree inventory: no completed tree may acquire a new/live file.
  for root in recipe['roots']:
   actual={str(p.resolve()) for p in Path(root['source']).rglob('*') if p.is_file()}
   assert actual==set(root['inventory']),root['source']
  expected=recipe['production_reference_commit'];check=subprocess.run(['git','diff','--exit-code',expected,'--','src','CMakeLists.txt'],cwd=R,capture_output=True)
  (INV/'production-diff.stdout').write_bytes(check.stdout);(INV/'production-diff.stderr').write_bytes(check.stderr);assert check.returncode==0,'production differs from recorded runtime source'
  dest=Path(recipe['destination']);assert not dest.exists(),'never collect into existing destination'
  record.update(authorization=authpin,recipe=pin(P/'recipe.json'),source_index=pin(P/'source-index.json'),launch_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),destination=str(dest))
  dump(INV/'receipt.json',record)
  dest.mkdir(parents=True,exist_ok=False)
  copied={};omitted={};json_count=0
  for item in recipe['planned_files']:
   src=Path(item['source']);rel=item['archive_relative'];reason=file_policy(src,item['bytes'])
   assert reason==item['omission_reason'],'policy drift '+str(src)
   value={k:item[k] for k in ('source','sha256','bytes','role')}
   if reason:omitted[rel]={**value,'reason':reason};continue
   target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
   assert target.read_bytes()==src.read_bytes(),'byte copy mismatch'
   if target.suffix=='.json':load(target);json_count+=1
   copied[rel]=value
  for name in ['collect_once.py','prepare_metadata.py','recipe.json','source-index.json','PLAN.md','release-schema.json']:
   src=P/name;rel='collection-source/'+name;item=pin(src);reason=file_policy(src,item['bytes'])
   if reason:omitted[rel]={**item,'role':'collector source/recipe','reason':reason}
   else:
    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);assert target.read_bytes()==src.read_bytes();copied[rel]={**item,'role':'collector source/recipe'}
  # Metadata-only outside-tree dependencies include all original source/runtime/exes/objects/raw arrays.
  deps=[row for part in recipe['dependency_input_manifests'] for row in load(part['source'])['inputs']];dependency_parts=[]
  for i in range(0,len(deps),256):
   name=f'dependency-metadata/part-{i//256:03d}.json'
   (dest/'dependency-metadata').mkdir(exist_ok=True);dump(dest/name,{'scope':'Exact external byte identities only; never copied executable/object/array/runtime payloads.','inputs':deps[i:i+256]});dependency_parts.append(name)
  prod={'runtime_production_commit':expected,'source_scope':['src','CMakeLists.txt'],'git_diff_returncode':check.returncode,'launch_HEAD':record['launch_HEAD'],'scope':'Working production files match runtime27c19; no production edits by this collector.'};dump(dest/'production-source-identity.json',prod)
  readme='''This compact archive contains only completed N24 evidence selected in its source-reviewed recipe.\n\nThe wave-map standard, half-step-large and small-pulse processes all aborted before targett2. Their stopped-run observers and independent manual field checks certify only the saved arrays, not unsaved abort states or completed native runs. The standard context is cross-linked from the earlier checkpoint. The C0N24 process completedt2 and its unchanged completed-result analyzer passed81 saved-array gates; this is finite-time native evidence, not a continuum/stability proof or a completed wave-map control.\n\nThe figure uses only pinned completed observer/manual savedJSON, root scalar comparison and exact failurestderr. No scientific source query, native advance, matrix/operator assembly or spectrum occurs in this collection.\n\nCopied files retain exact original bytes including whitespace. All RST/BIN/NPY/NPZ/JSONL/raw arrays, executables, libraries, objects and everyfile>1MiB remain metadata-only. Large console/analyzer records are not normalized or truncated; their full originals are identified by SHA256 and bytecount. Selected dependency metadata identifies external source/compiler/runtime inputs without copying those payloads.\n\nNo liveparent batch, running values attempt, bulk008 source/operator/attempt, or existing frozen archive is collected or modified.\n'''
  (dest/'README.md').write_text(readme)
  for rel,item in copied.items():assert pin(dest/rel)['sha256']==item['sha256']
  verify(protected);assert_gates(recipe)
  for root in recipe['roots']:assert {str(p.resolve()) for p in Path(root['source']).rglob('*') if p.is_file()}==set(root['inventory'])
  check2=subprocess.run(['git','diff','--exit-code',expected,'--','src','CMakeLists.txt'],cwd=R,capture_output=True);assert check2.returncode==0
  catalog={'scope':record['scope'],'files':copied,'omitted_large_payloads':omitted,'roots':recipe['roots'],'policy':recipe['policy'],'source_recipe':pin(P/'recipe.json'),'collector_source':pin(__file__),'external_dependency_parts':dependency_parts,'production_identity_sha256':pin(dest/'production-source-identity.json')['sha256'],'README_sha256':pin(dest/'README.md')['sha256']};dump(dest/'catalog.json',catalog)
  record.update(completed=True,inputs_unchanged=True,source_inventories_unchanged=True,production_unchanged=True,copied_files=len(copied),copied_bytes=sum(x['bytes'] for x in copied.values()),omitted_payloads=len(omitted),external_dependency_records=len(deps),finite_copied_JSONs=json_count,catalog=pin(dest/'catalog.json'),generated_metadata_files=len(dependency_parts)+4)
  dump(dest/'collection-receipt.json',record)
 except BaseException as exc:record['failure']=type(exc).__name__+': '+str(exc)
 finally:
  record['seconds']=time.monotonic()-started;record['finished_ns']=time.time_ns();dump(INV/'receipt.json',record)
 print(json.dumps(record,indent=2));assert record['completed'],'Collection failure preserved; do not retry into existing destination'
if __name__=='__main__':main()
