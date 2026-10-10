"""SOURCE-ONLY prepared one-shot certificate launcher; root must review then run."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OWNER=BASE/'continuum/manufactured-Gaussian-a2-cache-v6-instrumented-held-20261009'
INDEX_SHA='cafc4df240ba9988919f2f4d97522bda2f71a19f1899a625d2be98afa2144397'
RECIPE_SHA='9a7ee237b6fd4d764bac38e62ff4307248fbc4beb57b7a0dbd307e690d0e0742'
ROOT_PINS_SHA='15190ec460a653afc492c38d2d8b4d313c8d12e6be620da18eac6869c6e6141b'
ROOT_REVIEW_SHA='84aeccdca882ed14b26e268a10244cf93d7fa1b09b0e1460191c47ccb4adb0ce'
INDEPENDENT_RECEIPT=BASE/'boundary/manufactured-Gaussian-a2-cache-v6-independent-source-review-20261009/receipt.json'
INDEPENDENT_SHA='a110a702faeea2d73338034e0c3aed9d83e2dde4b9338c22ac2c7d015f0f9aca'
PROCESS_GROUP_SECONDS=720
ENVIRONMENT={'PYTHONOPTIMIZE':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
             'OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def write(path,value):
 with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def require(value,message):
 if not value:raise RuntimeError(message)
def verify(pins):
 for name,digest in pins.items():require(sha(name)==digest,'changed protected input '+name)
def snapshot(pins):
 result={}
 for name in pins:
  try:result[name]=sha(name)
  except OSError as error:result[name]={'error':str(error)}
 return result

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('stage',choices=('certificate',))
 args=parser.parse_args()
 out=HERE/'certificate-invocation001'
 out.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();pins={};before={};child_path=OWNER/'attempts/certificate001';child=None
 record={'completed':False,'accepted_stage':False,'producer_passed':False,'returncode':None,
         'stage':'certificate','global_slicing_accepted':False,'replay_authorized':False,
         'process_group_limit_seconds':PROCESS_GROUP_SECONDS,'process_group_timeout':False}
 try:
  require(sys.flags.optimize==0 and sys.flags.isolated==1 and sys.dont_write_bytecode,
          'exact unoptimized -I -B root launcher route required')
  release_path=HERE/'certificate-release.json';auth_path=HERE/'certificate-authorization.json'
  release=load(release_path)
  require(release['stage']==args.stage=='certificate','certificate-only release')
  require(Path(release['owner']).resolve()==OWNER and Path(release['output']).resolve()==child_path,'exact fixed owner/output')
  require(not child_path.exists(),'fresh certificate001 output required')
  require(release['source_index_sha256']==INDEX_SHA and release['recipe_sha256']==RECIPE_SHA,'exact released scientific source')
  require(sha(HERE/'source-pins002.json')==ROOT_PINS_SHA and sha(HERE/'source-review002.json')==ROOT_REVIEW_SHA,'root review002 provenance')
  require(sha(INDEPENDENT_RECEIPT)==INDEPENDENT_SHA,'independent source review identity')
  pins=dict(release['pins']);root_pins=load(HERE/'source-pins002.json')
  require(len(root_pins)==237 and all(pins.get(p)==h for p,h in root_pins.items()),'all237 reviewed root pins required')
  pins[str(auth_path)]=release['authorization_sha256'];pins[str(release_path)]=sha(release_path)
  require(pins.get(str(INDEPENDENT_RECEIPT))==INDEPENDENT_SHA,'independent review in protected set')
  verify(pins);before=snapshot(pins);require(before==pins,'prelaunch snapshot drift');write(out/'pins-before.json',before)
  recipe=load(OWNER/'recipe.json');auth=load(auth_path)
  require(sha(OWNER/'recipe.json')==RECIPE_SHA and sha(OWNER/'source-index.json')==INDEX_SHA,'scientific source identity')
  require(recipe['domain_wall_seconds']==600 and recipe['stage_timeouts']['certificate']==660,'unchanged600/660 resource limits')
  require(auth['allow_execution'] is True and auth['stage']=='certificate' and auth['output']==str(child_path),'exact single-use authorization')
  env=os.environ.copy()
  for name in ('PYTHONHOME','PYTHONPATH','PYTHONWARNINGS'):env.pop(name,None)
  env.update(ENVIRONMENT)
  command=[recipe['python']['path'],'-I','-B',str(OWNER/'run_once.py'),'--stage','certificate',
           '--recipe',str(OWNER/'recipe.json'),'--authorization',str(auth_path),'--output',str(child_path)]
  write(out/'invocation.json',{'command':command,'environment':ENVIRONMENT,
       'removed_environment':['PYTHONHOME','PYTHONPATH','PYTHONWARNINGS'],'cwd':str(HERE),
       'producer_seconds':600,'inner_wrapper_seconds':660,'root_process_group_seconds':720,
       'start_new_session':True,'timeout_action':'SIGKILL entire new process group; preserve all partial output'})
  print(json.dumps({'event':'launch','stage':'certificate','output':str(child_path),'process_group_seconds':720}),flush=True)
  with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
   child=subprocess.Popen(command,cwd=str(HERE),env=env,stdout=stdout,stderr=stderr,start_new_session=True)
   write(out/'process.json',{'pid':child.pid,'process_group_id':child.pid,'start_new_session':True})
   try:record['returncode']=child.wait(timeout=PROCESS_GROUP_SECONDS)
   except subprocess.TimeoutExpired:
    record['process_group_timeout']=True
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    record['returncode']=child.wait()
    raise TimeoutError('fixed720-second process-group cap; failed/partial diagnostic preserved')
  receipt=load(child_path/'receipt.json') if (child_path/'receipt.json').exists() else {}
  report=load(child_path/'report.json') if (child_path/'report.json').exists() else {}
  accepted=bool(record['returncode']==0 and receipt.get('completed') is True and receipt.get('passed') is True
      and type(receipt.get('returncode')) is int and receipt.get('returncode')==0 and receipt.get('inputs_unchanged') is True
      and receipt.get('stage')=='certificate' and receipt.get('source_index_sha256')==INDEX_SHA
      and receipt.get('recipe_sha256')==RECIPE_SHA and report.get('passed') is True
      and report.get('stage')=='certificate' and report.get('source_index_sha256')==INDEX_SHA
      and report.get('coverage_complete') is True and report.get('independent_replay_passed') is False
      and report.get('global_slicing_acceptance') is False)
  record.update(completed=accepted,accepted_stage=accepted,producer_passed=accepted,
      child_receipt_sha256=sha(child_path/'receipt.json') if (child_path/'receipt.json').exists() else None,
      report_sha256=sha(child_path/'report.json') if (child_path/'report.json').exists() else None)
  if not accepted:raise RuntimeError('actual producer did not pass all fixed completion/source gates; keep failure/partial classification')
 except BaseException as error:
  record.update(exception_type=type(error).__name__,exception=str(error),completed=False,accepted_stage=False,producer_passed=False)
  (out/'failure.txt').write_text(traceback.format_exc())
 finally:
  # Any unexpected exception after launch must not leave an orphan producer.
  if child is not None and child.poll() is None:
   try:os.killpg(child.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   record['returncode']=child.wait()
   record['process_group_cleanup_kill']=True
  after=snapshot(pins);write(out/'pins-after.json',after)
  record['inputs_unchanged']=bool(before) and before==after
  if not record['inputs_unchanged']:record.update(completed=False,accepted_stage=False,producer_passed=False)
  record['seconds']=time.monotonic()-started
  record['output_inventory']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),
      'policy':'large_payload' if p.suffix in ('.npz','.npy','.jsonl') or p.stat().st_size>1048576 else 'source_or_receipt'}
      for folder in (out,child_path) if folder.exists() for p in sorted(folder.rglob('*')) if p.is_file() and p!=out/'receipt.json']
  write(out/'receipt.json',record)
 print(json.dumps({k:v for k,v in record.items() if k!='output_inventory'}),flush=True)
 return 0 if record['accepted_stage'] else 1

if __name__=='__main__':raise SystemExit(main())
