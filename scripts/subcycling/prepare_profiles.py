"""Prepare isolated fixed-length, live-AMR restarts; never edit production files."""
from pathlib import Path
import json,sys,hashlib,re,argparse
ROOT=Path('/pscratch/sd/h/hzhu/vc-subcycling-20260911')
CAMPAIGN=Path('/pscratch/sd/h/hzhu/n128-chite-bisection-20260910')
sys.path.insert(0,str(CAMPAIGN))
from resume_case import checkpoint_info,sha
from workflow_common import setparam
source=CAMPAIGN/'cycle_03_recovery_24000'
parser=argparse.ArgumentParser()
parser.add_argument('--set-name',default='profiles')
parser.add_argument('--source-compatibility',action='store_true')
args=parser.parse_args()
if Path(args.set_name).name != args.set_name: raise SystemExit('set-name must be a directory name')
manifest=[]
for tag,number in [('early',63),('late',66)]:
 checkpoint=source/'rst'/('lapse200.%05d.rst'%number)
 info=checkpoint_info(checkpoint)
 checkpoint_hash=sha(checkpoint)
 for mode in ['baseline','profile']:
  case=ROOT/args.set_name/(tag+'_'+mode)
  case.mkdir(parents=True,exist_ok=False)
  with (source/'amr_history.jsonl').open('rb') as f:
   prefix=f.read(info['history_bytes'])
  assert len(prefix)==info['history_bytes'] and prefix.endswith(b'\n')
  (case/'amr_history.jsonl').write_bytes(prefix)
  inp=(source/'input.athinput').read_text()
  for section,key,value in [('time','nlim',info['cycle']+12),('time','ndiag',1),('time','execution_profile',str(mode=='profile').lower()),('mesh_refinement','amr_history_file',case/'amr_history.jsonl'),('problem','brill_global_coefficients_file',source/'initial.coefficients'),('mesh_refinement','max_nmb_per_rank',24000)]:
   inp=setparam(inp,section,key,value)
  if args.source_compatibility and mode=='profile':
   recorded_source=json.loads(prefix.splitlines()[0])['source_id']
   inp=setparam(inp,'mesh_refinement','amr_history_compatible_source_id',recorded_source)
  (case/'input.athinput').write_text(inp)
  exe=CAMPAIGN/'bundle/athena.history_extrema' if mode=='baseline' else ROOT/'build/src/athena'
  manifest.append(dict(case=str(case),tag=tag,mode=mode,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,info=info,input_sha256=sha(case/'input.athinput'),exe=str(exe),steps=12))
(ROOT/'profiles.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
