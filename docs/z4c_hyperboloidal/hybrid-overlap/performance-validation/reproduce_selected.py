"""Reproduce the completed N12 26..30M0 trajectory with qualified candidate."""
from pathlib import Path
import argparse,importlib.util,json,subprocess,os,time,hashlib,struct,re
ROOT=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111');B=ROOT/'full-overlap-perf20-20261114'
s=importlib.util.spec_from_file_location('old',ROOT/'hybrid_science_segment_v4.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
p=argparse.ArgumentParser();p.add_argument('--tag',default='');p.add_argument('--pair',required=True);p.add_argument('--numerical',action='store_true');p.add_argument('--wrapper',required=True);p.add_argument('--binding',required=True);a=p.parse_args()
paired=json.loads((B/a.pair/'results.json').read_text())
assert paired['source_checkpoints_unchanged'] and all(x.get('passed',x.get('bitwise_equal',False)) for x in paired['equivalence'])
assert paired['original_13M_reproduction'].get('passed',paired['original_13M_reproduction']['bitwise_equal'])
assert paired['original_legacy_reproduction']['bitwise_equal']
assert paired['pilot_gain']
selection=paired['selection']
exe=Path(selection['candidate_executable'])
assert old.sha(exe)==selection['candidate_executable_sha256']
ratio=paired['total_wall_ratio']
run=B/('reproduce-'+os.environ['PBS_JOBID'].split('.')[0]+a.tag);run.mkdir(exist_ok=False)
launch=B/('launch-'+os.environ['PBS_JOBID'].split('.')[0]);env=[x.split() for x in (launch/'rank-environment.txt').read_text().splitlines()];assert len(env)==24 and len({x[0] for x in env})==2 and all(x[3:8]==['1','1','specialized','limited','128'] for x in env)
for f in ['nodes.txt','modules.txt','rank-environment.txt']:(run/f).write_bytes((launch/f).read_bytes())
source=ROOT/'science/N12-full-overlap-30M-20261114/segment-9089673';baseline=json.loads((source/'segment-result.json').read_text());ref=Path(baseline['last_checkpoint']);assert baseline['complete'] and old.sha(ref)==baseline['last_checkpoint_sha256']
rst=source/'rst/headon.00026.rst';assert old.header(rst)['cycle']==18462 and 26<=old.header(rst)['time_M0']<26.01
inp=run/'input.athinput';inp.write_bytes((source/'input.athinput').read_bytes());assert 'nlim = 21301' in inp.read_text()
command=['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+a.binding,a.wrapper,str(exe),'-r',str(rst),'-i',str(inp),'-d',str(run),'-t','00:52:00']
provenance={'command':command,'input_sha256':old.sha(inp),'executable_sha256':old.sha(exe),'restart_sha256':old.sha(rst),'reference_final_sha256':old.sha(ref),'legacy_paired_total_wall_ratio':ratio,'scope':'26M0 saved-state continuation to30M0; compare entire final accepted payload to archived original' };(run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
begin=time.monotonic()
with (run/'run.log').open('w') as log:r=subprocess.run(command,cwd=run,stdout=log,stderr=subprocess.STDOUT,timeout=3450)
(run/'exit-code.txt').write_text(str(r.returncode)+'\n');r.check_returncode()
new=max((run/'rst').glob('*.rst'),key=lambda f:old.header(f)['cycle']);h=old.header(new);assert h==baseline['last'] and h['cycle']==21301
# Identical restart packing: parameter text may differ; compare every payload byte.
def offset(p):
 with p.open('rb') as f:
  text=bytearray()
  while not text.endswith(b'<par_end>\n'):text+=f.read(1)
  fixed=f.read(struct.calcsize('=ii9d38i2di'));nmb=struct.unpack_from('=i',fixed)[0];f.seek(struct.unpack('=Q',f.read(8))[0]+nmb*24,1);meta=struct.unpack('=8Q',f.read(64));f.seek(meta[2]+8,1);return f.tell()
if a.numerical:
 from numerical_check import compare as numeric
 comparison=numeric(ref,new,'long');size=new.stat().st_size-offset(new)
 payload_sha=comparison['candidate_payload_sha256'];bitwise=comparison['bitwise_equal']
else:
 digest=hashlib.sha256();size=0
 with ref.open('rb') as x,new.open('rb') as y:
  x.seek(offset(ref));y.seek(offset(new))
  while True:
   b=x.read(8*1024*1024);c=y.read(8*1024*1024);assert b==c,'Long evolution changed accepted payload'
   if not b:break
   digest.update(b);size+=len(b)
 payload_sha=digest.hexdigest();bitwise=True;comparison={'bitwise_equal':True,'passed':True}
assert old.sha(ref)==baseline['last_checkpoint_sha256'] and old.sha(rst)==provenance['restart_sha256']
result={'complete':True,'final':h,'wall_seconds':time.monotonic()-begin,'accepted_payload_bitwise_equal':bitwise,'numerical_comparison':comparison,'payload_sha256':payload_sha,'payload_bytes':size,'candidate_checkpoint':str(new),'candidate_checkpoint_sha256':old.sha(new),'reference_checkpoint':str(ref),'original_checkpoints_unchanged':True}
from numerical_check import history
result['constraint_history']=history(source/'headon.z4c_hyperboloidal.hst',run/'headon.z4c_hyperboloidal.hst',16)
if not a.numerical:assert result['constraint_history']['all_12_columns_bitwise_equal']
(run/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

# Attribution follows the uninstrumented reproduction and its exact payload check.
subprocess.run(['/usr/bin/python3',str(B/'profile_selected.py'),'--run',str(run),'--exe',str(exe),'--binding',a.binding,'--wrapper',a.wrapper,*(['--numerical'] if a.numerical else [])],check=True,timeout=1000)
