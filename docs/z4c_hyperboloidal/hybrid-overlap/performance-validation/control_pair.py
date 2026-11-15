"""Matched short timing windows; original legacy science data stay read-only."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,statistics,struct,subprocess,time
ROOT=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111')
B=ROOT/'full-overlap-perf20-20261114'
s=importlib.util.spec_from_file_location('old',ROOT/'hybrid_science_segment_v4.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--binding',required=True);p.add_argument('--wrapper',required=True);a=p.parse_args();run=a.run
parent=json.loads((B/'late-isolation-9091104.json').read_text());assert parent['complete']
selected=parent['selected'];actual=Path(selected['executable'])
if actual==B/'athena-reuse_unroll-kernel':
 (run/'selection.json').write_text(json.dumps({'selected_pair':'reuse-pair-9090714','selected_reproduction':selected['run'],'reason':'Only simpler reuse passed late check; already has matched legacy timing'},indent=2)+'\n');raise SystemExit(0)
assert actual==B/'athena-cache-only'
assert old.sha(actual)==json.loads((B/'cache-only-receipt.json').read_text())['executable_sha256']
exe={'candidate':actual,'legacy':ROOT/'build-full-N12-20261114/src/athena'}
assert old.sha(exe['legacy'])=='0b0173704b67b4e3b868c6ccc8a15adc7cab480a1e61065b517384bc6c222a09'
legacy=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/headon-6M-20261109/runs/headon20-9055906/rst/headon_6M.00012.rst')
full=ROOT/'science/N12-full-overlap-20261114/segment-9089261/rst/headon.00012.rst'
assert old.header(legacy)['cycle']==8521 and old.header(full)['cycle']==8520
sources={str(p):old.sha(p) for p in [legacy,full]}
base=(ROOT/'science/N12-full-overlap-20261114/input.athinput').read_text();dt=.00046944203735617445
env=[x.split() for x in (run/'rank-environment.txt').read_text().splitlines()]
assert len(env)==24 and len({x[0] for x in env})==2 and all(x[3:8]==['1','1','specialized','limited','128'] for x in env)
cpus=[set(range(int(x.split('-')[0]),int(x.split('-')[1])+1)) for x in a.binding.split(':')]
assert all(set(map(int,x[8].split(',')))==cpus[int(x[1])] for x in env)
(run/'provenance.json').write_text(json.dumps({'source_checkpoints':sources,'executable_sha256':{k:old.sha(v) for k,v in exe.items()},'late_qualification':selected,'protocol':'W4 without donor conversion; originalI/O. Fresh8 exact, L,C,C,L1065 matchedoutputtimings. Frozen original legacy exe0b017, qualified W4full exe de633. All checks remain strictly bitwise. Follow-up to intermittent legacy-only repeat failure9091219; preserve all failed evidence.'},indent=2)+'\n')

def launch(label,name,rst,steps):
 folder=run/name;folder.mkdir();start=old.header(rst)['cycle'] if rst else 0;text=re.sub(r'(?m)^nlim = .*$', 'nlim = '+str(start+steps),base);text=re.sub(r'(?m)^tlim = .*$', 'tlim = 10',text);text=text.replace('ndiag = 100','ndiag = 1')
 if label.startswith('legacy'):text=text.replace('overlap_mode = agreement_full','overlap_mode = legacy')
 inp=folder/'input.athinput';inp.write_text(text)
 command=['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+a.binding,a.wrapper,str(exe[label]),*(['-r',str(rst)] if rst else []),'-i',str(inp),'-d',str(folder)]
 if start==8521:
  # The one-step preparation forced a final dump and advanced output counters.
  # Restore archived legacy scheduling for BOTH modes; only output metadata changes.
  command+=['output1/last_time=4.0000000000000009','output2/last_time=4','output2/file_number=13']
 (folder/'provenance.json').write_text(json.dumps({'command':command,'input_sha256':old.sha(inp),'executable_sha256':old.sha(exe[label])},indent=2)+'\n')
 begin=time.monotonic()
 with (folder/'run.log').open('w') as log:r=subprocess.run(command,cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=1050)
 wall=time.monotonic()-begin;(folder/'exit-code.txt').write_text(str(r.returncode)+'\n');r.check_returncode()
 checkpoint_cycles=sorted(old.header(p)['cycle'] for p in (folder/'rst').glob('*.rst'))
 if start==8521 and steps==1065:assert checkpoint_cycles==[9231,9586],checkpoint_cycles
 last=max((folder/'rst').glob('*.rst'),key=lambda p:old.header(p)['cycle']);h=old.header(last);assert h['cycle']==start+steps and h['dt']==dt
 rows=[(float(e),int(c)) for e,c in re.findall(r'elapsed=(\S+) cycle=(\d+) time=\S+ dt=\S+',(folder/'run.log').read_text())]
 selected=[x for x in rows if start+20<=x[1]<start+steps];intervals=[y[0]-x[0] for x,y in zip(selected,selected[1:]) if y[1]==x[1]+1]
 d={'label':label,'name':name,'start_cycle':start,'steps':steps,'wall_seconds':wall,'wall_s_per_M0':wall/(steps*3*dt),'restart':str(last),'header':h,'checkpoint_cycles':checkpoint_cycles,'checkpoint_sha256':old.sha(last)}
 if intervals:d.update(mean_step_s=statistics.mean(intervals),median_step_s=statistics.median(intervals))
 (folder/'result.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True);return d
def offset(p):
 with p.open('rb') as f:
  text=bytearray()
  while not text.endswith(b'<par_end>\n'):
   c=f.read(1);assert c;text+=c
  nmb=struct.unpack_from('=i',f.read(struct.calcsize('=ii9d38i2di')))[0];f.seek(struct.unpack('=Q',f.read(8))[0]+nmb*24,1);meta=struct.unpack('=8Q',f.read(64));f.seek(meta[2]+8,1);return f.tell()
def compare(p,q):
 assert old.header(p)==old.header(q);digest=hashlib.sha256();size=0;equal=True
 with p.open('rb') as x,q.open('rb') as y:
  x.seek(offset(p));y.seek(offset(q))
  while True:
   b,c=x.read(8*1024*1024),y.read(8*1024*1024);equal &= b==c;assert len(b)==len(c)
   if not b:break
   digest.update(b);size+=len(b)
 return {'bitwise_equal':equal,'payload_bytes':size,'reference_sha256':digest.hexdigest()}

results={'measurements':[],'equivalence':[],'selection':{'candidate_executable':str(actual),'candidate_executable_sha256':old.sha(actual)},'qualified_late_reproduction':selected['run'],'legacy_executable_sha256':old.sha(exe['legacy'])}
def save():(run/'results.json').write_text(json.dumps(results,indent=2)+'\n')
d=launch('candidate','fresh-cache-only',None,8)
reference=json.loads((B/'paired-9090513/fresh-candidate/result.json').read_text())
results['equivalence'].append({'case':'fresh',**compare(Path(reference['restart']),Path(d['restart']))});save()
prep=json.loads((B/'legacy-pair-9090542/prepare-cycle8521/result.json').read_text());prepared=Path(prep['restart'])
assert old.header(prepared)['cycle']==8521 and old.sha(prepared)==prep['checkpoint_sha256'];sources[str(prepared)]=old.sha(prepared)
first={}
for i,label in enumerate(['legacy','candidate','candidate','legacy']):
 d=launch(label,'timing-%d-%s'%(i,label),legacy if label=='legacy' else prepared,1065);results['measurements'].append(d);save()
 if label in first:results['equivalence'].append({'case':d['name'],**compare(Path(first[label]['restart']),Path(d['restart']))})
 else:first[label]=d
 save()
ref=ROOT/'science/N12-full-overlap-30M-20261114/segment-9089673/rst/headon.00013.rst'
intermediate=next(p for p in Path(first['candidate']['restart']).parent.glob('*.rst') if old.header(p)['cycle']==9231)
results['original_13M_reproduction']=compare(ref,intermediate)
legacy_ref=Path(json.loads((B/'legacy-pair-9090542/timing-0-legacy_original/result.json').read_text())['restart'])
results['original_legacy_reproduction']=compare(legacy_ref,Path(first['legacy']['restart']))
assert all(old.sha(Path(p))==h for p,h in sources.items());results['source_checkpoints_unchanged']=True
def history(reference,candidate,minimum_rows):
 def rows(p):
  out={}
  for line in p.read_text().splitlines():
   if line.strip() and not line.startswith('#'):
    r=tuple(map(float,line.split()));assert len(r)==12;out[round(r[0]/dt)]=r
  return out
 x,y=rows(reference),rows(candidate);common=sorted(x.keys()&y.keys());assert len(common)>=minimum_rows
 return {'matched_rows':len(common),'all_12_columns_bitwise_equal':all(x[c]==y[c] for c in common)}
results['constraint_histories']=[]
for d in results['measurements']:
 original=ROOT/'science/N12-full-overlap-30M-20261114/segment-9089673' if d['label']=='candidate' else legacy.parent.parent
 h=history(next(original.glob('*.z4c_hyperboloidal.hst')),next((run/d['name']).glob('*.z4c_hyperboloidal.hst')),6);results['constraint_histories'].append(h)
l=[x for x in results['measurements'] if x['label']=='legacy'];c=[x for x in results['measurements'] if x['label']=='candidate']
results['total_wall_ratio']=sum(x['wall_seconds'] for x in c)/sum(x['wall_seconds'] for x in l)
results['paired_total_wall_ratios']=[c[0]['wall_seconds']/l[0]['wall_seconds'],c[1]['wall_seconds']/l[1]['wall_seconds']]
results['within_20_percent_both_pairs']=max(results['paired_total_wall_ratios'])<=1.20
results['validation_passed']=all(x['bitwise_equal'] for x in results['equivalence']) and results['original_13M_reproduction']['bitwise_equal'] and results['original_legacy_reproduction']['bitwise_equal'] and all(x['all_12_columns_bitwise_equal'] for x in results['constraint_histories'])
save();print(json.dumps(results),flush=True)
assert results['validation_passed'],'Strict bitwise qualification failed; all evidence retained'
