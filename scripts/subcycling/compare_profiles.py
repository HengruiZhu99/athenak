"""Compare complete restart payloads, histories and AMR after instrumented restarts."""
from pathlib import Path
import json,csv,sys,hashlib,math
ROOT=Path('/pscratch/sd/h/hzhu/vc-subcycling-20260911')
sys.path.insert(0,'/pscratch/sd/h/hzhu/n128-chite-bisection-20260910')
from criterion import read_history

def payload_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  prefix=f.read(1024*1024);mark=prefix.index(b'<par_end>\n')+len(b'<par_end>\n')
  h.update(prefix[mark:])
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
results={}
for tag in ['early','late']:
 a=ROOT/'profiles'/(tag+'_baseline');b=ROOT/'profiles'/(tag+'_profile')
 ha=read_history(next(a.glob('*.hst')));hb=read_history(next(b.glob('*.hst')))
 assert len(ha)==len(hb) and ha[-1]['time']==hb[-1]['time']
 differences={k:max(abs(x[k]-y[k]) for x,y in zip(ha,hb)) for k in ha[0]}
 restarts=[sorted((p/'rst').glob('*.rst'))[-1] for p in [a,b]]
 hashes=[payload_hash(p) for p in restarts]
 same_tree=(a/'amr_history.jsonl').read_bytes()==(b/'amr_history.jsonl').read_bytes()
 rows=list(csv.DictReader((b/'execution_profile.rank0.csv').open()))
 timing={r['region']:{k:float(v) for k,v in r.items() if k!='region'} for r in rows}
 item=dict(final_time=ha[-1]['time'],history_max_abs_differences=differences,restart_payload_hashes=hashes,all_restart_payload_bytes_equal=hashes[0]==hashes[1],amr_history_identical=same_tree,profile=timing)
 results[tag]=item
(ROOT/'profile_comparison.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
if any(not r['all_restart_payload_bytes_equal'] or not r['amr_history_identical'] for r in results.values()):
 raise SystemExit('Full-state comparison requires investigation')
