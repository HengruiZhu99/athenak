import sys,json
from pathlib import Path
sys.path.insert(0,'/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910')
from criterion import read_history
root=Path('/pscratch/sd/h/hzhu/chi-truncation-amr-20260910');report=[]
for n,limit in [(128,8),(512,6)]:
 for tag,d in [('old',root/'production'/f'N{n}'),('fixed',root/'validation_symmetry'/f'N{n}')]:
  if not (d/'amr_history.jsonl').exists():continue
  counts=[];created=deleted=0;last=None
  for line in (d/'amr_history.jsonl').open():
   try:e=json.loads(line)
   except json.JSONDecodeError:break
   if e['type']=='header':hdr=e;continue
   if e['type']!='event':continue
   t=float(e['time'])
   if t>limit+.01:break
   leaves=set(map(tuple,e['leaves']));m=sum((lev,x,hdr['root_blocks'][1]*2**(lev-hdr['root_level'])-1-y,z) not in leaves for lev,x,y,z in leaves)
   counts.append((t,m));created+=e.get('created',0);deleted+=e.get('deleted',0);last=e
  files=list(d.glob('*.hst'));h=read_history(files[0]) if files else []
  h=[x for x in h if x['time']<=limit+.01];tail=h[-1] if h else {}
  item=dict(N=n,version=tag,time=tail.get('time'),first_asymmetric=next((a for a in counts if a[1]),None),max_unmatched=max([a[1] for a in counts],default=0),events=len(counts),created=created,deleted=deleted,final_blocks=tail.get('nmb_total'),C2=tail.get('C-norm2'),min_lapse=tail.get('minLapse'),status=(d/'run-status').read_text().strip() if (d/'run-status').exists() else None)
  report.append(item)
(root/'validation_symmetry'/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
