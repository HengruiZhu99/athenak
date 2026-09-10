from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/pscratch/sd/h/hzhu/chi-truncation-amr-20260910');fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for row,(n,limit) in enumerate([(128,8),(512,6)]):
 for tag,d,col in [('Original',root/'production'/f'N{n}','#df7e21'),('Fixed',root/'validation_symmetry'/f'N{n}','#2878b5')]:
  if not (d/'amr_history.jsonl').exists():continue
  t=[];m=[];churn=[];levels=[];total=0
  for line in (d/'amr_history.jsonl').open():
   try:e=json.loads(line)
   except json.JSONDecodeError:break
   if e['type']=='header':hdr=e;continue
   if e['type']!='event':continue
   if float(e['time'])>limit:break
   leaves=set(map(tuple,e['leaves']));t.append(float(e['time']));m.append(sum((lev,x,hdr['root_blocks'][1]*2**(lev-hdr['root_level'])-1-y,z) not in leaves for lev,x,y,z in leaves));total+=e.get('created',0)+e.get('deleted',0);churn.append(total);levels.append(e['max_level']-hdr['root_level'])
  if tag=='Fixed':
   log=d/'run-status'
   if log.exists() and log.read_text().strip()=='0' and t:t.append(limit);m.append(m[-1]);churn.append(churn[-1]);levels.append(levels[-1])
  for ax,v in zip(axs[row],[m,churn,levels]):ax.step(t,v,where='post',color=col,label=tag);ax.set(xlabel='Coordinate time',xlim=(0,limit));ax.grid(alpha=.25);ax.legend()
 for ax,label in zip(axs[row],['Unmatched reflected mesh leaves','Cumulative created + deleted blocks','Maximum physical refinement level']):ax.set(ylabel=label,title=f'N{n}')
fig.suptitle('Short fresh-data VC Cartoon validation | A = −0.04896875\nMeasured tree reflection symmetry; no forced reflection of fields or mesh')
fig.savefig(root/'validation_symmetry'/'comparison.png',dpi=170)
