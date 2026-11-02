import json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
data={(c,m):json.loads((p/f'{c}-{m}.json').read_text()) for c in ['c','d'] for m in ['native','first4','first6']}
for c,mode,old in [('c','first4','abc-versus-sg1'),('d','first6','abcd-versus-sg1')]:
 a=data[c,'native'];b=data[c,mode];o=json.loads((p.parent/old/f'{c}-radial.json').read_text())
 for key in ['norms','radial_cumulative']:
  assert a[key]==b[key] and a[key]==o[key],(c,key)
def region(d,lo,hi):
 a=next(x for x in d['radial_cumulative'] if abs(x['cutoff']-lo)<1e-10);b=next(x for x in d['radial_cumulative'] if abs(x['cutoff']-hi)<1e-10)
 return {k:a[k]-b[k] for k in ['volume','H2','M2','Mr2']}
result={'scope':'Analysis-only common bulk outer first derivative sampling; native boundary refresh and donor-seven last-three-row derivative closure retained per case. Not a full common boundary operator or a resolution-convergence test.', 'native_reproduction_exact':True,'cases':{},'D_vs_C_percent':{}}
for (c,m),d in data.items():result['cases'][f'{c}-{m}']={'norms':d['norms'],'outer_0775_09':region(d,.775,.9),'inner_016_03':region(d,.16,.3)}
for m in ['native','first4','first6']:
 c=result['cases'][f'c-{m}'];d=result['cases'][f'd-{m}'];result['D_vs_C_percent'][m]={'exterior':{k:100*(d['norms']['exterior'][k]/c['norms']['exterior'][k]-1) for k in ['H_L2','M_L2','Mr_L2']},'outer_squared':{k:100*(d['outer_0775_09'][k]/c['outer_0775_09'][k]-1) for k in ['H2','M2','Mr2']}}
(p/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,field,title in zip(axes,['H2','M2'],['Hamiltonian','Momentum magnitude']):
 for c in ['c','d']:
  for mode,ls in [('first4','-'),('first6','--')]:
   rows=data[c,mode]['radial_cumulative'];x=[];y=[]
   for a,b in zip(rows,rows[1:]):
    if b['volume']<a['volume']:
     x.append((a['cutoff']+b['cutoff'])/2);y.append(math.sqrt(max(0,(a[field]-b[field])/(a['volume']-b['volume']))))
   ax.plot(x,y,ls,label=f'{c.upper()} state, {mode}',color={'c':'#2171b5','d':'#d95f02'}[c])
 ax.set(xlim=(.55,1),ylim=(2e-4,.15),yscale='log',xlabel='Compactified radius',ylabel='Regional RMS',title=title);ax.axvspan(.775,.9,color='grey',alpha=.12);ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.suptitle('10M N16 endpoints: common bulk sampling, native boundary closure retained')
fig.savefig(p/'comparison.png',dpi=170)
print(json.dumps(result['D_vs_C_percent'],indent=2))
