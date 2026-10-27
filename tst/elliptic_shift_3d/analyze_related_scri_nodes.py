#!/usr/bin/env python3
"""Localize all original pole components of the actual N10 midpoint."""
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();src=a.input.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);m=json.loads((src/'manifest.json').read_text());data=np.genfromtxt(src/'poles.csv.gz',delimiter=',',names=True);names=['chi','Khat','Theta','alpha','Axx','Axy','Axz','Ayy','Ayz','Azz','LambdaX','LambdaY','LambdaZ'];summary=dict(scope='Same actual N10 midpoint; node8 rejected, no new evolution/solve; local midpoint pole is distinct from final-accepted surface maximum',manifest_sha256=hashlib.sha256((src/'manifest.json').read_bytes()).hexdigest(),result=m['result'],condition=m['independent_condition_diagnostic'],component_maxima={},all_rows_retained=True,scientific_milestone_complete=False)
for method in m['result']['methods']:
 if not method['supported']:continue
 q=data[data['nodes']==method['nodes']];assert len(q)==600
 per={}
 for name in names:
  i=int(np.argmax(abs(q[name])));row=q[i];per[name]=dict(maximum=float(abs(row[name])),signed=float(row[name]),row=int(row['row']),face=int(row['face']),j=int(row['j']),k=int(row['k']),xyz=[float(row[f]) for f in 'xyz'])
 summary['component_maxima'][str(method['nodes'])]=per
 i=int(np.where(q['row']==method['row'])[0][0]);row=q[i];component=method['component'];prefix='Lambda'+str('XYZ'.index(component[-1]))+'_' if component.startswith('Lambda') else component+'_';summary['dominant_decomposition']={f:float(row[f]) for f in q.dtype.names if f.startswith(prefix)}
np.savez_compressed(out/'all-midpoint-poles.npz',columns=np.array(data.dtype.names),values=np.column_stack([data[f] for f in data.dtype.names]));(out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'.gitignore').write_text('/plot-cache/\n');os.environ['MPLCONFIGDIR']=str(out/'plot-cache');os.environ['XDG_CACHE_HOME']=str(out/'plot-cache');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained');q=data[data['nodes']==7]
for name in ['Axx','Ayy','Azz','LambdaX','LambdaY','LambdaZ']:ax[0].semilogy(q['row'],abs(q[name]),label=name,alpha=.8,lw=.8)
ax[0].axhline(1e-7,color='black',ls='--',label='Pole target1e-7');ax[0].set(xlabel='All600 actual surface rows',ylabel='|original pole numerator|',title='N10 midpoint t=.0001, Hermite7');ax[0].legend(fontsize=7,ncol=2)
parts=summary['dominant_decomposition'];keys=[k for k in parts if not k.endswith('sum_error')];ax[1].bar(range(len(keys)),[parts[k] for k in keys]);ax[1].set_xticks(range(len(keys)));ax[1].set_xticklabels([k.split('_',1)[1] for k in keys],rotation=45,ha='right');ax[1].set_ylabel('Additive LambdaZ pole contribution');ax[1].set_title('Maximum at row533;8nodes rejected by condition guard');fig.savefig(out/'midpoint-pole-localization.png',dpi=160);fig.savefig(out/'midpoint-pole-localization.pdf');plt.close(fig);print(json.dumps(dict(component_maxima=summary['component_maxima'],dominant_decomposition=summary['dominant_decomposition'])))
