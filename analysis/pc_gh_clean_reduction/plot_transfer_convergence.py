#!/usr/bin/env python3
"""Plot operator errors; no evolution or physical-stability claim."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
h=np.array([1/16,1/32,1/64])
for col,dim in enumerate([2,3]):
 residual_before=[];residual_after=[];curl_before=[];curl_after=[]
 for n in [8,16,32]:
  path=args.root/f'n{n}-smr'/f'fd6-{dim}d'
  records=[json.loads(line) for f in path.glob('transfer-mesh-rank*.jsonl') for line in f.read_text().splitlines()]
  records=[r for r in records if r['repeat']==0]
  residual_before.append(max(r['before_reference_residual_error'] for r in records))
  residual_after.append(max(r['after_reference_residual_error'] for r in records))
  curl_before.append(np.max([r['curl_tensor_error_before_max'] for r in records]))
  curl_after.append(np.max([r['curl_tensor_error_after_max'] for r in records]))
 for row,(old,new,label) in enumerate([(residual_before,residual_after,'Ghost residual error'),(curl_before,curl_after,'Active curl error')]):
  ax=axes[row,col]
  ax.loglog(h,old,'o--',color='#a66a26',label='Ordinary transfer')
  ax.loglog(h,new,'o-',color='#216a89',label='Residual transfer')
  power=(2 if dim==2 else 5)-row
  ax.loglog(h,new[0]*(h/h[0])**power,':',color='#888888',label=f'$h^{power}$ reference')
  ax.set_xticks(h,labels=['1/16','1/32','1/64'])
  ax.xaxis.set_minor_formatter(NullFormatter())
  ax.invert_xaxis();ax.grid(True,which='both',alpha=.2)
  ax.set_xlabel('Coarse spacing / M')
  ax.set_ylabel('Max '+label.lower()+(' [M⁻¹]' if row==0 else ' [M⁻²]'))
  if row==0:ax.set_title(f'{dim}D fixed refinement layout')
  if row==0 and col==0:ax.legend(fontsize=8)
fig.suptitle('FD6 periodic transfer fixture — seeded variable reductions\nOperator errors only; no evolution-stability conclusion',fontsize=13)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=170)
