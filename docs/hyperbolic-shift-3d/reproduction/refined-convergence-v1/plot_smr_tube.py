"""Actual native octree sections, with a qualified reference displacement overlay."""
from pathlib import Path
import json,struct,importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from matplotlib.colors import Normalize
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
spec=importlib.util.spec_from_file_location('reader',D/'check_smr_restart.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
a,raw,ages,arrays=reader.read(str(R/'output/refined-convergence/radial18-resolution-check/n12/state'))
blocks=[]
for i in range(a['leaves']):
 x,y,z,level=struct.unpack_from('<4i',raw,16*i);level-=2;h=.8/(4*2**level);lo=[-.4+h*q for q in [x,y,z]];blocks.append(dict(level=level,lo=lo,hi=[v+h for v in lo]))
tube=json.loads((D/'smr-tube-checks/topology-mpi1.json').read_text())['blocks'];assert len(blocks)==232 and len(tube)==400
k=.10116863541976967;drift=k*.1*10/3
fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
colors=plt.get_cmap('Blues');norm=Normalize(0,3)
for ax,data,title in zip(axes,[blocks,tube],['Original spherical target: 232 leaves','Symmetric tube target: 400 leaves']):
 for b in data:
  if b['lo'][2]<=.001<b['hi'][2]:ax.add_patch(Rectangle((b['lo'][0],b['lo'][1]),b['hi'][0]-b['lo'][0],b['hi'][1]-b['lo'][1],facecolor=colors(.2+.7*b['level']/3),edgecolor='gray',lw=.3))
 ax.plot([-drift,drift],[0,0],color='darkorange',lw=2,label='Reference ±kv(10/3), not live trajectory')
 for x in [-drift,drift]:ax.add_patch(Circle((x,0),.0168624312788484+4*(.025/12),fill=False,color='darkorange',ls='--',lw=1))
 ax.set(xlim=(-.205,.205),ylim=(-.205,.205),aspect='equal',xlabel='Compact x',ylabel='Compact y',title=title)
 ax.set_xticks([-.2,-.1,0,.1,.2]);ax.set_yticks([-.2,-.1,0,.1,.2]);ax.legend(fontsize=7,loc='upper left')
fig.suptitle('Nominal Cartesian block section near z=0; finest spacing unchanged\nDashed circles: initial boosted horizon enclosing radius + four N12 fine cells')
O=D/'smr-tube-checks'
for ext in ['png','pdf']:fig.savefig(O/('grid-coverage.'+ext),dpi=170)
