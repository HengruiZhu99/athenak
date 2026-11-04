"""Actual candidate SMR topology and geometric support; not boosted evolution."""
from pathlib import Path
import json,hashlib,shutil,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists());I=R/'output/refined-convergence/smr-expanded-topology';O=D/'expanded-smr-topology';O.mkdir(exist_ok=True)
base=json.loads((D/'refinement-coverage/verification.json').read_text());k=float(base['identity'].split(';k=')[1].split(';')[0]);M=1/3;drift=k*.1*10*M;results={};shapes={}
for name,folder,topfile,command,log in [('cube','capacity2048','radius005.json','radius005-command.json','radius005.log'),('offset-tube','tube','topology.json','command.json','native.log'),('symmetric-tube','symmetric-tube','topology.json','command.json','native.log')]:
 P=I/folder;x=json.loads((P/topfile).read_text());cmd=json.loads((P/command).read_text());build=json.loads((P/'build.json').read_text());assert cmd['returncode']==0 and hashlib.sha256((P/'probe-source.cpp').read_bytes()).hexdigest()==build['source_sha256'];assert x['n']==16 and x['leaves']==len(x['blocks'])
 lo=np.array([b['lo'] for b in x['blocks']]);hi=np.array([b['hi'] for b in x['blocks']]);levels=np.array([b['level'] for b in x['blocks']]);vol=np.prod(hi-lo,axis=1);assert abs(vol.sum()-.8**3)<1e-12
 for i in range(len(lo)):
  delta=np.minimum(hi[i],hi[i+1:])-np.maximum(lo[i],lo[i+1:]);assert not np.any(np.all(delta>1e-12,axis=1))
 fine=levels==3;fl=lo[fine].min(axis=0);fh=hi[fine].max(axis=0);assert abs(vol[fine].sum()-np.prod(fh-fl))<1e-12
 spacing=(hi[fine][0]-lo[fine][0])/16;assert np.max(abs(spacing-spacing[0]))<1e-14;buffer=4*spacing[0]
 # A sphere plus four fine cells fits iff every coordinate extent fits this
 # proven rectangular finest union. Reference +/- drift is not live tracker.
 margins={}
 for sign in (-1,1):
  center=np.array([sign*drift,0,0]);margins[str(sign)]=float(min(np.min(center-fl),np.min(fh-center))-buffer)
 results[name]=dict(leaves=len(lo),level_counts={str(l):int(n) for l,n in collections.Counter(levels.tolist()).items()},finest_lower=fl.tolist(),finest_upper=fh.tolist(),finest_dx=float(spacing[0]),four_cell_buffer=float(buffer),max_enclosing_radius_at_reference_10M=margins,nominal_core_cell_ratio_to_original=len(lo)/232,nominal_total_cell_ratio_with_radial18=(len(lo)+108)/(232+108),actual_four_node_capacity_blocks_per_rank=32,ceil_leaves_per_rank_48=(len(lo)+47)//48,qualified_volume_disjointness_and_rectangular_finest_union=True)
 shapes[name]=(fl,fh)
 q=O/name;q.mkdir(exist_ok=True)
 for f in (topfile,command,log,'build.json','probe-source.cpp'):shutil.copy2(P/f,q/f)
# Preserve original capacity rejections; these are analysis allocation failures.
for name in ('radius-0.05','radius-0.075'):
 for ext in ('.command.json','.log'):shutil.copy2(I/(name+ext),O/(name+ext))
record=dict(reference_x_at_10M=drift,original_leaves=232,original_finest_cube=[-.05,.05],original_max_enclosing_radius_at_positive_reference_10M=.05-drift-base['four_cell_buffer'],cases=results,qualification='Actual N16 AthenaK octree balance/redistribution and native field-transfer lifecycle, MPI2 topology probes, no evolution or admitted physics checkpoint. Wider cube required analysis capacity2048/ranks+8 after original local two-rank capacity rejection; production capacity not changed. Box variants use analysis-only rectangular flags with existing Regrid API and are not implemented as production runtime policies yet. Finest union proven rectangular/disjoint with total core volume. Reference +/- k*v*t used only for coverage estimates; actual gauge motion, horizon shape and gradients must be measured. Nominal cell-count ratios are not memory, runtime or speedup measurements. Same fine spacing is coverage expansion, not resolution refinement.')
(O/'analysis.json').write_text(json.dumps(record,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for color,(name,(fl,fh)) in zip(('C0','C1','C2'),shapes.items()):
 from matplotlib.patches import Rectangle
 axes[0].add_patch(Rectangle((fl[0],fl[1]),fh[0]-fl[0],fh[1]-fl[1],fill=False,label=name,lw=2,edgecolor=color))
axes[0].add_patch(Rectangle((-.05,-.05),.1,.1,fill=False,ls=':',color='black',label='original'))
axes[0].plot([-drift,drift],[0,0],'k.',label='reference +/- drift at10M');axes[0].set(xlim=(-.12,.12),ylim=(-.12,.12),xlabel='Compact x',ylabel='Compact y',aspect='equal');axes[0].legend(fontsize=7)
labels=['original']+list(results);axes[1].bar(labels,[232]+[results[n]['leaves'] for n in results]);axes[1].set(ylabel='Cartesian leaves',title='Actual balanced topology, same finest spacing');axes[1].tick_params(axis='x',labelrotation=20)
fig.suptitle('Candidate static coverage for boosted follow-up: topology only');fig.savefig(O/'coverage-cost.png',dpi=170);fig.savefig(O/'coverage-cost.pdf');plt.close(fig)
print(json.dumps(record,indent=2))
