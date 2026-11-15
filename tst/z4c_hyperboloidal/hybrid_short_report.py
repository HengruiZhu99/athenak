#!/usr/bin/env python3
"""Read-only report for the six matched short single-hole runs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from restart_roundtrip import parse,payload

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('runs',type=Path);a=p.parse_args()
 reports=json.loads((a.runs/'results.json').read_text());assert len(reports)==6
 modes=['legacy','masked','agreement'];colors=['#555555','#166b95','#bb542c']
 result={'scope':'short single-hole check; not late-time acceptance','cases':{},'resolution_ratios':{},'script_sha256':sha(Path(__file__))}
 fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for row in reports:
  n,mode=row['n'],row['mode'];d=a.runs/('N'+str(n)+'-'+mode)
  assert row['exit_code']==0
  for f,digest in row['history_sha256'].items():assert sha(d/f)==digest
  h=np.atleast_2d(np.loadtxt(d/'single_hole.z4c_hyperboloidal.hst'))
  b=np.atleast_2d(np.loadtxt(d/'single_hole.scri.hst'))
  assert np.isfinite(h).all() and np.isfinite(b).all() and abs(h[-1,0]-.005633304448274094)<1e-15
  info=max((parse(f) for f in (d/'rst').glob('*.rst')),key=lambda r:r['cycle'])
  u=payload(info);chi_counts=np.zeros(3)
  for block in u:
   assert np.isfinite(block).all();chi_counts+=block[:,33:36].sum(axis=0)
  report={'final_constraints_native':h[-1].tolist(),'max_scri_native':abs(b[:,2:]).max(axis=0).tolist(),
          'all_evolved_and_scri_cumulative_chi_counts':chi_counts.tolist(),'checkpoint':info['path'],
          'checkpoint_sha256':sha(Path(info['path'])),'input_sha256':row['input_sha256'],
          'executable_sha256':row['executable_sha256']}
  if mode!='legacy':
   o=np.atleast_2d(np.loadtxt(d/'single_hole.overlap.hst'))
   assert np.isfinite(o).all() and (o[:,8]<=0).all()
   if mode=='masked':assert (o[:,8]==0).all()
   report['final_overlap_native']=o[-1].tolist()
  result['cases'][d.name]=report
  for ax,col,label,factor in zip(axes.flat,[8,9,10,11],['H','M',r'\Theta','Z'],[1/9,1/9,1/3,1/3]):
   ax.plot(h[:,0]*3,h[:,col]*factor,color=colors[modes.index(mode)],ls='-' if n==12 else '--',label=mode+' N'+str(n))
   ax.set(xlabel=r'$t/M_0$',ylabel=(r'$M_0^2 '+label+r'_{\rm RMS}$' if col<10 else r'$M_0 '+label+r'_{\rm RMS}$'),title='Horizon-exterior '+('Theta' if col==10 else label))
   ax.grid(alpha=.2)
 for mode in modes:
  lo=np.array(result['cases']['N12-'+mode]['final_constraints_native']);hi=np.array(result['cases']['N16-'+mode]['final_constraints_native']);ratio=lo[8:12]/hi[8:12]
  result['resolution_ratios'][mode]={'exterior_H_M_Theta_Z_N12_over_N16':ratio.tolist(),'apparent_orders':(np.log(ratio)/np.log(16/12)).tolist()}
 result['limitations']=['Only 0.0168999133 M0; no late-time stability claim.','Two resolutions and norm ratios do not establish a full convergence order.','N12 uses v2 and N16 includes v3 AMR/initialization-receipt updates; steady-stage arithmetic is unchanged; each executable is hashed.','Single-hole geometry Rs=.08 Rc=.13 differs from head-on geometry Rs=.18 Rc=.25.']
 axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle('Short single-hole check: three schemes, two resolutions')
 fig.savefig(a.runs/'constraints-short.png',dpi=180)
 (a.runs/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'complete':True,'chi_counts':{k:v['all_evolved_and_scri_cumulative_chi_counts'] for k,v in result['cases'].items()}},indent=2))
if __name__=='__main__':main()
