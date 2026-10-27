#!/usr/bin/env python3
"""All-cell/region analysis of exact-initial-data transition derivative controls."""
import argparse,csv,hashlib,json,os,tempfile,time
from pathlib import Path
import numpy as np
from compare_coupled_family import norms
SOURCE=['H','M_norm','chi_rate','Khat_rate','Theta_rate','alpha_rate']+['metric_'+x for x in ['xx','xy','xz','yy','yz','zz']]+['A_'+x for x in ['xx','xy','xz','yy','yz','zz']]+['Lambda_'+x for x in 'xyz']+['flux_'+x for x in 'xyz']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--initial',type=Path,required=True);p.add_argument('--order',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(exist_ok=False);start=time.monotonic();hashes={};cases={};arrays={}
 def track(p):hashes[str(p)]=sha(p);return p
 scenarios=[('initial_ng2',a.initial/'diagnostic.csv',8,6)]
 if a.order:
  for name,n in [('n6_amr3',6),('n6_static',6),('n8_static',8),('n10_static',10)]:scenarios.append((name,a.order/name/'diagnostic.csv',5,n))
 for name,path,methods,n in scenarios:
  track(path)
  with path.open() as f:
   reader=csv.reader(f);header=next(reader);first=[next(reader) for _ in range(methods)];names=[row[header.index('method')] for row in first]
  cols=[i for i,f in enumerate(header) if f!='method'];fields=[header[i] for i in cols];data=np.loadtxt(path,delimiter=',',skiprows=1,usecols=cols);assert len(data)%methods==0 and np.all(np.isfinite(data));data=data.reshape(-1,methods,len(cols));lookup={f:i for i,f in enumerate(fields)}
  def channel(field,method=0):return data[:,method,lookup[field]]
  points=channel('point');assert np.array_equal(points,np.arange(len(points))) and np.all(data[:,:,lookup['point']]==points[:,None]);r=channel('r');w=channel('weight');face=channel('face');i,j,k=[channel(f) for f in ['i','j','k']];assert np.all(w>0);transition=(face>=0)&(r<.45);shell=face>=0;seams=(j<2)|(j>=n-2)|(k<2)|(k>=n-2)
  masks={'all':np.ones(len(points),bool),'puncture_r_lt_01':r<.1,'Cartesian':face<0,'transition':transition,'transition_inner_two_radial':transition&(i<2),'transition_interior_radial':transition&(i>=2),'transition_angular_seams':transition&seams,'other_shells':shell&~transition,'scri_r_ge_09':r>=.9}
  row={'cells':len(points),'n':n,'Penrose_volume':float(w.sum()),'methods':{}}
  for method,label in enumerate(names):
   result={'regions':{}}
   for region,mask in masks.items():
    if not mask.any():result['regions'][region]={'cells':0,'unavailable':True};continue
    ww=w[mask];result['regions'][region]={'cells':int(mask.sum()),'Penrose_volume':float(ww.sum()),'H':norms(channel('H',method)[mask],ww),'M':norms(channel('M',method)[mask],ww),'full_source_errors':{field:norms(channel('dq'+str(f),method)[mask],ww) for f,field in enumerate(SOURCE)},'positive_jet_componentmax_errors':{field:norms(channel(field,method)[mask],ww) for field in ['chi_d_error','chi_dd_error','alpha_d_error','alpha_dd_error']}}
   for target in [126]:
    if name in ['initial_ng2','n6_amr3']:
     p0=target;result['target126']={'point':p0,'xyz':[float(channel(c)[p0]) for c in ['x','y','z']],'H':float(channel('H',method)[p0]),'M':float(channel('M',method)[p0]),'positive_jet_componentmax_errors':{f:float(channel(f,method)[p0]) for f in ['chi_d_error','chi_dd_error','alpha_d_error','alpha_dd_error']},'source_errors':{field:float(channel('dq'+str(f),method)[p0]) for f,field in enumerate(SOURCE)}}
   row['methods'][label]=result
  # Raw complete source differences and scalar jets retained for independent
  # masks; the original CSV remains frozen as well.
  np.savez_compressed(out/(name+'-all-cell-source-errors.npz'),fields=np.array(fields),methods=np.array(names),data=data)
  cases[name]=row;arrays[name]=(r,{label:channel('H',m).copy() for m,label in enumerate(names)})
 summary={'scope':'Independent analytic INITIAL M1/3 wormhole jets and original tensor physics; no evolution/shift solve/adoption or scientific-pass claim.','physical_shift_control':'All source comparisons use the same independently refined radial shift jets, so derivative/source effects are separated from a numerical elliptic shift error.','source_columns':SOURCE,'weight_definition':'Original Penrose map-volume/chi^(3/2) weights, unchanged active values, every active cell retained. RMS/unnormalized L2/max and all regions retained. Other masks overlap and do not replace all.','oracle_contract':'Only interior halo values with0<r<.85 replaced analytically. Outer ghosts retain original scri treatment; exact full active jets are independently known for every cell.','order_contract':'Original supported ng2 baseline/cache embedded in ng3 sampling grid. Sixth-order replaces only transition chi/lapse jets; added third layer uses ordinary interpolation unless exact-halo control. First/second inverse-map J/H retained. Other fields and all other cells retain original baseline; this is not a global sixth-order scheme.','cases':cases,'scientific_acceptance':False}
 if a.order:
  summary['study_manifest']=json.loads(track(a.order/'manifest.json').read_text())
  summary['ng3_polynomial_controls']={}
  for name in ['n6_amr3','n6_static','n8_static','n10_static']:
   stdout=a.order/(name+'-diagnose.stdout');rows=[json.loads(line) for line in track(stdout).read_text().splitlines() if line.startswith('{')];summary['ng3_polynomial_controls'][name]=next(row for row in rows if row['phase']=='summary')
 with tempfile.TemporaryDirectory(prefix='transition-plot-',dir=out) as temporary:
  os.environ['MPLCONFIGDIR']=temporary;os.environ['XDG_CACHE_HOME']=temporary
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');base=cases['initial_ng2'];names=list(base['methods']);x=np.arange(len(names));axes[0].bar(x,[abs(base['methods'][n]['target126']['H']) for n in names]);axes[0].set_xticks(x,names,rotation=40,ha='right');axes[0].set_yscale('log');axes[0].set_title('Dominant point126: exact-jet substitutions');axes[0].set_ylabel('|H|')
  for label,row in base['methods'].items():axes[1].plot(['all','transition','puncture'],[row['regions'][r]['H']['penrose_volume_rms'] for r in ['all','transition','puncture_r_lt_01']],'.-',label=label)
  axes[1].set_yscale('symlog',linthresh=1e-12);axes[1].set_title('All-cell and overlapping region H RMS');axes[1].legend(fontsize=6);fig.suptitle('Exact initial wormhole diagnostic: Hessian stencil dominates; no evolution/adoption');fig.savefig(out/'transition-mechanism.png',dpi=150);plt.close(fig)
  if a.order:
   fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');selections=[('n6_static',6),('n8_static',8),('n10_static',10)];labels=['fourth_default','fourth_exact_halos','sixth_transition','sixth_transition_exact_halos'];colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
   for label,color in zip(labels,colors):
    ns=[n for _,n in selections]
    axes[0,0].semilogy(ns,[cases[k]['methods'][label]['regions']['transition']['H']['penrose_volume_rms'] for k,_ in selections],'.-',label=label,color=color)
    axes[0,1].semilogy(ns,[cases[k]['methods'][label]['regions']['transition']['positive_jet_componentmax_errors']['chi_dd_error']['penrose_volume_rms'] for k,_ in selections],'.-',color=color)
    axes[1,0].semilogy(ns,[cases[k]['methods'][label]['regions']['transition']['positive_jet_componentmax_errors']['alpha_dd_error']['penrose_volume_rms'] for k,_ in selections],'.-',color=color)
    axes[1,1].semilogy(ns,[cases[k]['methods'][label]['regions']['all']['H']['penrose_volume_rms'] for k,_ in selections],'.-',color=color)
   axes[0,0].legend(fontsize=7);axes[0,0].set_title('Transition H RMS');axes[0,1].set_title('Physical chi Hessian component-error RMS');axes[1,0].set_title('Physical lapse Hessian component-error RMS');axes[1,1].set_title('All-cell H RMS; core remains coarse')
   for ax in axes.flat:ax.set_xticks([6,8,10]);ax.set_xlabel('N');ax.grid(alpha=.2)
   fig.suptitle('Transition-only sixth-order fixed-stencil study\nNo global order/evolution claim; all-cell norms include every point');fig.savefig(out/'transition-order-controls.png',dpi=150);plt.close(fig)
 summary['inputs_sha256']=hashes;summary['inputs_unchanged']=all(sha(Path(p))==h for p,h in hashes.items());assert summary['inputs_unchanged'];summary['analysis_script_sha256']=sha(Path(__file__));summary['seconds']=time.monotonic()-start;(out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({name:{method:{'all_H':r['regions']['all']['H']['penrose_volume_rms'],'transition_H':r['regions']['transition']['H']['penrose_volume_rms']} for method,r in row['methods'].items()} for name,row in cases.items()}))

if __name__=='__main__':main()
