#!/usr/bin/env python3
"""Export a single-grid puncture pilot, with an explicit cutoff for interim evidence.

Validate exact frozen bindings, checkpoint ancestry, and saved-state RMS norms.
Never interpret reaching a target or a sampled norm as a convergence result.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
sys.path.insert(0,str(ROOT/'tst/elliptic_shift_1d'))
from analyze_telegraph_time import numerical_binding
from telegraph_analysis_io import validate_telegraph_profile
from horizon import apparent_horizons
case=ROOT/'output/elliptic-shift-1d/telegraph-long-puncture-500-v1/n1025'
parser=argparse.ArgumentParser();parser.add_argument('--through',type=float,help='Explicit saved-time cutoff for a running pilot');parser.add_argument('--output',type=Path,help='New export directory; defaults to the paper data directory');parser.add_argument('--case',type=Path,default=case);parser.add_argument('--protocol',type=Path);a=parser.parse_args();case=a.case.resolve()
manifest_bytes=(case/'manifest.json').read_bytes();m=json.loads(manifest_bytes)
assert m['status'] in ['finished','failed'] or a.through is not None,'Running exports require an explicit cutoff'
assert a.through is None or 0<=a.through<=m['time']
cutoff=m['time'] if a.through is None else a.through
D=a.output if a.output is not None else P/'data/long-puncture';D.mkdir(parents=True,exist_ok=False)
receipts=[];folders=[]
def receipt(path,content=None):
 receipts.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes() if content is None else content).hexdigest()))
def ancestors(folder):
 assert folder not in folders,'Ancestry cycle'
 raw=manifest_bytes if folder==case else (folder/'manifest.json').read_bytes()
 mm=json.loads(raw);numerical_binding(folder,mm)
 if 'checkpoint_parent' in mm:
  cp=Path(mm['checkpoint_parent']['path']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==mm['checkpoint_parent']['sha256'];receipt(cp);ancestors(cp.parent)
 folders.append(folder);receipt(folder/'manifest.json',raw)
 (D/('manifest-parent.json' if folder!=case else 'manifest-long.json')).write_bytes(raw)
ancestors(case)
rows={};profiles={};prev=None;last_t=None;last_path=None
for folder in folders:
 raw_history=(folder/'history.jsonl').read_bytes()
 history={round(h['t'],9):h for h in map(json.loads,raw_history.splitlines())};receipt(folder/'history.jsonl',raw_history)
 for path in sorted(folder.glob('profile-*.npz'),key=lambda p:int(p.stem.rsplit('-',1)[1])):
  # Filename indices are report numbers; check the time before loading full arrays.
  with np.load(path,allow_pickle=False) as z:
   if float(z['t'])>cutoff+1e-10:continue
   d={k:z[k].copy() for k in z.files}
  validate_telegraph_profile(d);t=round(float(d['t']),9);r=d['r'];diag=d['diagnostic'];h=history[t]
  if t==last_t:
   for k in ['q','B','rays','q_physical','diagnostic']:np.testing.assert_array_equal(d[k],prev[k])
  else:assert last_t is None or t>last_t
  w=r*r*d['grid_jacobian'];C=diag[:,:4];norms=np.sqrt(np.sum(w[:,None]*C*C,axis=0)/sum(w))
  np.testing.assert_allclose(norms,[h[k+'_rms'] for k in ['H','M','Theta','Z']],rtol=2e-12,atol=1e-25)
  ah_check=apparent_horizons(r,diag[:,8],diag[:,11])
  for key in ['apparent_horizon_r','apparent_horizon_mass']:
   if h[key] is None:assert ah_check[key] is None
   else:np.testing.assert_allclose(ah_check[key],h[key],rtol=2e-13,atol=1e-15)
  i=int(np.argmax(abs(C[:,0])));ah=h['apparent_horizon_r'];outside=r>=ah if ah is not None else np.zeros_like(r,dtype=bool)
  rr=d['rays'][::2];active=(rr[:,0]>0)&(rr[:,0]<1);gaps=np.diff(rr[active,0]);O=(1-r*r)/2
  # The r=0 logarithm is an AF coefficient, not a positive-radius sample.
  # Mask before exponentiation: its growth can otherwise overflow an
  # unused temporary even when every physical sample remains well represented.
  positive=r>0;rp=r[positive];Op=O[positive]
  chi_positive=rp**4/(rp**4+Op**4)*np.exp(d['q'][positive,0])
  row=dict(t=t,**{k+'_rms':float(v) for k,v in zip(['H','M','Theta','Z'],norms)},H_max=float(abs(C[i,0])),H_peak_r=float(r[i]),H_outside_AH=float(max(abs(C[outside,0]))) if outside.any() else None,AH_r=ah,AH_mass_error=abs(h['apparent_horizon_mass']-1/3) if h['apparent_horizon_mass'] is not None else None,AF_mass_error=abs(h['AF_inner_mass']-1/3),MS_mass_error=float(max(abs(diag[r<1,10]-1/3))),scri_pole=h['scri_pole'],scri_null=h['scri_null'],scri_clock=h['scri_clock'],ray_min_logJ=h['ray_minus_min_logJ'],ray_min_gap=float(min(gaps)) if len(gaps) else None,min_alpha_positive_nodes=float(min(diag[positive,7])),min_chi_positive_nodes=float(min(chi_positive)))
  if outside.any():
   exterior_norms=np.sqrt(np.sum(w[outside,None]*C[outside]*C[outside],axis=0)/sum(w[outside]))
   row.update({k+'_exterior_rms':float(v) for k,v in zip(['H','M','Theta','Z'],exterior_norms)})
   row.update({k+'_exterior_max':float(max(abs(C[outside,j]))) for j,k in enumerate(['H','M','Theta','Z'])})
   row['MS_exterior_mass_error']=float(max(abs(diag[outside&(r<1),10]-1/3)))
  else:
   row.update({k+'_exterior_rms':None for k in ['H','M','Theta','Z']})
   row.update({k+'_exterior_max':None for k in ['H','M','Theta','Z']})
   row['MS_exterior_mass_error']=None
  rows[t]=row
  if t in [0,2.5,6,8,10,12,14,16,20,25,50,100,250,500]:profiles[t]=(r.copy(),diag[:,7].copy(),np.divide(diag[:,7],O,out=np.full_like(r,np.nan),where=O>0),C[:,0].copy())
  last_t=t;prev=d;last_path=path;receipt(path)
# The failed checkpoint may contain an attempted stage; use only validated saved profiles for plots.
r=prev['r'];O=(1-r*r)/2;profiles[last_t]=(r.copy(),prev['diagnostic'][:,7].copy(),np.divide(prev['diagnostic'][:,7],O,out=np.full_like(r,np.nan),where=O>0),prev['diagnostic'][:,0].copy())
arrays={}
for t,values in profiles.items():
 for k,v in zip(['r','alpha','physical','H'],values):arrays[f't{round(t*100):06d}_{k}']=v
np.savez_compressed(D/'profiles.npz',**arrays)
assert a.through is None or abs(last_t-a.through)<1e-9,'Missing requested snapshot'
result=dict(target=500,interim=m['status']=='running',status=m['status'],failure=m.get('failure'),integrator_time=m['time'],last_saved_time=last_t,resolution=m['arguments']['n'],parameters=m['arguments'],parent_checkpoint=m.get('checkpoint_parent'),profile_times=sorted(profiles),rows=list(rows.values()),scope='Single-resolution survival pilot; interim when marked running. Both full-domain diagnostics and a separately labeled apparent-horizon-exterior budget; exterior RMS is normalized by the exterior r^2 dr/dx weight sum. No evolution points are excised; saved states are validated independently. Failed attempted stages are not plotted. Survival does not establish accuracy, convergence, or continuum stability.')
(D/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
protocol=a.protocol.resolve() if a.protocol else ROOT/'docs/elliptic-shift-1d/telegraph-long-puncture-500-v1.json';receipt(protocol);shutil.copy2(protocol,D/'protocol.json')
receipt(case.parent/'launch.json');shutil.copy2(case.parent/'launch.json',D/'launch.json')
amendment=ROOT/'docs/elliptic-shift-1d/telegraph-long-puncture-exterior-budget-v1.json';receipt(amendment);shutil.copy2(amendment,D/'exterior-budget.json')
(D/'source-manifest.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['target','status','failure','integrator_time','last_saved_time','resolution']},indent=2))
