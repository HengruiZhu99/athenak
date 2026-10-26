#!/usr/bin/env python3
"""Reweight saved independent MP readbacks with the declared scientific measure."""
from pathlib import Path
import argparse,json,hashlib,ast
import numpy as np
p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path);a=p.parse_args();utility=Path(__file__).with_name('review_goal1_clock_pilots.py');tree=ast.parse(utility.read_text());ns={'np':np};exec(compile(ast.Module(body=[v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name in ['reference_weights','norm']],type_ignores=[]),str(utility),'exec'),ns);norm=ns['norm'];out={'scope':'Reweight already saved independent80digit native target readbacks with fixed-reference Penrose measure; no repeated MP generation or PDE trajectory.','cases':{}}
for name in ['independent-clock-pulse4','independent-clock-t002','independent-clock-fine-t002','independent-clock-fine-t03']:
 work=a.work/name;review=json.loads((work/'review.json').read_text());case=Path(review['case'])
 with (case/'cubed_sphere_native_last_jets.bin').open('rb')as f:n,c=np.fromfile(f,np.uint64,2);x=np.fromfile(f,np.float64).reshape(int(n),int(c))
 arrays=[]
 for file in ['independent-targets.bin','runtime-targets.bin']:
  with (work/file).open('rb')as f:nn,cc=np.fromfile(f,np.uint64,2);arr=np.fromfile(f,np.float64).reshape(-1,22,13)
  assert nn==n and cc==286;arrays.append(arr)
 exact,runtime=arrays;native=x[:,14:].reshape(-1,22,13);config=[0,1,2,3,4,5,6,18,19,20,21];amp=.0005;L=.2;scale=np.full(22,amp/L);scale[config]=amp;w=ns['reference_weights'](x)['fixed_reference_Penrose'];entry={}
 for region,mask in [('all',np.ones(len(x),bool)),('bulk',x[:,13]==0),('shell',x[:,13]==1)]:
  quantities=[]
  for difference in [(native-exact)/scale[None,:,None],(runtime-exact)/scale[None,:,None]]:quantities.append({k:norm(v,w,mask)for k,v in [('fields',difference[:,:,0]),('first',L*difference[:,:,1:4]),('configuration_second',L*L*difference[:,config,4:])]})
  error,uncertainty=quantities;ratio={k:{metric:uncertainty[k][metric]/max(error[k][metric],1e-300)for metric in ['rms','linf']}for k in error};entry[region]={'native_exact_errors':error,'runtime_oracle_uncertainty':uncertainty,'fraction':ratio}
 out['cases'][name]={'time':review['time'],'active_nodes':int(n),'fixed_reference_Penrose':entry}
maximum=max(v for case in out['cases'].values()for region in case['fixed_reference_Penrose'].values()for metrics in region['fraction'].values()for v in metrics.values());out.update({'max_oracle_fraction_of_native_error':maximum,'all_below_10percent':maximum<.1,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reused_norm_source_sha256':hashlib.sha256(utility.read_bytes()).hexdigest()});a.output.write_text(json.dumps(out,indent=2)+'\n');assert out['all_below_10percent'];print(json.dumps({'all_below_10percent':True,'maximum':maximum,'readbacks':len(out['cases'])}))
