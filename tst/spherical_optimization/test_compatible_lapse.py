#!/usr/bin/env python3
"""Default parity and independent smooth-jet checks for composed lapse trial."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel
from evolution import make_evolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--before',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((a.before/'config.json').read_text());saved=np.load(a.before/'before.npz');k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg);assert np.array_equal(e.rhs(saved['state']),saved['rhs'])
 for key in ('d','dd'):assert np.array_equal(e.last[key],saved[key])
 e1=make_evolution(k,dict(cfg,compatible_lapse=1.));assert np.array_equal(e1.rhs(e1.initial),e.rhs(e.initial))
 q=e.materialize(saved['state']);d,dd=e.spatial_jets(q);d1,dd1=e1.spatial_jets(q);assert np.array_equal(d,d1);assert np.array_equal(dd[:,np.arange(12)!=8],dd1[:,np.arange(12)!=8]);assert np.array_equal(dd[-3:],dd1[-3:])
 rows=[]
 for n in (64,128,256):
  trial=make_evolution(k,dict(cfg,n=n,compatible_lapse=1.));q=trial.materialize(trial.initial);q[:,8]+=.01*np.sin(12*trial.r);d,dd=trial.spatial_jets(q);mask=(trial.r>trial.compatible_window[1])&(trial.r<trial.compatible_window[2]);error=float(max(abs(dd[mask,8]-trial.reference_dd[mask,8]+1.44*np.sin(12*trial.r[mask]))));rows.append(dict(n=n,error=error))
 orders=[float(np.log2(rows[i]['error']/rows[i+1]['error'])) for i in range(2)];assert min(orders)>5.5
 out=dict(default_rhs_and_jets_bitwise_equal=True,initial_rhs_bitwise_equal=True,only_lapse_second_jet_changed=True,last_three_jets_unchanged=True,window=e1.compatible_window.tolist(),smooth_second_derivative=rows,orders=orders,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));k.close()
if __name__=='__main__':main()
