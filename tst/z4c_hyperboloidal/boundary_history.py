#!/usr/bin/env python3
"""Check passive scri history, per-window maxima, and repartitioned restart."""
import argparse
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from restart_roundtrip import run,compare
parser=argparse.ArgumentParser()
parser.add_argument('--binary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
binary=args.binary.resolve()
fixture=(Path(__file__).resolve().parent/'n12_initialize.athinput').read_text()
base=fixture+'\n<time>\nevolution = dynamic\nintegrator = rk4_classical\ntlim = 1\nnlim = 8\n<output1>\nfile_type = hst\nhyperboloidal_scri = false\ndcycle = 1\ndata_format = %24.16e\n<output2>\nfile_type = rst\ndt = 100\nsingle_file_per_rank = false\n'
path=out/'input';path.write_text(base)
results={}
a=run(binary,out,'off',2,['-i',str(path)])
b=run(binary,out,'on',2,['-i',str(path),'output1/hyperboloidal_scri=true'])
results['passive']=compare(a,b)
c=run(binary,out,'coarse',2,['-i',str(path),'output1/hyperboloidal_scri=true','output1/dcycle=4'])
results['cadence_passive']=compare(a,c)
fine=np.loadtxt(out/'on/native_n12.scri.hst');coarse=np.loadtxt(out/'coarse/native_n12.scri.hst')
assert fine.shape[1]==22 and np.isfinite(fine).all()
for row in coarse:
 t=row[0]; idx=np.flatnonzero(np.isclose(fine[:,0],t,atol=1e-15,rtol=0))[0]
 if idx==0: expected=fine[0,2:]
 else: expected=np.max(fine[max(1,idx-3):idx+1,2:],axis=0)
 np.testing.assert_allclose(row[2:],expected,rtol=1e-14,atol=1e-30)
results['window_maxima']=True
r=run(binary,out,'restart',4,['-r',b['path'],'time/nlim=10','output1/last_time=-1'])
full=run(binary,out,'full10',4,['-i',str(path),'time/nlim=10','output1/hyperboloidal_scri=true'])
results['restart']=compare(r,full)
np.testing.assert_array_equal(fine[:9],np.loadtxt(out/'full10/native_n12.scri.hst')[:9])
results['history_2_vs_4_ranks_bitwise_equal']=True
results['rows']=fine.shape[0]
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
