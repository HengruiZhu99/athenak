#!/usr/bin/env python3
"""Check default-A preservation, passive signed histories, and retired mode rejection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from restart_roundtrip import run,compare
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--baseline-binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
repo=Path(__file__).resolve().parents[2]
text=(repo/'docs/z4c_hyperboloidal/evidence/scri-investigation/baseline.athinput').read_text()
text+='\n<time>\nnlim = 4\n<output1>\nhyperboloidal_scri = true\nhyperboloidal_scri_investigation = false\ndcycle = 1\ndt = -1\n<output2>\ndt = 100\n<z4c_hyperboloidal>\nscri_treatment = A\n'
path=out/'input.athinput';path.write_text(text)
results={}
def checkpoint(name,binary,opts,ranks=2):
 print('Running',name,flush=True)
 return run(binary.resolve(),out,name,ranks,['-i',str(path)]+opts)
legacy=checkpoint('snapshot-A',args.baseline_binary,[])
a=checkpoint('A-off',args.binary,[]);results['default_A']=compare(legacy,a)
b=checkpoint('A-on',args.binary,['output1/hyperboloidal_scri_investigation=true']);results['passive']=compare(a,b)
c=checkpoint('A-window',args.binary,['output1/hyperboloidal_scri_investigation=true','output1/dcycle=2']);results['cadence_passive']=compare(a,c)
for group in range(4):
 fine=np.loadtxt(out/f'A-on/single_hole_20M.scri-investigation-{group}.hst',ndmin=2)
 coarse=np.loadtxt(out/f'A-window/single_hole_20M.scri-investigation-{group}.hst',ndmin=2)
 assert np.isfinite(fine).all()
 for row in coarse:
  idx=np.argmin(abs(fine[:,0]-row[0]));assert abs(fine[idx,0]-row[0])<1e-14
  block=fine[idx:idx+1,2:] if idx==0 else fine[max(1,idx-1):idx+1,2:]
  for col in range(block.shape[1]):
   absolute=20*group+col; minimum=absolute<56 and absolute%2==0
   expected=np.min(block[:,col]) if minimum else np.max(block[:,col])
   np.testing.assert_allclose(row[2+col],expected,rtol=1e-13,atol=1e-28)
results['signed_windows']=True
for mode in ('B', 'C'):
    run(args.binary.resolve(), out, 'reject-' + mode, 2,
        ['-i', str(path), 'z4c_hyperboloidal/scri_treatment=' + mode],
        fail='Only scri treatment A')
results['mode_rejection'] = True
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
