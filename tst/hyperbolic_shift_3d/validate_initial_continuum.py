"""Analytic/AD initial constraint reference; analysis only, not production pgen."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tst/elliptic_shift_1d'))
from wormhole_isotropic import IsotropicWormhole

def run(executable,out):
 out.mkdir(parents=True,exist_ok=True)
 w=IsotropicWormhole(1/3,192)
 r=np.unique(np.r_[np.linspace(.005,.999,501),np.linspace(.35,.71,401),.063])
 q,d,dd,_=w.data(r);file=out/'analytic-jets.txt'
 with file.open('w') as f:
  f.write(str(len(r))+'\n');np.savetxt(f,np.c_[r,q,d,dd],fmt='%.17g')
 with (out/'continuum.csv').open('w') as f:
  subprocess.run([str(executable),str(file)],stdout=f,check=True)
 a=np.genfromtxt(out/'continuum.csv',delimiter=',',names=True);result={}
 for region,mask in [('all',np.ones(len(a),dtype=bool)),('exterior',a['r']>.063)]:
  result[region]={}
  for name in ['H','M','Mr']:
   rows=a[mask];idx=np.argmax(abs(rows[name]));result[region][name]=dict(maximum=float(abs(rows[name][idx])),radius=float(rows['r'][idx]))
 assert all(v['maximum']<1e-8 for v in result['exterior'].values())
 (out/'continuum-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.executable.resolve(),a.output)
