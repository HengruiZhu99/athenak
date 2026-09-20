#!/usr/bin/env python3
"""Test static AMR floor, exact vacuum transfers and nonzero MPI response."""
import argparse,json,os,subprocess,sys,re
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--exe',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
source=Path(__file__).resolve().parents[2]/'analysis/tde_revalidation';sys.path.insert(0,str(source))
from validate_checkpoint import validate
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);exe=a.exe.resolve()
region='\n<refined_region0>\nlevel = 1\nx1min = -4\nx1max = 4\nx2min = -4\nx2max = 4\nx3min = -4\nx3max = 4\n'
results={}
for case,ranks,floor in [('zero',1,True),('zero',2,True),('zero',2,False),('pulse',2,True)]:
 name=case+'-'+str(ranks)+'-'+str(int(floor));d=out/name;d.mkdir()
 text=(source/('preflight-'+case+'.athinput')).read_text()+region
 if not floor:text=text.replace('amr_static_halfwidth = 4','amr_static_halfwidth = 0')
 (d/'input.athinput').write_text(text)
 with(d/'run.log').open('w')as log:
  r=subprocess.run(['mpiexec','-n',str(ranks),str(exe),'-i',str(d/'input.athinput'),'-d',str(d)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 assert r.returncode==0,(name,r.returncode)
 v=validate(d,ranks,cycle=3,exact_zero=case=='zero');assert v['passed'],name
 assert v['blocks']==(64 if floor else 8),(name,v['blocks'])
 assert v['residual_exactly_zero']==(case=='zero'),name
 if case=='pulse':
  def history(file):
   lines=(d/file).read_text().splitlines();names=re.findall(r'\[\d+\]=(\S+)',next(x for x in lines if '[1]='in x));return dict(zip(names,np.loadtxt(d/file,ndmin=2)[-1]))
  h=history('ks_background.user.hst');c=history('ks_background.z4c.user.hst')
  peak=max(v['theta_regions'][k]['peak']['absolute_Theta']for k in ['horizon_interior','horizon_exterior'])
  assert np.isclose(peak,h['Theta-max'],rtol=1e-12,atol=0)
  assert np.isclose(v['theta_regions']['horizon_exterior']['proper_volume_Theta_RMS'],(c['Theta-norm']/c['Volume'])**.5,rtol=1e-11,atol=0)
 (d/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
 results[name]={k:v[k]for k in ['passed','blocks','time_M','residual_exactly_zero','invalid_metric_cells_including_ghosts']}
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
