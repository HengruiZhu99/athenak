"""Bounded t0 controls; no evolution or checkpoint writes."""
import subprocess,json,pathlib,time,hashlib
ROOT=next(p for p in pathlib.Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
out=ROOT/'output/refined-convergence/endpoint-diagnostic';binary=out/'reader';base=ROOT/'output/refined-convergence';config=ROOT/'inputs/hyperbolic_shift_3d/refined_convergence'
for label,mode in [('radial12','native'),('radial12','analytic'),('radial9','analytic'),('radial12','cartesian-donors'),('radial12','physical-frame'),('radial12','legacy-frame')]:
 cp=base/f'central20-n16-{label}-initial.checkpoint';prefix=out/f'central20-n16-{label}-initial-{mode}'
 cmd=['mpiexec','-n','2',str(binary),'16',str(cp),'.000125',str(config/f'central20_stationary_smr025_{label}.cfg'),'smr',str(prefix),'.3']
 if mode!='native':cmd+=[{'analytic':'analytic-initial-halos','cartesian-donors':'analytic-cartesian-to-shell-halos','physical-frame':'physical-frame-cartesian-to-shell-halos','legacy-frame':'legacy-frame-cartesian-to-shell-halos'}[mode]]
 before=sha(cp);start=time.time()
 with open(str(prefix)+'.log','w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 rec=dict(command=cmd,returncode=r.returncode,seconds=time.time()-start,input_before=before,input_after=sha(cp),binary_sha256=sha(binary));assert rec['input_before']==rec['input_after'];pathlib.Path(str(prefix)+'-command.json').write_text(json.dumps(rec,indent=2)+'\n');print(label,mode,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
