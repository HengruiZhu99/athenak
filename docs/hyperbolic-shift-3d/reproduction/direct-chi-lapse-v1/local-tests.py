from pathlib import Path
import subprocess,json,math,struct,hashlib
root=Path.cwd();out=root/'output/hyperbolic-core-shell/direct-chi-lapse-v1';exe=root/'build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_direct_chi_lapse';initial=out/'puncture-ac1-at3m.checkpoint'
def run(name,kind,end,restart,good=True):
 prefix=out/name
 cmd=['/opt/homebrew/bin/mpiexec','-n','2',str(exe),'16',str(prefix),kind,repr(end),'.0005','-',str(restart),'180','.05','.001','1','4','2']
 with prefix.with_suffix('.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=210)
 assert (r.returncode==0)==good,(name,r.returncode)
 if not good:return dict(returncode=r.returncode,log=prefix.with_suffix('.log').read_text())
 report=json.loads(prefix.with_suffix('.report.json').read_text());return report
results={}
results['minkowski']=run('minkowski','minkowski',.001,'-')
end1=math.nextafter(1.0005,math.inf);end2=math.nextafter(1.001,math.inf)
results['puncture_one']=run('puncture-one','puncture',end1,initial)
results['restart']=run('restart','puncture',end2,out/'puncture-one.checkpoint')
results['two_steps']=run('two-steps','puncture',end2,initial)
a=(out/'restart.checkpoint').read_bytes();b=(out/'two-steps.checkpoint').read_bytes();assert a==b
results['restart_bitwise_equal']=True
results['reject_al1']=run('reject-al1','puncture',end1,root/'output/hyperbolic-core-shell/direct-lapse-v1/puncture-al1-at3m.checkpoint',False)
# Deliberately force finite negative final candidates with linear RK residuals.
data=bytearray(initial.read_bytes());h=struct.unpack('<16s4i3Q2d',data[:72]);cells=h[5]
for f in [0,18]:struct.pack_into('<d',data,168+8*(31*cells+31*4095+f),1.)
synth=out/'synthetic-input.checkpoint';synth.write_bytes(data)
r=run('synthetic-signed','puncture',end1,synth);events=r['candidate_sign_events'];assert len(events)==2 and all(e[0]==4095 and e[6]==1 and e[9]<0 for e in events)
results['synthetic']=r
(out/'local-test-results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:({z:v.get(z) for z in ['ok','t','error','max_abs','last_hamiltonian_l2','candidate_sign_events']} if isinstance(v,dict) else v) for k,v in results.items()},indent=2))
