"""Independent real-FD readback of saved analytical energy coefficients; no evolution."""
from pathlib import Path
import hashlib,json,subprocess,time,sys
import numpy as np
P=Path(__file__).resolve().parent
S=P/'J0-N8-rb.98-Q64-a12x24-readback002'
sys.path.insert(0,str(S))
from energy_coefficients import coefficients,mm
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
exe=P/'build-attempts/release-005/radial-bridge-release'
assert sha(exe)=='2293e9be6f75042f926f22232039c3c3bdd28826eb9e80061905c272b7adce15'
assert sha(S/'energy_coefficients.py')=='eb15ff31262e5907d2825c357f055c1f8c0fc3c189f1fe8f7b51e0b0cf0c3e40'
dest=P/'coefficient-FD-gate-001';assert not dest.exists();dest.mkdir()
radii=[.025,.15,.3,.5,.7,.86,.875,.89,.92,.98];hs=[.0002/2**j for j in range(5)]
plan={'radii':radii,'h_sequence':hs,'first_derivative':'fourth-order centered, complete reference reevaluated at every point','threshold_final_scaled':2e-7,'scope':'H,Kn and coefficient divergence Gamma binding; finite-radius continuum coefficients only','source':{'path':str(S/'energy_coefficients.py'),'sha256':sha(S/'energy_coefficients.py')},'executable':{'path':str(exe),'sha256':sha(exe)}}
(dest/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
queries=sorted(set([r for r in radii]+[r+d*h for r in radii for h in hs for d in (-2,-1,1,2)]))
payload=''.join(format(r,'.17g')+'\n' for r in queries);(dest/'queries.txt').write_text(payload)
start=time.monotonic();run=subprocess.run([str(exe),'--reference-batch'],input=payload,text=True,capture_output=True);(dest/'stdout.txt').write_text(run.stdout);(dest/'stderr.txt').write_text(run.stderr)
assert run.returncode==0 and not run.stderr
refs=np.fromstring(run.stdout,sep=' ').reshape(-1,17);cf={r:coefficients(r,a) for r,a in zip(queries,refs)}
def err(a,b):
 d=a-b;return {'scaled':float(np.linalg.norm(d)/max(1.,np.linalg.norm(a),np.linalg.norm(b))),'absolute':float(np.linalg.norm(d)),'maxabs':float(np.max(np.abs(d)))}
rows=[]
for r in radii:
 c=cf[r]['c'];div=cf[r]['div_s'];Q=mm(cf[r]['H'],cf[r]['Kn'])
 for field,analytic in [('H',cf[r]['Hr']),('Kn',cf[r]['Knr']),('HKn',(cf[r]['Gamma']-div*Q)/c)]:
  seq=[]
  for h in hs:
   def val(j):
    f=cf[r+j*h];return mm(f['H'],f['Kn']) if field=='HKn' else f[field]
   fd=(val(-2)-8*val(-1)+8*val(1)-val(2))/(12*h)
   seq.append(err(fd,analytic))
  rows.append({'r':r,'field':field,'sequence':seq,'final_pass':seq[-1]['scaled']<=2e-7,'classification':'fourth_order_evidence' if any(seq[j+1]['scaled']<seq[j]['scaled']/8 for j in range(4)) else 'within_tolerance_order_unclassified'})
report={'passed':all(x['final_pass'] for x in rows),'plan_sha256':sha(dest/'plan.json'),'source_sha256':sha(__file__),'elapsed':time.monotonic()-start,'rows':rows,'maximum_final_scaled':max(x['sequence'][-1]['scaled'] for x in rows),'maximum_final_absolute':max(x['sequence'][-1]['absolute'] for x in rows),'scope':plan['scope']}
(dest/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2));assert report['passed']
