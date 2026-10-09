from pathlib import Path
import hashlib,json,time
Q=Path(__file__).resolve().parent;P=Q.parent;old=P/'inertial-family-held-003';new=P/'inertial-identity-held-006';A=Q/'saved-output-readback001';A.mkdir();start=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[old/'local-gates-release-001/coordinate.stdout',old/'local-gates-release-001/receipt.json',new/'local-gates-release-001/coordinate.stdout',new/'local-gates-release-001/receipt.json',new/'local-gates-debug-001/coordinate.stdout',new/'local-gates-debug-001/receipt.json'];pins=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}for p in paths]
a=[json.loads(x)for x in paths[0].read_text().splitlines()];b=[json.loads(x)for x in paths[2].read_text().splitlines()];c=[json.loads(x)for x in paths[4].read_text().splitlines()];assert len(a)==len(b)==len(c)==629;common=set(a[0]);assert all(all(x[k]==y[k]for k in common)for x,y in zip(a,b));assert b==c
r=json.loads(paths[3].read_text());d=json.loads(paths[5].read_text());assert r['passed']and d['passed'];assert all(x['stdout_sha256']==y['stdout_sha256']for x,y in zip(r['runs'],d['runs']))
poles=[];cases=json.loads((old/'cases.json').read_text())
for case,row in zip(cases,b):
 if case['r']>=.95 and case['name']=='field2-rho0':
  rho=sum(x*x for x in case['point']);o=1-rho;alpha=1+rho
  predicted_alpha=o*(-8*rho-1.5*alpha)-4*alpha*(1+3*rho);predicted_beta=[(12+10*o)*x for x in case['point']]
  actual=[o*x for x in row['actual_gauge4']];err=max(abs(actual[0]-predicted_alpha),max(abs(x-y)for x,y in zip(actual[1:],predicted_beta)))
  assert err<5e-10;poles.append({'r':case['r'],'point':case['point'],'OmegaF_actual4':actual,'OmegaF_predicted4':[predicted_alpha]+predicted_beta,'absolute_error':err})
r={'kind':'independent saved-data exact readback of diagnostic-only change','source_sha256':sha(__file__),'inputs':pins,'all629_old_common_fields_equal_exactly':True,'common_keys':sorted(common),'stable_Release_Debug_all4_stdout_byteidentical':True,'stable_Release_Debug_all629_JSON_equal':True,'original_direct_identity_still_failed':True,'pure_inertial_v1_outer_pole_saved_readback':poles,'pole_scope':'analytic C0 control plus already saved finiteOmega rows; no new query or bounded-gauge-source claim','no_operator_or_eigen_or_evolution':True,'seconds':time.monotonic()-start}
assert all(sha(p['path'])==p['sha256']for p in pins);(A/'receipt.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'receipt_sha256':sha(A/'receipt.json'),'rows':629,'pole_rows':len(poles),'pole_error_max':max(x['absolute_error']for x in poles),'seconds':r['seconds']},indent=2))
