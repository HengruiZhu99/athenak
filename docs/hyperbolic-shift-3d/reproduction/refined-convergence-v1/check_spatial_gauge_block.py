"""Independent analytic spatial gauge block versus extracted RF2 production RHS.

Uses preserved principal matrices; no evolution or change to reference source.
The restricted 5x5 block alone is not a full-system hyperbolicity certificate.
"""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'spatial-gauge-block';OUT.mkdir(exist_ok=True)
reference='origin/project/telegrapher_lapse'
commit=subprocess.check_output(['git','rev-parse',reference],text=True).strip()
path='experiments/telegrapher_aurora_20260929/inputs_v4/g1_bhahaha_L5.athinput'
text=subprocess.check_output(['git','show',commit+':'+path])
(OUT/'reference-input.athinput').write_bytes(text)
results=[]
for name,omega in [('central6_stationary_matched_plateau',.4712),('central20_stationary_smr025_radial9',.42)]:
 rows=[json.loads(x) for x in (HERE/'principal-audit'/(name+'-crossings.jsonl')).read_text().splitlines()]
 q=min((q for q in rows if q['inner_weight']==1),key=lambda q:abs(q['gap']))
 alpha=q['alpha']/omega;chi=omega**2*q['chi'];guard=max(chi,1e-5);gamma=q['inner_gamma'];ratio=2.
 # Variables: d_x alpha_phys, d_x B_x, Khat, d_x beta_x, Lambda_x.
 # Generalization retains the physical-chi guard in the lapse/B coupling.
 P=np.array([[0,guard,-2*alpha,0,0],[ratio,0,0,0,0],[-chi,0,0,0,0],[0,0,0,0,gamma],[0,0,-4*alpha/3,4/3,0]])
 ids=[18,19,13,22,15];S=np.diag([1/omega,1,1,1,1]);actual=S@np.array(q['P'])[np.ix_(ids,ids)]@np.linalg.inv(S)
 error=float(np.max(abs(actual-P)));assert error<1e-12
 lapse2=2*alpha*chi+ratio*guard;shift2=4*gamma/3
 assert abs(lapse2-shift2)<1e-12
 T=P-np.sqrt(shift2)*np.eye(5)
 singular=np.linalg.svd(T,compute_uv=False);squared=np.linalg.svd(T@T,compute_uv=False)
 assert sum(singular<1e-10)==1 and sum(squared<1e-10)==2
 results.append(dict(config=name,r=q['r'],omega=omega,alpha_physical=alpha,chi_physical=chi,gamma=gamma,actual_rhs_block=actual.tolist(),analytic_block=P.tolist(),max_difference=error,lapse_speed2=lapse2,shift_speed2=shift2,nullity_T=1,nullity_T_squared=2))
result=dict(passed=True,reference_commit=commit,reference_input=path,reference_input_sha256=hashlib.sha256(text).hexdigest(),reference_tau=.1,reference_kappa=.2,reference_shift_Gamma=1,reference_shift_eta=2,checks=results,qualification='The successful spatial-reference input uses kappa/tau=2. Under central coordinate scaling its normalized Gamma coefficient has the same possible inner lapse/shift speed coincidence. This explains why the inner defect is not unique to hyperboloidal stitching; it does not prove that it caused either long-run error. The full production-symbol rank evidence is separate. The 5x5 restriction sets other perturbations to zero, and is not a complete characteristic analysis.')
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
