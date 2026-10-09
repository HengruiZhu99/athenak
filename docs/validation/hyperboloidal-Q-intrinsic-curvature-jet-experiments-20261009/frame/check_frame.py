from pathlib import Path
import json,hashlib,sympy as s
P=Path(__file__).resolve().parent;ROOT=P.parents[2];j=json.loads((P/'actual-release.json').read_text());errors={'Y_t':0,'q_t':0,'q_tt':0};TF=0;kill=0
for t in j['rows']:
 for row in t['levels']:
  errors['Y_t']=max(errors['Y_t'],max(abs(a-b)for a,b in zip(row['Y_t'],t['Y_t_expected'])))
  for A in range(2):
   for B in range(2):errors['q_t']=max(errors['q_t'],abs(row['q_t'][A][B]));errors['q_tt']=max(errors['q_tt'],abs(row['q_tt'][A][B]-t['q_tt_expected'][A][B]))
  tr=(row['q_tt'][0][0]+row['q_tt'][1][1])/2
  if t['which']==0:TF=max(TF,max(abs(row['q_tt'][A][B]-(tr if A==B else 0))for A in range(2)for B in range(2)))
  else:kill=max(kill,max(abs(v)for r in row['q_tt']for v in r))
q=j['summary'];assert q['initial_constraints_max']<1e-12 and q['initial_R0_max']<1e-12 and q['first_RHS_next_R0_max']<1e-12 and q['actual_complete_first_RHS_field_match_error']<1e-8
assert max(q[k]for k in ['initial_N0_max','initial_N1_max','initial_Q0_max','first_RHS_N1_time_max'])<1e-12
assert max(errors.values())<1e-6 and TF>7.9 and kill<1e-6
# Preserve explicit distinction: additional stationary spatial-pullback family,
# eta_outer1 only; its data are read from the prior frozen actual gate.
prior=ROOT/'build-layer-research/continuum/q-null-spatial-diffeo/immutable-Q-spatial-Einstein-pullback-timejet-20261009';index=json.loads((prior/'index.json').read_text());assert hashlib.sha256((prior/'index.json').read_bytes()).hexdigest()=='de31bb51e55c159100a58d14acaae0f00231a7d6128bc6c3771d535b75ff5662';pin=next(f['sha256']for f in index['files']if f['path']=='actual-release.json');assert hashlib.sha256((prior/'actual-release.json').read_bytes()).hexdigest()==pin;d=json.loads((prior/'actual-release.json').read_text());etaerr=0;extra=[]
for a in [.5,.75,1.,2.]:
 for di in [0,1]:
  n=[.36,-.48,.8]if di else[1,0,0];T=[-n[0]*n[1],1-n[1]**2,-n[2]*n[1]];F=[sum(w*next(t for t in d['controls']if t['a']==a and t['sigma']==3 and t['dir']==di and t['col']==c)['F0'][k]for c,w in [(14,1),(19,1),(5,-1),(28,-1)])for k in range(20)];rad=sum(n[i]*F[4+i]for i in range(3));actual=[F[4+i]-rad*n[i]for i in range(3)];expected=[(a-1)*v/a**3 for v in T];etaerr=max(etaerr,max(abs(x-y)for x,y in zip(actual,expected)));extra.append(dict(a=a,dir=di,Y_t_actual=actual,Y_t_expected=expected))
assert etaerr<1e-8
# Symbolic projection of Penrose regularity, retaining trace/scale.
nu,alpha,B=s.symbols('nu alpha Box0');k,K,qt,Y=s.symbols('k K qdot LieYq');constraint=-nu*(k+K)-B/4;Ksolve=s.solve(constraint,K)[0];qdot=s.simplify(Y-2*alpha*k-2*alpha*Ksolve);assert s.simplify(qdot-Y-alpha*B/(2*nu))==0
# On the unit sphere h=Lie_T q has deltaR=Lie_T(2)=0.
zz=s.symbols('z');df=1-3*zz**2;ddf=-6+18*zz**2;trh=2*df;divdivh=2*(ddf+df);deltatr=2*ddf;assert s.expand(divdivh-deltatr-trh)==0
r={'coordinate_q_drift_changes_intrinsic_roundness':False,'intrinsic_roundness_preservation_proved_without_null_hierarchy':False,'passed_local_boundary_frame_negative_gate':True,'fixed_initial_q_and_Y_sufficient_for_invariance':False,'sigma3_evolution_admitted':False,'eta_outer':1,'rows':len(j['rows']),'kernel_radial_points':len(j['rows'])*12,'actual':q,'errors':errors,'nonKilling_q_tt_tracefree_max':TF,'Killing_q_tt_max':kill,'geometric_shift_formula':'Y=beta+alpha*s on null branch; deltaY_T=deltabeta_T-deltabargamma_nT/a','boundary_two_metric_evolution':'qdot=Lie_Y q+(alpha Box0/(2nu))q; Einstein N=O(Omega^2),preferredBox=O(Omega) gives qdot=Lie_Y q','gauge_only_beta_Omega_T':'Y_t=2T/a^2 including moving spatial normal; q_tt=Lie_(2T/a^2)q','additional_frozen_spatial_pullback_readback':{'prior_index_sha256':'de31bb51e55c159100a58d14acaae0f00231a7d6128bc6c3771d535b75ff5662','actual_file_sha256':pin,'scope':'xi=Omega(r^2 e_y-y*x), exact Einstein spatial-pullback family, eta_outer1. Distinct from gauge-only family.','formula':'Y_t=(eta*a-1)T/a^3','max_error':etaerr,'rows':extra}}
(P/'check-report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
