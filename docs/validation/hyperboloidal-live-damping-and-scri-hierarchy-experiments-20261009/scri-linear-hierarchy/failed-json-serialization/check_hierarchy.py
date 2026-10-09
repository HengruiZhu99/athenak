from pathlib import Path
import json,numpy as np,sympy as s
p=Path(__file__).resolve().parent;data=json.loads((p/'kernel.json').read_text());report={'scope':'Linear actual Cartesian C0 physical-P/spatial-norm gauge Taylor compatibility, outer CMC reference S1,kappa10. Baseline and live V1 damping compared. No native integration, PDE amplitude blowup or regular scri closure claim.'}
report['boundary']=[]
for x in data['boundary']:
 B=s.Matrix([[s.Rational(float(z)).limit_denominator(1000000) for z in row] for row in x['B']]);err=float(abs(np.array(B,dtype=float)-np.asarray(x['B'])).max());assert err<1e-11
 assert B.rank()==15
 # Necessary value constraints from ALL actual first-jet pole rows, not just
 # zero-jet eigenvectors. Tangential jets are left unrestricted here, making
 # necessity stronger; compatible angular derivatives are added in the report.
 conditions=[]
 for col in [0,2,3,4,5,6]:
  row=s.zeros(1,80);row[col]=1;assert B.col_join(row).rank()==15;conditions.append(str(col))
 row=s.zeros(1,80);row[1]=1;row[7]=-1;assert B.col_join(row).rank()==15
 P=B[:,:20];assert 20-P.rank()==20-(P*P).rank()==5
 lam=s.Symbol('lambda');char=P.charpoly(lam).as_expr();q=s.cancel(char/lam**5);coeff=s.Poly(q,lam).all_coeffs();Q=s.zeros(20)
 for c in coeff:Q=Q*P+c*s.eye(20)
 Pi=Q/q.subs(lam,0);assert Pi*Pi==Pi and P*Pi==s.zeros(20)
 for col in [0,2,3,4,5,6]:assert Pi[col,:]==s.zeros(1,20)
 assert Pi[1,:]-Pi[7,:]==s.zeros(1,20)
 D=(P+Pi).inv()-Pi;assert P*D==D*P==s.eye(20)-Pi
 theta=s.zeros(1,20);theta[3]=1;C=s.zeros(1,20);C[2]=1;C[0]=-3;C[4]=-3
 assert theta*Pi==C*Pi==s.zeros(1,20)
 report['boundary'].append({'a':x['a'],'profile':x['profile'],'rational_reconstruction_error':err,'full_firstjet_pole_rank':15,'zerojet_nullity':5,'zerojet_square_nullity':5,'necessary_values':'deltaalpha0=deltaP0=deltaTheta0=deltabeta0^i=0; deltachi0=h_nn0','kernel_projector_exact_identities':True,'Theta_and_C_annihilate_kernel_projector':True,'group_inverse_exact_identities':True,'norm_Theta_group_inverse':float(np.linalg.norm(np.asarray(theta*D,dtype=float))),'norm_C_group_inverse':float(np.linalg.norm(np.asarray(C*D,dtype=float)))})
report['cases']=[]
for a in [.5,.75,1.,2.]:
 for profile in [0,1]:
  rows=[x for x in data['cases'] if x['a']==a and x['profile']==profile]
  sets={which:sorted([x for x in rows if x['which']==which],key=lambda x:x['Omega']) for which in [0,1,2]}
  q=1.;l=4/(30*a*a+2);ar=-2*l/3
  z=sets[0][0];assert z['Omega']==0 and max(abs(np.array(z['pole'])))<1e-13
  theta_limit=-2/a**2;omega_Q_limit=(2-3*l)/a**2;N_limit=2*(l-2)/(3*a**3)
  x=sets[0][1];assert abs(x['RHS'][3]-theta_limit)<2e-5 and abs(x['Omega']*x['Qdot']-omega_Q_limit)<2e-5 and abs(x['Nrawdot']-N_limit)<2e-5
  z1=sets[1][0];assert max(abs(np.asarray(z1['pole'])))==0 and z1['H']==z1['Mrad']==z1['Zrad']==0
  exact_rhs_error=0
  for x in sets[1][1:]:
   alpha=1/a-x['Omega'];expected=np.zeros(20);expected[0]=-alpha**2*x['Omega'];expected[1]=2*alpha*x['Omega']/3;expected[2]=-2*x['Omega']**2/a;expected[3]=-2*alpha*x['Omega']/a;expected[17]=8*alpha*x['r']/(3*a)
   exact_rhs_error=max(exact_rhs_error,float(abs(np.asarray(x['RHS'])-expected).max()));assert exact_rhs_error<1e-11
  # which2 is that exact Cartesian RHS as the next dual seed. Its nonzero pole
  # is d_t B0, proving initial algebraic+scalar tangency does not close B0.
  next_pole=np.zeros(20);next_pole[12]=-8/(3*a**4);next_pole[15]=4/(3*a**4);next_pole[17]=(12-80*a*a)/(3*a**4)
  actual=np.asarray(sets[2][0]['pole']);assert abs(actual-next_pole).max()<1e-12
  report['cases'].append({'a':a,'profile':profile,'q':1,'all_initial_pole_cancelled_family':{'l':l,'A_rad':ar,'pole0_max':max(abs(np.asarray(z['pole']))),'H0':z['H'],'Mrad0':z['Mrad'],'Zrad0':z['Zrad'],'Theta_t0':theta_limit,'Omega_Q_t_limit':omega_Q_limit,'Nraw_t0':N_limit,'refinement':[{k:x[k] for k in ['Omega','Qdot','Nrawdot']}|{'Theta_t':x['RHS'][3]} for x in sets[0][1:]]},'Omega_squared_P_family':{'initial_pole0_max':0,'scalar_boundary_rates_zero':True,'exact_first_actual_RHS_error':exact_rhs_error,'next_pole_A_rad':float(actual[12]),'next_pole_Lambda_rad':float(actual[17]),'next_pole_expected_A_rad':float(next_pole[12]),'next_pole_expected_Lambda_rad':float(next_pole[17])}})
report['necessary_general_firstjet_conditions']={'value':'deltaalpha0=deltaP0=deltaTheta0=deltabeta0=0; deltachi0=h_nn0 (S1, spatial-norm rho1.5)', 'nonfree_angular':'partial_A deltachi0=partial_A h_nn0+2h_nA0 on unit sphere; derivatives of alpha0/P0/Theta0/beta0 are zero, and rotating radial tensor/vector derivatives must be retained.', 'Htensor':'H_ij=h_ij+(deltaGamma_bar^n_ij)^TF, where Gamma_bar depends on h/chi first spatial jets.', 'normal_residue':'d_i=deltaLambda_i-deltaGamma_tilde_i; d_A=4H_nA/(kappa a^2), d_n=[12H_nn+4deltaP1+2deltaTheta1]/(3kappa a^2+2); A_ij=H_ij-(d_i n_j+n_i d_j)^TF/2.', 'Theta_corner':'0=(2/a)deltaLapOmega-(2/a^2)deltaQ-kappa(2+kappa2_scri)Theta1, deltaQ=P1-3(alpha1+beta_n1).','further':'Finite smooth Q additionally requires (P_t-3w_t)_0=0. Preserving all leading pole constraints requires d_t B0=B0(F0,F1,angular F0)=0; F1 and the A/Lambda F0 include second spatial Taylor jets. The Omega^2P family violates this next condition despite vanishing initial pole and scalar rates.'}
report['amplitude_caveat']={'actual_zerojet_model':'For actual reference P with five semisimple zeros and Hurwitz nonzero roots, Pi=Q(P)/Q(0) projects to kerP and D=(P+Pi)^(-1)-Pi is its group inverse. Constant finite forcing f and zero initial data yield u=t Pi f+Omega D(exp(tP/Omega)-I)(I-Pi)f. Theta and C=P-3w annihilate Pi, hence this FROZEN NORMAL MODEL can keep them O(Omega) despite O(1) initial time derivatives.','scope':'This is an exact algebraic statement for the rationally reconstructed actual20 zero-jet matrix, not a substitute scalar toy or a PDE amplitude estimate. The PDE pole also contains first spatial derivatives, coefficient variation and higher hierarchy. Spatial jets/forcing, boundary conditions and nonnormal/global evolution remain uncontrolled; the smooth-corner counterexamples alone do not imply finite-Q amplitude blowup.'}
report['passed_actual_kernel_linear_hierarchy_gate']=True
(p/'check-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('PASS actual20 firstjet/hierarchy counterexamples and exact reconstructed projector/group inverse; amplitude/PDE scope retained')
