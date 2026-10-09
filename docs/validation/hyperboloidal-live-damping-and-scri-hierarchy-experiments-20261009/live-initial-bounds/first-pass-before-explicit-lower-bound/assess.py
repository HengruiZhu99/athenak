# Mathematical source-formula assessment only: no new actual-kernel/native gate.
from pathlib import Path
import hashlib,json,subprocess,sys,time
import sympy as sy,mpmath as mp
p=Path(__file__).resolve().parent;repo=p.parents[2];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();start=time.monotonic()
files=[p/'assess.py',repo/'src/z4c/hyperboloidal/layer_reference.hpp',repo/'src/z4c/hyperboloidal/conformal_rhs.hpp',repo/'src/z4c/z4c_hyperboloidal.cpp']
before={str(f.relative_to(repo)):sha(f) for f in files}
alpha,w,o,k,bO,V=sy.symbols('alpha w Omega kappa betaDotOmega V',real=True)
eff=2*bO+k*o;k2=eff/k-1
sigma=sy.cancel((-2*alpha*w-eff)/o).subs(w,-bO/alpha);assert sy.simplify(sigma+k)==0
m=V*(2*bO+k*(o-1));sigmaV=sy.cancel((2*bO-k-m)/o)
sigmaBase=(2*bO-k)/o;assert sy.simplify(sigmaV-((1-V)*sigmaBase-V*k))==0
report={'scope':'Mathematical assessment from exact production reference/pulse formulas only. No compiled actual20/new native gates; no evolution admissibility or energy theorem.','algebra':{'kappa_eff_live':str(eff),'kappa2_live':str(k2),'sigma_live_exact':'-kappa_input','m_blended':str(m),'sigma_blended':str(sigmaV),'delta_RHS':'Delta P_t=Delta Theta_t=-m*Theta/Omega; all other actual geometric RHS fields unchanged','nonlinear_ADM_constraint_correction':'Delta H_t=-4 K m Theta/Omega; Delta Mi_t=2 partial_i(m Theta/Omega); Delta Zi_t=0; Delta Theta_t=-m Theta/Omega','gradient_m':'m_i=V_i[2 beta^j Omega_j+kappa(Omega-1)]+V[2 beta^j_i Omega_j+2 beta^j Omega_ij+kappa Omega_i]','linearization_about_Einstein_reference':'Derivative of live kappa2 times backgroundTheta is zero; reference-linear generator equals that of the prescribed background kappa2(r). At finiteTheta, value-only beta cross-couplings appear; principal derivative orders remain unchanged.','outer_general':'kappa_eff_ref=2S/a^2+(kappa_input-4/a)Omega. It equals the preceding core-matched fixed-Omega profile only if a=S/2.'}}
mp.mp.dps=80
r0=mp.mpf('.05');r1=mp.mpf('.95');width=r1-r0;kap=mp.mpf(10);amp=mp.mpf('.02')
def source(r):
 if r<=r0:w=mp.mpf(0);wp=mp.mpf(0)
 elif r>=r1:w=mp.mpf(1);wp=mp.mpf(0)
 else:
  g=-width/(r-r0)+width/(r1-r);w=1/(1+mp.exp(-g));wp=w*(1-w)*width*(1/(r-r0)**2+1/(r1-r)**2)
 O=1-w*r*r;op=-wp*r*r-2*w*r;b=2*r*w;al=mp.sqrt(O*O+b*b);L=O-r*op;beta=-b*al/L
 return O,op,beta,w,wp,al,L

def eff_ref(r):O,op,beta,*_=source(r);return 2*beta*op+kap*O

def pulse_negative_x(r):
 O,op,beta,*_=source(r);shape=(1-r*r)**4*mp.exp(-4*r*r)
 # Actual angular vector at x=-r,y=z=0 is +x, i.e. negative radial shift.
 return 2*(beta-amp*shape)*op+kap*O
critical=mp.findroot(lambda r:mp.diff(pulse_negative_x,r),(.103,.108));boundary=mp.findroot(lambda r:pulse_negative_x(r)-kap,(.108,.112));excess=pulse_negative_x(critical)-kap
assert excess>mp.mpf('1e-9')
report['strict_bound_counterexample']={'point':'x=-r,y=z=0; actual shift_pulse=.02,pulse_angular=true,width=.5','r_max_excess':mp.nstr(critical,70),'kappa_eff_excess_above_10':mp.nstr(excess,70),'positive_kappa2':mp.nstr(excess/kap,70),'upper_violation_end_radius':mp.nstr(boundary,70),'asymptotic_reason':'Near r0+, upper admissible inward radial shift margin (kappa-eff_ref)/(2|Omega_r|)~kappa(r-r0)^2/(2width)->0, whereas pulse inward shift is nonzero. Hence no uniform strict-upper-bound neighbourhood of this reference for the uncut live formula.'}
# Directed interval arithmetic verifies the analytic source/pulse inequalities.
# This is a reference+initial-pulse bound, NOT a kernel/PDE stability gate.
mp.iv.dps=50;iv=mp.iv;I=lambda lo,hi:iv.mpf([str(lo),str(hi)])
kapI=I(10,10);ampI=I('.02','.02')
def interval(r,w,wp):
 O=1-w*r*r;dd=wp*r*r+2*w*r;al=iv.sqrt(O*O+4*r*r*w*w);L=O+r*dd
 ratio=4*al*dd/(r*L) # eff_ref=10-w*r^2*(10-ratio)
 shape=(1-r*r)**4*iv.exp(-4*r*r)
 vectorbound=iv.sqrt((1+iv.mpf('.15')*r*r)**2+(iv.mpf('.2')*r)**2+(iv.mpf('.05')*r*r)**2)
 pulse_ratio=2*ampI*shape*vectorbound*(wp/w+2/r)/(kapI-ratio) if float(w.a)>0 else None
 # For radial angular pulse |n·vector| <= this Euclidean component bound.
 eff=2*(2*r*w*al/L)*dd+10*O
 pulse_size=2*ampI*shape*vectorbound*dd
 return ratio,pulse_ratio,eff,pulse_size
maxR=0.;pulse_bounds={'.15':0.,'.2':0.};count=0
# All interior intervals: positive Omega and beta*Omega_r follow analytically.
lo=mp.mpf('.050001');hi=mp.mpf('.949999');steps=9000
for j in range(steps):
 l=lo+(hi-lo)*j/steps;h=lo+(hi-lo)*(j+1)/steps;r=I(mp.nstr(l,70),mp.nstr(h,70));g=-iv.mpf('.9')/(r-iv.mpf('.05'))+iv.mpf('.9')/(iv.mpf('.95')-r)
 w=1/(1+iv.exp(-g));wp=w*(1-w)*iv.mpf('.9')*(1/(r-iv.mpf('.05'))**2+1/(iv.mpf('.95')-r)**2)
 R,pr,eff,ps=interval(r,w,wp);upper=float(R.b);assert upper<10;maxR=max(maxR,upper);count+=1
 for cut in pulse_bounds:
  # Include intervals intersecting the cut, conservatively from slightly below it.
  if h>=mp.mpf(cut):assert pr is not None and float(pr.b)<1;pulse_bounds[cut]=max(pulse_bounds[cut],float(pr.b))
# End tails use analytic exponential inequalities, not underflow/floors.
# At delta<=1e-6, exp[-.9/delta+.9/(.9-delta)]<1e-1000 and
# w' or (1-w)' magnitude<1e-900. Monotonic exp(-.9/delta)/delta^2 proves it.
eps=mp.mpf('1e-6');E=mp.exp(-width/eps+width/(width-eps));Wprime=E*(width/eps**2+width/(width-eps)**2)
assert E<mp.mpf('1e-1000') and Wprime<mp.mpf('1e-900')
# Near r0, ratio <=4 sqrt(1+4rmax^2)*dmax/[r0(1-rmax^2)] <<10.
rmax=r0+eps;dmax=Wprime*rmax*rmax+2*E*rmax;tailR=4*mp.sqrt(1+4*rmax*rmax)*dmax/(r0*(1-rmax*rmax));assert tailR<1
# Near r1, use deliberately loose directed interval bounds w in[1-1e-40,1], w'<=1e-30.
r=I('.949999','.95');w=I(mp.nstr(1-mp.mpf('1e-40'),70),1);wp=I(0,'1e-30');R,pr,_,_=interval(r,w,wp);assert float(R.b)<10 and float(pr.b)<1
maxR=max(maxR,float(R.b));pulse_bounds={key:max(val,float(pr.b)) for key,val in pulse_bounds.items()}
# Outer exactCMC r>=.95: eff=8+2Omega in[8,8.195]; the pulse bound is tiny.
r=I('.95',1);w=I(1,1);wp=I(0,0);R,pr,_,_=interval(r,w,wp)
# This broad dependency interval can exceed the exact ratio8: use exact outer formula.
outer_pulse_upper=float((iv.mpf('.04')*(1-I('.95','.95')**2)**4*iv.exp(-4*I('.95','.95')**2)*iv.sqrt((1+iv.mpf('.15'))**2+iv.mpf('.2')**2+iv.mpf('.05')**2)/iv.mpf('.95')).b)
assert outer_pulse_upper<1
pulse_bounds={key:max(val,outer_pulse_upper) for key,val in pulse_bounds.items()}
report['continuous_reference_initial_pulse_bounds']={'method':'50-digit directed mpmath.iv intervals over9000 subintervals; analytic exponential tail bounds; exact outerCMC. Conditions and source-derived formulas in script.','reference':'0<eff_ref<=10 for0<=r<1; atscri limitingeff_ref8.','reference_upper_ratio_bound':maxR,'interval_count':count,'analytic_near_core_w_bound_log10':mp.nstr(mp.log10(E),30),'initial_pulse_sufficient_relative_upper_bound':pulse_bounds,'angular_component_bound':'|n·vector| <= sqrt((1+.15r²)^2+(.2r)^2+(.05r²)^2)','initial_positive_lower_bound':'eff_ref>=10Omega and eff_ref>0; for r>=.15 a separate conservative bound eff_ref>=8 follows from scan only, but strict positivity follows beta_ref<0 and bounded pulse: actual outward radial pulse amplitude<=.02*vectorbound<|beta_ref| where needed, or eff_ref−pulse_size>0 checked below.','smooth_turnon_implication':'Any0<=V<=1 identically0 below.15 (or.2) preserves0<eff_blend<=10 for the stated initial pulse, by convexity with baseeff10. This does not prove evolution preservation.'}
# Cheap source sampling for context only, clearly not the interval proof.
scans=[]
for j in range(1,1000):
 r=mp.mpf(j)/1000;O,op,beta,w,wp,al,L=source(r);shape=(1-r*r)**4*mp.exp(-4*r*r);A=mp.sqrt((1+mp.mpf('.15')*r*r)**2+(mp.mpf('.2')*r)**2+(mp.mpf('.05')*r*r)**2);ps=2*amp*shape*A*abs(op)
 assert eff_ref(r)-ps>0
 scans.append((eff_ref(r),eff_ref(r)-(8+2*O)))
report['context_samples_only']={'reference_eff_min_999points':float(min(z[0] for z in scans)),'reference_eff_max_999points':float(max(z[0] for z in scans)),'live_reference_minus_fixed_profile_min':float(min(z[1] for z in scans)),'live_reference_minus_fixed_profile_max':float(max(z[1] for z in scans)),'positivity_sampled_pulse':True}
(p/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');after={str(f.relative_to(repo)):sha(f) for f in files};assert before==after
receipt={'mathematical_assessment_completed':True,'actual_kernel_or_native_gate_run':False,'no_evolution_admissibility_claim':True,'source_before':before,'source_after':after,'sources_unchanged':True,'result_sha256':sha(p/'result.json'),'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'python':sys.version,'sympy':sy.__version__,'mpmath':mp.__version__,'seconds':time.monotonic()-start}
(p/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS mathematical live/blended assessment only',report['strict_bound_counterexample']);print(report['continuous_reference_initial_pulse_bounds'])
