#!/usr/bin/env python3
"""Independent scalar-fsum energy audit and exact polynomial-ball control."""
import argparse,hashlib,json,math,os
from fractions import Fraction as F
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Exact rational polynomial coefficients, independent of C++ AD/metric kernels.
def const(x):return {(0,0,0):F(x)}
def add(*polys):
 out={}
 for p in polys:
  for e,c in p.items():out[e]=out.get(e,F(0))+c
 return {e:c for e,c in out.items() if c}
def scale(p,c):return {e:v*F(c) for e,v in p.items() if v*c}
def mul(a,b):
 out={}
 for e,c in a.items():
  for f,d in b.items():g=tuple(x+y for x,y in zip(e,f));out[g]=out.get(g,F(0))+c*d
 return {e:c for e,c in out.items() if c}
def derivative(p,i):
 out={}
 for e,c in p.items():
  if e[i]:f=list(e);f[i]-=1;out[tuple(f)]=c*e[i]
 return out
def integral_coefficient(p):
 # Integral is pi times an EXACT rational number for even Cartesian powers.
 def half_gamma(a):return F(math.factorial(2*a),4**a*math.factorial(a))
 total=F(0)
 for e,c in p.items():
  if any(x%2 for x in e):continue
  aa=[x//2 for x in e];A=sum(aa);moment=F(2,2*A+3)*math.prod(half_gamma(x) for x in aa)/half_gamma(A+1);total+=c*moment
 return total
def integrate(p):return float(integral_coefficient(p))*math.pi

X=[{tuple(int(i==j) for j in range(3)):F(1)} for i in range(3)];o=add(const(1),*[scale(mul(x,x),-1) for x in X]);o2=mul(o,o)
beta=[mul(o2,add(const(1),scale(X[0],F(1,5)),scale(X[1],F(-3,10)))),mul(o2,add(const(F(3,10)),scale(mul(X[1],X[2]),F(2,5)))),mul(o2,add(const(F(-1,5)),scale(mul(X[0],X[2]),F(1,2))))]
div=add(*[derivative(beta[i],i) for i in range(3)]);H=[[add(derivative(beta[i],j),derivative(beta[j],i),scale(div,F(-2,3)) if i==j else {}) for j in range(3)] for i in range(3)];A=[scale(add(*[derivative(derivative(beta[i],j),j) for j in range(3)],scale(derivative(div,i),F(1,3))),-1) for i in range(3)]
Epoly=add(*[mul(beta[i],A[i]) for i in range(3)]);Qpoly=scale(add(*[mul(H[i][j],H[i][j]) for i in range(3) for j in range(3)]),F(1,2));Bpoly=add(*[mul(b,b) for b in beta]);exact={'E_integral':integrate(Epoly),'Q_integral':integrate(Qpoly),'beta2_integral':integrate(Bpoly),'ball_volume':4*math.pi/3,'zero_boundary_flux':'beta=(1-r²)^2 p(x); beta and first derivatives vanish at unit sphere','method':'exact rational polynomial coefficients and unit-ball moments; final multiply by pi','E_over_pi_exact_fraction':str(integral_coefficient(Epoly)),'Q_over_pi_exact_fraction':str(integral_coefficient(Qpoly)),'exact_rational_energy_identity':integral_coefficient(Epoly)==integral_coefficient(Qpoly)}
report={'input_manifest_sha256':sha(a.input/'manifest.json'),'analysis_source_sha256':sha(Path(__file__)),'exact_smooth_control':exact,'regional_warning':'Only whole-domain E=Q is the zero-Dirichlet IBP identity. Regional E-Q includes crossregion boundary flux and quadrature error; contributions localize the global defect, not independent regional violations.','cases':{},'all_cells_retained':True}
m=json.loads((a.input/'manifest.json').read_text())
for name in ['negative_physical_witness','smooth_zero_boundary']:
 path=a.input/('energy.'+name+'.csv.gz');q=np.genfromtxt(path,delimiter=',',names=True);w=q['volume'];b2=q['beta2_density'];E=q['E_density'];Q=q['Q_density'];defect=E-Q;den=math.fsum(float(x*y) for x,y in zip(w,b2));energy=math.fsum(float(x*y) for x,y in zip(w,E));strain=math.fsum(float(x*y) for x,y in zip(w,Q));volume=math.fsum(map(float,w));bad=np.argsort(w*defect)[:20];entry={'raw_sha256':sha(path),'rows':len(q),'volume':volume,'beta2_integral':den,'energy_integral_fsum':energy,'positive_strain_integral_fsum':strain,'defect_integral_fsum':energy-strain,'rayleigh':energy/den,'strain_over_beta2':strain/den,'relative_defect_over_strain':(energy-strain)/strain,'negative_energy_cells':int((E<0).sum()),'nonnegative_strain_all_cells':bool((Q>=0).all()),'maximum_strain_trace_residual':float(np.max(np.abs(q['H00']+q['H11']+q['H22']))),'most_negative_weighted_defect_points':[{'point':int(q['point'][i]),'block':int(q['block'][i]),'face':int(q['face'][i]),'layer':int(q['layer'][i]),'logical_ijk':[int(q[k][i]) for k in ['i','j','k']],'xyz':[float(q[k][i]) for k in ['x','y','z']],'volume':float(w[i]),'energy_density':float(E[i]),'strain_density':float(Q[i]),'weighted_defect':float(w[i]*defect[i])} for i in bad]}
 c=m['result']['cases'][name]['regions'];entry['region_defect_fraction_of_global']={r:data['defect_integral']/(energy-strain) for r,data in c.items()};entry['fresh_Az_max_difference']=m['result']['cases'][name]['fresh_Az_max_difference'];entry['cpp_fsum_relative_energy_difference']=abs(energy-c['all']['energy_integral'])/max(abs(energy),1e-300)
 if name=='smooth_zero_boundary':
  ea=math.fsum(float(x*y) for x,y in zip(w,q['analytic_E_density']));qa=math.fsum(float(x*y) for x,y in zip(w,q['analytic_Q_density']));entry.update({'analytic_same_quadrature_E':ea,'analytic_same_quadrature_Q':qa,'analytic_same_quadrature_defect':ea-qa,'additional_discretization_defect_vs_analytic_quadrature':(energy-strain)-(ea-qa),'discrete_E_relative_exact_ball':(energy-exact['E_integral'])/exact['E_integral'],'discrete_Q_relative_exact_ball':(strain-exact['Q_integral'])/exact['Q_integral']})
 report['cases'][name]=entry
report['exact_smooth_energy_identity_error']=exact['E_integral']-exact['Q_integral'];os.environ['MPLCONFIGDIR']=str(a.output/'.mplconfig');os.environ['XDG_CACHE_HOME']=str(a.output/'.cache');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,(name,case) in zip(axs,report['cases'].items()):
 ax.bar(['Discrete energy','Positive strain'],[case['rayleigh'],case['strain_over_beta2']],color=['#bd4450','#376ea6']);ax.axhline(0,color='black',lw=.7);ax.set_title(name.replace('_',' '));ax.set_ylabel('Integral / global integral beta²');ax.grid(axis='y',alpha=.2)
fig.savefig(a.output/'global-energy-versus-strain.png',dpi=160);plt.close(fig)
report['plot_sha256']=sha(a.output/'global-energy-versus-strain.png');(a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(a.output/'.gitignore').write_text('/.mplconfig/\n/.cache/\n');print(json.dumps({'exact':exact,'cases':{name:{k:v for k,v in case.items() if k in ['rayleigh','strain_over_beta2','relative_defect_over_strain','additional_discretization_defect_vs_analytic_quadrature']} for name,case in report['cases'].items()}},indent=2))
