#!/usr/bin/env python3
"""Build explicit, source-linked experiment inventory; null means not extracted.

This inventory does not certify reproduction or identical equations across papers.
"""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'docs/elliptic-shift-1d'
rows=[]
common={
 '1412.3827':dict(equations='Appendix C (48a-j),(49a-c); C_Z4c=0,1 and GBSSN compared',gauge_reference='S=a=1 CMC Minkowski; harmonic reference-reduced lapse (40),(48g), fixed beta=-r/a',damping={'xi_alpha':1,'kappa1':1.5,'kappa2':0},grid='staggered regular center to scri',dissipation=.5,resolution=[400],CFL=.2,dt=.0005,method='FD4/RK4; polynomial ghost extrapolation'),
 '1412.4801':dict(equations='Appendix C of 1412.3827; gauges (9)-(11)',gauge_reference='CMC Schwarzschild trumpet; harmonic lapse with restoring initial Schwarzschild lapse; Gamma driver',damping={'xi_beta':5,'lambda':.75,'mu':'0.15+4(1-r^2)','eta':.1,'kappa1':None,'kappa2':None},grid='staggered; scri extrapolated',dissipation=None,resolution=None,CFL=None,dt=None,method='FD4/RK4+KO'),
 '1705.06298':dict(equations='Appendix A; drop nonprincipal C_Z4c terms; gauge equations Sections 3-4 and Appendix B',gauge_reference='CMC=-3, S=1; specified lapse/shift family',damping={'kappa1':1.5,'kappa2':0},grid='staggered except Section 6',dissipation=.25,resolution=[200],CFL=.1,dt=.0005,method='FD4/RK4, one-point off-centered advection'),
 '2304.05384':dict(equations='GBSSN, Section II; lapse (10), integrated Gamma shift (12) or (13)',gauge_reference='M=1, K_CMC=-1, CMC trumpet (24),(25) or gauge-adapted statio data',damping={'kappa1':.5,'kappa2':0,'kappa0':0,'m_cK':.1,'xi_cK':.5,'lambda':1,'xi_beta':0},grid='staggered at puncture and scri; outflow polynomial extrapolation',dissipation=.1,resolution=[456],CFL=None,dt=.0006667,method='FD4/RK4, shift-dependent upwind advection'),
 '2311.04972':dict(equations='Einstein+scalar setup of prior work; eikonal/advection diagram equations (49)',gauge_reference='CMC trumpet relaxation or regular scalar collapse',damping=None,grid=None,dissipation=None,resolution=None,CFL=None,dt=None,method=None),
 '2408.08952':dict(equations='BSSN/Z4c Section II; reference-reduced gauges (9),(10)',gauge_reference='Minkowski; K_CMC=-1, S=1; reference metrics Section V',damping={'kappa1':1.5,'kappa2':0,'xi_cK':1.5,'n_cK':1,'eta':0,'lambda':1},grid='staggered; reference-dependent origin parity',dissipation=.1,resolution=None,CFL=None,dt=None,method='FD4/RK4; upwind advection')}
# Explicit keyword spelling above is kept simple in the generated output.

def add(paper,id,section,initial_data,measured,expected,**changes):
 row=dict(common.get(paper,{}));row.update(id=id,paper=paper,section_figure=section,
  evidence_url='https://arxiv.org/html/'+paper,initial_data=initial_data,
  measured_quantities=measured,expected_result=expected,duration=None,
  status='unrun',relationship='Physical experiment target; hybrid equations/gauge differ from the published control',
  additional_physics=False,missing_information=[])
 row.update(changes)
 for key in ['equations','initial_data','gauge_reference','damping','grid','dissipation','resolution','CFL','dt','duration','measured_quantities','expected_result']:
  if row.get(key) is None:row['missing_information'].append(key+': not yet extracted or not specified; consult source figures/supporting files')
 rows.append(row)

p='1412.3827'
add(p,'regular-gauge','Sec. 3.2,8.1-8.2; Fig.3','alpha perturbation A=.1, sigma=.1,c=.25, Eq.(23)','all fields; outgoing/center-reflected pulses','return to Minkowski',duration=6)
add(p,'regular-scalar','Sec.3.3,8; Figs.4-6','physical scalar A=.058,sigma=.1,c=.25; Hamiltonian solve Eq.(24)','rescaled phi,Pi at scri; Bondi corrected signal; constraint convergence','subcritical dispersal and two outgoing signals',duration=10000)
add(p,'scalar-convergence','Sec.8.1-8.2; Fig.6','same .058 pulse; three resolutions','scri rescaled waveform differences','fourth order while differences resolved',resolution=[1200,1800,2700],CFL=.108,dt=[.00009,.00006,.00004],duration=None)
for form in ['GBSSN','Z4c C=0','Z4c C=1']:
 add(p,'large-a-'+form,'Sec.8.2','same lapse/scalar families, a=1000','stability and endpoint propagation','stable reported',gauge_reference='a=1000, fixed beta=-r/a',damping={'kappa1':.003,'kappa2':0},grid='staggered with fifth-order ghost extrapolation',equations=form)
for form,a,k in [('GBSSN',.8,3),('Z4c C=1',.88,1)]:
 add(p,'small-a-'+form,'Sec.8.2','same regular data','bounded evolution','reported stable lower-a example',equations=form,gauge_reference=f'a={a}',damping={'kappa1':k,'kappa2':0})
for form,cfl in [('GBSSN',.252),('Z4c C=1',.224),('Z4c C=0',.2),('Z4c C=0 k2=-1',.224)]:
 add(p,'CFL-'+form,'Sec.8.2','regular data, S=a=1','largest stable CFL','source threshold comparison; hybrid threshold separately measured',equations=form,resolution=[400],CFL=cfl)
p='1412.4801'
add(p,'trumpet-lapse','Sec.4 (unplotted gauge-wave experiment)','already-CMC trumpet plus lapse perturbation; amplitude/width not specified here','relaxation, mass, horizon','stable nonlinear trumpet perturbation; not wormhole-to-trumpet')
add(p,'BH-scalar-tail','Sec.4; Figs.3-4','CMC trumpet plus Eq.(12) scalar A=1e-4; c,sigma not given in this section','phi/Omega finite radius and scri; tail slopes, convergence','finite-radius slope -3, scri -2; reported scri -2.08 +/- .09; fourth order until about100',duration=100)
p='1705.06298'
for id,gauge in [('fixed-harmonic','fixed shift, harmonic lapse'),('fixed-cK','fixed shift,cK m=1'),('fixed-matched-cK','fixed shift,cK m=1 superposed n_alpha=2'),('1log-Gamma','1+log n=1,integrated Gamma lambda=.75,xi_beta=2'),('harmonic-both','harmonic lapse/shift with lambda=.1,n_beta=2 origin stabilization'),('PCG-origin','preferred conformal gauge matched r-=0,r+=.05;cK n=.4,Gamma lambda=1'),('PCG-layer','preferred conformal gauge matched r-=.4,r+=.9;cK n=.4,Gamma lambda=1')]:
 add(p,'scalar-signal-'+id,'Sec.5,5.1; Figs.2-3','constraint-solved .058 physical scalar pulse,sigma=.1,c=.25','phi/Omega,omega,delta-alpha,delta-beta at scri; Bondi reference','source-dependent gauge distortion and relaxation; retain distinct profiles',gauge_reference=gauge,duration=None)
add(p,'actual-v-staggered','Sec.6.1-6.3; Figs.4-5','same scalar pulse; preferred conformal gauge','rescaled scri waveform convergence','fourth order in both layouts',grid='actual scri and staggered pair',resolution=[1600,2400,3600],CFL=.1,dt=None)
for id,note in [('restoring-strength','xi1=2,xi2=0 or xi1=0,xi2=2; weak fixed-shift xi1=1 fails'),('matching-location','r-<=.4 stable; later matching can oscillate'),('origin-shift','alpha^2 chi coefficient can excite origin instability; lambda=.1,n_beta=2 cure'),('shift-scri','integrated harmonic-coefficient Gamma needs xi_beta>=2')]:
 add(p,'gauge-sensitivity-'+id,'Sec.5','same regular scalar data','origin/scri fluctuations; relaxation times about6 or30',note)
p='2304.05384'
for data in ['CMC','statio']:
 for eta in [0,1]:
  add(p,f'{data}-advective-eta{eta}','Sec.VI.2; Fig.12',data+' trumpet data','profiles, constraints,horizon, invariant mass','eta=0 growing oscillations; eta=1 relaxation to common final slice',gauge_reference='slicing(10),shift(12)',damping={**common[p]['damping'],'eta':eta},duration=100)
 add(p,f'{data}-nonadvective','Sec.VI.3; Figs.13-14',data+' trumpet data','Lambda norm,H profiles,self-convergence','statio visually stationary; CMC relaxes; preserve reported convergence failures',gauge_reference='slicing(10),shift(13)',duration=1000,resolution=[304,456,684],dt=[.001,.0006667,.0004444])
 add(p,f'{data}-lapse-pulse','Sec.VI.3; Fig.15',data+' trumpet plus Eq.(41) lapse A=.05,sigma=.1,rc=.25','H radial convergence at .1 and4; outgoing/ingoing gauge waves','constraint convergence; coordinate profiles need not match hybrid',duration=4,resolution=[304,456,684],dt=[.001,.0006667,.0004444])
for id,change in [('kappa1',{'kappa1':1.5}),('slicing',{'m_cK':1,'xi_cK':1})]:
 add(p,'CMC-nonadvective-'+id,'Sec.VI.3; Figs.13-14','CMC trumpet','field and lapse convergence; long-term constraint floor','retain loss of convergence; stronger lapse case recovers fourth order near300',duration=1000,gauge_reference='slicing(10),shift(13)',damping={**common[p]['damping'],**change},resolution=[304,456,684])
p='2311.04972'
add(p,'diagram-trumpet','Sec.IV.2; Figs.9-11','CMC trumpet, metric evolution plus diagram eikonal fields','relaxing slices, stationary height-function cross-check with time rescaling c','visible settling by about4; independent height/eikonal agreement',duration=8,relationship='Physical relaxation overlaps 2304.05384; eikonal diagram diagnostic is an additional extraction test')
add(p,'scalar-collapse','Sec.IV.3; Fig.12','regular constraint-solved scalar pulse amplitude .07; Fig.6 initial slice','horizon onset and mass; inner limiting R; conformal diagrams','AH around4.4, final mass about.11, limiting R about.16 in source gauge',duration=None)
add(p,'stationary-diagrams','Sec.III; Figs.1-8,Appendix B','CMC and stationary Schwarzschild slices','height functions and diagrams','analytic/numerical initial-data construction only',status='not_an_evolution',relationship='Initial-data/diagram verification supporting trumpet evolution')
p='2408.08952'
refs=['CMC','c+=1','cos','sin','sim','mod','modvar','layCMC','laymod','layvar']
for ref in refs:
 for form in ['BSSN','Z4c']:
  add(p,'reference-'+ref+'-'+form,'Secs.V-VI; Fig.4',ref+' Minkowski reference; '+('noise amplitude1e-4' if form=='Z4c' else 'unperturbed'),
      'constraint monitor Eq.(38); drift; origin and layer errors',
      'Z4c all reported stable; BSSN only CMC,sin,sim stable; retain constraint floors',equations=form+' Section II',
      gauge_reference=ref+' reference, equations(9),(10), C=2.8 for mod,2 otherwise')
for ref in ['CMC','modvar','sin','layvar']:
 add(p,'convergence-'+ref,'Sec.VI; Fig.5',ref+' with small Gaussian-like lapse pulse; amplitude not specified in text',
     'all-field norm self-convergence; Arr pointwise near layer at .2',
     'CMC/modvar approach4; sin drift; layvar first order near joint',
     resolution=[[200,600,1800],[300,900,2700],[400,1200,3600]],duration=None)
add(p,'modvar-unperturbed','Sec.VI; Fig.5 top-right','modvar reference without pulse','all-field norm convergence','fourth order',resolution=[[200,600,1800],[300,900,2700],[400,1200,3600]])
# Additional explicitly required experiments beyond these papers.
for id,initial,measured in [('wormhole-to-trumpet','isotropic interior with physical lapse psi^-2 and independently derived hyperboloidal Schwarzschild exterior','R_inner,proper distance,lapse collapse,AH and invariant mass'),('fixed-scalar-oracle','analytic radial Minkowski scalar pulse or independent fixed-Schwarzschild solver','waveform and endpoint convergence'),('compatible-violations','weighted compatible H/M/Theta/Z perturbations without projections','propagation,damping and weights'),('scri-pulse-sweep','vary amplitude,width,location approaching scri','clock,pole compatibility,bulk/endpoint error')]:
 add('user-required',id,'User objective',initial,measured,'derive quantitative criteria before acceptance',evidence_url='../objective.txt',relationship='Mandatory extension beyond published experiments')
followups=[
 {'arxiv':'2409.02994','url':'https://arxiv.org/abs/2409.02994','relevance':'Generalized harmonic dual-foliation spherical scalar, pure gauge, constraint violating/satisfying evolutions. Physical controls relevant; equations differ. Detailed test inventory still open.'},
 {'arxiv':'2505.17176','url':'https://arxiv.org/abs/2505.17176','relevance':'Einstein-Maxwell-charged-scalar regular pulses,BH perturbation,collapse. Neutral vacuum/massless limits relevant; charged cases require Maxwell and complex scalar physics.'},
 {'arxiv':'2506.15311','url':'https://arxiv.org/abs/2506.15311','relevance':'Nonlinear BH scalar QNM/tails. Explicit uncharged fundamental-mode control applicable; charged cases require additional physics. Detailed parameter extraction open.'},
 {'arxiv':'2506.05467','url':'https://arxiv.org/abs/2506.05467','relevance':'Neutron-star/BH constraint-solved initial data construction, not an evolution reproduction; hydrodynamic matter requires additional physics.'},
 {'arxiv':'2511.21406','url':'https://arxiv.org/abs/2511.21406','relevance':'Comment disputing charged-tail claims. Must inspect before interpreting charged extensions; not itself an evolution target.'}
]
output={'status':'initial source-linked inventory; follow-up full-text audit and figure-axis duration extraction unfinished',
 'retrieval':'retrieval.json','rows':rows,'followups':followups,
 'accounting':'All rows unrun unless explicitly non-evolution; physical experiments shared between papers can share compatible trajectories, but different gauges/equations must remain labeled controls. Missing values are not invented.'}
(root/'literature'/'reproduction-matrix.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'followups':len(followups)}))
