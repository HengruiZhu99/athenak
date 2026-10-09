from pathlib import Path
import hashlib,json,subprocess
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources=[p for p in sorted(P.rglob('*')) if p.is_file() and 'build-attempts' not in p.parts and p.name not in ('local-recipe.json','cases.json','cases.txt','held-preparation.json')]
radii=[0.,.025,.049,.05,.1,.3,.45,.6,.85,.9,.95,.98,.995]
directions=[[1.,0.,0.],[.36,-.48,.8],[.8,.36,-.48]]
coeff=[]
for field in range(4):
 for degree in range(4):
  c=[[0.]*4 for _ in range(4)];c[field][degree]=1.;coeff.append({'name':f'field{field}-rho{degree}','coeff':c})
coeff.append({'name':'mixed','coeff':[[.17,-.11,.07,-.03],[.13,.19,-.05,.02],[-.23,.09,.04,-.01],[.21,-.08,.06,-.025]]})
cases=[]
for r in radii:
 for d in directions:
  if r==0 and d!=directions[0]:continue
  for c in coeff:cases.append({'r':r,'direction':d,'point':[r*a for a in d],**c})
(P/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
(P/'cases.txt').write_text(''.join(' '.join(format(a,'.17g') for a in c['point']+sum(c['coeff'],[]))+'\n' for c in cases))
r={'kind':'NARROW stable embedding identity diagnostic, unchanged inertial witness/source gates','prepared_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'runtime_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2',
 'sources':[{ 'path':str(p.relative_to(P)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sources],
 'reference_input_pins':[{'path':'../'+n,'sha256':sha(P.parent/n)} for n in ['taylor.hpp','complete_reference.hpp','reference-local-recipe.json','reference-gate-attempt001/receipt.json']],
 'independent_reference_gate':{'path':'../../../continuum/einstein-reference-jet-oracle-20261009/radial-attempt002/receipt.json','sha256':'b435db2e1be942ed9eb75bd4340c41c728dc1b2728232efdf8510b4f29843ef9'},
 'cases':{'count':len(cases),'metadata_sha256':sha(P/'cases.json'),'input_sha256':sha(P/'cases.txt'),'radii':radii,'directions':directions,'families':'four independent PHYSICAL INERTIAL displacement/velocity envelopes times rho^0..3 plus fixed mixed, then exact stationary inverse-embedding adapter'},
 'thresholds':{'synthetic_scaled':5e-13,'core_full_jet_scaled':5e-11,'core_acceleration_scaled':5e-11,'geometry_rhs_entry_scaled':5e-10,'gauge_attribution_entry_scaled':5e-11,'physical_constraint_scaled_by_full_input_jet':5e-10,'input_normal_scaled_by_input_values':5e-11,'spatial_normal_scaled_by_full_input_jet':5e-11,'output_normal_scaled_by_rhs':5e-11,'reference_rhs_absolute':5e-10,'reference_physical_constraint_absolute':5e-10,'directional_fd_final_scaled':2e-7,'negative_wrapper_min_difference':1.,'physical_vs_factored_moderate_r_scaled':2e-7,'inertial_adapter_identity_scaled':5e-10},
 'fd_eps':[1e-5,3e-6,1e-6,3e-7,1e-7],'additive_refinement':'new witness family only, prior broad-coordinate controls retain FAILED status; fixed five-epsilon sequence inherited; no epsilon/radius/threshold relaxation',
 'fd_acceptance':'all sequences retained; final h must pass; classify truncation decrease if final<=.5 first, else within declared absolute floor if every error<=2e-7, otherwise unclassified/fail',
 'binding':'all raw22 independent input directions plus native free20 chart at finite background 0/.003 and r .025/.3/.6/.9/.98; all22 actual output rows; actual double wrapper versus exact generic dual',
 'reference_export':{'mode':'--reference','fields':'ordinary Cartesian derivatives, fixed totaldegree3 index order i=0..3,j=0..3-i,k=0..3-i-j; names omega alpha chi P physical_lapse beta0..2 and g/bar/physical/A/K Cartesian row-major 00..22','scope':'C++ full reference through3; independent radial Omega4 remains separately gated'},
 'source_only_HELD_until_root_review':True,
 'scope':{'local_source_and_coordinate_lift_only':True,'operator_or_boundary_or_eigen_or_evolution_admitted':False,'actual_momentum_source_spatial_derivatives_or_Cdot_admitted':False,'expected_kinematic_output_is_not_independent_actual_Cdot':True,'strictly_finite_Omega':True,'no_higher_reference_placeholders_consumed':True}}
(P/'local-recipe.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({'recipe_sha256':sha(P/'local-recipe.json'),'cases':len(cases),'sources':len(sources)}))
