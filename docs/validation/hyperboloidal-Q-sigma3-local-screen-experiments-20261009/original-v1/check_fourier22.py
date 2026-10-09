"""Actual C0/Q frozen local screen; no subsidiary/global/native conclusion."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
meta=json.loads((P/'metadata-release.json').read_text());data=np.fromfile(P/'matrices-release.bin',dtype=np.float64)
assert len(meta['rows'])==1960 and data.size==1960*1768
z=data.reshape(1960,1768);u=z[:,:968].reshape(-1,22,22,2);v=z[:,968:].reshape(-1,20,20,2)
raw=u[:,:,:,0]+1j*u[:,:,:,1];chart=v[:,:,:,0]+1j*v[:,:,:,1]
assert np.isfinite(raw).all() and np.isfinite(chart).all()
ix=[18,0,7,17,19,20,21,1,2,3,4,5,8,9,10,11,12,14,15,16]
coeff={(d['a'],d['r']):d for d in meta['lifts']};lookup={tuple(d[k]for k in ['a','r','k','dir','form']):i for i,d in enumerate(meta['rows'])}
values=np.array([m[ix,:]@np.array(coeff[(d['a'],d['r'])]['B']) for d,m in zip(meta['rows'],raw)])
control_error=early_error=0
old_path=ROOT/'build-layer-research/continuum/conformal-q-followup/immutable-Q-null-finite-frequency-negative-20261009/fourier.json'
old=json.loads(old_path.read_text());assert len(old)==1120
for d in old:
 q=np.array(d['M']);m=q[:,:,0]+1j*q[:,:,1];i=lookup[tuple(d[k]for k in ['a','r','k','dir','form'])]
 control_error=max(control_error,float(np.max(np.abs(chart[i]-m))))
early_path=ROOT/'build-layer-research/continuum/q-null-early-feedback/immutable-Q-null-early-feedback-local-20261009/fourier.json'
early=json.loads(early_path.read_text());assert len(early)==560
for d in early:
 if d['form']!=4:continue
 q=np.array(d['M']);i=lookup[tuple(d[k]for k in ['a','r','k','dir'])+(4,)]
 early_error=max(early_error,float(np.max(np.abs(chart[i]-(q[:,:,0]+1j*q[:,:,1])))))
assert control_error==early_error==0

def delta_expected(c,weight):
    alpha,chi,omega=c['alpha'],c['chi'],c['Omega'];g=np.array(c['inverse']);grad=np.array(c['dOmega']);beta=np.array(c['beta']);bg=beta@grad;w=g@grad
    dN=np.zeros(22);dN[0]=grad@g@grad;dN[18]=2*bg*bg/alpha**3;dN[19:22]=-2*bg*grad/alpha**2
    for j,(n,m)in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):dN[1+j]=-chi*w[n]*w[m]*(1 if n==m else 2)
    D=np.zeros((22,22));D[19:22,:]=(-2*weight*alpha*alpha/(omega*(grad@grad)))*grad[:,None]*dN[None,:]
    return D
rank_one_error=chart_rank_one_error=values_rank_one_error=0
k0_error=geometry_error=outer_error=core_error=0
raw_geometry=[i for i in range(22)if i not in [18,19,20,21]]
for d,m in zip(meta['rows'],raw):
 key=tuple(d[k]for k in ['a','r','k','dir']);form=d['form'];i=lookup[key+(form,)];c=coeff[key[:2]]
 geometry_error=max(geometry_error,float(np.max(np.abs(m[raw_geometry,:]-raw[lookup[key+(3,)]][raw_geometry,:]))))
 if d['k']==0:
  j=lookup[(d['a'],d['r'],0,1-d['dir'],form)]
  k0_error=max(k0_error,float(np.max(np.abs(m.imag))),float(np.max(np.abs(m-raw[j]))),float(np.max(np.abs(chart[i].imag))),float(np.max(np.abs(chart[i]-chart[j]))))
 if form in [5,6]:
  oldform=2 if form==5 else 4;weight=c['Vlate'] if form==5 else c['W'];expect=delta_expected(c,weight);B=np.array(c['B']);j=lookup[key+(oldform,)]
  rank_one_error=max(rank_one_error,float(np.max(np.abs(m-raw[j]-expect))))
  chart_rank_one_error=max(chart_rank_one_error,float(np.max(np.abs(chart[i]-chart[j]-expect[ix,:]@B))))
  values_rank_one_error=max(values_rank_one_error,float(np.max(np.abs(values[i]-values[j]-expect[ix,:]@B))))
 if d['r']>=.95 and form in [4,6]:
  same=2 if form==4 else 5;j=lookup[key+(same,)]
  outer_error=max(outer_error,float(np.max(np.abs(m-raw[j]))),float(np.max(np.abs(chart[i]-chart[j]))))
 if d['r']==.45 and form in [2,4,5,6]:
  j=lookup[key+(3,)];core_error=max(core_error,float(np.max(np.abs(m-raw[j]))),float(np.max(np.abs(chart[i]-chart[j]))))
assert max(rank_one_error,chart_rank_one_error,values_rank_one_error)<1e-9
assert geometry_error==k0_error==outer_error==core_error==0
normal=meta['normal_gate'];binding=meta['binding']
assert normal['intrinsic_input_consumed_jet_scaled']<1e-10 and normal['intrinsic_output_scaled']<1e-11
assert binding['cases']==2352 and binding['single_beta_pole_assembly_error']<1e-12
assert binding['relative_FD_error'][2]<1e-7
assert binding['relative_FD_error'][1]<binding['relative_FD_error'][0]/50
assert binding['relative_FD_error'][2]<binding['relative_FD_error'][1]/20
assert json.loads((P/'metadata-debug.json').read_text())==meta
assert (P/'matrices-debug.bin').read_bytes()==(P/'matrices-release.bin').read_bytes()

form_names={0:'global Q preferred sigma0',1:'global Q preferred sigma5 late',2:'physical-inner Q preferred sigma5 late',3:'physical-P sourceoff spatial-norm baseline',4:'physical-inner Q preferred sigma5 early Wgauge',5:'physical-inner Q preferred sigma3 late',6:'physical-inner Q preferred sigma3 early Wgauge'}
kinds={'raw22':raw,'intrinsic20':chart,'value_only_RJB20':values}
allrows=[];eigenvalues={};worst=[];byk=[];byr=[];band=[]
for kind,mat in kinds.items():
    roots=np.linalg.eigvals(mat);eigenvalues[kind]=roots
    for d,e in zip(meta['rows'],roots):
        j=int(e.real.argmax());allrows.append(d|{'kind':kind,'max_real':float(e[j].real),'imag':float(e[j].imag),'positive_roots':int(np.sum(e.real>1e-8))})
    for a in [.5,.75,1,2]:
      for form in range(7):
        selected=[d for d in allrows if d['kind']==kind and d['a']==a and d['form']==form]
        worst.append(max(selected,key=lambda d:d['max_real']))
        for k in [0,4,16,64,256]:byk.append(max([d for d in selected if d['k']==k],key=lambda d:d['max_real']))
        for r in [.45,.65,.8,.85,.9,.95,.98]:byr.append(max([d for d in selected if d['r']==r],key=lambda d:d['max_real']))
        for N,span in [(16,2.2),(24,2.1),(36,2.1),(48,2.1)]:
          ny=float(np.pi*N/span);sample=[d for d in selected if d['k']<=ny]
          band.append(max(sample,key=lambda d:d['max_real'])|{'N':N,'span':span,'coordinate_pi_over_h':ny,'sampled_k_below_scalar_coordinate_Nyquist':sorted(set(d['k']for d in sample))})
np.savez_compressed(P/'roots.npz',**eigenvalues)
(P/'root-maxima.json').write_text(json.dumps(allrows,indent=2)+'\n')
report={'passed_actual_binding_normals_control_and_rank_one_gates':True,'local_matrices_per_kind':1960,'total_primitive_spectra':5880,'matrix_kinds':{
 'raw22':'All 22 primitive components independently have frozen exp(+ik n.x) jets; includes det/trace algebraic normals.',
 'intrinsic20':'20 free amplitudes with nonlinear det1/tracefree completion before linearization. Differentiated coefficient lift B(x) is retained in consumed jets.',
 'value_only_RJB20':'R J22 B(x0) using only the value lift. Not the intrinsic coefficient-jet restriction on a nonflat background and may violate differentiated algebraic constraints.'},
 'forms':form_names,'grid':{'a':[.5,.75,1,2],'S':1,'kappa_input':10,'kappa2':0,'r':[.45,.65,.8,.85,.9,.95,.98],'k':[0,4,16,64,256],'directions':[[1,0,0],[.36,-.48,.8]],'geometry':[.05,.95],'gauge':[.45,.85],'xi':'1/a','late_feedback':[.85,.95],'early_feedback':'Wgauge'},
 'old_1120_intrinsic20_control_error':control_error,'early_280_sigma5_control_error':early_error,
 'sigma3_minus_sigma5_analytic_rank_one_error_raw22':rank_one_error,'sigma3_minus_sigma5_analytic_rank_one_error_intrinsic20':chart_rank_one_error,'sigma3_minus_sigma5_analytic_rank_one_error_RJB20':values_rank_one_error,
 'geometric_rows_identical_all_forms_error':geometry_error,'k0_real_and_direction_identity_error':k0_error,'outer_equal_weight_identity_error':outer_error,'W0_physical_core_identity_error':core_error,
 'consumed_jet_and_output_normals':normal,'source_binding':binding,'Release_ASan_UBSan_matrix_bytes_and_metadata_equal':True,
 'max_intrinsic_vs_value_only_matrix_difference':float(np.max(np.abs(chart-values))),
 'positive_primitive_matrix_counts':{kind:int(np.sum(np.max(e.real,axis=1)>1e-8))for kind,e in eigenvalues.items()},
 'worst_by_a_form_kind':worst,'worst_by_a_form_kind_k':byk,'worst_by_a_form_kind_radius':byr,'sampled_bands_below_scalar_coordinate_Nyquist':band,
 'phase_and_scale':'Unscaled Cartesian coordinate phase exp(+ik n.x), |n|Euclidean=1; d=ikn, dd=-k²nn before chart completion. No Omega amplitude normalization. Coordinate wavelength 2pi/k. k_penrose=k*sqrt(chi*gtilde_inverse(n,n)); physical spatial k=Omega*k_penrose. k=0 is constant phase.',
 'Nyquist_scope':'Band tables only select tested coordinate k below pi/h; they are not a discrete spectrum or an accuracy claim. No native FD symbol, upwind, KO, boundary, or global coefficients are represented. Oblique Cartesian component Nyquist conditions differ from scalar |k|<pi/h.',
 'interpretation':'Positive frozen primitive roots are retained. This screen does not classify subsidiary growth, global continuum instability, or native stability. Sigma3 conditional finite-jet null/curvature tangency does not establish lower-order all-frequency stability.',
 'native_global_admission':False,'hierarchy_nonlinear_or_radiative_admission':False}
(P/'check-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'old_control_error':control_error,'rank_one_error':rank_one_error,'positive_matrix_counts':report['positive_primitive_matrix_counts'],'target_a0p5_intrinsic_by_k':[d for d in byk if d['a']==.5 and d['kind']=='intrinsic20' and d['form']in[2,3,4,5,6]]},indent=2))
