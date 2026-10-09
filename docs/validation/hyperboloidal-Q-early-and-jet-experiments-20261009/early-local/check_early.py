from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2];C=ROOT/'build-layer-research/continuum/conformal-q-followup/immutable-Q-null-finite-frequency-negative-20261009'
old=json.loads((C/'fourier.json').read_text());new=json.loads((P/'fourier.json').read_text());assert len(new)==560

def matrix(d):
 m=np.array(d['M']);return m[:,:,0]+1j*m[:,:,1]
original={(d['a'],d['r'],d['k'],d['dir'],d['form']):matrix(d)for d in old};oldroots=json.loads((C/'fourier-roots.json').read_text());rows=[];delta_error=outer_error=core_error=k0_error=0.
for d in new:
 m=matrix(d);assert np.isfinite(m).all();key=(d['a'],d['r'],d['k'],d['dir']);delta=m-original[key+(2,)]
 allowed=np.zeros((20,20),bool);allowed[4,[0,1,4,7]]=True
 delta_error=max(delta_error,float(np.max(np.abs(delta[~allowed]))),float(np.max(np.abs(delta.imag))))
 if d['r']>=.95:outer_error=max(outer_error,float(np.max(np.abs(delta))))
 if d['r']==.45:core_error=max(core_error,float(np.max(np.abs(m-original[key+(3,)]))))
 e=np.linalg.eigvals(m);j=e.real.argmax();rows.append({k:d[k]for k in ['a','r','Omega','k','dir','form']}|{'max_real':float(e[j].real),'imag':float(e[j].imag),'positive_roots':int(np.sum(e.real>1e-8))})
assert max(delta_error,outer_error,core_error)<1e-10
source=json.loads((P/'source.json').read_text());assert source['source_rows']==192 and source['pole_columns']==160
assert source['reference_fixedpoint_error']<1e-11 and source['generic_4D_Box_feedback_delta_error']<1e-11
assert source['identical_outer_gauge_error']==source['identical_leading_gauge_pole_error']==0
allrows=oldroots+rows;byk=[];byr=[];resolved=[]
for a in [.5,.75,1,2]:
 for f in [0,2,3,4,5]:
  for k in [0,4,16,64,256]:byk.append(max([r for r in allrows if r['a']==a and r['form']==f and r['k']==k],key=lambda r:r['max_real']))
  for radius in [.45,.65,.8,.85,.9,.95,.98]:byr.append(max([r for r in allrows if r['a']==a and r['form']==f and r['r']==radius],key=lambda r:r['max_real']))
  resolved.append(max([r for r in allrows if r['a']==a and r['form']==f and r['k'] in [0,4,16]],key=lambda r:r['max_real']))
report={'passed_source_principal_and_outer_pole_identity_gates':True,'sampled_frozen_spectrum_nonpositive':False,'global_native_accepted':False,'new_matrix_count':560,'frozen_control_matrix_count':1120,'matched_parameter_points':280,'form4':'sigma5 with v=Wgauge(.45,.85),alpha inner blend','form5':'sigma5 with v=Smooth(.65,.85),alpha inner blend','k_convention':'Unscaled Cartesian coordinate phase exp(+ik n.x), primitive20 amplitude constant; no Omega amplitude normalization. d=ikn,dd=-k²nn before algebraic dual completion.','N16_span2p2_Nyquist':float(np.pi*16/2.2),'N24_span2p1_Nyquist':float(np.pi*24/2.1),'sampled_coarse_resolved_k':[0,4,16],'resolved_scope':'Maxima only over sampledk, not all frequencies below Nyquist; no k32 or direct Nyquist probe here. k16 is resolved by both native resolutions.','worst_by_a_form_k':byk,'worst_by_a_form_radius':byr,'worst_over_sampled_coarse_resolved_k':resolved,'delta_outside_beta_radial_allowed_value_columns_error':delta_error,'identical_outer_matrices_error':outer_error,'identical_W0_baseline_matrices_error':core_error,'source':source,'admissibility':'Fixed current geometry .05/.95, S1,a>=.5 and gauge r1<=.85; on earlier supports Ωr=wgeo_prime*(outer−1)+wgeo*(-r/a)<0. sigma5 fixed finite. Not parameter-general admission; strict norm guard has no floor.','recommendation':'Earlier onset reduces late-onset resolved primitive growth substantially but retains positive k16 roots. One cheap actualglobal W control is a defensible attribution experiment, not stability/native admission. No earlier-weight native/global run performed here.'}
(P/'fourier-roots.json').write_text(json.dumps(rows,indent=2)+'\n');(P/'check-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'new_matrices':560,'delta_error':delta_error,'outer_error':outer_error,'core_error':core_error,'targeta0p5_byk':[r for r in byk if r['a']==.5]},indent=2))
