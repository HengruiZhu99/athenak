"""Freeze completed finite-time global tangent evidence, without large arrays."""
from pathlib import Path
import hashlib,json,math,shlex,shutil,subprocess,sys,zipfile
import numpy as np
import scipy
here=Path(__file__).resolve().parent
root=here.parents[2]
work=here.parent/'full-tensor-propagator'
v1=work/'projected-v1'
v2=work/'full22-v2'
interim=here.parent/'full-tensor-stage-interim'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(name,data):
 p=here/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def copy(src,dst):
 dst=here/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def artifact(p):
 d={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'copied':False}
 if p.suffix=='.npz':
  arrays={}
  # Read NPY headers only; never duplicate whole propagated states in the archive.
  with zipfile.ZipFile(p) as z:
   for name in z.namelist():
    with z.open(name) as f:
     version=np.lib.format.read_magic(f)
     shape,order,dtype=np.lib.format._read_array_header(f,version)
     arrays[name]={'shape':list(shape),'dtype':str(dtype),'fortran_order':order}
  d['arrays']=arrays
 return d
source_names=['full22_server.cpp','projected_base.hpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp','expm_propagate.py','krylov_propagate.py','validate_krylov_pilot.py','analyze_history.py','analyze_fields.py','verify_global_operators.py','diagnostic_fields.cpp','diagnostic_constraint_norms.cpp','plot_global.py','build-production.json','build-spatialnorm.json','build-production.log','build-spatialnorm.log','build-provenance.json','build-diagnostic-fields.json','build-diagnostic-constraint-norms.json','build-diagnostic-fields.log','build-diagnostic-constraint-norms.log']
for name in source_names:copy(v2/name,Path('sources/full22-v2')/name)
for name in ['tangent_server.cpp','old-jv-source.cpp','native_injection.hpp','spatial_norm_control.hpp','validate.py','build-production.json','build-spatialnorm.json']:
 copy(v1/name,Path('sources/projected-v1')/name)
for name in ['canonical-vs-krylov-all-states.json','global-native-operator-diagnostic-verification.json']:
 copy(v2/name,Path('receipts')/name)
for g in ['production','spatialnorm']:
 for suffix in ['projected-expm-t2.0.json','projected-krylov-m50-80-h0.1-t2.0.json','projected-expm-analysis.json','projected-krylov-analysis.json','projected-krylov-field-analysis.json']:
  copy(v2/f'{g}-{suffix}',Path('receipts')/f'{g}-{suffix}')
 for suffix in ['expm.log','krylov-t2.log','canonical-analysis.log','krylov-analysis.log','field-analysis.log','fields-native.stderr','global-operator-check.stderr']:
  copy(v2/f'{g}-{suffix}',Path('logs')/f'{g}-{suffix}')
for name in ['production-projected-krylov-m50-80-h0.1-t0.05.json','production-krylov-pilot-independent-validation.json']:
 copy(v2/name,Path('receipts')/name)
for name in ['production-krylov-independent-validation.log','global-native-verification.log','direct-native-norm-check.stderr']:
 copy(v2/name,Path('logs')/name)
for name in ['global-tangent-histories.png','global-tangent-histories.svg']:copy(v2/name,name)

# Recheck every byte identified by the original compiler dependency receipt.
prov=read(v2/'build-provenance.json');identity={'implementation_reference':prov['implementation_reference'],'launch_HEAD':prov['launch_HEAD'],'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'compiler_version':prov['compiler_version'],'builds':{},'diagnostic_builds':{}}
for g,b in prov['builds'].items():
 mismatch=[p for p,s in b['compiler_reported_dependency_hashes'].items() if sha(Path(p))!=s]
 archives=[p for p,s in b['archive_hashes'].items() if sha(Path(p))!=s]
 identity['builds'][g]={'dependency_count':len(b['compiler_reported_dependency_hashes']),'changed_dependencies_since_compilation_receipt':mismatch,'changed_link_archives':archives,'production_header_mismatches_vs27c':b['production_header_mismatches'],'executable_sha256':sha(v2/f'server-{g}'),'executable_matches_build_receipt':sha(v2/f'server-{g}')==b['executable_sha256'],'native_RHS_oracle_sha256':sha(work/f'server-{g}'),'native_RHS_oracle_matches_frozen_v1':sha(work/f'server-{g}')==sha(v1/f'server-{g}')}
for stem,exe in [('diagnostic-fields','diagnostic-fields'),('diagnostic-constraint-norms','diagnostic-constraint-norms')]:
 cmd=read(v2/f'build-{stem}.json');dc=[];skip=False
 for x in cmd:
  if skip:skip=False;continue
  if x=='-o':skip=True;continue
  if x.endswith('.a'):continue
  dc.append(x)
 dc+=['-M','-MT','diagnostic'];out=subprocess.check_output(dc,text=True)
 (here/'sources/full22-v2'/f'dependencies-{stem}.make').write_text(out)
 deps=sorted(set(str(Path(p).resolve()) for p in shlex.split(out.replace('\\\n',' ').split(':',1)[1])))
 hashes={p:sha(Path(p)) for p in deps}
 mismatch=[]
 for p,s in hashes.items():
  q=Path(p)
  if q.is_relative_to(root/'src'):
   rel=str(q.relative_to(root));expected=subprocess.check_output(['git','show',prov['implementation_reference']+':'+rel],cwd=root)
   if hashlib.sha256(expected).hexdigest()!=s:mismatch.append(rel)
 identity['diagnostic_builds'][stem]={'command':cmd,'dependency_command':dc,'dependency_hashes':hashes,'archive_hashes':{p:sha(Path(p)) for p in cmd if p.endswith('.a')},'executable_sha256':sha(v2/exe),'production_header_mismatches_vs27c':mismatch}
identity['frozen_spatialnorm_wrapper_sha256']=sha(v2/'native_injection.hpp')
identity['frozen_spatialnorm_math_sha256']=sha(v2/'spatial_norm_control.hpp')
identity['wrapper_matches_build_receipt']=identity['frozen_spatialnorm_wrapper_sha256']==prov['frozen_spatialnorm_wrapper_sha256']
identity['math_matches_build_receipt']=identity['frozen_spatialnorm_math_sha256']==prov['frozen_spatialnorm_math_sha256']
identity['interim_manifest']={'path':str(interim/'manifest.json'),'sha256':sha(interim/'manifest.json'),'REPORT_sha256':sha(interim/'REPORT.md'),'scope':'Frozen native full22 stage/direct-one-step consistency checks; unchanged, not copied or modified here.'}
identity['python']={'version':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'runtime_commands':'OPENBLAS_NUM_THREADS=1 PYTHONPATH=build-layer-research/boundary/python-deps python3 SCRIPT.py GAUGE [arguments]'}
save('source-identity-verification.json',identity)

compare=read(v2/'canonical-vs-krylov-all-states.json');native=read(v2/'global-native-operator-diagnostic-verification.json');summary={'parameters':{'N':16,'span':2.2,'h':.1375,'active_points':1640,'free_dimension':32800,'raw_dimension':36080,'S':1,'a':.5,'r0':.05,'r1':.95,'kappa':10,'KO':.1,'symmetric_ghost_degree':2,'strict_nonrecursive_donors':True,'outward_reference_crossing_time':.7457643839234269,'final_time':2.,'crossings':2/.7457643839234269},'semantics':'exp(t J20), J20=P_ref J22 Lift, projected continuous semidiscrete generator. Native final-only RK3 map separately validated in immutable interim; not the same finite-step map.','norm_limit':'configuration H1 plus momentum L2 is a reference-weighted stored-component units diagnostic, not invariant tensor energy, symmetrizer, or proven bound.','receipt_semantics_note':'The historical field-analysis JSON semantics string says canonical pending; the complete all-state independent comparison in this archive supersedes that run-time status without changing original receipt bytes.','gauges':{},'retrospective_implementation_checks':[]}
large={}
for g in ['production','spatialnorm']:
 canonical=read(v2/f'{g}-projected-expm-t2.0.json');analysis=read(v2/f'{g}-projected-expm-analysis.json');fields=read(v2/f'{g}-projected-krylov-field-analysis.json');independent=read(v2/f'{g}-projected-krylov-m50-80-h0.1-t2.0.json');meta=read(v2/f'{g}-cache0.0001-metadata.json')
 rows=[]
 for ah,fh in zip(analysis['histories'],fields['histories']):
  hist=ah['history'];f=fh['history'];assert ah['name']==fh['name']
  rows.append({'seed':ah['name'],'initial_euclidean_free20_norm':hist[0]['euclidean_component_l2'],'initial_native_H_M_Z_rms':hist[0]['native_H_M_Z_rms'],'final':hist[-1],'final_field_units_norm':f[-1],'sampled_max_euclidean_amplification':max((x['euclidean_component_amplification'],x['time']) for x in hist),'sampled_max_configuration_H1_momentum_L2_amplification':max((x['configuration_H1_momentum_L2_amplification'],x['time']) for x in f)})
 op=max(min(x['relative_l2'] for x in r['amplitude_sweep']) for r in native['operator_checks'][g]['rows'])
 diag=max(min(max(x['relative_error_H_M_Z']) for x in r['amplitude_sweep']) for r in native['diagnostic_checks'][g] if r['direction']!='gauge_t0')
 field=native['field_derivative_identity'][g]['maxrelative_total_native_H1_vs_sum_field_value_gradient_norms']
 summary['gauges'][g]={'matrix_shape':canonical['shape'],'matrix_nnz':canonical['nnz'],'canonical_seconds':canonical['seconds'],'independent_seconds':independent['seconds'],'independent_matvecs':independent['matvecs'],'initial_original_seed_norms':canonical['initial_original_norms'],'seeds':rows,'allstate_comparison':compare[g],'worst_best_native_directional_RHS_relative_l2':op,'worst_best_native_Diagnose_H_M_Z_relative_error':diag,'field_derivative_identity_relative_error':field,'ritz_candidates_only':analysis['ritz_from_history'][:2]}
 checks=[('G1 source and operator identity',all(not identity['builds'][g][k] for k in ['changed_dependencies_since_compilation_receipt','changed_link_archives','production_header_mismatches_vs27c']) and identity['builds'][g]['executable_matches_build_receipt'] and identity['builds'][g]['native_RHS_oracle_matches_frozen_v1'],None,None),('G2 canonical versus independent all-state relative L2',compare[g]['max_relative_state_l2_all_81times_2seeds']<1e-10,compare[g]['max_relative_state_l2_all_81times_2seeds'],1e-10),('G3 native propagated-direction RHS sweep',op<1e-7,op,1e-7),('G4 native nonlinear Diagnose versus signed constraint tangent',diag<1e-8,diag,1e-8),('G5 native total versus grouped field derivatives',field<1e-10,field,1e-10),('G6 finite states, common times, native server success',compare[g]['all_canonical_finite'] and compare[g]['all_krylov_finite'] and compare[g]['canonical_times_equal_krylov'] and analysis['diagnostic_server_exit']==0 and fields['server_exit']==0 and native['operator_checks'][g]['exit_status']==0,None,None)]
 summary['retrospective_implementation_checks'] += [{'gauge':g,'check':name,'pass':bool(ok),'observed':value,'threshold':threshold} for name,ok,value,threshold in checks]
 paths=[work/f'server-{g}',v1/f'server-{g}',v1/f'{g}-validation-vectors.npz',v2/f'server-{g}',v2/f'{g}-projected-J20.npz',v2/f'{g}-projected-expm-t2.0.npz',v2/f'{g}-projected-krylov-m50-80-h0.1-t2.0.npz',v2/f'{g}-projected-expm-native-diagnostics.npz',v2/f'{g}-projected-krylov-native-diagnostics.npz',v2/f'{g}-projected-expm-ritz-vectors.npz',v2/f'{g}-projected-krylov-ritz-vectors.npz',v2/f'{g}-cache0.0001-metadata.json']
 paths += [v2/f'{g}-cache0.0001-{suffix}' for suffix in ['J22.npz','indptr.bin','indices.bin','data.bin','lift.bin','restrict.bin']]
 large[g]=[artifact(p) for p in paths]
large['diagnostic_executables']=[artifact(v2/x) for x in ['diagnostic-fields','diagnostic-constraint-norms']]
assert all(x['pass'] for x in summary['retrospective_implementation_checks'])
assert identity['wrapper_matches_build_receipt'] and identity['math_matches_build_receipt']
assert all(not b['production_header_mismatches_vs27c'] for b in identity['diagnostic_builds'].values())
save('summary.json',summary);save('large-artifacts-metadata-only.json',large)
report='''# Full tensor global native tangent: completed finite-time gate

The actual 3D Cartesian full20 projected continuous generator develops substantial smooth-gauge constraint error in both gauge choices. The private spatial-norm gauge reduces a chosen component/derivative norm but does not reduce all constraints. Controlled shell constraint-bearing data decays in that norm over the sampled window. These are single-grid finite-time observations, not an energy bound, convergent instability rate, whole-system stability proof, or boundary attribution. No production files, stencils, ghosts, or parameters were changed.

## Operator and time semantics

The grid has N16, span2.2, h=.1375,1640 active ball nodes,32800 free20 and36080 unconstrained22 variables. Parameters S1,a.5,r0=.05,r1=.95,kappa10,symmetric quadratic ray ghosts,native KO=.1. Both gauges use the same exact LayerReference jets, strict interior/nonrecursive ghost donors, native centered/mixed/one-sided advection stencils, KO mask, determinant-one and tracefree algebraic Lift/Restrict. The gauge source is production physical-lapse or the frozen private spatialnorm wrapper/math. No primitive Fourier approximation or scalar substitute is used here. The three dormant B fields in this physical gauge are absent from both operators.

This archive propagates J20=P_ref J22 Lift continuously: u(t)=exp(t J20)u0. Native SSPRK(3,3) projects only its final stage in the current no-dyngr lifecycle; its actual derivative is P_ref R3(dt J22)Lift. The separate immutable stage-interim archive validates that native map directly and quantifies its difference from every-RHS projected20. The present continuous histories must not be called exact native finite-RK3 histories.

## Independent propagation and native diagnostics

SciPy expm_multiply (Taylor action with exact trace) propagated both columns through81 times from0 to2:1902.53s production,1770.40s spatialnorm. Independent two-pass Arnoldi m50/80 with empirical coarse/fine tolerance1e-10 took about79s per gauge. Canonical and Arnoldi full states agree at all81 times and both seeds to relative L2 1.75451e-13/1.47518e-13. The truncation-pair test alone is not a rigorous nonnormal forward-error bound; the completed canonical comparison is the independent accuracy evidence. The exact block-similarity pilot is preserved in the interim archive; no cancelled long similarity result is used.

The actual native nonlinear RHS centered amplitude sweeps in propagated gauge(t1,t2) and shell(t2) directions agree with J20 to worst best relative L2 5.00091e-8 production and1.59826e-8 spatialnorm. All sweep points are retained, including roundoff deterioration at small amplitude. Direct native ProjectAlgebraic+Prepare+Diagnose norms agree with the signed linear constraint diagnostic to worst best H/M/Z relative1.74835e-9/9.92937e-11. Pure gauge initial H/M/Z is exactly0 in the linear diagnostic; nonlinear reference roundoff is retained as absolute error. Both checks differentiate at the stationary reference in the indicated direction, not about a finite evolved state.

H is the native physical Hamiltonian; M and Z are native conformal covectors. Their reported RMS uses reference gamma_bar inverse for M/Z and the native unweighted cell mean, not volume quadrature. Component field norms use Cartesian quadrature h^3 sqrt(det gamma_bar). The units diagnostic is sqrt integral [sum_(chi,g,alpha,beta)(value^2/S^2+gamma_barInv gradient^2)+sum_(P,A,Lambda,Theta)value^2]. Upper tensor components are counted once. This is an explicitly defined stored-component norm, not a coordinate-invariant tensor energy or a proved symmetrizer. Its reconstruction from grouped native values/derivatives agrees with the existing whole-field diagnostic within1.71e-13. The figure was visually checked after regeneration.

## Finite-time results and localization

Both seed vectors are independently normalized to unit initial Euclidean free20 L2. The smooth gauge seed exactly follows the native width.5 angular shape: s=r^2, f=(1-s)^4 exp(-4s), deltaAlpha=.1f(1+.2x+.3yz), deltaBeta=.02f(1+.3yz,.2x,.1xy). Its original free20 norm is.5724344949330551. The shell seed uses the recorded RNG690 stream, independent white20 components times exp(-((r-.88)/.065)^2)(1-r^2)^4; its original norm is.3531828899512467. The exact generation source is copied from projected-v1. This is a controlled random realization, not a worst-case perturbation search.

At t2 (2.68181 outward reference light-crossing times, T=.7457643839234269), smooth-gauge Euclidean amplification is106.5178 production /91.9671 spatialnorm. Configuration-H1 plus momentum-L2 amplification is47.0983/35.7290. The native H/M/Z RMS is1.720467/1.217738/.272567 versus2.192661/1.220403/.369710. Thus lower component amplitude does not imply lower constraint error. Squared H/M/Z fractions at r>=.9 are.06616/.37020/.64077 production versus.03331/.45522/.76909 spatialnorm. Peak H lies at r=.357235 in the bulk for both. Final peak Z is near r=.99865. The raw component norm peaks before t2 and then partly recedes; this does not establish an exponential growth rate.

Shell seed final Euclidean amplification is.615589/.165343; configuration-H1 plus momentum-L2 amplification is.0748867/.0169943. Its sampled maximum in the latter norm is1 at initial time. Final H/M/Z is.00347946/.00387373/.00096910 versus.00250440/.00228468/.00099605. Early raw Euclidean amplification reflects the different units and derivative orders of stored configuration/momentum fields; it is not by itself a physical growth measure.

The late-trajectory reduced Ritz candidates have positive real parts (~1.992+/-6.794i and1.915+/-6.792i), but full-generator relative action residuals1.60e-3/1.22e-3 are inadequate for an eigenvalue claim. They are archived only as exploratory candidates; this gate makes no spectral stability/instability conclusion. NumPy2 complex BLAS RuntimeWarnings in this exploratory extraction remain in original logs; all saved states/candidate arrays and scalar receipts are finite. The long propagation result does not depend on Ritz extraction.

## Provenance and limits

Production compiled headers and native lifecycle sources are byte-identical to27c19d20696ea6dd4704032c51dfd026218f64f2. Original launch HEAD is b37a20f2d7a8ccc42148f1a814a80b2957e17b53. All1153/1155 original compiler dependencies and four static Kokkos link archives were rehashed unchanged. AppleClang21 arm64 flags are -O3 -DNDEBUG -std=c++17; exact commands, wrapper/math identities, diagnostic dependency inventories, Python versions, original receipts/logs and six retrospective implementation checks per gauge are preserved. They are consistency checks, not pre-registered physical acceptance thresholds. Historical field receipts retain their original 'canonical pending' status string; the completed all-state comparison supersedes that status.

Sources are byte-preserved copies, and original scripts/commands operate at the recorded scratch paths with the metadata-only large arrays. CSR matrices, seed vectors, propagated states, diagnostic arrays, candidate vectors and executables are not embedded in this compact archive; paths, dimensions, byte counts and SHA256 identify them. This is finite-time N16 evidence without refinement, all-time norm control, nonlinear acceptance, optimized global spectrum, or a new closure/SBP claim. Existing wave/scalar/stage frozen archives are unchanged. No costly propagation remains active.
'''
(here/'REPORT.md').write_text(report)
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(y) for y in x.values())
 if isinstance(x,list):return all(finite(y) for y in x)
 return True
jsons=list(here.rglob('*.json'))
assert all(finite(read(p)) for p in jsons)
save('receipt-audit.json',{'all_json_finite':True,'json_count_excluding_manifest':len(jsons),'all_12_retrospective_consistency_checks_pass':True,'no_production_edits':True,'figure_visually_checked':True,'large_arrays_metadata_only':True,'old_interim_manifest_sha256':sha(interim/'manifest.json')})
catalog={str(p.relative_to(here)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(here.rglob('*')) if p.is_file() and p.name!='manifest.json'}
save('manifest.json',{'scope':'completed single-grid full20 projected continuous global finite-time propagation; exact native stage relationship separately frozen','files':catalog})
print(json.dumps({'files':len(catalog),'bytes':sum(x['bytes'] for x in catalog.values()),'report_sha256':sha(here/'REPORT.md'),'summary_sha256':sha(here/'summary.json'),'manifest_sha256':sha(here/'manifest.json'),'all_checks_pass':True},indent=2))
