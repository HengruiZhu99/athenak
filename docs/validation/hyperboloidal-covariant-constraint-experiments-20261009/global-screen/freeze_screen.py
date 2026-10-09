"""Freeze C1 stage/verified-short/exploratory-long negative screen separately."""
from pathlib import Path
import hashlib,json,math,shlex,shutil,subprocess,zipfile
import numpy as np
here=Path(__file__).resolve().parent;work=here.parent;v2=work/'full22-candidate';root=work.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(n,d):
 p=here/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def cp(p,n):
 q=here/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def artifact(p):
 d={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'copied':False}
 if p.suffix=='.npz':
  a={}
  with zipfile.ZipFile(p) as z:
   for n in z.namelist():
    with z.open(n) as f:
     version=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format._read_array_header(f,version)
     a[n]={'shape':list(shape),'dtype':str(dtype),'fortran_order':order}
  d['arrays']=a
 return d
for folder,label in [(work,'native20'),(v2,'full22')]:
 for p in sorted(folder.iterdir()):
  if p.is_file() and (p.suffix in ['.cpp','.hpp','.py','.diff'] or p.name.startswith('build-') or p.name.startswith('dependencies-')):
   cp(p,Path('sources')/label/p.name)
for p in (work/'overlay').rglob('*'):
 if p.is_file():cp(p,Path('sources/overlay')/p.relative_to(work/'overlay'))
for n in ['build-provenance.json','source-preparation.json','scratch-build-correction.json']:
 cp(work/n,Path('receipts')/n)
for n in ['pilot-vs-frozen-C0.json','exploratory-t2-vs-frozen-C0.json']:
 cp(v2/n,Path('receipts')/n)
summary={'scope':'finiteOmega C1+derived covector Lambda; same actual native Cartesian stencils/ghosts; short independent numerical gate and exploratory long negative screen, no adoption','parameters':{'N':16,'span':2.2,'h':.1375,'points':1640,'free20':32800,'raw22':36080,'S':1,'a':.5,'r0':.05,'r1':.95,'kappa':10,'sym_ghost_degree':2,'KO':.1,'min_Omega':.0026953124999994555,'nominal_pole_dt':8.085937499998367e-5,'outward_crossing_time':.7457643839234269},'native_or_global_stability_accepted':False,'long_canonical_comparison_completed':False,'gauges':{}}
large={}
for g in ['production','spatialnorm']:
 p=v2/f'{g}-cache0.0001-validation.json';d=read(p);coords=d['metadata'].pop('xyz_omega_volume_ginv_chi');d['metadata']['coordinates_metadata_only']={'shape':[len(coords),len(coords[0])],'original_receipt_path':str(p),'original_receipt_sha256':sha(p)};save(Path('receipts')/p.name,d)
 for n in [f'{g}-pre-pilot-gate.json',f'{g}-projected-krylov-m50-80-h0.1-t0.05.json',f'{g}-projected-krylov-m50-80-h0.1-t2.0.json',f'{g}-krylov-pilot-independent-validation.json']:
  cp(v2/n,Path('receipts')/n)
 for t in ['0.05','2.0']:
  for kind in ['analysis','field-analysis']:cp(v2/f'{g}-projected-krylov-t{t}-{kind}.json',Path('receipts')/f'{g}-projected-krylov-t{t}-{kind}.json')
 for suffix in ['validation.log','cache0.0001-validation.stderr','krylov-pilot.log','pilot-independent.log','pilot-native-analysis.log','pilot-field-analysis.log','krylov-t2.log','t2-history-analysis.log','t2-fields-analysis.log']:
  cp(v2/f'{g}-{suffix}',Path('logs')/f'{g}-{suffix}')
 gate=read(v2/f'{g}-pre-pilot-gate.json');short=read(v2/f'{g}-krylov-pilot-independent-validation.json');long=read(v2/f'{g}-projected-krylov-m50-80-h0.1-t2.0.json');analysis=read(v2/f'{g}-projected-krylov-t2.0-analysis.json');fields=read(v2/f'{g}-projected-krylov-t2.0-field-analysis.json')
 assert gate['all_consistency_checks_pass']
 shorterr=max(v for row in short['checks'] for v in row['relative_state_l2']);assert shorterr<1e-10
 pair=max(s['coarse_fine_max_relative_difference'] for col in long['columns'] for s in col['steps']);assert pair<=1e-10
 constraint_eps=max(x['eps1e-5_vs3e-5_relative_constraints_l2'] for h in analysis['histories'] for x in h['constraint_amplitude_convergence']);assert constraint_eps<1e-7
 assert analysis['diagnostic_server_exit']==fields['server_exit']==0
 summary['gauges'][g]={'stage_consistency_checks_pass':True,'short_canonical_max_relative_state_error':shorterr,'short_canonical_total_seconds':short['total_seconds'],'exploratory_long_seconds':long['seconds'],'exploratory_long_matvecs':long['matvecs'],'max_long_coarse_fine_relative_difference':pair,'long_error_scope':'empirical local Arnoldi truncation difference; no independent long canonical forward-error validation','max_native_constraint_amplitude_check_relative_error':constraint_eps,'long_vs_frozen_C0':read(v2/'exploratory-t2-vs-frozen-C0.json')[g]}
 paths=[work/f'server-{g}',v2/f'server-{g}',work/f'{g}-validation-vectors.npz',v2/f'{g}-projected-J20.npz',v2/f'{g}-cache0.0001-metadata.json']
 paths += [v2/f'{g}-cache0.0001-{s}' for s in ['J22.npz','indptr.bin','indices.bin','data.bin','lift.bin','restrict.bin']]
 paths += [v2/f'{g}-projected-krylov-m50-80-h0.1-t{t}.npz' for t in ['0.05','2.0']]
 paths += [v2/f'{g}-projected-krylov-t{t}-native-diagnostics.npz' for t in ['0.05','2.0']]
 paths += [v2/f'{g}-krylov-pilot-canonical-vectors.npz']
 large[g]=[artifact(p) for p in paths]
large['other']=[artifact(v2/'diagnostic-fields'),artifact(work/'failed-misdirected-scratch-build-server-production')]
prov=read(work/'build-provenance.json');checks={}
for name,b in prov['builds'].items():
 bad=[p for p,s in b['compiler_dependency_hashes'].items() if sha(Path(p))!=s];archives=[p for p,s in b['link_archive_hashes'].items() if sha(Path(p))!=s]
 checks[name]={'dependencies':len(b['compiler_dependency_hashes']),'changed_dependencies':bad,'changed_archives':archives,'production_header_mismatches_vs27c':b['production_header_mismatches_vs27c']};assert not bad and not archives and not b['production_header_mismatches_vs27c']
# Capture the field-only diagnostic's exact build dependencies too.
cmd=read(v2/'build-diagnostic-fields.json');dc=[];skip=False
for x in cmd:
 if skip:skip=False;continue
 if x=='-o':skip=True;continue
 if x.endswith('.a'):continue
 dc.append(x)
dc+=['-M','-MT','fields'];dep=subprocess.check_output(dc,text=True);paths=sorted(set(str(Path(p).resolve()) for p in shlex.split(dep.replace('\\\n',' ').split(':',1)[1])))
save('field-diagnostic-build.json',{'command':cmd,'dependency_command':dc,'dependency_hashes':{p:sha(Path(p)) for p in paths},'executable_sha256':sha(v2/'diagnostic-fields')})
old=work.parent/'full-tensor-propagator';oracles={g:{'working_sha256':sha(old/f'server-{g}'),'preserved_sha256':sha(old/'projected-v1'/f'server-{g}')} for g in ['production','spatialnorm']};assert all(x['working_sha256']==x['preserved_sha256'] for x in oracles.values())
gateindex=root/'build-layer-research/continuum/covariant-z4-candidate/immutable-C1-stiffness-20261009/index.json'
identity={'launch_HEAD':prov['launch_HEAD'],'runtime_implementation':prov['runtime_implementation'],'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'dependency_identity_checks':checks,'explicit_overlay_hashes':prov['explicit_overlay_hashes'],'C1_math_sha256':sha(v2/'c1_additions.hpp'),'frozen_local_gate_index':{'path':str(gateindex),'sha256':sha(gateindex)},'restored_original_oracle_hashes':oracles,'unchanged_prior_global_manifest_sha256':sha(work.parent/'full-tensor-global-final/manifest.json'),'no_production_edits':subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''}
assert identity['no_production_edits'];assert identity['C1_math_sha256']=='908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7';save('source-identity-verification.json',identity);save('summary.json',summary);save('large-artifacts-metadata-only.json',large)
report='''# Covariant C1 Cartesian tangent: negative exploratory screen

This tests the mechanical Appendix-B C1 additions plus the separately derived covector Lambda correction on the actual 3D Cartesian full22/native20 infrastructure. Finite-Omega tensor/principal/local-pole gates passed first. The short independent numerical propagation gate passes, but the longer exploratory screen worsens M/Z and gives no basis for adoption or a costly long canonical/native pulse. It is not an all-time failure theorem. Production runtime remains27c19d20; launch HEADaef47b0a. Original global/stage/wave frozen archives are unchanged.

Parameters match the prior completed global control: N16,span2.2,h=.1375,1640 active nodes,32800 free20/36080 raw22,S1,a.5,wide(.05,.95),kappa10,symmetric quadratic ghosts,nativeKO.1. Both production and frozen private spatialnorm gauge branches are retained. The source addition changes P,Theta,A,Lambda only and uses live values/jets. The exact ignored CartesianPatch copy adds C1 after the existing C0 analytic-reference residual subtraction; C1 itself is never reference-subtracted. Cached Point adds the identical delta. No gauge changes, ghost/stencil changes, fields/floors/Theta falloff conditions, or production options were introduced. Strict interior/nonrecursive support is unchanged.

## Actual native implementation gate

Prepare changes the reference by0. Reference RHS max6.08357e-12 and H/M/Z3.23947e-14/1.33057e-15/6.64311e-16 match the old operator. All63576 donor references are strictly active; weight sum error1.33e-15. Raw CSR agrees with actual cached native LoadMeshJet/mixed/Lx/KO application to1.11e-15 relative. Six-amplitude nonlinear native22 RHS sweeps agree to worst best7.25e-10. The actual final-only nonlinear SSPRK3 derivative agrees with P_ref R3(dtJ22)Lift to worst best5.69e-10 over all tested seeds anddt/2,dt,2dt; nominaldt=.03minOmega=8.0859375e-5. All twelve stage consistency checks pass. These are implementation checks, not physical acceptance criteria.

Continuous propagation uses J20=P_ref J22Lift. It is distinct from the actual finite native final-only RK3 map. Independent raw Taylor expm_multiply versus m50/80 two-pass Arnoldi at t=.025,.05 agrees to maxrelative2.78e-14 for both seeds and both gauges. Each short Arnoldi run costs~5.9s; raw canonical controls cost105.78/93.47s. The matrix and all RHS/one-step/source checks use the actual native spherical ghost continuation; no scalar or primitive local Fourier substitute is used here.

The long t2 runs cost83.58/81.21s and have empirical local coarse/fine difference<=1e-10. Their all-time forward error is not independently certified: a full t2 Taylor comparison is intentionally pending, because the negative screen does not justify that cost. Native signed constraint derivative amplitude comparisons remain below1e-7. No generator eigensolve/Ritz search is used. All saved states and scalar receipts are finite. The long results below are explicitly exploratory.

## Matched comparison with frozen C0

The same original seed arrays are normalized to unit Euclidean free20 L2: native width.5 angular gauge pulse (.1 lapse/.02 shift before normalization) and controlled RNG690 shell random data with the original radial profile. Their generation source and exact vector hashes are retained. No new asymptotic condition is imposed by using these particular test vectors; raw white22 RHS and one-step gates separately include generic finite physicalTheta on the strict interior grid.

| Gauge/seed at t2 | C1 H/M/Z | C1/C0 H/M/Z | C1 cfgH1+momL2 amplification (C0) |
| --- | --- | --- | --- |
| Production gauge |1.648961/1.403795/.349237|.9584/1.1528/1.2813|44.1803(47.0983)|
| Spatialnorm gauge |2.335094/1.873042/.591101|1.0650/1.5348/1.5988|35.7733(35.7290)|
| Production shell |.00333776/.00388160/.00105359|.9593/1.0020/1.0872|.072925(.074887)|
| Spatialnorm shell |.00273790/.00262309/.00123051|1.0932/1.1481/1.2354|.017908(.016994)|

H is the native physical Hamiltonian; M/Z use reference Penrose inverse contractions and unweighted active-cell RMS. The stored-component units norm uses h^3sqrtgamma quadrature, configuration={chi,g,alpha,beta} H1 and momenta={P,A,Lambda,Theta} L2, with S1. It is not an invariant tensor energy, symmetrizer, or stability bound. Shell sampled maxima in this norm remain1 at initial time. At t=.05 the smooth M errors were already36.2%/40.6% larger, despite slightly lower H/Z; these negative short observations are independently Taylor-verified.

Final smooth squared r>=.9 fractions H/M/Z are.05778/.54038/.66572 production and.04798/.75784/.79728 spatialnorm. Peak H is in the bulk (r=.22802/.35724); peakM/Z near r=.99865. This locates the measured constraint error but does not isolate boundary causation. The local C1 gate separately retains large raw finiteTheta transients from the genuine Lambda double pole. Its Omega A/Omega Lambda similarity is analysis only, not imposed falloff or a runtime modification. No uniform unweighted propagator, regular nonlinear scri closure, finite-pulse stability, or accepted generator eigenvalues are claimed.

## Provenance and command correction

Exact source copies, original and patched CartesianPatch diffs, cached Point diff, source/math hashes, four AppleClang21 arm64 build/dependency inventories, static link archives, commands/logs and all stage/sweep/short/long observations are archived. Compiled production headers match27c19d20; the only geometry evolution change is the explicit C1 overlay. The frozen math is908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7. Large CSR/vector/state arrays and executables are metadata-only with original paths, shapes, byte counts and SHA256.

A copied native20 build command initially retained the old working scratch output path. Before any candidate result used it, that scratch binary was restored from its frozen projected-v1 copy and the command corrected. Both original working oracle hashes again match565f428.../495647..., and the correction receipt preserves the exact erroneous command, unused output hash, restoration path/hash and corrected source. No production source/executable or frozen archive bytes changed. Historical source-preparation.json predates this correction; authoritative compiled-source identity is build-provenance.json plus source-identity-verification.json and the current source catalog. The failed scratch binary is retained metadata-only and was never used for an observation.

This is a rejected exploratory screen with a verified short numerical gate, not an independently verified long stability result. Any blended lower-order follow-up requires its own formula/principal/coefficient/Einstein gates and separate source/receipt; this archive stays frozen.
'''
(here/'REPORT.md').write_text(report)
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(y) for y in x.values())
 if isinstance(x,list):return all(finite(y) for y in x)
 return True
assert all(finite(read(p)) for p in here.rglob('*.json'))
catalog={str(p.relative_to(here)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(here.rglob('*')) if p.is_file() and p.name!='manifest.json'}
save('manifest.json',{'scope':'actual full22/native short gate and exploratory t2 C1 negative screen; no long canonical/no adoption','files':catalog})
print(json.dumps({'files':len(catalog),'bytes':sum(x['bytes'] for x in catalog.values()),'REPORT_sha256':sha(here/'REPORT.md'),'summary_sha256':sha(here/'summary.json'),'manifest_sha256':sha(here/'manifest.json')},indent=2))
