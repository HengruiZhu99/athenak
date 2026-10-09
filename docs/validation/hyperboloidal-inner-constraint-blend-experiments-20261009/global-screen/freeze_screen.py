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
for n in ['build-provenance.json','gate-bound-sources.json','SOURCE_PREPARATION_HOLD.json']:
 cp(work/n,Path('receipts')/n)
for n in ['pilot-vs-frozen-C0.json','exploratory-t2.0-vs-frozen-C0.json','actual-matrix-support-attribution.json']:
 cp(v2/n,Path('receipts')/n)
summary={'scope':'prescribed bulk1−Wgauge C1+covector repair; same actual native Cartesian stencils/ghosts; short verified and long exploratory screen, no adoption','parameters':{'N':16,'span':2.2,'h':.1375,'points':1640,'free20':32800,'raw22':36080,'S':1,'a':.5,'r0':.05,'r1':.95,'kappa':10,'sym_ghost_degree':2,'KO':.1,'min_Omega':.0026953124999994555,'nominal_pole_dt':8.085937499998367e-5,'outward_crossing_time':.7457643839234269},'native_or_global_stability_accepted':False,'long_canonical_comparison_completed':False,'gauges':{}}
large={}
for g in ['production','spatialnorm']:
 p=v2/f'{g}-cache0.0001-validation.json';d=read(p);coords=d['metadata'].pop('xyz_omega_volume_ginv_chi');d['metadata']['coordinates_metadata_only']={'shape':[len(coords),len(coords[0])],'original_receipt_path':str(p),'original_receipt_sha256':sha(p)};save(Path('receipts')/p.name,d)
 for n in [f'{g}-pre-pilot-gate.json',f'{g}-projected-krylov-m50-80-h0.1-t0.05.json',f'{g}-projected-krylov-m50-80-h0.1-t2.0.json',f'{g}-krylov-pilot-independent-validation.json']:
  cp(v2/n,Path('receipts')/n)
 for t in ['0.05','2.0']:
  for kind in ['analysis','field-analysis']:cp(v2/f'{g}-projected-krylov-t{t}-{kind}.json',Path('receipts')/f'{g}-projected-krylov-t{t}-{kind}.json')
 for suffix in ['validation.log','cache0.0001-validation.stderr','krylov-pilot.log','pilot-independent.log','pilot-history-analysis.log','pilot-fields-analysis.log','krylov-t2.log','t2-history-analysis.log','t2-fields-analysis.log']:
  cp(v2/f'{g}-{suffix}',Path('logs')/f'{g}-{suffix}')
 gate=read(v2/f'{g}-pre-pilot-gate.json');short=read(v2/f'{g}-krylov-pilot-independent-validation.json');long=read(v2/f'{g}-projected-krylov-m50-80-h0.1-t2.0.json');analysis=read(v2/f'{g}-projected-krylov-t2.0-analysis.json');fields=read(v2/f'{g}-projected-krylov-t2.0-field-analysis.json')
 assert gate['all_consistency_checks_pass']
 shorterr=max(v for row in short['checks'] for v in row['relative_state_l2']);assert shorterr<1e-10
 pair=max(s['coarse_fine_max_relative_difference'] for col in long['columns'] for s in col['steps']);assert pair<=1e-10
 constraint_eps=max(x['eps1e-5_vs3e-5_relative_constraints_l2'] for h in analysis['histories'] for x in h['constraint_amplitude_convergence']);assert constraint_eps<1e-7
 assert analysis['diagnostic_server_exit']==fields['server_exit']==0
 summary['gauges'][g]={'stage_consistency_checks_pass':True,'short_canonical_max_relative_state_error':shorterr,'short_canonical_total_seconds':short['total_seconds'],'exploratory_long_seconds':long['seconds'],'exploratory_long_matvecs':long['matvecs'],'max_long_coarse_fine_relative_difference':pair,'long_error_scope':'empirical local Arnoldi truncation difference; no independent long canonical forward-error validation','max_native_constraint_amplitude_check_relative_error':constraint_eps,'long_vs_frozen_C0':read(v2/'exploratory-t2.0-vs-frozen-C0.json')['gauges'][g]['rows']}
 paths=[work/f'server-{g}',v2/f'server-{g}',work/f'{g}-validation-vectors.npz',v2/f'{g}-projected-J20.npz',v2/f'{g}-cache0.0001-metadata.json']
 paths += [v2/f'{g}-cache0.0001-{s}' for s in ['J22.npz','indptr.bin','indices.bin','data.bin','lift.bin','restrict.bin']]
 paths += [v2/f'{g}-projected-krylov-m50-80-h0.1-t{t}.npz' for t in ['0.05','2.0']]
 paths += [v2/f'{g}-projected-krylov-t{t}-native-diagnostics.npz' for t in ['0.05','2.0']]
 paths += [v2/f'{g}-krylov-pilot-canonical-vectors.npz']
 large[g]=[artifact(p) for p in paths]
large['other']=[artifact(v2/'diagnostic-fields')]
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
gateindex=root/'build-layer-research/continuum/covariant-constraint-propagation/immutable-C1-blend-constraint-20261009/index.json'
identity={'launch_HEAD':prov['launch_HEAD'],'runtime_implementation':prov['runtime_implementation'],'freeze_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'dependency_identity_checks':checks,'explicit_overlay_hashes':prov['explicit_overlay_hashes'],'C1_math_sha256':sha(v2/'c1_additions.hpp'),'bulk_helper_sha256':sha(v2/'bulk_c1_additions.hpp'),'frozen_local_gate_index':{'path':str(gateindex),'sha256':sha(gateindex)},'unchanged_original_oracle_hashes':oracles,'unchanged_prior_global_manifest_sha256':sha(work.parent/'full-tensor-global-final/manifest.json'),'no_production_edits':subprocess.check_output(['git','diff','--name-only','--','src','CMakeLists.txt'],cwd=root,text=True)==''}
assert identity['no_production_edits'];assert identity['C1_math_sha256']=='908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7';save('source-identity-verification.json',identity);save('summary.json',summary);save('large-artifacts-metadata-only.json',large)
report="# Prescribed bulk C1 Cartesian tangent: small mixed exploratory screen\n\nThe exact shared helper multiplies every mechanical C1 term and the derived covector Lambda repair by C(r)=1−W_gauge. It is the prescribed bulk blend, not the full covariant system. The scientific coefficient-gradient/principal/Einstein/outer-pole gate passed before compiling this separate actual Cartesian full22/native20 experiment. The short independent propagation gate passes; the t2 screen gives only small mixed changes and does not justify a long native/Taylor run. No production option or stabilization is adopted.\n\nParameters match the frozen C0 and full-C1 comparisons: N16,span2.2,h=.1375,1640 active nodes,32800 free20/36080 raw22,S1,a.5,wide reference(.05,.95),gauge(.45,.85),kappa10,symmetric quadratic ghosts,nativeKO.1. Theta remains unrestricted on the strict interior grid. Source helperBulkC1Additions is byte-pinned4c7b6637fc9c38339134d5d5824e589986110edc9efa55489ebe0fc941bafbe2, common C1 math908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7. All27 frozen gate-index files were checked against index1bb5f697691404c7a8ed19c20fc77aa79c5aa3fa31a81248d10c7e958d3ac2f0 before compilation.\n\nThe copied CartesianPatch calls the live shared helper after existing C0 analytic-reference roundoff subtraction. No blended C1 reference subtraction is added. Cached Point makes the identical call with p.radius and the same layer_gauge. No lapse/shift, stencil, ghost, projector, mask or KO change is made. Mathematical dC terms belong to the derived constraint propagation gate; the actual global operator multiplies live pointwise RHS additions by C and its native spatial constraint differentiation samples that variation. No value-only interpolation of subsidiary matrices is used here.\n\n## Native implementation and exact support\n\nPrepare change0, referenceRHS6.08357e-12 and H/M/Z3.23947e-14/1.33057e-15/6.64311e-16 match C0. All63576 donor references remain strict interior/nonrecursive, constant-weight error1.33e-15. Raw CSR/cached native stencil agreement is1.11e-15. Six-amplitude actual native22 RHS derivatives agree to worst best7.25e-10; actual final-only SSPRK3 derivative versus P_refR3(dtJ22)Lift agrees to1.93e-10 across tested seeds and dt/2,dt,2dt. Nominaldt=.03minOmega=8.0859375e-5. Twelve retrospective implementation checks pass; they are not physical acceptance criteria.\n\nThe actual projected sparse matrix provides a direct support check: all20 rows at each of672 nodes r>=.85 are numerically EXACTLY equal to frozen C0. Chi/g/alpha/beta rows are exactly equal everywhere. Initial pure-gauge pulse J20 action is exactly equal to C0. The maximum changed coefficient is7.7524414 in inner geometry/momentum rows. These are matrix support identities, not a global boundary energy or stability theorem. In particular, the original instantaneous native gauge constraint source is not removed by this lower-order blend.\n\n## Verified short and exploratory long propagation\n\nThe propagated generator is J20=P_ref J22Lift, continuously projected semidiscrete evolution. It is distinct from actual native finite final-only RK3. Direct Taylor expm_multiply and independent two-pass Arnoldi m50/80 agree at t=.025,.05 within2.86e-14 for both seeds and both gauges. Taylor short controls cost72.59/70.85s. At t=.05 smooth H/M/Z ratios to C0 are.997807/1.002004/1.000494 production and.997810/1.002099/1.000663 spatialnorm; shell differences are<=.33%. The chosen component/derivative norm is likewise nearly unchanged.\n\nLong t2 Arnoldi costs77.66/77.63s, with local coarse/fine differences<=1e-10. Independent long canonical comparison remains pending by design: the small mixed screen does not justify that cost. This local empirical truncation test is not a rigorous nonnormal forward-error bound. The following results are exploratory, single-grid, finite-window evidence, not exact native t2 histories or a theorem of all-time failure/success. No eigenvalue/Ritz search is used.\n\n| Gauge/seed at t2 | Blend H/M/Z | Blend/C0 H/M/Z | Blend cfgH1+momL2 amplification(C0) |\n| --- | --- | --- | --- |\n| Production gauge |1.691591/1.218061/.264725|.98322/1.00027/.97123|46.7621(47.0983)|\n| Spatialnorm gauge |2.209822/1.268840/.352081|1.00783/1.03969/.95232|35.9798(35.7290)|\n| Production shell |.00345536/.00385030/.00097727|.99308/.99395/1.00843|.075721(.074887)|\n| Spatialnorm shell |.00272555/.00261090/.00116520|1.08830/1.14278/1.16982|.018220(.016994)|\n\nSeed arrays and normalization exactly match prior frozen controls: native width.5 angular gauge pulse and controlled RNG690 radial-shell random vector, each normalized to unit Euclidean free20 L2. Generation sources/hashes are retained. H is physical Hamiltonian; M/Z use native Penrose inverse contractions and unweighted active-cell RMS. The field units diagnostic is h^3sqrtgamma quadrature of configuration{chi,g,alpha,beta} H1 plus momenta{P,A,Lambda,Theta} L2 at S1, counting stored upper tensor components once. It is not invariant tensor energy, symmetrizer or a proven bound. Both shell sampled maxima in that norm are1 initially. No field positivity/SPD claim is made for finite states from a linear tangent vector.\n\nSmooth final squared outer r>=.9 H/M/Z fractions are.06917/.39923/.60230 production and.03471/.48848/.72904 spatialnorm. PeakH remains in the bulk (r=.22802/.35724); peakZ near r=.99865. These are localizations of a measured error, not its boundary-causation proof. Native signed constraint amplitude sweeps and all finite-output checks are preserved. Diagnostic shell-pole columns remain C0 in this tangent-only copied header and are neither used nor claimed; the comparator uses actual EvolvedConstraints H/M/Z. Root's independent native build adds separate complete pole diagnostics without changing evolution.\n\n## Provenance and limits\n\nBuilds launch at aef47b0a with production headers byte-identical27c19d20 plus the explicit blend overlay/math. Freeze HEAD is recorded separately after parent documentation commit80b83eab. Exact commands, AppleClang21 arm64 flags, four native/tangent compiler dependency inventories and Kokkos static archives are rehashed unchanged. All build outputs remain in this fresh blend tree. The prior C1 command-path correction stays in its immutable archive; both original C0 scratch oracle hashes remain unchanged here. The initial SOURCE_PREPARATION_HOLD.json is retained as historical source-only status, superseded by gate-bound-sources.json and compiled build-provenance.json.\n\nSources/overlay diffs, gate-index identity, all native stage/sweep observations, short canonical checks, long empirical receipts, exact sparse support audit and native constraint/field diagnostics are copied. Large CSR/state/seed/diagnostic arrays and executables are metadata-only with path/shape/size/SHA256. Existing C0/global/fullC1/stage/wave frozen archives are unchanged. This screen accepts no nonlinear scri closure, complete covariant formulation, uniform energy estimate, finite angular pulse or black-hole transition. Any damping-profile follow-up stays in a separate source tree and requires its own coefficient-gradient/pole/principal gates.\n"
(here/'REPORT.md').write_text(report)
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,dict):return all(finite(y) for y in x.values())
 if isinstance(x,list):return all(finite(y) for y in x)
 return True
assert all(finite(read(p)) for p in here.rglob('*.json'))
catalog={str(p.relative_to(here)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(here.rglob('*')) if p.is_file() and p.name!='manifest.json'}
save('manifest.json',{'scope':'actual full22/native short gate and exploratory t2 prescribed bulk C1 mixed screen; no long canonical/no adoption','files':catalog})
print(json.dumps({'files':len(catalog),'bytes':sum(x['bytes'] for x in catalog.values()),'REPORT_sha256':sha(here/'REPORT.md'),'summary_sha256':sha(here/'summary.json'),'manifest_sha256':sha(here/'manifest.json')},indent=2))
