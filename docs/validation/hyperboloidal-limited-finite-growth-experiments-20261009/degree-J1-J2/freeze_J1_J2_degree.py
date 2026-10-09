"""Immutable J1/J2 N12/N16 consistency control; no spectra or propagation."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
D=P/'immutable-J1-J2-finite-rb-degree-control-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def pin(p):return {'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}
assert not D.exists()
rows=[];review=[]
for J in (1,2):
 for N in (12,16):
  base=P/f'J{J}-N{N}-rb.98-segmentedQ64-a12x24-primary001'
  angular=P/f'J{J}-N{N}-rb.98-segmentedQ64-a16x32-angular001'
  coarse=P/f'J{J}-N{N}-rb.98-segmentedQ32-a12x24-pair001'
  family=P/f'J{J}-N{N}-forcing-family-replay001'
  decorated=P/f'J{J}-N{N}-sector-readback00{1 if J==1 else 2}'
  r=read(base/'report.json');assert sha(base/'operator.npz')==r['operator_sha256'] and r['passed_single_quadrature_algebra']
  assert r['observed_incoming_rank']==r['expected_incoming_rank']==(8 if J==1 else 10)
  assert read(angular/'report.json')['passed_single_quadrature_algebra']
  f=read(family/'report.json');assert f['passed'] and f['family_count']==4*(16 if J==1 else 20)+1
  mass=read(base/'auxiliary-readback/report.json');assert mass['passed']
  q=read(P/f'J{J}-N{N}-quadrature-comparison.json');a=read(P/f'J{J}-N{N}-angular-comparison.json');assert q['passed'] and a['passed']
  c=read(coarse/'report.json');assert not c['passed_single_quadrature_algebra']
  for kind,path in [('primary',base),('angular',angular)]:
   receipt=read(path/'run-receipt.json');assert receipt['exit_code']==0 and receipt['source_unchanged'] and receipt['executable_unchanged']
  assert read(coarse/'run-receipt.json')['exit_code']!=0
  for kind in ('forcing','mass','decoration','radial-comparison','angular-comparison'):
   suffix='002' if J==2 and kind=='decoration' else '001'
   rr=read(P/f'J{J}-N{N}-{kind}-command{suffix}/receipt.json');assert rr['exit_code']==0 and rr['script_unchanged']
  rows.append({'J':J,'N':N,'dofs':r['dofs'],'E_modal_condition':r['energy_modal_condition'],
   'weak_strong':r['checks']['weak_strong'],'volume_identity':r['checks']['bulk_identity'],
   'forcing_fields':f['family_count'],'forcing_family_coefficient_max_scaled':f['max_coefficient_error_scaled'],
   'forcing_family_energy_max_scaled':f['max_energy_error_scaled'],
   'mass_max_scaled':max(z[k]['scaled'] for z in mass['mass'] for k in ('modal_I','nodal_exact_congruence','back_congruence')),
   'radial_pair_max_scaled':max(v['scaled'] for v in q['rows'].values()),
   'angular_pair_max_scaled':max(v['scaled'] for v in a['rows'].values()),
   'coarse32_forcing_failed_scaled':c['checks']['manufactured_forced_rhs']['scaled'],
   'incoming_rank':r['observed_incoming_rank'],'owner_complete_consistency_controls_passed':True})
  paths=[base/'operator.npz',base/'report.json',base/'auxiliary-readback/report.json',
   family/'report.json',family/'forcing-family.npz',
   P/f'J{J}-N{N}-quadrature-comparison.json',P/f'J{J}-N{N}-angular-comparison.json',
   P/f'polynomial-mass-N{N}/exact.json',decorated/'operator.npz',decorated/'matrix-metadata.json',
   coarse/'report.json',angular/'report.json']
  review.append({'J':J,'N':N,'inputs':[pin(p) for p in paths]})
recipe=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-held-20261009'
upstream=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009/immutable-total-J-finite-rb-N8-matrix-control-20261009/index.json'
j0=P/'immutable-J0-finite-rb-degree-control-20261009/index.json'
root_j0=ROOT/'build-layer-research/continuum/finite-rb-J0-degree-independent-root-review-20261009/receipt.json'
assert sha(upstream)=='d083ac45cd9f471898837fe7a1a54d965ad40c954db2f8d2c65ea183d72338e0'
assert sha(j0)=='056942564fdbd6f36510a63cddde38c0b2a0b6b811ad02942a665883e125fdb7'
assert sha(root_j0)=='2d20660d548c6bd78be5a405ae8907b40dc167157e1e547fea33d4cf68a7686b'
write(P/'J1-J2-degree-review-inputs.json',{'J':[1,2],'rb':.98,'degree_controls':review,
 'recipe_pins':[pin(recipe/'RECIPE.md'),pin(recipe/'FINAL-ADDENDUM.md')],
 'source_preparation':pin(P/'preparation-receipt.json'),'plan':pin(P/'degree-control-plan.json'),
 'upstream_N8_compiled_source_and_gate_index':pin(upstream),'separate_J0_index':pin(j0),
 'additive_J0_only_independent_review':pin(root_j0),
 'owner_controls_passed':True,'independent_saved_readback':'pending root review of J1/J2; J0 review is separate',
 'no_eigen_or_propagation_by_owner':True})
write(P/'J1-J2-degree-summary.json',{'rows':rows,'source_sha256':sha(P/'assemble_degree.py'),
 'point_executable_sha256':sha(P/'radial-bridge-release'),
 'scope':'unchanged actual C0 physical-P spatial-norm reference/source; finite-boundary energy-Galerkin consistency only',
 'general_nongauge_continuum_constraint_comparator':'unresolved; separate analytic projected point diagnostic only',
 'mechanical_failures':'Two decoration commands used cwd-relative incomplete input paths; preserved errors and empty created directories, then corrected complete paths with same source.',
 'no_generator_eigs_or_propagation_by_owner':True})
lines=['J1/J2 finite-ball degree controls at rb=.98 pass the owner source/mass/weak-strong/volume/forcing/integration gates for N12 and N16. Each channel keeps an independent regular polynomial envelope over the whole ball. The actual point executable2293e9be, C0 physical-P/spatial-norm equations, reference, coefficients, complete lift and SAT are unchanged. Degree-independent saved point queries and reference rows are reused; no kernel query, compile, eigenanalysis or propagation is performed here.','',
 '| J | N | DOFs | modal E condition | weak/strong scaled | volume identity scaled | forcing coefficient max scaled | radial32/64 change | angular change |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
 lines.append('| {J} | {N} | {dofs} | {E_modal_condition:.6g} | {w:.5g} | {g:.5g} | {forcing_family_coefficient_max_scaled:.5g} | {radial_pair_max_scaled:.5g} | {angular_pair_max_scaled:.5g} |'.format(**row,w=row['weak_strong']['scaled'],g=row['volume_identity']['scaled']))
lines += ['', 'The final rule uses64 nodes per fixed radial panel. Coarse32 rules fail the original2e-9 mixed-forcing gate and remain preserved; their saved-array changes relative to64 satisfy the unchanged2e-8 integration threshold. All65 J1 and81 J2 prescribed forcing fields (four polynomial envelopes per channel plus one mixed field) pass the direct cached pointwise forcing replay at each degree. Exact720/1280 rational mass entries, independent overintegrated dense/modal/nodal mass congruences, cached raw22 normal gates and incoming rank8/10 pass. The exact mass oracle uses coefficient-product convolution of the same polynomial integral; J0 received a separate independent Fraction readback. All absolute errors and conditions remain in reports.',
 '', 'Two mechanical decoration attempts failed before reading a matrix because incomplete relative paths were supplied. Their stderr/commands and empty created output folders are retained; complete-path retries use the identical decorator and preserve all scientific input arrays bitwise. No equation, source, SAT or tolerance was tuned. Independent J1/J2 saved-matrix readback is pending root review at this freeze; the separately linked J0 review does not certify these new matrices.',
 '', 'The N8 compiled-source/dependency, all-m angular and independent core controls remain pinned through d083ac45; no new compiler or independent physical source oracle is claimed. Large arrays and executable bytes remain locally retained and appear here as metadata only. Both ordinary-FD projected constraint protocols remain failed elsewhere; separate analytic projected-point readback does not resolve the general nongauge continuum comparator.',
 '', 'This is a finite-boundary energy-Galerkin consistency control. It is not CPBC, an exact-scri closure, an all-direction symmetrizer or a uniform stability estimate. No generator spectrum, time evolution, physical pulse/BH acceptance or unique Cartesian-boundary attribution follows. Original N8 and J0 degree checkpoints stay unchanged.']
(P/'REPORT-J1-J2-degree.md').write_text('\n'.join(lines)+'\n')
D.mkdir();files=[];large=[];aliases=[];finite_json=0
common={'prepare.py','preparation-receipt.json','degree-control-plan.json','assemble_degree.py','run_degree.py','run_degree_angular.py',
 'run_auxiliary.py','replay_forcing_degree.py','forcing-readback-preparation.json','exact_mass_degree.py','check_mass_degree.py',
 'mass-readback-preparation.json','decorate_degree_operator.py','compare_degree_matrices.py','energy_coefficients.py',
 'canceled_basis_complex.py','freeze_J1_J2_degree.py','J1-J2-degree-review-inputs.json','J1-J2-degree-summary.json','REPORT-J1-J2-degree.md'}
for origin in sorted(P.rglob('*')):
 rel=origin.relative_to(P)
 if any(part=='__pycache__' or part.startswith('immutable-') for part in rel.parts):continue
 if len(rel.parts)==1 and rel.name not in common and not rel.name.startswith('radial-bridge-'):continue
 if len(rel.parts)>1 and not(rel.parts[0].startswith(('J1-','J2-','polynomial-mass-')) or rel.parts[0]=='inputs'):continue
 if origin.is_symlink():
  aliases.append({'path':str(rel),'resolved':str(origin.resolve()),'upstream_index_sha256':sha(upstream)});continue
 if not origin.is_file():continue
 if origin.suffix in ('.npz','.bin') or origin.name.startswith('radial-bridge-') or origin.stat().st_size>1024*1024:
  entry={'path':str(rel),'origin':str(origin.resolve()),'bytes':origin.stat().st_size,'sha256':sha(origin)}
  if origin.suffix=='.npz':
   with np.load(origin,allow_pickle=False) as z:
    assert all(np.isfinite(z[k]).all() for k in z.files if np.issubdtype(z[k].dtype,np.number))
    entry['arrays']={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
   entry['numeric_arrays_finite']=True
  large.append(entry);continue
 target=D/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,target);assert sha(target)==sha(origin)
 if target.suffix=='.json':read(target);finite_json+=1
 files.append({'path':str(rel),'origin':str(origin.resolve()),'sha256':sha(target),'bytes':target.stat().st_size})
for name,origin in [('upstream-N8-index.json',upstream),('separate-J0-index.json',j0),('additive-J0-only-independent-review.json',root_j0),
 ('held-RECIPE.md',recipe/'RECIPE.md'),('held-FINAL-ADDENDUM.md',recipe/'FINAL-ADDENDUM.md')]:
 shutil.copyfile(origin,D/name);files.append({'path':name,'origin':str(origin),'sha256':sha(origin),'bytes':origin.stat().st_size})
 if name.endswith('.json'):read(D/name);finite_json+=1
record={'kind':'Immutable owner J1/J2 finite-ball N12/N16 consistency controls; independent review pending',
 'freeze_HEAD':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
 'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'production_source_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2',
 'small_files':len(files),'small_bytes':sum(x['bytes'] for x in files),'finite_JSON':finite_json,
 'files':files,'external_large_files':large,'symlink_aliases_to_pinned_N8_inputs':aliases,
 'owner_consistency_controls_passed':True,'independent_review':'pending root J1/J2 readback; additive J0 review is scoped separately',
 'no_generator_eigs_or_propagation_by_owner':True}
for item in files:assert sha(D/item['path'])==item['sha256']
write(D/'index.json',record)
print(json.dumps({'path':str(D),'index_sha256':sha(D/'index.json'),'files':len(files),'bytes':record['small_bytes'],'large_metadata':len(large),'finite_JSON':finite_json},indent=2))
