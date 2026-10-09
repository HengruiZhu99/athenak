"""Freeze owner J0 N12/N16 consistency controls; no spectra/propagation."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
D=P/'immutable-J0-finite-rb-degree-control-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(path.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def pin(path):return {'path':str(path.resolve()),'sha256':sha(path),'bytes':path.stat().st_size}

assert not D.exists()
rows=[];review=[]
for N in (12,16):
    base=P/f'J0-N{N}-rb.98-segmentedQ64-a12x24-primary001'
    r=read(base/'report.json');assert sha(base/'operator.npz')==r['operator_sha256'] and r['passed_single_quadrature_algebra']
    f=read(P/f'J0-N{N}-forcing-family-replay001/report.json');assert f['passed'] and f['family_count']==33
    mass=read(base/'auxiliary-readback/report.json');assert mass['passed']
    q=read(P/f'J0-N{N}-quadrature-comparison.json');ang=read(P/f'J0-N{N}-angular-comparison.json');assert q['passed'] and ang['passed']
    coarse=read(P/f'J0-N{N}-rb.98-segmentedQ32-a12x24-paired001/report.json');assert not coarse['passed_single_quadrature_algebra']
    rows.append({'N':N,'dofs':r['dofs'],'E_modal_condition':r['energy_modal_condition'],
                 'weak_strong':r['checks']['weak_strong'],'volume_identity':r['checks']['bulk_identity'],
                 'forcing_family_coefficient_max_scaled':f['max_coefficient_error_scaled'],
                 'forcing_family_energy_max_scaled':f['max_energy_error_scaled'],
                 'mass_max_scaled':max(z[k]['scaled'] for z in mass['mass'] for k in ('modal_I','nodal_exact_congruence','back_congruence')),
                 'radial_pair_max_scaled':max(v['scaled'] for v in q['rows'].values()),
                 'angular_pair_max_scaled':max(v['scaled'] for v in ang['rows'].values()),
                 'coarse32_forcing_failed_scaled':coarse['checks']['manufactured_forced_rhs']['scaled'],
                 'owner_complete_consistency_controls_passed':True})
    paths=[base/'operator.npz',base/'report.json',base/'auxiliary-readback/report.json',
           P/f'J0-N{N}-forcing-family-replay001/report.json',P/f'J0-N{N}-forcing-family-replay001/forcing-family.npz',
           P/f'J0-N{N}-quadrature-comparison.json',P/f'J0-N{N}-angular-comparison.json',
           P/f'polynomial-mass-N{N}/exact.json',P/f'J0-N{N}-sector-readback001/operator.npz',
           P/f'J0-N{N}-sector-readback001/matrix-metadata.json']
    review.append({'N':N,'inputs':[pin(path) for path in paths]})
recipe=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-held-20261009'
upstream=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009/immutable-total-J-finite-rb-N8-matrix-control-20261009/index.json'
assert sha(upstream)=='d083ac45cd9f471898837fe7a1a54d965ad40c954db2f8d2c65ea183d72338e0'
write(P/'J0-degree-review-inputs.json',{'J':0,'rb':.98,'degree_controls':review,
    'recipe_pins':[pin(recipe/'RECIPE.md'),pin(recipe/'FINAL-ADDENDUM.md')],
    'source_preparation':pin(P/'preparation-receipt.json'),'plan':pin(P/'degree-control-plan.json'),
    'upstream_N8_compiled_source_and_gate_index':pin(upstream),
    'owner_controls_passed':True,'independent_saved_readback':'pending root review; no independent numerical rerun claimed here',
    'no_eigen_or_propagation_by_owner':True})
write(P/'J0-degree-summary.json',{'rows':rows,'source_sha256':sha(P/'assemble_degree.py'),
    'point_executable_sha256':sha(P/'radial-bridge-release'),
    'scope':'same actual C0 physical-P spatial-norm reference/source; energy-Galerkin degree controls only; no stability or CPBC acceptance',
    'general_nongauge_continuum_constraint_comparator':'unresolved; separate analytic point work is not admitted by this checkpoint',
    'no_generator_eigs_or_propagation_by_owner':True})
text=['J0 finite-ball degree controls at rb=.98 pass the owner source/mass/weak-strong/volume/forcing/integration gates for N12 and N16. Each trial retains arbitrary independent regular envelopes over the whole ball. The actual point executable2293e9be and all C0 physical-P/spatial-norm equations, reference, coefficients and SAT are unchanged. The fresh assembler differs from the N8 BLAS source only by N12/16 admission, copied basename, and validated reuse of degree-independent reference rows. No kernel query, compile, eigenanalysis or propagation was performed for these controls.','',
      '| N | DOFs | modal E condition | weak/strong scaled | volume identity scaled | full forcing family max scaled | radial32/64 change | angular change |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    text.append('| {N} | {dofs} | {E_modal_condition:.6g} | {w:.5g} | {g:.5g} | {forcing_family_coefficient_max_scaled:.5g} | {radial_pair_max_scaled:.5g} | {angular_pair_max_scaled:.5g} |'.format(**row,w=row['weak_strong']['scaled'],g=row['volume_identity']['scaled']))
text+=['','The final rule uses64 nodes per fixed panel. Coarse32 rules fail the original2e-9 mixed-forcing gate and are preserved; their changes relative to64 nevertheless meet the unchanged2e-8 integration threshold. All33 prescribed per-channel cubic-or-lower fields pass the direct cached pointwise forcing replay at each degree. Exact720/1280 rational mass entries, overintegrated dense/modal/nodal mass congruences, all cached raw22 normals and measured incoming rank4 pass. All absolute errors and modal/raw conditions remain in source reports; the table shows scaled discrepancies.',
       '', 'Independent saved-matrix readback is pending root review. The N8 compiled source/dependency and all-m/core/normal gates remain pinned through d083ac45; this checkpoint does not silently claim a new compiler run or a new independent source oracle. All large arrays and executable bytes remain local, represented here by metadata only. Original direct ordinary-FD projected-constraint stops remain failed elsewhere; a separate analytic point diagnostic does not supply the missing general nongauge continuum comparator.',
       '', 'This is a finite-boundary energy-Galerkin consistency control, not CPBC, an exact-scri closure, an all-direction symmetrizer or a uniform stability estimate. No generator spectrum, time evolution, physical pulse/BH acceptance or unique Cartesian-boundary attribution follows. J1/J2 degree controls are separate future work.']
(P/'REPORT-J0-degree.md').write_text('\n'.join(text)+'\n')
D.mkdir();files=[];large=[];aliases=[];finite_json=0
for origin in sorted(P.rglob('*')):
    rel=origin.relative_to(P)
    if any(part in ('__pycache__',D.name) for part in rel.parts):continue
    if len(rel.parts)>1 and not (rel.parts[0].startswith('J0-') or rel.parts[0].startswith('polynomial-mass-') or rel.parts[0]=='inputs'):continue
    if origin.is_symlink():
        aliases.append({'path':str(rel),'resolved':str(origin.resolve()),'upstream_index_sha256':sha(upstream)})
        continue
    if not origin.is_file():continue
    if origin.suffix in ('.npz','.bin') or origin.name.startswith('radial-bridge-'):
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
for name,origin in [('upstream-N8-index.json',upstream),('held-RECIPE.md',recipe/'RECIPE.md'),('held-FINAL-ADDENDUM.md',recipe/'FINAL-ADDENDUM.md')]:
    shutil.copyfile(origin,D/name);files.append({'path':name,'origin':str(origin),'sha256':sha(origin),'bytes':origin.stat().st_size})
    if name.endswith('.json'):read(D/name);finite_json+=1
record={'kind':'Immutable owner J0 finite-ball N12/N16 consistency controls; independent review pending',
        'freeze_HEAD':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
        'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'production_source_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2',
        'small_files':len(files),'small_bytes':sum(x['bytes'] for x in files),'finite_JSON':finite_json,
        'files':files,'external_large_files':large,'symlink_aliases_to_pinned_N8_inputs':aliases,
        'owner_consistency_controls_passed':True,'independent_review':'pending root readback',
        'no_generator_eigs_or_propagation_by_owner':True}
for item in files:assert sha(D/item['path'])==item['sha256']
write(D/'index.json',record)
print(json.dumps({'path':str(D),'index_sha256':sha(D/'index.json'),'files':len(files),'bytes':record['small_bytes'],'large_metadata':len(large),'finite_JSON':finite_json},indent=2))
