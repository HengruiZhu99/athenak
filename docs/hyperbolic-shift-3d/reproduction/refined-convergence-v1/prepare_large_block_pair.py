"""Package the requested larger-block comparison; does not submit or evolve."""
from pathlib import Path
import ast, hashlib, json, shutil
D=Path(__file__).resolve().parent
R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
E=D/'large-block-control';E.mkdir(exist_ok=True)
O=R/'output/refined-convergence/large-block-topology'
v=json.loads((O/'block24.json').read_text())
assert abs(v['supports'][0]['finest_spacing']-.8/(4*24*4))<2e-18
assert v['blocks']==254 and v['core_leaves']==176
assert json.loads((D/'smr-tube-checks/topology-mpi2.json').read_text())['leaves']==400
for name in ['block24.json','build.json','native.log']:(E/name).write_bytes((O/name).read_bytes())
report=dict(status='native_topology_passed_not_evolved',
 old=dict(n=12,smr_levels=3,core_leaves=400,blocks=832,active_cells=1513728),
 candidate={k:v[k] for k in ['n','core_leaves','blocks','active_cells','nominal_cells','core_level_counts','supports']},
 finest_spacing_equal_within_roundoff=True,active_cell_ratio=v['active_cells']/1513728,
 block_ratio=v['blocks']/832,
 scope='Native AthenaK MPI2 topology with actual N24 allocation, no coupled evolution. Finest spacing identical; finest volume/coarse-core/angular/boundary resolution differ. No pure repacking, runtime or accuracy claim.')
(E/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
P=R/'output/refined-convergence/large-block-pair-v1';P.mkdir(exist_ok=False)
namespace='launcher-boosted-transition-v1';original=D/'gpu-partition-build'
s=(original/'partition_pair_worker.py').read_text().replace('launcher-v7',namespace).replace("'static_ownership_pair'","'large_block_pair'")
s=s.replace("('n', 'dt', 'identity', 'config', 'restart', 'start_time',\n                    'start_steps', 'target_time', 'diagnostic_interval', 'binary')",
 "('dt', 'restart', 'start_time', 'start_steps', 'target_time', 'diagnostic_interval', 'binary')")
s=s.replace("env = dict(os.environ, HYP3D_ANALYSIS_PARTITION=str(root / case['partition']))","env = dict(os.environ)")
s=s.replace("str(root / case['restart'])","'-'").replace("'900'","'1150'")
s=s.replace("partition_sha256=digest(root / case['partition'])","configuration_sha256=digest(root / case['config'])")
s=s.replace('> 2900 or time.monotonic() - begin > 1050','> 3050 or time.monotonic() - begin > 1400')
s=s.replace("    state['checkpoint_byte_identical'] = (\n        state['cases'][0]['verification']['sha256'] ==\n        state['cases'][1]['verification']['sha256'])\n",'')
s=s.replace("'Bounded ownership experiment only. Compare stepping '\n        'timers separately from setup/verification/I/O. Adoption requires physical '\n        'checkpoint parity and measured improvement; no strong-scaling claim.'",
 "'Bounded fresh-t0 large-block comparison at matched dt and physical time. Distinct meshes and identities; compare native constraints and common-coordinate fields, not checkpoint equality. Compare stepping and total cost separately. Not long stability or convergence evidence.'")
s=s[s.index('from pathlib import Path'):]
(P/'worker.py').write_text('"""Fresh mesh-block pair; frozen RF2 verifier, no automatic replay."""\n'+s)
s=(original/'collect_partition_pair.py').read_text().replace('launcher-v7',namespace).replace("'static_ownership_pair'","'large_block_pair'").replace('parity_passed=False','comparison_pending=True')
start=s.index("    result['parity_passed'] =");end=s.index('\nelse:',start)
s=s[:start]+"    result['qualification'] = 'Both bounded fresh cases admitted; distinct grids require field/norm comparison and measured performance analysis before adoption.'"+s[end:]
(P/'collect.py').write_text(s)
(P/'configuration_identity.cpp').write_text((D/'configuration_identity.cpp').read_text().replace('for(int n:{12,14,16})','for(int n:{24})'))
(P/'resolve_configuration_identity.py').write_text((D/'resolve_configuration_identity.py').read_text().replace('==[12,14,16]','==[24]'))
cfg=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_block24_smr2_merged13_reference.cfg'
shutil.copy2(cfg,P/'candidate.cfg')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(P/'qualified-local-identity.json').write_text(json.dumps(dict(identity=v['identity']+';amr_policy=smr',config_sha256=sha(cfg),
 source_commit='d98e8d9e7f6131a4af55afe96ef7589a73d07b9e',
 qualification='Native MPI2 topology and parser only; first coupled-field physics comparison occurs in bounded fresh pair. Not prior GPU evolution.'),indent=2)+'\n')
shutil.copy2(D/'launcher-v7/identity-calibration.json',P/'known-aurora-calibration.json')
shutil.copy2(D/'plan_large_block_pair.py',P/'plan.py')
for p in P.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
for name in ['worker.py','collect.py','configuration_identity.cpp','resolve_configuration_identity.py','plan.py']:(E/name).write_bytes((P/name).read_bytes())
records={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in P.iterdir() if p.is_file()}
(P/'package-records.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(report,indent=2))
