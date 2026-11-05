import hashlib,json,re
from pathlib import Path
root=Path.cwd();base=Path('docs/spherical-optimization');e=base/'evidence';plan=json.loads((e/'serial-cost-v28/plan.json').read_text());index=json.loads((e/'penalty-controls-source-v28/run-index.json').read_text());comp=json.loads((e/'serial-cost-final-v28/comparison.json').read_text())
assert index['total']==52 and index['failed']==3 and index['completed']==49
assert all(x['floor_calls']==0 for x in index['runs'])
assert len(comp['summary'])==9 and all(s['repeated_states_bitwise_equal'] and s['completed']==3 for s in comp['summary'])
for name,sha in plan['execution_sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
assert hashlib.sha256(Path('build-spherical-v26/libcurrent_spherical.dylib').read_bytes()).hexdigest()==plan['binary_sha256']
meta=json.loads((e/'penalty-controls-source-v28/manifest.json').read_text())
for name,sha in meta['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
checked=[]
for name in ['README.md','PENALTY-TRADEOFF.md','HANDOFF-AUDIT.md','FINAL-COST.md','COST-PROTOCOL.md']:
 p=base/name
 for dest in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' not in dest and not dest.startswith('#'):assert (p.parent/dest.split('#')[0]).exists(),(name,dest)
 checked.append(name)
log=e/'outer-lapse-build-v26/spherical-v26-ctest-verified.log';assert re.search(r'100% tests passed(?:, 0 tests failed)? out of 10\b',log.read_text())
out=dict(phase_runs=52,completed=49,preserved_failures=3,all_phase_runs_no_floors=True,benchmark_runs=27,benchmark_configs=9,all_repeat_states_bitwise=True,execution_sources_match_frozen_plan=True,phase_sources_match_archive_manifest=True,binary_sha256=plan['binary_sha256'],compiled_tests=dict(path=str(log),sha256=hashlib.sha256(log.read_bytes()).hexdigest(),passed=10,qualification='Existing v26 verification; no compiled source change in v27/v28'),python_compile_passed=['run_cost_plan.py','analyze_cost_plan.py','report_cost_plan.py','analyze_final_diagnostics.py','analyze_convergence.py','analyze_response_controls.py','test_shift_normal.py'],checked_report_links=checked,visually_checked=['serial-cost-final-v28/accuracy-cost.png','final-diagnostics-revised-v28/final-diagnostics.png','penalty05-gauge-controls-v28/response-controls.png','penalty05-temporal-v28/convergence.png','penalty05-spatial-v28/convergence.png','penalty05-n128-timestep-analysis-v28/timestep.png','penalty05-refinement-v28/boundary-refinement.png','penalty05-rk-margin-v28/rk-margin.png'],verification_note='Initial packaging assertion expected the alternate CTest summary wording with zero-failure clause; the preserved log says 100% tests passed out of 10. Parser corrected without rerunning or changing the test record.')
(e/'final-verification-v28.json').write_text(json.dumps(out,indent=2)+'\n')
p=base/'STATE.json';d=json.loads(p.read_text());d['latest']['phase_runs']=52;d['latest']['phase_completed_runs']=49;d['latest']['phase_preserved_failures']=3;d['latest']['verification']='evidence/final-verification-v28.json';p.write_text(json.dumps(d,indent=2)+'\n');print('Verified 52 phase runs, 27 benchmark runs, source/binary identities, report links and preserved 10-test record.')
