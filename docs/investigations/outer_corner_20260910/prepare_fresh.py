from pathlib import Path
import sys,json
sys.path.insert(0,'/pscratch/sd/h/hzhu/lapse-bisection-t200-20260909')
from controller import setparam,sha
root=Path(__file__).resolve().parent
base=Path('/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/history-extrema-20260901/bisection_N256_20260907')
case=root/'fresh_super_cpbc_linear'
case.mkdir(exist_ok=False)
(case/'amplitude.txt').write_text('-0.05\n')
inp=(base/'template.athinput').read_text()
for sec,key,value in [('job','basename','boundary200'),('time','tlim',200),('fastflow','num_horizons',0),('problem','stop_on_horizon','false'),('problem','stop_on_dispersion','false'),('mesh_refinement','amr_history_file',case/'amr_history.jsonl'),('problem','brill_global_coefficients_file','initial.coefficients'),('problem','constraint_summary_file','initial-constraints.dat'),('z4c','boundary_rhs','full_constraint_bjorhus'),('z4c','extrap_order',2),('z4c','vc_single_rank_device_sync','true'),('z4c','history_constraint_radius',1000),('output5','dt',5)]:
 inp=setparam(inp,sec,key,value)
(case/'input.athinput').write_text(inp)
runner=(base/'run_cycle.sh').read_text().replace('campaign=$(dirname -- "$case_dir")','campaign='+str(base)).replace('03:30:00','01:20:00')
(case/'run.sh').write_text(runner)
(case/'provenance.json').write_text(json.dumps(dict(baseline=str(base),exe_sha256=sha(base/'athena.history_extrema'),input_sha256=sha(case/'input.athinput'),purpose='Fresh A=-0.05 N256 boundary qualification, not a bisection midpoint',changes=['full_constraint_bjorhus','extrap_order=2','device sync','full-domain constraint diagnostics','curvature output dt5']),indent=2))
print(case)
