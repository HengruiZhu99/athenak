#!/usr/bin/env python3
"""Compare three native CK05 clock pilots at identical time and coordinates."""
from pathlib import Path
import argparse, ast, csv, hashlib, json, math
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('work', type=Path)
p.add_argument('output', type=Path)
p.add_argument('cases', nargs=3)
a = p.parse_args()
# Reuse the reviewed snapshot reader, measures, and tensor norms without running
# its command-line driver. This is analysis only; no evolved variables are added.
source = Path(__file__).with_name('review_goal1_tangent_pilots.py')
tree = ast.parse(source.read_text())
ns = dict(np=np, struct=__import__('struct'), a=a, json=json)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), str(source), 'exec'), ns)
config = [0,1,2,3,4,5,6,18,19,20,21]
datasets = [ns['load'](label) for label in a.cases]
assert all(d[1:3] == datasets[0][1:3] for d in datasets)
caps=[]
for label in a.cases:
    cmd=json.loads((a.work/label/'receipt.json').read_text())['command']
    caps.append(float(next(s.split('=')[1] for s in cmd if s.startswith('cubed_sphere/conformal_fixed_dt='))))
assert caps[0] == 2*caps[1] == 4*caps[2]
x = datasets[0][3]
assert all(np.array_equal(d[3][:,:4], x[:,:4]) for d in datasets)
w = ns['reference_weights'](x)['fixed_reference_Penrose']
mask = np.ones(len(x), bool)
scales = np.full(22, .0005/.2); scales[config] = .0005
fields = [d[3][:,14:].reshape(-1,22,13)/scales[None,:,None] for d in datasets]
def measures(z):
    return {k: ns['norm'](v,w,mask) for k,v in [('fields',z[:,:,0]),('first',.2*z[:,:,1:4]),('configuration_second',.04*z[:,config,4:])]}
diff = [measures(fields[0]-fields[1]), measures(fields[1]-fields[2])]
review = json.loads((a.output/'native-oracle-pilot-review.json').read_text())
oracle = review['cases'][a.cases[-1]]['native_exact_errors']['fixed_reference_Penrose']['all']
temporal = measures(fields[0]-fields[2])
result = {'scope':'Same-grid native22 total-RHS tangent clock controls; no spatial convergence, TT order, or exact-scri acceptance claim.', 'time':datasets[0][1], 'cases':a.cases,'timestep_caps':caps,'common_coordinates_exactly_equal':True,'differences_full_half_and_half_quarter':diff, 'empirical_order':{},'full_quarter_over_quarter_oracle':{},'stage_monitors':{}}
for k in diff[0]:
    result['empirical_order'][k] = {n:math.log2(diff[0][k][n]/diff[1][k][n]) for n in ['rms','linf']}
    result['full_quarter_over_quarter_oracle'][k] = {n:temporal[k][n]/oracle[k][n] for n in ['rms','linf']}
labels=['stationary4']
for label in a.cases:
    while label:
        assert label not in labels
        labels.append(label)
        label=json.loads((a.work/label/'receipt.json').read_text())['restart']
for label in labels:
    with (a.work/label/'cubed_sphere_native_tangent_rhs.csv').open() as f:
        rows=list(csv.DictReader(f))
    keys=['metric_rate_before','A_rate_before','metric_rate_after','A_rate_after','metric_correction','A_correction','raw_all_rhs']
    result['stage_monitors'][label] = {k:max(abs(float(r[k])) for r in rows) for k in keys}
result['order_diagnostic_gate_pass'] = all(v['rms']>=2.5 for v in result['empirical_order'].values())
result['temporal_contamination_gate_pass'] = all(v[n]<.1 for v in result['full_quarter_over_quarter_oracle'].values() for n in ['rms','linf'])
result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(a.output/'timestep-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
