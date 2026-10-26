#!/usr/bin/env python3
"""Independent dt/dt2/dt4 trajectory differences with frozen-source verification.

Raw differences and observed orders are evidence, not certified error bounds.
A temporal result never admits a candidate with failed spatial or boundary gates.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from telegraph_analysis_io import trajectory, physical_lapse_gradient


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerical_binding(folder, manifest):
    root = folder / 'frozen-source'
    binding = manifest['binding']
    for name, digest in binding['source_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError('Changed frozen source: ' + str(root / name))
    queue = ['run_telegraph']; seen = set()
    while queue:
        name = queue.pop(); path = root / (name + '.py')
        if name in seen or not path.exists():
            continue
        seen.add(name)
        if path.name not in binding['source_sha256']:
            raise ValueError('Unbound numerical import: ' + str(path))
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                queue.extend(x.name.split('.')[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                queue.append(node.module.split('.')[0])
    names = sorted([x + '.py' for x in seen] + ['fast_integrator.cpp', 'telegraph_integrator.cpp'])
    result = {name: binding['source_sha256'][name] for name in names}
    for argument, key in [('library', 'library_sha256'), ('fast_library', 'fast_sha256')]:
        path = root / Path(manifest['arguments'][argument]).name
        if sha(path) != binding[key]:
            raise ValueError('Changed executable: ' + str(path))
        result[key] = binding[key]
    return result


def load(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['status'] != 'finished':
        raise ValueError('Require a completed run: ' + str(folder))
    data, ancestry = trajectory(folder)
    binding = numerical_binding(folder, manifest)
    receipts = []
    for segment in ancestry:
        root = Path(segment['path']); old = json.loads((root / 'manifest.json').read_text())
        if old['status'] != 'finished' or numerical_binding(root, old) != binding:
            raise ValueError('Unverified or changed numerical ancestry')
        receipts.extend(dict(path=str(p), sha256=sha(p)) for p in sorted(root.glob('profile-*.npz')))
    args = manifest['arguments']; end = manifest['time']; cadence = args['report_dt']
    expected = {round(i * cadence, 9) for i in range(round(end / cadence) + 1)}
    expected.add(round(end, 9))
    if set(data) != expected or end != args['t']:
        raise ValueError('Incomplete complete-time trajectory')
    return dict(folder=str(folder), manifest=manifest, data=data, binding=binding,
                ancestry=ancestry, profile_receipts=receipts)


def controlled_inputs(base, other, changed):
    ignored = {'output', 'resume', 'library', 'fast_library'} | set(changed)
    a, b = base['manifest']['arguments'], other['manifest']['arguments']
    if {k:v for k,v in a.items() if k not in ignored} != {k:v for k,v in b.items() if k not in ignored}:
        raise ValueError('Uncontrolled numerical inputs')
    if base['binding'] != other['binding']:
        raise ValueError('Different numerical source or executable')
    if set(base['data']) != set(other['data']):
        raise ValueError('Different trajectory coverage')


def differences(a, b, gradient_scale=None):
    """Use the coarser physical nodes and their mapped radial measure."""
    ra, rb = a['r'], b['r']; ratio = (len(rb)-1)//(len(ra)-1)
    if ratio < 1 or (len(rb)-1) != ratio*(len(ra)-1) or not np.array_equal(ra, rb[::ratio]):
        raise ValueError('Require identical or nested physical nodes')
    measure = ra**2*a['grid_jacobian']; out = {}
    def norm(values, mask):
        return float(np.sqrt(np.sum(measure[mask]*values[mask]**2)/np.sum(measure[mask])))
    for region, mask in [('bulk', ra<.8), ('endpoint', ra>.9)]:
        delta = a['q_physical'][:,:7]-b['q_physical'][::ratio,:7]
        out['fields_'+region] = float(np.sqrt(np.mean([norm(delta[:,j],mask)**2 for j in range(7)])))
        for j in range(7):
            out[f'field{j}_{region}'] = norm(delta[:,j],mask)
    full = np.ones(len(ra), dtype=bool)
    for j, name in enumerate(['H','M','Theta','Z']):
        delta = a['diagnostic'][:,j]-b['diagnostic'][::ratio,j]
        out[name+'_rms'] = norm(delta,full)
        out[name+'_max'] = float(max(abs(delta)))
    ga, gb = physical_lapse_gradient(a), physical_lapse_gradient(b)[::ratio]
    inner = ra<.3; out['lapse_gradient_max'] = float(max(abs(ga[inner]-gb[inner])))
    scale = float(max(abs(ga[inner]))) if gradient_scale is None else gradient_scale
    out['lapse_gradient_relative'] = out['lapse_gradient_max']/scale
    out['B_max'] = float(max(abs(a['B']-b['B'][::ratio])))
    out['ray_position_max'] = float(np.max(abs(a['rays'][:,0]-b['rays'][:,0])))
    out['ray_logJ_max'] = float(np.max(abs(a['rays'][:,1]-b['rays'][:,1])))
    ma, mb = a['diagnostic'][:,10], b['diagnostic'][::ratio,10]
    if not np.array_equal(np.isfinite(ma), np.isfinite(mb)):
        raise ValueError('Changed finite mass-diagnostic support')
    finite = np.isfinite(ma); out['mass_max'] = float(max(abs(ma[finite]-mb[finite])))
    if not all(np.isfinite(v) for v in out.values()):
        raise ValueError('Nonfinite comparison metric')
    return out


def summarize(histories):
    names = histories[0]['base_half'].keys(); result = {}
    spatial = 'spatial' in histories[0]
    for key in names:
        peaks = {name:max(row[name][key] for row in histories) for name in ['base_half','half_quarter','base_quarter']+(['spatial'] if spatial else [])}
        e1, e2 = peaks['base_half'], peaks['half_quarter']
        order = float(np.log2(e1/e2)) if e1>0 and e2>0 else None
        fraction = None
        if spatial:
            denominator = peaks['spatial']
            fraction = peaks['base_quarter']/denominator if denominator>0 else (0. if peaks['base_quarter']==0 else None)
        result[key] = dict(peaks=peaks,observed_time_order=order,
            raw_time_to_space_fraction=fraction,
            below_ten_percent=bool(fraction is not None and fraction<.1) if spatial else None)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ['baseline','half','quarter']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--spatial',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();runs=[load(x) for x in [args.baseline,args.half,args.quarter]]
    base=runs[0]
    for other,factor in zip(runs[1:],[2,4]):
        controlled_inputs(base,other,{'cfl'})
        if base['manifest']['time_lattice']['dt'] != factor*other['manifest']['time_lattice']['dt']:
            raise ValueError('Require exact dt, dt/2, dt/4')
    space=load(args.spatial) if args.spatial else None
    if space:
        controlled_inputs(base,space,{'n'})
        if space['manifest']['arguments']['n'] <= base['manifest']['arguments']['n']:
            raise ValueError('Spatial reference must be finer')
    rows=[]
    for t in sorted(base['data']):
        a,b,c=[run['data'][t] for run in runs]
        scale=float(max(abs(physical_lapse_gradient(a)[a['r']<.3])))
        row=dict(t=t,base_half=differences(a,b,scale),half_quarter=differences(b,c,scale),base_quarter=differences(a,c,scale))
        if space:row['spatial']=differences(a,space['data'][t],scale)
        rows.append(row)
    records=[]
    for run in runs+([space] if space else []):
        records.append({k:v for k,v in run.items() if k!='data'})
    result=dict(records=records,history=rows,metrics=summarize(rows),goal_status='incomplete',
        scope='Verified frozen numerical import closure and libraries; complete exact saved times including ancestry. Raw baseline-to-dt/4 differences are compared with the raw middle-to-fine spatial difference. Observed orders are diagnostic, not error-bound proofs. Fixed-CFL spatial differences contain a temporal contribution. Report all failed or unresolved metrics; no temporal result overrides spatial/boundary/parameter failures.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['metrics'],indent=2))

if __name__=='__main__':main()
