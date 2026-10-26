#!/usr/bin/env python3
"""Re-differentiate saved states; no evolution or replacement of accepted gates."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np


def audit(folder):
    from analyze_telegraph_time import numerical_binding,sha
    from telegraph import options
    from puncture_chart import PunctureChart
    m=json.loads((folder/'manifest.json').read_text())
    if m['status']!='finished':raise ValueError('Require completed case')
    binding=numerical_binding(folder,m)
    # The worker imports the exact frozen numerical modules for this case.
    for name,digest in binding.items():
        module=sys.modules.get(name[:-3]) if name.endswith('.py') else None
        if module is not None and sha(module.__file__)!=digest:
            raise ValueError('Loaded numerical dependency differs: '+name)
    args=m['arguments'];path=sorted(folder.glob('profile-*.npz'))[-1]
    with np.load(path,allow_pickle=False) as z:
        r=z['r'];q=z['q'];saved=z['diagnostic'];t=float(z['t'])
    rows=[]
    for width in [7,9,11]:
        lib=folder/'frozen-source'/Path(args['library']).name
        opts=options(len(r),lib,args['nu'],args['cfl']);opts.width=width
        for key in ['grid_map','grid_stretch','quadrature_width','mapped_ko']:
            setattr(opts,key,args[key])
        ev=PunctureChart(opts)
        if not np.array_equal(ev.r,r):raise ValueError('Grid reconstruction changed')
        _,_,diagnostic,_=ev.parts(q)
        if width==7 and not np.array_equal(diagnostic,saved,equal_nan=True):
            raise ValueError('Original diagnostic not reproduced bitwise')
        row=dict(width=width)
        measure=r*r*ev.grid_jacobian
        for j,name in enumerate(['H','M','Theta','Z']):
            v=diagnostic[:,j];peak=int(np.argmax(abs(v)))
            row[name]=dict(maximum=float(abs(v[peak])),location=float(r[peak]),
                rms=float(np.sqrt(np.sum(measure*v*v)/np.sum(measure))),
                max_change_from_original=float(max(abs(v-saved[:,j]))))
        rows.append(row)
    return dict(case=str(folder.resolve()),profile=str(path.resolve()),profile_sha256=sha(path),
        manifest_sha256=sha(folder/'manifest.json'),time=t,n=len(r),original_diagnostic_bitwise=True,rows=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
    # Separate interpreters prevent one case's numerical imports leaking into
    # another case with a different frozen source snapshot.
    worker='import sys,runpy,json; from pathlib import Path; sys.path.insert(0,str(Path(sys.argv[1])/"frozen-source")); sys.path.append(str(Path(sys.argv[2]).parent)); ns=runpy.run_path(sys.argv[2]); print(json.dumps(ns["audit"](Path(sys.argv[1]))))'
    for folder in a.cases:
        result=subprocess.run([sys.executable,'-c',worker,str(folder.resolve()),str(Path(__file__).resolve())],check=True,capture_output=True,text=True)
        records.append(json.loads(result.stdout))
    result=dict(records=records,scope='Re-differentiation of identical saved states with 7/9/11-point stencils. Original 7-point diagnostic must reproduce bitwise. Frozen numerical dependencies and libraries verified. These are extraction sensitivity controls only; the evolution stencil, accepted-state extrema and acceptance criteria are not modified.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(records,indent=2))

if __name__=='__main__':main()
