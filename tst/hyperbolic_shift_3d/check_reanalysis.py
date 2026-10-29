"""Compare exported cell constraints with a native V6 report; never evolve."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('Output must be new')
    e=args.export;m=json.loads((e/'export.json').read_text())
    if m['status']!='completed' or not m['provenance']['execution_artifacts_verified']:
        raise ValueError('Export lacks verified execution provenance')
    if m['header']['magic']!='HYP3D_MPI_V6' or m['reanalyze']['bulk_diagnostic_operator']!='donor_seven_last3':
        raise ValueError('Expected V6 checkpoint and bulk diagnostic operator')
    for name in ['evolution.diagnostics.csv','evolution.coordinates.csv','source.report.json']:
        path=e/name;bound=m['output_files'][name]
        if path.stat().st_size!=bound['bytes'] or digest(path)!=bound['sha256']:
            raise ValueError('Changed exported artifact: '+name)
    report=json.loads((e/'source.report.json').read_text())
    if not report['diagnostics_current']:raise ValueError('Native diagnostics are stale')
    v=np.genfromtxt(e/'evolution.diagnostics.csv',delimiter=',',names=True)
    coords=np.genfromtxt(e/'evolution.coordinates.csv',delimiter=',',names=True)
    if len(v)!=report['cells'] or len(coords)!=len(v) or not np.all(np.isfinite(coords['volume'])&(coords['volume']>0)):
        raise ValueError('Invalid volume data')
    for k in ['x','y','z','r']:
        if not np.allclose(v[k],coords[k],atol=1e-14,rtol=1e-14):raise ValueError('Coordinate order differs')
    rows=[]
    for exterior in [False,True]:
        mask=v['r']>report['exterior_coordinate_cutoff'] if exterior else np.ones(len(v),bool)
        if not mask.any():raise ValueError('Empty diagnostic region')
        w=coords['volume'][mask];w=w/w.sum()
        for field,metric in [('H','H'),('M_norm','M')]:
            measured=float(np.sqrt(np.dot(w,v[field][mask]**2)))
            name=metric+('_exterior' if exterior else '')+'_L2';ref=report[name];error=abs(measured-ref)
            if not np.isfinite(error) or error>2e-12*(1+abs(ref)):raise ValueError('Constraint parity failed: '+name)
            rows.append(dict(field=name,reanalyzed=measured,native_sycl=ref,absolute_error=error))
    result=dict(passed=True,scope='Read-only coordinate-volume native diagnostic parity; not evolution accuracy',
        rows=rows,bulk_diagnostic_operator=m['reanalyze']['bulk_diagnostic_operator'],
        ray_diagnostic_operator=m['reanalyze']['ray_diagnostic_operator'],
        source_checkpoint_sha256=m['checkpoint_sha256'],export_manifest_sha256=digest(e/'export.json'),
        script_sha256=digest(Path(__file__)))
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
