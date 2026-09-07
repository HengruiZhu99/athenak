#!/usr/bin/env python3
"""Native-grid Schwarzschild and refinement-interface qualification, zero evolution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def rows(path):
    return [line.split() for line in path.read_text().splitlines() if line and not line.startswith('#')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--athena',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    exe=args.athena.resolve(strict=True)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2]
    template=(repo/'inputs/z4c/onepuncture/z4c_kerr_puncture_cartoon.athinput').read_text()
    template=template.split('<output1>')[0]
    for old,new in [('x1min = -12.0','x1min = 0.0'),('x1max = 12.0','x1max = 2.0'),
                    ('x2min = -12.0','x2min = -2.0'),('x2max = 12.0','x2max = 2.0'),
                    ('ix1_bc = outflow','ix1_bc = axis'),('signed_rho_z_suppressed_y_v1','half_rho_z_suppressed_y_v2'),
                    ('symmetry_schema = 1','symmetry_schema = 2'),('chi = 0.99','chi = 0.0'),
                    ('z_h = 0.0','z_h = 0.0137'),('nlim = 4','nlim = 0'),
                    ('symmetry = cartoon_so2','grid_centering = vertex\nsymmetry = cartoon_so2')]:
        template=template.replace(old,new)
    template+='\n<fastflow>\nnum_horizons = 1\nlmax = 8\nntheta = 20\n<output1>\nfile_type = rst\ndt = 1.0\n'
    results=[]
    for n,amr in [(64,False),(128,False),(256,False),(512,False),(128,True),(256,True)]:
        name=f'n{n}'+('-amr' if amr else '')
        case=out/name;case.mkdir()
        text=template.replace('nx1 = 64',f'nx1 = {n}').replace('nx2 = 64',f'nx2 = {n}')
        if amr:
            text=text.replace('refinement = none','refinement = static\nnum_levels = 2')
            text+='\n<refined_region1>\nlevel = 1\nx1min = 0\nx1max = 0.5\nx2min = -0.5\nx2max = 0.5\n'
        fixture=case/'input.athinput';fixture.write_text(text)
        with (case/'initialize.log').open('w') as log:
            subprocess.run([str(exe),'-i',str(fixture),'-d',str(case/'initial')],stdout=log,stderr=log,check=True)
        subprocess.run([sys.executable,str(repo/'scripts/mots/search_checkpoint.py'),'--athena',str(exe),
            '--checkpoint',str(case/'initial/rst/kerr_puncture_cartoon.00000.rst'),
            '--output',str(case/'analysis'),'--lmax','8','--radius-min','0.4','--radius-max','0.6',
            '--radii','3','--axis-bound','0.2','--axis-samples','65'],check=True)
        report=json.loads((case/'analysis/search/frozen_mots.json').read_text())
        data=rows(case/'analysis/search/mots.cartoon_m0_horizon_0.txt')
        best=min(data,key=lambda r:float(r[13]))
        result=dict(case=name,residual=float(best[13]),area=float(best[7]),verified=report['verified_candidate'])
        assert report['time']==0 and report['active_state_unchanged'] and report['mesh_unchanged']
        results.append(result)
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    uniform=[r for r in results if '-amr' not in r['case']]
    assert all(b['residual']<a['residual'] for a,b in zip(uniform,uniform[1:]))
    assert uniform[-1]['verified']
    assert results[-1]['residual']<results[-2]['residual']
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
