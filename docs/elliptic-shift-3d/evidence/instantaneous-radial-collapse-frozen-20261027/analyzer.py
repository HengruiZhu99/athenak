#!/usr/bin/env python3
"""Analyze frozen instantaneous solves; no evolution or acceptance exclusions."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(path):
    manifest=json.loads((path/'manifest.json').read_text());cases=[]
    for case in manifest['cases']:
        if 'result' not in case:
            cases.append(dict(name=case['name'],status=case['status'],error=case.get('error')));continue
        result=case['result'];diag=result['regional_diagnostics']
        with gzip.open(path/(case['name']+'.cells.csv.gz'),'rt') as stream:
            points=np.genfromtxt(stream,delimiter=',',names=True)
        xyz=np.column_stack([points[i] for i in ['x','y','z']]);radius=points['r'];normal=xyz/radius[:,None]
        beta=np.column_stack([points[f'beta_target_{i}'] for i in range(3)])
        maxima={}
        for name in ['FD','ref7','ref9']:
            error=np.column_stack([points[f'beta_{name}_{i}'] for i in range(3)])-beta
            norm=np.linalg.norm(error,axis=1);index=int(norm.argmax());radial=np.sum(error*normal,axis=1)
            maxima[name]=dict(max_vector=float(norm[index]),radius=float(radius[index]),point=xyz[index].tolist(),
                maximum_radial_error=float(abs(radial).max()),maximum_tangential_error=float(np.linalg.norm(error-radial[:,None]*normal,axis=1).max()))
        source_error=np.column_stack([points[f'source_FD_{i}']-points[f'source_ref7_{i}'] for i in range(3)])
        norm=np.linalg.norm(source_error,axis=1);index=int(norm.argmax())
        maxima['FD_source_minus_ref7']=dict(max_vector=float(norm[index]),radius=float(radius[index]),point=xyz[index].tolist())
        angular=[]
        for row in result['angular_comparison']:
            angular.append(dict(radius=row['r'],reference_radial=row['target_radial'],
                FD=row['series'][0],interpolated7=row['series'][1],target_projection=row['series'][3]))
        cases.append(dict(name=case['name'],time=result['snapshot_time'],solves=result['solves'],
            minimum_radius=result['minimum_cell_radius'],minimum_chi=result['minimum_chi'],minimum_alpha=result['minimum_alpha'],
            scri_normal_residual=result['scri_normal_ghost_residual'],scri_tangential_residual=result['scri_tangential_ghost_residual'],
            maxima=maxima,regional_diagnostics=diag,angular_comparison=angular))
    return dict(scope='Instantaneous global3D elliptic comparison only; no evolution; no spatial convergence claim',
        norm_definition='Cartesian vector component Euclidean square, averaged using physical spatial volume or conformal coordinate volume as separately labeled; every active cell retained in all-region norms',
        source='Original curvature source from physical alpha and A; target beta only initializes solves/diagnostics; no artificial L(beta_target) replacement',
        endpoint_limitation='Minimum sampled radius0.0360844; no r0 limiting data supplied; late inner fronts at1e-7 are unresolved',
        source_unchanged=manifest['sources_unchanged_during_run'],binary_unchanged=manifest['binary_unchanged_during_run'],cases=cases)


def plot(path,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(summary['cases']),2,figsize=(10,9),layout='constrained')
    for axes_row,case in zip(axes,summary['cases']):
        angular=case['angular_comparison'];r=np.array([a['radius'] for a in angular]);target=np.array([a['reference_radial'] for a in angular])
        axes_row[0].plot(r,target,'k-',label='frozen radial beta')
        for name,label,color in [('FD','FD source','tab:blue'),('interpolated7','independent interpolated source','tab:orange')]:
            mean=np.array([a[name]['mean'] for a in angular]);scatter=np.array([a[name]['raw_scatter'] for a in angular])
            axes_row[0].plot(r,mean,'.-',color=color,label=label);axes_row[0].fill_between(r,mean-scatter,mean+scatter,color=color,alpha=.15)
            axes_row[1].semilogy(r,abs(mean-target),'.-',color=color,label=label+' mean error')
            axes_row[1].semilogy(r,scatter,'--',color=color,label=label+' angular scatter')
        axes_row[1].semilogy(r,[a['target_projection']['raw_scatter'] for a in angular],'k:',label='same-grid target projection scatter')
        for ax in axes_row:ax.set_xlabel('compactified radius');ax.grid(alpha=.25)
        axes_row[0].set_ylabel(f't={case["time"]:g}: radial shift');axes_row[1].set_ylabel('absolute error / angular scatter')
    for ax in axes[0]:ax.legend(fontsize=7)
    fig.suptitle('Instantaneous N6 composite3D:5616 cells; no evolution or convergence claim')
    fig.savefig(path/'angular-comparison.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--plot',action='store_true');args=parser.parse_args()
    output=args.directory/'analysis.json'
    if output.exists():raise FileExistsError(output)
    result=analyze(args.directory);output.write_text(json.dumps(result,indent=2)+'\n')
    if args.plot:plot(args.directory,result)
    print(json.dumps(dict(cases=len(result['cases']),source_unchanged=result['source_unchanged'],binary_unchanged=result['binary_unchanged'],analysis_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
