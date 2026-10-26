#!/usr/bin/env python3
"""Recount native failures and compare coordinate-volume puncture norms."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_spherical import analyze_case, rows


def native(case):
    h = np.atleast_2d(np.loadtxt(case/'hyp.z4c.user.hst'))
    assert np.isfinite(h).all()
    r = json.loads((case/'receipt.json').read_text())
    np.testing.assert_array_equal(h, r['history_rows'])
    log = (case/'run.log').read_text()
    pattern = (r'invalid hyperboloidal physical ADM state: mesh_time=(\S+) cycle=(\d+) '
               r'xyz=\(([^)]+)\) Omega=(\S+) alpha=(\S+) chi=(\S+) determinant=(\S+) '
               r'P=(\S+) Theta=(\S+)')
    matches = list(re.finditer(pattern, log))
    failure = None
    if matches:
        assert len(matches) == 1
        m = matches[0]
        failure = dict(zip(['mesh_time','cycle','xyz','Omega','alpha','chi',
                            'metric_determinant','P','Theta'], m.groups()))
        failure['xyz'] = [float(v) for v in failure['xyz'].split(',')]
        for k in failure:
            if k != 'xyz': failure[k] = int(failure[k]) if k == 'cycle' else float(failure[k])
        failure['radius'] = math.sqrt(sum(v*v for v in failure['xyz']))
    assert (r['outcome'] == 'failed') == bool(failure)
    return {'path':str(case), 'outcome':r['outcome'], 'elapsed_seconds':r['elapsed_seconds'],
            'history_finite_and_matches_receipt':True, 'last_time':float(h[-1,0]),
            'last_H_l2':float(h[-1,2]), 'last_M_l2':float(h[-1,3]),
            'failure_recounted_from_raw_log':failure,
            'receipt_terminal_extractor_missed_failure':bool(failure and not r['terminal_messages'])}, h


def spherical(case):
    review = analyze_case(case)
    # Those thresholds belong to the wave-map gauge test, not this puncture test.
    review.pop('passed_old_failure_times')
    data = rows(case/'cubed_sphere_conformal.csv')
    groups = {}
    for r in data: groups.setdefault(float(r['time']),[]).append(r)
    values=[]
    for t, pair in sorted(groups.items()):
        assert len(pair) == 2 and {r['region'] for r in pair} == {'bulk','shell'}
        v = sum(float(r['volume']) for r in pair)
        norms = [math.sqrt(sum(float(r['volume'])*float(r[k])**2 for r in pair)/v)
                 for k in ['H_l2','M_conformal_l2']]
        values.append([t,*norms])
    review['combined_coordinate_volume_last'] = dict(zip(['time','H_l2','M_l2'],values[-1]))
    final_time = data[-1]['time']
    subsets = [r for r in rows(case/'cubed_sphere_conformal_regions.csv') if r['time']==final_time]
    for r in subsets:
        r['peak_radius'] = math.sqrt(sum(float(r['max_H_'+a])**2 for a in 'xyz'))
    review['last_spatial_subsets'] = subsets
    if 'first_bad_stage' in review:
        failure = review['first_bad_stage']
        radius = math.sqrt(sum(v*v for v in failure['xyz']))
        failure['radius'] = radius
        if radius >= .75: failure['Omega'] = (1-radius**2)/2
    r = json.loads((case/'receipt.json').read_text())
    review.update({k:r[k] for k in ['initial_profile_unchanged','initial_profile_payloads',
                                  'reference_payloads'] if k in r})
    return review, np.array(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capsule',type=Path)
    p.add_argument('--gpu-evolution',type=Path,required=True)
    args=p.parse_args(); cap=args.capsule.resolve()
    evidence={'scope':'Coordinate-volume RMS; different grids/operators, not a boundary-only comparison.',
              'native':{}, 'spherical':{}}
    series=[]
    for degree, style in [('degree3','-'),('degree4','--'),('degree3-halfstep',':')]:
        n,h=native(cap/'cpu'/('puncture-cartesian-t02-'+degree))
        evidence['native'][degree]=n
        series.append(('Cut sphere '+degree, '#6a526a', style, h[:,[0,2,3]]))
        s,h=spherical(args.gpu_evolution/degree)
        evidence['spherical'][degree]=s
        series.append(('Spherical '+degree, '#007a9e', style, h))
    for name in ['puncture-cartesian-original-two-step-admitted','puncture-cartesian-t002']:
        evidence['native'][name]=native(cap/'cpu'/name)[0]
    for name in ['two-step','t002']:
        evidence['spherical']['GPU-admission-'+name]=spherical(cap/'aurora/admission'/name)[0]
    (cap/'puncture-diagnostic-review.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    fig,axs=plt.subplots(2,2,figsize=(12,7.7),layout='constrained')
    for label,color,style,h in series:
        h=h[h[:,0]>0]  # Do not invent an initial ramp between sparse diagnostics.
        for col,key in enumerate(['Hamiltonian','Conformal momentum']):
            for row in [0,1]:
                ax=axs[row,col]
                ax.semilogy(h[:,0],h[:,col+1],style,color=color,label=label,lw=1.6)
                ax.plot(h[-1,0],h[-1,col+1],'o',color=color,ms=3)
                ax.set_ylabel(key+' RMS');ax.set_xlabel('Coordinate time')
                ax.set_xlim(0,.2 if row==0 else .02)
                ax.set_ylim(1e-8,1e8 if row==0 else 1)
    axs[0,0].set_title('Full bounded comparison')
    axs[0,1].set_title('Full bounded comparison')
    axs[1,0].set_title('Early interval')
    axs[1,1].set_title('Early interval')
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    fig.suptitle('Wormhole puncture: fixed Minkowski hyperboloidal reference\n'
                 'Different grids; endpoints mark last diagnostics, not necessarily failure stages',fontsize=13)
    fig.savefig(cap/'puncture-constraint-comparison.png',dpi=180)
    fig.savefig(cap/'puncture-constraint-comparison.pdf')
    print(json.dumps({'native':{k:v['failure_recounted_from_raw_log'] for k,v in evidence['native'].items()},
                      'spherical':{k:v.get('first_bad_stage',v['combined_coordinate_volume_last'])
                                   for k,v in evidence['spherical'].items()}},indent=2))


if __name__=='__main__':main()
