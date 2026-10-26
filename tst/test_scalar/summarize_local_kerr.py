#!/usr/bin/env python3
"""Summarize local exploratory Kerr runs without changing admission decisions."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scalar_evidence import parent, snapshot, waveform, history
from fit_ringdown import fit
from localize_schwarzschild_error import Sampler, sha, save, COLORS
from validate_kerr import reconstruct


def scri_history(run):
    file=run/'waveform-scri.csv';rows=np.atleast_1d(np.genfromtxt(file,delimiter=',',names=True));previous=parent(run)
    if previous is not None:
        old=scri_history(previous);rows=np.concatenate((old[old['time']<rows['time'][0]-1e-12],rows))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args();R=a.run.resolve();E=a.evidence.resolve();out=R/'analysis';scripts=Path(__file__).resolve().parent
    while not (R/'completion.json').exists():
        if not a.wait:raise RuntimeError('Local exploration has not completed')
        time.sleep(15)
    if out.exists():raise RuntimeError('Existing analysis is immutable')
    out.mkdir();complete=json.loads((R/'completion.json').read_text());ns=[12,16,20];runs=[Path(complete['finished'][str(n)]) for n in ns]
    reference=Path(complete['independent']);ref=np.genfromtxt(reference,delimiter=',',names=True);refz=ref['real22']+1j*ref['imag22'];expected=E/'spin-zero-qnm-both-branches.json'
    # The existing validator records failed checks as well as successful ones.
    review=out/'numerical-review.json'
    subprocess.run([sys.executable,str(scripts/'validate_kerr.py'),'--runs',*[str(r) for r in runs],'--oracle-waveform',str(reference),'--reference-sampler','lagrange6','--expected',str(expected),'--output',str(review)],check=True)
    numerical=json.loads(review.read_text());waves=[waveform(run) for run in runs];scri=[scri_history(run) for run in runs]
    for n,w,s in zip(ns,waves,scri):
        if not np.array_equal(w['time'],s['time']) or w['time'][-1]!=100:raise RuntimeError('Dense local scri/finite-radius times differ')
        for name,values in [('finite-radius',w),('scri',s)]:
            np.savetxt(out/f'{name}-n{n}.csv',np.column_stack([values[key] for key in values.dtype.names]),delimiter=',',header=','.join(values.dtype.names),comments='')
    # Distinct, preselected earlier windows for radiation that reaches scri sooner.
    scrifits=[]
    for n,w in zip(ns,scri):
        z=w['real22']+1j*w['imag22'];fits=[]
        for lo,hi in [(20,50),(25,55),(30,60)]:
            mask=(w['time']>=lo)&(w['time']<=hi);f=fit(w['time'][mask],z[mask],True);f['window']=[lo,hi];fits.append(f)
        scrifits.append(dict(n=n,fits=fits))
    # Independent scri reference at its preserved profile times, without reruns.
    table=E/'kerr-oracle/coefficients-spin05-n600-a8.txt'
    with table.open() as stream:
        nr,na,h=map(float,stream.readline().split());stream.readline();weights=np.fromstring(stream.readline(),sep=' ')
    normalization=2*np.pi*np.sqrt(15/(32*np.pi));scriref=[]
    for t in [2,6,12,35,65,100]:
        profile=np.genfromtxt(str(reference)+f'-profile-{t}.csv',delimiter=',',names=True);end=profile[-int(na):];z=24*normalization*np.sum(weights*(end['W_real']+1j*end['W_imag']));scriref.append(dict(time=t,real=float(z.real),imag=float(z.imag)))

    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for n,w,s,color in zip(ns,waves,scri,COLORS):
        axes[0,0].plot(w['time'],w['real22'],color=color,label=f'N={n}')
        axes[0,1].plot(w['time'],w['imag22'],color=color,label=f'N={n}')
        axes[1,0].plot(s['time'],s['real22'],color=color,label=f'N={n}')
        axes[1,1].plot(s['time'],s['imag22'],color=color,label=f'N={n}')
    axes[0,0].plot(ref['time'],ref['real22'],'k--',lw=1,label='independent600x8');axes[0,1].plot(ref['time'],ref['imag22'],'k--',lw=1,label='independent600x8')
    for ax,key in [(axes[1,0],'real'),(axes[1,1],'imag')]:ax.scatter([s['time'] for s in scriref],[s[key] for s in scriref],marker='x',s=45,color='black',label='independent scri samples',zorder=4)
    for ax,title in zip(axes.flat,['Finite-radius Re mode','Finite-radius Im mode','Exact-scri Re radiation mode','Exact-scri Im radiation mode']):ax.set(xlabel='t/M',title=title);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Local a/M=0.5 exploration: real scalar pulse on a frozen analytical Kerr trumpet\nN12/16/20 are coarse exploratory grids; the600x8 independent solution is not resolution-certified',fontsize=12)
    save(fig,out,'waveforms')

    modes=json.loads(expected.read_text())['modes'];co=next(m for m in modes if m['a_over_M']==.5 and m['m']==2);counter=next(m for m in modes if m['a_over_M']==.5 and m['m']==-2)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for n,rec,color in zip(ns,numerical['QNM'],COLORS):
        for k,(key,gamma,target) in enumerate([('omega_real','decay',co),('counterrotating_real','counterrotating_decay',counter)]):
            axes[0,k].scatter([f[key] for f in rec['fits']],[f[gamma] for f in rec['fits']],color=color,label=f'N={n}: three windows',s=40)
    independent=json.loads((R/'qnm-independent.json').read_text())
    for k,(key,gamma,target) in enumerate([('omega_real','decay',co),('counterrotating_real','counterrotating_decay',counter)]):
        axes[0,k].scatter([f[key] for f in independent['records']],[f[gamma] for f in independent['records']],marker='s',facecolors='none',edgecolors='black',label='independent600x8')
        axes[0,k].scatter([target['Momega_real']],[-target['Momega_imag']],marker='x',s=90,color='black',label='independent Leaver target')
        axes[0,k].set(xlabel='M omega_R',ylabel='M decay',title='Co-rotating branch' if k==0 else 'Counter-rotating branch');axes[0,k].legend(fontsize=8)
    axes[1,0].loglog(ns,[w['relative_L2'] for w in numerical['independent_waveform']],'o-');axes[1,0].set(xlabel='resolution N',ylabel='relative waveform L2 difference',title='Finite-radius comparison to exploratory reference')
    for n,s,color in zip(ns,scri,COLORS):
        mask=(s['time']>=20)&(s['time']<=70);axes[1,1].semilogy(s['time'][mask],abs(s['real22'][mask]+1j*s['imag22'][mask]),color=color,label=f'N={n}')
    axes[1,1].set(xlabel='t/M',ylabel='absolute radiation mode',title='Exact-scri ringdown amplitude');axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Exploratory spin-zero Kerr QNMs: complex frequencies and amplitudes fitted freely\nAll predeclared finite-radius windows35-65,40-70,45-75 shown; no formal admission is claimed',fontsize=12)
    save(fig,out,'ringdown-comparison')

    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);rho=np.linspace(1.2,24,1201);direction=np.array([.8,.6,0.]);xyz=rho[:,None]*direction;fake=np.column_stack((xyz,np.zeros((len(xyz),9))))
    radial={}
    for col,t in enumerate([12,100]):
        profile=np.genfromtxt(str(reference)+f'-profile-{t}.csv',delimiter=',',names=True);_,truth=reconstruct(fake,profile,sampler='lagrange6');axes[0,col].plot(rho,truth[:,0],'k-',label='independent600x8')
        for n,run,color in zip(ns,runs,COLORS):
            data=snapshot(run,t);sampler=Sampler(data,n);values=sampler(xyz)[:,0];axes[0,col].plot(rho,values,color=color,label=f'N={n}');axes[1,col].semilogy(rho,abs(values-truth[:,0]),color=color,label=f'N={n}');radial[f't{t}-n{n}']=values
            del sampler,data
        axes[0,col].set(title=f'Radial slice at t={t}M',ylabel='rescaled field psi');axes[1,col].set(xlabel='compact radius rho/M',ylabel='absolute field difference')
        for row in [0,1]:axes[row,col].axvspan(8,16,color='gray',alpha=.1);axes[row,col].legend(fontsize=8);axes[row,col].grid(alpha=.2)
    fig.suptitle('Local Kerr radial profiles along direction(0.8,0.6,0); exact scri at rho=24M\nSix-point sampling for display; common exterior field/derivative norms recorded separately',fontsize=12)
    save(fig,out,'radial-profiles')
    np.savez_compressed(out/'radial-data.npz',rho=rho,**radial)
    protected=json.loads((E/'frozen-baseline-sha256.json').read_text());bad=[name for name,digest in protected.items() if sha(name)!=digest]
    if bad:raise RuntimeError('Protected production baseline changed')
    summary=dict(exploration_complete=True,formal_validation=False,spin=.5,resolutions=ns,duration=100,
        waveform_relative_L2=[w['relative_L2'] for w in numerical['independent_waveform']],
        finite_radius_QNM=numerical['QNM'],scri_QNM=scrifits,scri_reference_samples=scriref,
        independent_reference_resolution=[600,8],independent_reference_QNM=independent,
        reduction=numerical['reduction'],field_checks_passed=numerical['field_checks_passed'],
        frozen_all_unchanged=all(json.loads((run/'frozen-integrity.json').read_text())['unchanged'] for run in runs),
        protected_baseline_files=len(protected),protected_baseline_mismatches=bad,
        remote_submissions=0,future_Aurora=dict(queue='capacity',nodes=1,walltime='00:30:00',project='CompactBinaryMerger',maximum_task_allocations=1),
        limits='Coarse local exploration only. No resolved independent-reference uncertainty, timestep controls, boundary controls or GPU comparison. All failed numerical checks remain recorded.',
        inputs_sha256={str(p):sha(p) for p in [R/'completion.json',R/'plan.json',R/'observer-review.json',expected,table,review]},
        source_sha256=sha(__file__),outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps(dict(exploration_complete=True,waveform_relative_L2=summary['waveform_relative_L2'],field_checks_passed=summary['field_checks_passed'],summary=str(out/'summary.json')),indent=2))


if __name__=='__main__':main()
