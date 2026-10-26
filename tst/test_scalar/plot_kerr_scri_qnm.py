#!/usr/bin/env python3
"""Free two-branch QNM fits of the saved exact-scri local Kerr radiation."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fit_ringdown import fit
from localize_schwarzschild_error import sha, save, COLORS


def predict(t, record):
    dt=t-record['window'][0]
    amps=np.array([complex(*v) for v in record['amplitudes']])
    basis=np.column_stack((np.exp((-record['decay']-1j*record['omega_real'])*dt),
                           np.exp((-record['counterrotating_decay']+1j*record['counterrotating_real'])*dt)))
    return basis@amps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();R=a.run.resolve();A=R/'analysis';out=a.output.resolve()
    if out.exists():raise RuntimeError('Preserve previous evidence; use a new output directory')
    out.mkdir(parents=True)
    ns=[12,16,20];starts=list(range(30,71,5));width=30
    protocol=dict(extraction='Exact scri endpoint, cap=1; radiation r Phi = S psi with S=24M',
        signal='Complex scalar (2,2) mode, diagonal real/imaginary harmonic Gram normalization; raw projection sensitivity checked on N20.',
        convention='z=Aplus exp[(-Gamma_plus-i omega_plus)(t-t0)]+Aminus exp[(-Gamma_minus+i omega_minus)(t-t0)]. Counter-rotating frequencies plotted as positive magnitudes.',
        spin=.5,resolutions=ns,window_starts=starts,window_width=width,
        preserved_common_windows=[[35,65],[40,70],[45,75]],
        exploratory_extension='All sliding windows30-60 through70-100M, fixed before calculating additional fits. No window selected using the target frequency.',
        method='Existing multistart variable projection: four real frequency/damping parameters and two free complex amplitudes. Targets used only for evaluation.',
        qualification='Coarse local exploration only; no evolution or new allocation. A low residual alone is not a reliable QNM identification.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    inputs=[a.expected.resolve(),A/'qa.json',A/'scri-common-window-fits.json',R/'plan.json',R/'observer-review.json',R/'test_scalar.build.json',R/'source/main.cpp']
    qa=json.loads((A/'qa.json').read_text())
    if not qa['passed']:raise RuntimeError('Preserved local-run QA failed')
    cached=json.loads((A/'scri-common-window-fits.json').read_text())
    modes=json.loads(a.expected.read_text())['modes'];co=next(m for m in modes if m['a_over_M']==.5 and m['m']==2);counter=next(m for m in modes if m['a_over_M']==.5 and m['m']==-2)
    targets={'omega_real':co['Momega_real'],'decay':-co['Momega_imag'],
             'counterrotating_real':counter['Momega_real'],'counterrotating_decay':-counter['Momega_imag']}
    records=[];reused=0;additional=0;data={}
    for n in ns:
        file=A/f'scri-n{n}.csv';inputs.append(file)
        if qa['output_sha256'][file.name]!=sha(file):raise RuntimeError('Scri waveform differs from preserved QA')
        w=np.genfromtxt(file,delimiter=',',names=True);data[n]=w
        if not np.array_equal(w['time'],np.arange(201)/2) or not np.isfinite(w.view(float)).all():raise RuntimeError('Incomplete dense scri trace')
        previous=next(r for r in cached['records'] if r['n']==n)
        for lo in starts:
            hi=lo+width;mask=(w['time']>=lo)&(w['time']<=hi);t=w['time'][mask];z=w['real22'][mask]+1j*w['imag22'][mask]
            old=next((f for f in previous['fits'] if f['window']==[lo,hi]),None)
            if old is None:
                f=fit(t,z,True);f['window']=[lo,hi];additional+=1
            else:
                f=dict(old);reused+=1
            residual=float(np.linalg.norm(predict(t,f)-z)/np.linalg.norm(z))
            if abs(residual-f['relative_residual'])>1e-11:raise RuntimeError('Saved fit cannot reproduce its residual')
            f.update(n=n,projection='normalized',relative_errors={key:(f[key]-target)/target for key,target in targets.items()},
                     rechecked_relative_residual=residual)
            records.append(f)
    raw=[]
    for lo in starts:
        w=data[20];mask=(w['time']>=lo)&(w['time']<=lo+width);z=w['raw_real22'][mask]+1j*w['raw_imag22'][mask]
        f=fit(w['time'][mask],z,True);f.update(n=20,window=[lo,lo+width],projection='raw');raw.append(f);additional+=1
    raw_sensitivity={key:max(abs(r[key]-next(f for f in records if f['n']==20 and f['window']==r['window'])[key])/target for r in raw) for key,target in targets.items()}
    with (out/'fits.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['N','projection','start','end',*targets,'relative_residual','basis_condition'])
        for f in records+raw:writer.writerow([f['n'],f['projection'],*f['window'],*[f[key] for key in targets],f['relative_residual'],f['basis_condition']])
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.5),constrained_layout=True)
    quantities=[('omega_real','Co-rotating frequency',r'$M\omega_{R,+}$',.02),
                ('counterrotating_real','Counter-rotating frequency magnitude',r'$M|\omega_{R,-}|$',.02),
                ('decay','Co-rotating damping rate',r'$M\Gamma_+$',.05),
                ('counterrotating_decay','Counter-rotating damping rate',r'$M\Gamma_-$',.05)]
    for ax,(key,title,ylabel,tol) in zip(axes.flat,quantities):
        target=targets[key]
        ax.axhspan(target*(1-tol),target*(1+tol),color='gray',alpha=.12,label=f'previous {100*tol:g}% accuracy band')
        ax.axhline(target,color='black',ls='--',lw=1.4,label=f'Leaver target {target:.6f}')
        for n,color in zip(ns,COLORS):
            r=[f for f in records if f['n']==n];ax.plot(starts,[f[key] for f in r],'o-',color=color,label=f'N={n}',ms=4)
        ax.plot(starts,[f[key] for f in raw],'s',color='#666666',mfc='none',ms=5,label='N20 raw projection')
        ax.set(xlabel=r'Window start $t_0/M$; end = $t_0/M+30$',ylabel=ylabel,title=title,xticks=starts)
        ax.grid(alpha=.2);ax.legend(fontsize=7.5,loc='best')
    fig.suptitle(r'Exact-$\mathscr{I}^{+}$ scalar Kerr QNMs: $a/M=0.5$, $s=0$, $\ell=2$, fundamental branches'+'\nFree frequencies and amplitudes; every fixed-width window shown. Gray bands are accuracy guides, not confidence intervals.',fontsize=12)
    save(fig,out,'scri-frequencies-damping')

    # Show what the fitting windows cover and the actual complex residuals.
    fig,axes=plt.subplots(2,2,figsize=(12.5,8),constrained_layout=True)
    w=data[20];z=w['real22']+1j*w['imag22'];mask=(w['time']>=30)&(w['time']<=100)
    axes[0,0].plot(w['time'][mask],z.real[mask],'k-',label='N20 scri data')
    axes[0,1].plot(w['time'][mask],z.imag[mask],'k-',label='N20 scri data')
    for lo,color in zip([35,45,60],['#0072b2','#d55e00','#009e73']):
        f=next(f for f in records if f['n']==20 and f['window']==[lo,lo+30]);m=(w['time']>=lo)&(w['time']<=lo+30);pred=predict(w['time'][m],f)
        axes[0,0].plot(w['time'][m],pred.real,'--',color=color,label=f'fit {lo}-{lo+30}M')
        axes[0,1].plot(w['time'][m],pred.imag,'--',color=color,label=f'fit {lo}-{lo+30}M')
    for n,color in zip(ns,COLORS):
        r=[f for f in records if f['n']==n];axes[1,0].semilogy(starts,[f['relative_residual'] for f in r],'o-',color=color,label=f'N={n}')
    axes[0,0].set(xlabel='t/M',ylabel='Re radiation mode');axes[0,1].set(xlabel='t/M',ylabel='Im radiation mode')
    axes[1,0].set(xlabel='Window start t0/M; width 30M',ylabel='complex relative fit residual',title='Small residual does not establish the correct frequencies')
    for n,color in zip(ns,COLORS):
        r=[f for f in records if f['n']==n]
        ratios=[max(abs(f['relative_errors'][key])/tol for key,_,_,tol in quantities) for f in r]
        axes[1,1].semilogy(starts,ratios,'o-',color=color,label=f'N={n}')
    axes[1,1].axhline(1,color='black',ls='--');axes[1,1].set(xlabel='Window start t0/M; width 30M',ylabel='largest error / previous accuracy limit',title='All four quantities must be below 1 for a window')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Exact-scri two-fundamental fits: signal, residual and window sensitivity\nAll three grids are exploratory; no time/boundary/reference-resolution certification.',fontsize=12)
    save(fig,out,'scri-fit-diagnostics')
    original=[f for f in records if f['n']==20 and f['window'][0] in [35,40,45]]
    result=dict(protocol=protocol,targets=targets,expected_modes=[co,counter],fits=records,raw_fits=raw,
                raw_projection_max_difference_relative_to_target=raw_sensitivity,
                representative_window=[45,75],representative_N20=next(f for f in records if f['n']==20 and f['window']==[45,75]),
                N20_original_window_spreads={key:(max(f[key] for f in original)-min(f[key] for f in original))/np.mean([f[key] for f in original]) for key in targets},
                all_four_within_previous_accuracy={n:[f['window'] for f in records if f['n']==n and all(abs(f['relative_errors'][key])<=tol for key,_,_,tol in quantities)] for n in ns},
                normalized_fits_reused=reused,additional_postprocessing_fits=additional,new_evolutions=0,new_allocations=0,
                qualification='Window scan is exploratory. No statistical uncertainty is inferred from overlapping windows or from optimizer success.',
                input_sha256={str(p):sha(p) for p in inputs},source_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(__file__).with_name('fit_ringdown.py').resolve()):sha(Path(__file__).with_name('fit_ringdown.py'))},
                output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps({key:result[key] for key in ['targets','representative_N20','raw_projection_max_difference_relative_to_target','all_four_within_previous_accuracy']},indent=2))


if __name__=='__main__':main()
