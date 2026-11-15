#!/usr/bin/env python3
"""Plot complete A/B run histories without inventing floors for exact zeros."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scri_run_summary import history

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',action='append',required=True,help='label=run_directory')
p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
cases=[]
for entry in args.run:
    label,directory=entry.split('=',1);root=Path(directory)
    a,_=history(next(root.glob('*.z4c_hyperboloidal.hst')))
    columns={};streams=[]
    for group in range(4):
        b,c=history(next(root.glob(f'*.scri-investigation-{group}.hst')))
        streams.append((b,c))
    # Live copies can have different trailing rows. Use only exact common dump
    # times, without interpolating extrema or merging different stage windows.
    times=streams[0][0][:,0]*3
    for b,_ in streams[1:]:times=np.intersect1d(times,b[:,0]*3)
    manifest=root/'merge-manifest.json'
    if manifest.exists():
        # A restart between dumps loses the earlier part of the accumulated
        # extrema window. Do not present that short window as a matched one.
        for stream in json.loads(manifest.read_text())['streams'].values():
            if stream['windowed']:
                partial=[3*row['time_native'] for row in stream['rows'] if not row['complete_window']]
                times=times[~np.isin(times,partial)]
    assert len(times)>0
    for b,c in streams:
        index=np.searchsorted(b[:,0]*3,times)
        assert np.array_equal((b[:,0]*3)[index],times)
        columns.update({k:v[index] for k,v in c.items() if k not in ('time','dt')})
    cases.append(dict(label=label,time=a[:,0]*3,constraints=a[:,8:]*[1/9,1/9,1/3,1/3],
                      scri_time=times,columns=columns,
                      shells=[history(next(root.glob(f'*.scri-shell-{group}.hst')))[0] for group in range(4)],
                      horizon=np.loadtxt(next(root.glob('*.horizon_summary_0.txt')),ndmin=2)))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def magnitude(c,key):
    if key in c:return c[key]
    return np.maximum(abs(c[key+'-min']),abs(c[key+'-max']))

def save(fig,name):
    for ext in ('png','pdf'):fig.savefig(args.output/f'{name}.{ext}',dpi=160)
    plt.close(fig)

def panel_plot(name,title,spec,shape):
    fig,axs=plt.subplots(*shape,figsize=(11,3.1*shape[0]),sharex=True,layout='constrained')
    for ax,(label,keys) in zip(axs.flat,spec):
        positive=False
        for case in cases:
            if isinstance(keys,str):v=magnitude(case['columns'],keys)
            else:v=np.maximum.reduce([magnitude(case['columns'],key) for key in keys])
            positive|=bool(np.any(v>0))
            ax.plot(case['scri_time'],np.ma.masked_less_equal(v,0),label=case['label'],lw=1.4)
        ax.set_yscale('log');ax.set_title(label);ax.grid(alpha=.2,which='both')
        if not positive:
            ax.set_ylim(1e-18,1);ax.text(.5,.5,'All recorded values zero',transform=ax.transAxes,ha='center')
    axs.flat[0].legend(fontsize=8)
    for ax in axs[-1,:]:ax.set_xlabel(r'$t/M_0$')
    fig.suptitle(title+'\nNative conformal units; maxima over points and RK stages between dumps')
    fig.supxlabel('Exact zeros and incomplete restart windows omitted; no artificial floor.',fontsize=9)
    save(fig,name)

fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True,layout='constrained')
for i,(ax,label) in enumerate(zip(axs.flat,[r'$M_0^2 H_{\rm RMS}$',r'$M_0^2 \mathcal{M}_{\rm RMS}$',r'$M_0\Theta_{\rm RMS}$',r'$M_0 Z_{\rm RMS}$'])):
    for case in cases:ax.semilogy(case['time'],np.ma.masked_less_equal(case['constraints'][:,i],0),label=case['label'])
    ax.set_ylabel(label);ax.grid(alpha=.2,which='both')
axs[0,0].legend(fontsize=8)
for ax in axs[-1]:ax.set_xlabel(r'$t/M_0$')
fig.suptitle('Constraints exterior to the accepted apparent horizon')
fig.supxlabel('Compact-coordinate volume-weighted RMS; mass convention M0=1.',fontsize=9)
save(fig,'exterior-constraints')

fig,axs=plt.subplots(4,4,figsize=(14,11),sharex=True,layout='constrained')
bands=['[.80,.88)', '[.88,.94)', '[.94,.97)', '[.97,1)']
units=[r'$M_0^2 H_{\rm RMS}$',r'$M_0^2\mathcal{M}_{\rm RMS}$',r'$M_0\Theta_{\rm RMS}$',r'$M_0 Z_{\rm RMS}$']
for group in range(4):
    for component in range(4):
        ax=axs[component,group]
        for case in cases:
            a=case['shells'][group]
            v=a[:,3+component]*[1/9,1/9,1/3,1/3][component]
            ax.semilogy(a[:,0]*3,np.ma.masked_less_equal(v,0),label=case['label'])
        ax.grid(alpha=.2,which='both')
        if component==0:ax.set_title('Compact radius '+bands[group])
        if group==0:ax.set_ylabel(units[component])
        if component==3:ax.set_xlabel(r'$t/M_0$')
axs[0,0].legend(fontsize=7)
fig.suptitle('Constraints in four fixed outer radial shells')
fig.supxlabel('Compact-coordinate volume-weighted RMS; fixed regions, independent of the horizon mask.',fontsize=9)
save(fig,'shell-constraints')

panel_plot('scri-compatibility','Scri compatibility and clock obstruction',[
    (r'$\max|s-1|$','s-1'),(r'$\max|s_t|$ after reconstruction','z-post'),
    (r'$\max_i|C_{\beta^i}|$',['Bx','By','Bz']),('Original clock residual','c-orig'),
    ('Invariant I','I'),(r'$|I-I_{\rm geom}|$','I-Igeom')],(3,2))
panel_plot('scri-regularity','Remaining regularity and pole numerators divided by local h',[
    (r'$|\partial_n\Theta|$','Th-n'),('Scalar normal defect','scalar-n'),
    (r'$C_\Lambda$','reg-Lam'),(r'$C_A$','reg-A'),(r'$|N_\chi|/h$','Nchi/h'),
    (r'$|N_P|/h$','NP/h'),(r'$|N_\Theta|/h$','NTh/h'),
    (r'$|N_\Lambda|/h$','NLam/h'),(r'$|N_A|/h$','NA/h')],(3,3))
panel_plot('scri-corrections','Normal derivative corrections and their lift Hessians',[
    (r'$|\ell|$','ell'),(r'$\max_i|q^i|$',['qx','qy','qz']),
    ('Lapse lift Hessian','Hess-alp'),('Shift lift Hessian','Hess-bet'),
    (r'$|N_\alpha|/h$','Nalp/h'),(r'$|N_\beta|/h$','Nbet/h')],(3,2))
panel_plot('scri-penalties','Penalty contributions to boundary rates',[
    (r'$s_t$ penalty','pen-z'),(r'$\chi_t$ penalty','pen-chi'),
    (r'$P_t$ penalty','pen-P'),(r'$\Theta_t$ penalty','pen-Th'),
    (r'$\Lambda_t$ penalty','pen-Lam'),(r'$A_t$ penalty','pen-A')],(3,2))
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for case in cases:
    a=case['horizon'];axs[0].plot(a[:,1]*3,a[:,2]*3-1,label=case['label'])
    assert np.all(a[:,8]>=0)
    axs[1].semilogy(a[:,1]*3,np.ma.masked_less_equal(np.sqrt(a[:,8]),0),label=case['label'])
axs[0].set_ylabel(r'$M_{\rm AH}/M_0-1$');axs[1].set_ylabel('Expansion RMS (native units)')
for ax in axs:ax.set_xlabel(r'$t/M_0$');ax.grid(alpha=.2,which='both')
axs[0].legend(fontsize=8);fig.suptitle('Horizon mass and measured expansion residual');save(fig,'horizon')
