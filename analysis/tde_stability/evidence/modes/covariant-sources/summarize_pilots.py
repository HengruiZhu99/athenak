from pathlib import Path
import io,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
CASES=[('Z4c α0.1',ROOT.parent/'lapse-damping/baseline','#386cb0','--'),
 ('Z4c constant0.1',ROOT.parent/'lapse-damping/scaled_01','#e68a24','--'),
 ('Z4c constant0.3',ROOT.parent/'lapse-damping/scaled_03','#31985f','--'),
 ('Coupled α0.1',ROOT/'covariant_alpha01','#386cb0','-'),
 ('Coupled constant0.1',ROOT/'covariant_const01','#e68a24','-'),
 ('Coupled constant0.3',ROOT/'covariant_const03','#31985f','-'),
 ('Coupled constant1.0',ROOT/'covariant_const10','#a34e9c','-')]
def history(p):
    text=p.read_text()
    if not text.endswith('\n'):text=text[:text.rfind('\n')+1]
    return np.atleast_2d(np.loadtxt(io.StringIO(text)))
def fit(t,y,a,b,valid_until=np.inf):
    select=(t>=a-1e-8)&(t<=b+1e-8)&(y>0)&np.isfinite(y)
    if np.count_nonzero(select)<10:return None
    tt=t[select];yy=np.log(y[select]);coef=np.polyfit(tt,yy,1);pred=np.polyval(coef,tt)
    denom=np.sum((yy-np.mean(yy))**2)
    return dict(requested=[a,b],actual=[float(tt.min()),float(tt.max())],
        complete=bool(tt.max()>=b-1.05 and valid_until>=b),gamma=float(coef[0]),
        r_squared=float(1-np.sum((yy-pred)**2)/denom) if denom>0 else None)
fig,axes=plt.subplots(2,2,figsize=(11.5,7),layout='constrained',facecolor='white')
report=[]
for name,d,color,style in CASES:
    if not (d/'ks_background.user.hst').exists():continue
    u=history(d/'ks_background.user.hst');z=history(d/'ks_background.z4c.user.hst')
    t=u[:,0];ext=np.sqrt(np.maximum(z[:,9],0));interior=np.sqrt(np.maximum(z[:,14],0))
    log=(d/'run.log').read_text();ended=(d/'exit_code.txt').exists()
    invalid_lines=[line for line in log.splitlines() if line.startswith('C2P_INVALID_ADM_INPUT')]
    first_invalid=None
    valid_until=np.inf
    if invalid_lines:
        first_invalid=dict(re.findall(r'(\w+)=([^ ]+)',invalid_lines[0]))
        valid_until=float(first_invalid['cycle_start_time'])
    fields=[u[:,9],ext,u[:,12],u[:,15]];times=[t,z[:,0],t,t]
    for ax,xx,yy in zip(axes.ravel(),times,fields):
        good=xx<valid_until
        ax.semilogy(xx[good],np.where(yy[good]>0,yy[good],np.nan),color=color,ls=style,lw=1.5,label=name)
        if not good.all():
            ax.semilogy(xx[~good],np.where(yy[~good]>0,yy[~good],np.nan),color=color,ls=style,lw=1,alpha=.25)
            first=np.flatnonzero(~good)[0]
            ax.plot(xx[first],yy[first],marker='x',color=color,ms=7)
    rec=dict(name=name,path=str(d),time=float(t[-1]),process_ended=ended,
        reached_time_target=bool(t[-1]>=300-1e-8),
        exit_code=int((d/'exit_code.txt').read_text()) if ended else None,
        bad_metric_max=float(u[:,11].max()),finite=bool(np.isfinite(u).all() and np.isfinite(z).all()),
        invalid_log=bool(first_invalid or 'Z4C_INVALID_STATE' in log or 'FATAL ERROR' in log),
        first_invalid_adm=first_invalid,
        final=dict(rho_max_passive_fluid=float(u[-1,2]),Theta_max=float(u[-1,9]),Theta_exterior_L2=float(ext[-1]),
                   Theta_interior_L2=float(interior[-1]),alpha_res=float(u[-1,12]),Gamma_res=float(u[-1,15])),
        windows={name:[fit(xx[xx<valid_until],yy[xx<valid_until],a,b,valid_until) for a,b in [(50,100),(100,200),(200,300)]]
                  for name,xx,yy in [('Theta_max',t,u[:,9]),('Theta_exterior_L2',z[:,0],ext)]})
    # Initialization prints the actual trumpet horizon; input history radius is overwritten.
    rec['horizon_radius_M']=1.0
    checkpoint=d/'checkpoint-validity.json'
    if checkpoint.exists():
        q=json.loads(checkpoint.read_text())
        rec['saved_checkpoint']={k:q[k] for k in ['time_M','cycle','all_payload_finite','invalid_metric_cells_including_ghosts','passed']}
        rec['saved_checkpoint']['note']='Saved-state validity only, not a stability result.'
    report.append(rec)
for ax,ylabel in zip(axes.ravel(),[r'Active max $|\Theta|$',r'$\|\Theta\|_{L^2(r>1M)}$',r'Active max $|\delta\alpha|$',r'Active max $|\delta\Gamma|$']):
    ax.set(xlabel='Time / M',ylabel=ylabel,xlim=(0,300));ax.grid(alpha=.2)
axes[0,0].legend(fontsize=7,ncol=2)
fig.suptitle('G2 vacuum lapse pulse: coupled constraint sources, dx=0.25M, dt=0.0375M',fontsize=11)
fig.savefig(ROOT/'pilot-comparison.png',dpi=180)
plt.close(fig)
out=dict(scope='Finite single-block CPU controls; incomplete windows marked explicitly. Theta norm uses proper-volume square integral, r>1M; curves do not prove continuum or matter stability.',cases=report)
(ROOT/'pilot-results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([{k:v for k,v in q.items() if k not in ['windows','path']} for q in report],indent=2))
