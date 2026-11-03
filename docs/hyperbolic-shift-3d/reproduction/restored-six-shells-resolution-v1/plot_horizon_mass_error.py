"""Accepted apparent-horizon irreducible-mass errors; gaps and signs preserved."""
from pathlib import Path
import hashlib, io, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
OUT=HERE/'n16-n20-n24-history-snapshot'
prior=json.loads((HERE/'n16-n20/comparison.json').read_text())
receipt=json.loads((OUT/'snapshot-provenance.json').read_text())
data={};sources={}
for n,key,raw in [(16,'N16*','restored-six-shells'),(20,'N20','restored-six-shells-resolution')]:
    seg,=prior['cases'][key]['segments']
    name,record=next((k,v) for k,v in seg['artifacts'].items() if k.endswith('/puncture.history.csv'))
    path=ROOT/'output'/raw/'aurora'/name
    b=path.read_bytes(); assert len(b)==record['bytes'] and hashlib.sha256(b).hexdigest()==record['sha256']
    data[n]=np.genfromtxt(io.BytesIO(b),delimiter=',',names=True); sources[n]=dict(path=str(path.relative_to(ROOT)),**record)
name='output/puncture.history.csv';record=receipt['files'][name];b=(OUT/'snapshot'/name).read_bytes()
assert len(b)==record['bytes'] and hashlib.sha256(b).hexdigest()==record['sha256']
data[24]=np.genfromtxt(io.BytesIO(b),delimiter=',',names=True); sources[24]=record
summary=dict(definition='abs(M_irr/(1/3)-1)',reference_mass=1/3,domain='Accepted apparent-horizon irreducible mass only',sources=sources,cases={},qualifications=['Rejected horizon rows are NaN gaps, not zero errors or interpolated values.','Plot is absolute relative error; signed errors remain in CSV/JSON.','Near-zero dips can reflect crossing the reference mass, not sustained accuracy improvement.','N24 snapshot ends at9.233M; N16/N20 extend to10M.','Horizon angular lmax12 and ntheta24 are fixed across all volume resolutions.','N16 historical campaign penalty-policy failure remains recorded.'])
fig,ax=plt.subplots(figsize=(11,5.5));colors={16:'#238b45',20:'#2166ac',24:'#d95f02'}
for n,d in data.items():
    accepted=(d['horizon_found']==1)&np.isfinite(d['horizon_irreducible_mass'])&(d['horizon_irreducible_mass']>0)
    signed=np.where(accepted,3*d['horizon_irreducible_mass']-1,np.nan)
    absolute=np.abs(signed); plotted=np.where(absolute>0,absolute,np.nan)
    ax.semilogy(d['t_over_M'],plotted,'o-',color=colors[n],lw=1.3,ms=3,label=f'N{n}')
    np.savetxt(OUT/f'horizon-mass-error-n{n}.csv',np.column_stack([d['t_over_M'],d['horizon_found'],d['horizon_irreducible_mass'],signed,absolute,d['horizon_expansion_rms']]),delimiter=',',header='t_over_M,horizon_accepted,M_irreducible,signed_relative_mass_error,absolute_relative_mass_error,expansion_rms',comments='')
    idx=np.flatnonzero(accepted);assert len(idx)>0;i=idx[-1]
    summary['cases'][str(n)]=dict(rows=len(d),accepted_rows=int(accepted.sum()),exact_zero_accepted_errors=int(np.sum(accepted&(absolute==0))),last_recorded_time=float(d['t_over_M'][-1]),last_accepted_time=float(d['t_over_M'][i]),last_signed_relative_error=float(signed[i]),last_absolute_relative_error=float(absolute[i]),maximum_accepted_absolute_error=float(np.nanmax(absolute)))
ax.set(xlabel=r'$t/M$',ylabel=r'$|M_{\mathrm{irr}}/M-1|$',title='Apparent-horizon mass error: six-shell N16, N20 and N24',xlim=(0,10))
ax.grid(alpha=.2,which='both');ax.legend(ncol=3)
ax.axvline(data[24]['t_over_M'][-1],ls=':',lw=1,color='gray')
fig.tight_layout(rect=(0,.13,1,1))
fig.text(.5,.035,'Accepted surfaces only; gaps mark rejected finder results. Absolute relative error, with no artificial log floor.\nN24 snapshot through 9.233M (dotted line); near-zero dips may be mass-error sign crossings.',ha='center',fontsize=9)
fig.savefig(OUT/'horizon-mass-error.png',dpi=180);plt.close(fig)
(OUT/'horizon-mass-error.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary['cases'],indent=2))
