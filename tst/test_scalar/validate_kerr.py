#!/usr/bin/env python3
"""Compare a real seven-block scalar with an independent regular m=2 oracle."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as leg
from scipy.interpolate import PchipInterpolator
from fit_ringdown import fit
from scalar_evidence import snapshot,history,waveform
from validate_schwarzschild import norms
from reference_interpolation import lagrange6

def reconstruct(data,profile,sampler='pchip'):
    rho_grid=np.unique(profile['rho']);z_grid=np.unique(profile['z']);nr,na=len(rho_grid),len(z_grid)
    if len(profile)!=nr*na:raise RuntimeError('Oracle is not a tensor grid')
    if not np.array_equal(profile['z'].reshape(nr,na),np.tile(z_grid,(nr,1))):raise RuntimeError('Unexpected oracle ordering')
    inverse=np.linalg.inv(leg.legvander(z_grid,na-1));tables={}
    for key,re,im in [('W','W_real','W_imag'),('v','v_real','v_imag'),('Dr','Dr_real','Dr_imag')]:
        values=(profile[re]+1j*profile[im]).reshape(nr,na)
        tables[key]=values
    xyz=data[:,:3];radius=np.linalg.norm(xyz,axis=1);mask=radius>=1.2;d=data[mask];radius=radius[mask];n=xyz[mask]/radius[:,None]
    if radius.min()<rho_grid[0] or radius.max()>rho_grid[-1]+1e-10:raise RuntimeError('Oracle does not cover comparison domain')
    exact=np.empty((len(d),5))
    for lo in range(0,len(d),65536):
        sl=slice(lo,lo+65536);r=radius[sl];v=n[sl];z=v[:,2];angular=leg.legvander(z,na-1)@inverse
        derivative=np.column_stack([leg.legval(z,leg.legder(np.eye(na)[:,k])) for k in range(na)])@inverse
        interpolate=lambda values:lagrange6(rho_grid,values,r) if sampler=='lagrange6' else PchipInterpolator(rho_grid,values.real,axis=0)(r)+1j*PchipInterpolator(rho_grid,values.imag,axis=0)(r)
        sampled={key:interpolate(values) for key,values in tables.items()}
        W=np.sum(angular*sampled['W'],axis=1);Wz=np.sum(derivative*sampled['W'],axis=1);Wr=np.sum(angular*sampled['Dr'],axis=1);V=np.sum(angular*sampled['v'],axis=1)
        a=v[:,0]+1j*v[:,1];A=a*a;exact[sl,0]=2*(A*W).real;exact[sl,1]=2*(A*V).real
        for k in range(3):exact[sl,2+k]=2*(A*Wr*v[:,k]+A*Wz*((k==2)-z*v[:,k])/r+2*a*W*((k==0)+1j*(k==1)-a*v[:,k])/r).real
    return d,exact

def compare(data,profile,sampler='pchip'):
    d,exact=reconstruct(data,profile,sampler)
    return dict(psi=norms(d[:,4]-exact[:,0],d[:,3]),v=norms(d[:,5]-exact[:,1],d[:,3]),Dpsi=norms(d[:,9:12]-exact[:,2:5],d[:,3]),q=norms(d[:,6:9]-exact[:,2:5],d[:,3]),reduction=norms(d[:,6:9]-d[:,9:12],d[:,3]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs=3,type=Path,required=True);p.add_argument('--oracle-waveform',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-sampler',choices=['pchip','lagrange6'],default='pchip');a=p.parse_args()
    if a.output.exists():raise RuntimeError('Existing evidence is immutable')
    parameters=[json.loads((d/'parameters.json').read_text()) for d in a.runs]
    if any(p['M']!=1 or p['a_over_M']!=.5 for p in parameters):raise RuntimeError('Initial Kerr admission is M1, spin .5 only')
    ns=[p['n'] for p in parameters];complete=all((d/'complete.txt').exists() for d in a.runs);records=[];checks=[];oracle_files=[a.oracle_waveform,a.expected]
    for t in [2,6,12,100]:
        profile=np.genfromtxt(str(a.oracle_waveform)+f'-profile-{t}.csv',delimiter=',',names=True)
        oracle_files.append(Path(str(a.oracle_waveform)+f'-profile-{t}.csv'))
        try:errors=[compare(snapshot(d,t),profile,a.reference_sampler) for d in a.runs]
        except FileNotFoundError:continue
        records.append(dict(time=t,errors=errors))
        for field in ['psi','v','Dpsi','reduction']:
            for norm in ['L2','Linf']:
                values=[r[field][norm] for r in errors];orders=[math.log(values[k]/values[k+1])/math.log(ns[k+1]/ns[k]) for k in [0,1]]
                checks.append(dict(time=t,field=field,norm=norm,errors=values,orders=orders,passed=values[0]>values[1]>values[2] and min(orders)>=1.5))
    modes=json.loads(a.expected.read_text())['modes'];co=next(m for m in modes if m['a_over_M']==.5 and m['m']==2);counter=next(m for m in modes if m['a_over_M']==.5 and m['m']==-2)
    reference=np.genfromtxt(a.oracle_waveform,delimiter=',',names=True);fits=[];waves=[];reductions=[]
    for d in a.runs:
        w=waveform(d);ref=reference[reference['time']<=w['time'][-1]+1e-12]
        if not np.array_equal(w['time'],ref['time']):raise RuntimeError('Independent waveform times differ')
        z=w['real22']+1j*w['imag22'];r=ref['real22']+1j*ref['imag22'];error=float(np.linalg.norm(z-r)/np.linalg.norm(r));waves.append(dict(run=str(d),relative_L2=error,passed=error<=.02))
        local=[]
        for lo,hi in [(35,65),(40,70),(45,75)]:
            select=(w['time']>=lo)&(w['time']<=hi)
            if np.count_nonzero(select)<40:continue
            f=fit(w['time'][select],z[select],True);f['window']=[lo,hi];f['relative_errors']=[abs(f['omega_real']-co['Momega_real'])/co['Momega_real'],abs(f['decay']+co['Momega_imag'])/(-co['Momega_imag']),abs(f['counterrotating_real']-counter['Momega_real'])/counter['Momega_real'],abs(f['counterrotating_decay']+counter['Momega_imag'])/(-counter['Momega_imag'])];f['passed']=f['success'] and all(e<=b for e,b in zip(f['relative_errors'],[.02,.05,.02,.05]));local.append(f)
        spreads={key:(max(f[key] for f in local)-min(f[key] for f in local))/np.mean([f[key] for f in local]) for key in ['omega_real','decay','counterrotating_real','counterrotating_decay']} if len(local)==3 else {}
        fits.append(dict(run=str(d),fits=local,window_spreads=spreads,passed=len(local)==3 and all(f['passed'] for f in local) and all(e<=.02 for e in spreads.values())))
        initial=snapshot(d,0);mask=np.linalg.norm(initial[:,:3],axis=1)>=1.2;scale=float(np.max(np.linalg.norm(initial[mask,6:9],axis=1)));exterior=[float(r['reduction_Linf']) for (t,region),r in history(d).items() if region=='exterior'];maximum=max(exterior) if exterior else None
        reductions.append(dict(run=str(d),normalized_max=maximum/scale if maximum is not None else None,all_half_time_reports_checked=bool(exterior),passed=bool(exterior) and maximum/scale<=.02))
    result=dict(stage='kerr',runs=[str(d) for d in a.runs],resolutions=ns,complete=complete,passed=False,field_checks=checks,records=records,QNM=fits,independent_waveform=waves,reduction=reductions,field_checks_passed=complete and len(checks)==32 and all(c['passed'] for c in checks),finest_accuracy_passed=complete and waves[-1]['passed'] and fits[-1]['passed'] and reductions[-1]['passed'],scope='Numerical checks only; complete CPU admission also requires time/inner/outer controls and a resolved independent oracle.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result['oracle']=dict(waveform=str(a.oracle_waveform.resolve()),files={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in oracle_files})
    result['reference_sampler']=a.reference_sampler;result['interpolator_sha256']=hashlib.sha256(Path(__file__).with_name('reference_interpolation.py').read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['complete','field_checks_passed','finest_accuracy_passed']},indent=2))
if __name__=='__main__':main()
