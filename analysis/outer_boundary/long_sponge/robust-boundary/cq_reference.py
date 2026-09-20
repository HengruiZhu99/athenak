"""Causal BDF2-CQ full-state DtN reference, frozen planar Fourier mode only."""
from rational_dtn import *
from scipy.signal import fftconvolve

def weights(k,cfg,dt,n,exponent=12.):
 rho=np.exp(-exponent/n);z=rho*np.exp(-2j*np.pi*np.arange(n)/n);s=(1.5-2*z+.5*z*z)/dt
 F=np.array([exterior(v,k,cfg)for v in s]);w=np.fft.ifft(F,axis=0)/(rho**np.arange(n))[:,None,None]
 return w,rho,F

def response(k,cfg,dt,tf,pulseT=1024.):
 n=2**int(np.ceil(np.log2(tf/dt+1)));w,rho,F=weights(k,cfg,dt,n)
 t=np.arange(n)*dt;g=(1-np.exp(-t/pulseT))**4*np.exp(-t/pulseT)
 q=np.zeros((n,10));q[:,0]=g*1e-8;q[:,6]=g*2e-8
 dn=np.stack([sum(fftconvolve(w[:,i,j],q[:,j],mode='full')[:n]for j in [0,6])for i in range(10)],axis=1)
 zero=np.stack([sum(fftconvolve(w[:,i,j],np.zeros(n),mode='full')[:n]for j in [0,6])for i in range(10)],axis=1)
 # Value of the finite weight power series at independently chosen points.
 checks=[]
 for z in [.85,.95,.975,.97*np.exp(.02j)]:
  got=np.einsum('n,nij->ij',z**np.arange(n),w);want=exterior((1.5-2*z+.5*z*z)/dt,k,cfg)
  checks.append(float(np.linalg.norm(got-want)/np.linalg.norm(want)))
 return dict(dt=dt,n=n,rho=rho,t=t,dn=dn,g=g,zero_max=float(abs(zero).max()),series_errors=checks,finite=bool(np.isfinite(w).all()and np.isfinite(dn).all()))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=Config(alpha=.999665896620774,chi=.9993319048666159,beta_n=.0003157233701617295);k=np.pi/256
 # dt levels compare only first 16000M, before significant inverse-FFT
 # amplification and well before periodic alias wraparound near the end.
 runs=[response(k,cfg,dt,65536)for dt in [16,8,4]];fine=runs[-1];rows=[]
 for run in runs:
  ids=run['t']<=16000;stride=int(run['dt']/fine['dt']);ref=fine['dn'][::stride][:len(run['t'])];den=np.linalg.norm(ref[ids]);dif=np.linalg.norm((run['dn']-ref)[ids])/den
  rows.append({key:run[key]for key in ['dt','n','rho','zero_max','series_errors','finite']}|{'relative_error_vs_dt4_until16000':float(dif),'peak_nonzero_response':float(abs(run['dn'][ids]).max())})
 out={'scope':'Time-domain operator-response and causality reference only. BDF2 convolution quadrature requires matched interior/interface time discretization. No AthenaK RK3, nonlinear, grid corner, or AMR stability claim.','config':asdict(cfg),'k':k,'rows':rows,'difference_ratio':rows[0]['relative_error_vs_dt4_until16000']/rows[1]['relative_error_vs_dt4_until16000']}
 a.output.write_text(json.dumps(out,indent=2)+'\n');np.savez(a.output.with_suffix('.npz'),t=fine['t'],dn=fine['dn'],g=fine['g']);print(json.dumps(out),flush=True)
