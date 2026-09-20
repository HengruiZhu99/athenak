"""Flat linear physical H/M and connection Q=2Z diagnostics; sixth-order FD."""
from pathlib import Path
import json,numpy as np

def diagnostics(u,h=32.,angle_y=np.pi/8):
 N=len(u);f=np.fft.fft(u,axis=0);a=2*np.pi*np.fft.fftfreq(N)
 d1=lambda z:1j*(1.5*np.sin(z)-.3*np.sin(2*z)+np.sin(3*z)/30)/h
 d2=lambda z:(2*np.cos(3*z)/90-.3*np.cos(2*z)+3*np.cos(z)-49/18)/h**2
 x,y=d1(a),d1(angle_y);xx,yy=d2(a),d2(angle_y)
 H=xx*f[:,1]+2*x*y*f[:,3]+yy*f[:,2]+2*(xx+yy)*f[:,0]
 Q=np.stack([f[:,13]-x*f[:,1]-y*f[:,3],f[:,14]-x*f[:,3]-y*f[:,2],f[:,15]-x*f[:,4]-y*f[:,5]],axis=1)
 K=f[:,6]+2*f[:,7]
 M=np.stack([x*f[:,8]+y*f[:,10]-2*x*K/3,x*f[:,10]+y*f[:,9]-2*y*K/3,x*f[:,11]+y*f[:,12]],axis=1)
 fields={'Theta':u[:,7], 'H':np.fft.ifft(H), 'M':np.fft.ifft(M,axis=0), 'Q':np.fft.ifft(Q,axis=0), 'Z':np.fft.ifft(Q,axis=0)/2}
 i=(N-32)//2;out={}
 for name,v in fields.items():
  point=abs(v)if v.ndim==1 else np.linalg.norm(v,axis=1)
  out[name]={'full_L2':float(np.linalg.norm(v)),'full_max':float(point.max()),'interior_L2':float(np.linalg.norm(v[i:i+32])),'interior_max':float(point[i:i+32].max()),'max_normal_x':float((np.argmax(point)-N/2+.5)*h)}
 return out
if __name__=='__main__':
 p=Path(__file__).parent;z=np.load(p/'coupled-rk3-final-dt0.3.npz');initial=diagnostics(z['initial']);final=diagnostics(z['u']);ratios={name:{key:final[name][key]/initial[name][key]for key in ['full_L2','full_max','interior_L2','interior_max']}for name in initial}
 out={'scope':'Flat linearized physical Hamiltonian H=partial_i partial_j h_ij+2 Laplacian chi, momentum M_i=partial_j A_ij−2/3 partial_i(Khat+2Theta), Q_i=Gamma_i−partial_j h_ij, Z_i=Q_i/2. Same sixth-order D1/D2; D2 is not replaced by D1 squared. D2 is independent of D1 squared, so H+div Q need not exactly reproduce the discrete Theta RHS identity. No nonlinear K/A products or curvature-background terms. Complex Fourier tangent uses modal amplitude norms, not a real-tangent spatial max. L2 keys denote unweighted discrete l2 without sqrt(dx) or proper-volume factors. Ratios use the same zero-extended periodic initial/final states, including extension derivative terms.','time_initial':0,'time_final':50000.4,'initial':initial,'final':final,'ratios':ratios};(p/'linear-constraints.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['ratios']),flush=True)
