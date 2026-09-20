"""Independent physical constraints and linear Weyl tensors of frozen states."""
import numpy as np

def physical(u,ux,uxx,k,chi):
 def ds(v,j):return (ux if j==0 else 1j*k*u if j==1 else 0*u)[v]
 def dds(v,i,j):
  if i==0 and j==0:return uxx[v]
  if i==2 or j==2:return 0j
  if i==1 and j==1:return -k*k*u[v]
  return 1j*k*ux[v]
 def hh(i,j):return [(1,1)] if i==j==0 else [(2,1)] if i==j==1 else [(1,-1),(2,-1)] if i==j==2 else [(3,1)] if set([i,j])=={0,1} else [(4,1)] if set([i,j])=={0,2} else [(5,1)]
 def aa(i,j):return [(f+7,w) for f,w in hh(i,j)]
 Q=np.array([u[13+i]-sum(sum(w*ds(f,j)for f,w in hh(i,j))for j in range(3))for i in range(3)])
 H=chi*sum(sum(w*dds(f,i,j)for f,w in hh(i,j))for i in range(3)for j in range(3))+2*sum(dds(0,i,i)for i in range(3))
 M=np.array([sum(sum(w*ds(f,j)for f,w in aa(i,j))for j in range(3))-2/3*(ds(6,i)+2*ds(7,i))for i in range(3)])
 E=np.array([[chi/2*(sum(sum(w*dds(f,l,i)for f,w in hh(l,j))+sum(w*dds(f,l,j)for f,w in hh(l,i))for l in range(3))-sum(sum(w*dds(f,l,l)for f,w in hh(i,j))for l in range(3)))+.5*(dds(0,i,j)+(sum(dds(0,l,l)for l in range(3))if i==j else 0))for j in range(3)]for i in range(3)])
 B=np.zeros((3,3),complex)
 for i in range(3):
  for j in range(3):
   for p in range(3):
    for q in range(3):B[i,j]+=np.sqrt(chi)*(i-p)*(p-q)*(q-i)/2*(sum(w*ds(f,p)for f,w in aa(q,j))+((ds(6,p)+2*ds(7,p))/3 if q==j else 0))
 return {'Theta':float(abs(u[7])),'Q_max':float(max(abs(Q))),'H':float(abs(H)),'M_max':float(max(abs(M))),'electric_Weyl':float(np.linalg.norm(E)),'magnetic_Weyl':float(np.linalg.norm(B)),'lapse':float(abs(u[16])),'shift_max':float(max(abs(u[17:20])))}

def coordinate_mode(cfg,k):
 """Exact mixed-metric + normal-shift Dirichlet counterexample at lambda=beta*k."""
 d=np.array([k,1j*k,0]);av=np.array([-4.,3j,0]);C=-6*k
 h0=np.outer(d,av)+np.outer(av,d)+np.outer([1,0,0],d)+np.outer(d,[1,0,0])-2/3*np.eye(3)*C
 h1=2*np.outer(d,d);A=cfg.beta_n/cfg.alpha*np.outer(d,d)
 u=np.zeros(20,complex);ux=np.zeros(20,complex);uxx=np.zeros(20,complex)
 u[0]=4*cfg.chi*k;ux[0]=k*u[0];uxx[0]=k*k*u[0]
 for (i,j),f in zip([(0,0),(1,1),(0,1),(0,2),(1,2)],range(1,6)):
  u[f]=h0[i,j];ux[f]=k*h0[i,j]+h1[i,j];uxx[f]=k*k*h0[i,j]+2*k*h1[i,j]
  u[f+7]=A[i,j];ux[f+7]=k*A[i,j];uxx[f+7]=k*k*A[i,j]
 return u,ux,uxx
