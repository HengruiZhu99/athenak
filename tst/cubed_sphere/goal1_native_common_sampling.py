"""Passive tensor interpolation of native readbacks; never supplies evolved fields.

Use existing Cartesian components and native derivative readbacks as scalars of
position. The analytic cubed-sphere inverse matches the verified fixed map.
"""
import numpy as np


def sample(fine, values, coarse, B, A, Nr, width=6, cube=.15, outer=.7):
    assert width in [4,6] and min(B,A,Nr)>=width
    assert len(fine)==B**3+6*A*A*Nr and len(values)==len(fine)
    shape=values.shape[1:];src=values.reshape(len(values),-1);out=np.empty((len(coarse),src.shape[1]));worst=0.
    for bulk in [True,False]:
        select=np.flatnonzero((coarse[:,13]==0)==bulk);pos=coarse[select,:3]
        if bulk:
            xi=((pos+cube)/(2*cube/B)-.5)[:,::-1];sizes=[B,B,B];patch=None
        else:
            axis=np.argmax(abs(pos),axis=1);patch=2*axis+(pos[np.arange(len(pos)),axis]<0);d=abs(pos[np.arange(len(pos)),axis]);u=pos[np.arange(len(pos)),(axis+1)%3]/d;v=pos[np.arange(len(pos)),(axis+2)%3]/d;s=(d-cube)/(outer/np.sqrt(1+u*u+v*v)-cube);xi=np.column_stack(((v+1)*A/2-.5,(u+1)*A/2-.5,s*Nr-.5));sizes=[A,A,Nr]
        indices=[];weights=[]
        for k,n in enumerate(sizes):
            start=np.clip(np.floor(xi[:,k]).astype(int)-(width-2)//2,0,n-width);ind=start[:,None]+np.arange(width);w=np.ones_like(ind,dtype=float)
            for j in range(width):
                for l in range(width):
                    if j!=l:w[:,j]*=(xi[:,k]-ind[:,l])/(ind[:,j]-ind[:,l])
            worst=max(worst,float(np.max(abs(w.sum(axis=1)-1))));indices.append(ind);weights.append(w)
        for offset in range(0,len(select),96):
            sl=slice(offset,offset+96);k,j,i=[q[sl]for q in indices];wk,wj,wi=[q[sl]for q in weights];address=(k[:,:,None,None]*sizes[1]+j[:,None,:,None])*sizes[2]+i[:,None,None,:]
            if not bulk:address+=B**3+patch[sl,None,None,None]*A*A*Nr
            w=wk[:,:,None,None]*wj[:,None,:,None]*wi[:,None,None,:];out[select[sl]]=np.einsum('pabc,pabcf->pf',w,src[address])
    assert worst<1e-12,worst
    return out.reshape((len(coarse),)+shape)
