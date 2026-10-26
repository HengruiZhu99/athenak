#!/usr/bin/env python3
"""Independent outer-CMC Z recount from an unchanged research state checkpoint.

Restricted to mass-positive runs with layer ending at .75, unit reference
scri radius, and physical outer radius .9<R<=1. Here the
initial twice-conformal metric is exactly Cartesian identity and Lambda=0,
so direct differencing of g-I agrees with the native reconstruction policy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('receipt',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();receipt=json.loads(a.receipt.read_text());payload=receipt['checkpoint_payload']
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    assert digest==payload['sha256'] and a.checkpoint.stat().st_size==payload['bytes']
    with a.checkpoint.open('rb') as f:
        assert f.read(16) in [b'ATHENA_CSP_CK01!',b'ATHENA_CSP_CK02!']
        tag=f.read(64).decode();signature=struct.unpack('<17d',f.read(136))
        fingerprint,time,cycle=struct.unpack('<Qdq',f.read(24));packs=[]
        for region in range(2):
            shape=struct.unpack('<5Q',f.read(40));offset=f.tell()
            packs.append(np.memmap(a.checkpoint,mode='r',dtype='<f8',offset=offset,shape=shape))
            f.seek(int(np.prod(shape))*8,1)
        assert f.tell()+8==a.checkpoint.stat().st_size
    assert tag==receipt['extra_parameters']['cubed_sphere/conformal_checkpoint_tag']
    L,R,stretch,curvature,r0,r1,mass=signature[:7]
    assert .9<R<=1 and r1==.75 and mass>0 and signature[-1]==8
    domain=a.receipt.with_name('cubed_sphere_conformal_domain.json')
    if R!=1 or domain.exists():
        descriptor=json.loads(domain.read_text())
        assert descriptor['domain_radius']==R and descriptor['reference_scri_radius']==1
    closure=int(signature[12]);interior=closure>=100
    assert closure in [2,3,4,5,102,103,104,105], 'unknown closure signature'
    # Six-node polynomial derivatives, independently checked by moments.
    weights=np.array([[-1/5,5/4,-10/3,5,-5,137/60],
                      [1/20,-1/3,1,-2,13/12,1/5],
                      [-1/30,1/4,-1,1/3,1/2,-1/20]])
    for depth,w in enumerate(weights):
        for power in range(6):
            assert abs(np.dot(w,(np.arange(6)-5+depth).astype(float)**power)-(1 if power==1 else 0))<1e-12
    state=packs[1];assert state.shape[0]==6 and state.shape[1]==25
    ng=4;na=state.shape[2]-2*ng;nr=state.shape[4]-2*ng
    assert state.shape[3]==state.shape[2]
    # Enough whole radial lines to cover r>.9 at every angular location.
    tail=int(np.ceil(.125/(R-L*np.sqrt(3))*nr))+4
    m,k,j,i=np.meshgrid(np.arange(6),np.arange(ng,ng+na),np.arange(ng,ng+na),
                        np.arange(ng+nr-tail,ng+nr),indexing='ij')
    q0=(i-ng+.5)/nr;u=-1+2*(j-ng+.5)/na;v=-1+2*(k-ng+.5)/na
    t=q0 if stretch==0 else -np.expm1(-stretch*q0)/(-np.expm1(-stretch))
    d=(1-t)*L+t*R/np.sqrt(1+u*u+v*v)
    x=np.zeros(m.shape+(3,));J=np.zeros(m.shape+(3,3))
    for patch in range(6):
        aa=patch//2;bb=(aa+1)%3;cc=(aa+2)%3;sign=1 if patch%2==0 else -1
        x[patch,...,aa]=sign*d[patch];x[patch,...,bb]=d[patch]*u[patch];x[patch,...,cc]=d[patch]*v[patch]
    radius=np.linalg.norm(x,axis=-1)
    # The allocated tail includes unused inner rows on coarse grids. Check
    # actual stencil positions for the reported collar, not those unused rows.
    collar=radius>.9
    for axis,spacing in enumerate([1/nr,2/na,2/na]):
        for shift in [-2,-1,1,2]:
            coords=[q0,u,v];coords[axis]=coords[axis]+shift*spacing
            ss,uu,vv=coords
            tt=ss if stretch==0 else -np.expm1(-stretch*ss)/(-np.expm1(-stretch))
            rr=(1-tt)*L*np.sqrt(1+uu*uu+vv*vv)+tt*R
            assert np.min(rr[collar])>r1,'collar stencil enters nonconstant reference metric'
    if interior:
        ss=(nr-5.5)/nr
        tt=ss if stretch==0 else -np.expm1(-stretch*ss)/(-np.expm1(-stretch))
        assert np.min((1-tt)*L*np.sqrt(1+u*u+v*v)+tt*R)>r1
    for patch in range(6):
        aa=patch//2;bb=(aa+1)%3;cc=(aa+2)%3;sign=1 if patch%2==0 else -1
        e=np.zeros(3);e[aa]=sign
        rr=radius[patch];dd=d[patch];den=R*dd/rr-L
        dden=R*(e/rr[...,None]-dd[...,None]*x[patch]/rr[...,None]**3)
        dt=(e*den[...,None]-(dd-L)[...,None]*dden)/den[...,None]**2
        factor=1 if stretch==0 else -np.expm1(-stretch)/(stretch*(1+np.expm1(-stretch)*t[patch]))
        J[patch,...,0,:]=dt*np.asarray(factor)[...,None] if stretch else dt
        eb=np.zeros(3);eb[bb]=1;ec=np.zeros(3);ec[cc]=1
        J[patch,...,1,:]=(eb*dd[...,None]-x[patch,...,bb,None]*e)/dd[...,None]**2
        J[patch,...,2,:]=(ec*dd[...,None]-x[patch,...,cc,None]*e)/dd[...,None]**2
    g=np.zeros(m.shape+(3,3));Dlog=np.zeros(m.shape+(3,3,3))
    for field,(aa,bb) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)],start=1):
        g[...,aa,bb]=g[...,bb,aa]=state[m,field,k,j,i]
        for axis,spacing in enumerate([1/nr,2/na,2/na]):
            def sample(shift):
                idx=[i,j,k];idx[axis]=idx[axis]+shift
                return state[m,field,idx[2],idx[1],idx[0]]-(1 if aa==bb else 0)
            derivative=((sample(-2)-sample(2))/12+2*(sample(1)-sample(-1))/3)/spacing
            if interior and axis==0:
                mm,kk,jj=m[...,0],k[...,0],j[...,0]
                anchor=state[mm,field,kk,jj,ng+nr-1]
                for depth,w in enumerate(weights):
                    value=np.zeros_like(anchor)
                    for node,coefficient in enumerate(w):
                        value+=coefficient*(state[mm,field,kk,jj,ng+nr-6+node]-anchor)
                    derivative[...,-1-depth]=value/spacing
            Dlog[...,axis,aa,bb]=Dlog[...,axis,bb,aa]=derivative
    dg=np.einsum('...Ad,...Aij->...dij',J,Dlog)
    inverse=np.linalg.inv(g)
    connection=np.einsum('...ia,...a->...i',inverse,
        np.einsum('...jk,...jak->...a',inverse,dg)-.5*np.einsum('...jk,...ajk->...a',inverse,dg))
    lam=np.stack([state[m,f,k,j,i] for f in [14,15,16]],axis=-1)
    z=.5*np.einsum('...ij,...j->...i',g,lam-connection)
    z2=state[m,0,k,j,i]*np.einsum('...ij,...i,...j->...',inverse,z,z)
    assert np.all(np.isfinite(z2)) and np.min(z2)>=0
    volume=(1/nr)*(2/na)**2/np.abs(np.linalg.det(J))
    def summarize(mask):
        if not np.any(mask):return {'cells':0}
        peak=np.unravel_index(np.argmax(np.where(mask,z2,-1)),z2.shape)
        return {'cells':int(np.count_nonzero(mask)),'volume':float(np.sum(volume[mask])),
                'Z_conformal_l2':float(np.sqrt(np.sum(volume[mask]*z2[mask])/np.sum(volume[mask]))),
                'Z_conformal_linf':float(np.sqrt(z2[peak])),'max_xyz':x[peak].tolist(),
                'max_logical_depth':int(nr-1-(i[peak]-ng)),
                'max_angular_indices':[int(j[peak]-ng),int(k[peak]-ng)],
                'max_patch':int(m[peak])}
    collar=radius>.9;last=radius>=.9375
    seam=np.minimum.reduce([j-ng,ng+na-1-j,k-ng,ng+na-1-k])
    summaries={n:summarize(mask) for n,mask in [('collar',collar),('last_bin',last),
               ('collar_seam_2_cells',collar&(seam<2)),('collar_face_interior',collar&(seam>=2))]}
    for depth in range(8):summaries['outer_depth_'+str(depth)]=summarize(collar&(i-ng==nr-1-depth))
    native=list(csv.DictReader(a.receipt.with_name('cubed_sphere_conformal_regions.csv').open()))
    match=[r for r in native if float(r['time'])==time and r['subset']=='fixed_outer_collar']
    assert len(match)==1
    native_z=float(match[0]['Z_conformal_l2']);relative=abs(summaries['collar']['Z_conformal_l2']/native_z-1)
    assert relative<1e-9,(relative,summaries['collar'],native_z)
    result={'checkpoint_sha256':digest,'binary_tag':tag,'reference_fingerprint':fingerprint,
            'radial_closure':'six-active-node interior' if interior else 'centered with polynomial ghosts',
            'time':time,'cycle':cycle,'signature':signature,'domain_radius':R,
            'reference_scri_radius':1,'native_collar_Z_relative_difference':relative,
            'regions':summaries,'scope':'Independent NumPy inverse/Jacobian/connection evaluation on the unchanged checkpoint in the constant-reference outer CMC region. Exact native norm agreement required. No evolution or closure repair.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
