#!/usr/bin/env python3
"""Measure global-periodic semidiscrete reduction/curl defects with compiled PointRHS.

Derivatives and signed KO are reconstructed independently on the whole periodic
array. This is not a raw production u_rhs/stage-budget dump. The same compiled
continuum kernel is used; the mesh FD/KO consumer is separately oracle-tested.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.dont_write_bytecode=True
from intrinsic_diagnostics import derivative
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)

def families(u):
    return np.array([np.concatenate([u[20+i:21+i],u[23+i:24+i],u[26+5*i:31+5*i],u[41+3*i:44+3*i]]) for i in range(3)])

def potentials(u):return np.concatenate([u[0:1],(u[0]*u[1])[None],u[2:7],u[7:10]])

def ko(f,spacing,order,amplitude):
    radius=order//2+1;out=np.zeros_like(f)
    for d,h in enumerate(spacing):
        if f.shape[-1-d]==1:continue
        for j in range(-radius,radius+1):out-=amplitude*(-1.)**j*math.comb(2*radius,radius+j)*np.roll(f,-j,axis=-1-d)/(2**(2*radius)*h)
    return out

records=[]
for order in [2,4,6]:
    ladders={eps:[] for eps in [0.,.3]}
    for n in [16,32,64]:
        d=a.output/f'fd{order}-n{n}';d.mkdir()
        j,i=np.indices((n,n));phase=2*np.pi*((i+.5)/n+(j+.5)/n)
        u=np.array([(0.6 if v==0 else 1.4 if v==1 else 0)+.02*np.sin(phase+.17*v)/(1+.03*v) for v in range(50)])[:,None]
        spacing=[2*np.pi/n,3*np.pi/n,1.];D=lambda f,i:derivative(f,i,spacing,order)
        du=np.array([D(u,i) for i in range(3)]);rate=u[0]*u[1]
        inputs=np.concatenate([np.moveaxis(u,0,-1).reshape(-1,50),np.moveaxis(du,(0,1),(-2,-1)).reshape(-1,150),rate.reshape(-1,1),np.full((n*n,1),2.),np.ones((n*n,1))],axis=1)
        np.savetxt(d/'input.txt',inputs,fmt='%.17g',header=str(n*n),comments='')
        command=[str(a.binary.resolve()),str((d/'input.txt').resolve()),str((d/'output.txt').resolve()),str((d/'kokkos.txt').resolve())]
        with (d/'run.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        assert result.returncode==0
        raw=np.loadtxt(d/'output.txt');assert np.isfinite(raw).all()
        rhs=np.moveaxis(raw[:,:50].reshape(1,n,n,50),-1,0)
        g=families(u);phi=potentials(u);e=g-np.array([D(phi,i) for i in range(3)])
        omega=np.array([[D(g[j],i)-D(g[i],j) for j in range(3)] for i in range(3)])
        dbeta=np.array([D(u[7:10],i) for i in range(3)]);dlambda=np.array([D(rate,i) for i in range(3)])
        transport=np.zeros_like(e);lie=np.zeros_like(omega)
        for k in range(3):
            transport+=u[7+k]*D(e,k);lie+=u[7+k]*D(omega,k)
            for i in range(3):
                transport[i]+=dbeta[i,k]*e[k]
                for j in range(3):lie[i,j]+=dbeta[i,k]*omega[k,j]+dbeta[j,k]*omega[i,k]
        chain=np.array([u[1]*D(u[0],i)+u[0]*D(u[1],i)-D(u[0]*u[1],i) for i in range(3)])
        for eps in [0.,.3]:
            udot=rhs+ko(u,spacing,order,eps);gdot=families(udot)
            phidot=np.concatenate([udot[0:1],(u[1]*udot[0]+u[0]*udot[1])[None],udot[2:7],udot[7:10]])
            edot=gdot-np.array([D(phidot,i) for i in range(3)])
            injection=edot-transport+rate*e-ko(e,spacing,order,eps)
            curl_dot=np.array([[D(gdot[j],i)-D(gdot[i],j) for j in range(3)] for i in range(3)])
            curl_injection=curl_dot-lie+rate*omega-ko(omega,spacing,order,eps)
            for i in range(3):
                for j in range(3):curl_injection[i,j]+=dlambda[i]*e[j]-dlambda[j]*e[i]
            imax=np.max(abs(injection),axis=(0,2,3,4));omax=np.max(abs(curl_injection),axis=(0,1,3,4,5))
            entry=dict(n=n,KO=eps,reduction_injection_max=float(imax.max()),curl_injection_max=float(omax.max()),reduction_family_max=imax.tolist(),curl_family_max=omax.tolist(),lapse_chain_product_max=float(np.max(abs(chain))))
            ladders[eps].append(entry)
            np.savez(d/f'defect-ko{eps}.npz',state=u,rhs=udot,reduction=e,omega=omega,injection=injection,curl_injection=curl_injection,lapse_chain_minus_product=chain)
        (d/'manifest.json').write_text(json.dumps(dict(command=command,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),input_sha256=hashlib.sha256((d/'input.txt').read_bytes()).hexdigest()),indent=2)+'\n')
    for eps,entries in ladders.items():
        rates={}
        for key in ['reduction_injection_max','curl_injection_max','lapse_chain_product_max']:
            values=np.array([e[key] for e in entries]);rates[key]=np.log2(values[:-1]/values[1:]).tolist()
        record=dict(order=order,KO=eps,entries=entries,observed_orders=rates,minimum_order=order-1,status='PASS' if min(v for rr in rates.values() for v in rr)>=order-1 else 'FAIL')
        records.append(record);(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
        assert record['status']=='PASS'
