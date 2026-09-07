#!/usr/bin/env python3
"""Analytic curved conformal/sheared primary constraints and auxiliary independence."""
import argparse
import json
from pathlib import Path
import numpy as np
from intrinsic_diagnostics import diagnostics,derivative
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[]
for order in [2,4,6]:
    errors=[];h_errors=[];m_errors=[]
    for n in [16,32,64]:
        y=(np.arange(n)+.5)*2*np.pi/n
        y=np.broadcast_to(y[None,:,None],(1,n,4));u=np.zeros((50,1,n,4))
        u[0]=1+.1*np.cos(y);u[1]=1/u[0];u[4]=.15*np.cos(y)
        u[10]=.03*np.sin(y);u[11]=.02;u[14]=-.02
        spacing=[.25,2*np.pi/n,1.]
        values=diagnostics(u,spacing,order)
        dw=-.1*np.sin(y);ddw=-.1*np.cos(y)
        h=(2/3)*u[10]**2-2*.02**2+4*u[0]*ddw-6*dw*dw
        m=np.zeros((3,1,n,4));m[1]=3*.02*dw/u[0]-(2/3)*.03*np.cos(y)
        h_error=float(np.max(abs(values['H_physical'][0]-h)));m_error=float(np.max(abs(values['M_physical']-m)))
        h_errors.append(h_error);m_errors.append(m_error);error=max(h_error,m_error)
        # Physical constraints must not read C, Z or any auxiliary.
        poisoned=u.copy();poisoned[16:]=np.arange(34)[:,None,None,None]*.07+np.sin(y)
        other=diagnostics(poisoned,spacing,order)
        assert np.array_equal(values['H_physical'],other['H_physical'])
        assert np.array_equal(values['M_physical'],other['M_physical'])
        assert np.max(abs(values['alpha_M_physical']-values['M_physical']))<2e-12
        # All ten auxiliary families, with a seeded x-directed residual varying in y.
        potentials=np.concatenate([u[0:1],(u[0]*u[1])[None],u[2:7],u[7:10]])
        amp=np.arange(1,11)/1000
        expected=np.zeros((3,10,1,n,4));expected[0]=amp[:,None,None,None]*np.cos(y)
        g=np.array([derivative(potentials,d,spacing,order) for d in range(3)])+expected
        for d in range(3):
            u[20+d]=g[d,0];u[23+d]=g[d,1];u[26+5*d:31+5*d]=g[d,2:7];u[41+3*d:44+3*d]=g[d,7:10]
        v=diagnostics(u,spacing,order)
        assert np.max(abs(v['reduction'].reshape(expected.shape)-expected))<2e-12
        exact_curl=np.zeros_like(expected);exact_curl[0]=-derivative(expected[0],1,spacing,order)
        assert np.max(abs(v['curl'].reshape(expected.shape)-exact_curl))<2e-12
        errors.append(error);np.savez(a.output/f'fd{order}-n{n}.npz',state=u,physical_h=values['H_physical'],physical_m=values['M_physical'],expected_h=h,expected_m=m)
    orders=np.log2(np.array(errors[:-1])/errors[1:])
    h_orders=np.log2(np.array(h_errors[:-1])/h_errors[1:]);m_orders=np.log2(np.array(m_errors[:-1])/m_errors[1:])
    record=dict(order=order,H_errors=h_errors,M_errors=m_errors,H_orders=h_orders.tolist(),M_orders=m_orders.tolist(),required_min_order=order-.6,status='PASS' if min(min(h_orders),min(m_orders))>=order-.6 else 'FAIL')
    records.append(record);(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
    assert record['status']=='PASS'
