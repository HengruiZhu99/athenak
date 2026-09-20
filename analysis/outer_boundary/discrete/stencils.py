"""Frozen Cartesian scalar Z4c principal operator with actual p-only face updates.

State [Khat,Theta,A_nn,Gamma_x,chi,h_nn,alpha,beta_x], variable-major.
Metric and A have transverse components -h_nn/2 and -A_nn/2.
No radial 1/r terms; this is not the variable-coefficient trumpet operator.
"""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('characteristics',Path(__file__).with_name('characteristic_reference.py'))
char=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(char)

def weights(x, point, derivative=1):
    x=np.asarray(x,dtype=float)-point
    return np.linalg.solve(np.array([x**k/math.factorial(k) for k in range(len(x))]),np.eye(len(x))[derivative])

def extension(n,degree=3,ng=4):
    E=np.zeros((n+2*ng,n)); E[ng:ng+n]=np.eye(n)
    for pos in range(-ng,n+ng):
        if 0<=pos<n: continue
        ids=np.arange(degree+1) if pos<0 else np.arange(n-degree-1,n)
        E[pos+ng,ids]=weights(ids,pos,0)
    return E

def stencil(n,h,offsets,values,E):
    S=np.zeros((n,E.shape[0])); ng=(E.shape[0]-n)//2
    for i in range(n): S[i,i+ng+np.asarray(offsets)]=values
    return S@E/h

def active_derivative(n,h,order):
    D=np.zeros((n,n)); count=order+1
    for i in range(n):
        first=min(max(i-order//2,0),n-count); ids=np.arange(first,first+count)
        D[i,ids]=weights(ids,i)/h
    return D

def operators(n,h,degree=3,beta=0.,diss=.5,ng=4):
    E=extension(n,degree,ng)
    if ng==4:
        D=stencil(n,h,range(-3,4),[-1/60,3/20,-3/4,0,3/4,-3/20,1/60],E)
        Dxx=stencil(n,h*h,range(-3,4),[1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90],E)
        up_offsets=np.arange(-2,5); up_values=np.array([1/30,-2/5,-7/12,4/3,-1/2,2/15,-1/60])
        ko_offsets=np.arange(-4,5); ko_values=np.array([1,-8,28,-56,70,-56,28,-8,1])*(-diss/256)
    elif ng==2:
        D=stencil(n,h,[-1,0,1],[-.5,0,.5],E)
        Dxx=stencil(n,h*h,[-1,0,1],[1,-2,1],E)
        up_offsets=np.arange(3); up_values=np.array([-1.5,2,-.5])
        ko_offsets=np.arange(-2,3); ko_values=np.array([1,-4,6,-4,1])*(-diss/16)
    else: raise ValueError(ng)
    if beta<0: up_offsets=-up_offsets;up_values=-up_values
    adv=beta*stencil(n,h,up_offsets,up_values,E)
    KO=stencil(n,h,ko_offsets,ko_values,E)
    return D,Dxx,adv,KO,E
