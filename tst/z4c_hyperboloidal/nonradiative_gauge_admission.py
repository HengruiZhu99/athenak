#!/usr/bin/env python3
"""Manufacture lapse and physical K directly from the independent metric oracle."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import nonradiative_analytic as fixture

def data(t,x,family,parameter=0,height=1,mass=1/3):
    g,dg,_=fixture.compact(t,x,family,parameter=parameter,height=height,mass=mass)
    gamma=g[:,1:,1:];inverse=np.linalg.inv(g);ig=np.linalg.inv(gamma)
    assert np.min(np.linalg.eigvalsh(gamma))>0
    alpha=np.sqrt(-1/inverse[:,0,0]);beta=np.einsum('qij,qj->qi',ig,g[:,0,1:])
    di=-np.einsum('qam,qemn,qnb->qeab',inverse,dg,inverse)
    da=.5*alpha[:,None]**3*di[:,:,0,0]
    db=np.einsum('qij,qej->qei',ig,dg[:,:,0,1:]-np.einsum('qeij,qj->qei',dg[:,:,1:,1:],beta))
    lie=np.einsum('qk,qkij->qij',beta,dg[:,1:,1:,1:])+np.einsum('qkj,qik->qij',gamma,db[:,1:])+np.einsum('qik,qjk->qij',gamma,db[:,1:])
    omega=(1-np.sum(x*x,axis=1))/2
    K=omega/(2*alpha)*np.einsum('qij,qij->q',ig,-dg[:,0,1:,1:]+lie)+3*np.einsum('qi,qi->q',beta,x)/alpha
    return np.column_stack((x,alpha,da[:,0],da[:,1:],beta,K,np.linalg.det(gamma)**(-1/3)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
    x=np.array([.94,.96,.98])[:,None]*np.array([.3,.4,np.sqrt(.75)])[None,:]
    mink=data(.2,x,'schwarzschild',mass=0);np.testing.assert_allclose(mink[:,-2],-3,atol=2e-14)
    cases=[('schwarzschild',0,1,.2),('rotation',.1,1,.2),('rotation',.8,1,.2),('retarded_bounce',2,1,5/3),('gauge',.01,1,.2),('gauge',1,1,.2),('gauge',10,16,.2)]
    rows=[]
    for i,(family,parameter,height,t) in enumerate(cases):
        values=data(t,x,family,parameter,height);rows.extend(np.column_stack((np.full(len(x),i),values)))
    np.savetxt(a.output/'oracle-input.txt',rows,fmt='%.17g')
    with (a.output/'oracle-input.txt').open() as stream,(a.output/'equation.csv').open('w') as log:
        subprocess.run([str(a.binary.resolve())],stdin=stream,stdout=log,check=True)
    result=np.loadtxt(a.output/'equation.csv',delimiter=',',skiprows=1)
    report={'assumption':'outer direct gauge, B=0 (decoupled); native stationary source and kappa=.25 retained','cases':[]}
    for i,case in enumerate(cases):
        surface,_=fixture.grid();g=fixture.compact(case[3],surface,case[0],parameter=case[1],height=case[2])[0]
        inverse=np.linalg.inv(g);alpha=np.sqrt(-1/inverse[:,0,0]);ig=np.linalg.inv(g[:,1:,1:])
        beta=np.einsum('qij,qj->qi',ig,g[:,0,1:]);speed=np.sqrt(np.einsum('qi,qij,qj->q',surface,ig,surface))
        report['cases'].append(dict(family=case[0],parameter=case[1],height=case[2],time=case[3],values=result[result[:,0]==i].tolist(),
            endpoint_alpha_defect=float(np.max(abs(alpha-1))),endpoint_A_shift_defect=float(np.max(np.linalg.norm(beta+speed[:,None]*surface,axis=1)))))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
