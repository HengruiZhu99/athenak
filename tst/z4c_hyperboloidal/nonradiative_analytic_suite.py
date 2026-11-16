#!/usr/bin/env python3
"""Bounded paper-parameter and donor-row matrix on two allocated CPU ranks."""
import hashlib,json,time,traceback
from pathlib import Path
import nonradiative_analytic as a
import nonradiative_sampling as s

def run(root,out,rank):
    tasks=[]
    def exact(name,family,parameter=0,**kw):
        tasks.append((name,lambda: a.extract_case(family,parameter,out/name,**kw)))
    for mass in (0,1/3):
        exact('minkowski' if mass==0 else 'schwarzschild','schwarzschild',mass=mass,duration=.4,dt=.01,lmax=8)
    for k in range(1,9):
        exact(f'rotation-{k/10:.1f}','rotation',k/10,duration=.4,dt=.01,lmax=8)
    exact('rotation-long','rotation',.8,duration=8/3,dt=.05/3,lmax=8)
    exact('retarded-bounce-period','retarded_bounce',2,duration=40/3,dt=.1/3,lmax=8)
    for amplitude in (.01,.1,1,10):
        exact(f'gauge-{amplitude}','gauge',amplitude,duration=.4,dt=.01,lmax=8)
    for dt,lmax in ((.01,4),(.005,4),(.01,8)):
        exact(f'conformal-{dt}-L{lmax}','schwarzschild',conformal_warp=.1,duration=.4,dt=dt,lmax=lmax)
    exact('angular-clock','schwarzschild',time_warp=.04,duration=.4,dt=.01,lmax=8)
    for n in (12,16,20):
        for family,par,at in [('schwarzschild',0,.2),('rotation',.1,.2),('rotation',.8,.2),('retarded_bounce',2,5/3),('gauge',.01,.2),('gauge',10,.2)]:
            name=f'rows-N{n}-{family}-{par}'
            tasks.append((name,lambda n=n,f=family,p=par,t=at,name=name:s.run(root/'layouts'/f'N{n}',out/name,f,p,at=t)))
    for dt in (.001,.004):
        name=f'rows-N12-gauge10-dt{dt}'
        tasks.append((name,lambda dt=dt,name=name:s.run(root/'layouts/N12',out/name,'gauge',10,at=.2,dt=dt)))
    report={'rank':rank,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('nonradiative*.py')},'cases':[]}
    start=time.monotonic()
    if rank==0:report['finite_physical']=a.finite_checks()
    failures=[]
    for index,(name,fn) in enumerate(tasks):
        if index%2!=rank:continue
        before=time.monotonic()
        try:
            result=fn();result['name']=name;result['seconds']=time.monotonic()-before
        except Exception as error:
            traceback.print_exc();result={'name':name,'error':str(error)};failures.append(name)
        report['cases'].append(result);report['seconds']=time.monotonic()-start
        (out/f'analytic-rank-{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,result,flush=True)
    report['failures']=failures;report['complete']=True
    (out/f'analytic-rank-{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def gauge_followup(root,out,rank):
    """Retain failed H=1 admission; test a separately declared H=16 chart."""
    out.mkdir(exist_ok=True);tasks=[]
    for n in (12,16,20):
        for family,par in [('schwarzschild',0),('gauge',.01),('gauge',.1),('gauge',1),('gauge',10)]:
            name=f'rows-N{n}-{family}-{par}-H16'
            tasks.append((name,lambda n=n,f=family,p=par,name=name:s.run(root/'layouts'/f'N{n}',out/name,f,p,at=.2,height=16)))
    for dt in (.001,.004):
        name=f'rows-N12-gauge10-H16-dt{dt}'
        tasks.append((name,lambda dt=dt,name=name:s.run(root/'layouts/N12',out/name,'gauge',10,at=.2,dt=dt,height=16)))
    for family,par in [('rotation',.8),('gauge',.01)]:
        name=f'rows-N20-{family}-{par}'
        tasks.append((name,lambda f=family,p=par,name=name:s.run(root/'layouts/N20',out/name,f,p,at=.2)))
    for family in ('schwarzschild','gauge'):
        name=f'exact-{family}-H16'
        tasks.append((name,lambda f=family,name=name:a.extract_case(f,10,out/name,duration=.4,dt=.01,lmax=8,height=16)))
    for n in (12,16,20):
        name=f'rows-N{n}-minkowski'
        tasks.append((name,lambda n=n,name=name:s.run(root/'layouts'/f'N{n}',out/name,mass=0)))
    report={'rank':rank,'cases':[],'complete':False};start=time.monotonic()
    for index,(name,fn) in enumerate(tasks):
        if index%2!=rank:continue
        try:result=fn();result['name']=name
        except Exception as error:traceback.print_exc();result={'name':name,'error':str(error)}
        report['cases'].append(result);report['seconds']=time.monotonic()-start
        (out/f'followup-rank-{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (out/f'followup-rank-{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
