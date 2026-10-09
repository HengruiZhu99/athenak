#!/usr/bin/env python3
"""Independent Cartesian continuum-rate readback; source-only until authorized.

No radial assembly, projection, discrete KO/upwind term, spectrum or evolution.
The polynomial core oracle below is independent of the actual-kernel point API.
Only the analytic solid-basis coefficient DATA are shared with that API.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

P = Path(__file__).resolve().parents[2] / "boundary/total-j-finite-rb-control-20261009"
SUB = P / '../../continuum/constraint-propagation/immutable-constraint-propagation-20261009/subsidiary.hpp'
HELD = P / '../total-j-finite-rb-control-held-20261009/FINAL-ADDENDUM.md'
SUB_HASH = '3d840f2f731ec7fab34579ec2a0132acc4066311b5a4876ae9f484d41f78be54'
HELD_HASH = '69c13c353efd91f08315578f5c7dfaa6413dfed8be36262c1df22eb7dffab163'
Q_NAMES = ['H', 'Mx', 'My', 'Mz', 'Zx', 'Zy', 'Zz', 'Theta_physical']
TI, TK = [0,0,0,1,1,2], [0,1,2,1,2,2]
OFFSETS, FIRST, SECOND = [-2,-1,1,2], [1,-8,8,-1], [-1,16,16,-1]
DIRS = [(1.,0.,0.), tuple(v/math.sqrt(14) for v in (1,2,3)),
        tuple(v/math.sqrt(14) for v in (2,-3,1))]
KAPPA = 10.

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def norm(x): return math.sqrt(math.fsum(v*v for v in x))
def difference(a,b): return [u-v for u,v in zip(a,b)]
def errors(a,b):
    absolute = norm(difference(a,b))
    return {'absolute_l2': absolute, 'scaled_l2': absolute/max(1.,norm(a),norm(b)),
            'absolute_peak': max(map(abs,difference(a,b)), default=0.)}

# Sparse Cartesian polynomials.  Integer differentiation is exact algebra;
# archived solid-basis coefficients and their evaluation remain binary64.
def add(*items):
    terms = {}
    for a in items:
        for k,v in a.items(): terms.setdefault(k,[]).append(v)
    return {k:z for k,v in terms.items() if (z:=math.fsum(v)) != 0}
def scale(a,c): return {k:v*c for k,v in a.items() if v*c != 0}
def mul(a,b):
    terms = {}
    for p,x in a.items():
        for q,y in b.items():
            k=tuple(u+v for u,v in zip(p,q)); terms.setdefault(k,[]).append(x*y)
    return {k:z for k,v in terms.items() if (z:=math.fsum(v)) != 0}
def diff(a,i):
    out={}
    for p,v in a.items():
        if p[i]:
            q=list(p);q[i]-=1;out[tuple(q)]=v*p[i]
    return out
def lap(a): return add(*(diff(diff(a,i),i) for i in range(3)))
def evaluate(a,x): return math.fsum(v*math.prod(x[i]**p[i] for i in range(3)) for p,v in a.items())
def polyjet(a,x):
    return [evaluate(a,x)]+[evaluate(diff(a,i),x) for i in range(3)]+[
        evaluate(diff(diff(a,i),j),x) for i in range(3) for j in range(3)]

class Basis:
    def __init__(self, path):
        text=Path(path).read_text();block=text.split('inline constexpr Term terms[]={',1)[1].split('};',1)[0]
        self.terms=[(int(c),tuple(map(int,(x,y,z))),float(re_),float(im))
            for c,x,y,z,re_,im in re.findall(r'\{(\d+),\{(\d+),(\d+),(\d+)\},([^,]+),([^}]+)\}',block)]
        block=text.split('inline constexpr Record records[]={',1)[1].split('};',1)[0]
        self.records={tuple(map(int,v[:4])):tuple(map(int,v[4:]))
            for v in re.findall(r'\{(-?\d+),(-?\d+),(\d+),(\d+),(\d+),(\d+)\}',block)}
        self.channels={}
        for J in range(3):
            block=text.split(f'inline constexpr Channel channels{J}[]={{',1)[1].split('};',1)[0]
            self.channels[J]=[tuple(map(int,v)) for v in re.findall(r'\{(\d+),(\d+),(\d+)\}',block)]
        assert len(self.terms)==1057 and [len(self.channels[J]) for J in range(3)]==[8,16,20]
        rho={(2,0,0):1.,(0,2,0):1.,(0,0,2):1.};one={(0,0,0):1.}
        self.envelopes=[one,rho,mul(rho,rho),mul(mul(rho,rho),rho)]
        self.envelopes.append(add(one,scale(rho,1/3),scale(self.envelopes[2],-1/5),scale(self.envelopes[3],1/7)))
    def components(self,J,m,c,phase):
        _,spin,L=self.channels[J][c];start,count=self.records[J,m,spin,L]
        out=[{} for _ in range((1,3,9)[spin])]
        for component,p,re_,im in self.terms[start:start+count]:
            v=im if phase else re_
            if v: out[component]=add(out[component],{p:v})
        return out
    def nonzero(self,J,m,c,phase): return any(self.components(J,m,c,phase))
    def raw(self,J,m,c,phase,envelope):
        kind,_,_=self.channels[J][c]
        w=self.envelopes[envelope if envelope<4 else 4]
        comp=[mul(a,w) for a in self.components(J,m,c,phase)]
        out=[{} for _ in range(22)]
        if kind==0: out[18]=comp[0]
        elif kind==1: out[0]=scale(comp[0],-1/math.sqrt(3))
        elif kind==2: out[7]=comp[0]
        elif kind==3: out[17]=comp[0]
        elif kind in (4,5):
            start=19 if kind==4 else 14
            for i in range(3):out[start+i]=comp[i]
        else:
            start=1 if kind==6 else 8
            for q,(i,j) in enumerate(zip(TI,TK)):out[start+q]=comp[3*i+j]
        return out

def flat_constraints(a):
    g=[[{} for _ in range(3)] for _ in range(3)];A=[[{} for _ in range(3)] for _ in range(3)]
    for q,(i,j) in enumerate(zip(TI,TK)):g[i][j]=g[j][i]=a[1+q];A[i][j]=A[j][i]=a[8+q]
    H=add(scale(lap(a[0]),2),*(diff(diff(g[i][j],i),j) for i in range(3) for j in range(3)))
    M=[add(*(diff(A[i][j],j) for j in range(3)),scale(diff(add(a[7],scale(a[17],2)),i),-2/3)) for i in range(3)]
    Z=[scale(add(a[14+i],scale(add(*(diff(g[i][j],j) for j in range(3))),-1)),.5) for i in range(3)]
    return [H]+M+Z+[a[17]]

def flat_source(a):
    """Independent flat-core raw22 source, mu_inner=3/8, kappa=10."""
    f=[{} for _ in range(22)];g=[[{} for _ in range(3)] for _ in range(3)];A=[[{} for _ in range(3)] for _ in range(3)]
    for q,(i,j) in enumerate(zip(TI,TK)):g[i][j]=g[j][i]=a[1+q];A[i][j]=A[j][i]=a[8+q]
    divbeta=add(*(diff(a[19+i],i) for i in range(3)))
    divlambda=add(*(diff(a[14+i],i) for i in range(3)))
    f[0]=scale(add(a[7],scale(a[17],2),scale(divbeta,-1)),2/3)
    f[7]=add(scale(lap(a[18]),-1),scale(a[17],KAPPA))
    f[17]=add(lap(a[0]),scale(divlambda,.5),scale(a[17],-2*KAPPA))
    f[18]=scale(a[7],-3)
    trace=add(scale(lap(a[18]),-1),scale(lap(a[0]),2),divlambda)
    for q,(i,j) in enumerate(zip(TI,TK)):
        f[1+q]=add(scale(A[i][j],-2),diff(a[19+j],i),diff(a[19+i],j),scale(divbeta,-2/3) if i==j else {})
        f[8+q]=add(scale(diff(diff(a[18],i),j),-1),scale(lap(g[i][j]),-.5),
            scale(diff(diff(a[0],i),j),.5),scale(add(diff(a[14+j],i),diff(a[14+i],j)),.5),
            add(scale(lap(a[0]),.5),scale(trace,-1/3)) if i==j else {})
    for i in range(3):
        gamma=add(*(diff(g[i][j],j) for j in range(3)))
        f[14+i]=add(lap(a[19+i]),scale(diff(divbeta,i),1/3),scale(diff(a[7],i),-4/3),
            scale(diff(a[17],i),-2/3),scale(add(a[14+i],scale(gamma,-1)),-KAPPA))
        f[19+i]=scale(a[14+i],3/8)
    return f

def flat_rates(q):
    H=q[0];M=q[1:4];Z=q[4:7];theta=q[7]
    divM=add(*(diff(M[i],i) for i in range(3)))
    divZ=add(*(diff(Z[i],i) for i in range(3)))
    return [scale(divM,-2)]+[add(scale(diff(H,i),-.5),lap(Z[i]),scale(diff(divZ,i),-1),
        scale(diff(theta,i),2*KAPPA)) for i in range(3)]+[
        add(M[i],diff(theta,i),scale(Z[i],-KAPPA)) for i in range(3)]+[
        add(scale(H,.5),divZ,scale(theta,-2*KAPPA))]

def columns(basis,J,m,phase,kind):
    choices=[]
    for c,(field,_,_) in enumerate(basis.channels[J]):
        if not basis.nonzero(J,m,c,phase):continue
        if kind=='gauge' and field not in (0,4):continue
        if kind=='shell' and field in (0,4):continue
        choices.append(c)
    return choices

def specs(basis,Js,stage):
    for J in Js:
        for m in range(J+1):
            for phase in range(2):
                cs=columns(basis,J,m,phase,stage)
                for c in cs:
                    for env in (range(4) if stage=='core' else [4 if stage=='gauge' else 5]):
                        yield {'J':J,'m':m,'phase':phase,'channels':[[c,1.]],'envelope':env,'stage':stage}
                if cs and stage!='gauge':
                    yield {
                        'J':J,'m':m,'phase':phase,'channels':[[c,(-1.)**c/(c+1)] for c in cs],
                        'envelope':6 if stage=='core' else 5,'stage':stage,'mixed':True}

def cases(basis,Js,stage,rb):
    radii=[0.,.025,.049] if stage=='core' else [.15,.30,.50,.70,.90,.96,.975]+([.99] if rb==.995 else [])
    for spec in specs(basis,Js,stage):
        for r in radii:
            for direction_id,n in enumerate(DIRS[:1] if r==0 else DIRS):
                x=tuple(r*v for v in n)
                yield dict(spec,r=r,direction_id=direction_id,x=x)

def stencil(x,h):
    out={(0,0,0):x}
    for i in range(3):
        for o in OFFSETS:
            k=[0,0,0];k[i]=o;out[tuple(k)]=tuple(x[d]+h*k[d] for d in range(3))
    for i in range(3):
        for j in range(i):
            for a in OFFSETS:
                for b in OFFSETS:
                    k=[0,0,0];k[i]=a;k[j]=b
                    out[tuple(k)]=tuple(x[d]+h*k[d] for d in range(3))
    assert len(out)==61
    return out

def fd_jets(samples,h):
    center=samples[0,0,0];out=[]
    for f,value in enumerate(center):
        first=[];dd=[[0.]*3 for _ in range(3)]
        for i in range(3):
            k=[]
            for o in OFFSETS:
                v=[0,0,0];v[i]=o;k.append(tuple(v))
            # Center-subtracted, compensated sums are the same frozen stencils.
            first.append(math.fsum(w*(samples[z][f]-value) for z,w in zip(k,FIRST))/(12*h))
            dd[i][i]=math.fsum(w*(samples[z][f]-value) for z,w in zip(k,SECOND))/(12*h*h)
            for j in range(i):
                terms=[]
                for a,wa in zip(OFFSETS,FIRST):
                    for b,wb in zip(OFFSETS,FIRST):
                        z=[0,0,0];z[i]=a;z[j]=b
                        terms.append(wa*wb*(samples[tuple(z)][f]-value))
                dd[i][j]=dd[j][i]=math.fsum(terms)/(144*h*h)
        out.append([value]+first+[dd[i][j] for i in range(3) for j in range(3)])
    return out

class API:
    def __init__(self,exe,attempt):self.exe=exe;self.attempt=attempt;self.calls=[]
    def run(self,mode,rows,nout):
        ans=[]
        for start in range(0,len(rows),8192):
            chunk=rows[start:start+8192];index=len(self.calls);cmd=[str(self.exe),mode]
            inp=''.join(' '.join(format(v,'.17g') if isinstance(v,float) else str(v) for v in row)+'\n' for row in chunk)
            begin=time.monotonic();run=subprocess.run(cmd,input=inp,text=True,capture_output=True)
            stem=self.attempt/'calls'/f'{index:06d}';stem.with_suffix('.input').write_text(inp)
            stem.with_suffix('.stdout').write_text(run.stdout);stem.with_suffix('.stderr').write_text(run.stderr)
            item={'command':cmd,'rows':len(chunk),'seconds':time.monotonic()-begin,'returncode':run.returncode,
                  'input_sha256':sha(stem.with_suffix('.input')),'stdout_sha256':sha(stem.with_suffix('.stdout')),
                  'stderr_sha256':sha(stem.with_suffix('.stderr')),'stderr_bytes':len(run.stderr.encode())}
            self.calls.append(item)
            with (self.attempt/'calls.jsonl').open('a') as log: log.write(json.dumps(item,allow_nan=False)+'\n')
            if run.returncode or run.stderr:raise RuntimeError(f'API call {index} failed; receipt preserved')
            values=[[float(v) for v in line.split()] for line in run.stdout.splitlines()]
            if len(values)!=len(chunk) or any(len(v)!=nout or not all(map(math.isfinite,v)) for v in values):
                raise RuntimeError(f'API call {index} shape/nonfinite output; receipt preserved')
            ans.extend(values)
        return ans
    def manufactured(self,case,points):
        rows=[]
        for x in points:
            for c,_ in case['channels']:rows.append([case['J'],case['m'],c,case['phase'],case['envelope'],*x])
        values=self.run('--manufactured-rate-batch',rows,30);count=len(case['channels']);out=[]
        for p in range(len(points)):
            out.append([math.fsum(w*values[p*count+i][f] for i,(_,w) in enumerate(case['channels'])) for f in range(30)])
        return out
    def jets(self,mode,x,jets):return [*x,*itertools.chain.from_iterable(jets)]

def core_check(api,basis,case):
    a=[{} for _ in range(22)]
    for c,w in case['channels']:
        column=basis.raw(case['J'],case['m'],c,case['phase'],case['envelope'])
        a=[add(u,scale(v,w)) for u,v in zip(a,column)]
    q=flat_constraints(a);f=flat_source(a);expected=flat_rates(q);by_chain=flat_constraints(f);x=case['x']
    fv=[evaluate(v,x) for v in f];qv=[evaluate(v,x) for v in q];rate=[evaluate(v,x) for v in expected]
    oracle_chain=[evaluate(v,x) for v in by_chain]
    actual=api.manufactured(case,[x])[0]
    actual_rate=api.run('--constraint-rate-batch',[api.jets('',x,[polyjet(v,x) for v in f])],8)[0]
    predicted=api.run('--subsidiary-batch',[api.jets('',x,[polyjet(v,x) for v in q])],8)[0]
    comparisons={'source':errors(actual[:22],fv),'constraints':errors(actual[22:],qv),
        'oracle_chain_identity':errors(oracle_chain,rate),'actual_exact_source_rate':errors(actual_rate,rate),
        'subsidiary_exact_polynomial_rate':errors(predicted,rate)}
    passed=all(v['scaled_l2']<=5e-11 for v in comparisons.values())
    return {'case':case,'passed':passed,'method':'exact Cartesian polynomial jets; no FD across r=.05',
        'comparisons':comparisons,'actual_source':actual[:22],'oracle_source':fv,'actual_constraints':actual[22:],
        'oracle_constraints':qv,'actual_rate':actual_rate,'subsidiary_rate':predicted,'oracle_rate':rate}

def sequence_info(rows):
    inc=[norm(difference(b,a)) for a,b in zip(rows,rows[1:])]
    scales=[max(1.,norm(a),norm(b)) for a,b in zip(rows,rows[1:])]
    ratios=[inc[i]/inc[i+1] if inc[i+1] else None for i in range(3)]
    orders=[math.log2(v) if v and v>0 else None for v in ratios]
    evidence=any(inc[i]>1e-10*scales[i] and inc[i+1]>1e-10*scales[i+1]
                 and ratios[i]>=8 for i in range(3))
    small=all(v/s<=2e-7 for v,s in zip(inc,scales))
    return {'increments_absolute_l2':inc,'increments_scaled_l2':[v/s for v,s in zip(inc,scales)],
        'increment_ratios':ratios,'observed_orders':orders,'fourth_order_evidence':evidence,
        'order_status':'classified_at_least_one_pair' if evidence else 'within_tolerance_order_unclassified' if small else 'unresolved',
        'last_increment_scaled_l2':inc[-1]/scales[-1],
        'richardson_last':[ (16*b-a)/15 for a,b in zip(rows[-2],rows[-1])]}

def fd_check(api,case,rb):
    x=case['x'];h0=min(.002,(rb-case['r'])/4);hs=[h0/2**i for i in range(5)]
    grids=[stencil(x,h) for h in hs];points=list(dict.fromkeys(p for grid in grids for p in grid.values()))
    if not all(norm(p)<rb for p in points):raise RuntimeError('FD sample is not strictly inside rb')
    values=api.manufactured(case,points);lookup=dict(zip(points,values));actual_rows=[];sub_rows=[]
    for grid,h in zip(grids,hs):
        sampled={k:lookup[p] for k,p in grid.items()};j=fd_jets(sampled,h)
        actual_rows.append(api.jets('',x,j[:22]));sub_rows.append(api.jets('',x,j[22:]))
    actual=api.run('--constraint-rate-batch',actual_rows,8)
    sub=api.run('--subsidiary-batch',sub_rows,8);aq=sequence_info(actual);sq=sequence_info(sub)
    initial=lookup[x][22:];base=lookup[x][:22];checks={}
    if case['stage']=='gauge':
        checks={'initial_constraints':norm(initial)<=5e-11,
                'final_zero_rate':norm(actual[-1])<=2e-7,'extrapolated_zero_rate':norm(aq['richardson_last'])<=2e-7,
                'actual_sequence_resolved':aq['order_status']!='unresolved'}
    else:
        checks={'final_actual_subsidiary':errors(actual[-1],sub[-1])['scaled_l2']<=2e-7,
            'extrapolated_actual_subsidiary':errors(aq['richardson_last'],sq['richardson_last'])['scaled_l2']<=2e-7,
            'actual_last_increment':aq['last_increment_scaled_l2']<=2e-7,
            'subsidiary_last_increment':sq['last_increment_scaled_l2']<=2e-7,
            'actual_sequence_resolved':aq['order_status']!='unresolved','subsidiary_sequence_resolved':sq['order_status']!='unresolved'}
    return {'case':case,'passed':all(checks.values()),'checks':checks,'h':hs,'initial_constraints':initial,
        'actual_source_center':base,'actual_rate_sequence':actual,'subsidiary_rate_sequence':sub,
        'actual_sequence':aq,'subsidiary_sequence':sq,'final_error':errors(actual[-1],sub[-1]),
        'extrapolated_error':errors(aq['richardson_last'],sq['richardson_last']),
        'absolute_error_sequence':[difference(a,b) for a,b in zip(actual,sub)]}

def plan():
    return {'scope':'pointwise continuum constraint rates only; no radial assembly/spectrum/evolution',
        'physical_order':Q_NAMES,'kappa_input':10,'kappa2':0,'reference':{'S':1,'a':.5,'geometry':[.05,.95]},
        'gauge':'C0 physical-P plus frozen spatial-norm control xi2; unchanged complete reference jets',
        'radii':[.15,.30,.50,.70,.90,.96,.975],'extra_r_if_rb_0995':.99,'directions':DIRS,
        'h':'min(.002,(rb-r)/4)/2^j, j0..4','stencil':'61 Cartesian points; standard fourth-order first/diagonal second and composed mixed',
        'tolerances':{'core':5e-11,'gauge_initial':5e-11,'final_rate':2e-7,'order_floor':1e-10},
        'no_omega_rescaling':True,'norm_scope':'Euclidean point comparisons and sample RMS/peaks only; no integrated energy assertion',
        'source_authorization':'requires explicit API-admission JSON and matching executable/API hashes'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--exe',type=Path)
    parser.add_argument('--authorization',type=Path);parser.add_argument('--rb',type=float,default=.98)
    parser.add_argument('--stage',choices=['core','gauge','shell','all'],default='all')
    parser.add_argument('--j',type=int,nargs='+',default=[0,1,2]);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.execute:print(json.dumps(plan(),indent=2));return
    if args.rb not in (.98,.995) or any(J not in (0,1,2) for J in args.j):raise ValueError('undeclared rb/J')
    if not args.exe or not args.authorization:raise ValueError('source-gated executable and authorization required')
    admission=json.loads(args.authorization.read_text())
    if admission.get('continuum_constraint_rate_api_admitted') is not True:raise ValueError('API not admitted')
    exe=args.exe.resolve();api_header=P/'constraint_rate_api.hpp'
    if sha(exe)!=admission['executable_sha256'] or sha(api_header)!=admission['api_sha256']:
        raise ValueError('API/executable hash mismatch')
    source_gate=Path(admission['source_configuration_gate'])
    if json.loads(source_gate.read_text()).get('passed_source_configuration_derivative_gate') is not True:
        raise ValueError('configuration derivative/source gate failed')
    if sha(SUB)!=SUB_HASH or sha(HELD)!=HELD_HASH:raise ValueError('frozen mathematical recipe changed')
    paths=[Path(__file__),api_header,P/'all_m_data.hpp',SUB,HELD,exe,args.authorization,source_gate,
        P/'actual_bridge.cpp',P/'baseline_dual_spatial.hpp',P/'spatial_dual.hpp',P/'configuration_rows.hpp',
        P/'radial_bridge.cpp',P/'build-release-latest.json']
    before={str(p.resolve()):sha(p) for p in paths}
    launch_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip()
    driver_command=[sys.executable,*sys.argv]
    output=args.output or P/'constraint-rate-attempts';output.mkdir(parents=True,exist_ok=True)
    attempt=output/f'attempt-{time.time_ns()}';attempt.mkdir();(attempt/'calls').mkdir()
    for p in (Path(__file__),api_header,P/'all_m_data.hpp'):shutil.copyfile(p,attempt/p.name)
    write_json(attempt/'plan.json',plan());write_json(attempt/'source-before.json',before)
    api=API(exe,attempt);basis=Basis(P/'all_m_data.hpp');start=time.monotonic();results=[];error=None
    try:
        with (attempt/'cases.jsonl').open('w') as stream:
            for stage in (['core','gauge','shell'] if args.stage=='all' else [args.stage]):
                for case in cases(basis,args.j,stage,args.rb):
                    row=core_check(api,basis,case) if stage=='core' else fd_check(api,case,args.rb)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();results.append(row)
                    if len(results)%50==0:print(f'{stage}: {len(results)} saved cases',flush=True)
                    if not row['passed']:raise RuntimeError(f'failed {stage} case {len(results)}; exact data saved')
    except Exception as e:error={'type':type(e).__name__,'message':str(e)}
    write_json(attempt/'calls.json',api.calls)
    after={str(p.resolve()):sha(p) for p in paths}
    groups={}
    for stage in ('core','gauge','shell'):
        rows=[r for r in results if r['case']['stage']==stage]
        group={'cases':len(rows),'passed':all(r['passed'] for r in rows) if rows else None}
        for key in ('actual_constraints','initial_constraints','actual_rate','oracle_rate','subsidiary_rate'):
            values=[r[key] for r in rows if key in r]
            if values:group[key+'_sample_rms']=[math.sqrt(math.fsum(v[i]**2 for v in values)/len(values)) for i in range(8)]
            if values:group[key+'_sample_peak']=[max(abs(v[i]) for v in values) for i in range(8)]
        for key in ('actual_rate_sequence','subsidiary_rate_sequence','absolute_error_sequence'):
            values=[r[key][-1] for r in rows if key in r]
            if values:group[key+'_final_sample_rms']=[math.sqrt(math.fsum(v[i]**2 for v in values)/len(values)) for i in range(8)]
            if values:group[key+'_final_sample_peak']=[max(abs(v[i]) for v in values) for i in range(8)]
        groups[stage]=group
    report={'passed_requested_continuum_rate_gate':error is None and before==after,
        'driver_version':'v2 append-only call logging; stricter absolute zero initial-gauge check; unchanged source/FD formulas',
        'driver_command':driver_command,'launch_HEAD':launch_head,
        'requested_stage':args.stage,'requested_J':args.j,'rb':args.rb,'seconds':time.monotonic()-start,
        'case_count':len(results),'calls':len(api.calls),'error':error,'source_before':before,'source_after':after,
        'source_unchanged':before==after,'groups':groups,'plan':plan()}
    write_json(attempt/'receipt.json',report);print(json.dumps(report,indent=2),flush=True)
    if not report['passed_requested_continuum_rate_gate']:sys.exit(1)

if __name__=='__main__':main()
