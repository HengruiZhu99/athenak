#!/usr/bin/env python3
"""HELD direct literal-RWM MP dual + independent Fraction closed oracle."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def guard(rows):
    for row in rows:
        if Path(row['path']).stat().st_size!=row['bytes'] or sha(row['path'])!=row['sha256']:
            raise RuntimeError('Protected input drift: '+row['path'])
def rows(path):
    with Path(path).open() as stream:
        for number,line in enumerate(stream,1):
            yield number,json.loads(line,parse_int=lambda token:-0.0 if token=='-0' else int(token))
def frac(x):return Fraction(x)
def pow2(n):return Fraction(1<<n) if n>=0 else Fraction(1,1<<(-n))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',required=True)
    parser.add_argument('--recipe',required=True);parser.add_argument('--attempt',required=True);args=parser.parse_args()
    if sys.flags.optimize or not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError('Require isolated unoptimized -I -B Python')
    auth=read(args.authorization);recipe=read(args.recipe)
    if not auth.get('far_complete_dual_local_execution_admitted'):
        raise RuntimeError('No exact far-dual local authorization')
    if auth['recipe_sha256']!=sha(args.recipe) or auth['source_index_sha256']!=sha(recipe['source_index']):
        raise RuntimeError('Root source/recipe authorization mismatch')
    inputs=read(recipe['input_pins']);source_files=read(recipe['source_index'])['files'];guard(inputs);guard(source_files)
    for name,value in recipe['environment'].items():
        if os.environ.get(name)!=value:raise RuntimeError('Environment guard: '+name)
    registry=read(recipe['case_registry']);assert registry['counts']['records']==2373
    # Only the separately released scientific oracle stage may import mpmath.
    sys.path.insert(0,recipe['mpmath_parent'])
    import mpmath as mp
    if str(Path(mp.__file__).resolve())!=recipe['mpmath_init']:raise RuntimeError('Unexpected mpmath')

    class Dual:
        def __init__(self,v=0,d=0):self.v,self.d=mp.mpf(v),mp.mpf(d)
        def __add__(self,x):x=ad(x);return Dual(self.v+x.v,self.d+x.d)
        __radd__=__add__
        def __neg__(self):return Dual(-self.v,-self.d)
        def __sub__(self,x):return self+-ad(x)
        def __rsub__(self,x):return ad(x)+-self
        def __mul__(self,x):x=ad(x);return Dual(self.v*x.v,self.d*x.v+self.v*x.d)
        __rmul__=__mul__
        def __truediv__(self,x):x=ad(x);return Dual(self.v/x.v,(self.d*x.v-self.v*x.d)/(x.v*x.v))
        def __rtruediv__(self,x):return ad(x)/self
    def ad(x):return x if isinstance(x,Dual) else Dual(x)
    def atom(x):return Dual(x[0],x[1])
    def field(u):
        return {k:atom(u[k]) if k in ('alpha','chi','P','Theta') else
                [[atom(v) for v in row] for row in u[k]] if k in ('g','beta_d','A') else
                [[[atom(v) for v in row] for row in plane] for plane in u[k]] if k=='g_d' else
                [[[[atom(v) for v in row] for row in plane] for plane in cube] for cube in u[k]] if k=='g_dd' else
                [atom(v) for v in u[k]] for k in u}
    def inverse(g):
        determinant=g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])-g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])+g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0])
        result=[]
        for i in range(3):
            line=[]
            for j in range(3):
                rr=[k for k in range(3) if k!=j];cc=[k for k in range(3) if k!=i]
                cofactor=((-1)**(i+j))*(g[rr[0]][cc[0]]*g[rr[1]][cc[1]]-g[rr[0]][cc[1]]*g[rr[1]][cc[0]])
                line.append(cofactor/determinant)
            result.append(line)
        assert determinant.v>0
        return result
    def raw_terms(u,c,od):
        a,x,b=u['alpha'],u['chi'],u['beta'];gi=inverse(u['g'])
        V=[[a*a*x*gi[i][j] for j in range(3)] for i in range(3)]
        L=[[V[i][j]-b[i]*b[j] for j in range(3)] for i in range(3)]
        regular=[[b[j]*u['alpha_d'][j] for j in range(3)]]
        pole=[[-a*a*u['P']]+[-a*b[j]*od[j] for j in range(3)]+[-a*L[j][k]*c[0][j][k] for j in range(3) for k in range(3)]]
        for i in range(3):
            regular.append([a*a*x*u['Lambda'][i]]+[b[j]*u['beta_d'][j][i] for j in range(3)]+
                [a*a*gi[i][j]*u['chi_d'][j]/2 for j in range(3)]+[-a*x*gi[i][j]*u['alpha_d'][j] for j in range(3)])
            pole.append([2*V[i][j]*od[j] for j in range(3)]+
                [-L[j][k]*(c[i+1][j][k]+b[i]*c[0][j][k]) for j in range(3) for k in range(3)])
        return regular,pole
    def target(row):
        u,h=field(row['input']),field(row['reference']);od=[atom(v) for v in row['Omega_d']]
        conn=[[[atom(v) for v in line] for line in plane] for plane in row['connection']]
        R,S=raw_terms(u,conn,od);Rh,Sh=raw_terms(h,conn,od);ratio=u['alpha']/h['alpha'];omega=atom(row['Omega'])
        terms=[R[0]+[-ratio*t for t in Rh[0]]]+[R[i]+[-t for t in Rh[i]] for i in range(1,4)]
        terms += [S[0]+[-ratio*t for t in Sh[0]]]+[S[i]+[-t for t in Sh[i]] for i in range(1,4)]
        terms += [terms[i]+[t/omega for t in terms[4+i]] for i in range(4)]
        result=[sum(values,Dual()) for values in terms]
        diagnostic=[(sum(abs(t.v) for t in values),sum(abs(t.d) for t in values)) for values in terms]
        return result,diagnostic
    maxima={};failures=[];counts={};FD=[];tiny_count=0;legacy_nonfinite=0;legacy_bit_rows=0;calls={}
    def require(ok,label):
        if not ok:failures.append({'check':label})
    def error(x,y):return abs(x-y)/max(mp.mpf(1),abs(x),abs(y))
    def update(name,x,y,tolerance,label,relative=False):
        e=abs(x-y)/abs(y) if relative and y else error(x,y)
        maxima[name]=max(maxima.get(name,mp.mpf(0)),e)
        if not(mp.isfinite(x) and mp.isfinite(y) and e<=mp.mpf(tolerance)):
            failures.append({'check':name,'label':label,'error':str(e),'got':str(x),'target':str(y)})
    def seed_check(got,wanted,label):
        # Fixed source-only declaration: exact rational seed formula, 2e-14
        # relative for nonzero rounded field seeds, exact zero otherwise.
        actual=frac(got)
        require(actual==0 if wanted==0 else abs(actual-wanted)<=Fraction(2,10**14)*abs(wanted),label)
    def verify_seed(row,seed):
        u=row['input'];xa=Fraction(seed['xi_alpha']);xc=Fraction(seed['xi_chi'])
        za=list(map(Fraction,seed['zeta_alpha']));zc=list(map(Fraction,seed['zeta_chi']))
        seed_check(u['alpha'][1],xa*frac(u['alpha'][0]),'registered alpha seed')
        seed_check(u['chi'][1],xc*frac(u['chi'][0]),'registered chi seed')
        for j in range(3):
            seed_check(u['alpha_d'][j][1],xa*frac(u['alpha_d'][j][0])+frac(u['alpha'][0])*za[j],'registered alpha gradient seed')
            seed_check(u['chi_d'][j][1],xc*frac(u['chi_d'][j][0])+frac(u['chi'][0])*zc[j],'registered chi gradient seed')
            seed_check(u['beta'][j][1],Fraction(seed['beta_value_dot'][j]),'registered beta seed')
            seed_check(u['Lambda'][j][1],Fraction(seed['Lambda_value_dot'][j]),'registered Lambda seed')
            for i in range(3):seed_check(u['beta_d'][j][i][1],Fraction(seed['beta_d_dot'][j][i]),'registered beta derivative seed')
        seed_check(u['P'][1],Fraction(seed['P_value_dot']),'registered physical P seed')
        seed_check(u['Theta'][1],Fraction(seed['Theta_value_dot']),'registered Theta seed')
        e=list(map(Fraction,registry['e']));diagonal=[Fraction(1,32),Fraction(-1,32),Fraction(0)]
        for i in range(3):
            for j in range(3):
                wanted=frac(u['g'][i][j][0])*(diagonal[i]+diagonal[j]) if seed['metric_value_dot']!='0' else Fraction(0)
                seed_check(u['g'][i][j][1],wanted,'registered metric seed')
                unused=seed['unconsumed_jets_dot']!='0'
                seed_check(u['A'][i][j][1],e[i]*e[j]/64 if unused else Fraction(0),'registered A seed')
                for k in range(3):
                    seed_check(u['g_d'][k][i][j][1],e[k]*e[i]*e[j]/128 if unused else Fraction(0),'registered metric first jet seed')
                    for l in range(3):seed_check(u['g_dd'][k][l][i][j][1],e[k]*e[l]*e[i]*e[j]/256 if unused else Fraction(0),'registered metric second jet seed')
    def verify_context(row):
        for key in ('all_input_finite','geometry_valid','positive_lapse_chi','SPD','reference_and_coefficients_zero_tangent'):
            require(row[key],key)
        require(row['uses_legacy_near'] is False,'far branch expected')
    def compare_native(row,truth,diagnostic):
        nonlocal tiny_count,legacy_nonfinite,legacy_bit_rows
        native=row['new'];require(native['valid'] and native['assembled'],'new source valid/assembled')
        values=native['parts']+native['rhs'];old=row['legacy']['parts']+row['legacy']['rhs']
        require(len(values)==12 and len(old)==12,'fixed output layout')
        bits_equal=True
        for index,(got,want,scale,previous) in enumerate(zip(values,truth,diagnostic,old)):
            for component in range(2):
                target_component=want.v if component==0 else want.d
                update('native-'+('parts' if index<8 else 'rhs')+('-primal' if component==0 else '-dual'),mp.mpf(got[component]),target_component,'2e-10',{'base':row.get('base_index'),'case':row.get('case_index'),'seed':row['seed_index'],'component':index,'dual':component})
                require(abs(target_component)<=mp.mpf(sys.float_info.max),'final target representable finite')
                if target_component and abs(target_component)<mp.mpf(float.fromhex('0x0.0000000000001p-1022')):tiny_count+=1
                maxima['literal-term-condition-diagnostic']=max(maxima.get('literal-term-condition-diagnostic',mp.mpf(0)),scale[component]/max(mp.mpf(1),abs(target_component)))
                try:
                    old_value=mp.mpf(previous[component]);bits_equal=bits_equal and float(got[component]).hex()==float(previous[component]).hex()
                    if mp.isfinite(old_value):update('legacy-error-metadata-only',old_value,target_component,mp.inf,'metadata only')
                    else:legacy_nonfinite+=1
                except (ValueError,TypeError):legacy_nonfinite+=1;bits_equal=False
        legacy_bit_rows+=int(bits_equal)
    attempt=Path(args.attempt);expected_direct=[]
    for base in registry['bases']:
        expected_direct.extend((base,seed,False) for seed in registry['seeds'])
        expected_direct.extend((base,next(s for s in registry['seeds'] if s['id']==name),True) for name in registry['zero_primal_gradient_variants']['seed_ids'])
    direct_count=fd_count=0;last_calls=0
    for number,row in rows(attempt/'direct.jsonl'):
        base,seed,variant=expected_direct[number-1];direct_count+=1
        require(row['kind']=='direct-dual','direct kind')
        require(row['base_index']==int(base['id'][1:]) and row['family']==base['family'] and row['direction']==base['direction'],'base registry identity')
        require(float(row['a']).hex()==float(base['a']).hex() and float(row['nominal_radius']).hex()==float(base['radius']).hex(),'registered a/radius')
        require(row['seed_id']==seed['id'] and row['seed_index']==registry['seeds'].index(seed) and row['zero_primal_gradients']==variant,'registered seed/variant')
        verify_context(row);verify_seed(row,seed)
        if variant:require(all(v[0]==0 for v in row['input']['alpha_d']+row['input']['chi_d']),'zero primal gradients')
        family=next(f for f in registry['State_constructor']['family_definitions'] if f['name']==row['family'])
        require(float(row['input']['alpha'][0]).hex()==float(family['alpha']).hex() and float(row['input']['chi'][0]).hex()==float(family['chi']).hex(),'original high-contrast primal')
        results=[]
        for precision in (480,560):
            with mp.workdps(precision):results.append(target(row))
        with mp.workdps(560):
            for index,(a,b) in enumerate(zip(results[0][0],results[1][0])):
                update('MP-precision-primal',a.v,b.v,'1e-220',number)
                update('MP-precision-dual',a.d,b.d,'1e-220',number)
            compare_native(row,*results[-1])
            if row['seed_id'] in ('zero','Theta-only','unconsumed-jet-only'):
                require(all(v[1]==0 for v in row['new']['parts']+row['new']['rhs']),'unused/zero exact source tangent')
            expected_fd=(not variant and row['seed_id']=='alpha-relative' and row['a']==.5 and row['direction']==1 and row['nominal_radius'] in (.025,.5,.84,.995))
            require(('FD' in row)==expected_fd,'fixed FD subset')
            if expected_fd:
                fd_count+=1;sequences=[]
                require(len(row['FD'])==5,'fixed five FD levels')
                for step,registered in zip(row['FD'],registry['FD']['steps']):
                    require(float(step['h']).hex()==float(registered).hex(),'fixed FD step')
                    require(step['side_fields_finite'] and step['side_positive_SPD'],'finite positive SPD FD sides')
                    require(all(step[s]['valid'] and step[s]['assembled'] for s in ('plus','minus')),'valid FD side source')
                    errors=[]
                    for plus,minus,tangent in zip(step['plus']['parts']+step['plus']['rhs'],step['minus']['parts']+step['minus']['rhs'],row['new']['parts']+row['new']['rhs']):
                        approximation=(mp.mpf(plus[0])-mp.mpf(minus[0]))/(2*mp.mpf(step['h']))
                        errors.append(error(approximation,mp.mpf(tangent[1])))
                    sequences.append(errors)
                for index in range(12):
                    errors=[level[index] for level in sequences]
                    require(errors[-1]<=mp.mpf('5e-7') and (errors[0]>=2*errors[-1] or max(errors)<=mp.mpf('5e-9')),'FD entry convergence/floor')
                FD.append({'base':row['base_index'],'errors':[[str(e) for e in level] for level in sequences]})
        last_calls+=2+10*int(expected_fd);require(row['helper_calls_cumulative']==last_calls,'direct call count')
    require(direct_count==2352 and fd_count==16 and last_calls==4864,'fixed direct+FD counts');calls['direct']=last_calls
    zero_old={};positive=negative=0;last_calls=0
    for number,row in rows(attempt/'closed.jsonl'):
        if row['kind']=='legacy-negative':
            negative+=1;which=row['case_index'];require(row['legacy']==zero_old[which],'negative reuses exact old export')
            index=5 if which==0 else 1;got=frac(row['legacy']['parts'][index][0]);prediction=pow2(301) if which==2 else Fraction(0)
            require(got==prediction,'legacy exact graph prediction');require(got!=[Fraction(1),Fraction(-1),Fraction(1)][which],'legacy wrong normal target detected')
            require(row['helper_calls_cumulative']==last_calls,'negative reuse has no call');continue
        positive+=1;which=row['case_index'];seed=row['seed_index'];verify_context(row)
        require(which==(number-1)//6 and seed==(number-1)%6 and row['case_id']==registry['closed_contexts'][which]['id'],'closed registry order')
        xa,xc,xg=map(Fraction,registry['closed_seed_order_xi_alpha_xi_chi_xi_gradient'][seed]);u=row['input'];h=row['reference']
        require(list(map(Fraction,row['xi']))==[xa,xc,xg],'closed fixed relative seeds')
        a=pow2(300 if which==1 else -300);x=pow2(601 if which==0 else -600 if which==1 else 600)
        require(frac(u['alpha'][0])==a and frac(u['chi'][0])==x and frac(h['alpha'][0])==frac(h['chi'][0])==1,'closed exact field values')
        require(frac(u['alpha'][1])==xa*a and frac(u['chi'][1])==xc*x,'closed complete value duals')
        for i in range(3):
            require(all(frac(v[c])==0 for v in (u['beta'][i],u['Lambda'][i],h['beta'][i],h['Lambda'][i]) for c in range(2)),'closed zero vector fields')
            for j in range(3):
                require(frac(u['g'][i][j][0])==frac(h['g'][i][j][0])==int(i==j) and u['g'][i][j][1]==h['g'][i][j][1]==0,'closed identity metric')
                require(u['beta_d'][i][j]==h['beta_d'][i][j]==[0,0],'closed zero beta gradient')
                require(u['A'][i][j]==h['A'][i][j]==[0,0],'closed zero A')
                for k in range(3):
                    require(u['g_d'][k][i][j]==h['g_d'][k][i][j]==[0,0],'closed zero first metric jet')
                    for l in range(3):require(u['g_dd'][k][l][i][j]==h['g_dd'][k][l][i][j]==[0,0],'closed zero second metric jet')
        for name in ('P','Theta'):require(u[name]==h[name]==[0,0],'closed zero P/Theta')
        require(row['Omega']==[1,0] and all(v==[0,0] for matrix in row['connection'] for line in matrix for v in line),'closed exact Omega/connection')
        expected_od=[[Fraction(1,2),Fraction(0),Fraction(0)],[Fraction(0)]*3,[Fraction(0)]*3][which]
        require([frac(v[0]) for v in row['Omega_d']]==expected_od,'closed supplied Omega derivatives')
        for j in range(3):
            ag=a if which==2 and j==0 else Fraction(0);cg=-x if which==1 and j==0 else Fraction(0)
            adual=a*(xa+xg) if which==2 and j==0 else Fraction(0);cdual=x*(-xc+xg) if which==1 and j==0 else Fraction(0)
            require(list(map(frac,u['alpha_d'][j]))==[ag,adual] and list(map(frac,u['chi_d'][j]))==[cg,cdual],'closed exact complete gradients')
            require(list(map(frac,h['alpha_d'][j]))==[Fraction(2 if which==2 and j==0 else 0),Fraction(0)] and list(map(frac,h['chi_d'][j]))==[Fraction(1 if which==1 and j==0 else 0),Fraction(0)],'closed fixed reference gradients')
        value=[Fraction(1),Fraction(-1),Fraction(1)][which]
        tangent=[4*xa+2*xc,-xa-xc/2+xg/2,-2*xa-xc-xg][which]
        exact=[[Fraction(0),Fraction(0)] for unused in range(12)];exact[5 if which==0 else 1]=[value,tangent];exact[9]=[value,tangent]
        native=row['new']['parts']+row['new']['rhs'];require(row['new']['valid'] and row['new']['assembled'],'closed new valid')
        with mp.workdps(560):
            literal,diagnostic=target(row)
            for index,(got,wanted) in enumerate(zip(native,exact)):
                for component in range(2):
                    truth=mp.mpf(wanted[component].numerator)/wanted[component].denominator
                    literal_value=literal[index].v if component==0 else literal[index].d
                    update('closed-literal-Fraction-identity',literal_value,truth,'1e-220',number)
                    update('closed-relative' if truth else 'closed-zero',mp.mpf(got[component]),truth,'2e-10',{'case':which,'seed':seed,'index':index,'dual':component},relative=bool(truth))
                    if not truth:require(got[component]==0,'closed unused exact zero output')
        if seed==0:zero_old[which]=row['legacy']
        last_calls+=2;require(row['helper_calls_cumulative']==last_calls,'closed call count')
    require(positive==18 and negative==3 and last_calls==36,'fixed closed/negative counts');calls['closed']=last_calls
    require(sum(calls.values())==4900,'total helper count')
    guard(inputs);guard(source_files)
    report={'passed':not failures,'failures':failures,'maxima':{k:str(v) for k,v in maxima.items()},
        'counts':{'direct_dual':direct_count,'closed_positive':positive,'legacy_negative_reuse':negative,
                  'records':direct_count+positive+negative,'FD_representatives':fd_count,'helper_calls':calls,'total_helper_calls':sum(calls.values())},
        'FD_sequences':FD,'below_binary64_minimum_nonzero_target_entries':tiny_count,
        'legacy_nonfinite_entries_metadata_only':legacy_nonfinite,'legacy_complete_output_bit_equal_rows_metadata_only':legacy_bit_rows,
        'inputs_unchanged':True,'scope':'direct finite-Omega RWM complete local field-dual arithmetic only; no compound inner/full22/stability/native acceptance'}
    path=attempt/'oracle-report.json'
    if path.exists():raise RuntimeError('Refuse overwrite oracle report')
    path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'passed':report['passed'],'counts':report['counts'],'failures':len(failures)},sort_keys=True))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
