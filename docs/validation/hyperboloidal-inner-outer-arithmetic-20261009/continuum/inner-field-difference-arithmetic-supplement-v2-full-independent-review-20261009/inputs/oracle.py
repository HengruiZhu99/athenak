#!/usr/bin/env python3
"""HELD stdlib Fraction oracle for fixed additive arithmetic witnesses."""
from fractions import Fraction as F
import hashlib,json,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gate_context import admitted,guard,pin,read,write_new

SEEDS=[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(1,-2,0),(0,0,1)]
REL=F(1,5000000000);ABS=REL
def require(ok,message):
    if not ok:raise RuntimeError(message)
def dyadic(text):
    x=float.fromhex(text)
    require(math.isfinite(x),'Nonfinite hexadecimal scalar')
    return F.from_float(x)
def pair(row):return dyadic(row['value_hex']),dyadic(row['dual_hex'])
def check(actual,target,label,errors):
    for component,(x,y) in enumerate(zip(pair(actual),target)):
        error=abs(x-y);relative=error/abs(y) if y else None
        require(error<=ABS if not y else relative<=REL,'Target mismatch '+label+' '+str(component))
        errors.append({'label':label,'component':component,'zero_target':not y,'absolute_error':str(error),'relative_error':str(relative) if relative is not None else None})
def exact(actual,target,label):require(pair(actual)==target,'Input identity '+label)
def audit(row,expected):
    keys=['dA_near','dA_far','dc_near','dc_far','dal_near','dal_far']
    require(set(row)==set(keys) and row=={key:expected.get(key,0) for key in keys},'Branch counters')

def main():
    require(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,'Require -I -B unoptimized')
    require(len(sys.argv)==4,'authorization, build, attempt required')
    here=Path(__file__).resolve().parent;auth,build,attempt=sys.argv[1],sys.argv[2],Path(sys.argv[3]).resolve()
    recipe,index,protected=admitted(here,auth,build)
    require(attempt==here/'attempts'/recipe['attempt_names'][build],'Exact fresh attempt path')
    rows=[json.loads(line) for line in (attempt/'probe.jsonl').read_text().splitlines()]
    require(len(rows)==129,'Exact total record count')
    seen=set();errors=[];counts={'witness':0,'negative-old-near':0,'near-bound':0}
    for row in rows:
        kind=row['kind'];seed=row['seed'];require(0<=seed<6 and tuple(row['xi'])==SEEDS[seed],'Fixed seeds')
        xa,xc,xg=SEEDS[seed]
        if kind in ('witness','negative-old-near'):
            witness=row['witness'];require(witness in (1,2,3),'Witness label');key=(kind,witness,seed)
            a=F(2)**(300 if witness==2 else -300);chi=F(2)**(601 if witness==1 else -600 if witness==2 else 600)
            exact(row['alpha'],(a,a*xa),'witness alpha');exact(row['chi'],(chi,chi*xc),'witness chi')
            exact(row['gradient'],(F(0),(chi if witness==2 else a)*xg),'zero-primal gradient seed')
            target=(F(1),F(4*xa+2*xc)) if witness==1 else (F(-1),F(-2*xa-xc+xg)) if witness==2 else (F(1),F(2*xa+xc-xg))
            if kind=='negative-old-near':
                require(seed==0 and pair(row['answer'])[0]==0 and pair(row['answer'])[0]!=target[0],'Old near-only lost-normal negative control')
                audit(row['audit'],{})
            else:
                check(row['answer'],target,str(key),errors)
                audit(row['audit'],{('dA_far' if witness==1 else 'dc_far' if witness==2 else 'dal_far'):1})
        elif kind=='near-bound':
            exponent,ratio_index=row['reference_exponent'],row['ratio_index'];require(exponent in (-400,0,400) and 0<=ratio_index<6,'Fixed bounds family')
            key=(kind,exponent,ratio_index,seed);h=F(2)**exponent;delta=F(2)**-40
            ratio=[F(1,2)-delta,F(1,2),F(1,2)+delta,F(2)-delta,F(2),F(2)+delta][ratio_index]
            a=h*ratio;near=F(1,2)<=a/h<=2
            require(row['witness']==0 and row['near'] is near,'Exact Fraction Near criterion')
            exact(row['reference'],(h,F(0)),'bound reference');exact(row['alpha'],(a,a*xa),'bound alpha');exact(row['chi'],(F(1),F(xc)),'bound chi');exact(row['gradient'],(a/8,a*xg),'bound gradient')
            check(row['dA'],(a*a-h*h,a*a*(2*xa+xc)),str(key)+' dA',errors)
            target=(-a/8,a*(F(xg)-F(xa,4)))
            check(row['dc'],target,str(key)+' dc',errors);check(row['dal'],target,str(key)+' dal',errors)
            suffix='near' if near else 'far';audit(row['audit'],{name+'_'+suffix:1 for name in ['dA','dc','dal']})
        else:raise RuntimeError('Unexpected row kind')
        require(key not in seen,'Duplicate query label');seen.add(key);counts[kind]+=1
    require(counts=={'witness':18,'negative-old-near':3,'near-bound':108},'Fixed family counts')
    guard(protected);guard(index['files'])
    report={'passed':True,'counts':counts,'records':len(rows),'component_target_checks':len(errors),'relative_nonzero_threshold':str(REL),'absolute_zero_threshold':str(ABS),'maximum_absolute_error':str(max(F(e['absolute_error']) for e in errors)),'maximum_relative_error':str(max(F(e['relative_error']) for e in errors if e['relative_error'] is not None)),'exact_zero_primal_nonzero_gradient_dual_covered':True,'negative_controls_all_three_lost_normal_primals_zero':True,'branch_endpoint_and_sides_verified':True,'floating_C1_continuity_claim':False,'main_suite_or_dV_changed':False,'arbitrary_nonlinear_gauge_or_evolution_acceptance':False,'errors':errors}
    write_new(attempt/'supplement-report.json',report)
    print(json.dumps({'passed':True,'records':len(rows),'report':pin(attempt/'supplement-report.json')}))

if __name__=='__main__':main()
