"""HELD exact-rational intermediate diagnosis; requires separately released query output."""
from pathlib import Path
from fractions import Fraction as Q
from collections import Counter
import argparse
import hashlib
import json
import struct
import sys

HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text(),parse_int=lambda x:-0.0 if x=='-0' else int(x))
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def verify(pins):
    for path,digest in pins.items():
        if sha(path)!=digest:raise RuntimeError('protected input changed: '+path)
def rat(x):return {'numerator':str(x.numerator),'denominator':str(x.denominator)}
class F:
    def __init__(self,v=0,d=0):self.v,self.d=Q(v),Q(d);self.sv,self.sd=abs(self.v),abs(self.d)
    @staticmethod
    def made(v,d,sv,sd):
        result=F(v,d);result.sv,result.sd=sv,sd;return result
    def __add__(self,x):x=asf(x);return F.made(self.v+x.v,self.d+x.d,self.sv+x.sv,self.sd+x.sd)
    __radd__=__add__
    def __neg__(self):return F.made(-self.v,-self.d,self.sv,self.sd)
    def __sub__(self,x):return self+-asf(x)
    def __rsub__(self,x):return asf(x)+-self
    def __mul__(self,x):
        x=asf(x);return F.made(self.v*x.v,self.d*x.v+self.v*x.d,self.sv*x.sv,self.sd*x.sv+self.sv*x.sd)
    __rmul__=__mul__
    def __truediv__(self,x):
        x=asf(x);return F.made(self.v/x.v,(self.d*x.v-self.v*x.d)/(x.v*x.v),self.sv/abs(x.v),self.sd/abs(x.v)+self.sv*x.sd/(x.v*x.v))
    def __rtruediv__(self,x):return asf(x)/self
def asf(x):return x if isinstance(x,F) else F(x)
def atom(x):return F(*x)
def matrix(x):return [[atom(y) for y in row] for row in x]
def inv(g):
    det=g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])-g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])+g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0])
    result=[]
    for i in range(3):
        row=[]
        for j in range(3):
            rr=[k for k in range(3) if k!=j];cc=[k for k in range(3) if k!=i]
            row.append(((-1)**(i+j))*(g[rr[0]][cc[0]]*g[rr[1]][cc[1]]-g[rr[0]][cc[1]]*g[rr[1]][cc[0]])/det)
        result.append(row)
    # Independent derivative identity, no STF-special implementation.
    for i in range(3):
        for j in range(3):
            assert result[i][j].d==-sum(result[i][k].v*g[k][l].d*result[l][j].v for k in range(3) for l in range(3))
    return det,result
def fields(u):
    return {k:atom(u[k]) for k in ('alpha','chi','P','Theta')}|{
        k:[atom(x) for x in u[k]] for k in ('alpha_d','chi_d','beta','Lambda')}|{
        k:matrix(u[k]) for k in ('g','beta_d')}
def model(row,G,Gh,aux=None):
    u,h=fields(row['input']),fields(row['reference']);a,x=u['alpha'],u['chi'];ar,y=h['alpha'],h['chi']
    c=[matrix(x) for x in row['connection']];od=[atom(x) for x in row['Omega_d']];o=atom(row['Omega'])
    db=[u['beta'][i]-h['beta'][i] for i in range(3)]
    dV=[[a*a*x*G[i][j]-ar*ar*y*Gh[i][j] for j in range(3)] for i in range(3)]
    Lh=[[ar*ar*y*Gh[i][j]-h['beta'][i]*h['beta'][j] for j in range(3)] for i in range(3)]
    dL=[[dV[i][j]-db[i]*u['beta'][j]-h['beta'][i]*db[j] for j in range(3)] for i in range(3)]
    if aux is not None:db,dV,Lh,dL=(aux[key] for key in ('db','dV','Lh','dL'))
    gc=[[a*a*G[i][j]*u['chi_d'][j]/2 for j in range(3)] for i in range(3)]
    gr=[[ar*ar*Gh[i][j]*h['chi_d'][j]/2 for j in range(3)] for i in range(3)]
    ga=[[a*x*G[i][j]*u['alpha_d'][j] for j in range(3)] for i in range(3)]
    gar=[[ar*y*Gh[i][j]*h['alpha_d'][j] for j in range(3)] for i in range(3)]
    groups=[[gc[i][j]-gr[i][j]-ga[i][j]+gar[i][j] for j in range(3)] for i in range(3)]
    pa=[[a*dL[i][j]*c[0][i][j] for j in range(3)] for i in range(3)]
    pv=[[2*dV[i][j]*od[j] for j in range(3)] for i in range(3)]
    pc1=[[[dL[j][l]*c[i+1][j][l] for l in range(3)] for j in range(3)] for i in range(3)]
    pc2=[[[dL[j][l]*u['beta'][i]*c[0][j][l] for l in range(3)] for j in range(3)] for i in range(3)]
    pc3=[[[Lh[j][l]*db[i]*c[0][j][l] for l in range(3)] for j in range(3)] for i in range(3)]
    R=[sum(u['beta'][j]*u['alpha_d'][j]-a/ar*h['beta'][j]*h['alpha_d'][j] for j in range(3))]
    S=[-a*a*u['P']+a*ar*h['P']-sum(a*db[i]*od[i] for i in range(3))-sum(pa[i][j] for i in range(3) for j in range(3))]
    for i in range(3):
        R.append(a*a*x*u['Lambda'][i]-ar*ar*y*h['Lambda'][i]+sum(u['beta'][j]*(u['beta_d'][j][i]-h['beta_d'][j][i])+db[j]*h['beta_d'][j][i]+groups[i][j] for j in range(3)))
        S.append(sum(pv[i][j] for j in range(3))-sum(pc1[i][j][l]+pc2[i][j][l]+pc3[i][j][l] for j in range(3) for l in range(3)))
    return {'db':db,'dV':dV,'Lh':Lh,'dL':dL,'gradient_chi_live':gc,'gradient_chi_reference':gr,
        'gradient_alpha_live':ga,'gradient_alpha_reference':gar,'gradient_group':groups,
        'pole_alpha_connection_unsigned':pa,'pole_beta_dV':pv,'pole_beta_connection1_unsigned':pc1,
        'pole_beta_connection2_unsigned':pc2,'pole_beta_connection3_unsigned':pc3},R+S+[R[i]+S[i]/o for i in range(4)]
def equal_bits(a,b):
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equal_bits(x,y) for x,y in zip(a,b))
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal_bits(a[k],b[k]) for k in a)
    if isinstance(a,(int,float)) and not isinstance(a,bool):return struct.pack('>d',float(a))==struct.pack('>d',float(b))
    return a==b
def leaves(got,wanted,path=()):
    if isinstance(wanted,F):yield path,atom(got),wanted
    else:
        assert len(got)==len(wanted)
        for i,(a,b) in enumerate(zip(got,wanted)):yield from leaves(a,b,path+(i,))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authorization',required=True);ap.add_argument('--attempt',required=True);args=ap.parse_args()
    if not(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize):raise RuntimeError('require -I -B unoptimized')
    r=load(HERE/'recipe.json');auth=load(args.authorization)
    if not(auth.get('metric_flux_diagnostic_local_admitted') is True and auth['recipe_sha256']==sha(HERE/'recipe.json') and auth['source_index_sha256']==sha(HERE/'source-index.json')):raise RuntimeError('no exact diagnostic release')
    pins=load(HERE/'input-pins.json');pins.update({str(HERE/'recipe.json'):sha(HERE/'recipe.json'),str(HERE/'source-index.json'):sha(HERE/'source-index.json'),str(Path(args.authorization).resolve()):sha(args.authorization)})
    for x in load(HERE/'source-index.json')['files']:pins[x['path']]=x['sha256']
    verify(pins);out=Path(args.attempt);write(out/'oracle-pins-before.json',pins)
    selected=set(r['selected_bases']);original={}
    with Path(r['original_direct_jsonl']).open() as stream:
        for number,line in enumerate(stream,1):
            row=json.loads(line,parse_int=lambda x:-0.0 if x=='-0' else int(x))
            if row['base_index'] in selected and row['seed_index']==13 and not row['zero_primal_gradients']:original[row['base_index']]=row
    assert set(original)==selected
    failed=load(r['original_oracle_report'])['failures'];assert len(failed)==63
    stats={};count=0;labels=[];failed_labels_checked=0
    def record(stream,kind,base,label,got,target):
        for derivative,(x,y) in enumerate(((got.v,target.v),(got.d,target.d))):
            error=abs(x-y);scaled=error/max(Q(1),abs(x),abs(y));key=(kind,derivative)
            term_bound=(target.sv,target.sd)[derivative];term_scaled=error/max(Q(1),term_bound)
            if key not in stats:stats[key]={'scaled':Q(-1),'absolute':Q(-1),'term_scaled':Q(-1)}
            if scaled>stats[key]['scaled']:stats[key].update(scaled=scaled,scaled_base=base,scaled_label=label,target_exact_zero=y==0)
            if error>stats[key]['absolute']:stats[key].update(absolute=error,absolute_base=base,absolute_label=label)
            if term_scaled>stats[key]['term_scaled']:stats[key].update(term_scaled=term_scaled,term_base=base,term_label=label)
            stream.write(json.dumps({'kind':kind,'base':base,'label':label,'dual':derivative,'absolute_error':rat(error),'scaled_error':rat(scaled),'expression_term_bound':rat(term_bound),'expression_term_scaled_error':rat(term_scaled),'target_exact_zero':y==0},sort_keys=True)+'\n')
    with (out/'entry-metrics.jsonl').open('w') as metrics,(out/'diagnostic.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line,parse_int=lambda x:-0.0 if x=='-0' else int(x));base=row['base_index'];count+=1
            assert base in selected and base not in labels;labels.append(base)
            old=original[base]
            for key in ('xyz','Omega','Omega_d','reference','input','connection','new','legacy'):
                if not equal_bits(row[key],old[key]):raise RuntimeError('original consumed/output bits differ: '+str(base)+'/'+key)
            for key in ('all_input_finite','geometry_valid','positive_lapse_chi','SPD','reference_and_coefficients_zero_tangent'):
                assert row[key] is True
            assert row['uses_legacy_near'] is False and row['helper_calls_cumulative']==2*count
            det,G=inv(matrix(row['input']['g']));deth,Gh=inv(matrix(row['reference']['g']))
            Gn=matrix(row['geometry_live']['inverse']);Ghn=matrix(row['geometry_reference']['inverse'])
            for name,native,truth in (('geometry_live',row['geometry_live'],(det,G)),('geometry_reference',row['geometry_reference'],(deth,Gh))):
                record(metrics,'inverse-stage',base,name+'.determinant',atom(native['determinant']),truth[0])
                for path,x,y in leaves(native['inverse'],truth[1]):record(metrics,'inverse-stage',base,name+'.inverse'+str(path),x,y)
            exact,full=model(row,G,Gh);held,fullheld=model(row,Gn,Ghn)
            data=row['intermediates']
            aux={'db':[atom(x) for x in data['db']],**{key:matrix(data[key]) for key in ('dV','Lh','dL')}}
            heldaux,fullaux=model(row,Gn,Ghn,aux)
            for key in exact:
                for path,x,y in leaves(row['intermediates'][key],exact[key]):
                    record(metrics,'complete-intermediate-error',base,key+str(path),x,y)
                for path,x,y in leaves(row['intermediates'][key],held[key]):
                    record(metrics,'downstream-with-native-inverse-error',base,key+str(path),x,y)
                if key.startswith('pole_'):
                    for path,x,y in leaves(data[key],heldaux[key]):
                        record(metrics,'pole-product-with-native-auxiliary-inputs',base,key+str(path),x,y)
            # Explicit exact addition-only error for the four already rounded terms.
            for i in range(3):
                for j in range(3):
                    rounded=sum(sign*atom(data[key][i][j]) for key,sign in (('gradient_chi_live',1),('gradient_chi_reference',-1),('gradient_alpha_live',-1),('gradient_alpha_reference',1)))
                    record(metrics,'gradient-addition-only',base,'gradient_group'+str((i,j)),atom(data['gradient_group'][i][j]),rounded)
            native=row['new']['parts']+row['new']['rhs']
            for ci,(observed,true,heldtrue,auxtrue) in enumerate(zip(native,full,fullheld,fullaux)):
                actual=atom(observed)
                record(metrics,'full-source-error',base,str(ci),actual,true)
                record(metrics,'inverse-effect-exact-downstream',base,str(ci),heldtrue,true)
                record(metrics,'remaining-arithmetic-after-native-inverse',base,str(ci),actual,heldtrue)
                record(metrics,'auxiliary-effect-exact-downstream',base,str(ci),auxtrue,heldtrue)
                record(metrics,'final-products-and-sums-after-native-auxiliary',base,str(ci),actual,auxtrue)
                for k in (0,1):
                    x=(actual.v,actual.d)[k];y=(true.v,true.d)[k];z=(heldtrue.v,heldtrue.d)[k]
                    assert x-y==(z-y)+(x-z)
                    t=(auxtrue.v,auxtrue.d)[k]
                    assert x-y==(z-y)+(t-z)+(x-t)
            for f in failed:
                if f['label']['base']!=base:continue
                ci=f['label']['component'];target=Q(f['target']);truth=full[ci].d
                assert abs(truth-target)<=Q(2,10**10)*max(Q(1),abs(truth),abs(target));failed_labels_checked+=1
    assert count==26 and labels==r['selected_bases'] and failed_labels_checked==63
    summary={'diagnostic_completed':True,'original_far_Release_passed':False,'rows':26,'helper_calls':52,'original_failed_labels_checked':63,
        'original_saved_zero_targets':sum(Q(f['target'])==0 for f in failed),'original_bits_equal':True,'exact_inverse_derivative_identity_passed':True,
        'stage_maxima':{kind+'/dual'+str(d):{**{k:v for k,v in value.items() if k not in ('scaled','absolute','term_scaled')},'scaled':rat(value['scaled']),'absolute':rat(value['absolute']),'expression_term_scaled':rat(value['term_scaled'])} for (kind,d),value in stats.items()},
        'term_bound_scope':'Absolute-sum propagation through the explicitly recorded rational expression tree, with exact nonzero denominator magnitude; diagnostic conditioning bound, not native error theorem.',
        'limits':['Observational26-row decomposition only; original63 failures and all main thresholds unchanged.','Holding native inverse exact isolates its effect in an ideal downstream model; remaining arithmetic includes all coefficient/product/sum stages and is not a unique causal percentage.','No new helper, source correction, Debug/native or evolution acceptance.']}
    verify(pins);summary['inputs_unchanged']=True;write(out/'oracle-report.json',summary);print(json.dumps({'diagnostic_completed':True,'rows':26,'original_far_Release_passed':False}))

if __name__=='__main__':main()
