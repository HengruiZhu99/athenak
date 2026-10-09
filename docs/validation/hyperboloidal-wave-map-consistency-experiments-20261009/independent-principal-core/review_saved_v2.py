"""Independent Fraction proof and saved-only capsule review. No owner code import/API."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,itertools,json,math,subprocess,sys,time,traceback
P=Path(__file__).resolve().parent;R=P.parents[2]
S=R/'build-layer-research/continuum/reference-wave-map-principal-core-20261009/immutable-principal-core-wave-map-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mm(a,b):return [[sum((x*y for x,y in zip(row,col)),F(0))for col in zip(*b)]for row in a]
def tr(a):return [list(x)for x in zip(*a)]
def add(a,b,c=F(1)):return [[x+c*y for x,y in zip(r,s)]for r,s in zip(a,b)]
def scale(a,c):return [[c*x for x in r]for r in a]
def rank(a):
 a=[r[:]for r in a];k=0
 for j in range(len(a[0])):
  p=next((i for i in range(k,len(a))if a[i][j]),None)
  if p is None:continue
  a[k],a[p]=a[p],a[k];v=a[k][j];a[k]=[x/v for x in a[k]]
  for i in range(len(a)):
   if i!=k and a[i][j]:v=a[i][j];a[i]=[x-v*y for x,y in zip(a[i],a[k])]
  k+=1
 return k
# Independent coefficient equations from flat C0 geometry and harmonic gauge.
# q=(alpha_n,chi_n,g_nn_n,P,Theta,A_nn,Lambda_n,beta_n_n), then two vector and two TT blocks.
def expected_matrix():
 m=[[F(0)for _ in range(20)]for _ in range(20)]
 def row(i,entries):
  for j,v in entries.items():m[i][j]=F(v)
 row(0,{3:-1});row(1,{3:F(2,3),4:F(4,3),7:F(-2,3)})
 row(2,{5:-2,7:F(4,3)});row(3,{0:-1});row(4,{1:1,6:F(1,2)})
 row(5,{0:F(-2,3),1:F(1,3),2:F(-1,2),6:F(2,3)})
 row(6,{3:F(-4,3),4:F(-2,3),7:F(4,3)});row(7,{0:-1,1:F(1,2),6:1})
 for b in [8,12]:
  row(b,{b+1:-2,b+3:1});row(b+1,{b:F(-1,2),b+2:F(1,2)})
  row(b+2,{b+3:1});row(b+3,{b+2:1})
 for b in [16,18]:row(b,{b+1:-2});row(b+1,{b:F(-1,2)})
 return m
# Sparse three-variable polynomial ring over Fraction, independent of owner AD.
Z={};one={(0,0,0):F(1)}
def C(q):return {}if not q else{(0,0,0):F(q)}
def pa(*args):
 o={}
 for a in args:
  for k,v in a.items():o[k]=o.get(k,F(0))+v
 return{k:v for k,v in o.items()if v}
def ps(a,c):return{k:v*F(c)for k,v in a.items()if v*c}
def pm(a,b):
 o={}
 for k,v in a.items():
  for l,w in b.items():q=tuple(x+y for x,y in zip(k,l));o[q]=o.get(q,F(0))+v*w
 return{k:v for k,v in o.items()if v}
def pd(a,i):return{tuple(q-(j==i)for j,q in enumerate(k)):v*k[i]for k,v in a.items()if k[i]}
def lap(a):return pa(*(pd(pd(a,i),i)for i in range(3)))
def ev(a,x):return sum((v*math.prod(x[i]**k[i]for i in range(3))for k,v in a.items()),F(0))
x=[{tuple(int(j==i)for j in range(3)):F(1)}for i in range(3)];rho=pa(*(pm(q,q)for q in x))
powers=[one]
for _ in range(3):powers.append(pm(powers[-1],rho))
def poly(cs):return pa(*(ps(powers[k],v)for k,v in enumerate(cs)))
def deriv(cs,n=1):
 out=list(cs)
 for _ in range(n):out=[(k+1)*out[k+1]for k in range(len(out)-1)]
 return out+[F(0)]*(4-len(out))
def fields(t,z,v,w):
 # Derived from hphys=Lie_(x*z) I, Kij=-Hess(t), fixedOmega=1.
 tp,zp=poly(deriv(t)),poly(deriv(z));T,Z,V,W=map(poly,[t,z,v,w])
 h=[[pa(ps(Z,2)if i==j else{},ps(pm(pm(x[i],x[j]),zp),4))for j in range(3)]for i in range(3)]
 ht=pa(*(h[i][i]for i in range(3)));chi=ps(ht,F(-1,3))
 g=[[pa(h[i][j],chi if i==j else{})for j in range(3)]for i in range(3)]
 P=ps(lap(T),-1);A=[[pa(ps(pd(pd(T,i),j),-1),ps(P,F(-1,3))if i==j else{})for j in range(3)]for i in range(3)]
 beta=[pa(pm(x[i],W),ps(pd(T,i),-1))for i in range(3)]
 lam=[pa(*(pd(g[i][j],j)for j in range(3)))for i in range(3)]
 return {'alpha':V,'chi':chi,'P':P,'Theta':{},'beta':beta,'g':g,'A':A,'Lambda':lam}
def flat_action(f):
 a,c,p,th,b,g,A,l=(f[k]for k in ['alpha','chi','P','Theta','beta','g','A','Lambda'])
 divb=pa(*(pd(b[i],i)for i in range(3)));divl=pa(*(pd(l[i],i)for i in range(3)))
 adot=ps(p,-1);bdot=[pa(l[i],ps(pd(c,i),F(1,2)),ps(pd(a,i),-1))for i in range(3)]
 cdot=ps(pa(p,ps(th,2),ps(divb,-1)),F(2,3));pdot=pa(ps(lap(a),-1),ps(th,10))
 tdot=pa(lap(c),ps(divl,F(1,2)),ps(th,-20))
 gd=[[pa(ps(A[i][j],-2),pd(b[j],i),pd(b[i],j),ps(divb,F(-2,3))if i==j else{})for j in range(3)]for i in range(3)]
 raw=[[pa(ps(pd(pd(a,i),j),-1),ps(lap(g[i][j]),F(-1,2)),ps(pd(pd(c,i),j),F(1,2)),ps(pa(pd(l[j],i),pd(l[i],j)),F(1,2)),ps(lap(c),F(1,2))if i==j else{})for j in range(3)]for i in range(3)]
 rt=pa(*(raw[i][i]for i in range(3)));Ad=[[pa(raw[i][j],ps(rt,F(-1,3))if i==j else{})for j in range(3)]for i in range(3)]
 ld=[pa(lap(b[i]),ps(pd(divb,i),F(1,3)),ps(pd(p,i),F(-4,3)),ps(pd(th,i),F(-2,3)),ps(pa(l[i],ps(pa(*(pd(g[i][j],j)for j in range(3))),-1)),-10))for i in range(3)]
 return {'alpha':adot,'chi':cdot,'P':pdot,'Theta':tdot,'beta':bdot,'g':gd,'A':Ad,'Lambda':ld}
def constraints(f):
 c,p,th,g,A,l=(f[k]for k in ['chi','P','Theta','g','A','Lambda'])
 H=pa(ps(lap(c),2),*(pd(pd(g[i][j],i),j)for i in range(3)for j in range(3)))
 M=[pa(*(pd(A[i][j],j)for j in range(3)),ps(pd(pa(p,ps(th,2)),i),F(-2,3)))for i in range(3)]
 Zc=[ps(pa(l[i],ps(pa(*(pd(g[i][j],j)for j in range(3))),-1)),F(1,2))for i in range(3)]
 return[H]+M+Zc+[th]
def raw(f):return[f[k]for k in ['alpha','chi','P','Theta']]+f['beta']+[f['g'][i][j]for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]]+[f['A'][i][j]for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]]+f['Lambda']
def exact_core():
 cases=[]
 for witness in range(17):
  cs=[[F(0)]*4 for _ in range(4)]
  if witness<16:cs[witness//4][witness%4]=F(1)
  else:cs=[[1,1,F(-1,5),0],[F(1,3),F(-1,7),0,F(1,11)],[F(-1,4),F(1,9),0,0],[F(1,6),0,F(-1,13),0]]
  t,z,v,w=cs;f=fields(*cs);actual=flat_action(f)
  vt=[6*q for q in deriv(t)];wt=[10*q for q in deriv(z)]
  for k,q in enumerate(deriv(t,2)[:-1]):vt[k+1]+=4*q
  for k,q in enumerate(deriv(z,2)[:-1]):wt[k+1]+=4*q
  target=fields(v,w,vt,wt)
  delta=[pa(a,ps(b,-1))for a,b in zip(raw(actual),raw(target))]
  normals=[pa(*(f[k][i][i]for i in range(3)))for k in ['g','A']]+[pa(*(actual[k][i][i]for i in range(3)))for k in ['g','A']]
  checks=delta+constraints(f)+normals
  assert not any(checks),(witness,'nonzero exact polynomial')
  acc=[pa(actual['alpha'],ps(poly(vt),-1))]+[pa(actual['beta'][i],pd(poly(v),i),ps(pm(x[i],poly(wt)),-1))for i in range(3)]
  assert not any(acc)
  for r in [0,.025,.049,.05]:
   for n in [[1,0,0],[.36,-.48,.8],[-.48,.64,.6]]:
    point=[F.from_float(float(r)*float(ni))for ni in n]
    assert all(ev(q,point)==0 for q in checks+acc)
    if r==0:
     origin=ps(pa(*(pd(f['Lambda'][i],i)for i in range(3)),ps(lap(f['chi']),F(1,2))),F(1,3))
     assert ev(pa(origin,ps(poly(wt),-1)),point)==0
    cases.append({'witness':witness,'r':r,'direction':n,'all22_exact':True,'physical8_exact_zero':True,'input_output_trace_normals_exact_zero':True,'acceleration4_exact':True,'origin_scalar_w_limit':r==0})
 assert len(cases)==204
 return cases

def run():
 t0=time.monotonic();recipe=json.loads((P/'recipe-v2.json').read_text())
 for rec in recipe['inputs']:assert sha(rec['path'])==rec['sha256'],rec['path']
 assert sha(__file__)==recipe['review_source_sha256']
 out=P/'attempt002';assert not out.exists();out.mkdir()
 result={'kind':'independent saved-only principal/core review','passed':False,'recipe_sha256':sha(P/'recipe-v2.json'),'source_sha256':sha(__file__),'no_sourcequeries_or_compile_or_matrix_generation_or_evolution':True}
 try:
  idx=json.loads((S/'index.json').read_text());assert sha(S/'index.json')==recipe['capsule_sha256']
  for f in idx['files']:
   q=S/f['path'];assert q.stat().st_size==f['bytes']and sha(q)==f['sha256'],q
   if q.suffix=='.json':json.dumps(json.loads(q.read_text()),allow_nan=False)
  dependencies=0
  for deps in idx['external_compiler_dependencies'].values():
   for p,h in deps.items():assert sha(p)==h,p;dependencies+=1
  for p,h in idx['external_source_inputs'].items():assert sha(R/p)==h,p
  A=S/idx['accepted_attempt'];r=json.loads((A/'receipt.json').read_text())
  assert r['passed_AB']and r['sources_unchanged']and r['release_debug_numeric_JSON_equal']and not r['C_queries_or_evolution_run']
  assert len(r['commands'])==11 and all(c['returncode']==0 and c['stderr_sha256']==hashlib.sha256(b'').hexdigest()for c in r['commands'])
  assert r['source_before']==r['source_after']and len(r['source_before'])==375
  M=expected_matrix();I=[[F(i==j)for j in range(20)]for i in range(20)];O=scale(I,F(0));pp=scale(add(I,M),F(1,2));pn=scale(add(I,M,F(-1)),F(1,2));H=scale(add(I,mm(tr(M),M)),F(1,2))
  assert mm(M,M)==I and sum(M[i][i]for i in range(20))==0
  assert mm(pp,pp)==pp and mm(pn,pn)==pn and mm(pp,pn)==O and add(pp,pn)==I
  assert rank(pp)==rank(pn)==10 and mm(H,M)==mm(tr(M),H)and H==add(mm(tr(pp),pp),mm(tr(pn),pn))
  exact={'M2_I':True,'trace_M_zero':True,'complementary_idempotent_projectors':True,'projector_ranks':[rank(pp),rank(pn)],'H_projector_and_IplusMtM_identity':True,'HM_MtH':True,'positive_quadratic_lower_bound':'x^T H x=(||x||^2+||Mx||^2)/2 >= ||x||^2/2, by exact coefficient identity'}
  (out/'exact-expected.json').write_text(json.dumps({'M':[[str(q)for q in row]for row in M],'H':[[str(q)for q in row]for row in H],'proof':exact},indent=2)+'\n')
  expected=[[float(q)for q in row]for row in M]
  def fmul(a,b):return[[math.fsum(x*y for x,y in zip(row,col))for col in zip(*b)]for row in a]
  nums={};payloads=[]
  for mode in ['release','debug']:
   rows=json.loads((A/('principal-'+mode+'.json')).read_text());assert len(rows)==792
   labels={(q['alpha'],q['chi'],q['oblique'],q['a'],q['r'])for q in rows};assert len(labels)==792
   assert labels==set(itertools.product([.2,1.,3.],[.4,1.,2.],[False,True],[.5,.75,1.,2.],[0.,.45,.5,.65,.8,.83,.84,.849,.85,.95,.98]))
   e=inv=sym=normal=0
   for q in rows:
    a=q['M'];assert len(a)==20 and all(len(v)==20 for v in a)and all(math.isfinite(v)for row in a for v in row)
    e=max(e,max(abs(a[i][j]-expected[i][j])for i in range(20)for j in range(20)))
    aa=fmul(a,a);inv=max(inv,max(abs(aa[i][j]-(i==j))for i in range(20)for j in range(20)))
    at=tr(a);ata=fmul(at,a);h=[[(ata[i][j]+(i==j))/2 for j in range(20)]for i in range(20)]
    hm=fmul(h,a);mth=fmul(at,h);sym=max(sym,max(abs(hm[i][j]-mth[i][j])for i in range(20)for j in range(20)))
    normal=max(normal,q['normal_scaled'])
   assert e<=2e-12 and inv<=2e-11 and sym<=2e-11 and normal<=5e-11
   nums[mode]={'cases':len(rows),'expected_error':e,'M2_error':inv,'HM_error':sym,'normal_scaled':normal};payloads.append(rows)
  assert payloads[0]==payloads[1]
  cases=exact_core();(out/'exact-core204-targets.json').write_text(json.dumps(cases,indent=2)+'\n')
  core=json.loads((A/'core-release.json').read_text());assert core==json.loads((A/'core-debug.json').read_text())
  assert core['core_cases']==204 and core['witnesses']==17 and all(math.isfinite(v)and v<=5e-11 for k,v in core.items()if k not in ['core_cases','witnesses'])
  fail=S/'attempts/1791564543283538000/receipt.json';assert sha(fail)=='40a8f2c3311a5dceb86d6d694718d75969e9c3733c67d4b67477cedaecc2daf4';fr=json.loads(fail.read_text());assert not fr['passed_AB']
  for rec in recipe['inputs']:assert sha(rec['path'])==rec['sha256']
  result.update(passed=True,capsule_files_verified=len(idx['files']),source_input_pins_verified=len(idx['external_source_inputs']),compiler_dependency_entries_verified=dependencies,exact_fraction_principal=exact,actual_saved_principal=nums,Release_Debug_actual_matrices_equal=True,exact_polynomial_core_targets_verified=len(cases),saved_actual_core_aggregate=core,core_limitation='Actual per-case rows were not saved. Independently verified all204 exact polynomial targets, physical8 constraints, input/output normals, accelerations and origin limits; actual source binding reviewed and saved aggregate maxima rechecked, but individual actual residuals are not recomputed.',first_failure_preserved=True,linear_principal_core_scope_only=True,no_finite_k_nonlinear_Cwave_or_scri_acceptance=True,seconds=time.monotonic()-t0)
 except Exception as e:result.update(exception=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-t0)
 (out/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));assert result['passed']
if __name__=='__main__':run()
