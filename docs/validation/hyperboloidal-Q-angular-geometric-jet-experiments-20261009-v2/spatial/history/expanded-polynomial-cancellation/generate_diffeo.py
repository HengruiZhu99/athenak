from pathlib import Path
import sympy as s,json,hashlib,time
P=Path(__file__).resolve().parent;t=time.monotonic();x,y,z,a=s.symbols('x y z a');v=[x,y,z];O=(1-x*x-y*y-z*z)/(2*a);h=(1+x*x+y*y+z*z)/(2*a);mon=[1,x,y,z,x*x,x*y,x*z,y*y,y*z,z*z];pairs=[(0,0),(0,1),(0,2),(1,1),(1,2)]
code=['// Generated exact polynomial pullback jets. No physical RHS subtraction.','Jet Initial(const hyp::LayerPoint<double>&p,double a,int col){auto u=Lift(p.state);const double x=p.radius==0?0:p.state.beta.value[0]*(-a),y=p.state.beta.value[1]*(-a),z=p.state.beta.value[2]*(-a);','J q[20]{};switch(col){'];meta=[]
for m in [2,3,4]:
 for j in range(3):
  for k,X in enumerate(mon):
   xi=[s.expand(O**m*X*(int(i==j)))for i in range(3)];div=sum(s.diff(xi[i],v[i])for i in range(3));F=[0]*20;F[0]=s.expand(O**(m-1)*v[j]*X/a**2);F[1]=s.expand(-s.Rational(2,3)*div+2*O**(m-1)*X*s.diff(O,v[j]))
   for i in range(3):F[4+i]=s.expand((-xi[i]+sum(v[t]*s.diff(xi[i],v[t])for t in range(3)))/a);F[17+i]=s.expand(sum(s.diff(xi[i],w,2)for w in v)+s.diff(div,v[i])/3)
   for c,(i,t) in enumerate(pairs):F[7+c]=s.expand(s.diff(xi[t],v[i])+s.diff(xi[i],v[t])-s.Rational(2,3)*div*int(i==t))
   expr=[];dest=[]
   for c in range(20):
    expr.append(F[c]);dest.append(f'q[{c}].v')
    for i in range(3):
     expr.append(s.diff(F[c],v[i]));dest.append(f'q[{c}].d[{i}]')
     for t in range(3):expr.append(s.diff(F[c],v[i],v[t]));dest.append(f'q[{c}].dd[{i}][{t}]')
   tmp,reduced=s.cse(expr,s.numbered_symbols('t'));col=len(meta);code.append(f'case {col}:{{')
   for name,val in tmp:code.append(f'const double {name}={s.ccode(val)};')
   for target,val in zip(dest,reduced):
    if val!=0:code.append(f'{target}=D(0,{s.ccode(val)});')
   code.append('break;}');meta.append({'col':col,'m':m,'axis':j,'monomial':str(X)})
code+=['default:throw std::runtime_error("invalid pullback column");}','for(int c=0;c<20;++c)Seed(u,c,q[c]);Consistent(u);return u;}'];(P/'generated_diffeo.hpp').write_text('\n'.join(code)+'\n');(P/'basis.json').write_text(json.dumps(meta,indent=2)+'\n');print(len(meta),time.monotonic()-t,hashlib.sha256((P/'generated_diffeo.hpp').read_bytes()).hexdigest())
