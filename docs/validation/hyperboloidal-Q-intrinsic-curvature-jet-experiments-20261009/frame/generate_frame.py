from pathlib import Path
import sympy as s,json
P=Path(__file__).resolve().parent;x,y,z,a,O=s.symbols('x y z a O');v=[x,y,z];Ts=[[-z*z*x,-z*z*y,z*(x*x+y*y)],[-y,x,0]];degrees=[3,1];pairs=[(0,0),(0,1),(0,2),(1,1),(1,2)]
def dx(F,i):return s.diff(F,v[i])-v[i]*s.diff(F,O)/a
code=['Jet Field(const hyp::LayerPoint<double>&p,double a,int which,bool first){auto u=Lift(p.state);const double x=-a*p.beta[0],y=-a*p.beta[1],z=-a*p.beta[2],O=p.omega;J q[20]{};switch(2*which+int(first)){']
for which,T in enumerate(Ts):
 for first in [False,True]:
  F=[0]*20;B=[O*t for t in T]
  if not first:
   for i in range(3):F[4+i]=B[i]
  else:
   div=sum(dx(B[i],i)for i in range(3));F[1]=-s.Rational(2,3)*div
   for i in range(3):F[4+i]=((1-2*a*O)/a**2-((degrees[which]+1)/a+1)*O)*T[i];F[17+i]=sum(dx(dx(B[i],j),j)for j in range(3))+dx(div,i)/3
   for k,(i,j)in enumerate(pairs):F[7+k]=dx(B[i],j)+dx(B[j],i)+F[1]*int(i==j)
  dest=[];expr=[]
  for k in range(20):
   dest.append(f'q[{k}].v');expr.append(s.factor(F[k]))
   for i in range(3):
    dest.append(f'q[{k}].d[{i}]');expr.append(s.factor(dx(F[k],i)))
    for j in range(3):dest.append(f'q[{k}].dd[{i}][{j}]');expr.append(s.factor(dx(dx(F[k],i),j)))
  tmp,vals=s.cse(expr,s.numbered_symbols('t'));code.append(f'case {2*which+int(first)}:{{')
  for name,val in tmp:code.append(f'const double {name}={s.ccode(val)};')
  for d,e in zip(dest,vals):
   if e!=0:code.append(f'{d}=D(0,{s.ccode(e)});')
  code.append('break;}')
code+=['default:throw std::runtime_error("bad frame field");}','for(int k=0;k<20;++k)Seed(u,k,q[k]);Consistent(u);return u;}'];(P/'generated_frame.hpp').write_text('\n'.join(code)+'\n')
