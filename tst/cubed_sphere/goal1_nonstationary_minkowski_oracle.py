"""80-digit local exterior inverse harmonic-time Minkowski oracle, no evolution."""
import mpmath as m
one=m.mpf(1)
def radial(r,t,e):
 o=1-r*r;L=1+r*r;op=-2*r;Lp=2*r;c=one/2+t;d=one/2-t
 A=r+e*o;Ap=1-2*e*r
 J=-d*r**3+(2-d)*r*r-(1+d)*r+c
 Jp=-3*d*r*r+2*(2-d)*r-(1+d)
 E=r*r/(1+r)**2-e*J/(1+r)+e*e*L
 Ep=2*r/(1+r)**3-e*(Jp/(1+r)-J/(1+r)**2)+e*e*Lp
 D=E/A**2;Dp=Ep/A**2-2*E*Ap/A**3
 B=2*L*D-o*o*D*D
 Bp=2*Lp*D+2*L*Dp-2*o*op*D*D-2*o*o*D*Dp
 kr=(2*L*op*D+L*o*Dp-o*D*Lp)/B**m.mpf('1.5')
 kt=-(L-o*o*D)/(r*m.sqrt(B))
 delta=(L*Dp+2*D/r-o*D*(L*D+B)/r)/B**m.mpf('1.5')
 Tt=r/A;TR=1-o*o*D/L
 alpha=L*Tt/m.sqrt(B);beta=-Tt*(L-o*o*D)/B
 return o,L,B,Bp,kr,kt,delta,alpha,beta,Tt,TR
def fields(x,t,e):
 r=m.sqrt(sum(z*z for z in x));n=[z/r for z in x]
 o,L,B,Bp,kr,kt,delta,alpha,beta,_,_=radial(r,t,e);chi=B**(-one/3)
 lr=(2*one/3)*B**(-5*one/3)*Bp+2*B**(-2*one/3)*(B-1)/r
 g=[[chi*((i==j)+(B-1)*n[i]*n[j])for j in range(3)]for i in range(3)]
 A=[[chi*delta/3*(2*B*n[i]*n[j]-((i==j)-n[i]*n[j]))for j in range(3)]for i in range(3)]
 return [chi,alpha,kr+2*kt,m.mpf(0)],[beta*z for z in n],[lr*z for z in n],g,A
pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
def native(x,t,e):
 s,b,l,g,A=fields(x,t,e)
 return [s[0]]+[g[i][j]for i,j in pairs]+[s[2]]+[A[i][j]for i,j in pairs]+l+[s[3],s[1]]+b
def configuration(x,t,e):
 s,b,l,g,A=fields(x,t,e);l00=m.sqrt(g[0][0]);l10=g[1][0]/l00;l20=g[2][0]/l00;l11=m.sqrt(g[1][1]-l10*l10);l21=(g[2][1]-l10*l20)/l11
 return [s[0],m.log(l00),m.log(l11),l10,l20,l21,s[1]]+b
def derivative(fn,x,indices):
 return m.diff(lambda*y:fn(y),tuple(x),tuple(indices.count(i)for i in range(len(x))))if indices else fn(x)
def scalar_jet(fn,x):
 return [fn(x)]+[derivative(fn,x,[i])for i in range(3)]+[derivative(fn,x,[i,j])for i in range(3)for j in range(i,3)]
