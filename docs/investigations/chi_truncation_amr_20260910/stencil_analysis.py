"""Independent rational checks of derivative leading terms and reference scaling."""
from fractions import Fraction as F
import json,math
x=list(range(-3,4));w5=list(map(F,[-1,4,-5,0,5,-4,1]));w5=[v/2 for v in w5];w6=list(map(F,[1,-6,15,-20,15,-6,1]))
checks={}
for derivative,w in [(5,w5),(6,w6)]:
 moments=[sum(a*F(b)**n for a,b in zip(w,x)) for n in range(derivative+2)]
 assert all(v==(math.factorial(derivative) if n==derivative else 0) for n,v in enumerate(moments))
 checks[str(derivative)]=list(map(str,moments))
# Actual fourth-order D1/D2 truncation coefficients from Taylor moments.
a=[-2,-1,0,1,2];d1=[F(1,12),F(-2,3),F(0),F(2,3),F(-1,12)];d2=[F(-1,12),F(4,3),F(-5,2),F(4,3),F(-1,12)]
assert sum(w*F(x)**5 for x,w in zip(a,d1))/math.factorial(5)==F(-1,30)
assert sum(w*F(x)**6 for x,w in zip(a,d2))/math.factorial(6)==F(-1,90)
checks['status']='PASS';print(json.dumps(checks,indent=2))
